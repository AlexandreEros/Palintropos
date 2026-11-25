"""How presets recorded before the shared illumination geometry relate to
the current ones (Operation Planet One, 2026-11-24).

The archived configurations used the ``tidally_locked``, ``instantaneous``
and default-period ``diurnal_mean`` insolation forms. Today's presets state
the same experiments in the orbit/rotation geometry, so their configuration
hashes differ by design. This module checks the scientific content of that
change instead of the hashes:

* Every other configuration entry is unchanged.
* Each legacy forcing is re-evaluated with its own legacy formulas, written
  out here, and compared with the new geometry. A fixed substellar point,
  daily means, declination and distance agree to round-off. The one
  intended difference is in ``instantaneous`` runs: they advanced the hour
  angle uniformly, and the geometry adds the equation of time. That
  difference is checked against its analytic form.
"""
from __future__ import annotations

import math

import numpy as np

from tropoi.physics.insolation import insolation_from_spec
from tropoi.physics.orbit import true_from_mean

TWO_PI = 2.0 * math.pi


def without_insolation(config: dict) -> dict:
    return {k: v for k, v in config.items() if k != "insolation"}


def _wrap(x):
    return (np.asarray(x) + math.pi) % TWO_PI - math.pi


def _legacy_true_anomaly(spec, t):
    period = spec.get("period", 365.25 * 86400.0)
    mean = TWO_PI * ((t - spec.get("time_of_perihelion", 0.0)) / period % 1.0)
    return true_from_mean(mean, spec.get("eccentricity", 0.0))


def _legacy_declination_and_flux(spec, t):
    e = spec.get("eccentricity", 0.0)
    nu = _legacy_true_anomaly(spec, t)
    ls = nu + math.radians(spec.get("perihelion_longitude_deg", 0.0))
    dec = math.asin(math.sin(math.radians(spec.get("obliquity_deg", 0.0))) * math.sin(ls))
    flux = spec["stellar_constant"] * ((1.0 + e * math.cos(nu)) / (1.0 - e * e)) ** 2
    return dec, flux, ls


def assert_insolation_migration(legacy: dict, current: dict) -> None:
    """``current`` is the same forcing as the archived ``legacy`` spec, apart
    from the documented equation-of-time change for ``instantaneous``."""
    kind = legacy["kind"]
    new = insolation_from_spec(current)
    lat = np.radians(np.linspace(-89.5, 89.5, 90))
    lon = np.radians(np.linspace(0.0, 358.0, 180))
    la, lo = (a.ravel() for a in np.meshgrid(lat, lon, indexing="ij"))
    if kind == "tidally_locked":
        assert current["kind"] == "orbital" and current.get("eccentricity", 0.0) == 0.0
        s = legacy["stellar_constant"]
        lon_ss = legacy.get("substellar_longitude", 0.0)
        old = s * np.maximum(np.cos(la) * np.cos(lo - lon_ss), 0.0)
        for t in np.linspace(0.0, 10 * current["period"], 7):
            flux, _ = new(la, lo, float(t))
            assert np.max(np.abs(flux - old)) <= 1e-12 * s
        return
    if kind == "diurnal_mean":
        assert current["kind"] == "diurnal_mean"
        for t in np.linspace(0.0, current["period"], 5):
            dec, s, _ = _legacy_declination_and_flux(legacy, float(t))
            h0 = np.arccos(np.clip(-np.tan(la) * math.tan(dec), -1.0, 1.0))
            old = s / math.pi * np.maximum(h0 * np.sin(la) * math.sin(dec)
                                           + np.cos(la) * math.cos(dec) * np.sin(h0), 0.0)
            flux, _ = new(la, lo, float(t))
            assert np.max(np.abs(flux - old)) <= 1e-12 * s
        return
    if kind == "instantaneous":
        assert current["kind"] == "orbital"
        ill = new.illumination
        sol = legacy["solar_day"]
        assert abs(ill.mean_solar_day / sol - 1.0) <= 1e-13
        eps = math.radians(legacy.get("obliquity_deg", 0.0))
        n = TWO_PI / legacy["period"]
        _, _, ls0 = _legacy_declination_and_flux(legacy, 0.0)
        alpha0 = math.atan2(math.cos(eps) * math.sin(ls0), math.cos(ls0))
        for t in np.linspace(0.0, legacy["period"], 61):
            dec, flux, ls = _legacy_declination_and_flux(legacy, float(t))
            lat_ss, lon_ss = ill.subsolar_point(float(t))
            assert abs(lat_ss - dec) <= 1e-12
            assert abs(ill.flux_at(float(t)) / flux - 1.0) <= 1e-12
            # Legacy subsolar longitude: -2 pi t / sol. The geometry adds the
            # equation of time, i.e. the right ascension of the Sun minus
            # that of the mean Sun, both measured from the epoch.
            alpha = math.atan2(math.cos(eps) * math.sin(ls), math.cos(ls))
            eot = _wrap(alpha - alpha0 - n * float(t))
            assert abs(_wrap(lon_ss - (-TWO_PI * float(t) / sol) - eot)) <= 1e-9
        return
    raise AssertionError(f"no recorded migration for insolation kind {kind!r}")


def assert_same_experiment_apart_from_insolation(cfg, recorded_config: dict) -> None:
    assert without_insolation(cfg.to_dict()) == without_insolation(recorded_config)
    legacy, current = recorded_config["insolation"], cfg.to_dict()["insolation"]
    if legacy == current:
        return
    assert_insolation_migration(legacy, current)
