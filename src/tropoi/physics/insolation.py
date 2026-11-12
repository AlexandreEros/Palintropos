"""Stellar flux at the top of the atmosphere.

Illumination is a *combination*: a star (how much light), an orbit (where
the planet is), a rotation (which way it points and how fast it spins), a
location and a time. :class:`StellarIllumination` owns that combination for
one planet, either assembled from the shared objects
(:meth:`StellarIllumination.from_components`: a
:class:`~tropoi.physics.star.Star`, a
:class:`~tropoi.physics.orbit.KeplerOrbit` and a
:class:`~tropoi.physics.rotation.Rotation`, with the gravitational
parameter from the star or given explicitly) or prescribed directly
(:meth:`StellarIllumination.prescribed`, which states its inputs the way
experiment configurations do: the irradiance *at the semimajor-axis
distance*, the period, eccentricity, obliquity and the solar longitude of
periapsis).

Every insolation geometry returns the pair the radiation scheme needs per
column: ``toa_flux`` (W m^-2 through a horizontal surface) and ``mu0``
(cosine of the zenith angle the beam is treated with). Geometries:

* :class:`DiurnalMeanInsolation` — the daily-mean flux of a planet on a
  Kepler orbit with obliquity; with zero obliquity and eccentricity it is the
  perpetual-equinox ``S0 cos(lat) / pi`` of Schneider & Liu (2009).
* :class:`InstantaneousInsolation` — the same orbit with the diurnal cycle.
* :class:`TidallyLockedInsolation` — a fixed substellar point (THAI).
* :class:`PrescribedInsolation` — any stated function of latitude, e.g. the
  net surface flux of Frierson et al. (2006).

The explicitly prescribed and daily-mean forcing modes are retained
unchanged; their numerics are identical to the pre-reconciliation ``Orbit``
(same Kepler solve, same formulas).

Time is seconds since periapsis passage shifted by ``time_of_periapsis``.
``L_s`` is the solar longitude (0 = northern spring equinox).

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from tropoi.physics.orbit import KeplerOrbit, true_from_mean
from tropoi.physics.rotation import Rotation
from tropoi.physics.star import Star

__all__ = ["DiurnalMeanInsolation", "InstantaneousInsolation",
           "PrescribedInsolation", "StellarIllumination",
           "TidallyLockedInsolation", "frierson_net_surface_flux"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


@dataclass(frozen=True)
class StellarIllumination:
    """Star + orbit + rotation of one planet, reduced to what illumination
    needs: the irradiance at the semimajor-axis distance, the orbit's
    period, eccentricity and timing, and the spin axis (obliquity and the
    solar longitude of periapsis, ``L_s`` of periapsis: the angle from the
    northern spring equinox to periapsis along the orbit).

    ``star``, ``orbit`` and ``rotation`` hold the full objects when the
    illumination was assembled from them (:meth:`from_components`); a
    :meth:`prescribed` illumination has none."""

    irradiance_at_semimajor: float          # W m^-2, facing the star at r = a
    period_s: float
    eccentricity: float = 0.0
    obliquity_rad: float = 0.0
    ls_periapsis_rad: float = 0.0
    time_of_periapsis_s: float = 0.0
    star: Optional[Star] = None
    orbit: Optional[KeplerOrbit] = None
    rotation: Optional[Rotation] = None

    def __post_init__(self):
        if not (0.0 <= self.eccentricity < 1.0):
            raise ValueError("eccentricity must be in [0, 1)")
        if not (self.irradiance_at_semimajor >= 0.0 and self.period_s > 0.0):
            raise ValueError("irradiance_at_semimajor must be >= 0 and "
                             "period_s > 0")

    # -- factories --------------------------------------------------------------

    @classmethod
    def prescribed(cls, stellar_constant: float,
                   period: float = 365.25 * 86400.0,
                   eccentricity: float = 0.0, obliquity_deg: float = 0.0,
                   perihelion_longitude_deg: float = 0.0,
                   time_of_perihelion: float = 0.0) -> "StellarIllumination":
        """Explicitly prescribed illumination, stated the way experiment
        configurations do. ``stellar_constant`` is the flux at the
        semimajor-axis distance — not at periapsis and not an orbital mean;
        ``perihelion_longitude_deg`` is the solar longitude L_s (0 =
        northern spring equinox) at perihelion."""
        return cls(float(stellar_constant), float(period),
                   float(eccentricity), math.radians(float(obliquity_deg)),
                   math.radians(float(perihelion_longitude_deg)),
                   float(time_of_perihelion))

    @classmethod
    def from_components(cls, star: Star, orbit: KeplerOrbit,
                        rotation: Rotation, *,
                        gravitational_parameter: float | None = None,
                        time_of_periapsis: float | None = None,
                        irradiance_at_semimajor: float | None = None
                        ) -> "StellarIllumination":
        """Combine the shared objects. The period comes from Kepler's third
        law with ``mu`` taken from ``gravitational_parameter`` or the star's
        mass; the irradiance at r = a from the star's luminosity, unless a
        measured value is stated explicitly (then the stated value wins and
        the star's luminosity is provenance only). By default the timing
        honours the orbit's epoch (``true_anomaly_epoch_rad``: the planet is
        there at t = 0, so ``flux_at(t)`` agrees with
        ``orbit.radius_at_time(t)``); an explicit ``time_of_periapsis``
        overrides it."""
        mu = gravitational_parameter
        if mu is None:
            mu = star.gravitational_parameter
        if mu is None:
            raise ValueError(
                f"propagating the orbit needs the gravitational parameter: "
                f"the star {star.name!r} has no mass; pass "
                "gravitational_parameter explicitly")
        s = (star.irradiance_at(orbit.semimajor_axis_m)
             if irradiance_at_semimajor is None
             else float(irradiance_at_semimajor))
        period = orbit.period(mu)
        if time_of_periapsis is None:
            # Mean anomaly M0 at t = 0 means the last periapsis passage was
            # M0 / n before the epoch.
            time_of_periapsis = (-orbit.mean_anomaly_epoch_rad * period
                                 / (2.0 * math.pi))
        return cls(s, period, orbit.eccentricity,
                   rotation.obliquity_rad,
                   orbit.periapsis_longitude_rad
                   - rotation.equinox_longitude_rad,
                   float(time_of_periapsis), star=star, orbit=orbit,
                   rotation=rotation)

    # -- propagation (identical numerics to the pre-reconciliation Orbit) -------

    def true_anomaly(self, t: float) -> float:
        mean = 2.0 * math.pi * ((t - self.time_of_periapsis_s)
                                / self.period_s % 1.0)
        return true_from_mean(mean, self.eccentricity)

    def solar_longitude(self, t: float) -> float:
        """L_s in radians, in [0, 2 pi)."""
        return (self.true_anomaly(t) + self.ls_periapsis_rad) % (2.0 * math.pi)

    def declination(self, t: float) -> float:
        return math.asin(math.sin(self.obliquity_rad)
                         * math.sin(self.solar_longitude(t)))

    def flux_at(self, t: float) -> float:
        """Stellar flux at the planet's instantaneous distance."""
        e = self.eccentricity
        return self.irradiance_at_semimajor * (
            (1.0 + e * math.cos(self.true_anomaly(t))) / (1.0 - e * e)) ** 2

    # -- description --------------------------------------------------------------

    @property
    def obliquity_deg(self) -> float:
        return math.degrees(self.obliquity_rad)

    def as_dict(self) -> dict:
        out = {"stellar_constant": self.irradiance_at_semimajor,
               "period": self.period_s, "eccentricity": self.eccentricity,
               "obliquity_deg": math.degrees(self.obliquity_rad),
               "perihelion_longitude_deg": math.degrees(self.ls_periapsis_rad),
               "time_of_perihelion": self.time_of_periapsis_s}
        if self.star is not None or self.orbit is not None:
            out["components"] = {
                "star": None if self.star is None else self.star.to_dict(),
                "orbit": None if self.orbit is None else self.orbit.to_dict(),
                "rotation": (None if self.rotation is None
                             else self.rotation.to_dict())}
        return out


class DiurnalMeanInsolation:
    """Daily-mean insolation ``(S / pi) (h0 sin(lat) sin(dec) + cos(lat)
    cos(dec) sin(h0))``; ``mu0`` is the daylight mean of the zenith cosine."""

    def __init__(self, illumination: StellarIllumination):
        self.illumination = illumination

    def signature(self) -> dict:
        return {"kind": "diurnal_mean", "orbit": self.illumination.as_dict()}

    def __call__(self, latitude, longitude, t: float):
        xp = _xp(latitude)
        dec = self.illumination.declination(t)
        s = self.illumination.flux_at(t)
        cos_h0 = xp.clip(-xp.tan(latitude) * math.tan(dec), -1.0, 1.0)
        h0 = xp.arccos(cos_h0)
        integral = (h0 * xp.sin(latitude) * math.sin(dec)
                    + xp.cos(latitude) * math.cos(dec) * xp.sin(h0))
        integral = xp.maximum(integral, 0.0)
        mu0 = xp.where(h0 > 0.0, integral / xp.maximum(h0, 1e-300), 0.0)
        return s * integral / math.pi, mu0


class InstantaneousInsolation:
    """Insolation with a diurnal cycle. ``solar_day`` is the length of the
    solar day in seconds; the substellar longitude is ``-2 pi t / solar_day``
    (local noon at longitude 0 at t = 0)."""

    def __init__(self, illumination: StellarIllumination, solar_day: float):
        if not solar_day > 0.0:
            raise ValueError("solar_day must be > 0")
        self.illumination = illumination
        self.solar_day = float(solar_day)

    def signature(self) -> dict:
        return {"kind": "instantaneous", "orbit": self.illumination.as_dict(),
                "solar_day": self.solar_day}

    def __call__(self, latitude, longitude, t: float):
        xp = _xp(latitude)
        dec = self.illumination.declination(t)
        hour_angle = longitude + 2.0 * math.pi * (t / self.solar_day)
        mu = (xp.sin(latitude) * math.sin(dec)
              + xp.cos(latitude) * math.cos(dec) * xp.cos(hour_angle))
        mu = xp.maximum(mu, 0.0)
        return self.illumination.flux_at(t) * mu, mu


class TidallyLockedInsolation:
    """A permanent substellar point at latitude 0 and ``substellar_longitude``."""

    def __init__(self, stellar_constant: float, substellar_longitude: float = 0.0):
        self.stellar_constant = float(stellar_constant)
        self.substellar_longitude = float(substellar_longitude)

    def signature(self) -> dict:
        return {"kind": "tidally_locked", "stellar_constant": self.stellar_constant,
                "substellar_longitude": self.substellar_longitude}

    def __call__(self, latitude, longitude, t: float):
        xp = _xp(latitude)
        mu = xp.maximum(xp.cos(latitude) * xp.cos(longitude - self.substellar_longitude), 0.0)
        return self.stellar_constant * mu, mu


class PrescribedInsolation:
    """``toa_flux = function(latitude)`` with a fixed ``mu0`` (irrelevant for
    a transparent atmosphere)."""

    def __init__(self, function: Callable, name: str, params: dict, mu0: float = 0.5):
        self.function = function
        self.name = name
        self.params = dict(params)
        self.mu0 = float(mu0)

    def signature(self) -> dict:
        return {"kind": "prescribed", "name": self.name, "params": self.params,
                "mu0": self.mu0}

    def __call__(self, latitude, longitude, t: float):
        flux = self.function(latitude)
        return flux, self.mu0 + 0.0 * flux


def frierson_net_surface_flux(net_solar_constant: float = 938.4,
                              delta_s: float = 1.4) -> PrescribedInsolation:
    """FHZ06 eqs. 2-3: ``R_S = R_S0 [1 + Delta_s p2(lat)]``, ``p2 = (1 - 3
    sin^2 lat) / 4``, all of it absorbed at the surface.

    Table 1 of the paper lists 938.4 W m^-2 as the "net solar constant
    (includes albedo)" while the text calls ``R_S0`` the global-mean net
    solar flux. Since ``p2`` has zero global mean, ``R_S0`` is the global
    mean; 938.4 is taken as four times it (938.4 / 4 = 234.6 W m^-2, i.e.
    1360 W m^-2 with albedo 0.31), the only reading with an Earth-like
    energy balance.
    """
    r_s0 = float(net_solar_constant) / 4.0
    ds = float(delta_s)

    def flux(latitude):
        xp = _xp(latitude)
        s = xp.sin(latitude)
        return r_s0 * (1.0 + ds * 0.25 * (1.0 - 3.0 * s * s))

    return PrescribedInsolation(flux, "frierson_net_surface_flux",
                                {"net_solar_constant": float(net_solar_constant),
                                 "delta_s": ds})
