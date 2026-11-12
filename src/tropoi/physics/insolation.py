"""Stellar flux at the top of the atmosphere.

Every geometry returns the pair the radiation scheme needs per column:
``toa_flux`` (W m^-2 through a horizontal surface) and ``mu0`` (cosine of
the zenith angle the beam is treated with). Geometries:

* :class:`DiurnalMeanInsolation` — the daily-mean flux of a planet on a
  Kepler orbit with obliquity; with zero obliquity and eccentricity it is the
  perpetual-equinox ``S0 cos(lat) / pi`` of Schneider & Liu (2009).
* :class:`InstantaneousInsolation` — the same orbit with the diurnal cycle.
* :class:`TidallyLockedInsolation` — a fixed substellar point (THAI).
* :class:`PrescribedInsolation` — any stated function of latitude, e.g. the
  net surface flux of Frierson et al. (2006).

Time is seconds since perihelion passage shifted by ``time_of_perihelion``.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["Orbit", "DiurnalMeanInsolation", "InstantaneousInsolation",
           "TidallyLockedInsolation", "PrescribedInsolation", "frierson_net_surface_flux"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


@dataclass(frozen=True)
class Orbit:
    """A Kepler orbit and spin axis. ``stellar_constant`` is the flux at the
    semi-major axis distance; ``perihelion_longitude_deg`` is the solar
    longitude L_s (0 = northern spring equinox) at perihelion."""

    stellar_constant: float
    period: float = 365.25 * 86400.0
    eccentricity: float = 0.0
    obliquity_deg: float = 0.0
    perihelion_longitude_deg: float = 0.0
    time_of_perihelion: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.eccentricity < 1.0):
            raise ValueError("eccentricity must be in [0, 1)")
        if not (self.stellar_constant >= 0.0 and self.period > 0.0):
            raise ValueError("stellar_constant must be >= 0 and period > 0")

    def true_anomaly(self, t: float) -> float:
        mean = 2.0 * math.pi * ((t - self.time_of_perihelion) / self.period % 1.0)
        e = self.eccentricity
        ecc = mean
        for _ in range(50):                         # Newton on Kepler's equation
            ecc -= (ecc - e * math.sin(ecc) - mean) / (1.0 - e * math.cos(ecc))
        return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * ecc),
                                math.sqrt(1.0 - e) * math.cos(0.5 * ecc))

    def solar_longitude(self, t: float) -> float:
        """L_s in radians, in [0, 2 pi)."""
        return (self.true_anomaly(t) + math.radians(self.perihelion_longitude_deg)) % (2.0 * math.pi)

    def declination(self, t: float) -> float:
        return math.asin(math.sin(math.radians(self.obliquity_deg))
                         * math.sin(self.solar_longitude(t)))

    def flux_at(self, t: float) -> float:
        """Stellar flux at the planet's instantaneous distance."""
        e = self.eccentricity
        return self.stellar_constant * ((1.0 + e * math.cos(self.true_anomaly(t)))
                                        / (1.0 - e * e)) ** 2

    def as_dict(self) -> dict:
        return {"stellar_constant": self.stellar_constant, "period": self.period,
                "eccentricity": self.eccentricity, "obliquity_deg": self.obliquity_deg,
                "perihelion_longitude_deg": self.perihelion_longitude_deg,
                "time_of_perihelion": self.time_of_perihelion}


class DiurnalMeanInsolation:
    """Daily-mean insolation ``(S / pi) (h0 sin(lat) sin(dec) + cos(lat)
    cos(dec) sin(h0))``; ``mu0`` is the daylight mean of the zenith cosine."""

    def __init__(self, orbit: Orbit):
        self.orbit = orbit

    def signature(self) -> dict:
        return {"kind": "diurnal_mean", "orbit": self.orbit.as_dict()}

    def __call__(self, latitude, longitude, t: float):
        xp = _xp(latitude)
        dec = self.orbit.declination(t)
        s = self.orbit.flux_at(t)
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

    def __init__(self, orbit: Orbit, solar_day: float):
        if not solar_day > 0.0:
            raise ValueError("solar_day must be > 0")
        self.orbit = orbit
        self.solar_day = float(solar_day)

    def signature(self) -> dict:
        return {"kind": "instantaneous", "orbit": self.orbit.as_dict(),
                "solar_day": self.solar_day}

    def __call__(self, latitude, longitude, t: float):
        xp = _xp(latitude)
        dec = self.orbit.declination(t)
        hour_angle = longitude + 2.0 * math.pi * (t / self.solar_day)
        mu = (xp.sin(latitude) * math.sin(dec)
              + xp.cos(latitude) * math.cos(dec) * xp.cos(hour_angle))
        mu = xp.maximum(mu, 0.0)
        return self.orbit.flux_at(t) * mu, mu


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
