"""A blackbody illuminating star (spectral radiance and irradiance).

The star is a uniform disk (no limb darkening) of Planck radiance
``B_lambda(T)`` and angular radius ``alpha`` seen from the planet, so the
irradiance on a surface facing it, above the atmosphere, is
``B_lambda * Omega`` with ``Omega = 2 pi (1 - cos alpha)``. A real stellar
spectrum is not a blackbody (absorption lines, limb darkening); this is
stated wherever the star is drawn.

``Star.sun_at_1au()`` uses IAU 2015 Resolution B3 nominal solar values
(T_eff = 5772 K, R = 695 700 km) and the IAU 2012 Resolution B2
astronomical unit (149 597 870.7 km).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.spatial.optics.constants import C, H, K_B

__all__ = ["Star", "planck_radiance"]


def planck_radiance(wavelength_nm, temperature_k: float) -> np.ndarray:
    """Planck spectral radiance B_lambda (W m^-2 sr^-1 nm^-1)."""
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    x = H * C / (lam * K_B * float(temperature_k))
    return 2.0 * H * C ** 2 / lam ** 5 / np.expm1(x) * 1e-9


@dataclass(frozen=True)
class Star:
    temperature_k: float
    angular_radius_deg: float
    label: str = "blackbody star"
    source: str = "stated by the user"

    def __post_init__(self) -> None:
        t, a = float(self.temperature_k), float(self.angular_radius_deg)
        if not (math.isfinite(t) and t > 0):
            raise ValueError(f"temperature_k must be > 0, got {t}")
        if not (math.isfinite(a) and 0.0 < a < 30.0):
            raise ValueError(f"angular_radius_deg must be in (0, 30), got {a}")

    @classmethod
    def sun_at_1au(cls) -> "Star":
        return cls(5772.0, math.degrees(math.asin(695700.0 / 149597870.7)),
                   label="Sun-like blackbody at 1 au",
                   source=("IAU 2015 B3 nominal T_eff = 5772 K, R = 695 700 "
                           "km; IAU 2012 B2 au"))

    @classmethod
    def sun_at(cls, distance_au: float, *, where: str,
               distance_source: str) -> "Star":
        """The IAU nominal Sun seen from ``distance_au`` (``where`` names the
        place, ``distance_source`` says where the distance comes from)."""
        d = float(distance_au)
        if not (math.isfinite(d) and d > 0.0):
            raise ValueError(f"distance_au must be > 0, got {d}")
        return cls(5772.0, math.degrees(math.asin(
            695700.0 / (149597870.7 * d))),
            label=f"Sun-like blackbody at {d:g} au ({where})",
            source=("IAU 2015 B3 nominal T_eff = 5772 K, R = 695 700 km; "
                    f"IAU 2012 B2 au; distance: {distance_source}"))

    @classmethod
    def with_sun_irradiance_at_1au(cls, temperature_k: float) -> "Star":
        """A blackbody of ``temperature_k`` sized to deliver the IAU nominal
        Sun's bolometric irradiance at 1 au (irradiance ~ T^4 sin^2 alpha)."""
        s = 695700.0 / 149597870.7 * (5772.0 / float(temperature_k)) ** 2
        if not 0.0 < s < math.sin(math.radians(30.0)):
            raise ValueError(f"no such star at {temperature_k} K "
                             f"(sin alpha = {s})")
        return cls(float(temperature_k), math.degrees(math.asin(s)),
                   label=f"{float(temperature_k):g} K blackbody star",
                   source=("angular radius chosen for the IAU nominal Sun's "
                           "bolometric irradiance at 1 au"))

    @property
    def solid_angle_sr(self) -> float:
        return 2.0 * math.pi * (1.0 - math.cos(math.radians(
            self.angular_radius_deg)))

    def radiance(self, wavelength_nm) -> np.ndarray:
        return planck_radiance(wavelength_nm, self.temperature_k)

    def irradiance(self, wavelength_nm) -> np.ndarray:
        """Spectral irradiance at the top of the atmosphere (W m^-2 nm^-1)."""
        return self.radiance(wavelength_nm) * self.solid_angle_sr

    def describe(self) -> str:
        return (f"{self.label}: blackbody {self.temperature_k:g} K, angular "
                f"radius {self.angular_radius_deg:.4g}° ({self.source})")

    def to_dict(self) -> dict:
        return {"label": self.label, "source": self.source,
                "temperature_k": float(self.temperature_k),
                "angular_radius_deg": float(self.angular_radius_deg)}
