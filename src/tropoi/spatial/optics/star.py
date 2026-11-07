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
