"""A blackbody illuminating star (spectral radiance and irradiance).

The star is a uniform disk (no limb darkening) of Planck radiance
``B_lambda(T)`` and angular radius ``alpha`` seen from the planet, so the
irradiance on a surface facing it, above the atmosphere, is
``B_lambda * Omega`` with ``Omega = 2 pi (1 - cos alpha)``. A real stellar
spectrum is not a blackbody (absorption lines, limb darkening); this is
stated wherever the star is drawn.

A star can be given by its angular radius alone, or by a physical radius
and a distance (``Star.of``): ``sin alpha = R / d``. Its luminosity is then
``L = 4 pi R^2 sigma T^4`` and its bolometric irradiance at the planet
``S = sigma T^4 sin^2 alpha = L / (4 pi d^2)``.

Nominal values: IAU 2015 Resolution B3 (solar T_eff = 5772 K, R = 695 700
km, L = 3.828e26 W, total solar irradiance 1361 W m^-2) and the IAU 2012
Resolution B2 astronomical unit (149 597 870.7 km). ``STARS`` lists real
stars, each with the source of its numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.spatial.optics.constants import C, H, K_B

__all__ = ["AU_KM", "L_SUN_W", "R_SUN_KM", "SIGMA_SB", "SOLAR_CONSTANT",
           "STARS", "Star", "StellarBody", "planck_radiance"]

R_SUN_KM = 695700.0                       # IAU 2015 B3 nominal solar radius
AU_KM = 149597870.7                       # IAU 2012 B2
L_SUN_W = 3.828e26                        # IAU 2015 B3 nominal luminosity
T_SUN_K = 5772.0                          # IAU 2015 B3 nominal T_eff
SOLAR_CONSTANT = 1361.0                   # IAU 2015 B3 nominal TSI, W m^-2
#: Stefan-Boltzmann constant from the exact SI constants (W m^-2 K^-4).
SIGMA_SB = 2.0 * math.pi ** 5 * K_B ** 4 / (15.0 * H ** 3 * C ** 2)
NOMINAL_SOURCE = ("IAU 2015 B3 nominal T_eff = 5772 K, R = 695 700 km; "
                  "IAU 2012 B2 au")


def planck_radiance(wavelength_nm, temperature_k: float) -> np.ndarray:
    """Planck spectral radiance B_lambda (W m^-2 sr^-1 nm^-1)."""
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    x = H * C / (lam * K_B * float(temperature_k))
    return 2.0 * H * C ** 2 / lam ** 5 / np.expm1(x) * 1e-9


def _alpha_deg(radius_rsun: float, distance_au: float) -> float:
    return math.degrees(math.asin(radius_rsun * R_SUN_KM
                                  / (distance_au * AU_KM)))


@dataclass(frozen=True)
class Star:
    """A blackbody disk. ``radius_rsun``/``distance_au`` are optional: given
    together they say what the angular radius comes from."""

    temperature_k: float
    angular_radius_deg: float
    label: str = "blackbody star"
    source: str = "stated by the user"
    radius_rsun: float | None = None
    distance_au: float | None = None

    def __post_init__(self) -> None:
        t, a = float(self.temperature_k), float(self.angular_radius_deg)
        if not (math.isfinite(t) and t > 0):
            raise ValueError(f"temperature_k must be > 0, got {t}")
        if not (math.isfinite(a) and 0.0 < a < 30.0):
            raise ValueError(f"angular_radius_deg must be in (0, 30), got {a}")
        if (self.radius_rsun is None) != (self.distance_au is None):
            raise ValueError("give radius_rsun and distance_au together")
        for name in ("radius_rsun", "distance_au"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(float(v))
                                      and float(v) > 0.0):
                raise ValueError(f"{name} must be > 0, got {v}")

    @classmethod
    def of(cls, temperature_k: float, radius_rsun: float, distance_au: float,
           *, label: str = "blackbody star",
           source: str = "stated by the user") -> "Star":
        """A star of radius ``radius_rsun`` (R_sun) seen from
        ``distance_au``; refused when it would fill more than 30 degrees."""
        r, d = float(radius_rsun), float(distance_au)
        if not (r > 0 and d > 0 and math.isfinite(r) and math.isfinite(d)):
            raise ValueError("radius_rsun and distance_au must be > 0")
        s = r * R_SUN_KM / (d * AU_KM)
        if not s < math.sin(math.radians(30.0)):
            raise ValueError(f"a star of {r:g} R_sun at {d:g} au would fill "
                             "more than 30 degrees of sky")
        return cls(float(temperature_k), _alpha_deg(r, d), label=label,
                   source=source, radius_rsun=r, distance_au=d)

    @classmethod
    def sun_at_1au(cls) -> "Star":
        return cls(T_SUN_K, math.degrees(math.asin(R_SUN_KM / AU_KM)),
                   label="Sun-like blackbody at 1 au", source=NOMINAL_SOURCE,
                   radius_rsun=1.0, distance_au=1.0)

    @classmethod
    def sun_at(cls, distance_au: float, *, where: str,
               distance_source: str) -> "Star":
        """The IAU nominal Sun seen from ``distance_au`` (``where`` names the
        place, ``distance_source`` says where the distance comes from)."""
        d = float(distance_au)
        if not (math.isfinite(d) and d > 0.0):
            raise ValueError(f"distance_au must be > 0, got {d}")
        return cls.of(T_SUN_K, 1.0, d,
                      label=f"Sun-like blackbody at {d:g} au ({where})",
                      source=f"{NOMINAL_SOURCE}; distance: {distance_source}")

    @classmethod
    def with_sun_irradiance_at_1au(cls, temperature_k: float) -> "Star":
        """A blackbody of ``temperature_k`` at 1 au, sized to deliver the IAU
        nominal Sun's bolometric irradiance there (irradiance ~ T^4 sin^2
        alpha, so its radius is (5772 K / T)^2 R_sun)."""
        s = R_SUN_KM / AU_KM * (T_SUN_K / float(temperature_k)) ** 2
        if not 0.0 < s < math.sin(math.radians(30.0)):
            raise ValueError(f"no such star at {temperature_k} K "
                             f"(sin alpha = {s})")
        return cls(float(temperature_k), math.degrees(math.asin(s)),
                   label=f"{float(temperature_k):g} K blackbody star",
                   source=("angular radius chosen for the IAU nominal Sun's "
                           "bolometric irradiance at 1 au"),
                   radius_rsun=(T_SUN_K / float(temperature_k)) ** 2,
                   distance_au=1.0)

    @property
    def solid_angle_sr(self) -> float:
        return 2.0 * math.pi * (1.0 - math.cos(math.radians(
            self.angular_radius_deg)))

    @property
    def bolometric_irradiance_w_m2(self) -> float:
        """sigma T^4 sin^2 alpha: the total irradiance at the planet."""
        return SIGMA_SB * float(self.temperature_k) ** 4 * math.sin(
            math.radians(self.angular_radius_deg)) ** 2

    @property
    def luminosity_lsun(self) -> float | None:
        """4 pi R^2 sigma T^4 in nominal solar luminosities (needs a radius)."""
        if self.radius_rsun is None:
            return None
        r = self.radius_rsun * R_SUN_KM * 1e3
        return 4.0 * math.pi * r * r * SIGMA_SB * \
            float(self.temperature_k) ** 4 / L_SUN_W

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
                "angular_radius_deg": float(self.angular_radius_deg),
                "radius_rsun": (None if self.radius_rsun is None
                                else float(self.radius_rsun)),
                "distance_au": (None if self.distance_au is None
                                else float(self.distance_au))}


@dataclass(frozen=True)
class StellarBody:
    """A real star as a blackbody: effective temperature and radius, with
    the source of both. Seen from a distance, it becomes a :class:`Star`."""

    key: str
    name: str
    kind: str
    temperature_k: float
    radius_rsun: float
    source: str

    @classmethod
    def from_luminosity(cls, key: str, name: str, kind: str, *,
                        radius_rsun: float, luminosity_lsun: float,
                        source: str) -> "StellarBody":
        """Temperature from L = 4 pi R^2 sigma T^4 (one source for both)."""
        r = radius_rsun * R_SUN_KM * 1e3
        t = (luminosity_lsun * L_SUN_W / (4.0 * math.pi * r * r * SIGMA_SB)
             ) ** 0.25
        return cls(key, name, kind, t, float(radius_rsun), source)

    def at(self, distance_au: float) -> Star:
        return Star.of(self.temperature_k, self.radius_rsun, distance_au,
                       label=f"{self.name} at {float(distance_au):g} au",
                       source=self.source)

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "kind": self.kind,
                "temperature_k": float(self.temperature_k),
                "radius_rsun": float(self.radius_rsun),
                "source": self.source}


_WIKI = "Wikipedia infobox, fetched 2026-11-07, citing "
#: Real stars as blackbodies, coolest first. Where one paper gives both the
#: radius and the luminosity, the temperature follows from them.
STARS: dict[str, StellarBody] = {s.key: s for s in (
    StellarBody("proxima_centauri", "Proxima Centauri", "M5.5 V red dwarf",
                2833.0, 0.162,
                _WIKI + "Anugu, Gallenne, Kervella & Rains (2026), "
                "arXiv:2610.33373 (preprint): T_eff 2833 ± 49 K, R 0.162 ± "
                "0.004 R_sun"),
    StellarBody.from_luminosity(
        "epsilon_eridani", "ε Eridani", "K2 V dwarf",
        radius_rsun=0.742, luminosity_lsun=0.326,
        source=_WIKI + "Soubiran et al. (2024), A&A 682, A145: R 0.742 ± "
        "0.004 R_sun, L 0.326 ± 0.007 L_sun; T_eff from L = 4πR²σT⁴"),
    StellarBody("sun", "the Sun", "G2 V dwarf", T_SUN_K, 1.0,
                "IAU 2015 B3 nominal T_eff = 5772 K, R = 695 700 km"),
    StellarBody.from_luminosity(
        "procyon_a", "Procyon A", "F5 IV–V subgiant",
        radius_rsun=2.043, luminosity_lsun=7.049,
        source=_WIKI + "Soubiran et al. (2024), A&A 682, A145: R 2.043 ± "
        "0.009 R_sun, L 7.049 ± 0.064 L_sun; T_eff from L = 4πR²σT⁴"),
    StellarBody.from_luminosity(
        "sirius_a", "Sirius A", "A0–A1 V star",
        radius_rsun=1.7144, luminosity_lsun=24.74,
        source=_WIKI + "its reference 'bond-11': R 1.7144 ± 0.0090 R_sun, "
        "L 24.74 ± 0.70 L_sun; T_eff from L = 4πR²σT⁴"),
)}
