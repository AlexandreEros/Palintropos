"""Stars: intrinsic properties, and the apparent star an observer sees.

:class:`Star` owns what a star *is*: mass, radius, luminosity and effective
temperature, with an explicit spectral model. For the ``"blackbody"`` model
the three radiative quantities are tied by ``L = 4 pi R^2 sigma T^4``:
constructors derive a missing one from the other two and reject supplied
combinations that contradict it. A real stellar spectrum is not a blackbody
(absorption lines, limb darkening); a non-blackbody spectrum can be
supplied as a ``spectrum`` object with a ``radiance(wavelength_nm)`` method
(none is packaged), and is stated wherever the star is drawn.

What an observer receives is viewing geometry, not an intrinsic property:
apparent angular size and irradiance depend on the distance.
:meth:`Star.seen_from` combines the two into an :class:`ApparentStar` — a
disk of stated temperature and angular radius, which is also constructible
directly from those two numbers alone (the sky page's workflow: no stellar
mass or distance is ever invented for it).

Nominal values: IAU 2015 Resolution B3 (solar T_eff, radius, luminosity,
total solar irradiance) and the IAU 2012 Resolution B2 astronomical unit.
``STARS`` lists real stars from the packaged catalogue
(``tropoi/physics/data/stars.toml``), each with the source of its numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.physics.constants import (AU_M, C, G, H, K_B, L_SUN_W,
                                      NOMINAL_SOLAR_SOURCE, R_SUN_M,
                                      SIGMA_SB, SOLAR_CONSTANT, T_SUN_K)
from tropoi.physics.species import load_dataset

__all__ = ["AU_KM", "ApparentStar", "BLACKBODY_TOLERANCE", "L_SUN_W",
           "R_SUN_KM", "SIGMA_SB", "SOLAR_CONSTANT", "STARS", "Star",
           "planck_radiance"]

#: Kilometre forms kept for display blocks (the SI values are in constants).
R_SUN_KM = R_SUN_M / 1000.0
AU_KM = AU_M / 1000.0

#: Relative tolerance within which supplied (T, R, L) must satisfy
#: L = 4 pi R^2 sigma T^4 for a blackbody star.
BLACKBODY_TOLERANCE = 1e-3


def planck_radiance(wavelength_nm, temperature_k: float) -> np.ndarray:
    """Planck spectral radiance B_lambda (W m^-2 sr^-1 nm^-1)."""
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    x = H * C / (lam * K_B * float(temperature_k))
    return 2.0 * H * C ** 2 / lam ** 5 / np.expm1(x) * 1e-9


def _blackbody_luminosity(radius_m: float, temperature_k: float) -> float:
    return 4.0 * math.pi * radius_m * radius_m * SIGMA_SB * temperature_k ** 4


@dataclass(frozen=True)
class Star:
    """Intrinsic stellar properties (SI), with an explicit spectral model."""

    name: str
    temperature_k: float
    radius_m: float
    luminosity_w: float
    mass_kg: float | None = None
    kind: str = ""
    key: str = ""
    spectral_model: str = "blackbody"
    #: Optional supplied spectrum with a ``radiance(wavelength_nm)`` method
    #: (W m^-2 sr^-1 nm^-1); None means the spectral model's own radiance.
    spectrum: object | None = None
    source: str = "stated by the user"

    def __post_init__(self) -> None:
        for field_name in ("temperature_k", "radius_m", "luminosity_w"):
            v = float(getattr(self, field_name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{field_name} must be finite and > 0, "
                                 f"got {v}")
        if self.mass_kg is not None and not (
                math.isfinite(float(self.mass_kg)) and self.mass_kg > 0.0):
            raise ValueError(f"mass_kg must be > 0, got {self.mass_kg}")
        if self.spectral_model == "blackbody" and self.spectrum is None:
            implied = _blackbody_luminosity(self.radius_m, self.temperature_k)
            rel = abs(self.luminosity_w - implied) / implied
            if rel > BLACKBODY_TOLERANCE:
                raise ValueError(
                    f"inconsistent blackbody star {self.name!r}: supplied "
                    f"L = {self.luminosity_w:g} W but 4πR²σT⁴ = {implied:g} W"
                    f" ({rel * 100:.2g} % apart). Supply two of (T, R, L) and"
                    " let the third follow, or fix the numbers")

    # -- constructors ----------------------------------------------------------

    @classmethod
    def blackbody(cls, name: str, *, temperature_k: float | None = None,
                  radius_rsun: float | None = None,
                  luminosity_lsun: float | None = None,
                  mass_kg: float | None = None,
                  gm_m3_s2: float | None = None,
                  kind: str = "", key: str = "",
                  source: str = "stated by the user") -> "Star":
        """A blackbody star from any two of (T_eff, radius, luminosity); the
        third is derived from ``L = 4 pi R^2 sigma T^4``. Supplying all
        three is allowed only when they already satisfy it (within
        ``BLACKBODY_TOLERANCE``)."""
        given = [v is not None
                 for v in (temperature_k, radius_rsun, luminosity_lsun)]
        if sum(given) < 2:
            raise ValueError("a blackbody star needs two of temperature_k, "
                             "radius_rsun, luminosity_lsun")
        t = None if temperature_k is None else float(temperature_k)
        r = None if radius_rsun is None else float(radius_rsun) * R_SUN_M
        lum = None if luminosity_lsun is None else float(luminosity_lsun) * L_SUN_W
        if lum is None:
            lum = _blackbody_luminosity(r, t)
        elif r is None:
            r = math.sqrt(lum / (4.0 * math.pi * SIGMA_SB * t ** 4))
        elif t is None:
            t = (lum / (4.0 * math.pi * r * r * SIGMA_SB)) ** 0.25
        mass = mass_kg
        if mass is None and gm_m3_s2 is not None:
            mass = float(gm_m3_s2) / G
        return cls(name, t, r, lum, mass_kg=mass, kind=kind, key=key,
                   source=source)

    @classmethod
    def sun(cls) -> "Star":
        """The IAU nominal Sun (T_eff and radius nominal; luminosity derived
        from them, 0.013 % below the separately nominal 3.828e26 W)."""
        return STARS["sun"]

    # -- intrinsic properties ---------------------------------------------------

    @property
    def radius_rsun(self) -> float:
        return self.radius_m / R_SUN_M

    @property
    def luminosity_lsun(self) -> float:
        return self.luminosity_w / L_SUN_W

    @property
    def gravitational_parameter(self) -> float | None:
        """G M (m^3 s^-2), when the mass is known."""
        return None if self.mass_kg is None else G * float(self.mass_kg)

    def radiance(self, wavelength_nm) -> np.ndarray:
        """Spectral radiance of the stellar disk (W m^-2 sr^-1 nm^-1)."""
        if self.spectrum is not None:
            return np.asarray(self.spectrum.radiance(wavelength_nm))
        if self.spectral_model != "blackbody":
            raise ValueError(f"no radiance for spectral model "
                             f"{self.spectral_model!r} without a spectrum")
        return planck_radiance(wavelength_nm, self.temperature_k)

    # -- viewing geometry -------------------------------------------------------

    def irradiance_at(self, distance_m: float) -> float:
        """Bolometric irradiance L / (4 pi d^2) at ``distance_m`` (W m^-2)."""
        d = float(distance_m)
        if not (math.isfinite(d) and d > 0.0):
            raise ValueError(f"distance_m must be > 0, got {d}")
        return self.luminosity_w / (4.0 * math.pi * d * d)

    def angular_radius_deg_at(self, distance_m: float) -> float:
        """Apparent angular radius asin(R / d) in degrees."""
        d = float(distance_m)
        if not d > self.radius_m:
            raise ValueError("the observer must be outside the star")
        return math.degrees(math.asin(self.radius_m / d))

    def seen_from(self, distance_m: float, *, label: str | None = None
                  ) -> "ApparentStar":
        """The apparent star at ``distance_m``: this spectrum at the
        apparent angular radius."""
        d = float(distance_m)
        alpha = self.angular_radius_deg_at(d)
        if not alpha < 30.0:
            raise ValueError(f"{self.name} from {d:g} m would fill more than "
                             "30 degrees of sky")
        return ApparentStar(
            self.temperature_k, alpha,
            label=label if label is not None
            else f"{self.name} at {d / AU_M:g} au",
            source=self.source, radius_rsun=self.radius_rsun,
            distance_au=d / AU_M, spectrum=self.spectrum)

    def at(self, distance_au: float) -> "ApparentStar":
        """``seen_from`` with the distance in au (catalogue convenience)."""
        return self.seen_from(float(distance_au) * AU_M)

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "kind": self.kind,
                "temperature_k": float(self.temperature_k),
                "radius_rsun": float(self.radius_rsun),
                "luminosity_lsun": float(self.luminosity_lsun),
                "mass_kg": None if self.mass_kg is None else float(self.mass_kg),
                "spectral_model": self.spectral_model,
                "source": self.source}


@dataclass(frozen=True)
class ApparentStar:
    """A star as seen: a uniform disk (no limb darkening) of stated
    temperature and angular radius. ``radius_rsun``/``distance_au`` are
    optional provenance: given together they say what the angular radius
    comes from; nothing invents them. The irradiance on a surface facing
    the disk, above the atmosphere, is ``B_lambda * Omega`` with
    ``Omega = 2 pi (1 - cos alpha)``."""

    temperature_k: float
    angular_radius_deg: float
    label: str = "blackbody star"
    source: str = "stated by the user"
    radius_rsun: float | None = None
    distance_au: float | None = None
    spectrum: object | None = None

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
           source: str = "stated by the user") -> "ApparentStar":
        """A star of radius ``radius_rsun`` (R_sun) seen from
        ``distance_au``; refused when it would fill more than 30 degrees."""
        r, d = float(radius_rsun), float(distance_au)
        if not (r > 0 and d > 0 and math.isfinite(r) and math.isfinite(d)):
            raise ValueError("radius_rsun and distance_au must be > 0")
        s = r * R_SUN_KM / (d * AU_KM)
        if not s < math.sin(math.radians(30.0)):
            raise ValueError(f"a star of {r:g} R_sun at {d:g} au would fill "
                             "more than 30 degrees of sky")
        return cls(float(temperature_k), math.degrees(math.asin(s)),
                   label=label, source=source, radius_rsun=r, distance_au=d)

    @classmethod
    def sun_at_1au(cls) -> "ApparentStar":
        return cls(T_SUN_K, math.degrees(math.asin(R_SUN_KM / AU_KM)),
                   label="Sun-like blackbody at 1 au",
                   source=NOMINAL_SOLAR_SOURCE, radius_rsun=1.0,
                   distance_au=1.0)

    @classmethod
    def sun_at(cls, distance_au: float, *, where: str,
               distance_source: str) -> "ApparentStar":
        """The IAU nominal Sun seen from ``distance_au`` (``where`` names the
        place, ``distance_source`` says where the distance comes from)."""
        d = float(distance_au)
        if not (math.isfinite(d) and d > 0.0):
            raise ValueError(f"distance_au must be > 0, got {d}")
        return cls.of(T_SUN_K, 1.0, d,
                      label=f"Sun-like blackbody at {d:g} au ({where})",
                      source=f"{NOMINAL_SOLAR_SOURCE}; distance: {distance_source}")

    @classmethod
    def with_sun_irradiance_at_1au(cls, temperature_k: float) -> "ApparentStar":
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
        r = self.radius_rsun * R_SUN_M
        return _blackbody_luminosity(r, float(self.temperature_k)) / L_SUN_W

    def radiance(self, wavelength_nm) -> np.ndarray:
        if self.spectrum is not None:
            return np.asarray(self.spectrum.radiance(wavelength_nm))
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


def _load_catalogue() -> dict[str, Star]:
    doc = load_dataset("stars.toml")
    out: dict[str, Star] = {}
    for key, rec in doc["stars"].items():
        out[key] = Star.blackbody(
            rec["name"], temperature_k=rec.get("temperature_k"),
            radius_rsun=rec.get("radius_rsun"),
            luminosity_lsun=rec.get("luminosity_lsun"),
            mass_kg=rec.get("mass_kg"), gm_m3_s2=rec.get("gm_m3_s2"),
            kind=rec.get("kind", ""), key=key, source=rec["source"])
    return out


#: Real stars as blackbodies, from the packaged catalogue, coolest first.
STARS: dict[str, Star] = _load_catalogue()
