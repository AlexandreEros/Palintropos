"""Rayleigh scattering cross sections of individual gases (sourced data).

For a gas whose refractive index ``n`` was measured at number density
``N_ref`` (ideal gas at the reference temperature and pressure of the
measurement), the Rayleigh cross section per molecule is

    sigma(lambda) = 24 pi^3 / (lambda^4 N_ref^2) * ((n^2 - 1)/(n^2 + 2))^2 * F_K(lambda)

(e.g. Bodhaine et al. 1999, eq. 22; Sneep & Ubachs 2005, eq. 2). The
Lorentz-Lorenz ratio is proportional to density, so sigma does not depend on
the reference state. ``F_K`` is the King correction factor (depolarization).

The measured records — dispersion coefficients, their validity ranges and
reference states, King coefficients, and every citation — are packaged data
(``tropoi/physics/data/rayleigh.toml``); this module holds the equations.
A record names its dispersion model by the refractiveindex.info formula
number; only formulas 2 and 6 are implemented. A species may carry several
records with different validity ranges; by default the first record that
covers the requested wavelengths is used, and wavelengths outside every
record's range are refused, never extrapolated.

As an end-to-end check, the illustrative dry-air preset's cross section
agrees with Bodhaine et al.'s eq. 29 fit for 360 ppm CO2 air to within
0.1 % over 400-760 nm (tests/test_sky_view.py), although it is built from
different refractive-index records.

The ideal-gas reference density neglects compressibility (a part in 10^3
or less at these conditions). Unsupported: absorption of any kind, Raman
scattering as a separate process, pressure dependence beyond ideal gas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.physics.composition import Composition
from tropoi.physics.constants import K_B
from tropoi.physics.species import SPECIES, load_dataset

__all__ = ["KING_CHECK", "OPTICS", "OpticalRecord", "composition_cross_section",
           "optical_records", "rayleigh_cross_section"]


@dataclass(frozen=True)
class OpticalRecord:
    """One measured refractive-index record of one gas, plus its King
    factor: dispersion formula (by identifier), coefficients, validity
    range, reference state and citations."""

    species: str
    formula: int                      # refractiveindex.info formula 2 or 6
    coefficients: tuple[float, ...]
    range_um: tuple[float, float]
    ref_temperature_k: float
    ref_pressure_pa: float
    refractive_source: str
    king: tuple[float, float, float]  # F = a + b / lambda^2 + c / lambda^4 (um)
    king_source: str
    king_verified: bool

    def covers(self, wavelength_um) -> bool:
        lam = np.asarray(wavelength_um, dtype=np.float64)
        lo, hi = self.range_um
        return bool(np.all(lam >= lo - 1e-12) and np.all(lam <= hi + 1e-12))

    def refractivity(self, wavelength_um) -> np.ndarray:
        """n - 1 at the record's reference state."""
        lam = np.asarray(wavelength_um, dtype=np.float64)
        lo, hi = self.range_um
        if not self.covers(lam):
            raise ValueError(
                f"{self.species}: wavelengths must lie in the measured range "
                f"{lo:g}-{hi:g} um of {self.refractive_source}; got "
                f"{float(lam.min()):g}-{float(lam.max()):g} um")
        c = self.coefficients
        if self.formula == 6:           # n - 1 = C1 + sum C_i / (C_i+1 - lam^-2)
            inv2 = lam ** -2
            out = np.full(lam.shape, c[0])
            for i in range(1, len(c), 2):
                out = out + c[i] / (c[i + 1] - inv2)
            return out
        if self.formula == 2:           # n^2 - 1 = C1 + sum C_i lam^2 / (lam^2 - C_i+1)
            l2 = lam ** 2
            n2m1 = np.full(lam.shape, c[0])
            for i in range(1, len(c), 2):
                n2m1 = n2m1 + c[i] * l2 / (l2 - c[i + 1])
            return np.sqrt(1.0 + n2m1) - 1.0
        raise ValueError(f"unsupported dispersion formula {self.formula}")

    def king_factor(self, wavelength_um) -> np.ndarray:
        lam = np.asarray(wavelength_um, dtype=np.float64)
        a, b, c = self.king
        return a + b / lam ** 2 + c / lam ** 4

    def cross_section(self, wavelength_nm) -> np.ndarray:
        """Rayleigh cross section per molecule (m^2)."""
        lam_um = np.asarray(wavelength_nm, dtype=np.float64) / 1000.0
        n = 1.0 + self.refractivity(lam_um)
        n_ref = self.ref_pressure_pa / (K_B * self.ref_temperature_k)
        lam_m = lam_um * 1e-6
        ll = (n ** 2 - 1.0) / (n ** 2 + 2.0)
        return (24.0 * math.pi ** 3 / (lam_m ** 4 * n_ref ** 2) * ll ** 2
                * self.king_factor(lam_um))

    def describe(self) -> str:
        return (f"{self.species}: n from {self.refractive_source} "
                f"({self.ref_temperature_k:g} K, {self.ref_pressure_pa:g} Pa, "
                f"{self.range_um[0]:g}-{self.range_um[1]:g} um); King factor "
                f"{self.king_source}"
                + ("" if self.king_verified else
                   " [transcription not yet checked against the paper]"))


def _load_records() -> tuple[dict[str, tuple[OpticalRecord, ...]], str]:
    doc = load_dataset("rayleigh.toml")
    king_source = doc["king_source"]
    out: dict[str, tuple[OpticalRecord, ...]] = {}
    for name, body in doc["species"].items():
        if name not in SPECIES:
            raise ValueError(f"rayleigh.toml lists {name!r}, which "
                             "species.toml does not identify")
        records = tuple(OpticalRecord(
            name, int(rec["formula"]),
            tuple(float(v) for v in rec["coefficients"]),
            (float(rec["range_um"][0]), float(rec["range_um"][1])),
            float(rec["ref_temperature_k"]), float(rec["ref_pressure_pa"]),
            rec["refractive_source"],
            (float(rec["king"][0]), float(rec["king"][1]),
             float(rec["king"][2])),
            king_source, bool(rec["king_verified"]))
            for rec in body["records"])
        if not records:
            raise ValueError(f"rayleigh.toml lists {name!r} with no records")
        out[name] = records
    return out, doc["king_check"]


#: Optical records by species name (packaged data); each species may carry
#: several records with different validity ranges.
OPTICS: dict[str, tuple[OpticalRecord, ...]]
OPTICS, KING_CHECK = _load_records()


def optical_records(species: str) -> tuple[OpticalRecord, ...]:
    """Every optical record of one named species."""
    try:
        return OPTICS[species]
    except KeyError:
        raise ValueError(f"no optical data for species {species!r}; known: "
                         f"{sorted(OPTICS)}") from None


def rayleigh_cross_section(species: str, wavelength_nm) -> np.ndarray:
    """Rayleigh cross section (m^2 per molecule) of one named species, from
    the first record covering the requested wavelengths."""
    records = optical_records(species)
    lam_um = np.asarray(wavelength_nm, dtype=np.float64) / 1000.0
    for record in records:
        if record.covers(lam_um):
            return record.cross_section(wavelength_nm)
    # No record covers the request: raise the first record's range error.
    return records[0].cross_section(wavelength_nm)


def composition_cross_section(composition: Composition, wavelength_nm
                              ) -> np.ndarray:
    """Mole-fraction-weighted Rayleigh cross section (m^2 per molecule).

    Independent scatterers: sigma_mix = sum x_i sigma_i. Refuses
    compositions containing species without optical data."""
    composition.require_coverage(OPTICS, "optical")
    lam = np.asarray(wavelength_nm, dtype=np.float64)
    return sum(x * rayleigh_cross_section(k, lam)
               for k, x in composition.mole_fractions)
