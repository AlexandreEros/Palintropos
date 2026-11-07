"""Rayleigh scattering cross sections of individual gases (sourced).

For a gas whose refractive index ``n`` was measured at number density
``N_ref`` (ideal gas at the reference temperature and pressure of the
measurement), the Rayleigh cross section per molecule is

    sigma(lambda) = 24 pi^3 / (lambda^4 N_ref^2) * ((n^2 - 1)/(n^2 + 2))^2 * F_K(lambda)

(e.g. Bodhaine et al. 1999, eq. 22; Sneep & Ubachs 2005, eq. 2). The
Lorentz-Lorenz ratio is proportional to density, so sigma does not depend on
the reference state. ``F_K`` is the King correction factor (depolarization).

Refractive indices are the dispersion formulas of the refractiveindex.info
database (M. N. Polyanskiy, Sci. Data 11, 94 (2024);
https://github.com/polyanskiy/refractiveindex.info-database, files
``data/main/<gas>/nk/<record>.yml``, fetched 2026-11-07), transcribed
verbatim with their measurement conditions and valid wavelength ranges.
Wavelengths outside a record's range are refused, never extrapolated.

King factors are those of Bates (1984) as given by Bodhaine et al. (1999,
J. Atmos. Oceanic Technol. 16, 1854): eq. 5 (N2), eq. 6 (O2) and the
sentence after it (Ar 1.00, CO2 1.15), with lambda in micrometres.
**Verified 2026-11-07** against the paper's printed text (``KING_CHECK``):
the coefficients match, and the paper's Fig. 1 (F(air) about 1.080 at
0.2 um and 1.047 at 1 um) confirms the micrometre unit. Bates (1984) itself
was not re-read; the citation chain is Bates via Bodhaine. As an end-to-end
check, the illustrative dry-air preset's cross section agrees with Bodhaine
et al.'s eq. 29 fit for 360 ppm CO2 air to within 0.1 % over 400-760 nm
(tests/test_sky_view.py), although it is built from different
refractive-index records.

The ideal-gas reference density neglects compressibility (a part in 10^3
or less at these conditions). Unsupported: absorption of any kind, Raman
scattering as a separate process, pressure dependence beyond ideal gas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.spatial.optics.constants import K_B

__all__ = ["KING_CHECK", "SPECIES", "Species", "rayleigh_cross_section"]

_DB = ("refractiveindex.info database (Polyanskiy, Sci. Data 11, 94, 2024), "
       "data/main/{path}.yml, fetched 2026-11-07")
_KING = ("Bates (1984), Planet. Space Sci. 32, 785, as given by Bodhaine "
         "et al. (1999), J. Atmos. Oceanic Technol. 16, 1854, eqs. 5-6 and "
         "the text after eq. 6")
#: How the King factors were checked (shown with the sources).
KING_CHECK = ("King-factor coefficients checked 2026-11-07 against the printed "
              "text of Bodhaine et al. (1999), eqs. 5-6 and the text after "
              "eq. 6; micrometre unit confirmed by its Fig. 1. Bates (1984) "
              "not re-read.")


@dataclass(frozen=True)
class Species:
    """One gas: refractive-index formula, reference state, King factor."""

    name: str
    formula_type: int                 # refractiveindex.info formula 2 or 6
    coefficients: tuple[float, ...]
    range_um: tuple[float, float]
    ref_temperature_k: float
    ref_pressure_pa: float
    refractive_source: str
    king: tuple[float, float, float]  # F = a + b / lambda^2 + c / lambda^4 (um)
    king_source: str
    king_verified: bool
    molar_mass_g_per_mol: float

    def refractivity(self, wavelength_um) -> np.ndarray:
        """n - 1 at the record's reference state."""
        lam = np.asarray(wavelength_um, dtype=np.float64)
        lo, hi = self.range_um
        if np.any(lam < lo - 1e-12) or np.any(lam > hi + 1e-12):
            raise ValueError(
                f"{self.name}: wavelengths must lie in the measured range "
                f"{lo:g}-{hi:g} um of {self.refractive_source}; got "
                f"{float(lam.min()):g}-{float(lam.max()):g} um")
        c = self.coefficients
        if self.formula_type == 6:      # n - 1 = C1 + sum C_i / (C_i+1 - lam^-2)
            inv2 = lam ** -2
            out = np.full(lam.shape, c[0])
            for i in range(1, len(c), 2):
                out = out + c[i] / (c[i + 1] - inv2)
            return out
        if self.formula_type == 2:      # n^2 - 1 = C1 + sum C_i lam^2 / (lam^2 - C_i+1)
            l2 = lam ** 2
            n2m1 = np.full(lam.shape, c[0])
            for i in range(1, len(c), 2):
                n2m1 = n2m1 + c[i] * l2 / (l2 - c[i + 1])
            return np.sqrt(1.0 + n2m1) - 1.0
        raise ValueError(f"unsupported formula type {self.formula_type}")

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
        return (f"{self.name}: n from {self.refractive_source} "
                f"({self.ref_temperature_k:g} K, {self.ref_pressure_pa:g} Pa, "
                f"{self.range_um[0]:g}-{self.range_um[1]:g} um); King factor "
                f"{self.king_source}"
                + ("" if self.king_verified else
                   " [transcription not yet checked against the paper]"))


def _mm(**atoms) -> float:
    from tropoi.spatial.optics.constants import ATOMIC_WEIGHT_G_PER_MOL as W
    return float(sum(W[a] * k for a, k in atoms.items()))


SPECIES: dict[str, Species] = {
    "N2": Species(
        "N2", 2, (0.0, 39209.95e-8, 1146.24e-6, 18806.48e-8, 13.476e-3),
        (0.4, 1.0), 273.0, 100000.0,
        "Börzsönyi et al. (2008), Appl. Opt. 47, 4856; "
        + _DB.format(path="N2/nk/Borzsonyi"),
        (1.034, 3.17e-4, 0.0), _KING, True, _mm(N=2)),
    "O2": Species(
        "O2", 6, (1.181494e-4, 9.708931e-3, 75.4),
        (0.4, 1.8), 293.0, 101325.0,
        "Zhang, Lu & Wang (2008), Appl. Opt. 47, 3143 (with Křen 2011, "
        "Appl. Opt. 50, 6484); " + _DB.format(path="O2/nk/Zhang"),
        (1.096, 1.385e-3, 1.448e-4), _KING, True, _mm(O=2)),
    "Ar": Species(
        "Ar", 2, (0.0, 20332.29e-8, 206.12e-6, 34458.31e-8, 8.066e-3),
        (0.4, 1.0), 273.0, 100000.0,
        "Börzsönyi et al. (2008), Appl. Opt. 47, 4856; "
        + _DB.format(path="Ar/nk/Borzsonyi"),
        (1.0, 0.0, 0.0), _KING, True, _mm(Ar=1)),
    "CO2": Species(
        "CO2", 6, (0.0, 6.99100e-2, 166.175, 1.44720e-3, 79.609,
                   6.42941e-5, 56.3064, 5.21306e-5, 46.0196, 1.46847e-6,
                   0.0584738),
        (0.1807, 1.6945), 273.15, 101325.0,
        "Bideau-Mehu et al. (1973), Opt. Commun. 9, 432; "
        + _DB.format(path="CO2/nk/Bideau-Mehu"),
        (1.15, 0.0, 0.0), _KING, True, _mm(C=1, O=2)),
}


def rayleigh_cross_section(species: str, wavelength_nm) -> np.ndarray:
    """Rayleigh cross section (m^2 per molecule) of one named species."""
    try:
        return SPECIES[species].cross_section(wavelength_nm)
    except KeyError:
        raise ValueError(f"no optical data for species {species!r}; known: "
                         f"{sorted(SPECIES)}") from None
