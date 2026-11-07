"""Explicit atmospheric composition (an additive, view-only capability).

A :class:`Composition` is a set of named species with mole fractions that
the user STATES. It is never inferred: not from a run's ``r_dry``, not from
``kappa``, not from a mean molecular mass. Many mixtures share one molar
mass, so the inverse is not defined.

Relation to the dry thermodynamics. A composition implies a mean molar mass
``M = sum x_i M_i`` and so a specific gas constant ``R_mix = R* / M``.
:func:`gas_constant_report` compares ``R_mix`` with a run's ``r_dry`` and
says whether they agree; it never changes ``r_dry`` and the solver never
reads a composition. ``cp`` is not compared in this version (it needs
sourced, temperature-dependent heat capacities). Today a composition only
supplies per-molecule scattering cross sections to the sky view; the
column's number density is ``p / (k_B T)``, which needs no composition.

Mole fractions must sum to 1 within ``SUM_TOLERANCE``; they are never
renormalized silently. Species without optical data are refused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tropoi.spatial.optics.constants import ATOMIC_WEIGHT_SOURCE, R_MOLAR
from tropoi.spatial.optics.rayleigh import SPECIES

__all__ = ["COMPOSITION_PRESETS", "Composition", "CompositionError",
           "R_DRY_TOLERANCE", "SUM_TOLERANCE", "gas_constant_report"]

SUM_TOLERANCE = 1e-6
#: Relative difference below which R_mix and r_dry are reported as agreeing
#: (a reporting threshold, not a physical one).
R_DRY_TOLERANCE = 1e-3


class CompositionError(ValueError):
    """An invalid or unsupported composition."""


@dataclass(frozen=True)
class Composition:
    """Named species -> mole fraction, with a label and a stated source."""

    mole_fractions: tuple[tuple[str, float], ...]
    label: str = "user composition"
    source: str = "stated by the user"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        items = tuple((str(k), float(v)) for k, v in
                      (self.mole_fractions.items()
                       if isinstance(self.mole_fractions, dict)
                       else self.mole_fractions))
        if not items:
            raise CompositionError("a composition needs at least one species")
        names = [k for k, _ in items]
        if len(set(names)) != len(names):
            raise CompositionError(f"repeated species in {names}")
        unknown = [k for k in names if k not in SPECIES]
        if unknown:
            raise CompositionError(
                f"no optical data for {unknown}; supported species are "
                f"{sorted(SPECIES)}. Remove them explicitly (and say so in "
                "the label) rather than expecting them to be ignored")
        for k, v in items:
            if not (math.isfinite(v) and v > 0.0):
                raise CompositionError(f"mole fraction of {k} must be > 0, "
                                       f"got {v}")
        total = sum(v for _, v in items)
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise CompositionError(
                f"mole fractions sum to {total:.9g}, not 1 (tolerance "
                f"{SUM_TOLERANCE:g}); they are never renormalized silently")
        object.__setattr__(self, "mole_fractions", items)

    @classmethod
    def of(cls, label: str = "user composition", source: str = "stated by "
           "the user", **fractions: float) -> "Composition":
        return cls(tuple(fractions.items()), label=label, source=source)

    @property
    def molar_mass_kg_per_mol(self) -> float:
        return sum(x * SPECIES[k].molar_mass_g_per_mol
                   for k, x in self.mole_fractions) / 1000.0

    @property
    def gas_constant(self) -> float:
        """R* / M (J kg^-1 K^-1) implied by this mixture."""
        return R_MOLAR / self.molar_mass_kg_per_mol

    def cross_section(self, wavelength_nm) -> np.ndarray:
        """Mole-fraction-weighted Rayleigh cross section (m^2 per molecule).

        Independent scatterers: sigma_mix = sum x_i sigma_i.
        """
        lam = np.asarray(wavelength_nm, dtype=np.float64)
        return sum(x * SPECIES[k].cross_section(lam)
                   for k, x in self.mole_fractions)

    def sources(self) -> list[str]:
        return [SPECIES[k].describe() for k, _ in self.mole_fractions] + [
            f"Molar masses: {ATOMIC_WEIGHT_SOURCE}"]

    def describe(self) -> str:
        parts = ", ".join(f"{k} {x:.6g}" for k, x in self.mole_fractions)
        return f"{self.label}: {parts} (mole fractions; {self.source})"

    def to_dict(self) -> dict:
        return {"label": self.label, "source": self.source,
                "mole_fractions": [[k, x] for k, x in self.mole_fractions],
                "molar_mass_kg_per_mol": self.molar_mass_kg_per_mol,
                "gas_constant": self.gas_constant, "notes": list(self.notes)}


def gas_constant_report(composition: Composition, r_dry: float | None, *,
                        against: str = "the run uses r_dry") -> dict:
    """Compare the mixture's R* / M with a run's ``r_dry`` (report only).

    ``status`` is ``"consistent"``, ``"inconsistent"`` or ``"no_r_dry"``.
    Nothing is changed either way: the run keeps its ``r_dry``.
    """
    r_mix = composition.gas_constant
    if r_dry is None:
        return {"status": "no_r_dry", "r_mix": r_mix, "r_dry": None,
                "relative_difference": None,
                "text": (f"Composition implies R = R*/M = {r_mix:.6g} "
                         "J kg⁻¹ K⁻¹; there is no run gas constant to "
                         "compare with.")}
    rel = (r_mix - float(r_dry)) / float(r_dry)
    ok = abs(rel) <= R_DRY_TOLERANCE
    text = (f"Composition implies R = R*/M = {r_mix:.6g} J kg⁻¹ K⁻¹; {against} "
            f"= {float(r_dry):.6g} ({rel * 100:+.3g} %). "
            + ("They agree within the 0.1 % reporting threshold."
               if ok else
               "They DISAGREE: the column was built with that gas constant, "
               "not this mixture's. The sky uses the composition only for "
               "scattering cross sections; nothing in any run is changed."))
    return {"status": "consistent" if ok else "inconsistent", "r_mix": r_mix,
            "r_dry": float(r_dry), "relative_difference": rel, "text": text}


#: Illustrative presets. Each says exactly where its numbers come from.
COMPOSITION_PRESETS: dict[str, Composition] = {
    "earth_dry_air_illustrative": Composition(
        (("N2", 0.78084), ("O2", 0.20946), ("Ar", 0.00934),
         ("CO2", 0.00036)),
        label="Earth-like dry air (illustrative)",
        source=("commonly tabulated dry-air volume fractions N2 78.084 %, "
                "O2 20.946 %, Ar 0.934 %; CO2 set to 360 ppm so the four "
                "species sum to 1. Ne, He, CH4 and other trace gases are "
                "omitted. Not a reference composition"),
        notes=("illustrative preset: not a validated reference mixture",)),
}
