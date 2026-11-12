"""Explicit atmospheric composition: named species with stated abundances.

A :class:`Composition` identifies species and their mole fractions. It is
STATED, never inferred: not from a run's ``r_dry``, not from ``kappa``, not
from a mean molecular mass (many mixtures share one molar mass, so the
inverse is not defined). Whether a particular calculation supports the
listed species is a separate question: the composition only requires each
name to be a known species identity (:mod:`tropoi.physics.species`), and
each property consumer checks the coverage it actually needs with
:meth:`Composition.require_coverage`, refusing missing species by name
rather than silently ignoring them.

Validation is strict by default: mole (or mass) fractions must be positive,
finite and sum to 1 within :data:`SUM_TOLERANCE` (a numerical tolerance for
float round-off, not an invitation to drop constituents); nothing is
renormalized silently. For mixtures stated as rounded numbers whose total
is not 1 (for example "95.1 % CO2, 2.6 % N2, 1.9 % Ar, ..."), the explicit
factory :meth:`Composition.normalized` rescales them and records the stated
fractions and their total in ``normalization``, so the original statement
is preserved. It still refuses totals further than
:data:`NORMALIZATION_GUARD` from 1: an input missing a major constituent is
an error, not something to rescale.

Mole/mass conversion is explicit both ways:
:meth:`Composition.from_mass_fractions` and
:attr:`Composition.mass_fractions`.

The mixture's mean molar mass ``M = sum x_i M_i`` and specific gas constant
``R = R*/M`` follow from the one packaged molar-mass table.
:func:`gas_constant_report` compares that ``R`` with a run's ``r_dry`` and
says whether they agree; it never changes ``r_dry`` and no solver ever
reads a composition to change its own constants.

``COMPOSITIONS`` holds the packaged presets
(``tropoi/physics/data/mixtures.toml``): the climate mixtures used to
configure idealized experiments and the illustrative sky mixtures, each
with its own provenance. They are deliberately distinct; two presets that
both describe "Mars" are different statements with different sources.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from tropoi.physics.constants import R_UNIVERSAL
from tropoi.physics.species import SPECIES, load_dataset

__all__ = ["COMPOSITIONS", "Composition", "CompositionError",
           "NORMALIZATION_GUARD", "R_DRY_TOLERANCE", "SUM_TOLERANCE",
           "gas_constant_report"]

#: Stated fractions must sum to 1 within this (float round-off headroom).
SUM_TOLERANCE = 1e-6
#: The explicit normalization factory refuses totals further than this from
#: 1: a composition missing a major constituent is incomplete, not rounded.
NORMALIZATION_GUARD = 0.02
#: Relative difference below which R_mix and r_dry are reported as agreeing
#: (a reporting threshold, not a physical one).
R_DRY_TOLERANCE = 1e-3

_ORTHO_PARA = ("normal", "equilibrium")


class CompositionError(ValueError):
    """An invalid composition, or one a calculation cannot support."""


def _as_items(fractions) -> tuple[tuple[str, float], ...]:
    items = tuple((str(k), float(v)) for k, v in
                  (fractions.items() if isinstance(fractions, Mapping)
                   else fractions))
    return items


def _check_names(names: Iterable[str]) -> None:
    names = list(names)
    if not names:
        raise CompositionError("a composition needs at least one species")
    if len(set(names)) != len(names):
        raise CompositionError(f"repeated species in {names}")
    unknown = [k for k in names if k not in SPECIES]
    if unknown:
        raise CompositionError(
            f"unknown species {unknown}; known species identities are "
            f"{sorted(SPECIES)}")


@dataclass(frozen=True)
class Composition:
    """Named species -> mole fraction, with a label and a stated source."""

    mole_fractions: tuple[tuple[str, float], ...]
    label: str = "user composition"
    source: str = "stated by the user"
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Hydrogen nuclear-spin assumption consumed by the thermodynamics
    #: ("normal" 3:1 ortho:para, or "equilibrium"); irrelevant without H2.
    ortho_para: str = "normal"
    #: None for strictly stated fractions; for :meth:`normalized` mixtures,
    #: the stated fractions and their total, preserved as provenance.
    normalization: dict | None = None

    def __post_init__(self) -> None:
        items = _as_items(self.mole_fractions)
        _check_names(k for k, _ in items)
        for k, v in items:
            if not (math.isfinite(v) and v > 0.0):
                raise CompositionError(f"mole fraction of {k} must be > 0, "
                                       f"got {v}")
        total = sum(v for _, v in items)
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise CompositionError(
                f"mole fractions sum to {total:.9g}, not 1 (tolerance "
                f"{SUM_TOLERANCE:g}); they are never renormalized silently. "
                "For a rounded statement, use Composition.normalized, which "
                "records what was stated")
        if self.ortho_para not in _ORTHO_PARA:
            raise CompositionError(f"ortho_para must be one of {_ORTHO_PARA},"
                                   f" got {self.ortho_para!r}")
        object.__setattr__(self, "mole_fractions", items)
        object.__setattr__(self, "notes", tuple(self.notes))

    # -- factories -----------------------------------------------------------

    @classmethod
    def of(cls, label: str = "user composition",
           source: str = "stated by the user", **fractions: float
           ) -> "Composition":
        return cls(tuple(fractions.items()), label=label, source=source)

    @classmethod
    def normalized(cls, fractions, *, label: str, source: str,
                   notes: tuple[str, ...] = (), ortho_para: str = "normal"
                   ) -> "Composition":
        """EXPLICIT normalization of rounded mole fractions.

        The stated fractions and their total are preserved in
        ``normalization``; totals further than :data:`NORMALIZATION_GUARD`
        from 1 are refused (state the composition completely instead)."""
        items = _as_items(fractions)
        _check_names(k for k, _ in items)
        for k, v in items:
            if not (math.isfinite(v) and v > 0.0):
                raise CompositionError(f"mole fraction of {k} must be > 0, "
                                       f"got {v}")
        total = sum(v for _, v in items)
        if abs(total - 1.0) > NORMALIZATION_GUARD:
            raise CompositionError(
                f"stated mole fractions of {label!r} sum to {total:.4f}; "
                "that is not a rounding remainder. State the composition "
                f"completely (within {NORMALIZATION_GUARD:g} of 1)")
        return cls(tuple((k, v / total) for k, v in items), label=label,
                   source=source, notes=tuple(notes), ortho_para=ortho_para,
                   normalization={"stated_mole_fractions": [[k, v] for k, v
                                                            in items],
                                  "stated_total": total})

    @classmethod
    def from_mass_fractions(cls, label: str = "user composition",
                            source: str = "stated by the user",
                            **fractions: float) -> "Composition":
        """Mass fractions w_i (summing to 1, never renormalized) to mole
        fractions x_i = (w_i / M_i) / sum_j (w_j / M_j)."""
        items = _as_items(fractions)
        _check_names(k for k, _ in items)
        for k, w in items:
            if not (math.isfinite(w) and w > 0.0):
                raise CompositionError(f"mass fraction of {k} must be > 0, "
                                       f"got {w}")
        total = sum(w for _, w in items)
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise CompositionError(
                f"mass fractions sum to {total:.9g}, not 1 (tolerance "
                f"{SUM_TOLERANCE:g}); they are never renormalized silently")
        moles = tuple((k, w / SPECIES[k].molar_mass) for k, w in items)
        n = sum(v for _, v in moles)
        return cls(tuple((k, v / n) for k, v in moles), label=label,
                   source=source)

    # -- identity and conversion ----------------------------------------------

    @property
    def species_names(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.mole_fractions)

    def mole_fraction(self, species: str) -> float:
        return dict(self.mole_fractions).get(species, 0.0)

    @property
    def mass_fractions(self) -> tuple[tuple[str, float], ...]:
        """w_i = x_i M_i / sum_j x_j M_j, in the stated order."""
        m = [(k, x * SPECIES[k].molar_mass) for k, x in self.mole_fractions]
        total = sum(v for _, v in m)
        return tuple((k, v / total) for k, v in m)

    def mass_fraction(self, species: str) -> float:
        return dict(self.mass_fractions).get(species, 0.0)

    @property
    def molar_mass(self) -> float:
        """Mean molar mass sum x_i M_i, kg/mol."""
        return sum(x * SPECIES[k].molar_mass for k, x in self.mole_fractions)

    @property
    def molar_mass_kg_per_mol(self) -> float:
        return self.molar_mass

    @property
    def molar_mass_g_per_mol(self) -> float:
        return self.molar_mass * 1000.0

    @property
    def gas_constant(self) -> float:
        """R* / M (J kg^-1 K^-1) implied by this mixture."""
        return R_UNIVERSAL / self.molar_mass

    # -- property coverage -----------------------------------------------------

    def require_coverage(self, supported: Iterable[str], dataset: str
                         ) -> None:
        """Refuse, by name, constituents absent from ``supported``.

        Property consumers call this with the species their dataset actually
        covers (``dataset`` names it in the error); unsupported constituents
        are never silently ignored."""
        supported = set(supported)
        missing = [k for k, _ in self.mole_fractions if k not in supported]
        if missing:
            raise CompositionError(
                f"no {dataset} data for {missing}; supported species are "
                f"{sorted(supported)}. Remove them explicitly (and say so in "
                "the label) rather than expecting them to be ignored")

    # -- description -----------------------------------------------------------

    def sources(self) -> list[str]:
        from tropoi.physics.species import ATOMIC_WEIGHT_SOURCE
        return [f"{self.label}: {self.source}",
                f"Molar masses: {ATOMIC_WEIGHT_SOURCE}"]

    def describe(self) -> str:
        parts = ", ".join(f"{k} {x:.6g}" for k, x in self.mole_fractions)
        return f"{self.label}: {parts} (mole fractions; {self.source})"

    def to_dict(self) -> dict:
        out = {"label": self.label, "source": self.source,
               "mole_fractions": [[k, x] for k, x in self.mole_fractions],
               "molar_mass_kg_per_mol": self.molar_mass,
               "gas_constant": self.gas_constant, "notes": list(self.notes)}
        if self.ortho_para != "normal":
            out["ortho_para"] = self.ortho_para
        if self.normalization is not None:
            out["normalization"] = {
                "stated_mole_fractions":
                    [list(p) for p in
                     self.normalization["stated_mole_fractions"]],
                "stated_total": self.normalization["stated_total"]}
        return out


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


def _load_presets() -> dict[str, Composition]:
    doc = load_dataset("mixtures.toml")
    out: dict[str, Composition] = {}
    for key, rec in doc["mixtures"].items():
        kwargs = {"label": rec.get("label", key), "source": rec["source"],
                  "notes": tuple(rec.get("notes", ())),
                  "ortho_para": rec.get("ortho_para", "normal")}
        fractions = tuple(rec["mole_fractions"].items())
        if rec.get("normalize", False):
            out[key] = Composition.normalized(fractions, **kwargs)
        else:
            out[key] = Composition(fractions, **kwargs)
    return out


#: Packaged preset compositions, by key. Climate mixtures (used by
#: tropoi.physics.planets) and illustrative sky mixtures are distinct
#: presets with distinct provenance, even where names look alike.
COMPOSITIONS: dict[str, Composition] = _load_presets()
