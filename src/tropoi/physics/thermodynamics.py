"""Ideal-gas thermodynamics of explicit compositions.

Given a :class:`~tropoi.physics.composition.Composition`, this module
provides the two numbers the hydrostatic primitive equations need — the
specific gas constant ``R`` (from the composition itself) and the isobaric
specific heat ``c_p`` — and, for checking whether a constant ``c_p`` is
defensible in a given regime, the temperature-dependent ``c_p(T)`` of the
mixture.

Heat capacities are computed, not tabulated: translation (5/2 R), classical
rotation (R for linear, 3/2 R for nonlinear molecules) and one Einstein
function per vibrational mode from the fundamental wavenumbers in the
packaged dataset ``tropoi/physics/data/thermodynamics.toml`` (the
rigid-rotor/harmonic-oscillator approximation, RRHO). Hydrogen is the
exception: its rotational spacing (``theta_rot`` = 85.3 K) is comparable to
giant-planet temperatures, so its rotational heat capacity is summed
explicitly over the ortho and para levels, either for a frozen 3:1 "normal"
mixture or for an equilibrium mixture (:func:`hydrogen_rotational_cv`).
Measured agreement with reference 298.15 K values is asserted in
``tests/test_physics_thermodynamics.py``; RRHO neglects anharmonicity and
centrifugal distortion, which is a per-cent-level error below ~1000 K.

Coverage is checked, never assumed: a composition containing a species
without a thermodynamic record is refused by name
(``Composition.require_coverage``). "Dry" here means no phase changes are
modelled; condensable species may still be listed as ordinary gases with
fixed mole fractions.

Import-light: NumPy only. Functions accept NumPy or CuPy temperature
arrays; the packaged data is loaded once at import, outside any hot loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tropoi.physics.composition import Composition
from tropoi.physics.constants import C2_CM_K, R_UNIVERSAL
from tropoi.physics.species import load_dataset

__all__ = ["R_UNIVERSAL", "THERMO", "ThermoRecord", "cp_molar",
           "heat_capacity", "heat_capacity_range", "hydrogen_rotational_cv",
           "kappa"]


def _xp(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: WPS433
        return cupy
    return np


@dataclass(frozen=True)
class ThermoRecord:
    """RRHO inputs of one gas: rotational degrees of freedom (0 atom, 2
    linear, 3 nonlinear) and vibrational fundamentals (cm^-1), each listed
    once per degenerate mode. ``quantum_rotor_theta_k`` is set for H2 only."""

    name: str
    rotational_dof: int
    vibrations_cm: tuple[float, ...] = ()
    quantum_rotor_theta_k: float | None = None

    def cp_molar(self, temperature, *, ortho_para: str = "normal"):
        """Isobaric molar heat capacity (J mol^-1 K^-1) at ``temperature``
        (K)."""
        xp = _xp(temperature)
        t = xp.asarray(temperature, dtype=np.float64)
        cv = 1.5 + xp.zeros_like(t)
        if self.quantum_rotor_theta_k is not None:
            cv = cv + hydrogen_rotational_cv(t, self.quantum_rotor_theta_k,
                                             ortho_para=ortho_para)
        else:
            cv = cv + 0.5 * self.rotational_dof
        for nu in self.vibrations_cm:
            x = (C2_CM_K * nu) / t
            # Einstein function x^2 e^x / (e^x - 1)^2, written to stay finite
            # for large x (cold gas): x^2 e^-x / (1 - e^-x)^2.
            e = xp.exp(-x)
            cv = cv + x * x * e / ((1.0 - e) ** 2)
        return R_UNIVERSAL * (cv + 1.0)


def hydrogen_rotational_cv(temperature, theta_rot: float = 85.3, *,
                           ortho_para: str = "normal", j_max: int = 24):
    """Rotational heat capacity of H2 in units of R.

    ``"normal"``: ortho and para are separate, non-interconverting gases in
    the high-temperature 3:1 ratio (the laboratory gas, and the usual
    assumption for fast vertical mixing). ``"equilibrium"``: the ortho/para
    ratio follows the temperature, which adds the conversion heat and
    produces the well-known maximum near 50 K. Giant-planet tropospheres lie
    between the two; this is a real modelling uncertainty, not a numerical
    one. Levels ``E_J = J(J+1) k theta_rot`` with degeneracy ``2J+1``; even J
    is para (nuclear weight 1), odd J is ortho (nuclear weight 3).
    """
    if ortho_para not in ("normal", "equilibrium"):
        raise ValueError(f"ortho_para must be 'normal' or 'equilibrium', got {ortho_para!r}")
    xp = _xp(temperature)
    t = xp.asarray(temperature, dtype=np.float64)
    j = np.arange(j_max + 1)
    energy = xp.asarray(j * (j + 1) * theta_rot, dtype=np.float64)   # in K
    degeneracy = 2 * j + 1

    def cv_of(weights):
        w = xp.asarray(weights, dtype=np.float64)
        x = energy / t[..., None]
        boltz = w * xp.exp(-x)
        z = boltz.sum(axis=-1)
        e1 = (boltz * x).sum(axis=-1) / z
        e2 = (boltz * x * x).sum(axis=-1) / z
        return e2 - e1 * e1

    para = degeneracy * (j % 2 == 0)
    ortho = degeneracy * (j % 2 == 1)
    if ortho_para == "equilibrium":
        return cv_of(para + 3.0 * ortho)
    return 0.25 * cv_of(para) + 0.75 * cv_of(ortho)


def _load_records() -> dict[str, ThermoRecord]:
    doc = load_dataset("thermodynamics.toml")
    out: dict[str, ThermoRecord] = {}
    for name, rec in doc["species"].items():
        theta = rec.get("quantum_rotor_theta_k")
        out[name] = ThermoRecord(
            name, int(rec["rotational_dof"]),
            tuple(float(v) for v in rec.get("vibrations_cm", ())),
            None if theta is None else float(theta))
    return out


#: Thermodynamic records by species name (packaged data).
THERMO: dict[str, ThermoRecord] = _load_records()


def cp_molar(species: str, temperature, *, ortho_para: str = "normal"):
    """Isobaric molar heat capacity (J mol^-1 K^-1) of one named species."""
    try:
        record = THERMO[species]
    except KeyError:
        raise ValueError(f"no thermodynamic data for species {species!r}; "
                         f"covered: {sorted(THERMO)}") from None
    return record.cp_molar(temperature, ortho_para=ortho_para)


def heat_capacity(composition: Composition, temperature):
    """Isobaric specific heat of the mixture, J kg^-1 K^-1, at T (K).

    Refuses compositions containing species without thermodynamic data."""
    composition.require_coverage(THERMO, "thermodynamic")
    total = 0.0
    for s, x in composition.mole_fractions:
        total = total + x * THERMO[s].cp_molar(
            temperature, ortho_para=composition.ortho_para)
    return total / composition.molar_mass


def kappa(composition: Composition, temperature):
    """R / c_p(T) of the mixture."""
    return composition.gas_constant / heat_capacity(composition, temperature)


def heat_capacity_range(composition: Composition, t_low: float,
                        t_high: float, n: int = 201) -> dict:
    """Extremes of c_p over a temperature interval and the relative spread
    ``(max - min) / mean``: the error scale of a constant-c_p model that
    spans the interval."""
    t = np.linspace(float(t_low), float(t_high), int(n))
    c = np.asarray(heat_capacity(composition, t))
    return {"t_low": float(t_low), "t_high": float(t_high),
            "cp_min": float(c.min()), "cp_max": float(c.max()),
            "cp_mean": float(c.mean()),
            "relative_spread": float((c.max() - c.min()) / c.mean())}
