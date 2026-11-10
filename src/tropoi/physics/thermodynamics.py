"""Ideal-gas thermodynamics of prescribed atmospheric compositions.

A :class:`GasMixture` is a fixed list of species and mole fractions. It
provides the two numbers the hydrostatic primitive equations need — the
specific gas constant ``R`` and the isobaric specific heat ``c_p`` — and, for
checking whether a constant ``c_p`` is defensible in a given regime, the
temperature-dependent ``c_p(T)`` of the mixture.

Heat capacities are computed, not tabulated: translation (5/2 R), classical
rotation (R for linear, 3/2 R for nonlinear molecules) and one Einstein
function per vibrational mode from the fundamental wavenumbers below (the
rigid-rotor/harmonic-oscillator approximation, RRHO). Hydrogen is the
exception: its rotational spacing (``theta_rot`` = 85.3 K) is comparable to
giant-planet temperatures, so its rotational heat capacity is summed
explicitly over the ortho and para levels, either for a frozen 3:1 "normal"
mixture or for an equilibrium mixture (Section "Hydrogen" below). Measured
agreement with reference 298.15 K values is asserted in
``tests/test_physics_thermodynamics.py``; RRHO neglects anharmonicity and
centrifugal distortion, which is a per-cent-level error below ~1000 K.

"Dry" here means no phase changes are modelled. Condensable species (H2O,
CO2, CH4, NH3) may still be listed as ordinary gases with fixed mole
fractions.

Import-light: NumPy only. Functions accept NumPy or CuPy temperature arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

__all__ = ["R_UNIVERSAL", "Species", "SPECIES", "GasMixture",
           "hydrogen_rotational_cv", "MIXTURES"]

#: Molar gas constant (exact SI value), J mol^-1 K^-1.
R_UNIVERSAL = 8.31446261815324
#: Second radiation constant hc/k_B in cm K (exact SI): converts a
#: wavenumber in cm^-1 to a characteristic temperature.
C2_CM_K = 1.438776877


def _xp(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: WPS433
        return cupy
    return np


@dataclass(frozen=True)
class Species:
    """One gas: molar mass (kg/mol), rotational degrees of freedom (0 atom,
    2 linear, 3 nonlinear) and vibrational fundamentals (cm^-1), each listed
    once per degenerate mode."""

    name: str
    molar_mass: float
    rotational_dof: int
    vibrations_cm: tuple[float, ...] = ()
    quantum_rotor_theta: float | None = None     # K; set for H2 only

    def cp_molar(self, temperature, *, ortho_para: str = "normal"):
        """Isobaric molar heat capacity (J mol^-1 K^-1) at ``temperature`` (K)."""
        xp = _xp(temperature)
        t = xp.asarray(temperature, dtype=np.float64)
        cv = 1.5 + xp.zeros_like(t)
        if self.quantum_rotor_theta is not None:
            cv = cv + hydrogen_rotational_cv(t, self.quantum_rotor_theta,
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


#: Molar masses (kg/mol) from standard atomic weights; vibrational
#: fundamentals (cm^-1) are the usual spectroscopic values, degenerate modes
#: repeated. NH3's inversion doubling is ignored (RRHO).
SPECIES: dict[str, Species] = {s.name: s for s in (
    Species("H2", 2.01588e-3, 2, (4161.2,), quantum_rotor_theta=85.3),
    Species("He", 4.002602e-3, 0),
    Species("Ar", 39.948e-3, 0),
    Species("N2", 28.0134e-3, 2, (2329.9,)),
    Species("O2", 31.9988e-3, 2, (1556.4,)),
    Species("CO", 28.0101e-3, 2, (2143.3,)),
    Species("CO2", 44.0095e-3, 2, (667.4, 667.4, 1333.0, 2349.1)),
    Species("H2O", 18.01528e-3, 3, (1594.7, 3657.1, 3755.9)),
    Species("CH4", 16.04246e-3, 3, (1306.0,) * 3 + (1534.0,) * 2 + (2916.5,)
            + (3019.5,) * 3),
    Species("NH3", 17.03052e-3, 3, (950.0,) + (1627.0,) * 2 + (3337.0,)
            + (3444.0,) * 2),
    Species("SO2", 64.0638e-3, 3, (517.7, 1151.4, 1362.0)),
)}


@dataclass(frozen=True)
class GasMixture:
    """A fixed composition: species name -> mole fraction (normalized on
    construction; the stated fractions must sum to within 2 % of 1, so a
    forgotten major constituent is an error and not silently renormalized)."""

    mole_fractions: Mapping[str, float]
    name: str = "mixture"
    ortho_para: str = "normal"

    def __post_init__(self):
        fr = {k: float(v) for k, v in dict(self.mole_fractions).items()}
        unknown = sorted(set(fr) - set(SPECIES))
        if unknown:
            raise ValueError(f"unknown species {unknown}; known: {sorted(SPECIES)}")
        if not fr or any((not math.isfinite(v)) or v < 0.0 for v in fr.values()):
            raise ValueError("mole fractions must be finite and >= 0")
        total = sum(fr.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"mole fractions of {self.name!r} sum to {total:.4f}; state the "
                "composition completely (within 2 % of 1)")
        object.__setattr__(self, "mole_fractions",
                           {k: v / total for k, v in sorted(fr.items())})

    @property
    def molar_mass(self) -> float:
        """Mean molar mass, kg/mol."""
        return sum(x * SPECIES[s].molar_mass for s, x in self.mole_fractions.items())

    @property
    def gas_constant(self) -> float:
        """Specific gas constant R = R*/M, J kg^-1 K^-1."""
        return R_UNIVERSAL / self.molar_mass

    def mass_fraction(self, species: str) -> float:
        x = self.mole_fractions.get(species, 0.0)
        return x * SPECIES[species].molar_mass / self.molar_mass

    def cp(self, temperature):
        """Isobaric specific heat of the mixture, J kg^-1 K^-1, at T (K)."""
        total = 0.0
        for s, x in self.mole_fractions.items():
            total = total + x * SPECIES[s].cp_molar(temperature,
                                                    ortho_para=self.ortho_para)
        return total / self.molar_mass

    def kappa(self, temperature):
        """R / c_p(T)."""
        return self.gas_constant / self.cp(temperature)

    def cp_range(self, t_low: float, t_high: float, n: int = 201) -> dict:
        """Extremes of c_p over a temperature interval and the relative
        spread ``(max - min) / mean``: the error scale of a constant-c_p
        model that spans the interval."""
        t = np.linspace(float(t_low), float(t_high), int(n))
        c = np.asarray(self.cp(t))
        return {"t_low": float(t_low), "t_high": float(t_high),
                "cp_min": float(c.min()), "cp_max": float(c.max()),
                "cp_mean": float(c.mean()),
                "relative_spread": float((c.max() - c.min()) / c.mean())}

    def as_dict(self) -> dict:
        return {"name": self.name, "mole_fractions": dict(self.mole_fractions),
                "ortho_para": self.ortho_para}


#: Representative bulk compositions (mole fractions). These are prescribed
#: round numbers for idealized configurations, not retrievals: trace
#: radiatively active gases are added per experiment.
MIXTURES: dict[str, GasMixture] = {m.name: m for m in (
    GasMixture({"N2": 0.7809, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}, "earth_dry_air"),
    GasMixture({"CO2": 0.951, "N2": 0.026, "Ar": 0.019, "O2": 0.002, "CO": 0.001}, "mars"),
    GasMixture({"CO2": 0.965, "N2": 0.035}, "venus"),
    GasMixture({"N2": 0.95, "CH4": 0.049, "H2": 0.001}, "titan"),
    GasMixture({"H2": 0.862, "He": 0.136, "CH4": 0.002}, "jupiter"),
    GasMixture({"H2": 0.88, "He": 0.115, "CH4": 0.005}, "saturn"),
    GasMixture({"N2": 0.9996, "CO2": 0.0004}, "thai_ben1"),
    GasMixture({"CO2": 1.0}, "thai_ben2"),
)}
