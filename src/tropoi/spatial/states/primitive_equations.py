"""Primitive-equation state description: one packed (3K+1, L+1, L+1) stack.

Rows ``[zeta_1..zeta_K, delta_1..delta_K, T_1..T_K, ln p_s]`` top to bottom.
Split verbatim from the PE core (now
:mod:`tropoi.temporal.tendencies.primitive_equations`, which re-exports
every name here), together with the exact resting-state constructor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cupy as cp

PROGNOSTICS = ("zeta", "delta", "temperature", "ln_ps")


class PrimitiveEquationsStateError(ValueError):
    """A primitive-equation state violated a hard physical/numerical rule."""


@dataclass
class PrimitiveEquationsState:
    """Spectral PE state: one (3*nlev+1, l_max+1, l_max+1) complex stack.

    Rows 0..K-1 = zeta (top to bottom), K..2K-1 = delta, 2K..3K-1 = T,
    row 3K = ln(p_s). ``nlev`` is inferred from the leading axis.
    """

    coeffs: cp.ndarray

    def __post_init__(self) -> None:
        if self.coeffs.ndim != 3 or (self.coeffs.shape[0] - 1) % 3 != 0 \
                or self.coeffs.shape[0] < 4 \
                or self.coeffs.shape[1] != self.coeffs.shape[2]:
            raise PrimitiveEquationsStateError(
                "primitive-equation coefficients must have shape "
                f"(3*nlev+1, l_max+1, l_max+1) with nlev >= 1, got "
                f"{self.coeffs.shape}")

    @property
    def nlev(self) -> int:
        return (self.coeffs.shape[0] - 1) // 3

    @property
    def zeta(self) -> cp.ndarray:
        """(nlev, l_max+1, l_max+1) relative-vorticity coefficients (s^-1)."""
        return self.coeffs[0:self.nlev]

    @property
    def delta(self) -> cp.ndarray:
        """(nlev, l_max+1, l_max+1) divergence coefficients (s^-1)."""
        return self.coeffs[self.nlev:2 * self.nlev]

    @property
    def temperature(self) -> cp.ndarray:
        """(nlev, l_max+1, l_max+1) full-temperature coefficients (K)."""
        return self.coeffs[2 * self.nlev:3 * self.nlev]

    @property
    def ln_ps(self) -> cp.ndarray:
        """(l_max+1, l_max+1) ln(p_s) coefficients (p_s in Pa)."""
        return self.coeffs[3 * self.nlev]

    @classmethod
    def from_fields(cls, zeta_lm: cp.ndarray, delta_lm: cp.ndarray,
                    temperature_lm: cp.ndarray,
                    ln_ps_lm: cp.ndarray) -> "PrimitiveEquationsState":
        """Assemble a state from per-variable coefficient arrays.

        ``zeta_lm``, ``delta_lm``, ``temperature_lm`` have shape
        (nlev, l_max+1, l_max+1); ``ln_ps_lm`` has shape
        (l_max+1, l_max+1).
        """
        parts = [cp.asarray(a, dtype=cp.complex128)
                 for a in (zeta_lm, delta_lm, temperature_lm)]
        lnps = cp.asarray(ln_ps_lm, dtype=cp.complex128)
        return cls(cp.concatenate(parts + [lnps[None]], axis=0))

    @classmethod
    def zeros(cls, l_max: int, nlev: int) -> "PrimitiveEquationsState":
        n = l_max + 1
        return cls(cp.zeros((3 * nlev + 1, n, n), dtype=cp.complex128))


def isothermal_rest_state(l_max: int, nlev: int, *,
                          temperature: float,
                          surface_pressure: float
                          ) -> PrimitiveEquationsState:
    """Exactly resting, horizontally uniform isothermal state.

    zeta = delta = 0 everywhere; T_k = ``temperature`` at every level;
    p_s = ``surface_pressure`` (Pa) everywhere. Constant fields are the
    pure (0,0) mode with coefficient value * sqrt(4*pi) (repository
    orthonormal-SH convention).
    """
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(
            f"temperature must be finite and > 0, got {temperature}")
    if not (math.isfinite(surface_pressure) and surface_pressure > 0):
        raise ValueError(
            f"surface_pressure must be finite and > 0, got "
            f"{surface_pressure}")
    state = PrimitiveEquationsState.zeros(l_max, nlev)
    monopole = math.sqrt(4.0 * math.pi)
    state.temperature[:, 0, 0] = temperature * monopole
    state.ln_ps[0, 0] = math.log(surface_pressure) * monopole
    return state
