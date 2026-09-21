"""Shallow-water state description: one packed (3, L+1, L+1) coefficient stack.

Split verbatim from the shallow-water core (now
:mod:`tropoi.temporal.tendencies.shallow_water`, which re-exports every
name here) so initial conditions and other state consumers need not
import the tendency implementation. Semantics (thickness geopotential,
pinned monopoles) are documented on that module.
"""
from __future__ import annotations

from dataclasses import dataclass

import cupy as cp

#: Stack indices of the prognostic variables in ShallowWaterState.coeffs.
ZETA, DELTA, PHI = 0, 1, 2
PROGNOSTICS = ("zeta", "delta", "phi")


class ShallowWaterStateError(ValueError):
    """A shallow-water state violated a hard physical/numerical constraint."""


@dataclass
class ShallowWaterState:
    """Spectral shallow-water state: one (3, l_max+1, l_max+1) complex stack.

    Index 0 = zeta, 1 = delta, 2 = phi (PERTURBATION geopotential). Keeping
    the three prognostics in a single array makes the RK4 stage arithmetic a
    plain array expression (see ``run.engine.rk4_step_array``).
    """

    coeffs: cp.ndarray

    @classmethod
    def from_fields(cls, zeta_lm: cp.ndarray, delta_lm: cp.ndarray,
                    phi_lm: cp.ndarray) -> "ShallowWaterState":
        return cls(cp.stack([
            cp.asarray(zeta_lm, dtype=cp.complex128),
            cp.asarray(delta_lm, dtype=cp.complex128),
            cp.asarray(phi_lm, dtype=cp.complex128),
        ]))

    @classmethod
    def zeros(cls, l_max: int) -> "ShallowWaterState":
        n = l_max + 1
        return cls(cp.zeros((3, n, n), dtype=cp.complex128))

    @property
    def zeta(self) -> cp.ndarray:
        return self.coeffs[ZETA]

    @property
    def delta(self) -> cp.ndarray:
        return self.coeffs[DELTA]

    @property
    def phi(self) -> cp.ndarray:
        return self.coeffs[PHI]
