"""Pluggable stepper hooks: explicit tendency terms and implicit diagonal damping.

A time stepper advances ``dX/dt = tendency(X)``. Experiments add physics to
that equation through two generic kinds of hook, kept here so the steppers
know nothing about any particular experiment:

* an **explicit term** is any callable ``term(X) -> dX/dt`` with a
  ``signature()`` method. A stepper adds it to the tendency it evaluates
  explicitly (for the semi-implicit leapfrog: to ``N(X^n)``).
* a :class:`DiagonalDamping` is a linear damping ``dX/dt = -r X`` whose rate
  ``r`` depends only on the row of the coefficient stack and on the
  spherical-harmonic degree ``l`` (so it is diagonal in the spectral basis).
  A multi-level scheme applies it implicitly to the new level with the
  backward-Euler factor ``1 / (1 + h r)`` (``h = 2 dt`` for leapfrog); a
  one-level reference scheme uses its explicit form ``-r X``
  (:func:`complete_tendency`). The backward-Euler factor is not the
  exponential ``exp(-h r)``: the two differ by ``O((h r)^2)`` per step,
  i.e. the damping is first-order accurate in time and unconditionally
  stable (docs/held_suarez/SEMI_IMPLICIT.md §8). The semi-implicit stepper
  can instead take the same rates centred inside its solve
  (``damping_scheme="centred"``, opt-in; SEMI_IMPLICIT.md §9), in which
  case ``apply_implicit`` is not called and only ``rates`` is read.

Import-light: NumPy only. Coefficient arrays may be NumPy or CuPy; the rate
and factor tables are mirrored to the array module of the state on first
use and cached per (module, h).
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = ["DiagonalDamping", "complete_tendency", "hook_signatures",
           "rk4_real_axis_limit"]


def _array_module(a):
    """NumPy or CuPy, from the array's type (no CuPy import unless needed)."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: WPS433  (optional backend)
        return cupy
    return np


def rk4_real_axis_limit() -> float:
    """Largest ``y`` with ``|R_4(-y)| <= 1`` for the classical RK4 polynomial
    (≈ 2.785): the explicit stability limit of RK4 for ``dX/dt = -r X``."""
    lo, hi = 2.0, 3.0

    def gain(y):
        z = -y
        return abs(1.0 + z + z * z / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gain(mid) <= 1.0:
            lo = mid
        else:
            hi = mid
    return lo


class DiagonalDamping:
    """Linear damping ``dX/dt = -rates[row, l] X`` of a spectral stack.

    ``rates`` is a real, finite, non-negative ``(n_rows, n_degrees)`` array
    for a coefficient stack of shape ``(n_rows, n_degrees, n_orders)``
    (degree on axis 1, order on axis 2, as in the PE stack). A zero rate
    gives a factor of exactly 1.0, so the damped rows are untouched
    bit for bit (e.g. the global means, where ``c_0 = 0``).
    """

    def __init__(self, name: str, rates, *, params: dict | None = None):
        rates = np.array(rates, dtype=np.float64)
        if rates.ndim != 2:
            raise ValueError(f"rates must be 2-D (rows, degrees), got shape {rates.shape}")
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError(f"damping {name!r}: rates must be finite and >= 0")
        self.name = str(name)
        self.rates = rates
        self.params = dict(params or {})
        self._dev: dict[tuple[str, float | None], Any] = {}

    @property
    def max_rate(self) -> float:
        return float(self.rates.max()) if self.rates.size else 0.0

    def factor(self, h: float) -> np.ndarray:
        """Backward-Euler factor ``1 / (1 + h r)`` per (row, degree)."""
        return 1.0 / (1.0 + float(h) * self.rates)

    def _table(self, xp, h: float | None):
        key = (xp.__name__, None if h is None else float(h))
        t = self._dev.get(key)
        if t is None:
            host = self.rates if h is None else self.factor(h)
            t = xp.asarray(host)[:, :, None]
            self._dev[key] = t
        return t

    def _check(self, x):
        if x.shape[:2] != self.rates.shape:
            raise ValueError(
                f"damping {self.name!r} has rates for {self.rates.shape} "
                f"(rows, degrees); state shape is {x.shape}")

    def tendency(self, x):
        """Explicit form ``-r X`` (for one-level reference schemes)."""
        self._check(x)
        return -(self._table(_array_module(x), None) * x)

    def apply_implicit(self, x, h: float):
        """``X / (1 + h r)`` as one multiplication by the cached factor."""
        self._check(x)
        return x * self._table(_array_module(x), h)

    def signature(self) -> dict[str, Any]:
        """Identity of the damping for checkpoint compatibility checks."""
        return {"kind": "diagonal_damping", "name": self.name,
                "shape": list(self.rates.shape),
                "rates_sha256": hashlib.sha256(
                    np.ascontiguousarray(self.rates).tobytes()).hexdigest(),
                "params": self.params}


def hook_signatures(explicit_terms: Iterable, dampers: Iterable) -> list:
    """Ordered signatures of every hook (order matters: it is the order of
    the floating-point operations)."""
    out = [{"role": "explicit", **t.signature()} for t in explicit_terms]
    out += [{"role": "damping", **d.signature()} for d in dampers]
    return out


def complete_tendency(tendency: Callable, explicit_terms: Sequence = (),
                      dampers: Sequence[DiagonalDamping] = ()) -> Callable:
    """``X -> tendency(X) + sum(term(X)) + sum(-r X)``: the full right-hand
    side, for explicit schemes (the RK4 reference and the SI startup).

    With no hooks this returns ``tendency`` itself (bitwise unchanged).
    """
    terms = tuple(explicit_terms)
    damps = tuple(dampers)
    if not terms and not damps:
        return tendency

    def f(x):
        out = tendency(x)
        for term in terms:
            out = out + term(x)
        for d in damps:
            out = out + d.tendency(x)
        return out

    return f
