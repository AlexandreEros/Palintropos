"""Backward-Euler time step of the conductive substrate.

The physical column (strata, mesh, materials, state, energy) is
:class:`tropoi.physics.substrate.ConductiveSubstrate`. This module owns its
time discretization and the nonlinear solve of each step.

**Discretization.** Finite volume, backward Euler, in enthalpy form. For
each cell, with C_i = rho_i dz_i, the residual in W m^-2 is

    r_i = C_i [h_i(T_i) - h_i(T_i^n)] / dt - (F_i-1/2 - F_i+1/2) - S_i

with all fluxes at the new time. Interior faces use series half-cell
resistances, F = (T_i - T_i+1) / (dz_i/(2 k_i(T_i)) + dz_i+1/(2 k_i+1(T_i+1))),
so flux is exactly continuous across material interfaces. The skin flux is
Kirchhoff-exact within the top half-cell:
q_s = 2 [Phi_1(T_s) - Phi_1(T_1)] / dz_1, with Phi_1 = integral of k_1 dT.
The skin satisfies

    r_s = f(T_s) - q_s = 0,
    f(T_s) = (1 - A) SW_down + eps LW_down - eps sigma T_s^4 + F_0 + a (T_s - T_ref)

where SW and LW are incident radiation, and F_0 + a (T_s - T_ref) is an
optional caller-linearized extra flux (for example an atmosphere's
turbulent fluxes, or a prescribed flux in a verification problem).

The L + 1 unknowns (T_s, T_1 .. T_L) form a tridiagonal nonlinear system
per column. It is solved by safeguarded Newton iteration with the exact
Jacobian, a backtracking line search and a positivity bound. A step that
does not reach the residual tolerance raises :class:`SubstrateSolveError`.

**Energy identity.** With E = sum_i C_i [h_i(T_i) - h_i(300 K)], and because
the skin holds no energy and the same q_s enters both r_s and r_1, each step
satisfies

    dE/dt - [f(T_s) + G + sum S] = sum_i r_i - r_s

so the per-step energy closure is bounded by the Newton tolerances. Every
:class:`StepReport` carries both sides of the identity.

**Implementation.** The arithmetic is the canonical native (C++)
implementation (``tropoi/{numerics,physics,temporal}/native/pt``), run by
:mod:`tropoi.temporal.substrate_native`: compiled for the GPU at run time
(CuPy arrays) or by the CPU library built at installation (NumPy arrays).
Python validates the inputs, prepares the per-column forcing and computes
the step report from the converged evaluation with the same reductions as
before. A step whose Newton iteration fails in any column raises
:class:`SubstrateSolveError`; no unconverged state is returned.

Import-light: NumPy at import; CuPy only for CuPy arrays.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from tropoi.physics.substrate import (ENERGY_REFERENCE_TEMPERATURE, TOLERANCE_FLOOR,
                                      TOLERANCE_ROUNDOFF, ConductiveSubstrate, SubstrateState)

__all__ = ["BackwardEulerSubstrate", "StepReport", "SubstrateSolveError"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def _host(a, xp):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class SubstrateSolveError(ArithmeticError):
    """A step's Newton iteration did not reach the residual tolerance."""



class StepReport(NamedTuple):
    """Diagnostics of one step, per column (npts,), in W m^-2 unless noted.

    ``energy_tendency`` is dE/dt. ``absorbed`` is everything in f(T_s) except
    emission: (1 - A) SW + eps LW + F_0 + a (T_s - T_ref). ``emitted`` is
    eps sigma T_s^4. ``conducted`` is q_s. ``bottom`` is G, and ``source``
    is sum S.

    The identity energy_tendency - (absorbed - emitted + bottom + source)
    = ``residual_sum`` (sum r_i - r_s) holds to round-off, and
    \\|residual_sum\\| <= ``tolerance_sum``.

    ``max_residual_ratio`` is the largest \\|r\\|/tau over the column (<= 1 once
    converged). ``extent`` holds the (min, max) temperature reached in each
    stratum over all columns, with the skin counted in the top stratum.
    Each pair contains backend scalars in kelvin."""
    iterations: int
    energy_tendency: object
    absorbed: object
    emitted: object
    conducted: object
    bottom: object
    source: object
    residual_sum: object
    tolerance_sum: object
    max_residual_ratio: object
    extent: tuple



class BackwardEulerSubstrate:
    """Backward-Euler stepping of one :class:`ConductiveSubstrate` (see the
    module docstring). It holds no state of its own: the substrate's state is
    passed to and returned from :meth:`step`."""

    def __init__(self, substrate: ConductiveSubstrate):
        self.substrate = substrate

    def describe(self, backend: str) -> dict:
        """Provenance of the implementation used on ``backend``
        (``"cupy"`` or ``"numpy"``)."""
        from tropoi.temporal.substrate_native import describe
        return describe(self.substrate, backend)

    def step(self, state: SubstrateState, dt: float, *, shortwave_down=0.0,
             longwave_down=0.0, flux=0.0, flux_derivative=0.0,
             flux_reference_temperature=None, bottom_heat_flux=None,
             source=None):
        """Advance one backward-Euler step of ``dt`` seconds.

        Surface forcing (all W m^-2 and positive downward, scalar or one
        value per column):

        * ``shortwave_down``: incident shortwave, of which (1 - A) is
          absorbed;
        * ``longwave_down``: incident longwave, of which eps is absorbed;
        * ``flux`` + ``flux_derivative`` (T_s - T_ref): an extra flux
          linearized by the caller about ``flux_reference_temperature``
          (default: the state's skin temperature). The derivative must
          be <= 0.

        ``bottom_heat_flux`` overrides G. ``source`` is a heat source per
        cell, integrated over the cell (W m^-2): (L, npts), (L, 1) or
        scalar.

        Returns ``(new_state, report)``. Raises :class:`SubstrateSolveError`
        if the Newton iteration fails to reach the residual tolerance."""
        from tropoi.temporal.substrate_native import run_step
        sub = self.substrate
        th_old = state.anomaly
        xp = _xp(th_old)
        dt = float(dt)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")
        n = th_old.shape[1]
        shape = (n,)

        def column(v):
            return xp.broadcast_to(xp.asarray(v, dtype=xp.float64), shape)

        sw = column(shortwave_down)
        lw = column(longwave_down)
        f0 = column(flux)
        a = column(flux_derivative)
        # the linearization reference, as an anomaly
        given_ref = flux_reference_temperature is not None
        t_ref = (column(flux_reference_temperature) - ENERGY_REFERENCE_TEMPERATURE
                 if given_ref else state.surface_anomaly)
        gb = column(sub.bottom_heat_flux if bottom_heat_flux is None else bottom_heat_flux)
        src = (xp.zeros((1, 1)) if source is None
               else xp.asarray(source, dtype=xp.float64))
        # One device synchronization validates every input; the detailed
        # message is worked out only on failure.
        named = {"shortwave_down": sw, "longwave_down": lw, "flux": f0,
                 "flux_derivative": a, "bottom_heat_flux": gb, "source": src}
        if given_ref:
            named["flux_reference_temperature"] = t_ref
        ok = xp.all(a <= 0.0)
        for arr in named.values():
            ok = ok & xp.all(xp.isfinite(arr))
        if not bool(ok):
            for name, arr in named.items():
                if not bool(xp.all(xp.isfinite(arr))):
                    raise ValueError(f"{name} must be finite")
            raise ValueError("flux_derivative must be finite and <= 0")
        const = (1.0 - sub.albedo) * sw + sub.emissivity * lw + f0
        cells = None if source is None else xp.broadcast_to(src, (sub.layers, n))
        out = run_step(sub, xp, th_old, state.surface_anomaly, dt, const, a, t_ref, gb, cells,
                       sub.max_iterations)
        # One device-to-host transfer per step: the iteration count and whether
        # any column failed (decision U2).
        summary = _host(xp.stack([xp.max(out["iterations"]), xp.max(out["failed"])]), xp)
        it, failed = int(summary[0]), bool(summary[1])
        if failed:
            bad = np.flatnonzero(_host(out["failed"], xp))
            raise SubstrateSolveError(
                f"Newton did not converge in {it} iterations for {bad.size} columns "
                f"(first {bad[:8].tolist()}); worst residual/tolerance "
                f"{float(xp.max(out['ratio'])):.3e}")
        new_state = SubstrateState(out["th"], out["ths"])
        return new_state, self._report(xp, out, it, gb, src, n, new_state)

    def _report(self, xp, out, it, gb, src, n, state):
        """The step report from the converged evaluation, with the reductions
        of the reference implementation (bitwise, decision U3)."""
        sub = self.substrate
        src_sum = (xp.broadcast_to(src, (sub.layers, n)).sum(axis=0)
                   if src.size > 1 else src.reshape(()) * sub.layers + xp.zeros(n))
        storage = out["storage"].sum(axis=0)
        resid = out["r"].sum(axis=0) - out["r_s"]
        tol = out["tau"].sum(axis=0) + out["tau_s"]
        ratio = xp.maximum(xp.max(xp.abs(out["r"]) / out["tau"], axis=0),
                           xp.abs(out["r_s"]) / out["tau_s"])
        return StepReport(it, storage, out["absorbed"], out["emitted"], out["q_s"],
                          gb + 0.0 * storage, src_sum, resid, tol, ratio,
                          sub._temperature_extent(state, xp))
