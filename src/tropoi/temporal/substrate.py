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
The native step also validates the forcing and forms the absorbed constant.
Python reads one small summary per step and computes the step report from
the converged evaluation, with the same reductions as before. Forcing that
is not finite, or a positive ``flux_derivative``, raises ValueError. A step
whose Newton iteration fails in any column raises
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
    Each pair contains backend scalars in kelvin. ``extent_bounds`` holds the
    same values as two (strata,) arrays (lowest, highest), for accumulating
    them without a host transfer."""
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
    extent_bounds: tuple = None



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
        gb = sub.bottom_heat_flux if bottom_heat_flux is None else bottom_heat_flux
        src = None if source is None else xp.asarray(source, dtype=xp.float64)
        # The native step validates the forcing, forms the absorbed constant
        # (1 - A) SW + eps LW + F_0 and solves; one device-to-host transfer per
        # step then reads the largest iteration count, whether any column
        # failed and whether any forcing was invalid (decision U2).
        out = run_step(sub, xp, th_old, state.surface_anomaly, dt, shortwave=shortwave_down,
                       longwave=longwave_down, flux=flux, dflux=flux_derivative,
                       fref_temperature=flux_reference_temperature, bottom=gb, source=src,
                       max_iterations=sub.max_iterations)
        summary = _host(out["summary"], xp)
        it, failed, invalid = int(summary[0]), bool(summary[1]), bool(summary[2])
        if invalid:
            self._raise_invalid(xp, n, shortwave_down, longwave_down, flux, flux_derivative,
                                flux_reference_temperature, gb, src)
        if failed:
            bad = np.flatnonzero(_host(out["failed"], xp))
            raise SubstrateSolveError(
                f"Newton did not converge in {it} iterations for {bad.size} columns "
                f"(first {bad[:8].tolist()}); worst residual/tolerance "
                f"{float(xp.max(out['ratio'])):.3e}")
        new_state = SubstrateState(out["th"], out["ths"])
        return new_state, self._report(xp, out, it, gb, src, n)

    @staticmethod
    def _raise_invalid(xp, n, sw, lw, f0, a, fref, gb, src):
        """The message for invalid forcing, checked in the documented order
        (only after the native step has flagged it)."""
        def column(v):
            return xp.broadcast_to(xp.asarray(v, dtype=xp.float64), (n,))
        named = {"shortwave_down": column(sw), "longwave_down": column(lw),
                 "flux": column(f0), "flux_derivative": column(a),
                 "bottom_heat_flux": column(gb),
                 "source": xp.zeros((1, 1)) if src is None else src}
        if fref is not None:
            named["flux_reference_temperature"] = column(fref) - ENERGY_REFERENCE_TEMPERATURE
        for name, arr in named.items():
            if not bool(xp.all(xp.isfinite(arr))):
                raise ValueError(f"{name} must be finite")
        if not bool(xp.all(named["flux_derivative"] <= 0.0)):
            raise ValueError("flux_derivative must be finite and <= 0")
        raise RuntimeError("the native step flagged forcing that validates here")

    def _report(self, xp, out, it, gb, src, n):
        """The step report from the converged evaluation, with the reductions
        of the reference implementation (bitwise, decision U3). The largest
        |r| / tau is the native evaluation's own (a maximum is exact in any
        order), and so are the extents (per column, then reduced here)."""
        sub = self.substrate
        storage = out["storage"].sum(axis=0)
        r_sum = out["r"].sum(axis=0)
        tau_sum = out["tau"].sum(axis=0)
        S = out["extent"].shape[0] // 2
        lows, highs = out["extent"][:S].min(axis=1), out["extent"][S:].max(axis=1)
        scalar_source = src is None or src.size == 1
        if xp is np:
            src_sum = ((src.reshape(()) if src is not None else np.zeros(())) * sub.layers
                       + np.zeros(n)) if scalar_source else \
                np.broadcast_to(src, (sub.layers, n)).sum(axis=0)
            resid = r_sum - out["r_s"]
            tol = tau_sum + out["tau_s"]
            bottom = gb + 0.0 * storage
        else:
            kernels = _fused_report_kernels()
            if scalar_source:
                s0 = 0.0 if src is None else src.reshape(())
                resid, tol, bottom, src_sum = kernels[1](r_sum, out["r_s"], tau_sum,
                                                         out["tau_s"], gb, storage, s0,
                                                         float(sub.layers))
            else:
                src_sum = xp.broadcast_to(src, (sub.layers, n)).sum(axis=0)
                resid, tol, bottom = kernels[0](r_sum, out["r_s"], tau_sum, out["tau_s"], gb,
                                                storage)
        extent = tuple((lows[s], highs[s]) for s in range(S))
        return StepReport(it, storage, out["absorbed"], out["emitted"], out["q_s"], bottom,
                          src_sum, resid, tol, out["ratio"], extent, (lows, highs))


_REPORT_KERNELS = None


def _fused_report_kernels():
    """The elementwise part of the report as one kernel each (CuPy), in the
    reference's operation order; products as __dmul_rn, never contracted."""
    global _REPORT_KERNELS
    if _REPORT_KERNELS is None:
        import cupy as cp
        params = ("float64 r_sum, float64 r_s, float64 tau_sum, float64 tau_s, float64 gb, "
                  "float64 storage")
        body = ("resid = r_sum - r_s; tol = tau_sum + tau_s; "
                "bottom = gb + __dmul_rn(0.0, storage);")
        _REPORT_KERNELS = (
            cp.ElementwiseKernel(params, "float64 resid, float64 tol, float64 bottom", body,
                                 "pt_substrate_report"),
            cp.ElementwiseKernel(params + ", float64 s0, float64 layers",
                                 "float64 resid, float64 tol, float64 bottom, float64 src_sum",
                                 body + " src_sum = __dmul_rn(s0, layers) + 0.0;",
                                 "pt_substrate_report_scalar_source"))
    return _REPORT_KERNELS
