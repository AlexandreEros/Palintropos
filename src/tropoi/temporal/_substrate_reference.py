"""MIGRATION REFERENCE (temporary): the Python implementation of the
substrate's backward-Euler step, as it was before the native C++
implementation (tropoi.temporal.substrate) replaced it.

It is not a production path. Tests compare the native implementation with
it: bitwise on the GPU (same arithmetic, same reductions), within the
approved bounds on the CPU. It is removed once those gates have passed
(decision U9); its source identity stays in Git.

Original description:

Backward-Euler time step of the conductive substrate.

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

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math

import numpy as np

from tropoi.numerics.tridiagonal import thomas
from tropoi.temporal.substrate import StepReport, SubstrateSolveError
from tropoi.physics.constants import SIGMA_SB
from tropoi.physics.materials import CellMaterials
from tropoi.physics.substrate import (ENERGY_REFERENCE_TEMPERATURE, TOLERANCE_FLOOR,
                                      TOLERANCE_ROUNDOFF, ConductiveSubstrate, SubstrateState)

__all__ = ["ReferenceBackwardEulerSubstrate"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def _host(a, xp):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class ReferenceBackwardEulerSubstrate:
    """Backward-Euler stepping of one :class:`ConductiveSubstrate` (see the
    module docstring). It holds no state of its own: the substrate's state is
    passed to and returned from :meth:`step`."""

    def __init__(self, substrate: ConductiveSubstrate):
        self.substrate = substrate

    def _system(self, xp, th, ths, th_old, dt, forcing, jacobian):
        """Residuals (and the Jacobian) at anomalies ``th`` (cells) and
        ``ths`` (skin). Differences are formed from anomalies."""
        sub = self.substrate
        rho, kc, chi = sub.materials.on(xp)
        dz, half = sub._consts(xp)
        const, a, t_ref, gb, src = forcing
        T = ENERGY_REFERENCE_TEMPERATURE + th
        Ts = ENERGY_REFERENCE_TEMPERATURE + ths
        T_old = ENERGY_REFERENCE_TEMPERATURE + th_old
        k = CellMaterials.conductivity(kc, chi, T)
        dz1 = float(sub.thickness[0])
        q_s = (2.0 / dz1) * CellMaterials.kirchhoff_difference(kc[0], chi[0], Ts, T[0],
                                                              delta=ths - th[0])
        res_half = half / k                                   # (L, n)
        g = 1.0 / (res_half[:-1] + res_half[1:])              # (L-1, n)
        f_int = g * (th[:-1] - th[1:])
        faces = xp.concatenate([q_s[None], f_int, xp.broadcast_to(-gb, Ts.shape)[None]],
                               axis=0)                       # (L+1, n), downward
        cap = rho * dz
        storage = cap * sub.materials.enthalpy_difference(T, T_old, delta=th - th_old) / dt
        r = storage - (faces[:-1] - faces[1:]) - src
        lin = a * (ths - t_ref)
        emitted = (sub.emissivity * SIGMA_SB) * Ts ** 4
        r_s = const + lin - emitted - q_s
        tau = TOLERANCE_FLOOR + TOLERANCE_ROUNDOFF * (
            xp.abs(storage) + xp.abs(faces[:-1]) + xp.abs(faces[1:]) + xp.abs(src))
        tau_s = TOLERANCE_FLOOR + TOLERANCE_ROUNDOFF * (
            xp.abs(const) + xp.abs(lin) + emitted + xp.abs(q_s))
        out = dict(r=r, r_s=r_s, tau=tau, tau_s=tau_s, storage=storage,
                   q_s=q_s, emitted=emitted, absorbed=const + lin, faces=faces)
        if jacobian:
            dk = CellMaterials.conductivity_derivative(kc, chi, T)
            dq_dts = (2.0 / dz1) * CellMaterials.conductivity(kc[0], chi[0], Ts)
            dq_dt1 = -(2.0 / dz1) * k[0]
            drift = th[:-1] - th[1:]
            gg = g * g
            dfa = g + drift * gg * half[:-1] * dk[:-1] / (k[:-1] * k[:-1])   # dF/dT_upper
            dfb = -g + drift * gg * half[1:] * dk[1:] / (k[1:] * k[1:])      # dF/dT_lower
            cpv = sub.materials.specific_heat(T)
            n = T.shape[1]
            L = T.shape[0]
            diag = xp.empty((L + 1, n))
            lower = xp.empty((L, n))
            upper = xp.empty((L, n))
            diag[0] = a - 4.0 * sub.emissivity * SIGMA_SB * Ts ** 3 - dq_dts
            upper[0] = -dq_dt1
            # cell rows: r_i = storage_i - F_{i-1/2} + F_{i+1/2} - S_i
            d_cells = cap * cpv / dt
            d_cells = d_cells + 0.0 * T
            d_cells[0] = d_cells[0] - dq_dt1
            d_cells[:-1] = d_cells[:-1] + dfa
            d_cells[1:] = d_cells[1:] - dfb
            diag[1:] = d_cells
            lower[0] = -dq_dts
            if L > 1:
                lower[1:] = -dfa
                upper[1:] = dfb
            out["jacobian"] = (lower, diag, upper)
        return out

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
        forcing = (const, a, t_ref, gb, src)

        T = th_old.copy()
        Ts = xp.array(state.surface_anomaly, dtype=xp.float64, copy=True)
        it = 0
        while True:
            sysm = self._system(xp, T, Ts, th_old, dt, forcing, jacobian=True)
            ratio = xp.maximum(xp.max(xp.abs(sysm["r"]) / sysm["tau"], axis=0),
                               xp.abs(sysm["r_s"]) / sysm["tau_s"])
            done = ratio <= 1.0
            if bool(xp.all(done)):
                break
            if it >= sub.max_iterations:
                bad = np.flatnonzero(~np.asarray(_host(done, xp)))
                raise SubstrateSolveError(
                    f"Newton did not converge in {it} iterations for {bad.size} columns "
                    f"(first {bad[:8].tolist()}); worst residual/tolerance "
                    f"{float(xp.max(ratio)):.3e}")
            lower, diag, upper = sysm["jacobian"]
            rhs = -xp.concatenate([sysm["r_s"][None], sysm["r"]], axis=0)
            delta = thomas(lower, diag, upper, rhs, xp)
            x = xp.concatenate([Ts[None], T], axis=0)
            # positivity: never more than halve an absolute temperature in one update
            absolute = ENERGY_REFERENCE_TEMPERATURE + x
            shrink = xp.where(delta < 0.0,
                              -0.5 * absolute / xp.where(delta < 0.0, delta, -1.0), xp.inf)
            lam = xp.minimum(1.0, xp.min(shrink, axis=0))
            lam = xp.where(done, 0.0, lam)
            for _ in range(30):
                xt = x + lam[None] * delta
                trial = self._system(xp, xt[1:], xt[0], th_old, dt, forcing, jacobian=False)
                tr = xp.maximum(xp.max(xp.abs(trial["r"]) / trial["tau"], axis=0),
                                xp.abs(trial["r_s"]) / trial["tau_s"])
                ok = (tr < ratio) | (tr <= 1.0) | done
                if bool(xp.all(ok)):
                    break
                lam = xp.where(ok, lam, 0.5 * lam)
            x = x + lam[None] * delta
            Ts, T = x[0], x[1:]
            it += 1
        new_state = SubstrateState(T, Ts)
        return new_state, self._report(xp, sysm, it, gb, src, n, new_state)

    def _report(self, xp, sysm, it, gb, src, n, state):
        sub = self.substrate
        src_sum = (xp.broadcast_to(src, (sub.layers, n)).sum(axis=0)
                   if src.size > 1 else src.reshape(()) * sub.layers + xp.zeros(n))
        storage = sysm["storage"].sum(axis=0)
        resid = sysm["r"].sum(axis=0) - sysm["r_s"]
        tol = sysm["tau"].sum(axis=0) + sysm["tau_s"]
        ratio = xp.maximum(xp.max(xp.abs(sysm["r"]) / sysm["tau"], axis=0),
                           xp.abs(sysm["r_s"]) / sysm["tau_s"])
        extent = sub._temperature_extent(state, xp)
        lows, highs = zip(*extent)
        return StepReport(it, storage, sysm["absorbed"], sysm["emitted"], sysm["q_s"],
                          gb + 0.0 * storage, src_sum, resid, tol, ratio, extent,
                          (xp.stack(lows), xp.stack(highs)))
