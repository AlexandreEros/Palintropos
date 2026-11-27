"""Conductive substrate: vertically resolved, enthalpy-conserving conduction
beneath a surface.

This is the one conduction solver of the project. A column is a stack of
physical strata (:class:`~tropoi.physics.materials.Stratum`) on a cell grid
whose faces include every stratum boundary. Properties may depend on
temperature and depth: k(T, z), c_p(T, z) and a prescribed rho(z). The
constant-property THAI substrate (Fauchez et al. 2020, Table 2: C = 2e6
J m^-3 K^-1, k = 2 W m^-1 K^-1, > 3 m) is one configuration; Mercury and
lunar regoliths are others.

**Conventions.** Depth z is positive downward. Face fluxes F are positive
downward. The surface is a diagnosed, massless skin at temperature T_s.

* top: F_1/2 = q_s, the conductive flux from the skin into cell 1;
* bottom: F_L+1/2 = -G, where G > 0 is the geothermal flux entering from
  below.

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

**Energy.** E = sum_i C_i [h_i(T_i) - h_i(300 K)]. Because the skin holds no
energy and the same q_s enters both r_s and r_1, the step satisfies the
identity

    dE/dt - [f(T_s) + G + sum S] = sum_i r_i - r_s

so the per-step energy closure is bounded by the Newton tolerances. Every
:class:`StepReport` carries both sides of the identity.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from tropoi.physics.constants import SIGMA_SB
from tropoi.physics.materials import CellMaterials, Stratum

__all__ = ["ConductiveSubstrate", "StepReport", "SubstrateSolveError",
           "SubstrateState", "ENERGY_REFERENCE_TEMPERATURE"]

#: Reference temperature (K) of the substrate energy.
ENERGY_REFERENCE_TEMPERATURE = 300.0

_EPS = 2.220446049250313e-16
#: Residual tolerance floor (W m^-2) and round-off multiple (frozen in
#: docs/validation/PLANET-ONE-GATES.md, G-B).
TOLERANCE_FLOOR = 1e-9
TOLERANCE_ROUNDOFF = 64.0 * _EPS


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


class SubstrateSolveError(ArithmeticError):
    """A step's Newton iteration did not reach the residual tolerance."""


class SubstrateState(NamedTuple):
    """Prognostic state of a set of substrate columns, as anomalies from
    :data:`ENERGY_REFERENCE_TEMPERATURE` (300 K). Storing anomalies keeps
    the round-off of temperature differences, and hence of the energy
    budget, relative to the anomaly instead of to 300 K.

    ``anomaly``: (L, npts) cell temperatures minus 300 K.
    ``surface_anomaly``: (npts,) diagnosed skin temperature minus 300 K.
    Use :meth:`ConductiveSubstrate.temperature` and
    :meth:`ConductiveSubstrate.surface_temperature` for absolute values."""
    anomaly: object
    surface_anomaly: object


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


def _tridiagonal(lower, diag, upper, rhs, xp):
    """Solve a batch of tridiagonal systems along axis 0 (Thomas algorithm,
    no pivoting; the Newton Jacobians here are diagonally dominant)."""
    n = diag.shape[0]
    cp = [None] * n
    dp = [None] * n
    cp[0] = upper[0] / diag[0] if n > 1 else None
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / den
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / den
    out = xp.empty_like(rhs)
    out[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


class ConductiveSubstrate:
    """A stratified conductive column (see the module docstring).

    ``strata`` are listed top to bottom. The last stratum's ``bottom`` is
    the column depth. ``interfaces`` are the cell faces (m, from 0, strictly
    increasing, containing every stratum boundary), built by the caller, for
    example with :func:`tropoi.spatial.vertical.stratified_interfaces`. The
    substrate never builds its own mesh.

    ``albedo`` and ``emissivity`` (both in [0, 1]) enter f(T_s).
    Emissivity 0 gives a non-radiating surface, as used by
    prescribed-flux verification problems. ``bottom_heat_flux`` is the
    geothermal G (W m^-2, upward into the column; a step may override it).
    ``initial_temperature`` is a scalar or a per-cell profile (K); it is an
    experiment's physical choice and is unrelated to the internal enthalpy
    reference :data:`ENERGY_REFERENCE_TEMPERATURE`. All four are required:
    they belong to the experiment, not to the substrate.
    ``max_iterations`` (numerical configuration, recorded in the signature)
    caps Newton iterations per step."""

    def __init__(self, strata: Sequence[Stratum], *, interfaces: Sequence[float],
                 albedo: float, emissivity: float, bottom_heat_flux: float,
                 initial_temperature, max_iterations: int = 50):
        strata = tuple(strata)
        e = np.asarray(tuple(float(x) for x in interfaces), dtype=np.float64)
        if (e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or e[0] != 0.0
                or not np.all(np.diff(e) > 0.0)):
            raise ValueError("interfaces must be finite, start at 0 and increase strictly")
        if not (0.0 <= albedo <= 1.0 and 0.0 <= emissivity <= 1.0):
            raise ValueError("albedo and emissivity must lie in [0, 1]")
        if not math.isfinite(float(bottom_heat_flux)):
            raise ValueError("bottom_heat_flux must be finite")
        self.strata = strata
        self.interfaces = e
        self.thickness = np.diff(e)
        self.centres = 0.5 * (e[1:] + e[:-1])
        self.materials = CellMaterials.on_grid(strata, e)
        self.albedo = float(albedo)
        self.emissivity = float(emissivity)
        self.bottom_heat_flux = float(bottom_heat_flux)
        self.max_iterations = int(max_iterations)
        init = np.asarray(initial_temperature, dtype=np.float64)
        if init.ndim not in (0, 1) or (init.ndim == 1 and init.size != self.layers):
            raise ValueError("initial_temperature must be a scalar or one value per cell")
        if not (np.all(np.isfinite(init)) and np.all(init > 0.0)):
            raise ValueError("initial_temperature must be finite and > 0 K")
        self.initial_temperature = init
        self._device: dict = {}

    @classmethod
    def constant(cls, *, interfaces: Sequence[float], volumetric_heat_capacity: float,
                 conductivity: Optional[float] = None,
                 thermal_inertia: Optional[float] = None,
                 **kw) -> "ConductiveSubstrate":
        """One constant-property stratum on the explicit grid ``interfaces``
        (build one with :mod:`tropoi.spatial.vertical`), given by its
        volumetric heat capacity C and exactly one of the conductivity k or
        the thermal inertia I = sqrt(k C). The column depth is the last
        interface. There are no material defaults: the experiment states
        them (THAI's sand, for example, is C = 2e6, I = 2000). The remaining
        keywords (albedo, emissivity, bottom_heat_flux, initial_temperature)
        are required as for the constructor."""
        cap = float(volumetric_heat_capacity)
        if not (math.isfinite(cap) and cap > 0.0):
            raise ValueError("volumetric_heat_capacity must be finite and > 0")
        if thermal_inertia is not None and conductivity is not None:
            raise ValueError("give thermal_inertia or conductivity, not both")
        if thermal_inertia is None and conductivity is None:
            raise ValueError("give the conductivity or the thermal inertia")
        if conductivity is None:
            inertia = float(thermal_inertia)
            if not (math.isfinite(inertia) and inertia > 0.0):
                raise ValueError("thermal_inertia must be finite and > 0")
            conductivity = inertia * inertia / cap
        if not (math.isfinite(float(conductivity)) and float(conductivity) > 0.0):
            raise ValueError("conductivity must be finite and > 0")
        bottom = float(interfaces[-1])
        if not math.isfinite(bottom):
            raise ValueError("interfaces must be finite")
        return cls([Stratum.volumetric(bottom, cap, float(conductivity))],
                   interfaces=interfaces, **kw)

    # -- description ----------------------------------------------------------------

    @property
    def layers(self) -> int:
        return int(self.thickness.size)

    @property
    def depth(self) -> float:
        return float(self.interfaces[-1])

    def signature(self) -> dict:
        init = self.initial_temperature
        return {"kind": "conductive",
                "scheme": "finite-volume backward-Euler enthalpy, Newton; "
                          "diagnosed massless skin (Kirchhoff half-cell)",
                "strata": [s.to_dict() for s in self.strata],
                "interfaces": [float(x) for x in self.interfaces],
                "albedo": self.albedo, "emissivity": self.emissivity,
                "bottom_heat_flux": self.bottom_heat_flux,
                "initial_temperature": (float(init) if init.ndim == 0
                                        else [float(x) for x in init]),
                "tolerance": {"floor_w_m2": TOLERANCE_FLOOR,
                              "roundoff_multiple_of_eps": TOLERANCE_ROUNDOFF / _EPS},
                "max_iterations": self.max_iterations}

    # -- state ----------------------------------------------------------------------

    def initial_state(self, npts: int, xp=np) -> SubstrateState:
        init = self.initial_temperature
        col = init.reshape(-1, 1) if init.ndim == 1 else np.full((self.layers, 1), float(init))
        temp = np.array(np.broadcast_to(col, (self.layers, int(npts))), dtype=np.float64)
        theta = temp - ENERGY_REFERENCE_TEMPERATURE
        return SubstrateState(xp.asarray(theta), xp.asarray(theta[0].copy()))

    def temperature(self, state: SubstrateState):
        """Cell temperatures (K), (L, npts)."""
        return ENERGY_REFERENCE_TEMPERATURE + state.anomaly

    def surface_temperature(self, state: SubstrateState):
        """Skin temperature (K), (npts,)."""
        return ENERGY_REFERENCE_TEMPERATURE + state.surface_anomaly

    def state_at(self, temperature, surface_temperature=None, xp=np) -> SubstrateState:
        """A state from absolute temperatures (K). The skin defaults to the
        top cell."""
        T = xp.asarray(temperature, dtype=xp.float64)
        Ts = T[0] if surface_temperature is None else xp.asarray(surface_temperature,
                                                                 dtype=xp.float64)
        return SubstrateState(T - ENERGY_REFERENCE_TEMPERATURE,
                              xp.array(Ts - ENERGY_REFERENCE_TEMPERATURE, copy=True))

    def energy(self, state: SubstrateState):
        """Stored enthalpy relative to 300 K (J m^-2), (npts,)."""
        xp = _xp(state.anomaly)
        rho, _, _ = self.materials.on(xp)
        cap = rho * xp.asarray(self.thickness)[:, None]
        dh = self.materials.enthalpy_difference(self.temperature(state),
                                                ENERGY_REFERENCE_TEMPERATURE,
                                                delta=state.anomaly)
        return (cap * dh).sum(axis=0)

    def conductivity(self, state: SubstrateState):
        """Cell conductivities k(T, z) (W m^-1 K^-1), (L, npts)."""
        xp = _xp(state.anomaly)
        _, kc, chi = self.materials.on(xp)
        return CellMaterials.conductivity(kc, chi, self.temperature(state))

    # -- one step ---------------------------------------------------------------------

    def _consts(self, xp):
        key = xp.__name__
        if key not in self._device:
            self._device[key] = (xp.asarray(self.thickness)[:, None],
                                 0.5 * xp.asarray(self.thickness)[:, None])
        return self._device[key]

    def _system(self, xp, th, ths, th_old, dt, forcing, jacobian):
        """Residuals (and the Jacobian) at anomalies ``th`` (cells) and
        ``ths`` (skin). Differences are formed from anomalies."""
        rho, kc, chi = self.materials.on(xp)
        dz, half = self._consts(xp)
        const, a, t_ref, gb, src = forcing
        T = ENERGY_REFERENCE_TEMPERATURE + th
        Ts = ENERGY_REFERENCE_TEMPERATURE + ths
        T_old = ENERGY_REFERENCE_TEMPERATURE + th_old
        k = CellMaterials.conductivity(kc, chi, T)
        dz1 = float(self.thickness[0])
        q_s = (2.0 / dz1) * CellMaterials.kirchhoff_difference(kc[0], chi[0], Ts, T[0],
                                                              delta=ths - th[0])
        res_half = half / k                                   # (L, n)
        g = 1.0 / (res_half[:-1] + res_half[1:])              # (L-1, n)
        f_int = g * (th[:-1] - th[1:])
        faces = xp.concatenate([q_s[None], f_int, xp.broadcast_to(-gb, Ts.shape)[None]],
                               axis=0)                       # (L+1, n), downward
        cap = rho * dz
        storage = cap * self.materials.enthalpy_difference(T, T_old, delta=th - th_old) / dt
        r = storage - (faces[:-1] - faces[1:]) - src
        lin = a * (ths - t_ref)
        emitted = (self.emissivity * SIGMA_SB) * Ts ** 4
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
            cpv = self.materials.specific_heat(T)
            n = T.shape[1]
            L = T.shape[0]
            diag = xp.empty((L + 1, n))
            lower = xp.empty((L, n))
            upper = xp.empty((L, n))
            diag[0] = a - 4.0 * self.emissivity * SIGMA_SB * Ts ** 3 - dq_dts
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
        gb = column(self.bottom_heat_flux if bottom_heat_flux is None else bottom_heat_flux)
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
        const = (1.0 - self.albedo) * sw + self.emissivity * lw + f0
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
            if it >= self.max_iterations:
                bad = np.flatnonzero(~np.asarray(_host(done, xp)))
                raise SubstrateSolveError(
                    f"Newton did not converge in {it} iterations for {bad.size} columns "
                    f"(first {bad[:8].tolist()}); worst residual/tolerance "
                    f"{float(xp.max(ratio)):.3e}")
            lower, diag, upper = sysm["jacobian"]
            rhs = -xp.concatenate([sysm["r_s"][None], sysm["r"]], axis=0)
            delta = _tridiagonal(lower, diag, upper, rhs, xp)
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
        src_sum = (xp.broadcast_to(src, (self.layers, n)).sum(axis=0)
                   if src.size > 1 else src.reshape(()) * self.layers + xp.zeros(n))
        storage = sysm["storage"].sum(axis=0)
        resid = sysm["r"].sum(axis=0) - sysm["r_s"]
        tol = sysm["tau"].sum(axis=0) + sysm["tau_s"]
        ratio = xp.maximum(xp.max(xp.abs(sysm["r"]) / sysm["tau"], axis=0),
                           xp.abs(sysm["r_s"]) / sysm["tau_s"])
        return StepReport(it, storage, sysm["absorbed"], sysm["emitted"], sysm["q_s"],
                          gb + 0.0 * storage, src_sum, resid, tol, ratio,
                          self._temperature_extent(state, xp))

    def _temperature_extent(self, state: SubstrateState, xp) -> tuple:
        bounds = []
        for start, end, _ in self.materials.segments:
            cells = state.anomaly[start:end]
            lo, hi = cells.min(), cells.max()
            if start == 0:
                skin = state.surface_anomaly
                lo, hi = xp.minimum(lo, skin.min()), xp.maximum(hi, skin.max())
            bounds.append((ENERGY_REFERENCE_TEMPERATURE + lo, ENERGY_REFERENCE_TEMPERATURE + hi))
        return tuple(bounds)

    def extent(self, state: SubstrateState) -> list:
        """Host (min, max) temperature pairs by stratum, including the skin
        in the top stratum."""
        return [(float(lo), float(hi))
                for lo, hi in self._temperature_extent(state, _xp(state.anomaly))]


def _host(a, xp):
    return a.get() if hasattr(a, "get") else np.asarray(a)
