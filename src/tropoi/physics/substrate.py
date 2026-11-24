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
           "SubstrateState", "geometric_interfaces", "ENERGY_REFERENCE_TEMPERATURE"]

#: Reference temperature (K) of the substrate energy.
ENERGY_REFERENCE_TEMPERATURE = 300.0

_MAX_STRETCH = 1.5
_RATIO_TOL = 1e-9          # round-off allowance on the [1, 1.5] ratio guarantee
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


def geometric_interfaces(top_thickness: float, depth: float, *,
                         layers: Optional[int] = None,
                         stretch: Optional[float] = None,
                         depth_is_minimum: bool = False) -> tuple:
    """Layer interfaces (m, from 0) with thicknesses growing geometrically
    from ``top_thickness``.

    Give exactly one of ``layers`` and ``stretch``. Both keep
    ``top_thickness`` exact.

    * ``layers``: the ratio is solved so that ``layers`` layers end exactly
      at ``depth``. The solved ratio must lie in [1, 1.5].
    * ``stretch``: layers of thickness ``top * stretch**i`` are added while
      they fit within ``depth``. A gap g left at the bottom is handled
      deterministically:

      - if g is zero to within 1e-12 relative, the grid ends exactly at
        ``depth``;
      - otherwise two terminations are considered: merging g into the last
        layer, or keeping g as an extra layer. Each is acceptable only if its
        final thickness ratio lies in [1, 1.5]. Of the acceptable ones, the
        one whose final ratio is closest to ``stretch`` is used (merging on a
        tie). The requested ratio then holds for every layer except the last;
      - if neither is acceptable, ``ValueError`` is raised. In that case the
        requested top thickness, depth, stretch and ratio bound are
        incompatible. Use ``layers`` for an exact geometric fit, choose
        another depth, or pass ``depth_is_minimum=True``;
      - with ``depth_is_minimum=True``, ``depth`` is a lower bound: the grid
        ends at the first exact geometric interface at or below ``depth``.
        Every ratio is then exactly ``stretch``, and the column is at least
        as deep as requested.

    **Ratio bounds.** The regolith specification requires only an upper
    bound, stretch <= 1.5. This function guarantees more: every
    adjacent-layer ratio of a grid it returns lies in [1, 1.5] (to a
    round-off allowance of 1e-9, checked on return), so thickness never
    decreases with depth. ``layers`` must be an integer. Grids passed explicitly as ``interfaces`` to
    :class:`ConductiveSubstrate`, for example uniform or historical fixture
    grids, are not subject to that guarantee."""
    top = float(top_thickness)
    total = float(depth)
    if not (math.isfinite(top) and math.isfinite(total)):
        raise ValueError("top_thickness and depth must be finite")
    if not (top > 0.0 and total > 0.0 and top <= total):
        raise ValueError("need 0 < top_thickness <= depth")
    if (layers is None) == (stretch is None):
        raise ValueError("give exactly one of layers and stretch")
    if layers is not None and depth_is_minimum:
        raise ValueError("depth_is_minimum applies to stretch, not layers")
    if layers is not None:
        if isinstance(layers, bool) or int(layers) != layers:
            raise ValueError(f"layers must be an integer, got {layers!r}")
        n = int(layers)
        if n < 1:
            raise ValueError("layers must be >= 1")
        if n == 1:
            if not math.isclose(top, total, rel_tol=1e-12):
                raise ValueError("one layer requires top_thickness == depth")
            return (0.0, total)
        if top * n > total * (1.0 + 1e-12):
            raise ValueError("top_thickness * layers exceeds depth (ratio < 1)")
        if _geometric_sum(top, _MAX_STRETCH, n) < total * (1.0 - 1e-12):
            raise ValueError(f"{n} layers from top_thickness={top} cannot reach depth={total} "
                             f"with a ratio <= {_MAX_STRETCH}")
        if math.isclose(top * n, total, rel_tol=1e-12):
            ratio = 1.0
        else:
            lo, hi = 1.0 + 1e-15, 2.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _geometric_sum(top, mid, n) < total:
                    lo = mid
                else:
                    hi = mid
            ratio = 0.5 * (lo + hi)
        if ratio > _MAX_STRETCH:
            raise ValueError(f"solved stretch {ratio:.6f} exceeds {_MAX_STRETCH}")
        edges = [0.0]
        t = top
        for _ in range(n - 1):
            edges.append(edges[-1] + t)
            t *= ratio
        edges.append(total)
        return _checked(edges)
    ratio = float(stretch)
    if not (math.isfinite(ratio) and 1.0 <= ratio <= _MAX_STRETCH):
        raise ValueError(f"stretch must lie in [1, {_MAX_STRETCH}]")
    edges = [0.0]
    t = top
    while edges[-1] + t <= total * (1.0 + 1e-12):
        edges.append(edges[-1] + t)
        t *= ratio
    gap = total - edges[-1]
    if gap <= 1e-12 * total:
        edges[-1] = total
        return _checked(edges)
    if depth_is_minimum:
        edges.append(edges[-1] + t)
        return _checked(edges)
    dz = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    candidates = []
    if len(dz) >= 2:                      # merge into the last layer (never the top one)
        merged = (dz[-1] + gap) / dz[-2]
        if 1.0 <= merged <= _MAX_STRETCH:
            candidates.append((abs(merged - ratio), 0, "merge"))
    else:
        merged = math.nan
    appended = gap / dz[-1]
    if 1.0 <= appended <= _MAX_STRETCH:
        candidates.append((abs(appended - ratio), 1, "append"))
    if not candidates:
        raise ValueError(
            f"top_thickness={top}, depth={total}, stretch={ratio} are incompatible with "
            f"final-layer ratios in [1, {_MAX_STRETCH}]: merging the {gap:.6g} m remainder "
            f"gives {merged:.4f}, keeping it as a layer gives {appended:.4f}. Use layers= "
            "for an exact geometric fit, another depth, or depth_is_minimum=True")
    if min(candidates)[2] == "merge":
        edges[-1] = total
    else:
        edges.append(total)
    return _checked(edges)


def _geometric_sum(top: float, ratio: float, n: int) -> float:
    """top * (ratio**n - 1) / (ratio - 1). Near ratio 1 it uses expm1/log1p to
    avoid cancellation; the plain form is kept elsewhere, so the default grid's
    arithmetic is unchanged."""
    x = ratio - 1.0
    if x < 1e-3:
        return top * math.expm1(n * math.log1p(x)) / x
    return top * (ratio ** n - 1.0) / (ratio - 1.0)


def _checked(edges) -> tuple:
    """Assert the documented ratio guarantee (to 1e-9 relative) on a generated grid."""
    dz = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    for i in range(1, len(dz)):
        q = dz[i] / dz[i - 1]
        if not (1.0 - _RATIO_TOL <= q <= _MAX_STRETCH + _RATIO_TOL):
            raise ValueError(f"generated grid ratio {q!r} at layer {i} is outside "
                             f"[1, {_MAX_STRETCH}]; this is a bug or a degenerate request")
    return tuple(edges)


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
    stratum over all columns, with the skin counted in the top stratum."""
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
    the column depth. The grid is either explicit ``interfaces`` (m, from 0,
    strictly increasing, containing every stratum boundary), or
    ``top_thickness`` with ``max_stretch``, which is passed to
    :func:`~tropoi.physics.materials.stratified_interfaces`.

    ``albedo`` and ``emissivity`` (both in [0, 1]) enter f(T_s).
    Emissivity 0 gives a non-radiating surface, as used by
    prescribed-flux verification problems. ``bottom_heat_flux`` is the
    default geothermal G (W m^-2, upward into the column).
    ``initial_temperature`` is a scalar or a per-cell profile (K).
    ``max_iterations`` caps Newton iterations per step."""

    def __init__(self, strata: Sequence[Stratum], *,
                 interfaces: Optional[Sequence[float]] = None,
                 top_thickness: Optional[float] = None,
                 max_stretch: Optional[float] = None,
                 albedo: float = 0.0, emissivity: float = 1.0,
                 bottom_heat_flux: float = 0.0, initial_temperature=300.0,
                 max_iterations: int = 50):
        from tropoi.physics.materials import stratified_interfaces
        strata = tuple(strata)
        if interfaces is not None:
            if top_thickness is not None or max_stretch is not None:
                raise ValueError("give interfaces or top_thickness/max_stretch, not both")
            edges = tuple(float(x) for x in interfaces)
        else:
            if top_thickness is None or max_stretch is None:
                raise ValueError("give interfaces, or top_thickness and max_stretch")
            edges = stratified_interfaces([s.bottom for s in strata],
                                          top_thickness, max_stretch)
        e = np.asarray(edges, dtype=np.float64)
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
    def constant(cls, *, volumetric_heat_capacity: float = 2.0e6,
                 conductivity: Optional[float] = None,
                 thermal_inertia: Optional[float] = None,
                 interfaces: Optional[Sequence[float]] = None,
                 depth: Optional[float] = None, top_thickness: Optional[float] = None,
                 layers: Optional[int] = None, stretch: Optional[float] = None,
                 **kw) -> "ConductiveSubstrate":
        """One constant-property stratum, given by C and either k or the
        thermal inertia I = sqrt(k C). The defaults are the THAI
        substrate's: C = 2e6, I = 2000 (k = 2), with 30 layers from a 2 cm
        top to 3.5 m.

        The grid takes exactly one form:

        * explicit ``interfaces``;
        * ``top_thickness`` to ``depth`` with ``layers`` (default 30);
        * ``top_thickness`` to ``depth`` with ``stretch``
          (:func:`geometric_interfaces`).

        Conflicting arguments raise ``ValueError``."""
        cap = float(volumetric_heat_capacity)
        if not (math.isfinite(cap) and cap > 0.0):
            raise ValueError("volumetric_heat_capacity must be finite and > 0")
        if thermal_inertia is not None and conductivity is not None:
            raise ValueError("give thermal_inertia or conductivity, not both")
        if conductivity is None:
            inertia = 2000.0 if thermal_inertia is None else float(thermal_inertia)
            if not (math.isfinite(inertia) and inertia > 0.0):
                raise ValueError("thermal_inertia must be finite and > 0")
            conductivity = inertia * inertia / cap
        if not (math.isfinite(float(conductivity)) and float(conductivity) > 0.0):
            raise ValueError("conductivity must be finite and > 0")
        if interfaces is not None:
            given = [n for n, v in (("depth", depth), ("top_thickness", top_thickness),
                                    ("layers", layers), ("stretch", stretch)) if v is not None]
            if given:
                raise ValueError(f"interfaces given together with {', '.join(given)}; "
                                 "explicit interfaces define the whole geometry")
        else:
            if stretch is not None and layers is not None:
                raise ValueError("give layers or stretch, not both")
            top = 0.02 if top_thickness is None else top_thickness
            total = 3.5 if depth is None else depth
            interfaces = (geometric_interfaces(top, total, stretch=stretch)
                          if stretch is not None else
                          geometric_interfaces(top, total, layers=30 if layers is None else layers))
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
        report = self._report(xp, sysm, it, gb, src, n)
        return SubstrateState(T, Ts), report

    def _report(self, xp, sysm, it, gb, src, n):
        src_sum = (xp.broadcast_to(src, (self.layers, n)).sum(axis=0)
                   if src.size > 1 else src.reshape(()) * self.layers + xp.zeros(n))
        storage = sysm["storage"].sum(axis=0)
        resid = sysm["r"].sum(axis=0) - sysm["r_s"]
        tol = sysm["tau"].sum(axis=0) + sysm["tau_s"]
        ratio = xp.maximum(xp.max(xp.abs(sysm["r"]) / sysm["tau"], axis=0),
                           xp.abs(sysm["r_s"]) / sysm["tau_s"])
        return StepReport(it, storage, sysm["absorbed"], sysm["emitted"], sysm["q_s"],
                          gb + 0.0 * storage, src_sum, resid, tol, ratio, ())

    def extent(self, state: SubstrateState) -> list:
        """(min, max) temperature in each stratum over all columns, with the
        skin counted in the top stratum. Use it to check each law's
        validity range."""
        xp = _xp(state.anomaly)
        T = _host(self.temperature(state), xp)
        which = self.materials.stratum_of_cell
        out = []
        for j in range(len(self.strata)):
            vals = T[which == j]
            lo, hi = float(vals.min()), float(vals.max())
            if j == 0:
                ts = _host(self.surface_temperature(state), xp)
                lo, hi = min(lo, float(ts.min())), max(hi, float(ts.max()))
            out.append((lo, hi))
        return out


def _host(a, xp):
    return a.get() if hasattr(a, "get") else np.asarray(a)
