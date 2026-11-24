"""Conductive substrate (regolith): a layered, implicit, energy-conserving ground.

The THAI protocol (Fauchez et al. 2020, Table 2) prescribes a conductive sand
surface: volumetric heat capacity 2e6 J m^-3 K^-1, thermal inertia
2000 J m^-2 K^-1 s^-1/2 (so k = I^2 / C = 2 W m^-1 K^-1) and a subsurface
deeper than 3 m. This module is that substrate as a standalone kernel. It
does not touch :class:`~tropoi.physics.surface.SlabSurface`, which stays the
default lower boundary.

Discretization (finite volume, backward Euler, flux form). Depth ``z`` is
positive downward from ``z = 0`` at the surface; every interface flux ``G``
is positive downward. Layer ``i`` has thickness ``dz_i`` and areal capacity
``m_i = C dz_i``; interior interfaces carry ``G = k (T'_i - T'_{i+1}) /
d_{i+1/2}`` with ``d`` the distance between layer centres. The update is

    m_i (T'_i - T_i) = dt (G_{i-1/2} - G_{i+1/2}),
    G_{1/2} = F_s          (net flux into the surface, positive downward),
    G_{L+1/2} = -F_b       (F_b: heat entering from below, positive upward),

so the column satisfies sum_i m_i (T'_i - T_i) = dt (F_s + F_b) exactly up
to round-off.

The surface flux enters linearized about the current surface temperature,
``F_s = F0 + a (T'_s - T_s)`` with ``a = dF/dT_s <= 0`` supplied by the
caller. Two surface temperatures are available (``skin``):

* ``"layer"``: the surface temperature is the top-layer mean T_1 (the
  provisional production choice).
* ``"diagnosed"``: T_skin = T_1 + F_s dz_1 / (2 k), the half-layer
  conduction drop. The linearized flux stays linear in T'_1:
  ``F = [F0 + a (T'_1 - T_s,old)] / (1 - a r)`` with ``r = dz_1 / (2k)``.

Either way, the stored energy lives only in the layers, and the flux
returned as applied is the one the solve used.

The state is stored as the anomaly ``T - 300 K``. 300 K is the energy
reference of the substrate budget. Storing the anomaly keeps round-off
relative to the anomaly rather than to 300 K. Absolute temperatures are
exposed through :meth:`ConductiveSubstrate.temperature`.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

__all__ = ["ConductiveSubstrate", "SubstrateState", "geometric_interfaces",
           "ENERGY_REFERENCE_TEMPERATURE", "SKIN_MODES"]

#: Reference temperature (K) of the stored anomaly and of the substrate energy.
ENERGY_REFERENCE_TEMPERATURE = 300.0

SKIN_MODES = ("layer", "diagnosed")

_MAX_STRETCH = 1.5


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def geometric_interfaces(top_thickness: float, depth: float, *,
                         layers: Optional[int] = None,
                         stretch: Optional[float] = None) -> tuple:
    """Layer interfaces (m, from 0 to ``depth``) with thicknesses growing
    geometrically from ``top_thickness``.

    Give exactly one of ``layers`` and ``stretch``.

    * ``layers``: the ratio is solved so that ``layers`` layers end exactly
      at ``depth``.
    * ``stretch``: geometric layers are added while they fit. A remaining
      gap thinner than half the next geometric layer is merged into the
      last layer; otherwise it becomes its own layer.

    The ratio must lie in [1, 1.5]."""
    top = float(top_thickness)
    total = float(depth)
    if not (top > 0.0 and total > 0.0 and top <= total):
        raise ValueError("need 0 < top_thickness <= depth")
    if (layers is None) == (stretch is None):
        raise ValueError("give exactly one of layers and stretch")
    if layers is not None:
        n = int(layers)
        if n < 1:
            raise ValueError("layers must be >= 1")
        if n == 1:
            if not math.isclose(top, total, rel_tol=1e-12):
                raise ValueError("one layer requires top_thickness == depth")
            return (0.0, total)
        if top * n > total * (1.0 + 1e-12):
            raise ValueError("top_thickness * layers exceeds depth (ratio < 1)")
        if math.isclose(top * n, total, rel_tol=1e-12):
            ratio = 1.0
        else:
            lo, hi = 1.0 + 1e-15, 2.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if top * (mid ** n - 1.0) / (mid - 1.0) < total:
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
        return tuple(edges)
    ratio = float(stretch)
    if not (1.0 <= ratio <= _MAX_STRETCH):
        raise ValueError(f"stretch must lie in [1, {_MAX_STRETCH}]")
    return _continue_geometric([0.0], top, ratio, total)


def _continue_geometric(edges: list, next_thickness: float, ratio: float,
                        total: float) -> tuple:
    edges = list(edges)
    t = float(next_thickness)
    while edges[-1] + t <= total * (1.0 + 1e-12):
        edges.append(edges[-1] + t)
        t *= ratio
    gap = total - edges[-1]
    if gap > 1e-12 * total:
        if gap < 0.5 * t and len(edges) > 1:
            edges[-1] = total
        else:
            edges.append(total)
    else:
        edges[-1] = total
    return tuple(edges)


class SubstrateState(NamedTuple):
    """Prognostic state of a substrate column set.

    ``anomaly``: ``(L, npts)`` layer temperatures minus 300 K.
    ``surface_flux``: ``(npts,)`` the net surface flux (W m^-2, positive
    downward) applied in the step that produced this state. The diagnosed
    skin needs it."""
    anomaly: object
    surface_flux: object


class ConductiveSubstrate:
    """A layered conductive ground with an implicit flux-form update.

    Material: ``volumetric_heat_capacity`` (J m^-3 K^-1), and exactly one of
    ``thermal_inertia`` (J m^-2 K^-1 s^-1/2) and ``conductivity``
    (W m^-1 K^-1). With neither given, the THAI inertia 2000 is used.

    Geometry, in order of precedence:
    * explicit ``interfaces`` (m, from 0, increasing);
    * ``top_thickness`` with ``stretch``;
    * ``top_thickness`` with ``layers`` (default 30 layers, top 2 cm, to
      ``depth`` 3.5 m).

    ``bottom_heat_flux`` is the default F_b (W m^-2, positive upward into
    the column). ``albedo`` and ``emissivity`` are carried for the coupling.
    ``initial_temperature`` is a scalar or a per-layer profile (K). ``skin``
    is one of :data:`SKIN_MODES`."""

    def __init__(self, *, volumetric_heat_capacity: float = 2.0e6,
                 thermal_inertia: Optional[float] = None,
                 conductivity: Optional[float] = None,
                 depth: float = 3.5, top_thickness: float = 0.02,
                 layers: Optional[int] = 30, stretch: Optional[float] = None,
                 interfaces: Optional[Sequence[float]] = None,
                 bottom_heat_flux: float = 0.0, albedo: float = 0.3,
                 emissivity: float = 1.0, initial_temperature=300.0,
                 skin: str = "layer"):
        cap = float(volumetric_heat_capacity)
        if not (math.isfinite(cap) and cap > 0.0):
            raise ValueError("volumetric_heat_capacity must be finite and > 0")
        if thermal_inertia is not None and conductivity is not None:
            raise ValueError("give thermal_inertia or conductivity, not both")
        if conductivity is None:
            inertia = 2000.0 if thermal_inertia is None else float(thermal_inertia)
            if not (math.isfinite(inertia) and inertia > 0.0):
                raise ValueError("thermal_inertia must be finite and > 0")
            cond = inertia * inertia / cap
        else:
            cond = float(conductivity)
            if not (math.isfinite(cond) and cond > 0.0):
                raise ValueError("conductivity must be finite and > 0")
        if skin not in SKIN_MODES:
            raise ValueError(f"skin must be one of {SKIN_MODES}")
        if not (0.0 <= albedo <= 1.0 and 0.0 < emissivity <= 1.0):
            raise ValueError("albedo must be in [0, 1] and emissivity in (0, 1]")
        if interfaces is not None:
            edges = tuple(float(e) for e in interfaces)
        elif stretch is not None:
            edges = geometric_interfaces(top_thickness, depth, stretch=stretch)
        else:
            edges = geometric_interfaces(top_thickness, depth, layers=layers)
        e = np.asarray(edges, dtype=np.float64)
        if e.ndim != 1 or e.size < 2 or e[0] != 0.0 or not np.all(np.diff(e) > 0.0):
            raise ValueError("interfaces must start at 0 and increase strictly")
        self.volumetric_heat_capacity = cap
        self.conductivity = cond
        self.thermal_inertia = math.sqrt(cond * cap)
        self.interfaces = e
        self.thickness = np.diff(e)
        self.centres = 0.5 * (e[1:] + e[:-1])
        self.layer_capacity = cap * self.thickness          # m_i, J m^-2 K^-1
        self.interface_conductance = cond / np.diff(self.centres)   # (L-1,)
        self.skin_resistance = self.thickness[0] / (2.0 * cond)    # r, m^2 K W^-1
        self.bottom_heat_flux = float(bottom_heat_flux)
        self.albedo = float(albedo)
        self.emissivity = float(emissivity)
        self.skin = skin
        init = np.asarray(initial_temperature, dtype=np.float64)
        if init.ndim not in (0, 1) or (init.ndim == 1 and init.size != self.layers):
            raise ValueError("initial_temperature must be a scalar or one value per layer")
        self.initial_temperature = init
        # Per-layer sum of the conductances to the neighbours (0 for L = 1).
        gsum = np.zeros(self.layers)
        gsum[:-1] += self.interface_conductance
        gsum[1:] += self.interface_conductance
        self._conductance_sum = gsum
        self._device_cache: dict = {}

    @property
    def layers(self) -> int:
        return int(self.thickness.size)

    @property
    def depth(self) -> float:
        return float(self.interfaces[-1])

    def signature(self) -> dict:
        init = self.initial_temperature
        return {"kind": "conductive",
                "volumetric_heat_capacity": self.volumetric_heat_capacity,
                "conductivity": self.conductivity,
                "interfaces": [float(x) for x in self.interfaces],
                "bottom_heat_flux": self.bottom_heat_flux,
                "albedo": self.albedo, "emissivity": self.emissivity,
                "initial_temperature": (float(init) if init.ndim == 0
                                        else [float(x) for x in init]),
                "skin": self.skin}

    def _constants(self, xp):
        key = xp.__name__
        if key not in self._device_cache:
            self._device_cache[key] = (
                xp.asarray(self.layer_capacity)[:, None],
                xp.asarray(self._conductance_sum)[:, None],
                xp.asarray(self.interface_conductance),
            )
        return self._device_cache[key]

    def initial_state(self, npts: int, xp=np) -> SubstrateState:
        anomaly = np.broadcast_to(
            (self.initial_temperature - ENERGY_REFERENCE_TEMPERATURE).reshape(-1, 1)
            if self.initial_temperature.ndim == 1
            else self.initial_temperature - ENERGY_REFERENCE_TEMPERATURE,
            (self.layers, int(npts)))
        return SubstrateState(xp.asarray(np.array(anomaly, dtype=np.float64)),
                              xp.zeros(int(npts), dtype=xp.float64))

    def temperature(self, state: SubstrateState):
        """Absolute layer temperatures (K), ``(L, npts)``."""
        return ENERGY_REFERENCE_TEMPERATURE + state.anomaly

    def surface_temperature(self, state: SubstrateState):
        """The surface temperature the fluxes see (K), per ``skin``."""
        top = state.anomaly[0]
        if self.skin == "diagnosed":
            top = top + self.skin_resistance * state.surface_flux
        return ENERGY_REFERENCE_TEMPERATURE + top

    def energy(self, state: SubstrateState):
        """Stored energy relative to 300 K (J m^-2), ``(npts,)``."""
        m = self._constants(_xp(state.anomaly))[0]
        return (m * state.anomaly).sum(axis=0)

    def step(self, state: SubstrateState, flux, dt: float, *,
             flux_derivative=0.0, bottom_heat_flux=None) -> SubstrateState:
        """Advance one backward-Euler step of ``dt`` seconds.

        ``flux`` is F0, the net surface flux into the ground (W m^-2,
        positive downward), evaluated at :meth:`surface_temperature` of
        ``state``. ``flux_derivative`` is a = dF/dT_s there (W m^-2 K^-1,
        <= 0), scalar or per column. ``bottom_heat_flux`` overrides the
        default F_b. Returns the new state; its ``surface_flux`` is the flux
        actually applied, ``F0 + a (T'_s - T_s)``."""
        theta = state.anomaly
        xp = _xp(theta)
        m, gsum, g = self._constants(xp)
        dt = float(dt)
        a = flux_derivative
        fb = self.bottom_heat_flux if bottom_heat_flux is None else bottom_heat_flux
        if self.skin == "layer":
            a_eff = a
            const = flux - a * theta[0]
        else:
            r = self.skin_resistance
            den = 1.0 - a * r
            a_eff = a / den
            const = (flux - a * (theta[0] + r * state.surface_flux)) / den
        # Applied flux: F = const + a_eff * theta'_0 (linear in the new top anomaly).
        rhs = m * theta
        rhs[0] = rhs[0] + dt * const
        rhs[-1] = rhs[-1] + dt * fb
        diag = m + dt * gsum
        diag = xp.broadcast_to(diag, theta.shape).copy()
        diag[0] = diag[0] - dt * a_eff
        new = self._thomas(diag, -dt * g, rhs, xp)
        applied = const + a_eff * new[0]
        if not hasattr(applied, "shape") or applied.shape != new[0].shape:
            applied = xp.broadcast_to(xp.asarray(applied, dtype=xp.float64),
                                      new[0].shape).copy()
        return SubstrateState(new, applied)

    @staticmethod
    def _thomas(diag, off, rhs, xp):
        """Solve the symmetric tridiagonal system with diagonal ``diag``
        (L, npts), off-diagonal ``off`` (L-1,) and right side ``rhs``."""
        n = diag.shape[0]
        out = xp.empty_like(rhs)
        if n == 1:
            out[0] = rhs[0] / diag[0]
            return out
        cprime = [None] * (n - 1)
        dprime = [None] * n
        cprime[0] = off[0] / diag[0]
        dprime[0] = rhs[0] / diag[0]
        for i in range(1, n):
            den = diag[i] - off[i - 1] * cprime[i - 1]
            if i < n - 1:
                cprime[i] = off[i] / den
            dprime[i] = (rhs[i] - off[i - 1] * dprime[i - 1]) / den
        out[n - 1] = dprime[n - 1]
        for i in range(n - 2, -1, -1):
            out[i] = dprime[i] - cprime[i] * out[i + 1]
        return out
