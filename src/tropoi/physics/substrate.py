"""Conductive substrate: vertically resolved, enthalpy-conserving conduction
beneath a surface.

This is the project's one conductive column. A column is a stack of
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

**Time stepping** (backward Euler, Newton) is not physics and lives in
:mod:`tropoi.temporal.substrate`.

**Energy.** E = sum_i C_i [h_i(T_i) - h_i(300 K)]. Because the skin holds no
energy and the same q_s enters both r_s and r_1, the step satisfies the
identity

    dE/dt - [f(T_s) + G + sum S] = sum_i r_i - r_s

(:mod:`tropoi.temporal.substrate` states the per-step identity it satisfies).

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from tropoi.physics.materials import CellMaterials, Stratum

__all__ = ["ConductiveSubstrate", "SubstrateState", "ENERGY_REFERENCE_TEMPERATURE"]

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
