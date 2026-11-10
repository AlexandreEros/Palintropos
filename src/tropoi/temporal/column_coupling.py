"""Couple column physics to the hydrostatic primitive-equation core.

:class:`ColumnPhysicsCoupling` is the host adapter between the spectral PE
stack and :class:`tropoi.physics.column_model.ColumnPhysics`. It is a
*lagged term* of the semi-implicit stepper (``semi_implicit.py``): called
with the old time level ``X^{n-1}`` and the leapfrog window ``2 dt``, it

1. synthesizes ``T_k`` and ``p_s`` on the model's state grid and builds the
   columns (``p_{k+1/2} = sigma_{k+1/2} p_s``, top interface at zero
   pressure);
2. evaluates the physics (radiation, lower boundary, interior heat,
   sensible exchange) and applies the heating over the window;
3. applies the dry convective adjustment to that provisional profile;
4. returns the total temperature change divided by the window as a spectral
   tendency, truncated at the model's product cut like every other analyzed
   term; and
5. advances the surface temperature by one model step and accumulates the
   area-mean fluxes for the energy budget.

Only the temperature rows receive a tendency: radiation and convection as
formulated here exchange no momentum and no mass.

The energy statement that can be checked (and is, by the run driver): the
change of the model's global energy :func:`global_energy` between two times
equals the time integral of ``budget["column_energy_tendency"]`` plus what
the dynamics itself gains or loses numerically (hyperdiffusion, drag, time
filter, truncation). The second part is measured with the physics off.

GPU (CuPy) when the model is; the physics itself is device-agnostic.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["ColumnPhysicsCoupling", "global_energy"]

_BUDGET_KEYS = ("toa_net_down", "olr", "absorbed_stellar", "bottom_net_down",
                "sensible", "interior", "column_energy_tendency", "surface_net")


def _xp(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: WPS433
        return cupy
    return np


def _synthesize(sh, coeffs):
    batch = getattr(sh, "inv_transform_batch", None)
    if batch is not None:
        return batch(coeffs)
    xp = _xp(coeffs)
    return xp.stack([sh.inv_transform(c).real for c in coeffs])


def _analyze(sh, fields):
    batch = getattr(sh, "transform_batch", None)
    if batch is not None:
        return batch(fields)
    xp = _xp(fields)
    return xp.stack([sh.transform(f) for f in fields])


def _area_weights(model):
    xp = _xp(model.grid.point_latitudes)
    w = xp.asarray(model.grid.cell_areas, dtype=np.float64).ravel()
    return w / w.sum()


def global_energy(model, coeffs, gravity: float) -> dict:
    """Area-mean column energy of a PE state, J m^-2.

    ``enthalpy = <p_s sum_k dsigma_k c_p T_k> / g``, ``kinetic = <p_s sum_k
    dsigma_k |V_k|^2 / 2> / g``, ``surface_potential = <p_s Phi_s> / g``;
    ``total`` is their sum (the hydrostatic total energy: internal plus
    potential energy of a column is its enthalpy plus ``p_s Phi_s / g``).
    """
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
    xp = _xp(coeffs)
    K = model.nlev
    state = PrimitiveEquationsState(coeffs)
    w = _area_weights(model)
    dsig = xp.asarray(model.sigma.thickness_array())[:, None]
    temperature = _synthesize(model.sh, coeffs[2 * K:3 * K]).reshape(K, -1)
    ps = xp.exp(model.sh.inv_transform(coeffs[3 * K]).real).ravel()
    u, v = model.wind_on_state_grid(state)
    u = u.reshape(K, -1)
    v = v.reshape(K, -1)
    phi_s = model.sh.inv_transform(model.phi_surface_lm).real.ravel()
    enthalpy = float((w * ps * (dsig * model.cp_dry * temperature).sum(axis=0)).sum()) / gravity
    kinetic = float((w * ps * (dsig * 0.5 * (u * u + v * v)).sum(axis=0)).sum()) / gravity
    potential = float((w * ps * phi_s).sum()) / gravity
    return {"enthalpy": enthalpy, "kinetic": kinetic, "surface_potential": potential,
            "total": enthalpy + kinetic + potential,
            "mean_surface_pressure": float((w * ps).sum())}


class ColumnPhysicsCoupling:
    """Lagged stepper term evaluating column physics on the state grid.

    ``dt`` is the model time step (the leapfrog window is ``2 dt``; the RK4
    startup calls with a window of ``dt``, during which the surface is not
    advanced and nothing is accumulated, so every model step is counted
    once). ``surface_temperature`` (npts,) is required when the physics has a
    surface.
    """

    def __init__(self, model, physics, dt: float, *, surface_temperature=None):
        self.model = model
        self.physics = physics
        self.dt = float(dt)
        grid = model.grid
        xp = _xp(grid.point_latitudes)
        self._xp = xp
        self.latitude = xp.asarray(grid.point_latitudes, dtype=np.float64).ravel()
        self.longitude = xp.asarray(grid.point_longitudes, dtype=np.float64).ravel()
        self.weights = _area_weights(model)
        self._sigma_int = xp.asarray(model.sigma.interfaces_array())[:, None]
        self._cut = model._trunc_cut
        if physics.surface is not None:
            if surface_temperature is None:
                raise ValueError("physics with a surface needs surface_temperature")
            self.surface_temperature = xp.asarray(surface_temperature,
                                                  dtype=np.float64).ravel().copy()
            if self.surface_temperature.shape != self.latitude.shape:
                raise ValueError("surface_temperature must have one value per state-grid point")
        else:
            self.surface_temperature = None
        self.budget_time = 0.0
        self.budget = {k: 0.0 for k in _BUDGET_KEYS}
        self.last: Optional[Any] = None

    def signature(self) -> dict:
        return {"kind": "column_physics_coupling", "dt": self.dt,
                "physics": self.physics.signature()}

    # ------------------------------------------------------------------
    def columns(self, coeffs):
        """(pressure_interfaces (K+1, npts), temperature (K, npts))."""
        xp = self._xp
        K = self.model.nlev
        sh = self.model.sh
        temperature = _synthesize(sh, coeffs[2 * K:3 * K]).reshape(K, -1)
        ps = xp.exp(sh.inv_transform(coeffs[3 * K]).real).ravel()
        return self._sigma_int * ps[None, :], temperature

    def _lowest_wind_speed(self, coeffs):
        xp = self._xp
        m = self.model
        K = m.nlev
        psi = m._inv_laplacian(coeffs[K - 1])
        chi = m._inv_laplacian(coeffs[2 * K - 1])
        u, v = m._wind_from(m.sh, m._state_coslat, psi, chi)
        return xp.sqrt(u * u + v * v).ravel()

    def __call__(self, coeffs, t: float, window: float):
        xp = self._xp
        m = self.model
        K = m.nlev
        p_int, temperature = self.columns(coeffs)
        wind = self._lowest_wind_speed(coeffs) if self.physics.sensible is not None else None
        out = self.physics.compute(p_int, temperature, t, latitude=self.latitude,
                                   longitude=self.longitude,
                                   surface_temperature=self.surface_temperature,
                                   wind_speed=wind)
        provisional = temperature + window * out.heating
        adjusted = self.physics.adjust(p_int, provisional, window)
        spec = _analyze(m.sh, (adjusted - temperature) / window)
        cut = self._cut
        spec[:, cut + 1:, :] = 0.0
        spec[:, :, cut + 1:] = 0.0
        tendency = xp.zeros_like(coeffs)
        tendency[2 * K:3 * K] = spec

        elapsed = window - self.dt          # dt for a leapfrog step, 0 at startup
        if elapsed > 0.0:
            if self.surface_temperature is not None:
                self.surface_temperature = self.physics.surface.advance(
                    self.surface_temperature, out.surface_net, elapsed)
            f = out.fluxes
            w = self.weights
            means = {"toa_net_down": f.toa_net_down, "olr": f.olr,
                     "absorbed_stellar": f.absorbed_stellar,
                     "bottom_net_down": f.bottom_net_down, "sensible": out.sensible,
                     "column_energy_tendency": out.column_energy_tendency,
                     "surface_net": out.surface_net}
            for key, field in means.items():
                self.budget[key] += elapsed * float((w * field).sum())
            self.budget["interior"] += elapsed * out.interior
            self.budget_time += elapsed
        self.last = out
        return tendency

    # ------------------------------------------------------------------
    def budget_means(self, *, reset: bool = False) -> dict:
        """Time- and area-mean fluxes (W m^-2) since the last reset, plus the
        accumulated time (s)."""
        t = self.budget_time
        out = {k: (v / t if t > 0 else 0.0) for k, v in self.budget.items()}
        out["seconds"] = t
        if reset:
            self.budget_time = 0.0
            self.budget = {k: 0.0 for k in _BUDGET_KEYS}
        return out

    def state_dict(self) -> dict:
        xp = self._xp
        ts = self.surface_temperature
        if ts is not None and xp is not np:
            ts = xp.asnumpy(ts)
        return {"surface_temperature": None if ts is None else np.array(ts),
                "budget_time": self.budget_time, "budget": dict(self.budget)}

    def load_state_dict(self, d: dict) -> None:
        ts = d.get("surface_temperature")
        if (ts is None) != (self.surface_temperature is None):
            raise ValueError("saved surface state does not match this physics")
        if ts is not None:
            self.surface_temperature = self._xp.asarray(ts, dtype=np.float64)
        self.budget_time = float(d.get("budget_time", 0.0))
        self.budget = {k: float(d.get("budget", {}).get(k, 0.0)) for k in _BUDGET_KEYS}
        self.physics.reset()
