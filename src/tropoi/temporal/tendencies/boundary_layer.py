"""Boundary-layer stress and diffusion as a lagged stepper term.

:class:`BoundaryLayerDiffusion` applies the dry boundary layer of
:mod:`tropoi.physics.boundary_layer` (MARS_MI_PLAN.md §2e) to the
primitive-equation state: one implicit column solve per step for u, v
and the dry static energy, with the bulk surface stress as the implicit
bottom condition of the momentum solve and a zero-flux bottom for the
dry static energy (the surface sensible flux stays in the column
physics). Like every other nonlinear momentum term it is evaluated on
the backend product grid: winds and temperature are synthesized there,
the momentum change returns through the weak-form vector analysis, the
temperature change through a scalar analysis, and both are truncated at
the model's product cut.

A *lagged* term of the semi-implicit stepper, evaluated at the old time
level with the leapfrog window: the solve runs over the window, so the
tendency it returns applies exactly the implicit solution - monotone,
no overshoot, whatever the stiffness (the measured §2e stiffnesses are
~8 for the stress and ~10^2 for the first interior interface, both far
beyond an explicit step).

Energy: the dry-static-energy solve conserves the mass-weighted column
enthalpy to round-off (zero-flux boundaries, geopotential frozen). The
momentum solve removes kinetic energy (stress plus diffusion). With
``heating=True`` the term adds

    Q = -[(u + u') du + (v + v') dv] / (2 c_p w)

whose mass-weighted column integral equals that removal identically
(the trapezoid identity u'^2 - u^2 = (u' + u)(u' - u)); with
``heating=False`` the removal is left to the measured residual, like
every other sink. Either way the term accumulates its area-mean
kinetic-energy tendency (W m^-2, negative = removal, via the analyzed
monopole) so a run records it (§2e "reported" quantities).

The slab temperature it reads is the column coupling's state-grid
field, carried to the product grid through the spectral basis (one
analysis and one synthesis per step; stated in §2e).

GPU (CuPy), like the model it acts on.
"""
from __future__ import annotations

import math

import numpy as np

from tropoi.physics.boundary_layer import (boundary_layer_depth, diffusivity_profile,
                                           exchange_coefficient, implicit_column_diffusion)

__all__ = ["BoundaryLayerDiffusion"]


class BoundaryLayerDiffusion:
    """FHZ06 bulk stress + implicit K-profile diffusion (plan §2e)."""

    def __init__(self, model, coupling, *, gravity: float, cp: float,
                 gas_constant: float, dt: float, roughness_length: float,
                 critical_richardson: float = 1.0, rich_crit_pbl: float = 1.0,
                 frac_inner: float = 0.1, gustiness: float = 0.0,
                 heating: bool = True):
        import cupy as cp_mod
        if coupling.surface_temperature is None:
            raise ValueError("the boundary layer needs a surface temperature")
        self.model = model
        self.coupling = coupling
        self.gravity = float(gravity)
        self.cp = float(cp)
        self.gas_constant = float(gas_constant)
        self.dt = float(dt)
        self.roughness_length = float(roughness_length)
        self.critical_richardson = float(critical_richardson)
        self.rich_crit_pbl = float(rich_crit_pbl)
        self.frac_inner = float(frac_inner)
        self.gustiness = float(gustiness)
        self.heating = bool(heating)
        ps = model._ps
        self._shape = ps.coslat.shape
        self._sigma_int = cp_mod.asarray(model.sigma.interfaces_array())[:, None]
        self._phi_s = ps.sh.inv_transform(model.phi_surface_lm).real.ravel()
        # budget: accumulated kinetic-energy removal (J m^-2) and time (s)
        self.budget_time = 0.0
        self.kinetic_sink = 0.0
        self.last_max_depth = 0.0

    def signature(self) -> dict:
        return {"kind": "boundary_layer_diffusion", "roughness_length": self.roughness_length,
                "critical_richardson": self.critical_richardson,
                "rich_crit_pbl": self.rich_crit_pbl, "frac_inner": self.frac_inner,
                "gustiness": self.gustiness, "heating": self.heating}

    # -- budget -----------------------------------------------------------
    def budget_means(self, *, reset: bool = False) -> dict:
        t = self.budget_time
        out = {"kinetic_sink": self.kinetic_sink / t if t > 0 else 0.0, "seconds": t,
               "max_depth": self.last_max_depth}
        if reset:
            self.budget_time = 0.0
            self.kinetic_sink = 0.0
        return out

    def state_dict(self) -> dict:
        return {"budget_time": self.budget_time, "kinetic_sink": self.kinetic_sink}

    def load_state_dict(self, d: dict) -> None:
        self.budget_time = float(d.get("budget_time", 0.0))
        self.kinetic_sink = float(d.get("kinetic_sink", 0.0))

    # -- the term ---------------------------------------------------------
    def _synthesize(self, sh, coeffs):
        import cupy as cp_mod
        batch = getattr(sh, "inv_transform_batch", None)
        if batch is not None:
            return batch(coeffs)
        return cp_mod.stack([sh.inv_transform(c).real for c in coeffs])

    def __call__(self, coeffs, t: float, window: float):
        import cupy as xp
        from tropoi.spatial.sigma_coordinate import hydrostatic_geopotential
        m = self.model
        K = m.nlev
        ps = m._ps
        sh_p = ps.sh
        g, cp, r = self.gravity, self.cp, self.gas_constant
        w = float(window)

        # Product-grid columns: T, p interfaces, winds, heights, slab T.
        temperature = self._synthesize(sh_p, coeffs[2 * K:3 * K]).real.reshape(K, -1)
        surf_p = xp.exp(sh_p.inv_transform(coeffs[3 * K]).real).ravel()
        p_int = self._sigma_int * surf_p[None, :]
        us, vs = [], []
        for k in range(K):
            u, v = m._wind_from(sh_p, ps.coslat, m._inv_laplacian(coeffs[k]),
                                m._inv_laplacian(coeffs[K + k]))
            us.append(u.ravel())
            vs.append(v.ravel())
        u = xp.stack(us)
        v = xp.stack(vs)
        phi_full, phi_below = hydrostatic_geopotential(m.sigma, temperature, self._phi_s, r)
        z_full = (phi_full - self._phi_s) / g
        z_interface = (phi_below[:K - 1] - self._phi_s) / g      # interior interfaces
        ts_state = self.coupling.surface_temperature.reshape(
            m.grid.point_latitudes.shape)
        ts = sh_p.inv_transform(m.sh.transform(ts_state)).real.ravel()

        # Surface exchange, depth, diffusivity, conductances.
        gust2 = self.gustiness * self.gustiness
        speed1 = xp.sqrt(u[-1] * u[-1] + v[-1] * v[-1] + gust2)
        coefficient = exchange_coefficient(
            ts, temperature[-1], z_full[-1], speed1, cp, g,
            roughness_length=self.roughness_length,
            critical_richardson=self.critical_richardson)
        u_star = xp.sqrt(coefficient) * speed1
        dse = cp * temperature + g * z_full
        h = boundary_layer_depth(z_full, dse, u * u + v * v, g,
                                 critical_richardson=self.rich_crit_pbl)
        diffusivity = diffusivity_profile(z_interface, u_star, h,
                                          frac_inner=self.frac_inner)
        t_interface = 0.5 * (temperature[:K - 1] + temperature[1:])
        rho_interface = p_int[1:K] / (r * t_interface)
        dz = z_full[:K - 1] - z_full[1:]
        conductance = rho_interface * diffusivity / dz
        mass = (p_int[1:] - p_int[:K]) / g
        rho1 = 0.5 * (p_int[-1] + p_int[-2]) / (r * temperature[-1])
        stress_conductance = rho1 * coefficient * speed1

        # The implicit solves and the changes they imply.
        u_new = implicit_column_diffusion(u, mass, conductance, w,
                                          bottom_conductance=stress_conductance)
        v_new = implicit_column_diffusion(v, mass, conductance, w,
                                          bottom_conductance=stress_conductance)
        s_new = implicit_column_diffusion(dse, mass, conductance, w)
        du = u_new - u
        dv = v_new - v
        d_temp = (s_new - dse) / cp
        kinetic_change = ((u_new + u) * du + (v_new + v) * dv) / (2.0 * w)  # W/kg
        if self.heating:
            d_temp = d_temp - (w / cp) * kinetic_change

        # Spectral tendencies: weak-form vector analysis for the momentum
        # rows, scalar analysis for temperature, both at the product cut.
        curl, div = m.so.vector_curl_div_spectral_batch(du / w, dv / w, truncate=False)
        out = xp.zeros_like(coeffs)
        shape = self._shape
        for k in range(K):
            out[k] = m._truncate(curl[k])
            out[K + k] = m._truncate(div[k])
            out[k, 0, :] = 0.0
            out[K + k, 0, :] = 0.0
            out[2 * K + k] = m._truncate(sh_p.transform((d_temp[k] / w).reshape(shape)))

        # Area-mean kinetic-energy tendency (W m^-2, negative = removal, the
        # sign the sink comparisons of the plan use) through the analyzed
        # monopole of the column integral; accumulated like the coupling's
        # budget (dt of real time per call).
        column_rate = (mass * kinetic_change).sum(axis=0)
        mean_sink = float(sh_p.transform(column_rate.reshape(shape))[0, 0].real
                          / math.sqrt(4.0 * math.pi))
        self.kinetic_sink += self.dt * mean_sink
        self.budget_time += self.dt
        self.last_max_depth = float(h.max())
        return out
