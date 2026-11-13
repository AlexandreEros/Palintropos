"""Rayleigh drag whose coefficient depends on latitude as well as on level.

Held–Suarez drag depends on the level only, so it is diagonal in the
spectral basis (``held_suarez.rayleigh_drag_damping``). Schneider & Liu
(2009) switch the drag off near the equator of their Jupiter model,

    dV/dt = -k(lat, sigma) V,
    k = k0(lat) max(0, (sigma - sigma_b) / (1 - sigma_b)),

with ``k0`` constant poleward of a cutoff latitude and decaying to zero
equatorward of it. That is not diagonal: the force is formed on the product
grid and mapped to vorticity and divergence tendencies with the same
weak-form vector analysis the momentum equations use.

It is a *lagged* stepper term (evaluated at the old time level): a forward
step of a damping is stable for ``2 dt k < 2``, far from binding here.

GPU (CuPy), like the model it acts on.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["schneider_liu_drag_profile", "LatitudeDependentRayleighDrag",
           "MomentumSinkHeating", "rayleigh_sponge_damping"]


def rayleigh_sponge_damping(sigma_full, n_degrees: int, *, reference_pressure: float,
                            bottom_pressure: float, efold_days: float):
    """Isca's top sponge (``damping_driver`` ``rayleigh``, commit d1321ac3) as a
    :class:`~tropoi.temporal.hooks.DiagonalDamping` on vorticity and divergence:

        rate = (1 / efold) ((p_b - p) / p_b)^2   for p < p_b,  0 below,

    with ``p = sigma_k * reference_pressure``. Isca uses the local full-level
    pressure; a diagonal damping needs one rate per level, and on a flat
    planet the two differ by the surface-pressure variation. Temperature and
    ln p_s are untouched. Its kinetic-energy loss is returned as heat by
    :class:`MomentumSinkHeating` like any other damper (Isca's
    ``do_conserve_energy``)."""
    from tropoi.temporal.hooks import DiagonalDamping
    sigma = np.asarray(sigma_full, dtype=np.float64)
    p_b = float(bottom_pressure)
    if not (p_b > 0.0 and efold_days > 0.0 and reference_pressure > 0.0):
        raise ValueError("sponge pressures and e-folding time must be positive")
    p = sigma * float(reference_pressure)
    rate = np.where(p < p_b, ((p_b - p) / p_b) ** 2 / (float(efold_days) * 86400.0), 0.0)
    K = sigma.size
    rates = np.zeros((3 * K + 1, int(n_degrees)))
    rates[0:K, :] = rate[:, None]
    rates[K:2 * K, :] = rate[:, None]
    return DiagonalDamping("rayleigh_sponge", rates,
                           params={"reference_pressure": float(reference_pressure),
                                   "bottom_pressure": p_b, "efold_days": float(efold_days),
                                   "sigma_full": [float(s) for s in sigma]})


def schneider_liu_drag_profile(latitude, k0: float, cutoff_deg: float = 16.3,
                               decay_deg: float = 2.0):
    """``k0`` poleward of ``cutoff_deg``, ``k0 exp(-((cutoff - |lat|) /
    decay)^2)`` equatorward of it.

    SL09 state the cutoff (16.3 degrees) and that the coefficient "decreases
    rapidly to zero at lower latitudes" without giving the function; the
    Gaussian shoulder and its 2-degree width are this code's choice.
    """
    xp = np
    if type(latitude).__module__.split(".")[0] == "cupy":
        import cupy as xp  # noqa: WPS433
    lat = xp.abs(latitude) * (180.0 / math.pi)
    shoulder = xp.exp(-((cutoff_deg - lat) / decay_deg) ** 2)
    return k0 * xp.where(lat >= cutoff_deg, 1.0, shoulder)


class LatitudeDependentRayleighDrag:
    """``-k0(lat) ramp(sigma_k) V_k`` as vorticity/divergence tendencies."""

    def __init__(self, model, k0_of_latitude, *, sigma_b: float, params: dict):
        import cupy as cp
        self.model = model
        self.sigma_b = float(sigma_b)
        self.params = dict(params)
        ps = model._ps
        geometry = ps.geometry if ps.geometry is not None else model.grid
        self._k0 = cp.asarray(k0_of_latitude(geometry.point_latitudes)).ravel()
        sigma = model.sigma.full_levels_array()
        ramp = np.maximum(0.0, (sigma - self.sigma_b) / (1.0 - self.sigma_b))
        self._levels = [int(k) for k in np.nonzero(ramp > 0.0)[0]]
        self._ramp = cp.asarray(ramp[self._levels])[:, None]
        self.max_rate = float(self._k0.max() * ramp.max()) if self._levels else 0.0

    def signature(self) -> dict:
        return {"kind": "latitude_dependent_rayleigh_drag", "sigma_b": self.sigma_b,
                "params": self.params}

    def __call__(self, coeffs, t: float, window: float):
        import cupy as cp
        m = self.model
        K = m.nlev
        out = cp.zeros_like(coeffs)
        if not self._levels:
            return out
        ps = m._ps
        us, vs = [], []
        for k in self._levels:
            u, v = m._wind_from(ps.sh, ps.coslat, m._inv_laplacian(coeffs[k]),
                                m._inv_laplacian(coeffs[K + k]))
            us.append(u.ravel())
            vs.append(v.ravel())
        rate = -self._ramp * self._k0[None, :]
        curl, div = m.so.vector_curl_div_spectral_batch(rate * cp.stack(us),
                                                        rate * cp.stack(vs), truncate=False)
        for i, k in enumerate(self._levels):
            out[k] = m._truncate(curl[i])
            out[K + k] = m._truncate(div[i])
            out[k, 0, :] = 0.0
            out[K + k, 0, :] = 0.0
        return out


class MomentumSinkHeating:
    """Return the energy removed by momentum sinks and diffusion as heat.

    ``sinks`` are callables ``X -> dX/dt`` (a drag, a hyperdiffusion). Two
    parts of what they do to the model's energy (``column_coupling.
    global_energy``) are returned to the temperature:

    * **kinetic.** With ``dV`` the wind tendency implied by their vorticity
      and divergence rows, ``Q = -V . dV / c_p`` at every grid point and
      level, so the mass-weighted enthalpy gained equals the kinetic energy
      lost: ``<p_s dsigma c_p Q> = -<p_s dsigma V . dV>``. Q is local and
      may be negative where a scale-selective diffusion carries kinetic
      energy into a point; only its integral is sign-definite.
    * **enthalpy.** A diffusion of T along sigma surfaces that is not
      weighted by ``p_s`` changes ``<p_s T_k>``. With ``dT_k`` their
      temperature rows, the horizontally uniform ``-<p_s dT_k> / <p_s>`` is
      added to level k, which cancels that change level by level. (Measured
      on a 100-bar Jupiter column at T9: 0.11 of the initial kinetic energy
      per day, against 0.78 through the momentum rows.)

    An *explicit* stepper term, evaluated at the centred time level ``X^n``
    like the dynamics: the energy a centred-implicit damper removes over a
    leapfrog step is ``2 dt r |(X^{n+1} + X^{n-1}) / 2|^2``, which ``r
    |X^n|^2`` matches to second order. The balance is therefore exact for
    the instantaneous tendencies and exact in a run only to the
    time-truncation error; the analysed heating is truncated like every
    other product.
    """

    def __init__(self, model, sinks, cp: float):
        self.model = model
        self.sinks = tuple(sinks)
        self.cp = float(cp)
        if not self.sinks:
            raise ValueError("no sinks to return as heat")

    def signature(self) -> dict:
        return {"kind": "momentum_sink_heating", "cp": self.cp, "sinks": len(self.sinks)}

    def __call__(self, coeffs):
        import cupy as cp
        m = self.model
        K = m.nlev
        sink = self.sinks[0](coeffs)[:3 * K]
        for other in self.sinks[1:]:
            sink = sink + other(coeffs)[:3 * K]
        # One synthesis for everything: the wind derivatives of the state and
        # of the sink tendency (4 fields x 2K levels), the sinks' temperature
        # rows (K) and ln p_s (1).
        psi = m._inv_laplacian(cp.concatenate([coeffs[0:K], sink[0:K]]))
        chi = m._inv_laplacian(cp.concatenate([coeffs[K:2 * K], sink[K:2 * K]]))
        psi_lam, psi_snt = m._stack_derivs(psi)
        chi_lam, chi_snt = m._stack_derivs(chi)
        fields = m.sh.inv_transform_batch(cp.concatenate(
            [psi_lam, psi_snt, chi_lam, chi_snt, sink[2 * K:3 * K], coeffs[3 * K][None]], axis=0))
        fields = fields.reshape(fields.shape[0], -1)
        n = 2 * K
        coslat = m._state_coslat.ravel()
        wind_u = (fields[n:2 * n] / m.R + fields[2 * n:3 * n]) / coslat
        wind_v = (fields[0:n] - fields[3 * n:4 * n] / m.R) / coslat
        heating = -(wind_u[:K] * wind_u[K:] + wind_v[:K] * wind_v[K:]) / self.cp
        d_temp = fields[4 * n:4 * n + K]
        ps = cp.exp(fields[4 * n + K])
        mass = cp.asarray(m.grid.cell_areas, dtype=cp.float64).ravel() * ps
        uniform = -(mass * d_temp).sum(axis=1) / mass.sum()
        spec = m.sh.transform_batch(heating)
        cut = m._trunc_cut
        spec[:, cut + 1:, :] = 0.0
        spec[:, :, cut + 1:] = 0.0
        spec[:, 0, 0] += uniform * math.sqrt(4.0 * math.pi)
        out = cp.zeros_like(coeffs)
        out[2 * K:3 * K] = spec
        return out
