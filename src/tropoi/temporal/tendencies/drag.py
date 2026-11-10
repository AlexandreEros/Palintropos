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

__all__ = ["schneider_liu_drag_profile", "LatitudeDependentRayleighDrag"]


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
