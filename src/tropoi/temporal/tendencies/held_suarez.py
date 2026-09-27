"""Held & Suarez (1994) forcing for the dry primitive-equation core.

The forcing is prescribed by the paper (docs/held_suarez/PROTOCOL.md §1.1),
not a modelling choice:

    dv/dt = ... - k_v(sigma) v
    dT/dt = ... - k_T(phi, sigma) [T - T_eq(phi, p)]

    T_eq = max{ T_min, [T_max - dT_y sin^2 phi - dtheta_z ln(p/p0) cos^2 phi] (p/p0)^kappa }
    k_T  = k_a + (k_s - k_a) max(0, (sigma - sigma_b)/(1 - sigma_b)) cos^4 phi
    k_v  = k_f max(0, (sigma - sigma_b)/(1 - sigma_b))

with sigma = p/p_s for the instantaneous p_s (so p = sigma p_s in T_eq).

How each term enters the time scheme (plan §3; SEMI_IMPLICIT.md §8):

* **Newtonian relaxation** (:class:`NewtonianRelaxation`) is an explicit
  tendency term: evaluated on the backend product sampling at the middle
  time level, analyzed, and truncated at the product cut exactly like every
  other analyzed nonlinear product of the PE tendency (so a state
  band-limited at the cut stays band-limited and the SI operator's mask
  remains the exact linearization).
* **Rayleigh drag** is linear, horizontally uniform and depends on the
  model level only, so it damps zeta_k and delta_k exactly per level:
  :func:`rayleigh_drag_damping`, a :class:`~tropoi.temporal.hooks.
  DiagonalDamping` applied implicitly with ``1 / (1 + 2 dt k_v,k)``.
* **del^8 hyperdiffusion** (a numerical choice of the core, not HS forcing)
  on zeta, delta and T with ``K8 = 1 / (tau c_ref^4)``, ``c_l = l(l+1)/a^2``:
  :func:`hyperdiffusion_damping`, per-degree factor
  ``1 / (1 + 2 dt K8 c_l^4)``. ``c_0 = 0``, so the global-mean temperature
  (and the zeta/delta monopoles, identically zero) are never damped: this is
  diffusion of T' = T - global mean. ln p_s is not diffused.

The formula functions are array-module agnostic (NumPy or CuPy); this
module imports no CuPy, so the formulas are testable on CPU. The Newtonian
term needs a :class:`PrimitiveEquationsModel` (GPU).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tropoi.temporal.hooks import DiagonalDamping

__all__ = ["DAY_SECONDS", "HeldSuarezParameters", "equilibrium_temperature",
           "newtonian_rate", "rayleigh_rate", "newtonian_tendency_grid",
           "NewtonianRelaxation", "rayleigh_drag_damping",
           "hyperdiffusion_coefficient", "hyperdiffusion_damping"]

DAY_SECONDS = 86400.0


def _xp(*arrays):
    """CuPy if any argument is a CuPy array, else NumPy (no eager CuPy import)."""
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


@dataclass(frozen=True)
class HeldSuarezParameters:
    """HS94 constants (PROTOCOL.md §1.1). Rates in s^-1, temperatures in K."""

    sigma_b: float = 0.7
    k_f: float = 1.0 / DAY_SECONDS
    k_a: float = 1.0 / (40.0 * DAY_SECONDS)
    k_s: float = 1.0 / (4.0 * DAY_SECONDS)
    delta_t_y: float = 60.0
    delta_theta_z: float = 10.0
    t_max: float = 315.0
    t_min: float = 200.0
    p0: float = 1.0e5
    kappa: float = 2.0 / 7.0

    def __post_init__(self):
        for name, val in asdict(self).items():
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
        if not (0.0 <= self.sigma_b < 1.0):
            raise ValueError(f"sigma_b must be in [0, 1), got {self.sigma_b}")
        for name in ("k_f", "k_a", "k_s"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.p0 <= 0.0 or self.kappa <= 0.0:
            raise ValueError("p0 and kappa must be > 0")

    def as_dict(self) -> dict:
        return asdict(self)


def _sigma_ramp(sigma, params: HeldSuarezParameters, xp):
    return xp.maximum(0.0, (sigma - params.sigma_b) / (1.0 - params.sigma_b))


def equilibrium_temperature(lat, p, params: HeldSuarezParameters):
    """T_eq(phi, p) in K; ``lat`` in radians, ``p`` in Pa (broadcastable)."""
    xp = _xp(lat, p)
    lat = xp.asarray(lat, dtype=np.float64)
    p = xp.asarray(p, dtype=np.float64)
    s = xp.sin(lat)
    c = xp.cos(lat)
    ratio = p / params.p0
    t = (params.t_max - params.delta_t_y * s * s
         - params.delta_theta_z * xp.log(ratio) * c * c) * ratio ** params.kappa
    return xp.maximum(params.t_min, t)


def newtonian_rate(lat, sigma, params: HeldSuarezParameters):
    """k_T(phi, sigma) in s^-1."""
    xp = _xp(lat, sigma)
    lat = xp.asarray(lat, dtype=np.float64)
    sigma = xp.asarray(sigma, dtype=np.float64)
    c = xp.cos(lat)
    c2 = c * c
    return params.k_a + (params.k_s - params.k_a) * _sigma_ramp(sigma, params, xp) * (c2 * c2)


def rayleigh_rate(sigma, params: HeldSuarezParameters):
    """k_v(sigma) in s^-1."""
    xp = _xp(sigma)
    sigma = xp.asarray(sigma, dtype=np.float64)
    return params.k_f * _sigma_ramp(sigma, params, xp)


def newtonian_tendency_grid(temperature, lat, sigma, surface_pressure,
                            params: HeldSuarezParameters, rate=None):
    """-k_T (T - T_eq(phi, sigma p_s)) pointwise.

    ``temperature`` (K, npts), ``lat`` (npts,) radians, ``sigma`` (K,) full
    levels, ``surface_pressure`` (npts,) Pa. ``rate`` may pass a precomputed
    k_T (K, npts) (it is state-independent).
    """
    xp = _xp(temperature)
    sigma = xp.asarray(sigma, dtype=np.float64)[:, None]
    lat = xp.asarray(lat, dtype=np.float64)
    if rate is None:
        rate = newtonian_rate(lat[None, :], sigma, params)
    t_eq = equilibrium_temperature(lat[None, :], sigma * surface_pressure[None, :], params)
    return -rate * (temperature - t_eq)


class NewtonianRelaxation:
    """Explicit Newtonian temperature relaxation as a stepper hook.

    ``term(X)`` returns a PE stack that is zero except in the T rows:
    ``analyze(-k_T (T - T_eq(phi, sigma_k p_s)))`` on the model's product
    sampling, truncated at the product cut. Rows ``[zeta, delta, T, ln p_s]``.
    """

    def __init__(self, model, params: HeldSuarezParameters):
        self.model = model
        self.params = params
        ps = model._ps
        geometry = ps.geometry if ps.geometry is not None else model.grid
        self._sh = ps.sh
        self._lat = geometry.point_latitudes
        self._sigma = model.sigma.full_levels_array()
        xp = _xp(self._lat)
        self._rate = newtonian_rate(self._lat[None, :],
                                    xp.asarray(self._sigma)[:, None], params)
        self.max_rate = float(self._rate.max())
        self._cut = model._trunc_cut
        self._batched = bool(getattr(model, "batched_transforms", False))

    def signature(self) -> dict:
        return {"kind": "held_suarez_newtonian", "params": self.params.as_dict(),
                "sigma_full": [float(s) for s in self._sigma]}

    def grid_fields(self, coeffs):
        """(T (K, npts), p_s (npts,)) on the product sampling."""
        K = self.model.nlev
        xp = _xp(coeffs)
        sh = self._sh
        if self._batched:
            g = sh.inv_transform_batch(coeffs[2 * K:3 * K + 1])
            temperature, lnps = g[0:K], g[K]
        else:
            temperature = xp.stack([sh.inv_transform(coeffs[2 * K + k]).real
                                    for k in range(K)])
            lnps = sh.inv_transform(coeffs[3 * K]).real
        return temperature, xp.exp(lnps)

    def __call__(self, coeffs):
        K = self.model.nlev
        xp = _xp(coeffs)
        temperature, p_s = self.grid_fields(coeffs)
        forcing = newtonian_tendency_grid(temperature, self._lat, self._sigma, p_s,
                                          self.params, rate=self._rate)
        sh = self._sh
        if self._batched:
            spec = sh.transform_batch(forcing)
        else:
            spec = xp.stack([sh.transform(forcing[k]) for k in range(K)])
        cut = self._cut
        spec[:, cut + 1:, :] = 0.0
        spec[:, :, cut + 1:] = 0.0
        out = xp.zeros_like(coeffs)
        out[2 * K:3 * K] = spec
        return out


def rayleigh_drag_damping(sigma_full, n_degrees: int, params: HeldSuarezParameters
                          ) -> DiagonalDamping:
    """Rayleigh drag on zeta_k and delta_k at rate k_v(sigma_k), every degree."""
    sigma_full = np.asarray(sigma_full, dtype=np.float64)
    K = sigma_full.size
    kv = rayleigh_rate(sigma_full, params)
    rates = np.zeros((3 * K + 1, int(n_degrees)))
    rates[0:K, :] = kv[:, None]
    rates[K:2 * K, :] = kv[:, None]
    return DiagonalDamping("held_suarez_rayleigh", rates,
                           params={"k_f": params.k_f, "sigma_b": params.sigma_b,
                                   "sigma_full": [float(s) for s in sigma_full]})


def hyperdiffusion_coefficient(radius: float, efold_seconds: float,
                               reference_degree: int, order: int = 4) -> float:
    """K such that K c_ref^order = 1/tau (c_l = l(l+1)/a^2)."""
    c_ref = reference_degree * (reference_degree + 1.0) / radius ** 2
    return 1.0 / (efold_seconds * c_ref ** order)


def hyperdiffusion_damping(nlev: int, l_max: int, radius: float,
                           efold_seconds: float, reference_degree: int,
                           order: int = 4) -> DiagonalDamping:
    """del^(2 order) hyperdiffusion on zeta, delta, T: rate K c_l^order.

    ``reference_degree`` is the degree whose e-folding time is
    ``efold_seconds``. ln p_s is not diffused; l = 0 has rate 0 exactly.
    """
    if not (math.isfinite(efold_seconds) and efold_seconds > 0):
        raise ValueError(f"efold_seconds must be finite and > 0, got {efold_seconds}")
    if not (1 <= int(reference_degree) <= int(l_max)):
        raise ValueError(f"reference_degree must be in [1, l_max={l_max}], got {reference_degree}")
    K = int(nlev)
    k8 = hyperdiffusion_coefficient(radius, efold_seconds, int(reference_degree), order)
    l = np.arange(int(l_max) + 1, dtype=np.float64)
    c_l = l * (l + 1.0) / radius ** 2
    per_degree = k8 * c_l ** order
    rates = np.zeros((3 * K + 1, l.size))
    rates[0:3 * K, :] = per_degree[None, :]
    return DiagonalDamping(f"hyperdiffusion_del{2 * order}", rates,
                           params={"coefficient": k8, "order": order,
                                   "efold_seconds": float(efold_seconds),
                                   "reference_degree": int(reference_degree),
                                   "radius": float(radius)})
