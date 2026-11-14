"""Dry boundary layer: depth, K profile and implicit column diffusion.

The scheme pre-registered in ``docs/radiation/MARS_MI_PLAN.md`` §2e: the
dry K-profile boundary layer of Frierson, Held & Zurita-Gotor (2006) in
the structure of Isca's ``diffusivity`` module (``do_simple``, no
entrainment, no free-atmosphere diffusion), with the FHZ06-simplified
exchange coefficient (:func:`tropoi.physics.surface.drag_coefficient`)
and neutral similarity in place of the full Monin-Obukhov functions.
That simplification is stated in the plan; Isca's grey-Mars case runs
the full forms.

Pieces, all column-wise over (nlev, npts) arrays ordered top to bottom:

* :func:`exchange_coefficient` - the surface exchange coefficient from
  the lowest-level state, the same arithmetic as the bulk sensible flux.
* :func:`boundary_layer_depth` - the lowest height where the bulk
  Richardson number of the dry static energy reaches a critical value
  (Isca ``pbl_depth``, stable/neutral branch), interpolated linearly in
  Ri between the bracketing full levels.
* :func:`diffusivity_profile` - K = k u* z below ``frac_inner`` h, the
  FHZ06 cubic taper K = k u* z (1 - (z - fb h)/((1 - fb) h))^2 up to h,
  zero above.
* :func:`implicit_column_diffusion` - one backward-Euler tridiagonal
  solve per column (Thomas algorithm), unconditionally stable, exactly
  conservative under zero-flux boundaries; an optional bottom
  conductance is the implicit bulk surface stress (the diffused quantity
  relaxes toward a zero surface value through it).

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import numpy as np

from tropoi.physics.surface import VON_KARMAN, drag_coefficient

__all__ = ["exchange_coefficient", "boundary_layer_depth", "diffusivity_profile",
           "implicit_column_diffusion"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def exchange_coefficient(surface_temperature, air_temperature, air_height, speed,
                         cp: float, gravity: float, *, roughness_length: float,
                         critical_richardson: float = 1.0):
    """The FHZ06 exchange coefficient from the lowest-level state: the
    same Richardson number and :func:`~tropoi.physics.surface.
    drag_coefficient` the bulk sensible flux uses, without the flux.
    ``speed`` already includes any gustiness."""
    xp = _xp(surface_temperature, air_temperature)
    dse_air = cp * air_temperature + gravity * air_height
    dse_sfc = cp * surface_temperature
    richardson = (gravity * air_height * (dse_air - dse_sfc) / dse_sfc
                  / xp.maximum(speed * speed, 1e-12))
    return drag_coefficient(air_height, roughness_length, richardson, critical_richardson)


def boundary_layer_depth(z_full, dry_static_energy, wind_speed_sq, gravity: float,
                         *, critical_richardson: float = 1.0):
    """Boundary-layer depth h (npts,) above the surface.

    ``z_full`` and ``dry_static_energy`` are (nlev, npts) full-level
    height above the surface (m) and dry static energy (J/kg), top to
    bottom; ``wind_speed_sq`` the squared full-level wind (m^2/s^2). The
    bulk Richardson number against the lowest level,

        Ri_k = g z_k (s_k - s_bot) / s_bot / max(|v_k|^2, 1e-12),

    is scanned upward from the lowest level; h interpolates linearly in
    Ri where it first exceeds ``critical_richardson`` (Isca
    ``diffusivity.pbl_depth``, the branch its ``do_simple`` always
    takes). A column that never crosses keeps h at its lowest full
    level, as there."""
    xp = _xp(z_full, dry_static_energy)
    K = z_full.shape[0]
    s_bot = dry_static_energy[-1]
    rich = (gravity * z_full * (dry_static_energy - s_bot) / s_bot
            / xp.maximum(wind_speed_sq, 1e-12))
    h = z_full[-1].copy()
    found = xp.zeros(h.shape, dtype=bool)
    rich_below, z_below = rich[-1], z_full[-1]
    for k in range(K - 2, -1, -1):
        rich_here, z_here = rich[k], z_full[k]
        cross = (~found) & (rich_here > critical_richardson)
        denom = xp.where(cross, rich_here - rich_below, 1.0)
        interpolated = z_here + ((z_below - z_here)
                                 * (rich_here - critical_richardson) / denom)
        h = xp.where(cross, interpolated, h)
        found = found | cross
        rich_below, z_below = rich_here, z_here
    return h


def diffusivity_profile(z, u_star, h, *, frac_inner: float = 0.1,
                        von_karman: float = VON_KARMAN):
    """K (same shape as ``z``) at heights ``z`` above the surface.

    ``z`` is (m, npts) (any set of heights, normally the interior
    interfaces), ``u_star`` and ``h`` (npts,). Neutral FHZ06 profile:
    K = k u* z below ``frac_inner`` h, the cubic taper
    K = k u* z (1 - (z - fb h)/((1 - fb) h))^2 between, zero at and
    above h."""
    xp = _xp(z, u_star)
    h_inner = frac_inner * h
    span = xp.maximum(h - h_inner, 1e-30)
    taper = xp.where(z < h_inner, 1.0,
                     xp.clip(1.0 - (z - h_inner) / span, 0.0, 1.0) ** 2)
    return von_karman * u_star * z * taper


def implicit_column_diffusion(q, layer_mass, conductance, window: float, *,
                              bottom_conductance=None):
    """Backward-Euler vertical diffusion of ``q`` over ``window`` seconds.

    ``q`` and ``layer_mass`` are (nlev, npts) top to bottom (mass in
    kg/m^2); ``conductance[i]`` (nlev-1, npts), in kg m^-2 s^-1, couples
    layers i and i+1 through the interface between them (rho K / dz
    there). ``bottom_conductance`` (npts,), when given, couples the
    lowest layer to a surface value of zero - the implicit bulk stress
    rho C |v| for momentum. The discrete system

        m_k (q'_k - q_k) = w [ g_{k-1} (q'_{k-1} - q'_k)
                             + g_k (q'_{k+1} - q'_k) ]  (- w g_b q'_bot)

    is symmetric positive definite and is solved by the Thomas
    algorithm, one vectorized sweep over levels. With zero-flux
    boundaries the mass-weighted column integral of q is conserved to
    round-off; every solution is free of over- and undershoots.
    Returns the new ``q'``."""
    xp = _xp(q, layer_mass)
    K = q.shape[0]
    w = float(window)
    g_above = [None] * K          # conductance to the layer above
    g_below = [None] * K          # conductance to the layer below
    zero = xp.zeros_like(q[0])
    for k in range(K):
        g_above[k] = conductance[k - 1] if k > 0 else zero
        g_below[k] = conductance[k] if k < K - 1 else (
            bottom_conductance if bottom_conductance is not None else zero)
    # Thomas: diag b_k = m_k + w (g_above + g_below), sub a_k = -w g_above,
    # super c_k = -w g_below (interior), rhs d_k = m_k q_k.
    b_prime = [None] * K
    d_prime = [None] * K
    b_prime[0] = layer_mass[0] + w * (g_above[0] + g_below[0])
    d_prime[0] = layer_mass[0] * q[0]
    for k in range(1, K):
        c_prev = -w * conductance[k - 1]              # super-diagonal of row k-1
        a_k = -w * conductance[k - 1]                 # sub-diagonal of row k
        m = a_k / b_prime[k - 1]
        b_prime[k] = layer_mass[k] + w * (g_above[k] + g_below[k]) - m * c_prev
        d_prime[k] = layer_mass[k] * q[k] - m * d_prime[k - 1]
    out = xp.empty_like(q)
    out[K - 1] = d_prime[K - 1] / b_prime[K - 1]
    for k in range(K - 2, -1, -1):
        out[k] = (d_prime[k] + w * conductance[k] * out[k + 1]) / b_prime[k]
    return out
