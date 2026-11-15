"""Dry boundary layer: depth, K profile and implicit column diffusion.

The scheme pre-registered in ``docs/radiation/MARS_MI_PLAN.md`` §2e: the
dry K-profile boundary layer of Frierson, Held & Zurita-Gotor (2006) in
the structure of Isca's ``diffusivity`` module (``do_simple``, no
entrainment, no free-atmosphere diffusion), with the FHZ06-simplified
exchange coefficient (:func:`tropoi.physics.surface.drag_coefficient`)
and neutral similarity in place of the full Monin-Obukhov functions.
That simplification is stated in the plan; Isca's grey-Mars case runs
the full forms, which :mod:`tropoi.physics.monin_obukhov` now provides
as an opt-in alternative (``scheme = "monin_obukhov"``). FHZ06 remains
the default and its arithmetic here is unchanged.

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
* :func:`monin_obukhov_diffusivity_profile` - the same structure with
  Isca's stability functions, returning distinct K_m and K_h; it reduces
  exactly to the previous function in the neutral limit.
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
           "monin_obukhov_diffusivity_profile", "implicit_column_diffusion"]


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
                         *, critical_richardson: float = 1.0,
                         wind_floor: float = 1e-12, wind_offset: float = 0.0):
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
    level, as there.

    Two mutually exclusive regularizations of a calm column are offered,
    because the two schemes this model carries disagree about it.
    ``wind_floor`` (the default, 1e-12 m^2/s^2) clamps the denominator, as
    the FHZ06 path has always done. ``wind_offset``, when non-zero, adds
    instead of clamping - Isca's ``small = 1e-4`` m^2/s^2 - which is what
    the Monin-Obukhov path uses so that the depth diagnosis and the surface
    exchange treat a calm column the same way. An offset of 1e-4 keeps Ri
    finite and *small* at vanishing wind, where a 1e-12 clamp lets it grow
    to ~1e12 and so reports a boundary layer one level deep; with the
    offset the column instead looks neutral and the first crossing is found
    higher up. Asking for both at once is an error; pass
    ``wind_floor=0.0`` alongside a ``wind_offset``."""
    xp = _xp(z_full, dry_static_energy)
    if wind_offset > 0.0 and wind_floor > 0.0:
        raise ValueError("give either wind_floor or wind_offset, not both "
                         "(pass wind_floor=0.0 with a wind_offset)")
    K = z_full.shape[0]
    s_bot = dry_static_energy[-1]
    denominator = (wind_speed_sq + wind_offset if wind_offset
                   else xp.maximum(wind_speed_sq, wind_floor))
    rich = (gravity * z_full * (dry_static_energy - s_bot) / s_bot
            / denominator)
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


def monin_obukhov_diffusivity_profile(z, u_star, b_star, h, *,
                                      frac_inner: float = 0.1,
                                      options=None):
    """Stability-dependent ``(K_m, K_h)`` at heights ``z`` above the
    surface: Isca ``diffusivity.diffusivity_pbl`` with ``mo_diff``.

    ``z`` is (m, npts), ``u_star``, ``b_star`` and ``h`` are (npts,).
    ``options`` is a
    :class:`~tropoi.physics.monin_obukhov.MoninObukhovOptions` (its
    defaults, i.e. Isca's, when None).

    Three regions, with ``h_i = frac_inner h``:

    * ``z < h_i`` - the surface layer proper, ``K = k u* z / phi(zeta(z))``
      with ``zeta(z) = -k b* z / u*^2``: the full Monin-Obukhov profile,
      evaluated at the height in question.
    * ``h_i <= z < h`` - the outer layer. Isca does *not* continue to
      evaluate ``phi`` here. It takes ``K(h_i)`` and scales it::

          K = K(h_i) (z / h_i) (1 - (z - h_i)/(h - h_i))^2

      so the stability function is frozen at its ``h_i`` value while the
      linear growth continues and the parabola closes the profile at
      ``z = h``. Because ``K(h_i) z / h_i = k u* z / phi(zeta(h_i))``,
      this is "neutral-shaped growth times a frozen stability factor",
      which is how the scheme avoids ``phi`` misbehaving where similarity
      theory no longer holds.
    * ``z >= h`` - zero.

    In the neutral limit (``phi = 1``) this collapses onto
    :func:`diffusivity_profile`, the FHZ06 form: there
    ``K(h_i) z / h_i = k u* z``, so the two differ only by the stability
    factor. The agreement is to rounding, not bit for bit, because this
    route multiplies and divides by ``h_i`` where the FHZ06 one does not;
    it is checked in the tests at ~1e-15 relative. That identity is what
    makes the FHZ06 taper already Isca-consistent in the neutral limit.
    """
    from tropoi.physics.monin_obukhov import MoninObukhovOptions, diffusivities
    xp = _xp(z, u_star)
    opts = MoninObukhovOptions() if options is None else options
    h_inner = frac_inner * h
    span = xp.maximum(h - h_inner, 1e-30)
    inner_m, inner_h = diffusivities(z, u_star, b_star, opts)
    ref_m, ref_h = diffusivities(h_inner, u_star, b_star, opts)
    # the outer-layer shape; h_inner > 0 wherever h > 0, and a zero-depth
    # column is caught by the z >= h branch below
    safe_inner = xp.maximum(h_inner, 1e-30)
    factor = (z / safe_inner) * xp.clip(1.0 - (z - h_inner) / span, 0.0, 1.0) ** 2
    below = z < h_inner
    above = z >= h
    k_m = xp.where(above, 0.0, xp.where(below, inner_m, ref_m * factor))
    k_h = xp.where(above, 0.0, xp.where(below, inner_h, ref_h * factor))
    return k_m, k_h


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
