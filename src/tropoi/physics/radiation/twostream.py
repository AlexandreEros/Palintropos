"""Two-stream radiative transfer for plane-parallel columns (thermal and stellar).

One solver serves every optics provider (gray, band, correlated-k) and both
absorbing and scattering media, so cloud and aerosol optics can be added
later by supplying ``ssa`` and ``asym`` without touching this file.

Conventions (shared with RADIATION.md):

* axis 0 is the vertical; index 0 is the TOP. ``K`` layers, ``K + 1``
  interfaces. Every other axis ("points": columns, spectral points) is
  broadcast, exactly like the sigma column operators of the dynamical core.
* fluxes are hemispheric flux densities in W m^-2 (or any consistent unit),
  each positive in its own direction: ``up`` and ``down``.
  ``net = down - up`` is the net downward flux; a layer gains
  ``net[k] - net[k + 1]``.
* ``tau`` is the layer extinction optical depth measured along the vertical.
* thermal sources are passed as flux-Planck values ``pi B`` (for a gray
  medium ``sigma T^4``), one at the top and one at the bottom of each layer;
  the source is linear in optical depth inside a layer.

Method: each homogeneous layer is reduced to a diffuse reflectance ``R``, a
diffuse transmittance ``T`` and emergent source fluxes from the generalized
two-stream equations (Meador & Weaver 1980; Toon et al. 1989)

    dF+/dtau = g1 F+ - g2 F- - S+ ,      dF-/dtau = g2 F+ - g1 F- + S- ,

and the layers are combined by the adding method (two sweeps, no matrix
solve). Thermal closure: ``g1 = D [1 - w (1 + g) / 2]``, ``g2 = D w (1 - g)
/ 2`` with diffusivity factor ``D`` (``D = 2`` is the hemispheric mean of
Toon et al.; ``D = 1.66`` the usual diffusivity approximation; a
non-scattering layer then transmits ``exp(-D tau)``). Stellar closures:
Eddington, quadrature or hemispheric mean, with optional delta scaling.

The layer formulas are checked against an independent numerical solution of
the same differential equations, against closed forms, and against exact
angular integration (exponential integrals) in
``tests/test_radiation_twostream.py``; the last comparison measures the
error of the two-stream approximation itself.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["thermal_coefficients", "stellar_coefficients", "delta_scale",
           "diffuse_layer", "thermal_layer_sources", "direct_layer",
           "adding", "thermal_fluxes", "stellar_fluxes"]

_STELLAR_CLOSURES = ("eddington", "quadrature", "hemispheric")


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def thermal_coefficients(ssa, asym, diffusivity: float):
    """(g1, g2) of the thermal two-stream closure with diffusivity ``D``."""
    d = float(diffusivity)
    return d * (1.0 - 0.5 * ssa * (1.0 + asym)), 0.5 * d * ssa * (1.0 - asym)


def stellar_coefficients(ssa, asym, mu0, closure: str = "quadrature"):
    """(g1, g2, g3) for a direct beam with cosine ``mu0`` (g4 = 1 - g3)."""
    if closure == "eddington":
        g1 = 0.25 * (7.0 - ssa * (4.0 + 3.0 * asym))
        g2 = -0.25 * (1.0 - ssa * (4.0 - 3.0 * asym))
        g3 = 0.25 * (2.0 - 3.0 * asym * mu0)
    elif closure == "quadrature":
        r3 = 3.0 ** 0.5
        g1 = 0.5 * r3 * (2.0 - ssa * (1.0 + asym))
        g2 = 0.5 * r3 * ssa * (1.0 - asym)
        g3 = 0.5 * (1.0 - r3 * asym * mu0)
    elif closure == "hemispheric":
        g1 = 2.0 - ssa * (1.0 + asym)
        g2 = ssa * (1.0 - asym)
        g3 = 0.5 * (1.0 - asym) + 0.0 * mu0
    else:
        raise ValueError(f"closure must be one of {_STELLAR_CLOSURES}, got {closure!r}")
    return g1, g2, g3


def delta_scale(tau, ssa, asym):
    """Delta scaling with forward fraction f = g^2 (Joseph et al. 1976)."""
    f = asym * asym
    denom = 1.0 - ssa * f
    return tau * denom, ssa * (1.0 - f) / denom, asym / (1.0 + asym)


def diffuse_layer(tau, g1, g2):
    """Diffuse reflectance and transmittance (R, T) of homogeneous layers.

    Written so the conservative limit (g1 = g2, k -> 0) is reached without
    cancellation: the denominator is ``k (1 + E) + g1 (1 - E)`` with
    ``1 - E`` from ``expm1``.
    """
    xp = _xp(tau, g1, g2)
    k = xp.sqrt(xp.maximum((g1 - g2) * (g1 + g2), 1e-24))
    e1 = xp.exp(-k * tau)
    one_minus_e2 = -xp.expm1(-2.0 * k * tau)
    denom = k * (2.0 - one_minus_e2) + g1 * one_minus_e2
    # tau = 0: denom = 2k, R = 0, T = 1 (no layer).
    return g2 * one_minus_e2 / denom, 2.0 * k * e1 / denom


def thermal_layer_sources(tau, g1, g2, refl, trans, planck_top, planck_bot):
    """Emergent fluxes (up at the layer top, down at the layer bottom) of a
    layer whose flux-Planck source is linear in tau from ``planck_top`` to
    ``planck_bot``, with no radiation incident on it.

    From the particular solution ``F+- = P(tau) +- P' / (g1 + g2)`` of the
    thermal closure. Optically thin layers (where ``P' = dP / tau`` is
    ill-conditioned) use the mean-source form ``(1 - R - T) (P_top + P_bot) /
    2``, which differs at second order in tau.
    """
    xp = _xp(tau, refl, planck_top)
    emis = 1.0 - refl - trans
    s = g1 + g2
    thin = tau * s < 1e-4
    safe = xp.where(thin, 1.0, tau * s)
    lin = (planck_bot - planck_top) * (1.0 + refl - trans) / safe
    mean = 0.5 * emis * (planck_top + planck_bot)
    up = xp.where(thin, mean, planck_top * (1.0 - refl) - trans * planck_bot + lin)
    down = xp.where(thin, mean, planck_bot * (1.0 - refl) - trans * planck_top - lin)
    return up, down


def direct_layer(tau, ssa, g1, g2, g3, mu0):
    """Response of homogeneous layers to a direct beam of unit flux through
    the layer top (measured on the horizontal).

    Returns ``(r_dir, t_dir, t_beam)``: diffuse flux leaving the top, diffuse
    flux leaving the bottom, and the unscattered beam transmission
    ``exp(-tau / mu0)`` (Meador & Weaver 1980, eqs. 14-15).
    """
    xp = _xp(tau, ssa, mu0)
    g4 = 1.0 - g3
    # Floor k at 1e-6: the closed form loses digits as k -> 0 (conservative
    # scattering); the floor costs O(k^2) = 1e-12 and keeps ~1e-10 accuracy.
    k = xp.sqrt(xp.maximum((g1 - g2) * (g1 + g2), 1e-12))
    # The closed form is singular at k mu0 = 1; step off it.
    kmu = k * mu0
    kmu = xp.where(xp.abs(1.0 - kmu) < 1e-6, kmu + 2e-6, kmu)
    a1 = g1 * g4 + g2 * g3
    a2 = g1 * g3 + g2 * g4
    e1 = xp.exp(-k * tau)
    e2 = e1 * e1
    t_beam = xp.exp(-tau / mu0)
    # Everything divided by exp(k tau) so no exponential grows.
    denom = (1.0 - kmu * kmu) * ((k + g1) + (k - g1) * e2)
    r_dir = ssa * ((1.0 - kmu) * (a2 + k * g3)
                   - (1.0 + kmu) * (a2 - k * g3) * e2
                   - 2.0 * k * (g3 - a2 * mu0) * e1 * t_beam) / denom
    t_tot = (t_beam * (1.0 - ssa * ((1.0 + kmu) * (a1 + k * g4)
                                    - (1.0 - kmu) * (a1 - k * g4) * e2) / denom)
             + ssa * 2.0 * k * (g4 + a1 * mu0) * e1 / denom)
    t_dir = t_tot - t_beam
    # Round-off can leave tiny negative diffuse fluxes for very thin layers.
    return xp.maximum(r_dir, 0.0), xp.maximum(t_dir, 0.0), t_beam


def adding(refl, trans, source_up, source_down, surface_albedo, surface_source,
           top_down=0.0):
    """Combine K layers and a lower boundary into interface fluxes.

    ``refl, trans, source_up, source_down`` have shape ``(K, ...)``;
    ``surface_albedo`` and ``surface_source`` (upward flux leaving the lower
    boundary by its own emission or by scattering of a direct beam) and
    ``top_down`` (diffuse flux incident at the top) broadcast against one
    level. Returns ``(up, down)`` of shape ``(K + 1, ...)``: the diffuse
    fluxes at every interface.
    """
    xp = _xp(refl, trans, source_up)
    K = refl.shape[0]
    shape = xp.broadcast(refl[0], xp.asarray(surface_albedo),
                         xp.asarray(surface_source), xp.asarray(top_down)).shape
    albedo = [None] * (K + 1)        # reflectance of everything below interface i
    src = [None] * (K + 1)           # upward flux at interface i with nothing incident
    denom = [None] * K
    albedo[K] = xp.broadcast_to(xp.asarray(surface_albedo, dtype=np.float64), shape)
    src[K] = xp.broadcast_to(xp.asarray(surface_source, dtype=np.float64), shape)
    for k in range(K - 1, -1, -1):
        denom[k] = 1.0 / (1.0 - refl[k] * albedo[k + 1])
        albedo[k] = refl[k] + trans[k] * trans[k] * albedo[k + 1] * denom[k]
        src[k] = source_up[k] + trans[k] * (src[k + 1] + albedo[k + 1] * source_down[k]) * denom[k]
    up = xp.empty((K + 1,) + shape, dtype=np.float64)
    down = xp.empty((K + 1,) + shape, dtype=np.float64)
    down[0] = top_down
    for k in range(K):
        up[k] = albedo[k] * down[k] + src[k]
        down[k + 1] = (trans[k] * down[k] + refl[k] * src[k + 1] + source_down[k]) * denom[k]
    up[K] = albedo[K] * down[K] + src[K]
    return up, down


def thermal_fluxes(tau, planck_top, planck_bot, *, surface_source, surface_albedo=0.0,
                   ssa=None, asym=None, diffusivity: float = 1.66, top_down=0.0):
    """Thermal (longwave) interface fluxes.

    ``tau`` (K, ...) layer optical depths; ``planck_top``/``planck_bot``
    (K, ...) flux-Planck values at the two edges of each layer (pass the same
    array twice for isothermal layers). ``surface_source`` is the upward flux
    emitted by the lower boundary and ``surface_albedo`` its reflectance of
    downwelling thermal radiation (``1 - emissivity`` for a surface; ``1``
    with ``surface_source = 0`` returns every downwelling watt, the
    energy-conserving closed bottom of an atmosphere with no surface).
    Without ``ssa`` the medium is purely absorbing.
    """
    xp = _xp(tau, planck_top)
    if ssa is None:
        ssa = 0.0 * tau
    if asym is None:
        asym = 0.0 * tau
    g1, g2 = thermal_coefficients(ssa, asym, diffusivity)
    refl, trans = diffuse_layer(tau, g1, g2)
    s_up, s_dn = thermal_layer_sources(tau, g1, g2, refl, trans, planck_top, planck_bot)
    refl, trans, s_up, s_dn = xp.broadcast_arrays(refl, trans, s_up, s_dn)
    return adding(refl, trans, s_up, s_dn, surface_albedo, surface_source, top_down)


def stellar_fluxes(tau, ssa, asym, mu0, toa_flux, *, surface_albedo=0.0,
                   closure: str = "quadrature", scale: bool = True,
                   diffuse_incidence: bool = False):
    """Stellar (shortwave) interface fluxes.

    ``toa_flux`` is the flux through a horizontal surface at the top; ``mu0``
    the cosine of the zenith angle (points with ``mu0 <= 0`` or zero flux
    return zeros). With ``diffuse_incidence`` the incident flux is isotropic
    over the downward hemisphere instead of a collimated beam (``mu0`` is
    then unused). Returns ``(up, down, down_direct)`` of shape ``(K + 1,
    ...)``; ``down`` includes the direct beam.
    """
    xp = _xp(tau, ssa, toa_flux, mu0)
    if scale:
        tau, ssa, asym = delta_scale(tau, ssa, asym)
    if diffuse_incidence:
        g1, g2, _ = stellar_coefficients(ssa, asym, 1.0, closure)
        refl, trans = diffuse_layer(tau, g1, g2)
        zero = 0.0 * refl
        refl, trans, zero = xp.broadcast_arrays(refl, trans, zero)
        up, down = adding(refl, trans, zero, zero, surface_albedo, 0.0, toa_flux)
        return up, down, 0.0 * down
    mu0 = xp.asarray(mu0, dtype=np.float64)
    lit = mu0 > 0.0
    mu = xp.where(lit, mu0, 1.0)
    flux = xp.where(lit, toa_flux, 0.0)
    g1, g2, g3 = stellar_coefficients(ssa, asym, mu, closure)
    refl, trans = diffuse_layer(tau, g1, g2)
    r_dir, t_dir, t_beam = direct_layer(tau, ssa, g1, g2, g3, mu)
    refl, trans, r_dir, t_dir, t_beam = xp.broadcast_arrays(refl, trans, r_dir, t_dir, t_beam)
    K = refl.shape[0]
    shape = xp.broadcast(refl[0], flux).shape
    direct = xp.empty((K + 1,) + shape, dtype=np.float64)
    direct[0] = flux
    for k in range(K):
        direct[k + 1] = direct[k] * t_beam[k]
    s_up = r_dir * direct[:-1]
    s_dn = t_dir * direct[:-1]
    up, down = adding(refl, trans, s_up, s_dn, surface_albedo,
                      surface_albedo * direct[K], 0.0)
    return up, down + direct, direct
