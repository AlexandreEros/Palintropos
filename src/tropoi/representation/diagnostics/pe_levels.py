"""Physical fields of a saved PE state on native sigma levels (CPU).

Given one packed PE frame (rows ``[zeta_1..K, delta_1..K, T_1..K, ln p_s]``,
top to bottom) and the run's :class:`~tropoi.representation.archive.
pe_context.PEColumnContext`, :func:`reconstruct_levels` evaluates on a
regular latitude-longitude display mesh, for selected FULL levels k:

* ``temperature``          T_k (K), the stored full-level temperature;
* ``pressure``             p_k = sigma_k p_s (Pa): the model's pure-sigma
                           convention (top interface sigma = 0), with the
                           local surface pressure p_s = exp(ln p_s);
* ``geopotential``         Phi_k (m^2 s^-2), the Simmons–Burridge full-level
                           geopotential of
                           :func:`~tropoi.spatial.sigma_coordinate.
                           hydrostatic_geopotential` -- the function the core
                           calls -- applied to the stored temperature
                           coefficients with the boundary condition
                           Phi(sigma = 1) = Phi_s = g h_s, exactly as the
                           tendency does (it is linear, so spectral and grid
                           evaluation agree to round-off; a test checks it).
                           Phi_k is therefore ABSOLUTE: it includes Phi_s once
                           and is zero on the reference sphere h = 0;
* ``geopotential_height``  Z_k = Phi_k / g with the model's g (m), the height
                           above the planet's reference sphere in the model's
                           constant-gravity, spherical-geopotential
                           convention (the same g makes h_s = Phi_s / g);
* ``zeta`` / ``delta``     relative vorticity and divergence (s^-1);
* ``u`` / ``v``            eastward / northward wind (m/s), the cores'
                           Helmholtz reconstruction
                           (:func:`~tropoi.spatial.transforms.host_synthesis.
                           synthesize_wind_latlon`).

plus the level-free ``surface_pressure``, ``surface_geopotential`` and
``surface_height`` (= h_s). All values come from exact host synthesis of the
saved coefficients (no CUDA, no model, no interpolation in the vertical).
Fields, pressure, geometry and winds all refer to the same full level.

NumPy only; never imports CuPy.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tropoi.spatial.sigma_coordinate import hydrostatic_geopotential
from tropoi.spatial.transforms.host_synthesis import (synthesize_latlon,
                                                      synthesize_wind_latlon)

__all__ = [
    "LEVEL_FIELDS",
    "LevelFields",
    "level_geopotential_coefficients",
    "reconstruct_levels",
]

#: Per-level outputs of :func:`reconstruct_levels`.
LEVEL_FIELDS = ("temperature", "pressure", "geopotential",
                "geopotential_height", "zeta", "delta", "u", "v")


@dataclass(frozen=True, eq=False)
class LevelFields:
    """Reconstructed fields; per-level arrays are ``(len(levels), nlat, nlon)``."""

    levels: tuple[int, ...]
    sigma: tuple[float, ...]
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    values: dict
    surface_pressure: np.ndarray
    surface_geopotential: np.ndarray
    surface_height: np.ndarray | None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]


def _frame_parts(frame, nlev: int, l_max: int):
    x = np.asarray(frame)
    expected = (3 * nlev + 1, l_max + 1, l_max + 1)
    if x.shape != expected:
        raise ValueError(f"PE frame has shape {x.shape}; expected {expected}")
    return (x[0:nlev], x[nlev:2 * nlev], x[2 * nlev:3 * nlev], x[3 * nlev])


def level_geopotential_coefficients(frame, context) -> np.ndarray:
    """Full-level geopotential coefficients ``(nlev, L+1, L+1)`` (absolute).

    ``hydrostatic_geopotential(sigma, T_lm, Phi_s_lm, R_d)``: the same call,
    with the same arguments, as ``PrimitiveEquationsModel._momentum_tendencies``.
    """
    _, _, temperature, _ = _frame_parts(frame, context.nlev, context.l_max)
    phi_full, _ = hydrostatic_geopotential(
        context.sigma, np.asarray(temperature, dtype=np.complex128),
        context.surface_geopotential_lm(), context.r_dry)
    return phi_full


def _levels(levels, nlev: int) -> tuple[int, ...]:
    if levels is None:
        return tuple(range(nlev))
    out = []
    for k in levels:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError(f"level indices must be integers, got {k!r}")
        k = int(k)
        if not 0 <= k < nlev:
            raise IndexError(f"level index {k} is out of range for {nlev} "
                             "native levels (0 = top)")
        out.append(k)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated level indices {list(levels)}")
    return tuple(out)


def reconstruct_levels(frame, context, lat_deg, lon_deg,
                       levels: Sequence[int] | None = None, *,
                       winds: bool = True) -> LevelFields:
    """Evaluate the native-level fields of one saved state (see module doc).

    ``levels`` are 0-based full-level indices, top to bottom (default:
    all). ``winds=False`` skips ``u``/``v``. Geopotential height needs the
    model's gravity: a context without one (a flat run that never defined
    g) raises.
    """
    if context.gravity is None:
        raise ValueError(
            "this run defines no gravity, so geopotential height Phi/g is "
            "undefined; supply the gravity the run should be read with "
            "(e.g. gravity=9.80665 for geopotential metres)")
    nlev, l_max = context.nlev, context.l_max
    chosen = _levels(levels, nlev)
    zeta, delta, _, ln_ps = _frame_parts(frame, nlev, l_max)
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    g = float(context.gravity)
    full = context.full_levels
    phi_lm = level_geopotential_coefficients(frame, context)
    _, _, temperature, _ = _frame_parts(frame, nlev, l_max)

    p_s = np.exp(synthesize_latlon(ln_ps, lat, lon))
    phi_s = synthesize_latlon(context.surface_geopotential_lm(), lat, lon)
    h_s = (None if context.is_flat
           else synthesize_latlon(context.surface_elevation_lm, lat, lon))
    names = LEVEL_FIELDS if winds else LEVEL_FIELDS[:6]
    out = {name: np.empty((len(chosen), lat.size, lon.size)) for name in names}
    for i, k in enumerate(chosen):
        out["temperature"][i] = synthesize_latlon(temperature[k], lat, lon)
        out["pressure"][i] = full[k] * p_s
        phi = synthesize_latlon(phi_lm[k], lat, lon)
        out["geopotential"][i] = phi
        out["geopotential_height"][i] = phi / g
        out["zeta"][i] = synthesize_latlon(zeta[k], lat, lon)
        out["delta"][i] = synthesize_latlon(delta[k], lat, lon)
        if winds:
            out["u"][i], out["v"][i] = synthesize_wind_latlon(
                zeta[k], delta[k], context.radius_m, lat, lon)
    for array in out.values():
        if not np.isfinite(array).all():
            raise ValueError("reconstructed fields are not finite")
    return LevelFields(levels=chosen, sigma=tuple(full[k] for k in chosen),
                       lat_deg=lat, lon_deg=lon, values=out,
                       surface_pressure=p_s, surface_geopotential=phi_s,
                       surface_height=h_s)
