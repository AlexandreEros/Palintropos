"""Lower-boundary energy exchange: a slab surface and bulk sensible heat flux.

A terrestrial column needs a surface that absorbs stellar radiation, emits
thermally and hands heat to the lowest atmospheric layer; without the last
step radiation leaves a temperature discontinuity that nothing removes. The
formulation is the dry part of Frierson, Held & Zurita-Gotor (2006): an
energy-conserving slab (their eq. 1 without evaporation) and a bulk sensible
heat flux with the simplified Monin-Obukhov drag coefficient of their eqs.
12-14, written with dry static energy as they prescribe.

Their K-profile boundary-layer diffusion (eqs. 16-20) is NOT implemented;
vertical redistribution above the lowest layer is left to the dry convective
adjustment. This is a stated difference from the published model.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["SlabSurface", "drag_coefficient", "sensible_heat_flux"]

VON_KARMAN = 0.4


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def drag_coefficient(height, roughness_length: float, richardson,
                     critical_richardson: float = 1.0):
    """FHZ06 eqs. 12-14: neutral value ``[k / ln(z / z0)]^2`` for unstable
    and neutral conditions, reduced by ``(1 - Ri / Ri_c)^2`` when stable and
    zero beyond ``Ri_c``."""
    xp = _xp(height, richardson)
    neutral = (VON_KARMAN / xp.log(height / roughness_length)) ** 2
    stable = xp.clip(1.0 - richardson / critical_richardson, 0.0, 1.0) ** 2
    return neutral * xp.where(richardson > 0.0, stable, 1.0)


def sensible_heat_flux(surface_temperature, air_temperature, air_height, wind_speed,
                       air_density, cp: float, gravity: float, *,
                       roughness_length: float = 3.21e-5,
                       critical_richardson: float = 1.0, gustiness: float = 0.0):
    """Upward sensible heat flux (W m^-2) from the surface to the lowest
    layer: ``rho C |v| (c_p T_s - c_p T_a - g z_a)``. ``gustiness`` (m/s) is
    added in quadrature to the wind (0 in FHZ06). Returns ``(flux,
    drag_coefficient)``."""
    xp = _xp(surface_temperature, air_temperature)
    speed = xp.sqrt(wind_speed * wind_speed + gustiness * gustiness)
    dse_air = cp * air_temperature + gravity * air_height
    dse_sfc = cp * surface_temperature
    richardson = (gravity * air_height * (dse_air - dse_sfc) / dse_sfc
                  / xp.maximum(speed * speed, 1e-12))
    c = drag_coefficient(air_height, roughness_length, richardson, critical_richardson)
    return air_density * c * speed * (dse_sfc - dse_air), c


class SlabSurface:
    """A surface layer of areal heat capacity ``heat_capacity`` (J K^-1 m^-2)
    with one temperature: ``C dT_s/dt = net flux into the surface``."""

    def __init__(self, heat_capacity: float, *, albedo: float = 0.0,
                 emissivity: float = 1.0):
        if not (math.isfinite(heat_capacity) and heat_capacity > 0.0):
            raise ValueError("heat_capacity must be finite and > 0")
        if not (0.0 <= albedo <= 1.0 and 0.0 < emissivity <= 1.0):
            raise ValueError("albedo must be in [0, 1] and emissivity in (0, 1]")
        self.heat_capacity = float(heat_capacity)
        self.albedo = float(albedo)
        self.emissivity = float(emissivity)

    def signature(self) -> dict:
        return {"kind": "slab_surface", "heat_capacity": self.heat_capacity,
                "albedo": self.albedo, "emissivity": self.emissivity}

    def advance(self, temperature, net_flux_into_surface, dt: float):
        return temperature + (float(dt) / self.heat_capacity) * net_flux_into_surface
