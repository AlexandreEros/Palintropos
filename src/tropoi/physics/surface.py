"""Lower-boundary energy exchange: a slab surface and bulk sensible heat flux.

A terrestrial column needs a surface that absorbs stellar radiation, emits
thermally and hands heat to the lowest atmospheric layer; without the last
step radiation leaves a temperature discontinuity that nothing removes. The
formulation is the dry part of Frierson, Held & Zurita-Gotor (2006): an
energy-conserving slab (their eq. 1 without evaporation) and a bulk sensible
heat flux with the simplified Monin-Obukhov drag coefficient of their eqs.
12-14, written with dry static energy as they prescribe.

The full Monin-Obukhov surface layer that Isca's grey-Mars case runs is
available as an opt-in alternative: :func:`monin_obukhov_sensible_heat_flux`
over :mod:`tropoi.physics.monin_obukhov`. FHZ06 stays the default and the
arithmetic of :func:`drag_coefficient` and :func:`sensible_heat_flux` is
untouched.

Their K-profile boundary-layer diffusion (eqs. 16-20) is NOT implemented;
vertical redistribution above the lowest layer is left to the dry convective
adjustment. This is a stated difference from the published model.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = ["SlabSurface", "drag_coefficient", "sensible_heat_flux",
           "monin_obukhov_sensible_heat_flux", "SENSIBLE_SCHEMES"]

SENSIBLE_SCHEMES = ("fhz06", "monin_obukhov")

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


def monin_obukhov_sensible_heat_flux(surface_temperature, air_temperature,
                                     air_height, wind_speed, air_density, cp: float,
                                     gas_constant: float, gravity: float, *,
                                     roughness_length: float = 3.21e-5,
                                     roughness_heat: Optional[float] = None,
                                     gustiness: float = 0.0, options=None):
    """Upward sensible heat flux (W m^-2) with Monin-Obukhov exchange, as
    Isca assembles it in ``surface_flux.F90``:

        H = rho c_p C_t |v| (T_s - theta_a),
        theta_a = T_a (p_s / p_a)^kappa,   kappa = R / c_p.

    The pressure ratio is formed from the level's own height rather than
    from a separately defined full-level pressure:
    ``p_s / p_a = exp(g z_a / (R T_a))``. For the hydrostatic layer the
    coordinate already assumes, that is the same number Isca gets from
    ``p_surf / p_full``, and taking it this way keeps ``theta_a`` tied to
    the very height ``z_a`` that the Richardson number and the profile
    functions use - one definition of where the lowest level is, not two.

    Two things differ from :func:`sensible_heat_flux`, beyond the exchange
    coefficient itself.

    *The thermodynamic difference is a potential-temperature difference,
    not a dry-static-energy one.* FHZ06 drives the flux with
    ``c_p T_s - (c_p T_a + g z_a)``; Isca drives it with
    ``c_p (T_s - T_a (p_s/p_a)^kappa)``. The two agree to first order,
    since ``(p_s/p_a)^kappa - 1 ~ g z_a / (c_p T_a)`` for a shallow
    hydrostatic layer, and they differ at second order in
    ``g z_a / (c_p T_a)``. On the Mars configuration the lowest level sits
    near ``z_a ~ 10`` m with ``c_p T_a ~ 1.7e5`` J/kg, so the discrepancy
    is parts in 1e9 of the difference itself - far below the difference the
    exchange coefficient makes.

    *The momentum and heat coefficients are no longer the same number.*
    The return value is ``(flux, exchange)`` where ``exchange`` is the
    whole :class:`~tropoi.physics.monin_obukhov.SurfaceExchange`, so a
    caller doing the momentum stress uses ``exchange.drag_m`` while the
    heat flux above used ``exchange.drag_t``. The FHZ06 path returns one
    coefficient used for both.

    ``roughness_heat`` defaults to ``roughness_length``, which is what
    grey-Mars sets (``roughness_mom = roughness_heat = 3.21e-5`` m).
    """
    from tropoi.physics.monin_obukhov import surface_drag
    xp = _xp(surface_temperature, air_temperature)
    speed = xp.sqrt(wind_speed * wind_speed + gustiness * gustiness)
    kappa = gas_constant / cp
    theta_air = air_temperature * xp.exp(
        kappa * gravity * air_height / (gas_constant * air_temperature))
    exchange = surface_drag(theta_air, surface_temperature, air_height, speed,
                            gravity=gravity, roughness_momentum=roughness_length,
                            roughness_heat=(roughness_length if roughness_heat is None
                                            else roughness_heat),
                            opts=options)
    flux = (air_density * cp * exchange.drag_t * speed
            * (surface_temperature - theta_air))
    return flux, exchange


class SlabSurface:
    """A surface layer of areal heat capacity ``heat_capacity`` (J K^-1 m^-2)
    with one temperature: ``C dT_s/dt = net flux into the surface``."""

    def __init__(self, heat_capacity: float, *, albedo: float, emissivity: float):
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
