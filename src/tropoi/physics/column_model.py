"""Column physics shared by every host: radiation, lower boundary, interior
heat, sensible exchange and dry convection for a batch of columns.

:class:`ColumnPhysics` is deliberately independent of any dynamical core. A
host passes pressures and temperatures of its columns and receives a
temperature tendency plus the fluxes needed for an energy budget; the
three-dimensional coupling (``tropoi.temporal.column_coupling``) and the
single-column model below are two hosts of the same object, so what is
validated in a column is the code that runs in the general circulation
model.

Energy accounting, per unit area of each column (all in W m^-2):

    d/dt [atmospheric enthalpy] = toa_net_down - bottom_net_down
                                  + sensible + interior
    C dT_s/dt                   = bottom_net_down - sensible      (surface)

``bottom_net_down`` is identically zero for a closed (surface-less) bottom,
where the interior flux is deposited in the lowest layer (Schneider & Liu
2009). The heating returned is exactly the divergence of these fluxes, so
the column-integrated heating equals the boundary fluxes to round-off
(tested); convective adjustment redistributes enthalpy without changing it.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from tropoi.physics.convection import DryConvectiveAdjustment
from tropoi.physics.radiation.scheme import (STEFAN_BOLTZMANN, Column, LowerBoundary,
                                             Radiation, RadiativeFluxes, heating_rate)
from tropoi.physics.surface import SlabSurface, sensible_heat_flux

__all__ = ["PhysicsOutput", "ColumnPhysics", "RadiativeConvectiveColumn"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


@dataclass
class PhysicsOutput:
    """One evaluation of the column physics."""

    heating: Any                 # (K, ncol) K/s, radiation + interior + sensible
    fluxes: RadiativeFluxes
    sensible: Any                # (ncol,) W m^-2 upward from the surface
    interior: float              # W m^-2 deposited in the lowest layer
    surface_net: Any             # (ncol,) W m^-2 into the surface (0 if closed)

    @property
    def column_energy_tendency(self):
        """d/dt of the atmospheric column enthalpy, W m^-2, from the fluxes."""
        f = self.fluxes
        return f.toa_net_down - f.bottom_net_down + self.sensible + self.interior


class ColumnPhysics:
    """Radiation + lower boundary + optional dry convection.

    ``surface=None`` is a closed bottom (no surface). ``radiation_interval``
    recomputes the radiative fluxes every that many calls and holds the
    interface fluxes in between (the heating is re-derived from them on
    every call). The held fluxes are not part of a checkpoint: a host that
    resumes must do so on a call where they are recomputed. ``sensible`` is None or a dict of :func:`~tropoi.physics.surface.
    sensible_heat_flux` keyword options (bulk exchange is then active).
    """

    def __init__(self, radiation: Radiation, *, gravity: float, cp: float,
                 gas_constant: float, insolation=None,
                 convection: Optional[DryConvectiveAdjustment] = None,
                 surface: Optional[SlabSurface] = None,
                 internal_heat_flux: float = 0.0, radiation_interval: int = 1,
                 sensible: Optional[dict] = None):
        if radiation_interval < 1:
            raise ValueError("radiation_interval must be >= 1")
        if sensible is not None and surface is None:
            raise ValueError("sensible exchange needs a surface")
        self.radiation = radiation
        self.gravity = float(gravity)
        self.cp = float(cp)
        self.gas_constant = float(gas_constant)
        self.insolation = insolation
        self.convection = convection
        self.surface = surface
        self.internal_heat_flux = float(internal_heat_flux)
        self.radiation_interval = int(radiation_interval)
        self.sensible = None if sensible is None else dict(sensible)
        # opt-in backward-Euler exchange (see compute); not a flux option
        self._implicit_exchange = bool((self.sensible or {}).get("implicit", False))
        self._flux_options = {k: v for k, v in (self.sensible or {}).items() if k != "implicit"}
        self._cached: Optional[RadiativeFluxes] = None
        self._calls = 0

    def signature(self) -> dict:
        return {"kind": "column_physics", "radiation": self.radiation.signature(),
                "gravity": self.gravity, "cp": self.cp, "gas_constant": self.gas_constant,
                "insolation": None if self.insolation is None else self.insolation.signature(),
                "convection": None if self.convection is None else self.convection.signature(),
                "surface": None if self.surface is None else self.surface.signature(),
                "internal_heat_flux": self.internal_heat_flux,
                "radiation_interval": self.radiation_interval, "sensible": self.sensible}

    def reset(self) -> None:
        """Forget the held radiative fluxes (after a restart or a state jump)."""
        self._cached = None
        self._calls = 0

    # ------------------------------------------------------------------
    def compute(self, pressure_interfaces, temperature, t: float, *, latitude=None,
                longitude=None, surface_temperature=None, wind_speed=None,
                exchange_window=None, surface_step=None) -> PhysicsOutput:
        """``exchange_window`` (s) is the interval over which the host applies
        the heating to the air, ``surface_step`` the one over which it
        advances the slab. With ``"implicit": true`` in the sensible options
        and both given, the bulk exchange is the backward-Euler flux between
        the two reservoirs with the exchange coefficient k = rho C |v| c_p
        frozen: F' = F / (1 + k w / Ca + k s / Cs), Ca = c_p dp / g. The same
        F' enters the air and leaves the slab. Without them, the flux is the
        explicit one, bit for bit."""
        xp = _xp(temperature)
        column = Column(pressure_interfaces, temperature, self.gravity, self.cp,
                        latitude=latitude)
        if self._cached is None or self._calls % self.radiation_interval == 0:
            if self.surface is None:
                boundary = LowerBoundary("closed")
            else:
                if surface_temperature is None:
                    raise ValueError("a surface needs surface_temperature")
                boundary = LowerBoundary("surface", temperature=surface_temperature,
                                         emissivity=self.surface.emissivity,
                                         albedo=self.surface.albedo)
            toa = mu0 = None
            if self.insolation is not None:
                toa, mu0 = self.insolation(latitude, longitude, t)
            self._cached = self.radiation.fluxes(column, boundary, toa_flux=toa, mu0=mu0)
        self._calls += 1
        fluxes = self._cached
        # The interface fluxes are what is held between radiation calls; the
        # heating is their divergence over the *current* layer masses, so the
        # column integral equals the reported boundary fluxes whatever the
        # surface pressure has done since they were computed.
        heating = heating_rate(fluxes.net_down, pressure_interfaces, self.gravity, self.cp)
        dp_low = pressure_interfaces[-1] - pressure_interfaces[-2]
        per_flux = self.gravity / (self.cp * dp_low)        # K/s per W m^-2
        zero = xp.zeros_like(temperature[0])
        sensible = zero
        if self.sensible is not None:
            t_air = temperature[-1]
            p_top = pressure_interfaces[-2]
            p_bot = pressure_interfaces[-1]
            # Simmons-Burridge height of the lowest full level above the surface
            alpha = 1.0 - (p_top / dp_low) * xp.log(p_bot / p_top)
            height = alpha * self.gas_constant * t_air / self.gravity
            density = 0.5 * (p_top + p_bot) / (self.gas_constant * t_air)
            speed = zero if wind_speed is None else wind_speed
            sensible, coefficient = sensible_heat_flux(surface_temperature, t_air, height, speed,
                                                       density, self.cp, self.gravity,
                                                       **self._flux_options)
            if self._implicit_exchange and exchange_window is not None:
                gust = float(self._flux_options.get("gustiness", 0.0))
                k = density * coefficient * xp.sqrt(speed * speed + gust * gust) * self.cp
                ca = self.cp * dp_low / self.gravity
                cs = self.surface.heat_capacity
                step = 0.0 if surface_step is None else float(surface_step)
                sensible = sensible / (1.0 + k * (float(exchange_window) / ca + step / cs))
            heating[-1] += per_flux * sensible
        interior = 0.0
        if self.internal_heat_flux:
            interior = self.internal_heat_flux
            heating[-1] += per_flux * interior
        surface_net = zero if self.surface is None else fluxes.bottom_net_down - sensible
        return PhysicsOutput(heating=heating, fluxes=fluxes, sensible=sensible,
                             interior=interior, surface_net=surface_net)

    def adjust(self, pressure_interfaces, temperature, dt: float):
        """Dry convective adjustment over ``dt`` (identity when disabled)."""
        if self.convection is None:
            return temperature
        dp = pressure_interfaces[1:] - pressure_interfaces[:-1]
        p_mid = 0.5 * (pressure_interfaces[1:] + pressure_interfaces[:-1])
        return self.convection.adjust(temperature, p_mid, dp, dt)


class RadiativeConvectiveColumn:
    """A single-column (or many independent columns) model on a fixed
    pressure grid: the reference host for the column physics.

    ``pressure_interfaces`` (K+1, ncol) stay fixed. :meth:`step` advances in
    physical time; :meth:`equilibrate` drives the columns to their steady
    state with a pseudo-time step that gives every layer the same temperature
    change per unit flux divergence (a layer's physical response time grows
    with its mass, so physical time stepping needs ~decades of model time for
    a deep atmosphere; the steady state is the same).
    """

    def __init__(self, physics: ColumnPhysics, pressure_interfaces, temperature, *,
                 latitude=None, longitude=None, surface_temperature=None, wind_speed=None):
        self.physics = physics
        self.pressure_interfaces = pressure_interfaces
        self.temperature = temperature.copy()
        self.latitude = latitude
        self.longitude = longitude
        self.surface_temperature = (None if surface_temperature is None
                                    else surface_temperature.copy())
        self.wind_speed = wind_speed
        self.time = 0.0

    def evaluate(self) -> PhysicsOutput:
        self.physics.reset()
        return self.physics.compute(self.pressure_interfaces, self.temperature, self.time,
                                    latitude=self.latitude, longitude=self.longitude,
                                    surface_temperature=self.surface_temperature,
                                    wind_speed=self.wind_speed)

    def step(self, dt: float) -> PhysicsOutput:
        """Forward step of length ``dt`` s: physics tendency, surface update,
        then convective adjustment."""
        out = self.physics.compute(self.pressure_interfaces, self.temperature, self.time,
                                   latitude=self.latitude, longitude=self.longitude,
                                   surface_temperature=self.surface_temperature,
                                   wind_speed=self.wind_speed)
        self.temperature = self.temperature + dt * out.heating
        if self.physics.surface is not None:
            self.surface_temperature = self.physics.surface.advance(
                self.surface_temperature, out.surface_net, dt)
        self.temperature = self.physics.adjust(self.pressure_interfaces, self.temperature, dt)
        self.time += dt
        return out

    def _stable_step(self, factor: float = 0.4) -> float:
        """An explicit time step (s) that keeps every layer's own radiative
        relaxation stable: ``factor / max |d heating_k / d T_k|``, with the
        diagonal sensitivity measured by warming the even and the odd layers
        by 1 K in turn (two extra evaluations)."""
        xp = _xp(self.temperature)
        base = self.evaluate().heating
        saved = self.temperature
        rate = 0.0
        for parity in (0, 1):
            bump = xp.zeros_like(saved)
            bump[parity::2] = 1.0
            self.temperature = saved + bump
            diff = self.evaluate().heating - base
            rate = max(rate, float(xp.abs(diff[parity::2]).max()))
        self.temperature = saved
        if self.physics.surface is not None:
            c = self.physics.surface.heat_capacity
            rate = max(rate, 4.0 * STEFAN_BOLTZMANN
                       * float(self.surface_temperature.max()) ** 3 / c)
        return factor / max(rate, 1e-300)

    def equilibrate(self, *, tolerance: float = 1e-3, max_iterations: int = 400000,
                    max_change: float = 0.5, refresh: int = 500) -> dict:
        """March in physical time to steady state.

        The step is the largest that is radiatively stable for every layer
        (re-measured every ``refresh`` steps) and changes no temperature by
        more than ``max_change`` K. Physical time is used on purpose: any
        scheme that steps layers at different rates changes the enthalpy
        balance of a convecting block and therefore its steady state.
        Converged when no layer (after adjustment) and no surface changes
        its heat content faster than ``tolerance`` W m^-2.
        """
        xp = _xp(self.temperature)
        phys = self.physics
        dp = self.pressure_interfaces[1:] - self.pressure_interfaces[:-1]
        mass_cp = phys.cp * dp / phys.gravity               # J K^-1 m^-2 per layer
        hard = None
        if phys.convection is not None:
            hard = DryConvectiveAdjustment(phys.convection.kappa, timescale=0.0,
                                           lapse_fraction=phys.convection.lapse_fraction)
        p_mid = 0.5 * (self.pressure_interfaces[1:] + self.pressure_interfaces[:-1])
        imbalance = float("inf")
        dt_stable = self._stable_step()
        for it in range(1, max_iterations + 1):
            if it % refresh == 0:
                dt_stable = self._stable_step()
            out = self.evaluate()
            peak = float(xp.abs(out.heating).max())
            dt = min(dt_stable, max_change / max(peak, 1e-300))
            new = self.temperature + dt * out.heating
            if hard is not None:
                new = hard.adjust(new, p_mid, dp, 0.0)
            resid = float(xp.abs((new - self.temperature) * mass_cp).max()) / dt
            self.temperature = new
            if phys.surface is not None:
                resid = max(resid, float(xp.abs(out.surface_net).max()))
                self.surface_temperature = phys.surface.advance(
                    self.surface_temperature, out.surface_net, dt)
            self.time += dt
            imbalance = resid
            if imbalance < tolerance:
                return {"iterations": it, "imbalance": imbalance, "converged": True,
                        "model_time": self.time}
        return {"iterations": max_iterations, "imbalance": imbalance, "converged": False,
                "model_time": self.time}
