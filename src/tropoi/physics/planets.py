"""Bulk parameters of idealized planetary configurations.

A :class:`PlanetaryBody` collects what the dynamical core and the column
physics need to be told about a planet: radius, gravity, rotation,
composition (hence ``R`` and ``c_p``), where the model's lower boundary is
and whether it is a surface, the stellar flux and orbit, interior heat, and
a vertical grid suited to the regime. They are round reference values (NASA
planetary fact sheets, rounded) for *configuring idealized experiments*.
Selecting ``"mars"`` sets Mars-like bulk parameters; it does not make the
model a validated representation of Mars (see docs/radiation/REGIMES.md).

Why each body carries its own vertical grid: the sigma interfaces must
resolve the layers where radiation is absorbed and emitted. A grid that is
uniform in sigma has its highest full level at ``p_bottom / (2 K)`` and
spends half its levels in the lowest scale height: adequate for a 3-bar
giant-planet weather layer with gray optics, useless for a 100-bar column
(top level at 1.7 bar for K = 30) and for any stratosphere. The grids here
are uniform in ``ln sigma`` aloft and uniform in ``sigma`` near the bottom
(:meth:`tropoi.spatial.sigma_coordinate.SigmaGrid.stretched`); their
resolution is measured in docs/radiation/VALIDATION.md.

Import-light: NumPy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from tropoi.physics.insolation import Orbit
from tropoi.physics.thermodynamics import MIXTURES, GasMixture

__all__ = ["VerticalGridSpec", "PlanetaryBody", "PLANETS", "DAY_SECONDS"]

DAY_SECONDS = 86400.0
_YEAR = 365.25 * DAY_SECONDS


@dataclass(frozen=True)
class VerticalGridSpec:
    """A stretched sigma grid: ``nlev`` layers, first interface below the
    top at ``sigma_top``, uniform in ``ln sigma`` above ``linear_scale`` and
    tending to uniform in sigma below it (None = logarithmic throughout)."""

    nlev: int
    sigma_top: float
    linear_scale: Optional[float] = None

    def as_dict(self) -> dict:
        return {"nlev": self.nlev, "sigma_top": self.sigma_top,
                "linear_scale": self.linear_scale}


@dataclass(frozen=True)
class PlanetaryBody:
    name: str
    radius: float                    # m
    gravity: float                   # m s^-2 at the reference level
    rotation_rate: float             # rad s^-1 (negative = retrograde)
    mixture: GasMixture
    reference_pressure: float        # Pa: mean pressure of the model's lower boundary
    has_surface: bool
    orbit: Orbit
    reference_temperature: float     # K: where the constant c_p is evaluated
    temperature_range: tuple[float, float]   # K: span the configuration is meant for
    internal_heat_flux: float = 0.0  # W m^-2
    bond_albedo: float = 0.3
    vertical_grid: VerticalGridSpec = VerticalGridSpec(30, 1.0e-3, 0.2)
    notes: str = ""

    @property
    def gas_constant(self) -> float:
        return self.mixture.gas_constant

    @property
    def cp(self) -> float:
        """Constant c_p used by the dynamical core: the mixture's c_p at the
        reference temperature."""
        return float(self.mixture.cp(self.reference_temperature))

    @property
    def kappa(self) -> float:
        return self.gas_constant / self.cp

    @property
    def scale_height(self) -> float:
        return self.gas_constant * self.reference_temperature / self.gravity

    def with_(self, **changes) -> "PlanetaryBody":
        return replace(self, **changes)

    def regime(self) -> dict:
        """Numbers that decide whether the hydrostatic, shallow, ideal-gas,
        constant-c_p primitive equations apply (docs/radiation/REGIMES.md)."""
        t_lo, t_hi = self.temperature_range
        spread = self.mixture.cp_range(t_lo, t_hi)
        h = self.scale_height
        top = self.vertical_grid.sigma_top
        depth = h * math.log(1.0 / (0.5 * top))      # to the top full level
        return {
            "name": self.name, "R": self.gas_constant, "cp": self.cp, "kappa": self.kappa,
            "scale_height_m": h,
            "model_depth_m": depth,
            "depth_over_radius": depth / self.radius,
            "gravity_change_over_depth": 1.0 - (self.radius / (self.radius + depth)) ** 2,
            "cp_relative_spread": spread["relative_spread"],
            "cp_min": spread["cp_min"], "cp_max": spread["cp_max"],
            "lamb_speed": math.sqrt(self.cp / (self.cp - self.gas_constant)
                                    * self.gas_constant * self.reference_temperature),
            "rotational_speed_equator": abs(self.rotation_rate) * self.radius,
            "top_full_level_pressure": 0.5 * top * self.reference_pressure,
        }


def _body(**kw) -> PlanetaryBody:
    return PlanetaryBody(**kw)


PLANETS: dict[str, PlanetaryBody] = {b.name: b for b in (
    _body(name="earth", radius=6.371e6, gravity=9.80665, rotation_rate=7.292115e-5,
          mixture=MIXTURES["earth_dry_air"], reference_pressure=1.0e5, has_surface=True,
          orbit=Orbit(1361.0, 365.256 * DAY_SECONDS, 0.0167, 23.44, 282.9),
          reference_temperature=260.0, temperature_range=(180.0, 320.0), bond_albedo=0.30,
          vertical_grid=VerticalGridSpec(30, 1.0e-3, 0.15),
          notes="dry air; no water vapour, ozone or clouds unless an experiment adds them"),
    _body(name="mars", radius=3.3895e6, gravity=3.71, rotation_rate=7.0882e-5,
          mixture=MIXTURES["mars"], reference_pressure=610.0, has_surface=True,
          orbit=Orbit(586.2, 686.98 * DAY_SECONDS, 0.0934, 25.19, 251.0),
          reference_temperature=200.0, temperature_range=(130.0, 290.0), bond_albedo=0.25,
          vertical_grid=VerticalGridSpec(30, 2.0e-4, 0.15),
          notes="no CO2 condensation, dust or seasonal mass cycle; top full level ~0.06 Pa"),
    _body(name="venus", radius=6.0518e6, gravity=8.87, rotation_rate=-2.9924e-7,
          mixture=MIXTURES["venus"], reference_pressure=9.2e6, has_surface=True,
          orbit=Orbit(2601.3, 224.701 * DAY_SECONDS, 0.0068, 177.36, 131.0),
          reference_temperature=500.0, temperature_range=(170.0, 740.0), bond_albedo=0.76,
          vertical_grid=VerticalGridSpec(50, 1.0e-6, 0.1),
          notes="no clouds (the real albedo and heating are cloud-controlled); c_p varies "
                "by a factor ~1.6 across the column"),
    _body(name="titan", radius=2.5747e6, gravity=1.352, rotation_rate=4.5607e-6,
          mixture=MIXTURES["titan"], reference_pressure=1.467e5, has_surface=True,
          orbit=Orbit(14.82, 29.457 * _YEAR, 0.0565, 26.7, 277.7),
          reference_temperature=90.0, temperature_range=(70.0, 180.0), bond_albedo=0.27,
          vertical_grid=VerticalGridSpec(40, 1.0e-5, 0.15),
          notes="no haze or methane cycle; the extended atmosphere strains the "
                "shallow-atmosphere approximation"),
    _body(name="jupiter", radius=6.9911e7, gravity=24.79, rotation_rate=1.7585e-4,
          mixture=MIXTURES["jupiter"], reference_pressure=1.0e7, has_surface=False,
          orbit=Orbit(50.26, 11.862 * _YEAR, 0.0489, 3.13, 57.0),
          reference_temperature=165.0, temperature_range=(105.0, 700.0),
          internal_heat_flux=5.7, bond_albedo=0.343,
          vertical_grid=VerticalGridSpec(48, 1.0e-4, None),
          notes="100-bar lower boundary, no surface; no clouds or hazes; hydrogen "
                "ortho/para state is a modelling choice"),
    _body(name="saturn", radius=5.8232e7, gravity=10.44, rotation_rate=1.6378e-4,
          mixture=MIXTURES["saturn"], reference_pressure=1.0e7, has_surface=False,
          orbit=Orbit(14.82, 29.457 * _YEAR, 0.0565, 26.73, 277.7),
          reference_temperature=135.0, temperature_range=(80.0, 700.0),
          internal_heat_flux=2.0, bond_albedo=0.34,
          vertical_grid=VerticalGridSpec(48, 1.0e-4, None),
          notes="as jupiter; rings are not represented"),
)}
