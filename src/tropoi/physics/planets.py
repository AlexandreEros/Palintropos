"""Bulk parameters of idealized planetary configurations.

A :class:`PlanetaryBody` collects what the dynamical core and the column
physics need to be told about a planet: radius, gravity, a
:class:`~tropoi.physics.rotation.Rotation` (spin rate and axis), a
:class:`~tropoi.physics.composition.Composition` (hence ``R`` and ``c_p``),
where the model's lower boundary is and whether it is a surface, the
illuminating :class:`~tropoi.physics.star.Star` and
:class:`~tropoi.physics.orbit.KeplerOrbit`, interior heat, and a vertical
grid suited to the regime. They are round reference values (NASA planetary
fact sheets, rounded) for *configuring idealized experiments*. Selecting
``"mars"`` sets Mars-like bulk parameters; it does not make the model a
validated representation of Mars (see docs/radiation/REGIMES.md).

Orbit frame convention (see :mod:`tropoi.physics.orbit`): each body's orbit
is written in its own orbital plane with the reference direction at its
northern spring equinox, so the orbit's longitude of periapsis *is* the
solar longitude L_s of perihelion and the rotation's equinox longitude is 0.

The body's ``illumination`` keeps the historically stated scalars exactly —
the fact-sheet stellar constant and period — with the star, orbit and
rotation attached as the shared objects they reduce from (the IAU nominal
Sun's luminosity and the fact-sheet distances reproduce the stated
constants to ~0.1 %, the accuracy of these round values).

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

from tropoi.physics.composition import COMPOSITIONS, Composition
from tropoi.physics.insolation import StellarIllumination
from tropoi.physics.orbit import KeplerOrbit
from tropoi.physics.rotation import Rotation
from tropoi.physics.star import STARS, Star
from tropoi.physics.thermodynamics import heat_capacity, heat_capacity_range

__all__ = ["VerticalGridSpec", "PlanetaryBody", "PLANETS", "DAY_SECONDS"]

DAY_SECONDS = 86400.0
_YEAR = 365.25 * DAY_SECONDS
_SUN = STARS["sun"]


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
    rotation: Rotation
    composition: Composition
    reference_pressure: float        # Pa: mean pressure of the model's lower boundary
    has_surface: bool
    star: Star
    orbit: KeplerOrbit
    illumination: StellarIllumination
    reference_temperature: float     # K: where the constant c_p is evaluated
    temperature_range: tuple[float, float]   # K: span the configuration is meant for
    internal_heat_flux: float = 0.0  # W m^-2
    bond_albedo: float = 0.3
    vertical_grid: VerticalGridSpec = VerticalGridSpec(30, 1.0e-3, 0.2)
    notes: str = ""

    @property
    def rotation_rate(self) -> float:
        """rad s^-1 (negative = retrograde): the rotation's angular velocity."""
        return self.rotation.angular_velocity

    @property
    def gas_constant(self) -> float:
        return self.composition.gas_constant

    @property
    def cp(self) -> float:
        """Constant c_p used by the dynamical core: the composition's c_p at
        the reference temperature."""
        return float(heat_capacity(self.composition, self.reference_temperature))

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
        spread = heat_capacity_range(self.composition, t_lo, t_hi)
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


def _solar_body(*, name: str, radius: float, gravity: float,
                rotation_rate: float, composition: str,
                reference_pressure: float, has_surface: bool,
                stellar_constant: float, period: float, eccentricity: float,
                obliquity_deg: float, perihelion_ls_deg: float,
                semimajor_axis_au: float, **kw) -> PlanetaryBody:
    """A body orbiting the Sun, with the historically stated scalars kept
    exactly: the fact-sheet stellar constant and period drive the
    illumination; the fact-sheet semimajor axis fixes the orbit geometry."""
    rotation = Rotation.of(rotation_rate, obliquity_deg=obliquity_deg)
    orbit = KeplerOrbit.from_au(semimajor_axis_au, eccentricity=eccentricity,
                                periapsis_longitude_rad=math.radians(
                                    perihelion_ls_deg))
    illumination = StellarIllumination(
        float(stellar_constant), float(period), float(eccentricity),
        math.radians(obliquity_deg), math.radians(perihelion_ls_deg),
        star=_SUN, orbit=orbit, rotation=rotation)
    return PlanetaryBody(name=name, radius=radius, gravity=gravity,
                         rotation=rotation,
                         composition=COMPOSITIONS[composition],
                         reference_pressure=reference_pressure,
                         has_surface=has_surface, star=_SUN, orbit=orbit,
                         illumination=illumination, **kw)


#: Semimajor axes (au) are NASA fact-sheet values; Titan takes Saturn's
#: heliocentric orbit (it orbits Saturn).
PLANETS: dict[str, PlanetaryBody] = {b.name: b for b in (
    _solar_body(name="earth", radius=6.371e6, gravity=9.80665, rotation_rate=7.292115e-5,
                composition="earth_dry_air", reference_pressure=1.0e5, has_surface=True,
                stellar_constant=1361.0, period=365.256 * DAY_SECONDS, eccentricity=0.0167,
                obliquity_deg=23.44, perihelion_ls_deg=282.9, semimajor_axis_au=1.000,
                reference_temperature=260.0, temperature_range=(180.0, 320.0), bond_albedo=0.30,
                vertical_grid=VerticalGridSpec(30, 1.0e-3, 0.15),
                notes="dry air; no water vapour, ozone or clouds unless an experiment adds them"),
    _solar_body(name="mars", radius=3.3895e6, gravity=3.71, rotation_rate=7.0882e-5,
                composition="mars", reference_pressure=610.0, has_surface=True,
                stellar_constant=586.2, period=686.98 * DAY_SECONDS, eccentricity=0.0934,
                obliquity_deg=25.19, perihelion_ls_deg=251.0, semimajor_axis_au=1.524,
                reference_temperature=200.0, temperature_range=(130.0, 290.0), bond_albedo=0.25,
                vertical_grid=VerticalGridSpec(30, 2.0e-4, 0.15),
                notes="no CO2 condensation, dust or seasonal mass cycle; top full level ~0.06 Pa"),
    _solar_body(name="venus", radius=6.0518e6, gravity=8.87, rotation_rate=-2.9924e-7,
                composition="venus", reference_pressure=9.2e6, has_surface=True,
                stellar_constant=2601.3, period=224.701 * DAY_SECONDS, eccentricity=0.0068,
                obliquity_deg=177.36, perihelion_ls_deg=131.0, semimajor_axis_au=0.7233,
                reference_temperature=500.0, temperature_range=(170.0, 740.0), bond_albedo=0.76,
                vertical_grid=VerticalGridSpec(50, 1.0e-6, 0.1),
                notes="no clouds (the real albedo and heating are cloud-controlled); c_p varies "
                      "by a factor ~1.6 across the column"),
    _solar_body(name="titan", radius=2.5747e6, gravity=1.352, rotation_rate=4.5607e-6,
                composition="titan", reference_pressure=1.467e5, has_surface=True,
                stellar_constant=14.82, period=29.457 * _YEAR, eccentricity=0.0565,
                obliquity_deg=26.7, perihelion_ls_deg=277.7, semimajor_axis_au=9.583,
                reference_temperature=90.0, temperature_range=(70.0, 180.0), bond_albedo=0.27,
                vertical_grid=VerticalGridSpec(40, 1.0e-5, 0.15),
                notes="no haze or methane cycle; the extended atmosphere strains the "
                      "shallow-atmosphere approximation"),
    _solar_body(name="jupiter", radius=6.9911e7, gravity=24.79, rotation_rate=1.7585e-4,
                composition="jupiter", reference_pressure=1.0e7, has_surface=False,
                stellar_constant=50.26, period=11.862 * _YEAR, eccentricity=0.0489,
                obliquity_deg=3.13, perihelion_ls_deg=57.0, semimajor_axis_au=5.204,
                reference_temperature=165.0, temperature_range=(105.0, 700.0),
                internal_heat_flux=5.7, bond_albedo=0.343,
                vertical_grid=VerticalGridSpec(48, 1.0e-4, None),
                notes="100-bar lower boundary, no surface; no clouds or hazes; hydrogen "
                      "ortho/para state is a modelling choice"),
    _solar_body(name="saturn", radius=5.8232e7, gravity=10.44, rotation_rate=1.6378e-4,
                composition="saturn", reference_pressure=1.0e7, has_surface=False,
                stellar_constant=14.82, period=29.457 * _YEAR, eccentricity=0.0565,
                obliquity_deg=26.73, perihelion_ls_deg=277.7, semimajor_axis_au=9.583,
                reference_temperature=135.0, temperature_range=(80.0, 700.0),
                internal_heat_flux=2.0, bond_albedo=0.34,
                vertical_grid=VerticalGridSpec(48, 1.0e-4, None),
                notes="as jupiter; rings are not represented"),
)}
