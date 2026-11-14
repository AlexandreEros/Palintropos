"""Configuration of a radiative–dynamical experiment (CPU-only, hashable).

Every quantity that determines the result is a field of
:class:`RadiativeConfig`; its canonical JSON is hashed into
``config_sha256``, which every checkpoint carries and every resume
compares. Physics components are plain ``{"kind": ..., ...}`` dictionaries
resolved by :mod:`tropoi.run.radiative.model`, so a configuration file is
readable without importing any solver. Nothing here imports CuPy.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

__all__ = ["RadiativeConfig", "PRESETS", "CONFIG_SCHEMA", "DAY_SECONDS", "mars_gray_flat_dev",
           "jupiter_schneider_liu", "earth_gray_dry"]

CONFIG_SCHEMA = "palintropos.radiative.config/1"
DAY_SECONDS = 86400.0


@dataclass(frozen=True)
class RadiativeConfig:
    """One experiment. SI units; times in seconds unless named *_days
    (a "day" is always 86400 s, whatever the planet's rotation)."""

    experiment_id: str
    # planet and gas
    radius: float
    omega: float
    gravity: float
    gas_constant: float
    cp: float
    reference_pressure: float
    # horizontal discretization (Gauss lat-lon)
    l_max: int
    retained_truncation: Optional[int]
    nlat: int
    nlon: int
    # vertical grid: {"kind": "uniform"|"stretched"|"frierson", "nlev": ...}
    vertical: dict
    # time integration
    dt: float
    days: int
    t_ref: float = 300.0
    raw_nu: float = 0.1
    raw_alpha: float = 0.53
    damping_scheme: str = "centred"
    hyperdiffusion_order: int = 4
    hyperdiffusion_efold_days: float = 0.1
    # physics (see model.py for the kinds)
    longwave: dict = field(default_factory=dict)
    shortwave: Optional[dict] = None
    insolation: Optional[dict] = None
    convection: Optional[dict] = None
    surface: Optional[dict] = None
    sensible: Optional[dict] = None
    drag: Optional[dict] = None
    # top Rayleigh sponge {"bottom_pressure": Pa, "efold_days": d} (Isca
    # damping_driver); left out of the canonical form when unset
    sponge: Optional[dict] = None
    # dry boundary layer (MARS_MI_PLAN.md 2e): bulk surface stress plus
    # implicit K-profile diffusion of momentum and dry static energy,
    # {"roughness_length": m, "critical_richardson": ..., "rich_crit_pbl":
    # ..., "frac_inner": ..., "gustiness": m/s}; left out of the canonical
    # form when unset so every earlier configuration keeps its hash
    boundary_layer: Optional[dict] = None
    internal_heat_flux: float = 0.0
    # return the kinetic energy removed by drag and hyperdiffusion as heat.
    # The field default stays False: configurations recorded before the
    # option existed omit it and must keep meaning (and hashing as) the
    # unheated experiment. The presets pass True explicitly.
    frictional_heating: bool = False
    radiation_interval: int = 1
    thermal_source: str = "linear"
    # initial state: {"kind": "radiative_convective"} or {"kind": "isothermal",
    # "temperature": T}; a seeded vorticity perturbation is added
    initial: dict = field(default_factory=lambda: {"kind": "radiative_convective"})
    perturbation_wind: float = 0.1          # m/s, grid RMS wind scale of the seed
    perturbation_lmax: int = 8
    seed: int = 20261110
    # output
    checkpoint_every_days: int = 10
    keep_checkpoints: int = 2
    state_every_days: int = 10
    mean_last_days: int = 0                 # 0 = second half of the run

    def __post_init__(self):
        if not self.experiment_id or any(c in self.experiment_id for c in "/\\ "):
            raise ValueError(f"bad experiment_id {self.experiment_id!r}")
        for name in ("radius", "gravity", "gas_constant", "cp", "reference_pressure", "dt"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        if not self.cp > self.gas_constant:
            raise ValueError("cp must exceed the gas constant")
        if self.nlat < self.l_max + 1 or self.nlon < 2 * self.l_max + 1:
            raise ValueError("state grid too coarse for exact analysis at l_max")
        rt = self.retained_truncation
        if rt is not None and not (1 <= int(rt) <= self.l_max - 1):
            raise ValueError("retained_truncation must be in [1, l_max - 1]")
        spd = DAY_SECONDS / self.dt
        if abs(spd - round(spd)) > 1e-9:
            raise ValueError(f"dt = {self.dt} s must divide 86400 s")
        if self.days < 1 or self.radiation_interval < 1:
            raise ValueError("days and radiation_interval must be >= 1")
        if round(spd) % self.radiation_interval:
            # Held radiative fluxes are not checkpointed. Runs stop and resume
            # at whole days, so a day must be a whole number of radiation
            # intervals for a resumed run to recompute them exactly where an
            # uninterrupted one does.
            raise ValueError(f"radiation_interval = {self.radiation_interval} must divide the "
                             f"{round(spd)} steps of a day")
        if self.damping_scheme not in ("lagged", "centred"):
            raise ValueError("damping_scheme must be 'lagged' or 'centred'")
        if "kind" not in self.vertical or "kind" not in self.longwave:
            raise ValueError("vertical and longwave need a 'kind'")
        if self.sensible is not None and self.surface is None:
            raise ValueError("sensible exchange needs a surface")
        if self.boundary_layer is not None and self.surface is None:
            raise ValueError("the boundary layer needs a surface")
        if self.surface is None and self.shortwave is None and self.internal_heat_flux <= 0:
            raise ValueError("a surface-less column needs a stellar or interior heat source")

    @property
    def kappa(self) -> float:
        return self.gas_constant / self.cp

    @property
    def steps_per_day(self) -> int:
        return int(round(DAY_SECONDS / self.dt))

    @property
    def total_steps(self) -> int:
        return self.days * self.steps_per_day

    @property
    def retained_cut(self) -> int:
        from tropoi.spatial.truncation import product_truncation_cut
        if self.retained_truncation is None:
            return product_truncation_cut(self.l_max)
        return int(self.retained_truncation)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["sponge"] is None:
            del d["sponge"]
        if d["boundary_layer"] is None:
            del d["boundary_layer"]
        if not d["frictional_heating"]:
            # written only when set, so configurations that predate the
            # option keep their hash
            del d["frictional_heating"]
        d["schema"] = CONFIG_SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RadiativeConfig":
        d = dict(d)
        schema = d.pop("schema", CONFIG_SCHEMA)
        if schema != CONFIG_SCHEMA:
            raise ValueError(f"unknown config schema {schema!r}")
        return cls(**d)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_(self, **changes) -> "RadiativeConfig":
        return replace(self, **changes)


def _t21() -> dict:
    return {"l_max": 22, "retained_truncation": 21, "nlat": 32, "nlon": 64}


def _preset_id(experiment_id: Optional[str], name: str, resolution: str,
               frictional_heating: bool) -> str:
    """The preset's default experiment id: ``<name>-heated-<resolution>``
    with frictional heating, ``<name>-<resolution>`` without. These are the
    ids of the runs recorded for each formulation (docs/radiation/evidence),
    so each default reproduces a recorded configuration hash."""
    if experiment_id is not None:
        return experiment_id
    return f"{name}-heated-{resolution}" if frictional_heating else f"{name}-{resolution}"


def jupiter_schneider_liu(experiment_id: Optional[str] = None, *, days: int = 200,
                          dt: float = 1200.0, frictional_heating: bool = True) -> RadiativeConfig:
    """The physics of Schneider & Liu (2009), appendix B and Table 1, at T21.

    Matched to the paper: planetary constants, gray thermal optical depth
    80 (p / 3 bar)^2, semi-infinite scattering solar flux (tau 3 p / 3 bar,
    single-scattering albedo 0.8, asymmetry 0.204), insolation (F0 / pi)
    cos(lat) with F0 = 50.7 W m^-2, interior heat 5.7 W m^-2 into the lowest
    layer, dry convective relaxation with 6 h, Rayleigh drag 0.05 / day
    below sigma = 0.8 poleward of 16.3 degrees, 30 uniform sigma levels to
    3 bar. NOT matched: horizontal resolution (T21, not T213), the form of
    the hyperdiffusion, the drag's equatorial shoulder (not specified in the
    paper), and the integration length (theirs: > 28,000 days).

    ``frictional_heating`` (default on) returns what drag and
    hyperdiffusion remove as heat; ``frictional_heating=False`` is the
    configuration of the recorded unheated run (``80f280d2…``).
    """
    return RadiativeConfig(
        experiment_id=_preset_id(experiment_id, "jupiter-sl09", "T21L30", frictional_heating),
        frictional_heating=frictional_heating, radius=69.86e6, omega=1.7587e-4, gravity=26.0,
        gas_constant=3605.38, cp=12619.0, reference_pressure=3.0e5,
        vertical={"kind": "uniform", "nlev": 30}, dt=dt, days=days,
        longwave={"kind": "power_law", "tau0": 80.0, "p0": 3.0e5, "exponent": 2.0},
        shortwave={"kind": "semi_infinite_scattering", "tau0": 3.0, "p0": 3.0e5,
                   "ssa": 0.8, "asym": 0.204},
        insolation={"kind": "diurnal_mean", "stellar_constant": 50.7},
        convection={"timescale": 21600.0}, internal_heat_flux=5.7,
        drag={"kind": "schneider_liu", "k0_per_day": 0.05, "sigma_b": 0.8,
              "cutoff_deg": 16.3, "decay_deg": 2.0},
        **_t21())


def earth_gray_dry(experiment_id: Optional[str] = None, *, days: int = 300,
                   dt: float = 1200.0, frictional_heating: bool = True) -> RadiativeConfig:
    """A dry Earth with the gray radiation, slab surface and bulk sensible
    exchange of Frierson, Held & Zurita-Gotor (2006), Table 1.

    Matched: constants, longwave optical depth (6 at the equator, 1.5 at the
    poles, linear fraction 0.1), net surface solar flux, slab heat capacity
    1e7 J K^-1 m^-2, simplified Monin-Obukhov drag coefficient, their sigma
    levels. NOT matched: their K-profile boundary-layer diffusion (dry
    convective adjustment redistributes heat instead), momentum exchange
    (Held-Suarez Rayleigh drag instead of surface stress), del^4 vs del^8
    diffusion, and resolution (T21, theirs T85/T170). This is therefore not
    the FHZ06 dry limit and is not presented as a reproduction of it.

    ``frictional_heating`` (default on) returns what drag and
    hyperdiffusion remove as heat; without it the eddying run fails the
    energy criteria A2/A3 (docs/radiation/README.md §6).
    ``frictional_heating=False`` is the configuration of the recorded
    unheated run (``e31ae183…``).
    """
    return RadiativeConfig(
        experiment_id=_preset_id(experiment_id, "earth-gray-dry", "T21L25", frictional_heating),
        frictional_heating=frictional_heating, radius=6.376e6, omega=7.292e-5, gravity=9.8,
        gas_constant=287.04, cp=1004.64, reference_pressure=1.0e5,
        vertical={"kind": "frierson", "nlev": 25}, dt=dt, days=days,
        longwave={"kind": "frierson", "tau_equator": 6.0, "tau_pole": 1.5,
                  "linear_fraction": 0.1},
        shortwave={"kind": "transparent"},
        insolation={"kind": "frierson", "net_solar_constant": 938.4, "delta_s": 1.4},
        convection={"timescale": 0.0},
        surface={"heat_capacity": 1.0e7, "albedo": 0.0, "emissivity": 1.0,
                 "initial_temperature": 285.0},
        sensible={"roughness_length": 3.21e-5, "critical_richardson": 1.0, "gustiness": 1.0},
        drag={"kind": "held_suarez", "k_f_per_day": 1.0, "sigma_b": 0.7},
        initial={"kind": "isothermal", "temperature": 264.0},
        **_t21())


# Isca exp/test_cases/grey_mars `bk` (25 levels, commit d1321ac3)
ISCA_GREY_MARS_SIGMA = (
    0.0, 1.53008955e-04, 4.63790800e-04, 1.10977640e-03, 2.37044150e-03, 4.74479200e-03,
    9.12245300e-03, 1.70677050e-02, 3.12516100e-02, 5.59939500e-02, 9.76165000e-02,
    1.63754900e-01, 2.60315150e-01, 3.85974250e-01, 5.28084800e-01, 6.65956600e-01,
    7.81088000e-01, 8.65400050e-01, 9.21109250e-01, 9.55343500e-01, 9.75416950e-01,
    9.86856800e-01, 9.93269300e-01, 9.96830200e-01, 9.98799150e-01, 1.0)
_ISCA_RHO_CP = 1.035e3 * 3989.24495292815      # Isca constants RHO0 * CP_OCEAN


def _time_of_perihelion_for_equinox_start(period: float, eccentricity: float,
                                          ls_perihelion_deg: float) -> float:
    """The ``time_of_perihelion`` that puts t = 0 at L_s = 0 (northern spring
    equinox), from Kepler's equation."""
    nu = math.radians((0.0 - ls_perihelion_deg) % 360.0)        # true anomaly at L_s = 0
    e = eccentricity
    ecc_anomaly = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(nu / 2.0))
    mean = (ecc_anomaly - e * math.sin(ecc_anomaly)) % (2.0 * math.pi)
    return -mean / (2.0 * math.pi) * period


def mars_gray_flat_dev(experiment_id: Optional[str] = None, *, days: int = 30,
                       dt: float = 600.0, frictional_heating: bool = True,
                       sponge: bool = False, implicit_surface: bool = False,
                       boundary_layer: bool = False) -> RadiativeConfig:
    """DEVELOPMENT EXPERIMENT, not a reproduction of Thomson & Vallis (2019)
    M-I (docs/radiation/MARS_MI_PLAN.md).

    Matched to M-I (Table 1, section 4.2) and, where the paper is silent, to
    Isca (commit d1321ac3): constants, 610 Pa, the 25-level grid, gray
    longwave 0.2 (p / 610 Pa) and shortwave 0.2 (p / 610 Pa)^4 along the
    vertical path, diurnal and seasonal insolation (589 W m^-2, e = 0.0935,
    obliquity 25.19 degrees, perihelion at L_s = 250.85 degrees, sol 88440 s,
    year 59166360 s, t = 0 at L_s = 0), a 2 m "land depth" slab
    (8.26e6 J m^-2 K^-1), albedo 0.3, roughness 3.21e-5 m, no gustiness, a
    264 K isothermal start, no topography.

    SUBSTITUTES for physics M-I has and this code lacks: Held-Suarez Rayleigh
    drag (1 day^-1 below sigma = 0.7) stands in for surface stress and
    vertical momentum diffusion, and instantaneous dry convective adjustment
    stands in for the turbulent diffusion of heat (M-I has no convection
    scheme). ``sponge=True`` (preset ``dev-mars-gray-flat-sponge``) adds
    M-I's top sponge, 1 / 0.125 day ((0.5 Pa - p) / 0.5 Pa)^2 above 0.5 Pa
    (Isca damping_driver). ``implicit_surface=True`` (preset
    ``dev-mars-gray-flat-sponge-implicit``) solves the bulk sensible exchange
    backward-Euler between bottom layer and slab (MARS_MI_PLAN.md §2c-2d;
    the explicit exchange is unstable in Isca's 0.73 Pa bottom layer once
    low-level winds pass ~25 m/s). ``boundary_layer=True`` (preset
    ``dev-mars-gray-flat-bl``) replaces BOTH substitutes with the §2e dry
    boundary layer: bulk surface stress plus implicit K-profile diffusion
    of momentum and dry static energy (drag and convection are then None,
    matching M-I's ``convection: None``); it implies the sponge and the
    implicit exchange. NOT matched: T21 (M-I: T42), del^8 and RAW (Isca:
    del^4 and Robert-Asselin 0.04), a uniform 285 K initial surface (Isca
    test case: 285 - 40 (3 sin^2 lat - 1) / 3), and the FHZ06-simplified
    exchange coefficient and neutral K profile against Isca's full
    Monin-Obukhov forms (stated in §2e).
    """
    if boundary_layer:
        sponge = implicit_surface = True
    year, ecc, ls_peri = 59166360.0, 0.0935, 250.85
    name = "dev-mars-gray-flat" + ("-bl" if boundary_layer else
                                   ("-sponge" if sponge else "") +
                                   ("-implicit" if implicit_surface else ""))
    sensible = {"roughness_length": 3.21e-5, "critical_richardson": 1.0, "gustiness": 0.0}
    if implicit_surface:
        sensible["implicit"] = True
    return RadiativeConfig(
        experiment_id=_preset_id(experiment_id, name, "T21L25", frictional_heating),
        sponge={"bottom_pressure": 0.5, "efold_days": 0.125} if sponge else None,
        frictional_heating=frictional_heating, radius=3.396e6,
        omega=2.0 * math.pi / 88308.0, gravity=3.71, gas_constant=192.0,
        cp=192.0 / 0.22727, reference_pressure=610.0,
        vertical={"kind": "interfaces", "sigma": list(ISCA_GREY_MARS_SIGMA)}, dt=dt, days=days,
        longwave={"kind": "power_law", "tau0": 0.2, "exponent": 1.0, "p0": 610.0},
        shortwave={"kind": "vertical_path", "tau0": 0.2, "exponent": 4.0, "p0": 610.0},
        insolation={"kind": "instantaneous", "stellar_constant": 589.0, "period": year,
                    "eccentricity": ecc, "obliquity_deg": 25.19,
                    "perihelion_longitude_deg": ls_peri,
                    "time_of_perihelion": _time_of_perihelion_for_equinox_start(year, ecc,
                                                                                ls_peri),
                    "solar_day": 88440.0},
        convection=None if boundary_layer else {"timescale": 0.0},
        surface={"heat_capacity": 2.0 * _ISCA_RHO_CP, "albedo": 0.3, "emissivity": 1.0,
                 "initial_temperature": 285.0},
        sensible=sensible,
        boundary_layer={"roughness_length": 3.21e-5, "critical_richardson": 1.0,
                        "rich_crit_pbl": 1.0, "frac_inner": 0.1,
                        "gustiness": 0.0} if boundary_layer else None,
        drag=None if boundary_layer else {"kind": "held_suarez", "k_f_per_day": 1.0,
                                          "sigma_b": 0.7},
        initial={"kind": "isothermal", "temperature": 264.0},
        checkpoint_every_days=10, state_every_days=1,
        **_t21())


PRESETS = {"jupiter-schneider-liu": jupiter_schneider_liu, "earth-gray-dry": earth_gray_dry,
           "dev-mars-gray-flat": mars_gray_flat_dev,
           "dev-mars-gray-flat-sponge": functools.partial(mars_gray_flat_dev, sponge=True),
           "dev-mars-gray-flat-sponge-implicit": functools.partial(
               mars_gray_flat_dev, sponge=True, implicit_surface=True, days=60),
           "dev-mars-gray-flat-bl": functools.partial(
               mars_gray_flat_dev, boundary_layer=True, days=60)}
