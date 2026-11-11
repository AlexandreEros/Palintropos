"""Configuration of a radiative–dynamical experiment (CPU-only, hashable).

Every quantity that determines the result is a field of
:class:`RadiativeConfig`; its canonical JSON is hashed into
``config_sha256``, which every checkpoint carries and every resume
compares. Physics components are plain ``{"kind": ..., ...}`` dictionaries
resolved by :mod:`tropoi.run.radiative.model`, so a configuration file is
readable without importing any solver. Nothing here imports CuPy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

__all__ = ["RadiativeConfig", "PRESETS", "CONFIG_SCHEMA", "DAY_SECONDS",
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
    internal_heat_flux: float = 0.0
    # return the kinetic energy removed by drag and hyperdiffusion as heat
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
        if self.damping_scheme not in ("lagged", "centred"):
            raise ValueError("damping_scheme must be 'lagged' or 'centred'")
        if "kind" not in self.vertical or "kind" not in self.longwave:
            raise ValueError("vertical and longwave need a 'kind'")
        if self.sensible is not None and self.surface is None:
            raise ValueError("sensible exchange needs a surface")
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


def jupiter_schneider_liu(experiment_id: str = "jupiter-sl09-T21L30", *, days: int = 200,
                          dt: float = 1200.0) -> RadiativeConfig:
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
    """
    return RadiativeConfig(
        experiment_id=experiment_id, radius=69.86e6, omega=1.7587e-4, gravity=26.0,
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


def earth_gray_dry(experiment_id: str = "earth-gray-dry-T21L25", *, days: int = 300,
                   dt: float = 1200.0) -> RadiativeConfig:
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
    """
    return RadiativeConfig(
        experiment_id=experiment_id, radius=6.376e6, omega=7.292e-5, gravity=9.8,
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


PRESETS = {"jupiter-schneider-liu": jupiter_schneider_liu, "earth-gray-dry": earth_gray_dry}
