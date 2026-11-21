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
           "jupiter_schneider_liu", "earth_gray_dry", "thai_dry_dev", "THAI_DRY_TABLES"]

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
    # bulk surface exchange: tropoi.physics.surface.sensible_heat_flux
    # options, plus "implicit" for the backward-Euler exchange. "scheme":
    # "monin_obukhov" with "monin_obukhov": {...} selects Isca's full
    # surface layer instead of FHZ06, and must match the same two keys in
    # "boundary_layer" (see _check_surface_layer_scheme)
    sensible: Optional[dict] = None
    drag: Optional[dict] = None
    # top Rayleigh sponge {"bottom_pressure": Pa, "efold_days": d} (Isca
    # damping_driver); left out of the canonical form when unset
    sponge: Optional[dict] = None
    # dry boundary layer (MARS_MI_PLAN.md 2e): bulk surface stress plus
    # implicit K-profile diffusion of momentum and dry static energy,
    # {"roughness_length": m, "critical_richardson": ..., "rich_crit_pbl":
    # ..., "frac_inner": ..., "gustiness": m/s}; left out of the canonical
    # form when unset so every earlier configuration keeps its hash.
    # "scheme": "monin_obukhov" with "monin_obukhov": {...} selects Isca's
    # full surface layer instead, and must match the same two keys in
    # "sensible" (see _check_surface_layer_scheme)
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
        self._check_surface_layer_scheme()
        if self.surface is None and self.shortwave is None and self.internal_heat_flux <= 0:
            raise ValueError("a surface-less column needs a stellar or interior heat source")

    def _check_surface_layer_scheme(self) -> None:
        """The surface exchange and the vertical diffusion must agree on the
        surface-layer scheme and on its parameters.

        They are separate objects evaluated on different grids - the bulk
        flux in the column physics, the stress and the K profile in the
        lagged tendency - so nothing structural stops a configuration from
        asking for a Monin-Obukhov heat flux over an FHZ06 K profile. That
        combination is not a scheme either model implements, and the two
        would disagree about the stability of the same column, so it is
        rejected here rather than left to produce a quietly meaningless
        run.
        """
        flux = (self.sensible or {}).get("scheme", "fhz06")
        diffusion = (self.boundary_layer or {}).get("scheme", "fhz06")
        if self.sensible is not None and self.boundary_layer is not None:
            if flux != diffusion:
                raise ValueError(
                    f"the sensible flux asks for scheme {flux!r} and the boundary "
                    f"layer for {diffusion!r}; the surface-layer scheme must be the "
                    "same for both")
            a = (self.sensible or {}).get("monin_obukhov") or {}
            b = (self.boundary_layer or {}).get("monin_obukhov") or {}
            if flux == "monin_obukhov" and a != b:
                raise ValueError(
                    "the sensible flux and the boundary layer must use the same "
                    f"monin_obukhov options, not {a} and {b}")
            for key in ("roughness_length", "gustiness"):
                if key in (self.sensible or {}) and key in (self.boundary_layer or {}):
                    if self.sensible[key] != self.boundary_layer[key]:
                        raise ValueError(
                            f"the sensible flux and the boundary layer disagree on "
                            f"{key}: {self.sensible[key]} and {self.boundary_layer[key]}")
        elif flux == "monin_obukhov" or diffusion == "monin_obukhov":
            # one of the two is absent; Monin-Obukhov is a surface-layer
            # scheme, so half of it is a configuration error
            raise ValueError("the monin_obukhov scheme needs both a sensible flux "
                             "and a boundary layer")

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
                       boundary_layer: bool = False,
                       monin_obukhov: bool = False) -> RadiativeConfig:
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
    Monin-Obukhov forms (stated in §2e), unless ``monin_obukhov=True``.

    ``monin_obukhov=True`` (preset ``dev-mars-gray-flat-bl-most``) keeps
    everything else and removes that last difference: the surface exchange
    becomes Isca's iterated Monin-Obukhov drag and the K profile its
    stability-dependent K_m and K_h
    (:mod:`tropoi.physics.monin_obukhov`; Isca's three published kernel
    checksums reproduce exactly as aggregates and its per-column published
    scales elementwise to a few ulp - see docs/radiation/MONIN_OBUKHOV.md
    §4, which keeps the two claims apart).
    It implies ``boundary_layer`` and runs Isca's own namelist defaults,
    which its grey-Mars test case does not override: rich_crit 2.0,
    drag_min 1e-5, stable_option 1. The surface-layer critical Richardson
    number is therefore 2.0, not the 1.0 the FHZ06 path uses; the depth
    diagnosis keeps rich_crit_pbl = 1.0, Isca's separate default for it.
    """
    if monin_obukhov:
        boundary_layer = True
    if boundary_layer:
        sponge = implicit_surface = True
    year, ecc, ls_peri = 59166360.0, 0.0935, 250.85
    name = "dev-mars-gray-flat" + (("-bl" + ("-most" if monin_obukhov else ""))
                                   if boundary_layer else
                                   ("-sponge" if sponge else "") +
                                   ("-implicit" if implicit_surface else ""))
    sensible = {"roughness_length": 3.21e-5, "critical_richardson": 1.0, "gustiness": 0.0}
    if monin_obukhov:
        # critical_richardson is the FHZ06 knob and is rejected here; the
        # Monin-Obukhov critical value is monin_obukhov.rich_crit, left at
        # Isca's default of 2.0 by the empty options dict
        sensible = {"roughness_length": 3.21e-5, "gustiness": 0.0,
                    "scheme": "monin_obukhov", "monin_obukhov": {}}
    if implicit_surface:
        sensible["implicit"] = True
    bl = None
    if boundary_layer:
        bl = {"roughness_length": 3.21e-5, "critical_richardson": 1.0,
              "rich_crit_pbl": 1.0, "frac_inner": 0.1, "gustiness": 0.0}
        if monin_obukhov:
            del bl["critical_richardson"]
            bl["scheme"] = "monin_obukhov"
            bl["monin_obukhov"] = {}
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
        boundary_layer=bl,
        drag=None if boundary_layer else {"kind": "held_suarez", "k_f_per_day": 1.0,
                                          "sigma_b": 0.7},
        initial={"kind": "isothermal", "temperature": 264.0},
        checkpoint_every_days=10, state_every_days=1,
        **_t21())


#: The k-tables of the THAI dry development cases, pinned by content hash
#: (docs/radiation/OPACITY.md: sources, coverage, assumptions, and the two
#: commands that fetch and import them into ``opacity_cache/tables``).
THAI_DRY_TABLES = {
    "ben1": {"composition": "thai_ben1", "longwave": {
        "file": "thai-ben1-sp-lw-15-dsa-ar10bar.npz",
        "sha256": "699c00492b5e5d5b9e6642da054a8daeee3f3269670a351ef689356c350e9038"},
        "shortwave": {
            "file": "thai-ben1-sp-sw-43-dsa-ar10bar-trappist1.npz",
            "sha256": "c473fca1eceb69cbb69016d856eff67093d0183e3c5542301cf28429ae35c09d"},
        # TRAPPIST-1 flux fraction of the table's 43 bands, increasing in
        # wavenumber (block 2 of sp_sw_43_dsa_ar10bar_trappist1)
        "stellar_band_fractions": (
            0.02656221023, 0.01482918161, 0.01508034607, 0.02695592583, 0.02959266107,
            0.02208575483, 0.01866939112, 0.01676674122, 0.07782652617, 0.0106526464,
            0.02536404824, 0.01529585662, 0.04146680899, 0.05677867426, 0.03676360419,
            0.01185539372, 0.02286220581, 0.02136834353, 0.01518227359, 0.01474203956,
            0.01975398087, 0.04499964863, 0.03198620258, 0.05168204686, 0.03599036771,
            0.03338955765, 0.03444872077, 0.01685195291, 0.01830330962, 0.0628369022,
            0.05388278132, 0.01439385132, 0.01384145534, 0.04184376402, 0.002530298546,
            0.00102034457, 0.0007771275866, 0.0005825989255, 0.0001044206189,
            5.747656744e-05, 1.116784097e-05, 1.136564469e-05, 2.481091694e-08)},
    "ben2": {"composition": "thai_ben2", "longwave": {
        "file": "thai-ben2-sp-lw-17-dsa-mars.npz",
        "sha256": "9d4c18a6822690aa12051598695ca4d6946324bb36f8752a35d57b9963087e18"},
        "shortwave": {
            "file": "thai-ben2-sp-sw-42-dsa-mars-trappist1.npz",
            "sha256": "c2e873d40e3da41dd52a9b3187fd47cf65631481b03a152363f6bfc6300e8386"},
        # the same for the 42 bands of sp_sw_42_dsa_mars_trappist1
        "stellar_band_fractions": (
            0.007258484028, 0.006780315862, 0.002870869908, 0.009652540429, 0.0395083342,
            0.02363320203, 0.01990910171, 0.02146444856, 0.02269817405, 0.01096201485,
            0.01248332998, 0.02849879125, 0.02621123844, 0.06775044413, 0.0348666271,
            0.03048862789, 0.03289022826, 0.09284954726, 0.01518227359, 0.01474203956,
            0.01975398087, 0.04499964863, 0.04929574556, 0.03437250387, 0.03599036771,
            0.0508297636, 0.05216377736, 0.04110577227, 0.09996624428, 0.003882973633,
            0.02612883711, 0.004724752917, 0.01099017399, 0.002530298546, 0.00102034457,
            0.0007771275866, 0.0005825989255, 0.0001044206189, 5.747656744e-05,
            1.116784097e-05, 1.136564469e-05, 2.481091694e-08)},
}
#: Temperature (K) at which the constant c_p of the THAI dry cases is evaluated.
THAI_CP_TEMPERATURE = 250.0


def thai_dry_dev(case: str, experiment_id: Optional[str] = None, *, days: int = 5,
                 dt: float = 1200.0, frictional_heating: bool = True,
                 radiation_interval: int = 3,
                 stellar_absorption: bool = False) -> RadiativeConfig:
    """DEVELOPMENT EXPERIMENT with a THAI dry composition and real-gas
    longwave absorption; not a THAI reproduction (docs/radiation/OPACITY.md,
    docs/radiation/THAI_BEN_PLAN.md).

    ``case`` is ``"ben1"`` (1 bar N2 with 400 ppmv CO2) or ``"ben2"`` (1 bar
    CO2). The two differ only through inputs: the composition preset
    (``thai_ben1`` / ``thai_ben2`` of ``tropoi.physics.composition``) and
    the premixed longwave k-table imported for it. The bulk thermodynamics
    come from that same composition - ``gas_constant`` from its molar mass
    and a constant ``cp`` from its heat capacity at 250 K - and the table
    loader refuses a table mixed for another gas constant.

    Matched to the THAI protocol (Fauchez et al. 2020): radius 5798 km,
    g = 9.12 m s^-2, synchronous 6.1-day rotation, 900 W m^-2 on a fixed
    substellar point, 1 bar, surface albedo 0.3, roughness 1e-2 m, a 300 K
    isothermal start at rest, dry.

    Longwave: correlated-k, 1-3000 cm^-1, from the ROCKE-3D SOCRATES
    spectral files ``sp_lw_17_dsa_mars`` (Ben 2: self-broadened CO2 lines,
    sub-Lorentzian wings, CO2-CO2 CIA) and ``sp_lw_15_dsa_ar10bar`` (Ben 1:
    air-broadened CO2 lines, N2-N2 CIA; no N2-CO2 CIA). States outside
    1 Pa - 10 bar or 100 - 400 K are an error.

    SHORTWAVE, two variants. By default (``dev-thai-ben1`` / ``-ben2``, id
    ``...-lwck-...``) the atmosphere is transparent to starlight: the
    surface absorbs 70 % of the incident flux. THAI's Bond albedos
    (0.27-0.30 Ben 1, 0.18-0.22 Ben 2) show the atmosphere absorbs a large
    share of an M dwarf's light in Ben 2, so that is the largest known
    departure of the default. ``stellar_absorption=True``
    (``dev-thai-ben1-ck`` / ``-ben2-ck``, id ``...-ck-...``) adds real-gas
    stellar absorption, 500-50000 cm^-1, from the TRAPPIST-1 files
    ``sp_sw_42_dsa_mars_trappist1`` and ``sp_sw_43_dsa_ar10bar_trappist1``
    with the star's band fluxes as those files give them. Still an
    approximation: absorption only, NO Rayleigh scattering (its
    flux-weighted optical depth for this star is 0.008 and 0.005).

    NOT matched: T21 L40 to 4 Pa with del^8 and no sponge; a one-layer slab
    of 2e6 J m^-2 K^-1 (the UM's); the FHZ06 dry boundary layer of the Mars
    development runs in place of each THAI model's own; constant c_p;
    radiation every ``radiation_interval`` steps (hourly at 1200 s). Short
    runs from the isothermal start are nowhere near equilibrium.
    """
    if case not in THAI_DRY_TABLES:
        raise ValueError(f"case must be one of {sorted(THAI_DRY_TABLES)}, not {case!r}")
    from tropoi.physics.composition import COMPOSITIONS
    from tropoi.physics.thermodynamics import heat_capacity
    entry = THAI_DRY_TABLES[case]
    composition = COMPOSITIONS[entry["composition"]]
    roughness = 1.0e-2
    shortwave: dict = {"kind": "transparent"}
    if stellar_absorption:
        shortwave = {"kind": "correlated_k", "tables": [dict(entry["shortwave"])],
                     "mass_fractions": {composition.label: 1.0}, "bounds": "error",
                     "band_fractions": list(entry["stellar_band_fractions"])}
    tag = "ck" if stellar_absorption else "lwck"
    return RadiativeConfig(
        experiment_id=_preset_id(experiment_id, f"dev-thai-{case}-{tag}", "T21L40",
                                 frictional_heating),
        frictional_heating=frictional_heating, radius=5.798e6,
        omega=2.0 * math.pi / (6.1 * DAY_SECONDS), gravity=9.12,
        gas_constant=float(composition.gas_constant),
        cp=float(heat_capacity(composition, THAI_CP_TEMPERATURE)),
        reference_pressure=1.0e5,
        vertical={"kind": "stretched", "nlev": 40, "sigma_top": 4.0e-5,
                  "linear_scale": 0.15},
        dt=dt, days=days,
        longwave={"kind": "correlated_k", "tables": [dict(entry["longwave"])],
                  "mass_fractions": {composition.label: 1.0}, "bounds": "error"},
        shortwave=shortwave,
        insolation={"kind": "tidally_locked", "stellar_constant": 900.0},
        convection=None,
        surface={"heat_capacity": 2.0e6, "albedo": 0.3, "emissivity": 1.0,
                 "initial_temperature": 300.0},
        sensible={"roughness_length": roughness, "critical_richardson": 1.0,
                  "gustiness": 0.0, "implicit": True},
        boundary_layer={"roughness_length": roughness, "critical_richardson": 1.0,
                        "rich_crit_pbl": 1.0, "frac_inner": 0.1, "gustiness": 0.0},
        drag=None, radiation_interval=radiation_interval,
        initial={"kind": "isothermal", "temperature": 300.0},
        checkpoint_every_days=1, state_every_days=1,
        **_t21())


PRESETS = {"jupiter-schneider-liu": jupiter_schneider_liu, "earth-gray-dry": earth_gray_dry,
           "dev-thai-ben1": functools.partial(thai_dry_dev, "ben1"),
           "dev-thai-ben2": functools.partial(thai_dry_dev, "ben2"),
           "dev-thai-ben1-ck": functools.partial(thai_dry_dev, "ben1",
                                                 stellar_absorption=True),
           "dev-thai-ben2-ck": functools.partial(thai_dry_dev, "ben2",
                                                 stellar_absorption=True),
           "dev-mars-gray-flat": mars_gray_flat_dev,
           "dev-mars-gray-flat-sponge": functools.partial(mars_gray_flat_dev, sponge=True),
           "dev-mars-gray-flat-sponge-implicit": functools.partial(
               mars_gray_flat_dev, sponge=True, implicit_surface=True, days=60),
           "dev-mars-gray-flat-bl": functools.partial(
               mars_gray_flat_dev, boundary_layer=True, days=60),
           "dev-mars-gray-flat-bl-most": functools.partial(
               mars_gray_flat_dev, monin_obukhov=True, days=60)}
