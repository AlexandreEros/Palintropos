"""Build the model, the column physics, the stepper and the initial state of a
radiative–dynamical experiment from a :class:`RadiativeConfig`.

``build_physics`` and ``build_sigma`` are CPU-only (they also serve the
single-column reference model); everything that touches the dynamical core
needs CuPy.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from tropoi.physics.column_model import ColumnPhysics, RadiativeConvectiveColumn
from tropoi.physics.convection import DryConvectiveAdjustment
from tropoi.physics.insolation import (DiurnalMeanInsolation, InstantaneousInsolation,
                                       PrescribedInsolation, StellarIllumination,
                                       TidallyLockedInsolation, frierson_net_surface_flux)
from tropoi.physics.radiation.gray import (GrayLongwave, GrayShortwave, OpticalDepthProfile,
                                           SemiInfiniteScatteringShortwave,
                                           frierson_longwave)
from tropoi.physics.radiation.scheme import Radiation
from tropoi.physics.surface import SlabSurface
from tropoi.run.radiative.config import DAY_SECONDS, RadiativeConfig
from tropoi.spatial.sigma_coordinate import SigmaGrid

__all__ = ["build_sigma", "build_radiation", "build_insolation", "build_physics",
           "build_model", "build_stepper", "reference_column", "initial_state"]


def _options(spec: dict, *required, **defaults) -> dict:
    """The entries of a ``{"kind": ...}`` spec other than the kind, with
    defaults applied and unknown or missing keys rejected."""
    given = {k: v for k, v in spec.items() if k != "kind"}
    unknown = sorted(set(given) - set(required) - set(defaults))
    missing = sorted(set(required) - set(given))
    if unknown or missing:
        raise ValueError(f"{spec.get('kind')!r}: unknown keys {unknown}, missing keys {missing}")
    return {**defaults, **given}


def build_sigma(cfg: RadiativeConfig) -> SigmaGrid:
    v = cfg.vertical
    kind = v["kind"]
    if kind == "uniform":
        return SigmaGrid.uniform(int(_options(v, "nlev")["nlev"]))
    if kind == "frierson":
        return SigmaGrid.frierson(int(_options(v, "nlev")["nlev"]))
    if kind == "stretched":
        o = _options(v, "nlev", "sigma_top", linear_scale=None)
        return SigmaGrid.stretched(int(o["nlev"]), float(o["sigma_top"]), o["linear_scale"])
    raise ValueError(f"unknown vertical grid kind {kind!r}")


def _longwave(spec: dict):
    kind = spec["kind"]
    if kind == "frierson":
        return frierson_longwave(**_options(spec, tau_equator=6.0, tau_pole=1.5,
                                            linear_fraction=0.1))
    if kind == "power_law":
        o = _options(spec, "tau0", exponent=1.0, p0="surface", tau_pole=None, diffusivity=1.0)
        profile = OpticalDepthProfile([(1.0, o["exponent"])], equator=o["tau0"],
                                      pole=o["tau_pole"], reference=o["p0"])
        return GrayLongwave(profile, diffusivity=o["diffusivity"])
    raise ValueError(f"unknown longwave kind {kind!r}")


def _shortwave(spec: Optional[dict]):
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "transparent":
        _options(spec)
        return GrayShortwave(OpticalDepthProfile([(1.0, 1.0)], equator=0.0))
    if kind == "semi_infinite_scattering":
        o = _options(spec, "tau0", "ssa", "asym", exponent=1.0, p0="surface")
        return SemiInfiniteScatteringShortwave(
            OpticalDepthProfile([(1.0, o["exponent"])], equator=o["tau0"], reference=o["p0"]),
            ssa=o["ssa"], asym=o["asym"])
    if kind == "power_law":
        o = _options(spec, "tau0", exponent=1.0, p0="surface", ssa=0.0, asym=0.0,
                     closure="quadrature")
        return GrayShortwave(
            OpticalDepthProfile([(1.0, o["exponent"])], equator=o["tau0"], reference=o["p0"]),
            ssa=o["ssa"], asym=o["asym"], closure=o["closure"])
    raise ValueError(f"unknown shortwave kind {kind!r}")


def build_radiation(cfg: RadiativeConfig) -> Radiation:
    return Radiation(_longwave(cfg.longwave), _shortwave(cfg.shortwave),
                     source=cfg.thermal_source)


def build_insolation(spec: Optional[dict]):
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "frierson":
        return frierson_net_surface_flux(**_options(spec, net_solar_constant=938.4,
                                                    delta_s=1.4))
    if kind == "uniform":
        flux = float(_options(spec, "flux")["flux"])
        return PrescribedInsolation(lambda lat: flux + 0.0 * lat, "uniform", {"flux": flux})
    if kind == "tidally_locked":
        return TidallyLockedInsolation(**_options(spec, "stellar_constant",
                                                  substellar_longitude=0.0))
    if kind in ("diurnal_mean", "instantaneous"):
        o = _options(spec, "stellar_constant", period=365.25 * DAY_SECONDS, eccentricity=0.0,
                     obliquity_deg=0.0, perihelion_longitude_deg=0.0, time_of_perihelion=0.0,
                     **({"solar_day": DAY_SECONDS} if kind == "instantaneous" else {}))
        solar_day = o.pop("solar_day", None)
        illumination = StellarIllumination.prescribed(**o)
        return (DiurnalMeanInsolation(illumination) if kind == "diurnal_mean"
                else InstantaneousInsolation(illumination, solar_day))
    raise ValueError(f"unknown insolation kind {kind!r}")


def _global_mean_insolation(cfg: RadiativeConfig) -> Optional[PrescribedInsolation]:
    """A uniform insolation equal to the configured one's global mean at
    t = 0 (for the horizontally uniform reference column)."""
    ins = build_insolation(cfg.insolation)
    if ins is None:
        return None
    lat = np.arcsin(np.linspace(-1.0, 1.0, 2001)[1::2])
    lon = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    flux, _ = ins(la.ravel(), lo.ravel(), 0.0)
    mean = float(np.mean(flux))
    return PrescribedInsolation(lambda latitude: mean + 0.0 * latitude, "global_mean",
                                {"flux": mean})


def build_physics(cfg: RadiativeConfig, *, insolation="configured") -> ColumnPhysics:
    """The column physics of ``cfg``; ``insolation`` may override the
    configured geometry (the reference column uses its global mean)."""
    convection = None
    if cfg.convection is not None:
        o = _options({"kind": "convection", **cfg.convection}, timescale=0.0,
                     lapse_fraction=1.0)
        convection = DryConvectiveAdjustment(cfg.kappa, timescale=o["timescale"],
                                             lapse_fraction=o["lapse_fraction"])
    surface = None
    if cfg.surface is not None:
        o = _options({"kind": "surface", **cfg.surface}, "heat_capacity", albedo=0.0,
                     emissivity=1.0, initial_temperature=None)
        surface = SlabSurface(o["heat_capacity"], albedo=o["albedo"],
                              emissivity=o["emissivity"])
    ins = build_insolation(cfg.insolation) if isinstance(insolation, str) else insolation
    return ColumnPhysics(build_radiation(cfg), gravity=cfg.gravity, cp=cfg.cp,
                         gas_constant=cfg.gas_constant, insolation=ins,
                         convection=convection, surface=surface,
                         internal_heat_flux=cfg.internal_heat_flux,
                         radiation_interval=cfg.radiation_interval, sensible=cfg.sensible)


def reference_column(cfg: RadiativeConfig, *, tolerance: float = 1e-3) -> dict:
    """Radiative(-convective) equilibrium of one column under the
    global-mean insolation (CPU). Returns the profile and its fluxes."""
    sigma = build_sigma(cfg)
    physics = build_physics(cfg, insolation=_global_mean_insolation(cfg))
    physics.sensible = None              # no wind in a column at rest
    p = (sigma.interfaces_array() * cfg.reference_pressure)[:, None]
    ts = None
    if physics.surface is not None:
        ts = np.full(1, float((cfg.surface or {}).get("initial_temperature") or 280.0))
    col = RadiativeConvectiveColumn(physics, p, np.full((sigma.nlev, 1), 200.0),
                                    latitude=np.zeros(1), longitude=np.zeros(1),
                                    surface_temperature=ts)
    result = col.equilibrate(tolerance=tolerance)
    out = col.evaluate()
    return {"temperature": col.temperature[:, 0].copy(),
            "surface_temperature": None if ts is None else float(col.surface_temperature[0]),
            "olr": float(out.fluxes.olr[0]),
            "absorbed_stellar": float(out.fluxes.absorbed_stellar[0]),
            "pressure_interfaces": p[:, 0].copy(), **result}


# ---------------------------------------------------------------------------
# GPU from here on
# ---------------------------------------------------------------------------

def build_model(cfg: RadiativeConfig):
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.primitive_equations import PrimitiveEquationsModel

    period = 2.0 * math.pi / abs(cfg.omega) if cfg.omega != 0.0 else 1.0e30
    params = PlanetaryParameters.ideal_sphere(cfg.radius, period)
    params.angular_velocity = float(cfg.omega)
    planet = Planet.generate(params=params, grid_type="latlon", nlat=cfg.nlat, nlon=cfg.nlon,
                             l_max=cfg.l_max, product_quadrature="fine")
    return PrimitiveEquationsModel(planet, build_sigma(cfg), r_dry=cfg.gas_constant,
                                   cp_dry=cfg.cp, batched_transforms=True,
                                   retained_truncation=cfg.retained_truncation)


def _drag(cfg: RadiativeConfig, model):
    """(dampers, lagged_terms) for the configured drag."""
    from tropoi.temporal.tendencies.drag import (LatitudeDependentRayleighDrag,
                                                 schneider_liu_drag_profile)
    from tropoi.temporal.tendencies.held_suarez import (HeldSuarezParameters,
                                                        rayleigh_drag_damping)
    spec = cfg.drag
    if spec is None:
        return (), ()
    kind = spec["kind"]
    if kind == "held_suarez":
        o = _options(spec, k_f_per_day=1.0, sigma_b=0.7)
        params = HeldSuarezParameters(k_f=o["k_f_per_day"] / DAY_SECONDS, sigma_b=o["sigma_b"])
        return (rayleigh_drag_damping(model.sigma.full_levels_array(), cfg.l_max + 1,
                                      params),), ()
    if kind == "schneider_liu":
        o = _options(spec, k0_per_day=0.05, sigma_b=0.8, cutoff_deg=16.3, decay_deg=2.0)
        k0 = o["k0_per_day"] / DAY_SECONDS

        def profile(latitude):
            return schneider_liu_drag_profile(latitude, k0, o["cutoff_deg"], o["decay_deg"])

        return (), (LatitudeDependentRayleighDrag(model, profile, sigma_b=o["sigma_b"],
                                                  params=o),)
    raise ValueError(f"unknown drag kind {kind!r}")


def _frictional_heating(cfg: RadiativeConfig, model, dampers, lagged_drag):
    """The heating that returns what ``dampers`` (implicit, diagonal) and
    ``lagged_drag`` (explicit) remove from the kinetic energy."""
    from tropoi.temporal.tendencies.drag import MomentumSinkHeating
    sinks = [damper.tendency for damper in dampers]
    sinks += [lambda x, term=term: term(x, 0.0, 0.0) for term in lagged_drag]
    return MomentumSinkHeating(model, sinks, cfg.cp)


def build_stepper(cfg: RadiativeConfig, model, coupling, *,
                  stage_validator: Optional[Callable] = None):
    """Semi-implicit leapfrog with del^(2n) hyperdiffusion, the configured
    drag and the column physics as a lagged term (``coupling=None`` gives
    the adiabatic model with the same numerics)."""
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator)
    from tropoi.temporal.tendencies.held_suarez import hyperdiffusion_damping
    op = SemiImplicitOperator.from_model(model, t_ref=cfg.t_ref)
    hyper = hyperdiffusion_damping(model.nlev, cfg.l_max, cfg.radius,
                                   cfg.hyperdiffusion_efold_days * DAY_SECONDS,
                                   cfg.retained_cut, order=cfg.hyperdiffusion_order)
    dampers, lagged = _drag(cfg, model)
    explicit = ()
    if cfg.frictional_heating:
        explicit = (_frictional_heating(cfg, model, dampers + (hyper,), lagged),)
    if coupling is not None:
        lagged = lagged + (coupling,)
    return SemiImplicitLeapfrogStepper(
        model.tendency, op, cfg.dt, raw_nu=cfg.raw_nu, raw_alpha=cfg.raw_alpha,
        stage_validator=stage_validator, explicit_terms=explicit, dampers=dampers + (hyper,),
        damping_scheme=cfg.damping_scheme, lagged_terms=lagged)


def initial_state(cfg: RadiativeConfig):
    """A resting, horizontally uniform state plus a seeded rotational
    perturbation. Returns ``(coeffs (CuPy), surface_temperature or None,
    info)``. The perturbation is vertically uniform (so it carries no
    vertical shear to radiate as gravity waves) with a grid RMS wind of
    about ``perturbation_wind``."""
    import cupy as cp
    from tropoi.spatial.states.primitive_equations import isothermal_rest_state
    sigma = build_sigma(cfg)
    K, n = sigma.nlev, cfg.l_max + 1
    kind = cfg.initial["kind"]
    info: dict = {"kind": kind}
    surface_t = None
    if kind == "isothermal":
        profile = np.full(K, float(cfg.initial["temperature"]))
        if cfg.surface is not None:
            surface_t = float(cfg.surface.get("initial_temperature") or profile[-1])
    elif kind == "radiative_convective":
        ref = reference_column(cfg)
        profile = ref["temperature"]
        surface_t = ref["surface_temperature"]
        info.update({k: ref[k] for k in ("olr", "absorbed_stellar", "iterations",
                                         "imbalance", "converged")})
    else:
        raise ValueError(f"unknown initial kind {kind!r}")
    x0 = isothermal_rest_state(cfg.l_max, K, temperature=float(profile[0]),
                               surface_pressure=cfg.reference_pressure).coeffs
    x0[2 * K:3 * K, 0, 0] = cp.asarray(profile) * math.sqrt(4.0 * math.pi)
    from tropoi.spatial.laplacian import laplacian_eigenvalues
    c_l = -laplacian_eigenvalues(cfg.l_max, cfg.radius)        # l (l+1) / a^2
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    zeta = np.zeros((n, n), dtype=np.complex128)
    for l in range(1, cfg.perturbation_lmax + 1):
        for m in range(l + 1):
            re, im = rng.standard_normal(2)
            zeta[l, m] = complex(re, im if m > 0 else 0.0) / math.sqrt(c_l[l])
    # mean of |V|^2 = sum |zeta_lm|^2 / c_l / (4 pi), m > 0 counted twice
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = np.where(c_l > 0, 1.0 / c_l, 0.0)[:, None]
    power = (fac * np.abs(zeta) ** 2)
    mean_v2 = (power[:, 0].sum() + 2.0 * power[:, 1:].sum()) / (4.0 * math.pi)
    zeta *= cfg.perturbation_wind / math.sqrt(mean_v2)
    x0[0:K] += cp.asarray(zeta)[None]
    info["temperature_profile"] = [float(t) for t in profile]
    return x0, surface_t, info
