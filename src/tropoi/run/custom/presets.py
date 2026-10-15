"""Resolved initial states of the existing presets, as :class:`InitialState`.

``initial_state_from_preset`` builds a preset run's model exactly as
``tropoi run bve|swe`` does (same configuration resolution, same builders,
same initial-condition factories) and captures the resolved spectral state
together with its specifications. Saved and re-run through
:func:`tropoi.run.custom.run_initial_state`, it reproduces the preset's
trajectory; that equivalence is what the preset-versus-reloaded tests check.

``planet_spec_for_run_config`` and ``topography_spec_for_run_config`` are
CPU-only: they state, as specifications, the planet and terrain a preset run
configuration implies (ordinary ``from_earth_like`` planets, or the exact
Williamson-5 ideal sphere).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

from tropoi.representation.archive.initial_state import InitialState
from tropoi.spatial.specification import (DiscretizationSpec, FluidLayerSpec,
                                          PlanetSpec, TopographySpec)
from tropoi.spatial.williamson5 import W5_CONE_HEIGHT_M, W5_RADIUS_M

__all__ = [
    "initial_state_from_config",
    "initial_state_from_preset",
    "planet_spec_for_run_config",
    "topography_spec_for_run_config",
]


def planet_spec_for_run_config(run_config: Mapping) -> PlanetSpec:
    """The planet a resolved run configuration builds, as a specification.

    Mirrors the builders (``cli.bve.build_planet``,
    ``cli.swe.build_swe_model``): an explicit ``planet`` block (custom runs)
    is authoritative; the Williamson-5 scenario uses the ideal sphere of
    radius ``radius_earth_units * W5_RADIUS_M``; every other preset uses
    ``from_earth_like(day_hours, radius_earth_units)``.
    """
    if "planet" in run_config:
        return PlanetSpec.from_dict(run_config["planet"])
    day_hours = float(run_config["day_hours"])
    units = float(run_config["radius_earth_units"])
    if run_config.get("scenario") == "williamson5":
        return PlanetSpec.ideal_sphere(units * W5_RADIUS_M, day_hours * 3600.0)
    return PlanetSpec.earth_like(day_hours=day_hours, radius_earth_units=units)


def topography_spec_for_run_config(run_config: Mapping) -> TopographySpec:
    """The terrain specification of a resolved SWE run configuration."""
    kind = run_config.get("topography", "flat")
    if kind == "flat":
        return TopographySpec.flat()
    if kind == "mountain":
        return TopographySpec.mountain(
            height_m=run_config["mountain_height_m"],
            lat_deg=run_config["mountain_lat_deg"],
            lon_deg=run_config["mountain_lon_deg"],
            width_deg=run_config["mountain_width_deg"])
    if kind == "williamson5_cone":
        return TopographySpec.williamson5_cone(height_m=W5_CONE_HEIGHT_M)
    raise ValueError(f"unknown topography {kind!r} in the run configuration")


def _finite_parameters(parameters: Mapping) -> dict:
    return {str(k): float(v) for k, v in parameters.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v))}


def initial_state_from_config(cfg, *, label: Optional[str] = None,
                              preset: Optional[str] = None) -> InitialState:
    """Resolve a preset run configuration's initial state (requires CUDA).

    ``cfg`` is a resolved ``BVERunConfig`` or ``SWERunConfig``. The model is
    built with the run's own builders and the state with the run's own
    initial-condition factory, so the captured coefficients are exactly the
    ones ``tropoi run`` would integrate.
    """
    import cupy as cp

    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.swe.config import SWERunConfig

    run_config = cfg.to_run_config_dict()
    planet_spec = planet_spec_for_run_config(run_config)
    provenance: dict = {"kind": "preset-export", "scenario": cfg.scenario,
                        "preset": preset}
    if isinstance(cfg, BVERunConfig):
        from tropoi.cli.bve import build_planet
        from tropoi.spatial.initialization.bve import make_ic

        planet = build_planet(run_config)
        zeta0_lm = planet.sh.transform(make_ic(cfg.scenario, planet))
        provenance["solver"] = "bve"
        return InitialState(
            solver="bve", coeffs=cp.asnumpy(zeta0_lm), planet=planet_spec,
            discretization=DiscretizationSpec(
                lmax=cfg.lmax, grid=cfg.grid, resolution=cfg.resolution,
                nlat=cfg.nlat, nlon=cfg.nlon,
                product_quadrature=cfg.product_quadrature),
            label=label or f"bve-{preset or cfg.scenario}",
            provenance=provenance)
    if isinstance(cfg, SWERunConfig):
        from tropoi.cli.swe import build_swe_model
        from tropoi.spatial.initialization.swe import make_swe_ic

        model = build_swe_model(run_config)
        state0 = make_swe_ic(cfg.scenario, model)
        topography_lm = None
        if model.has_topography:
            topography_lm = cp.asnumpy(model.topography.elevation_lm)
            provenance["topography_measured"] = _finite_parameters(
                model.topography.parameters)
        provenance["solver"] = "swe"
        return InitialState(
            solver="swe", coeffs=cp.asnumpy(state0.coeffs),
            planet=planet_spec,
            discretization=DiscretizationSpec(
                lmax=cfg.lmax, grid=cfg.grid, resolution=cfg.resolution,
                nlat=cfg.nlat, nlon=cfg.nlon, product_quadrature="fine"),
            fluid=FluidLayerSpec(cfg.gravity, cfg.mean_depth_m),
            topography=topography_spec_for_run_config(run_config),
            topography_lm=topography_lm,
            label=label or f"swe-{cfg.scenario}",
            provenance=provenance)
    raise TypeError(
        f"expected a BVERunConfig or SWERunConfig, got {type(cfg).__name__}")


def initial_state_from_preset(solver: str, settings: Optional[Mapping] = None,
                              *, preset: Optional[str] = None,
                              label: Optional[str] = None) -> InitialState:
    """Resolve the initial state of ``tropoi run <solver>`` (requires CUDA).

    ``settings`` are the explicit run settings, with the CLI's destination
    names (for example ``{"scenario": "rh4", "lmax": 21}`` or
    ``{"scenario": "williamson5", "grid": "latlon", "nlat": 64, "nlon":
    128, "lmax": 42}``); ``preset`` names a ``tropoi run bve --preset``
    bundle. Configuration errors are raised before CUDA is touched.
    """
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.swe.config import SWERunConfig

    settings = dict(settings or {})
    if solver == "bve":
        preset_settings = None
        if preset is not None:
            from tropoi.cli.main import PRESETS
            if preset not in PRESETS:
                raise ValueError(
                    f"unknown BVE preset {preset!r}; choose from "
                    f"{sorted(PRESETS)}")
            preset_settings = PRESETS[preset]["settings"]
        cfg = BVERunConfig.resolve(settings, preset_settings)
    elif solver == "swe":
        if preset is not None:
            raise ValueError("the SWE has no named presets; pass settings")
        cfg = SWERunConfig.resolve(settings)
    else:
        raise ValueError(
            f"solver must be 'bve' or 'swe' (PE is not supported), got "
            f"{solver!r}")
    return initial_state_from_config(cfg, label=label, preset=preset)
