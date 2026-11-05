"""Build a solver's model objects from a resolved run configuration.

One builder per solver, each taking the resolved run-configuration mapping —
exactly the values ``config.json`` persists — and returning the resources that
solver integrates: a :class:`~tropoi.spatial.planet.Planet` for the BVE, a
model object for the shallow-water and primitive equations. Two callers need
the same construction and must not drift apart: the runners, which build the
model of a run about to be integrated, and the saved-run interface, which
rebuilds the model of a finished run from its persisted configuration.

These builders live below both callers because they depend only on the
discretization and the cores (``spatial``, ``temporal.tendencies``), never on
the configuration or orchestration layers: they read a plain mapping and
never validate it. Validation, defaulting and preset resolution stay in the
configuration layer, which is why a legacy capsule can be rebuilt here
without being re-validated by newer preset guards. The historical import
paths ``tropoi.run.{bve,swe,pe}.model`` re-export these functions unchanged.

Heavy numerical imports (CuPy through ``Planet``, the cores) happen only when
a builder is called, so importing this module stays free.
"""
from __future__ import annotations

from typing import Mapping


def build_planet(run_config: Mapping):
    """The BVE run's Planet (grid, transform, backend, operators).

    Built from the resolved run configuration dict, so the same resources
    can be reconstructed later from a saved capsule's ``config.json`` (the
    saved-run interface does this lazily for plotting). A ``planet`` block
    (runs started from a saved initial state) is authoritative for the
    planet; otherwise the preset ``from_earth_like`` planet. Legacy capsules
    predating the ``grid`` / ``product_quadrature`` keys use the only
    backend and the ``Planet.generate`` default that existed then.
    Imports CuPy through Planet; call only after validation.
    """
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet

    if "planet" in run_config:
        # Custom runs (tropoi.run.custom) record an explicit PlanetSpec.
        from tropoi.spatial.specification import PlanetSpec
        params = PlanetSpec.from_dict(run_config["planet"]).to_parameters()
    else:
        params = PlanetaryParameters.from_earth_like(
            day_hours=float(run_config["day_hours"]),
            radius_earth_units=float(run_config["radius_earth_units"]))
    return Planet.generate(
        params=params,
        grid_resolution=int(run_config["resolution"]),
        l_max=int(run_config["lmax"]),
        product_quadrature=run_config.get("product_quadrature", "fine"),
        grid_type=run_config.get("grid", "geodesic"),
        nlat=int(run_config["nlat"]),
        nlon=int(run_config["nlon"]),
    )


def _w5_planet_params(run_config):
    """Exact canonical-planet parameters for the Williamson-5 scenario.

    The Williamson (1992) suite prescribes a PERFECT SPHERE of radius
    a = 6.37122e6 m rotating at Omega = 7.292e-5 s^-1. The ordinary
    ``from_earth_like`` path shrinks the dynamical radius ~0.06% through
    the oblateness model and uses the 6.371e6 m Earth radius — silently
    noncanonical for this benchmark. For williamson5,
    ``radius_earth_units`` therefore scales the CANONICAL radius on an
    ideal sphere, and the canonical ``day_hours`` (resolved by the config)
    reproduces Omega = 7.292e-5 exactly (2*pi/(day_hours*3600) round-trips
    bitwise; pinned by tests).
    """
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.williamson5 import W5_RADIUS_M

    # Accept the resolved SWERunConfig (historical callers/tests) as well as
    # the plain run-configuration mapping the saved-run interface passes.
    if isinstance(run_config, Mapping):
        radius_units = float(run_config["radius_earth_units"])
        day_hours = float(run_config["day_hours"])
    else:
        radius_units = float(run_config.radius_earth_units)
        day_hours = float(run_config.day_hours)
    return PlanetaryParameters.ideal_sphere(
        radius_m=radius_units * W5_RADIUS_M,
        sidereal_day_s=day_hours * 3600.0)


def build_swe_model(run_config: Mapping, *, topography_lm=None):
    """The shallow-water model (planet + terrain + core) of a run.

    Built from the resolved run configuration dict — exactly the values
    ``config.json`` persists — so a saved capsule's model can be
    reconstructed later without re-running preset validation (legacy
    capsules are never re-validated by newer preset guards). Topography is
    reconstructed deterministically from the same keys the scientific
    hash covers. Imports CuPy; call only after validation.

    Runs started from a saved initial state (``tropoi.run.custom``) carry
    an explicit ``planet`` block, which is authoritative for the planet,
    and use their persisted, already-resolved terrain: ``topography_lm``
    (host or device elevation coefficients, metres) is then required for
    any non-flat terrain and is used exactly, never re-projected.
    """
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.shallow_water import ShallowWaterModel
    from tropoi.spatial.terrain.topography import Topography

    if "planet" in run_config:
        from tropoi.spatial.specification import PlanetSpec
        params = PlanetSpec.from_dict(run_config["planet"]).to_parameters()
    elif run_config.get("scenario") == "williamson5":
        # Benchmark planets are exact ideal spheres (see _w5_planet_params).
        params = _w5_planet_params(run_config)
    else:
        params = PlanetaryParameters.from_earth_like(
            day_hours=float(run_config["day_hours"]),
            radius_earth_units=float(run_config["radius_earth_units"]))
    planet = Planet.generate(
        params=params,
        grid_resolution=int(run_config["resolution"]),
        l_max=int(run_config["lmax"]),
        product_quadrature="fine",
        grid_type=run_config.get("grid", "geodesic"),
        nlat=int(run_config["nlat"]),
        nlon=int(run_config["nlon"]),
    )
    # Topography is reconstructed deterministically from the resolved
    # configuration (which participates in the scientific hash), so no
    # terrain arrays need to be persisted with the run. Manifests without
    # a topography key are flat runs.
    topography_kind = run_config.get("topography", "flat")
    if topography_lm is not None:
        if topography_kind == "flat":
            raise ValueError(
                "resolved terrain coefficients were given for a flat-bottom "
                "run configuration")
        import cupy as cp
        topography = Topography(
            cp.asarray(topography_lm, dtype=cp.complex128),
            preset=topography_kind,
            parameters=dict(run_config.get("topography_parameters") or {}))
    elif "initial_state_digest" in run_config and topography_kind != "flat":
        raise ValueError(
            "this run started from a saved initial state whose resolved "
            "terrain must be supplied (topography_lm, persisted under the "
            "capsule's initial_state/); it is never re-projected")
    elif topography_kind == "mountain":
        topography = Topography.mountain(
            planet,
            height_m=float(run_config["mountain_height_m"]),
            lat_deg=float(run_config["mountain_lat_deg"]),
            lon_deg=float(run_config["mountain_lon_deg"]),
            width_deg=float(run_config["mountain_width_deg"]))
    elif topography_kind == "williamson5_cone":
        topography = Topography.williamson5_cone(planet)
    elif topography_kind == "flat":
        topography = None
    else:
        raise ValueError(
            f"unknown topography {topography_kind!r} in the run configuration")
    model = ShallowWaterModel(planet, gravity=float(run_config["gravity"]),
                              mean_depth=float(run_config["mean_depth_m"]),
                              topography=topography)
    return model


def build_pe_model(run_config: Mapping, *, topography_lm=None):
    """The primitive-equation model (planet + sigma grid + terrain) of a run.

    Built from the resolved run configuration dict — exactly the values
    ``config.json`` persists — so a saved capsule's model can be
    reconstructed later without re-running preset validation. Topography
    is scenario-independent fixed model data, reconstructed
    deterministically from the resolved configuration (which participates
    in the scientific hash) exactly like the SWE path; the PE model
    consumes the derived surface geopotential Phi_s = g * h_s. PE terrain
    is band-limited at the model's dealiased product-truncation cut
    (l_cut), the resolved band of the dealiased pressure-gradient
    pathway; the projection-residual gate validates the terrain actually
    used and fails loudly when the mountain is too narrow for the cut.

    A run over a prepared terrain (``terrain_digest`` in the configuration)
    is built from its persisted coefficients, ``topography_lm`` (the
    capsule's ``terrain/``), which must be supplied: they are never
    re-projected. Imports CuPy; call only after validation.
    """
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.primitive_equations import (
        PrimitiveEquationsModel)
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.spatial.terrain.topography import Topography

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=float(run_config["day_hours"]),
            radius_earth_units=float(run_config["radius_earth_units"])),
        grid_resolution=int(run_config["resolution"]),
        l_max=int(run_config["lmax"]),
        product_quadrature="fine",
        grid_type=run_config.get("grid", "geodesic"),
        nlat=int(run_config["nlat"]),
        nlon=int(run_config["nlon"]),
    )
    topography_kind = run_config.get("topography", "flat")
    if "terrain_digest" in run_config:
        if topography_lm is None:
            raise ValueError(
                "this run uses a prepared terrain; its resolved coefficients "
                "(persisted under the capsule's terrain/) must be supplied "
                "as topography_lm; they are never re-projected")
        import cupy as cp
        import numpy as np
        lm = np.asarray(topography_lm, dtype=np.complex128)
        cut = int(run_config["terrain_l_cut"])
        if np.any(lm[cut + 1:, :]) or np.any(lm[:, cut + 1:]):
            raise ValueError(
                f"prepared terrain has content above its cut {cut}")
        surface_geopotential_lm = float(run_config["gravity"]) * cp.asarray(lm)
    elif topography_kind == "mountain":
        from tropoi.temporal.tendencies.primitive_equations import (
            product_truncation_cut)
        topography = Topography.mountain(
            planet,
            height_m=float(run_config["mountain_height_m"]),
            lat_deg=float(run_config["mountain_lat_deg"]),
            lon_deg=float(run_config["mountain_lon_deg"]),
            width_deg=float(run_config["mountain_width_deg"]),
            l_cut=product_truncation_cut(int(run_config["lmax"])))
        surface_geopotential_lm = topography.surface_geopotential_lm(
            float(run_config["gravity"]))
    elif topography_kind == "flat":
        surface_geopotential_lm = None
    else:
        raise ValueError(
            f"unknown topography {topography_kind!r} in the run configuration")
    interfaces = run_config.get("sigma_interfaces")
    if interfaces is None:
        sigma = SigmaGrid.uniform(int(run_config["nlev"]))
    else:
        sigma = SigmaGrid(tuple(float(s) for s in interfaces))
    # Level-batched transforms are an opt-in model setting (not part of the
    # persisted run configuration or run identity): the default per-level
    # path is what every saved capsule's hash was produced with.
    return PrimitiveEquationsModel(
        planet, sigma, r_dry=float(run_config["r_dry"]),
        cp_dry=float(run_config["cp_dry"]),
        surface_geopotential_lm=surface_geopotential_lm,
        batched_transforms=bool(run_config.get("batched_transforms", False)))
