"""Construct SWE resources from resolved or saved run configuration.

Heavy numerical imports happen only when a builder is called.
"""
from __future__ import annotations

from typing import Mapping


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
    from tropoi.run.swe.config import W5_RADIUS_M

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
