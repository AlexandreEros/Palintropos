"""Construct PE resources from resolved or saved run configuration.

Heavy numerical imports happen only when a builder is called.
"""
from __future__ import annotations
from typing import Mapping

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
