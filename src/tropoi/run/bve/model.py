"""Construct BVE resources from resolved or saved run configuration.

Heavy numerical imports happen only when a builder is called.
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

