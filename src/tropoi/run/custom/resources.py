"""Shared specification validation and construction for state/terrain preparation."""
from __future__ import annotations
from tropoi.spatial.specification import PlanetSpec, DiscretizationSpec, TopographySpec

class PreparationError(ValueError):
    """An input cannot become a valid initial state under the policy.

    ``report`` holds whatever was measured before the failure.
    """

    def __init__(self, message: str, report: "PreparationReport | None" = None):
        super().__init__(message)
        self.report = report


def _base_config(planet: PlanetSpec, discretization: DiscretizationSpec
                 ) -> dict:
    return {**discretization.run_config_entries(),
            "planet": planet.to_dict(), "day_hours": planet.day_hours}


def _build_planet(planet: PlanetSpec, discretization: DiscretizationSpec):
    from tropoi.run.bve.model import build_planet

    return build_planet(_base_config(planet, discretization))


def _check_specs(planet, discretization) -> None:
    if not isinstance(planet, PlanetSpec):
        raise PreparationError(
            f"planet must be a PlanetSpec, got {type(planet).__name__}")
    if not isinstance(discretization, DiscretizationSpec):
        raise PreparationError(
            "discretization must be a DiscretizationSpec, got "
            f"{type(discretization).__name__}")


def _resolve_topography(planet_obj, spec: TopographySpec):
    """The model's Topography for a specification (CUDA), or None if flat."""
    from tropoi.spatial.terrain.topography import Topography

    if spec.is_flat:
        return None
    if spec.preset == "imported":
        raise PreparationError(
            "imported terrain is resolved once, from its file, by "
            "prepare_planet(...); pass the PreparedPlanet as topography")
    p = spec.parameters
    if spec.preset == "mountain":
        return Topography.mountain(planet_obj, height_m=p["height_m"],
                                   lat_deg=p["lat_deg"], lon_deg=p["lon_deg"],
                                   width_deg=p["width_deg"])
    return Topography.williamson5_cone(planet_obj, height_m=p["height_m"])

