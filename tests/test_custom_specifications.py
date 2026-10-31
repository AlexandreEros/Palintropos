"""Planet / fluid / topography / discretization specifications (CPU only).

The specifications are the persistable, versioned description of a custom
run's prescribed physics and discretization. These tests pin that they
reproduce the preset planets bit for bit, keep the ordinary-versus-
ideal-sphere radius distinction, keep SWE gravity explicit, round-trip
through strict JSON, and refuse anything they cannot reproduce.
"""
from __future__ import annotations

import ast
import json
import math
import pathlib

import pytest

from tropoi.spatial.environment import PlanetaryParameters
from tropoi.spatial.specification import (DiscretizationSpec, FluidLayerSpec,
                                          PlanetSpec, SpecificationError,
                                          TopographySpec)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "tropoi"

_PARAMETER_ATTRS = ("mass", "equatorial_radius", "sidereal_day",
                    "angular_velocity", "oblateness", "polar_radius",
                    "volume", "radius", "density")


def _same_parameters(a, b):
    for name in _PARAMETER_ATTRS:
        va, vb = float(getattr(a, name)), float(getattr(b, name))
        assert va == vb or (math.isnan(va) and math.isnan(vb)), name


def _strict_round_trip(document):
    return json.loads(json.dumps(document, allow_nan=False))


# ---------------------------------------------------------------------------
# PlanetSpec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("day_hours,units", [(24.0, 1.0), (math.inf, 1.0),
                                             (10.0, 0.5), (23.9345, 2.0)])
def test_earth_like_reproduces_from_earth_like_bitwise(day_hours, units):
    spec = PlanetSpec.earth_like(day_hours=day_hours, radius_earth_units=units)
    _same_parameters(spec.to_parameters(),
                     PlanetaryParameters.from_earth_like(
                         day_hours=day_hours, radius_earth_units=units))
    assert spec.construction == "oblate_volumetric_mean"
    assert spec.rotating is (day_hours != math.inf)


def test_preset_run_configs_map_to_the_planets_their_builders_use():
    from tropoi.run.swe.model import _w5_planet_params
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.custom import planet_spec_for_run_config
    from tropoi.run.swe.config import SWERunConfig

    w5 = SWERunConfig.resolve({"scenario": "williamson5"}).to_run_config_dict()
    spec = planet_spec_for_run_config(w5)
    assert spec.construction == "ideal_sphere"
    _same_parameters(spec.to_parameters(), _w5_planet_params(w5))
    # The benchmark keeps its exact dynamical radius and rotation rate.
    assert spec.dynamical_radius_m == 6.37122e6
    assert spec.angular_velocity_rad_s == 7.292e-5

    w2 = SWERunConfig.resolve({}).to_run_config_dict()
    _same_parameters(planet_spec_for_run_config(w2).to_parameters(),
                     PlanetaryParameters.from_earth_like(
                         day_hours=w2["day_hours"], radius_earth_units=1.0))
    bve = BVERunConfig.resolve({}).to_run_config_dict()
    spec = planet_spec_for_run_config(bve)
    assert not spec.rotating                       # BVE default: non-rotating
    _same_parameters(spec.to_parameters(),
                     PlanetaryParameters.from_earth_like(
                         day_hours=math.inf, radius_earth_units=1.0))


def test_ideal_sphere_radius_is_exact_and_oblate_radius_is_derived():
    sphere = PlanetSpec.ideal_sphere(3.4e6, 88642.0)
    assert sphere.dynamical_radius_m == 3.4e6
    assert sphere.derived()["oblateness"] == 0.0
    oblate = PlanetSpec.oblate(3.4e6, 88642.0)
    assert oblate.dynamical_radius_m < 3.4e6         # rotational flattening
    assert PlanetSpec.oblate(3.4e6, None).dynamical_radius_m == \
        pytest.approx(3.4e6, rel=1e-15)
    # Both record their convention explicitly.
    assert "exactly" in sphere.to_dict()["radius_convention"]
    assert "derived" in oblate.to_dict()["radius_convention"]


@pytest.mark.parametrize("spec", [
    PlanetSpec.ideal_sphere(6.37122e6, 2 * math.pi / 7.292e-5),
    PlanetSpec.ideal_sphere(1.0e6, None, mass_kg=1e22),
    PlanetSpec.earth_like(day_hours=math.inf),
    PlanetSpec.oblate(7.1492e7, 35730.0, mass_kg=1.898e27),
])
def test_planet_round_trips_through_strict_json(spec):
    document = _strict_round_trip(spec.to_dict())
    assert PlanetSpec.from_dict(document) == spec
    if not spec.rotating:
        assert document["sidereal_day_s"] is None
        assert spec.day_hours == math.inf


def test_planet_document_refuses_what_it_cannot_reproduce():
    document = PlanetSpec.earth_like().to_dict()
    tampered = dict(document, derived=dict(document["derived"],
                                           dynamical_radius_m=6.371e6))
    with pytest.raises(SpecificationError, match="derived"):
        PlanetSpec.from_dict(tampered)
    with pytest.raises(SpecificationError, match="version"):
        PlanetSpec.from_dict(dict(document, version=2))
    with pytest.raises(SpecificationError, match="unknown key"):
        PlanetSpec.from_dict(dict(document, gravity=9.81))
    with pytest.raises(SpecificationError, match="construction"):
        PlanetSpec.from_dict(dict(document, construction="geoid"))


@pytest.mark.parametrize("kwargs,match", [
    ({"radius_m": -1.0, "sidereal_day_s": 1.0}, "reference_radius_m"),
    ({"radius_m": math.nan, "sidereal_day_s": 1.0}, "finite"),
    ({"radius_m": 1.0, "sidereal_day_s": 0.0}, "sidereal_day_s"),
    ({"radius_m": 1.0, "sidereal_day_s": 1.0, "mass_kg": 0.0}, "mass_kg"),
    ({"radius_m": True, "sidereal_day_s": 1.0}, "real number"),
])
def test_planet_rejects_invalid_values(kwargs, match):
    with pytest.raises(SpecificationError, match=match):
        PlanetSpec.ideal_sphere(**kwargs)


# ---------------------------------------------------------------------------
# Fluid layer, topography, discretization
# ---------------------------------------------------------------------------

def test_fluid_layer_keeps_gravity_explicit_and_documents_phi():
    fluid = FluidLayerSpec(gravity_m_s2=3.71, mean_depth_m=1200.0)
    assert fluid.phi0 == 3.71 * 1200.0
    document = _strict_round_trip(fluid.to_dict())
    assert "h = mean_depth_m + phi / gravity_m_s2" in document["phi_definition"]
    assert FluidLayerSpec.from_dict(document) == fluid
    # No mass/radius anywhere: gravity is never derived.
    assert set(document) == {"version", "gravity_m_s2", "mean_depth_m",
                             "phi_definition"}
    for bad in ({"gravity_m_s2": 0.0, "mean_depth_m": 1.0},
                {"gravity_m_s2": 9.8, "mean_depth_m": -1.0},
                {"gravity_m_s2": math.inf, "mean_depth_m": 1.0}):
        with pytest.raises(SpecificationError):
            FluidLayerSpec(**bad)


def test_topography_specifications_reuse_the_existing_presets():
    assert TopographySpec.flat().run_config_entries() == {}
    mountain = TopographySpec.mountain(height_m=1500, lat_deg=10,
                                       lon_deg=-40, width_deg=15)
    entries = mountain.run_config_entries()
    assert entries["topography"] == "mountain"
    assert entries["mountain_height_m"] == 1500.0
    assert entries["mountain_width_deg"] == 15.0
    cone = TopographySpec.williamson5_cone()
    assert cone.parameters == {"height_m": 2000.0}
    for spec in (TopographySpec.flat(), mountain, cone):
        assert TopographySpec.from_dict(
            _strict_round_trip(spec.to_dict())) == spec
    with pytest.raises(SpecificationError, match="must be one of"):
        TopographySpec("fractal", {})
    with pytest.raises(SpecificationError, match="missing"):
        TopographySpec("mountain", {"height_m": 1.0})
    with pytest.raises(SpecificationError, match="no parameters"):
        TopographySpec("flat", {"height_m": 1.0})
    with pytest.raises(SpecificationError, match="lat_deg"):
        TopographySpec.mountain(height_m=1, lat_deg=91, lon_deg=0,
                                width_deg=10)


def test_discretization_round_trip_and_validation():
    spec = DiscretizationSpec.gauss_latlon(21, 32, 64)
    assert spec.grid == "latlon"
    assert DiscretizationSpec(lmax=5, grid="gauss-latlon").grid == "latlon"
    assert DiscretizationSpec.from_dict(_strict_round_trip(spec.to_dict())) \
        == spec
    assert spec.run_config_entries() == {
        "lmax": 21, "grid": "latlon", "resolution": 4, "nlat": 32,
        "nlon": 64, "product_quadrature": "fine"}
    with pytest.raises(SpecificationError, match="grid"):
        DiscretizationSpec(lmax=5, grid="cubed-sphere")
    with pytest.raises(SpecificationError, match="lmax"):
        DiscretizationSpec(lmax=0)
    with pytest.raises(SpecificationError, match="integer"):
        DiscretizationSpec(lmax=5.0)
    with pytest.raises(SpecificationError, match="nlat"):
        DiscretizationSpec(lmax=5, nlat=1, nlon=8)
    with pytest.raises(SpecificationError, match="product_quadrature"):
        DiscretizationSpec(lmax=5, product_quadrature="exact")


# ---------------------------------------------------------------------------
# Vocabulary kept in sync with the modules this one may not import
# ---------------------------------------------------------------------------

def _literal(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_specification_vocabulary_matches_its_sources():
    from tropoi.run.bve import config as bve_config
    from tropoi.run.swe import config as swe_config
    from tropoi.spatial import specification as spec

    assert spec.GRID_TYPES == bve_config.GRID_TYPES
    assert spec.PRODUCT_QUADRATURES == bve_config.PRODUCT_QUADRATURES
    assert (spec.MIN_NLAT, spec.MIN_NLON) == (bve_config.MIN_NLAT,
                                              bve_config.MIN_NLON)
    assert spec.MAX_MOUNTAIN_HEIGHT_M == swe_config.MAX_MOUNTAIN_HEIGHT_M
    topography = SRC / "spatial" / "terrain" / "topography.py"
    assert spec.TOPOGRAPHY_SPEC_PRESETS == _literal(topography,
                                                    "TOPOGRAPHY_PRESETS")
    assert spec.MAX_MOUNTAIN_HEIGHT_M == _literal(topography,
                                                  "MAX_MOUNTAIN_HEIGHT_M")
    assert (spec.EARTH_MASS_KG, spec.EARTH_RADIUS_M) == (5.972e24, 6.371e6)
