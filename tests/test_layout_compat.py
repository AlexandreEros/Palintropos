"""The spatial / temporal / representation layout and its compatibility paths.

Sprint 3 moved every implementation module under ``tropoi.spatial``,
``tropoi.temporal`` or ``tropoi.representation``. The pre-move import paths
remain valid for the 0.1 series as ``sys.modules`` aliases: the old name IS
the new module object, so classes, functions, module state and monkeypatch
targets are identical under both names. These tests pin that contract, the
dependency direction between the three layers, and the packaged CUDA
sources. Import-only identity checks need CuPy importable, not a device.
"""
from __future__ import annotations

import ast
import importlib
import importlib.resources
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "tropoi"

#: Every relocated module: legacy import path -> canonical module.
MOVED_MODULES = {
    # spatial: truncation policy, prescribed environment, planet facade
    "tropoi.support": "tropoi.spatial.truncation",
    "tropoi.planet.planetary_parameters": "tropoi.spatial.environment",
    "tropoi.planet.planet": "tropoi.spatial.planet",
    "tropoi.planet.elevation_data": "tropoi.spatial.terrain.elevation_data",
    "tropoi.planet.terrain_spectral": "tropoi.spatial.terrain.terrain_spectral",
    "tropoi.planet.tectonics": "tropoi.spatial.terrain.tectonics",
    "tropoi.physics.topography": "tropoi.spatial.terrain.topography",
    "tropoi.physics.sigma_coordinate": "tropoi.spatial.sigma_coordinate",
    # spatial: grids, transforms, backends, operators
    "tropoi.numerics.grid_base": "tropoi.spatial.grids.grid_base",
    "tropoi.numerics.grid": "tropoi.spatial.grids.grid",
    "tropoi.numerics.geodesic_grid": "tropoi.spatial.grids.geodesic_grid",
    "tropoi.numerics.latlon_grid": "tropoi.spatial.grids.latlon_grid",
    "tropoi.numerics.cartesian_to_spherical": "tropoi.spatial.grids.cartesian_to_spherical",
    "tropoi.numerics.integration": "tropoi.spatial.grids.integration",
    "tropoi.numerics.grid_interpolation": "tropoi.spatial.grids.grid_interpolation",
    "tropoi.numerics.spherical_harmonics": "tropoi.spatial.transforms.spherical_harmonics",
    "tropoi.numerics.fast_geodesic_sh": "tropoi.spatial.transforms.fast_geodesic_sh",
    "tropoi.numerics.optimized_geodesic_sh": "tropoi.spatial.transforms.optimized_geodesic_sh",
    "tropoi.numerics.compute_optimal_weights": "tropoi.spatial.transforms.compute_optimal_weights",
    "tropoi.numerics.cuda.cuda_utils": "tropoi.spatial.transforms.cuda.cuda_utils",
    "tropoi.numerics.spherical_backend": "tropoi.spatial.spherical_backend",
    "tropoi.numerics.spectral_operators": "tropoi.spatial.operators.spectral_operators",
    "tropoi.numerics.differential_operators_spherical": "tropoi.spatial.operators.differential_operators_spherical",
    # spatial: initialization
    "tropoi.run.bve.initial_conditions": "tropoi.spatial.initialization.bve",
    "tropoi.run.swe.initial_conditions": "tropoi.spatial.initialization.swe",
    "tropoi.run.pe.initial_conditions": "tropoi.spatial.initialization.pe",
    # temporal: integration and tendencies
    "tropoi.run.engine": "tropoi.temporal.integration",
    "tropoi.physics.barotropic": "tropoi.temporal.tendencies.barotropic",
    "tropoi.physics.shallow_water": "tropoi.temporal.tendencies.shallow_water",
    "tropoi.physics.primitive_equations": "tropoi.temporal.tendencies.primitive_equations",
    # representation: diagnostics, archive writer, visualization
    "tropoi.run.bve.diagnostics": "tropoi.representation.diagnostics.bve",
    "tropoi.run.swe.diagnostics": "tropoi.representation.diagnostics.swe",
    "tropoi.run.pe.diagnostics": "tropoi.representation.diagnostics.pe",
    "tropoi.run.bve.io": "tropoi.representation.archive.writer",
    "tropoi.run.bve.visualization": "tropoi.representation.visual.bve",
    "tropoi.run.swe.visualization": "tropoi.representation.visual.swe",
    "tropoi.run.pe.visualization": "tropoi.representation.visual.pe",
    "tropoi.run.pe.snapshot_visualization": "tropoi.representation.visual.pe_snapshots",
    "tropoi.viz.complex_encoding": "tropoi.representation.visual.complex_encoding",
    "tropoi.viz.fields": "tropoi.representation.visual.fields",
    "tropoi.viz.grid_adapter": "tropoi.representation.visual.grid_adapter",
    "tropoi.viz.maps": "tropoi.representation.visual.maps",
    "tropoi.viz.matplotlib_renderer": "tropoi.representation.visual.matplotlib_renderer",
    "tropoi.viz.normalization": "tropoi.representation.visual.normalization",
    "tropoi.viz.planet_viewer": "tropoi.representation.visual.planet_viewer",
    "tropoi.viz.renderers": "tropoi.representation.visual.renderers",
    "tropoi.viz.specs": "tropoi.representation.visual.specs",
    "tropoi.viz.timeline": "tropoi.representation.visual.timeline",
    "tropoi.viz.vorticity_viewer": "tropoi.representation.visual.vorticity_viewer",
}

LEGACY_PACKAGES = ("tropoi.numerics", "tropoi.physics", "tropoi.planet",
                   "tropoi.viz")
#: Every compatibility-only import path: the aliases, the legacy packages,
#: and the older BVE re-export module kept since the physics split.
LEGACY_NAMES = sorted([*MOVED_MODULES, *LEGACY_PACKAGES,
                       "tropoi.run.bve.barotropic_vorticity"])

#: Legacy package initializers / modules that re-export public objects:
#: name -> (canonical module, canonical name if different).
LEGACY_EXPORTS = {
    "tropoi.numerics": {
        "GridGeometryBase": ("tropoi.spatial.grids.grid_base", "GridGeometry"),
        "LatLonGridGeometry": ("tropoi.spatial.grids.grid", None),
        "GeodesicGridGeometry": ("tropoi.spatial.grids.geodesic_grid", None),
        "GridGeometry": ("tropoi.spatial.grids.geodesic_grid", "GeodesicGridGeometry"),
        "simpson_2d": ("tropoi.spatial.grids.integration", None),
        "LatLonSphericalHarmonics": ("tropoi.spatial.transforms.spherical_harmonics", None),
        "GaussLatLonGridGeometry": ("tropoi.spatial.grids.latlon_grid", None),
        "GaussLatLonSphericalHarmonics": ("tropoi.spatial.grids.latlon_grid", None),
        "PointSetSphericalHarmonics": ("tropoi.spatial.transforms.fast_geodesic_sh", None),
        "GeodesicSphericalHarmonics": ("tropoi.spatial.transforms.optimized_geodesic_sh", None),
        "OptimizedGeodesicSH": ("tropoi.spatial.transforms.optimized_geodesic_sh", None),
        "SphericalGridBackend": ("tropoi.spatial.spherical_backend", None),
        "GeodesicBackend": ("tropoi.spatial.spherical_backend", None),
        "LatLonBackend": ("tropoi.spatial.spherical_backend", None),
        "PointSetBackend": ("tropoi.spatial.spherical_backend", None),
        "ProductSpace": ("tropoi.spatial.spherical_backend", None),
        "make_backend": ("tropoi.spatial.spherical_backend", None),
        "SpectralOperators": ("tropoi.spatial.operators.spectral_operators", None),
        "geodesic_to_latlon_grid": ("tropoi.spatial.grids.grid_interpolation", None),
        "latlon_to_geodesic_grid": ("tropoi.spatial.grids.grid_interpolation", None),
    },
    "tropoi.planet": {
        "Planet": ("tropoi.spatial.planet", None),
        "PlanetaryParameters": ("tropoi.spatial.environment", None),
        "ElevationData": ("tropoi.spatial.terrain.elevation_data", None),
    },
    "tropoi.viz": {
        "FigureSpec": ("tropoi.representation.visual.specs", None),
        "FigureTimeline": ("tropoi.representation.visual.timeline", None),
        "NormalizationPolicy": ("tropoi.representation.visual.normalization", None),
        "ScalarGridField": ("tropoi.representation.visual.fields", None),
        "get_default_renderer": ("tropoi.representation.visual.renderers", None),
        "render_snapshot_product": ("tropoi.representation.visual.timeline", None),
        "phase_magnitude_hsv": ("tropoi.representation.visual.complex_encoding", None),
        "PlanetViewer": ("tropoi.representation.visual.planet_viewer", None),
    },
    "tropoi.run.bve": {
        "BarotropicVorticity": ("tropoi.temporal.tendencies.barotropic", None),
        "BarotropicState": ("tropoi.spatial.states.barotropic", None),
        "run_bve": ("tropoi.run.bve.runner", None),
    },
    "tropoi.run.bve.barotropic_vorticity": {
        "BarotropicVorticity": ("tropoi.temporal.tendencies.barotropic", None),
        "BarotropicState": ("tropoi.spatial.states.barotropic", None),
    },
    # names split out of a moved module, or relocated out of a config
    "tropoi.physics.shallow_water": {
        "ShallowWaterState": ("tropoi.spatial.states.shallow_water", None),
        "ShallowWaterStateError": ("tropoi.spatial.states.shallow_water", None),
        "ShallowWaterModel": ("tropoi.temporal.tendencies.shallow_water", None),
    },
    "tropoi.physics.primitive_equations": {
        "PrimitiveEquationsState": ("tropoi.spatial.states.primitive_equations", None),
        "PrimitiveEquationsStateError": ("tropoi.spatial.states.primitive_equations", None),
        "isothermal_rest_state": ("tropoi.spatial.states.primitive_equations", None),
        "PrimitiveEquationsModel": ("tropoi.temporal.tendencies.primitive_equations", None),
        "product_truncation_cut": ("tropoi.spatial.truncation", None),
    },
    "tropoi.run.swe.config": {
        "require_scenario_support": ("tropoi.spatial.truncation", None),
        "SWE_SCENARIO_SUPPORT": ("tropoi.spatial.truncation", None),
        "W5_U0_MS": ("tropoi.spatial.williamson5", None),
        "W5_MEAN_DEPTH_M": ("tropoi.spatial.williamson5", None),
    },
    "tropoi.run.pe.config": {
        "require_thermal_wave_support": ("tropoi.spatial.truncation", None),
    },
}


def _cupy_importable() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


needs_cupy = pytest.mark.skipif(not _cupy_importable(),
                                reason="CuPy is not importable")


@needs_cupy
@pytest.mark.parametrize("old,new", sorted(MOVED_MODULES.items()))
def test_legacy_module_path_is_the_canonical_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


@needs_cupy
@pytest.mark.parametrize("package", sorted(LEGACY_EXPORTS))
def test_legacy_exports_are_the_canonical_objects(package):
    legacy = importlib.import_module(package)
    for name, (module, canonical_name) in LEGACY_EXPORTS[package].items():
        canonical = getattr(importlib.import_module(module),
                            canonical_name or name)
        assert getattr(legacy, name) is canonical, f"{package}.{name}"


def test_every_legacy_module_file_is_only_an_alias():
    # One canonical implementation: an old module file holds a docstring and
    # the alias call, never code that could drift from the canonical module.
    for old, new in MOVED_MODULES.items():
        path = SRC.parent / pathlib.Path(*old.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = [node for node in tree.body
                if not (isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant))]
        assert [ast.unparse(node) for node in body] == [
            "from tropoi._compat import alias_module",
            f"alias_module(__name__, '{new}')"], old


@needs_cupy
def test_canonical_modules_never_import_legacy_paths():
    # Import every canonical module in a fresh interpreter: none of the old
    # module names or legacy packages may appear, so no new module depends
    # back on a compatibility path.
    probe = (
        "import importlib, pkgutil, sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        f"legacy = {LEGACY_NAMES!r}\n"
        "for pkg in ('tropoi.spatial', 'tropoi.temporal', "
        "'tropoi.representation', 'tropoi.run', 'tropoi.cli'):\n"
        "    mod = importlib.import_module(pkg)\n"
        "    for info in pkgutil.walk_packages(mod.__path__, pkg + '.'):\n"
        "        if info.name not in legacy:\n"
        "            importlib.import_module(info.name)\n"
        "importlib.import_module('tropoi.spatial.transforms.cuda.cuda_utils')\n"
        "loaded = [m for m in legacy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def _imports(path: pathlib.Path):
    """Every absolute import in a source file, TYPE_CHECKING blocks excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            skip.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


#: Layer -> the tropoi layers it may import (eagerly or lazily).
ALLOWED = {
    "tropoi.spatial": ("tropoi.spatial",),
    "tropoi.temporal": ("tropoi.spatial", "tropoi.temporal"),
    "tropoi.representation": ("tropoi.spatial", "tropoi.temporal",
                              "tropoi.representation"),
}
#: Deliberate, documented exceptions (docs/ARCHITECTURE.md).
EXCEPTIONS = {
    # the run-id hash consumes the stdlib-only configuration layer's
    # scientific subset
    ("tropoi.representation.archive.writer", "tropoi.run.bve.config"),
    # Snapshot.plot rebuilds model resources lazily through the solver
    # modules' own builders (orchestration owns model construction)
    ("tropoi.representation.visual.snapshot", "tropoi.cli.bve"),
    ("tropoi.representation.visual.snapshot", "tropoi.cli.swe"),
    ("tropoi.representation.visual.snapshot", "tropoi.cli.pe"),
}


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_dependency_direction(layer):
    root = SRC / layer.split(".", 1)[1]
    violations = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(SRC.parent).with_suffix("")
        module = ".".join(p for p in rel.parts if p != "__init__")
        for target in _imports(path):
            if not target.startswith("tropoi.") or target == "tropoi._compat":
                continue
            if target.startswith(ALLOWED[layer]):
                continue
            if (module, target) not in EXCEPTIONS:
                violations.append((module, target))
    assert not violations, violations


def test_cuda_kernel_sources_are_package_resources():
    sources = importlib.resources.files("tropoi.spatial.transforms.cuda")
    names = {entry.name for entry in sources.iterdir()}
    assert {"legendre.cu", "sh_matrix.cu", "sh_matrix_real.cu",
            "sph_harm.cu"} <= names
    # the legacy directory keeps only the alias, never a second copy
    assert not list((SRC / "numerics" / "cuda").glob("*.cu"))


@needs_cupy
def test_cuda_sources_load_through_old_and_new_paths():
    from tropoi.numerics.cuda.cuda_utils import load_cuda_source as legacy
    from tropoi.spatial.transforms.cuda.cuda_utils import load_cuda_source
    assert legacy is load_cuda_source
    assert "__global__" in load_cuda_source("legendre")


def test_pyproject_packages_the_relocated_kernels():
    text = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"tropoi.spatial.transforms.cuda" = ["*.cu"]' in text
