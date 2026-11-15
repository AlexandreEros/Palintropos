"""The two deprecated numerical paths, and the one rule that keeps
deprecating them from rewriting history.

`"separable"` is the default evaluator of the spherical-harmonic basis and
the geodesic grid is deprecated, but both legacy paths stay reachable: the
dense evaluator is the numerical reference, the fallback for grids that
cannot factorize, and the arithmetic every run recorded before the choice
existed. So the contract is "loud, not gone" -- and a replay must follow
the evaluator a run recorded rather than today's default, or changing the
default would silently re-derive every saved run's diagnostics on
different numerics.
"""
from __future__ import annotations

import warnings

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


# ---------------------------------------------------------------------------
# the deprecations are audible
# ---------------------------------------------------------------------------

@needs_cuda
def test_the_geodesic_grid_warns_that_it_is_deprecated():
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    params = PlanetaryParameters.from_earth_like(day_hours=24.0,
                                                 radius_earth_units=1.0)
    with pytest.warns(DeprecationWarning, match="DEPRECATED: the geodesic grid"):
        Planet.generate(params=params, grid_resolution=1, l_max=4,
                        grid_type="geodesic")


@needs_cuda
def test_a_lat_lon_grid_that_cannot_factorize_still_reports_the_dense_path():
    """The involuntary fallback is as deprecated as a requested dense run:
    the point is to explain a slow step, not to assign blame."""
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)
    # nlon = 40 with l_max = 22 fails 2*l_max < nlon, so separable cannot
    # be used however it is asked for.
    grid = GaussLatLonGridGeometry(32, 40, radius=6.371e6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sh = GaussLatLonSphericalHarmonics(grid, 22,
                                           transform_backend="separable")
    assert sh.transform_backend == "dense"
    assert any(issubclass(w.category, DeprecationWarning)
               and "DEPRECATED" in str(w.message) for w in caught)
    assert any("not applicable" in str(w.message) for w in caught)


@needs_cuda
def test_the_default_lat_lon_pairing_is_quiet():
    """A default run must not warn, or the deprecations become noise."""
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)
    grid = GaussLatLonGridGeometry(32, 64, radius=6.371e6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sh = GaussLatLonSphericalHarmonics(grid, 21)
    assert sh.transform_backend == "separable"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# a replay follows the run, not the default (CPU)
# ---------------------------------------------------------------------------

class _Storage:
    """Just the two attributes `build_resources` reads."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.solver = "bve"
        self.run_config = {}


@pytest.mark.parametrize("recorded", ["dense", "separable"])
def test_a_replay_uses_the_evaluator_the_manifest_records(recorded):
    from tropoi.representation.visual.snapshot import _recorded_transform_backend
    storage = _Storage({"numerics": {"transform_backend": recorded}})
    assert _recorded_transform_backend(storage) == recorded


@pytest.mark.parametrize("manifest", [
    {},                                     # no numerics block at all
    {"numerics": None},                     # manifest written without one
    {"numerics": {}},                       # numerics, but from before the choice
    {"numerics": {"product_quadrature": "fine"}},
])
def test_a_run_recorded_before_the_choice_replays_on_the_dense_path(manifest):
    """Runs predating the evaluator choice were necessarily dense, so that
    is what reproducing their diagnostics needs -- the same legacy
    inference as the missing ``grid`` key in temporal.models."""
    from tropoi.representation.visual.snapshot import _recorded_transform_backend
    assert _recorded_transform_backend(_Storage(manifest)) == "dense"
