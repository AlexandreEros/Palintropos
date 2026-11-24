"""The two deprecated numerical paths: reachable, never implicit, loud.

`"separable"` is the default evaluator of the spherical-harmonic basis and
Gauss lat-lon is the default grid, but both legacy paths stay *reachable*:
the dense evaluator is the numerical reference and the fallback for grids
that cannot factorize, and the geodesic grid carries the resolution-3
product-quadrature characterization and the exact mass-conservation check.
So the contract is "loud, not gone" -- choosing either warns, and so does
falling back to dense involuntarily.

What is deliberately *not* tested here any more: pinning a replay to the
evaluator a run's manifest records. A replay re-derives diagnostics on the
current default; a run that must reproduce bit-for-bit is reproduced from
the commit its manifest records, not by keeping obsolete arithmetic alive
in current code.
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
