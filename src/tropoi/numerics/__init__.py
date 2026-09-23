"""Compatibility package: the numerics moved to ``tropoi.spatial``.

Grids live in :mod:`tropoi.spatial.grids`, transforms (and their CUDA kernel
sources) in :mod:`tropoi.spatial.transforms`, operators in
:mod:`tropoi.spatial.operators`, and backends in
:mod:`tropoi.spatial.spherical_backend`. This initializer keeps the historical
eager re-exports (same objects); each ``tropoi.numerics.<module>`` path is an
alias of its canonical module.
"""
from tropoi.spatial.grids.grid_base import GridGeometry as GridGeometryBase
from tropoi.spatial.grids.grid import LatLonGridGeometry
from tropoi.spatial.grids.geodesic_grid import GeodesicGridGeometry
GridGeometry = GeodesicGridGeometry
from tropoi.spatial.grids.integration import simpson_2d
from tropoi.spatial.transforms.spherical_harmonics import LatLonSphericalHarmonics
from tropoi.spatial.grids.latlon_grid import GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics
from tropoi.spatial.transforms.fast_geodesic_sh import PointSetSphericalHarmonics
from tropoi.spatial.transforms.optimized_geodesic_sh import GeodesicSphericalHarmonics, OptimizedGeodesicSH
from tropoi.spatial.spherical_backend import (
    GeodesicBackend,
    LatLonBackend,
    PointSetBackend,
    ProductSpace,
    SphericalGridBackend,
    make_backend,
)
from tropoi.spatial.operators.spectral_operators import SpectralOperators
from tropoi.spatial.grids.grid_interpolation import geodesic_to_latlon_grid, latlon_to_geodesic_grid

__all__ = [
    "GridGeometryBase",
    "LatLonGridGeometry",
    "GeodesicGridGeometry",
    "GridGeometry",
    "simpson_2d",
    "LatLonSphericalHarmonics",
    "GaussLatLonGridGeometry",
    "GaussLatLonSphericalHarmonics",
    "PointSetSphericalHarmonics",
    "GeodesicSphericalHarmonics",
    "OptimizedGeodesicSH",
    "SphericalGridBackend",
    "GeodesicBackend",
    "LatLonBackend",
    "PointSetBackend",
    "ProductSpace",
    "make_backend",
    "SpectralOperators",
    "geodesic_to_latlon_grid",
    "latlon_to_geodesic_grid",
]
