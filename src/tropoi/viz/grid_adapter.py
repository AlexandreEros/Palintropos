"""Adapters from simulation samplings to the repository's map-view grid."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tropoi.viz.fields import ScalarGridField

if TYPE_CHECKING:
    from tropoi.spatial.grids.grid import LatLonGridGeometry


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def map_to_uniform_latlon(values, source_grid, *,
                          target_grid: "LatLonGridGeometry | None" = None
                          ) -> "tuple[LatLonGridGeometry, np.ndarray]":
    """Map any repository state sampling to the standard 91x181 view grid.

    The grid/interpolation machinery (``tropoi.spatial.grids``, which imports
    CuPy) is imported here, on use, so host-only compositions that never
    map a field (coefficient-space frames) keep this module importable
    without CUDA.
    """
    from tropoi.spatial.grids.grid import LatLonGridGeometry
    from tropoi.spatial.grids.grid_interpolation import geodesic_to_latlon_grid

    target = target_grid or LatLonGridGeometry.create((91, 181))
    values = _host(values)
    if values.ndim == 2 and values.shape == target.lat_grid.shape:
        return target, values
    mapped = geodesic_to_latlon_grid(
        values, source_grid, target, method="linear")
    if np.isnan(mapped).any():
        nearest = geodesic_to_latlon_grid(
            values, source_grid, target, method="nearest")
        mapped = np.where(np.isnan(mapped), nearest, mapped)
    return target, mapped


def scalar_field_on_uniform_latlon(values, source_grid, *, name: str,
                                   units: str,
                                   times: np.ndarray | None = None
                                   ) -> ScalarGridField:
    """Build a physical-grid field on the established uniform map sampling."""
    target, mapped = map_to_uniform_latlon(values, source_grid)
    return ScalarGridField(
        mapped, _host(target.latitudes), _host(target.longitudes),
        name=name, units=units, times=times)
