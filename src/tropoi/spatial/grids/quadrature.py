"""Horizontal quadrature on the sphere: Gauss-Legendre and regular
latitude-longitude column sets, and diagnostic point collections.

NumPy only. This module never imports CuPy and never performs a spherical
harmonic transform, so CPU-only code (the airless runner, archive readers,
diagnostics) can use exactly the nodes, ordering and weights of the
atmospheric grids without initializing CUDA.

**The one Gauss-Legendre construction.** :func:`gauss_legendre` is the
project's only call of ``numpy.polynomial.legendre.leggauss`` for a horizontal grid
(the g-space quadrature of ``physics.radiation.kdist`` is a different domain). The spectral
grid (:class:`~tropoi.spatial.grids.latlon_grid.GaussLatLonGridGeometry`),
the Gauss airless columns and the diagnostics all take their nodes from it:

* nodes ``x_i`` are sin(latitude). :func:`gauss_latitude_nodes` orders them
  north to south (``x`` descending), the grid's row order;
* latitudes are ``pi/2 - arccos(x_i)``, the grid's expression;
* longitudes are ``2 pi j / nlon``, ``j = 0 .. nlon-1``, starting at 0;
* points are the lat-major C-order flattening of ``(nlat, nlon)``;
* physical cell weights are ``W_ij = R^2 (2 pi / nlon) w_i``. Because
  ``sum_i w_i = 2``, they sum to ``4 pi R^2``.

**Regular grids** (:meth:`ColumnSet.regular`) are cell-centred, with
latitudes from south to north and longitudes ``(j + 1/2) 360/nlon`` degrees
east. Their weights are the exact cell areas
``R^2 dlambda (sin phi_N - sin phi_S)``.

**Diagnostic collections** (:meth:`ColumnSet.points`) are independent sample
locations. They have no physical area and no weights, so nothing global may
be computed from them.

Coordinates are held in radians *and* degrees, each constructed from the
primary form of its grid. A Gauss column's radians are therefore the grid's
radians exactly, not a round trip through degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["ColumnSet", "gauss_legendre", "gauss_latitude_nodes", "gauss_latitudes",
           "gauss_longitudes"]

#: Column-set kinds: a global discretization with physical area weights, or
#: a diagnostic collection without.
KINDS = ("global", "diagnostic")


def gauss_legendre(n: int) -> tuple:
    """Gauss-Legendre nodes and weights on [-1, 1], nodes ascending (the
    NumPy convention). The weights sum to 2."""
    n = int(n)
    if n < 1:
        raise ValueError(f"need at least one node, got {n}")
    return np.polynomial.legendre.leggauss(n)


def gauss_latitude_nodes(nlat: int) -> tuple:
    """(x, w): the nodes x = sin(latitude) and their weights, north to south
    (x descending), the row order of the Gauss grid."""
    x, w = gauss_legendre(nlat)
    order = np.argsort(-x)
    return x[order], w[order]


def gauss_latitudes(nlat: int) -> np.ndarray:
    """Gauss latitudes (rad), north to south: ``pi/2 - arccos(x)``."""
    x, _ = gauss_latitude_nodes(nlat)
    return np.pi / 2.0 - np.arccos(x)


def gauss_longitudes(nlon: int) -> np.ndarray:
    """Uniform longitudes (rad) from 0, ``2 pi j / nlon``."""
    return np.linspace(0.0, 2.0 * np.pi, int(nlon), endpoint=False)


@dataclass(frozen=True)
class ColumnSet:
    """Columns on a sphere of radius ``radius`` (m): flat per-column
    coordinates, a kind, and physical weights (m^2) or None.

    ``lat``/``lon`` are radians, ``lat_deg``/``lon_deg`` degrees east.
    ``shape`` is ``(nlat, nlon)`` for a grid (the flattening is lat-major C
    order), None for a point collection. ``layout`` states the ordering."""

    lat: np.ndarray
    lon: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    kind: str
    weights: Optional[np.ndarray]
    shape: Optional[tuple]
    layout: str
    radius: float

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if (self.weights is None) != (self.kind == "diagnostic"):
            raise ValueError("a global column set has weights; a diagnostic one has none")
        n = self.lat.size
        if not all(a.shape == (n,) for a in (self.lon, self.lat_deg, self.lon_deg)):
            raise ValueError("coordinates must be flat arrays of one length")
        if self.weights is not None:
            if self.weights.shape != (n,) or not np.all(self.weights > 0.0):
                raise ValueError("weights must be positive, one per column")
            # the weights are physical areas: they must tile the sphere (to
            # the round-off of an n-term sum)
            area = 4.0 * math.pi * self.radius * self.radius
            if abs(float(self.weights.sum()) / area - 1.0) > max(n, 1) * 4.0 * np.finfo(float).eps:
                raise ValueError("global weights do not sum to the sphere's area")

    @property
    def size(self) -> int:
        return int(self.lat.size)

    @classmethod
    def gauss(cls, nlat: int, nlon: int, radius: float) -> "ColumnSet":
        """The Gauss-Legendre grid of the spectral models: the same nodes,
        ordering and weights as ``GaussLatLonGridGeometry(nlat, nlon, radius)``."""
        nlat, nlon = int(nlat), int(nlon)
        if nlat < 2 or nlon < 4:
            raise ValueError(f"a Gauss grid needs nlat >= 2 and nlon >= 4, got ({nlat}, {nlon})")
        _, w = gauss_latitude_nodes(nlat)
        lat = np.repeat(gauss_latitudes(nlat), nlon)
        lon = np.tile(gauss_longitudes(nlon), nlat)
        weights = np.repeat(w, nlon) * (2.0 * np.pi / nlon) * float(radius) ** 2
        return cls(lat, lon, np.degrees(lat), np.degrees(lon), "global", weights, (nlat, nlon),
                   "Gauss-Legendre, lat-major, north to south, longitudes from 0",
                   float(radius))

    @classmethod
    def regular(cls, nlat: int, nlon: int, radius: float) -> "ColumnSet":
        """A regular cell-centred grid with exact cell areas."""
        nlat, nlon = int(nlat), int(nlon)
        if nlat < 1 or nlon < 1:
            raise ValueError(f"a regular grid needs nlat, nlon >= 1, got ({nlat}, {nlon})")
        edges = np.linspace(-90.0, 90.0, nlat + 1)
        lat = 0.5 * (edges[1:] + edges[:-1])
        lon = (np.arange(nlon) + 0.5) * (360.0 / nlon)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        band = np.sin(np.radians(edges[1:])) - np.sin(np.radians(edges[:-1]))
        R = float(radius)
        w = R ** 2 * math.radians(360.0 / nlon) * np.broadcast_to(band[:, None], la.shape)
        lat_deg, lon_deg = la.ravel(), lo.ravel()
        return cls(np.radians(lat_deg), np.radians(lon_deg), lat_deg, lon_deg, "global",
                   np.array(w).ravel(), (nlat, nlon),
                   "regular cell-centred, lat-major, south to north, longitudes (j + 1/2) dlon",
                   R)

    @classmethod
    def points(cls, lat_deg, lon_deg, radius: float) -> "ColumnSet":
        """Independent diagnostic sample locations (degrees): no weights."""
        lat_deg = np.asarray(lat_deg, dtype=float)
        lon_deg = np.asarray(lon_deg, dtype=float)
        if lat_deg.ndim != 1 or lat_deg.shape != lon_deg.shape or not lat_deg.size:
            raise ValueError("points need equally long, non-empty lists of latitudes and longitudes")
        return cls(np.radians(lat_deg), np.radians(lon_deg), lat_deg, lon_deg, "diagnostic",
                   None, None, "diagnostic points in the order given", float(radius))
