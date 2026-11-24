"""Corrected structured lat-lon grid: Gauss-Legendre latitudes, uniform
periodic longitudes, and an exact-quadrature SH transform.

Why "corrected": the legacy pairing (`grid.LatLonGridGeometry` +
`spherical_harmonics.LatLonSphericalHarmonics`) integrates with Simpson
panels over a longitude axis built with ``endpoint=False`` — the closing
panel [2pi-dl, 2pi) is missing entirely, biasing every analysis low by
~1/nlon — and uses equiangular-latitude Simpson quadrature with explicit
pole rows (where sin(colat)=0 wastes samples and the panel rule is only
O(h^4) accurate). This module replaces (not wraps) that scheme:

* latitude nodes/weights are Gauss-Legendre in sin(lat): exact for
  polynomial integrands up to degree 2*nlat-1,
* longitudes are uniform with equal weights 2*pi/nlon: exact for
  trigonometric polynomials of zonal degree < nlon (the correct quadrature
  for a periodic axis — no panels, no seam),

so analysis of a band-limited field of degree L against Y_lm (degree
l <= l_max) is EXACT whenever  L + l_max <= 2*nlat - 1  and
L + l_max < nlon. For the state grid (L = l_max) that is
nlat >= l_max + 1, nlon >= 2*l_max + 1; a product grid sized by the
3/2 rule (nlat >= ceil((3*l_max+1)/2), nlon >= 3*l_max + 1) integrates
quadratic products exactly.

The transform itself is `PointSetSphericalHarmonics` (the same GPU matrix
path the geodesic backend uses) fed with the tensor-product quadrature
weights; fields are flat (n_points,) arrays in lat-major order, matching the
unstructured convention used throughout the operators.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import cupy as cp

from tropoi.spatial.grids.grid_base import GridGeometry
from tropoi.spatial.transforms.fast_geodesic_sh import PointSetSphericalHarmonics
from tropoi.spatial.transforms.separable_gauss_sh import (
    SeparableGaussSphericalHarmonics, separable_is_applicable)

__all__ = ["GaussLatLonGridGeometry", "GaussLatLonSphericalHarmonics",
           "TRANSFORM_BACKENDS", "default_transform_backend"]

#: Evaluators this grid can pair with. ``"separable"`` is the FFT +
#: Legendre path and the default: 16x a dense T42 step on the reference
#: device. ``"dense"`` is the historical
#: :class:`PointSetSphericalHarmonics` basis-matrix path -- **deprecated**,
#: and retained as the numerical reference, as the fallback for grids that
#: cannot factorize, and to reproduce runs recorded before the evaluator
#: became selectable. The two agree only to round-off (different summation
#: order), so they are interchangeable for new work but not for
#: reproducing a pinned hash.
TRANSFORM_BACKENDS = ("dense", "separable")

_BACKEND_ENV = "TROPOI_SH_BACKEND"

#: Emitted whenever a Gauss lat-lon grid ends up on the dense evaluator,
#: whether a caller asked for it or the separable path was inapplicable.
_DENSE_DEPRECATED = (
    "DEPRECATED: the dense basis-matrix evaluator "
    "(transform_backend='dense') is deprecated on Gauss lat-lon grids. It "
    "builds an n_points x n_basis complex matrix and spends one GEMM per "
    "transform, which measured ~96% of a synchronized T42 step; the "
    "separable FFT+Legendre evaluator is the default and ~16x faster on "
    "the same step. Dense is kept as the numerical reference and as the "
    "fallback for grids that cannot factorize; the two agree to round-off "
    "but not bit-for-bit, so a run that must reproduce bit-for-bit is "
    "reproduced from the commit its manifest records.")


def default_transform_backend() -> str:
    """Backend used when a caller does not name one.

    ``"separable"`` unless the ``TROPOI_SH_BACKEND`` environment variable
    says otherwise. The variable exists so an existing run configuration
    can be pointed at either evaluator without editing call sites;
    whichever is chosen is recorded in the run manifest by
    :meth:`SphericalGridBackend.describe` and in every checkpoint by
    ``run.held_suarez.experiment.transform_identity``, so a run never
    silently loses track of which numerics produced it.

    This default changed with the separable evaluator: runs made before it
    were necessarily dense, which is why replaying a saved run pins the
    evaluator its manifest records rather than taking this default (see
    ``representation.visual.snapshot``).
    """
    value = os.environ.get(_BACKEND_ENV, "separable").strip().lower()
    if value not in TRANSFORM_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV}={value!r} is not one of {TRANSFORM_BACKENDS} "
            "(no silent fallback)")
    return value


class GaussLatLonGridGeometry(GridGeometry):
    """Structured Gauss-Legendre x uniform-longitude grid on the sphere.

    Latitude rows run north -> south (colatitude ascending); per-point
    arrays are the C-order flattening of the (nlat, nlon) grid.
    """

    def __init__(self, nlat: int, nlon: int, radius: float = 1.0):
        if nlat < 2 or nlon < 4:
            raise ValueError(
                f"GaussLatLonGridGeometry needs nlat >= 2 and nlon >= 4, "
                f"got ({nlat}, {nlon})")
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.radius = float(radius)

        # Gauss-Legendre nodes x_j = cos(colat_j) and weights on [-1, 1].
        x, w_gl = np.polynomial.legendre.leggauss(self.nlat)
        order = np.argsort(-x)              # x descending == colat ascending
        x, w_gl = x[order], w_gl[order]
        colat = np.arccos(x)                # (nlat,), in (0, pi): no poles
        self._latitudes_np = np.pi / 2.0 - colat
        self._longitudes_np = np.linspace(0.0, 2.0 * np.pi, self.nlon,
                                          endpoint=False)
        self._gl_weights = w_gl             # sum == 2 exactly

        self._latitudes = cp.asarray(self._latitudes_np)
        self._longitudes = cp.asarray(self._longitudes_np)

        # Per-point arrays: lat-major C-order flattening.
        self._point_latitudes = cp.asarray(
            np.repeat(self._latitudes_np, self.nlon))
        self._point_longitudes = cp.asarray(
            np.tile(self._longitudes_np, self.nlat))

        self.coslat = cp.cos(self._point_latitudes)
        self.sinlat = cp.sin(self._point_latitudes)

        # Tensor-product solid-angle weights: w_j * (2*pi/nlon), sum == 4*pi.
        w_point = np.repeat(w_gl, self.nlon) * (2.0 * np.pi / self.nlon)
        self.solid_angle_weights = cp.asarray(w_point)

    # -- GridGeometry interface ---------------------------------------------

    @property
    def latitudes(self) -> cp.ndarray:
        return self._latitudes

    @property
    def longitudes(self) -> cp.ndarray:
        return self._longitudes

    @property
    def point_latitudes(self) -> cp.ndarray:
        return self._point_latitudes

    @property
    def point_longitudes(self) -> cp.ndarray:
        return self._point_longitudes

    @property
    def n_points(self) -> int:
        return self.nlat * self.nlon

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)

    # -- quadrature / metric ---------------------------------------------------

    @property
    def cell_areas(self) -> cp.ndarray:
        """Physical per-point areas (solid angle x R^2); same role as the
        geodesic grid's Voronoi cell areas (viz weighting, diagnostics)."""
        return self.solid_angle_weights * self.radius**2

    @property
    def cfl_length_scale(self) -> float:
        """Minimum meridional node spacing, R * min(dcolat).

        The zonal direction imposes no tighter constraint for a spectral
        transform method: the zonal wavenumber of any representable field is
        capped by l_max, so the resolved zonal wavelength *per grid point*
        (2*pi*R*cos(lat)/m divided by nlon points at that latitude) is
        latitude-independent — pole-ward metric shrinkage of dx is matched by
        the shrinkage of the fastest representable zonal feature. GL colatitude
        nodes are near-uniform (theta_j ~ (j - 1/4)*pi/(nlat + 1/2)), so this
        scale is ~ pi*R/nlat, not pathologically small at the poles.

        This is a grid proxy for the fundamental spectral scale ~R/l_max
        (measured ~2x it at nlat~l_max). It is conservative in the resolved
        regime the transform recommends -- nlat >= l_max+1, which
        GaussLatLonSphericalHarmonics *warns* about but does not enforce: if a
        caller ignores that warning and runs nlat < l_max+1, pi*R/nlat can
        exceed R/l_max and this proxy is no longer guaranteed conservative
        (the transform is also inexact there). Oversampling (nlat >> l_max)
        stays safe but shrinks the proxy, making the timestep needlessly small.
        """
        colat = np.pi / 2.0 - self._latitudes_np
        return self.radius * float(np.min(np.abs(np.diff(colat))))


class GaussLatLonSphericalHarmonics:
    """SH analysis/synthesis on a GaussLatLonGridGeometry.

    Thin pairing of the grid's exact tensor-product quadrature with one of
    two GPU evaluators. Fields are flat (n_points,) arrays in the geometry's
    lat-major point order; 2D (nlat, nlon) inputs to `transform` are accepted
    and flattened.

    Parameters
    ----------
    transform_backend : {"dense", "separable"}, optional
        Which evaluator to pair with. ``"dense"`` is the historical
        `PointSetSphericalHarmonics` basis-matrix path: one n_points x
        n_basis complex matrix product per transform, the numerical
        reference, and the default. ``"separable"`` is
        `SeparableGaussSphericalHarmonics`: a cuFFT longitude stage plus a
        custom-kernel Legendre stage, which drops the nlon factor from the
        dominant stage and agrees with the dense path to round-off. Defaults
        to :func:`default_transform_backend`.

        ``"separable"`` falls back to the dense path, with a warning naming
        the reason, when this (grid, l_max) pairing cannot use it -- chiefly
        a longitude axis too coarse for the truncation (2*l_max >= nlon).

        Landing on ``"dense"`` raises a ``DeprecationWarning`` either way,
        asked for or fallen back to, so a slow run is never quiet about
        why.
    """

    def __init__(self, grid: GaussLatLonGridGeometry, l_max: int, *,
                 transform_backend: str | None = None):
        if not isinstance(grid, GaussLatLonGridGeometry):
            raise ValueError(
                f"GaussLatLonSphericalHarmonics requires a "
                f"GaussLatLonGridGeometry, got {type(grid).__name__}")
        self.grid = grid
        self.l_max = int(l_max)

        requested = (default_transform_backend() if transform_backend is None
                     else str(transform_backend).strip().lower())
        if requested not in TRANSFORM_BACKENDS:
            raise ValueError(
                f"transform_backend must be one of {TRANSFORM_BACKENDS}, got "
                f"{transform_backend!r} (no silent fallback)")

        # Exactness envelope for analysis of degree-l_max fields.
        if grid.nlat < l_max + 1 or grid.nlon < 2 * l_max + 1:
            warnings.warn(
                f"Gauss lat-lon SH is under-resolved: ({grid.nlat}, "
                f"{grid.nlon}) grid for l_max={l_max}. Analysis is exact "
                f"only for nlat >= {l_max + 1} and nlon >= {2 * l_max + 1}; "
                f"below that, round trips leak energy across modes.",
                stacklevel=2,
            )

        if requested == "separable":
            ok, reason = separable_is_applicable(
                grid.nlat, grid.nlon, self.l_max, grid.solid_angle_weights)
            if not ok:
                warnings.warn(
                    f"transform_backend='separable' requested but not "
                    f"applicable to this ({grid.nlat}, {grid.nlon}) grid at "
                    f"l_max={self.l_max}: {reason}. Falling back to the dense "
                    f"basis-matrix evaluator.",
                    stacklevel=2,
                )
                requested = "dense"
        self.transform_backend = requested

        # Deliberately after the fallback: an involuntary dense run is as
        # worth reporting as a requested one.
        if requested == "dense":
            warnings.warn(_DENSE_DEPRECATED, DeprecationWarning, stacklevel=2)

        if requested == "separable":
            self.sh = SeparableGaussSphericalHarmonics(
                grid.latitudes, grid.nlon, self.l_max,
                weights=grid.solid_angle_weights,
            )
        else:
            self.sh = PointSetSphericalHarmonics(
                grid.point_latitudes,
                grid.point_longitudes,
                self.l_max,
                weights=grid.solid_angle_weights,
            )
        self.weights = self.sh.weights

    def transform(self, values) -> cp.ndarray:
        """Analysis: field values (flat or (nlat, nlon)) -> coefficients."""
        return self.sh.transform(values)

    def inv_transform(self, coeffs) -> cp.ndarray:
        """Synthesis: coefficients -> flat (n_points,) field."""
        return self.sh.inv_transform(coeffs)

    def transform_batch(self, values) -> cp.ndarray:
        """Batched analysis of (K, n_points) fields -> (K, l+1, l+1)."""
        return self.sh.transform_batch(values)

    def inv_transform_batch(self, coeffs) -> cp.ndarray:
        """Batched synthesis of (K, l+1, l+1) -> (K, n_points)."""
        return self.sh.inv_transform_batch(coeffs)

    def __getattr__(self, name):
        return getattr(self.sh, name)
