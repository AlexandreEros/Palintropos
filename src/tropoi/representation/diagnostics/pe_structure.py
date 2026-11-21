"""Columns, vertical sections, zonal means and the meridional mass
streamfunction of saved PE states (CPU).

Every product is an exact host synthesis of the saved spectral coefficients
through :func:`~tropoi.representation.diagnostics.pe_levels.
reconstruct_levels`, on native full levels (0 = top). Nothing is
interpolated in the vertical.

* :func:`column` evaluates one geographic point. ``p_k = sigma_k p_s`` is
  the model's pure-sigma pressure at the nominal full level. ``Z_k`` is the
  Simmons–Burridge full-level geopotential divided by the model's gravity
  (absolute, including ``Phi_s``). It sits at the hydrostatic level position
  ``sigma^h_k``, not at ``sigma_k``. The lowest full level is an air
  temperature at ``sigma_{K-1}`` and never a surface temperature.
* :func:`vertical_section` evaluates a latitude x level section at one
  longitude, or a longitude x level section at one latitude. Points can be
  any latitudes or longitudes, since the synthesis is exact everywhere.
* :class:`ZonalMeanAccumulator` reduces states on the run's own Gaussian
  state grid (``nlat`` Gauss–Legendre latitudes, ``nlon`` equally spaced
  longitudes). The longitude mean ``[ ]`` is the plain mean over the
  ``nlon`` longitudes. It gives ``[u]``, ``[v]`` and ``[T]`` through the
  Held–Suarez :class:`~tropoi.representation.diagnostics.
  held_suarez_sections.SectionAccumulator`, so the two reductions share one
  definition. It also gives ``[p_s v_k]``: each state's product is formed on
  the grid before its longitude mean, and that mean is never replaced by
  ``[p_s][v_k]``. Several states enter with equal weights 1/N.
* :func:`mass_streamfunction` gives the meridional mass streamfunction on
  sigma interfaces for a pure-sigma column. It integrates downward from zero
  at the model top (``sigma = 0``)::

      psi(phi, sigma_{k+1/2}) = (2 pi a cos(phi) / g) sum_{j <= k} [p_s v_j] dsigma_j

  ``dsigma_j`` is the layer thickness between the saved interfaces, so the
  layer mass per unit area is ``p_s dsigma_j / g``. Units are kg s^-1.
  Positive values mean net northward mass transport above the interface.
  ``psi`` at ``sigma = 1`` is the whole column's northward mass flux. It is
  reported as a residual and never forced to zero. For one state it is
  nonzero wherever the column mass poleward of that latitude is changing,
  and only a long time mean makes it small.

NumPy only; never imports CuPy.
"""
from __future__ import annotations

import math

import numpy as np

from tropoi.representation.diagnostics.held_suarez_sections import (
    SectionAccumulator, state_grid)
from tropoi.representation.diagnostics.pe_levels import reconstruct_levels

__all__ = [
    "COLUMN_FIELDS",
    "SECTION_FIELDS",
    "ZonalMeanAccumulator",
    "column",
    "mass_streamfunction",
    "vertical_section",
]

#: Per-level outputs of :func:`column` and :func:`vertical_section`.
COLUMN_FIELDS = ("temperature", "u", "v", "pressure", "geopotential_height")
SECTION_FIELDS = COLUMN_FIELDS


def _finite_point(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat, lon = float(lat_deg), float(lon_deg)
    if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90.0:
        raise ValueError("a geographic point needs a finite latitude in "
                         "[-90, 90] and a finite longitude")
    return lat, lon % 360.0


def column(frame, context, lat_deg: float, lon_deg: float) -> dict:
    """Native-level profile of one saved state at one point (module doc)."""
    lat, lon = _finite_point(lat_deg, lon_deg)
    levels = reconstruct_levels(frame, context, [lat], [lon], None,
                                vorticity_divergence=False)
    out = {name: levels[name][:, 0, 0].copy() for name in COLUMN_FIELDS}
    out["surface_pressure"] = float(levels.surface_pressure[0, 0])
    out["surface_height"] = (0.0 if levels.surface_height is None
                             else float(levels.surface_height[0, 0]))
    out["sigma"] = np.asarray(context.full_levels, dtype=np.float64)
    out["sigma_hydrostatic"] = np.asarray(
        context.sigma.hydrostatic_full_levels, dtype=np.float64)
    out["sigma_interfaces"] = np.asarray(context.sigma_interfaces,
                                         dtype=np.float64)
    out["latitude"], out["longitude"] = lat, lon
    return out


def vertical_section(frame, context, *, latitude=None, longitude=None,
                     points) -> dict:
    """Latitude x level section at ``longitude``, or longitude x level
    section at ``latitude``, sampled at ``points`` (degrees).

    Per-level arrays are ``(nlev, len(points))``. ``surface_pressure`` and
    ``surface_height`` are ``(len(points),)``.
    """
    if (latitude is None) == (longitude is None):
        raise ValueError("give exactly one of latitude= or longitude=")
    points = np.asarray(points, dtype=np.float64).ravel()
    if points.size < 2 or not np.isfinite(points).all():
        raise ValueError("a section needs at least two finite sample points")
    if longitude is not None:
        _, lon = _finite_point(0.0, longitude)
        if np.abs(points).max() > 90.0:
            raise ValueError("section latitudes must lie in [-90, 90]")
        lat_axis, lon_axis, take = points, np.array([lon]), (slice(None), 0)
    else:
        lat, _ = _finite_point(latitude, 0.0)
        lat_axis, lon_axis, take = np.array([lat]), points % 360.0, (0, slice(None))
    levels = reconstruct_levels(frame, context, lat_axis, lon_axis, None,
                                vorticity_divergence=False)
    out = {name: levels[name][(slice(None), *take)].copy()
           for name in SECTION_FIELDS}
    out["surface_pressure"] = levels.surface_pressure[take].copy()
    out["surface_height"] = (np.zeros(points.size)
                             if levels.surface_height is None
                             else levels.surface_height[take].copy())
    return out


class ZonalMeanAccumulator:
    """Equal-weight zonal means of saved states on the run's state grid.

    ``nlat``/``nlon`` are the run's Gaussian grid sizes. :meth:`add` takes
    one packed PE frame. :meth:`means` returns ``(nlev, nlat)`` arrays with
    latitudes north to south (:attr:`lat_deg`).
    """

    def __init__(self, context, nlat: int, nlon: int) -> None:
        self.context = context
        self.lat_deg, self.lon_deg, self.latitude_source = state_grid(nlat, nlon)
        self.sections = SectionAccumulator(int(nlat), context.nlev, int(nlon))
        self.ps_v = np.zeros((context.nlev, int(nlat)))
        self.ps = np.zeros(int(nlat))

    @property
    def count(self) -> int:
        return self.sections.count

    def add(self, frame) -> None:
        levels = reconstruct_levels(frame, self.context, self.lat_deg,
                                    self.lon_deg, None,
                                    vorticity_divergence=False)
        u, v, temperature = levels["u"], levels["v"], levels["temperature"]
        self.sections.add(u, v, temperature)
        p_s = levels.surface_pressure                      # (nlat, nlon)
        # The product is formed at every grid point before its longitude
        # mean. [p_s v] differs from [p_s][v] by [p_s* v*].
        self.ps_v += (p_s[None] * v).mean(axis=2)
        self.ps += p_s.mean(axis=1)

    def means(self) -> dict:
        moments = self.sections.means()                    # (nlat, nlev)
        n = self.count
        return {"u": moments["u"].T, "v": moments["v"].T,
                "temperature": moments["T"].T, "ps_v": self.ps_v / n,
                "surface_pressure": self.ps / n}


def mass_streamfunction(ps_v, sigma_interfaces, lat_deg, radius_m: float,
                        gravity: float) -> np.ndarray:
    """Meridional mass streamfunction (kg s^-1) on interfaces (module doc).

    ``ps_v`` is ``[p_s v_k]`` with shape ``(nlev, nlat)``, in Pa m s^-1.
    Returns ``(nlev + 1, nlat)``: row 0 is the model top (zero) and the last
    row is the whole-column flux at ``sigma = 1``.
    """
    ps_v = np.asarray(ps_v, dtype=np.float64)
    interfaces = np.asarray(sigma_interfaces, dtype=np.float64)
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    if interfaces[0] != 0.0 or interfaces[-1] != 1.0:
        raise ValueError("the mass streamfunction needs a pure-sigma column "
                         "from sigma = 0 to 1")
    if ps_v.shape != (interfaces.size - 1, lat.size):
        raise ValueError(f"[p_s v] has shape {ps_v.shape}; expected "
                         f"{(interfaces.size - 1, lat.size)}")
    if not (radius_m > 0 and gravity > 0):
        raise ValueError("radius and gravity must be positive")
    layer = ps_v * np.diff(interfaces)[:, None]
    above = np.concatenate([np.zeros((1, lat.size)), np.cumsum(layer, axis=0)])
    return 2.0 * math.pi * radius_m * np.cos(lat)[None] / gravity * above
