"""Elevation rasters read from files: normalization and resampling (CPU).

This module is the single import path for external terrain. Every reader
(NetCDF, grey bitmaps such as 16-bit PNG, TIFF) returns the same normalized
:class:`ElevationRaster`, and one resampling path turns any raster into
values at a model's state-grid points
(:meth:`ElevationRaster.resample_to_points`); the projection onto the
spectral truncation happens in
:meth:`tropoi.spatial.terrain.topography.Topography.from_raster` with the
model's own transform.

Normalized form
---------------
* ``elevation_m``: float64 metres, shape ``(nlat, nlon)``, rows ordered
  **south to north** (latitude ascending), read-only.
* ``lat_deg``: ascending, uniformly spaced, covering the globe either as
  cell centres (``-90 + d/2 .. 90 - d/2``, "cell" registration) or as nodes
  including both poles ("node" registration).
* ``lon_deg``: ascending, uniformly spaced at ``360 / nlon``, covering one
  full turn exactly once (a duplicated seam column is dropped). The first
  longitude may be anywhere; sampling wraps.

What each reader needs, and why
-------------------------------
Inputs that cannot be interpreted unambiguously are rejected with an error
that names the missing argument; nothing is guessed.

* **NetCDF** (``.nc``, ``.nc4``, ``.cdf``; needs ``netCDF4``): coordinates
  come from 1-D coordinate variables in degrees; CF ``scale_factor`` /
  ``add_offset`` / ``_FillValue`` / ``missing_value`` are honoured; vertical
  units come from the variable's ``units`` attribute (``m`` or ``km``) or
  ``units=``. With several 2-D variables, ``variable=`` is required.
* **Bitmaps** (``.png``, ``.pgm``, ``.bmp``, ...; Pillow): single-channel
  grey only (colour maps cannot be inverted to heights). Row 0 is the
  northern edge (the raster convention). Pixels carry no units and no
  georeferencing, so ``scale``/``offset`` (or ``height_range``) and
  ``lon_west_deg`` (west edge of column 0) are required.
* **TIFF** (``.tif``, ``.tiff``; Pillow): GeoTIFF tags place the raster when
  present (geographic degrees, or equirectangular metres with standard
  parallel 0); without them it is a bitmap and needs ``lon_west_deg``.
  Vertical units must be given (``units=``) unless the file declares metres.

Missing values (NaN, fill values, ``nodata=``) raise unless ``fill_value=``
says what to put there. Only regular global latitude-longitude rasters are
supported: general GIS reprojection and regional mosaics are out of scope.

NumPy only at import; Pillow and netCDF4 are imported by the readers that
need them. Never imports CuPy.
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ElevationImportError",
    "ElevationRaster",
    "MAX_RASTER_PIXELS",
    "read_elevation",
    "working_resolution_deg",
]

#: Largest raster (pixels) read into memory (float64: 512 MB). Larger
#: global DEMs must be downsampled first, e.g. ``gdal_translate -outsize``.
MAX_RASTER_PIXELS = 1 << 26

#: Relative tolerance on coordinate spacing and global coverage.
_COORD_RTOL = 1e-4

_UNIT_FACTORS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "kilometre": 1000.0, "kilometres": 1000.0,
}

_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf", ".netcdf")
_TIFF_SUFFIXES = (".tif", ".tiff")
_IMAGE_SUFFIXES = (".png", ".pgm", ".bmp", ".gif", ".jpg", ".jpeg", ".webp")

#: Largest digital number of each grey Pillow mode (for ``height_range``).
_MODE_MAX_DN = {"L": 255, "I;16": 65535, "I;16B": 65535, "I;16L": 65535,
                "I;16N": 65535}
_GREY_MODES = ("L", "I;16", "I;16B", "I;16L", "I;16N", "I;16S", "I", "F")


class ElevationImportError(ValueError):
    """An elevation file is unsupported, ambiguous or inconsistent."""


def _unit_factor(units: str, where: str) -> float:
    key = str(units).strip().lower()
    if key not in _UNIT_FACTORS:
        raise ElevationImportError(
            f"{where}: unsupported elevation units {units!r}; supported: "
            "m (metres) or km")
    return _UNIT_FACTORS[key]


def working_resolution_deg(l_max: int) -> float:
    """Anti-aliasing working resolution for a truncation (degrees).

    About half the shortest resolved wavelength, ``180 / (2 (l_max + 1))``:
    a finer source is area-averaged to this before point sampling, so
    sub-grid relief is averaged instead of aliased onto resolved scales.
    """
    return 180.0 / (2.0 * (int(l_max) + 1))


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_step(values: np.ndarray, name: str) -> float:
    if values.size < 2:
        raise ElevationImportError(f"{name} needs at least two values")
    steps = np.diff(values)
    step = float(np.mean(steps))
    if step <= 0 or not np.allclose(steps, step, rtol=_COORD_RTOL,
                                    atol=1e-9):
        raise ElevationImportError(
            f"{name} must be uniformly spaced (a regular latitude-longitude "
            f"grid); spacing varies from {steps.min():g} to {steps.max():g} "
            "degrees")
    return step


def _cell_areas(lat_deg: np.ndarray, dlat: float) -> np.ndarray:
    """Relative solid angle of each latitude row (works at the poles)."""
    hi = np.radians(np.minimum(lat_deg + dlat / 2.0, 90.0))
    lo = np.radians(np.maximum(lat_deg - dlat / 2.0, -90.0))
    return np.sin(hi) - np.sin(lo)


@dataclass(frozen=True, eq=False)
class ElevationRaster:
    """A normalized global elevation raster (see the module docstring).

    Build with :func:`read_elevation` or :meth:`from_array`; the constructor
    validates an already-normalized grid. ``provenance`` holds JSON scalars
    describing the source and every interpretation applied to it.
    """

    elevation_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.elevation_m, dtype=np.float64)
        lat = np.array(self.lat_deg, dtype=np.float64).ravel()
        lon = np.array(self.lon_deg, dtype=np.float64).ravel()
        if values.ndim != 2 or values.shape != (lat.size, lon.size):
            raise ElevationImportError(
                f"elevation shape {values.shape} does not match "
                f"(nlat, nlon) = ({lat.size}, {lon.size})")
        if not np.isfinite(values).all():
            raise ElevationImportError(
                "elevation contains NaN/Inf after normalization")
        dlat = _uniform_step(lat, "latitudes")
        dlon = _uniform_step(lon, "longitudes")
        if not math.isclose(dlon * lon.size, 360.0, rel_tol=_COORD_RTOL):
            raise ElevationImportError(
                f"longitudes cover {dlon * lon.size:g} degrees; a global "
                "raster must cover 360 degrees exactly once")
        tol = _COORD_RTOL * 180.0 + 1e-9
        if abs(lat[0] + 90.0) <= tol and abs(lat[-1] - 90.0) <= tol:
            registration = "node"
        elif (abs(lat[0] - (-90.0 + dlat / 2)) <= tol
              and abs(lat[-1] - (90.0 - dlat / 2)) <= tol):
            registration = "cell"
        else:
            raise ElevationImportError(
                f"latitudes span [{lat[0]:g}, {lat[-1]:g}] with spacing "
                f"{dlat:g}: not a global raster (expected cell centres "
                f"[{-90 + dlat / 2:g}, {90 - dlat / 2:g}] or nodes "
                "[-90, 90]); regional rasters are not supported")
        for arr in (values, lat, lon):
            arr.flags.writeable = False
        object.__setattr__(self, "elevation_m", values)
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "lon_deg", lon)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))
        object.__setattr__(self, "_dlat", dlat)
        object.__setattr__(self, "_dlon", dlon)
        object.__setattr__(self, "_registration", registration)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_array(cls, elevation_m, lat_deg, lon_deg, *,
                   provenance: Mapping[str, Any] | None = None
                   ) -> "ElevationRaster":
        """Normalize an in-memory grid (rows in either latitude order).

        Latitudes may be ascending or descending; longitudes may start
        anywhere, use either the [-180, 180) or [0, 360) convention, and
        repeat the seam column. Values must be finite metres.
        """
        values, lat, lon, notes = _normalize_axes(
            np.asarray(elevation_m, dtype=np.float64),
            np.asarray(lat_deg, dtype=np.float64),
            np.asarray(lon_deg, dtype=np.float64))
        return cls(values, lat, lon, provenance={
            "format": "array", **(provenance or {}), **notes})

    # -- introspection -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.elevation_m.shape)

    @property
    def registration(self) -> str:
        """``"cell"`` (values are cell means/centres) or ``"node"``."""
        return self._registration

    @property
    def resolution_deg(self) -> float:
        """Latitude spacing in degrees."""
        return float(self._dlat)

    def area_mean(self) -> float:
        """Area-weighted global mean elevation (m)."""
        w = _cell_areas(self.lat_deg, self._dlat)
        return float(np.sum(w[:, None] * self.elevation_m)
                     / (np.sum(w) * self.elevation_m.shape[1]))

    def describe(self) -> str:
        p = self.provenance
        name = p.get("source_name") or p.get("format", "array")
        return (f"{name}: {self.shape[0]} x {self.shape[1]} "
                f"({self.resolution_deg:g} deg, {self.registration}-"
                f"registered), {float(self.elevation_m.min()):.0f} .. "
                f"{float(self.elevation_m.max()):.0f} m")

    # -- resampling ----------------------------------------------------------

    def coarsened(self, resolution_deg: float) -> "ElevationRaster":
        """Area-weighted block mean onto a coarser global cell grid.

        Returns ``self`` when the raster is already at least this coarse.
        Each output cell averages the source cells whose centres fall in it,
        weighted by solid angle, so the global mean is preserved.
        """
        target = float(resolution_deg)
        if not target > 0:
            raise ValueError(f"resolution_deg must be > 0, got {target}")
        ny = max(1, int(round(180.0 / target)))
        nx = max(1, int(round(360.0 / target)))
        nlat, nlon = self.shape
        if ny >= nlat and nx >= nlon:
            return self
        ny, nx = min(ny, nlat), min(nx, nlon)
        lat, lon, values = self.lat_deg, self.lon_deg, self.elevation_m
        rows = np.clip(np.floor((lat + 90.0) / 180.0 * ny).astype(int),
                       0, ny - 1)
        edge = lon[0] - self._dlon / 2.0
        cols = np.clip(np.floor(((lon - edge) % 360.0) / 360.0 * nx)
                       .astype(int), 0, nx - 1)
        row_starts = np.flatnonzero(np.r_[True, np.diff(rows) != 0])
        col_starts = np.flatnonzero(np.r_[True, np.diff(cols) != 0])
        if row_starts.size != ny or col_starts.size != nx:
            raise RuntimeError("coarsening produced empty bins")  # pragma: no cover
        w = _cell_areas(lat, self._dlat)
        by_row = (np.add.reduceat(values * w[:, None], row_starts, axis=0)
                  / np.add.reduceat(w, row_starts)[:, None])
        counts = np.diff(np.r_[col_starts, nlon])
        out = np.add.reduceat(by_row, col_starts, axis=1) / counts[None, :]
        out_lat = -90.0 + (np.arange(ny) + 0.5) * 180.0 / ny
        out_lon = edge + (np.arange(nx) + 0.5) * 360.0 / nx
        prov = dict(self.provenance)
        prov["coarsened_to_deg"] = 180.0 / ny
        return ElevationRaster(out, out_lat, out_lon, provenance=prov)

    def sample(self, lat_deg, lon_deg) -> np.ndarray:
        """Bilinear interpolation at arbitrary points (degrees).

        Longitudes wrap. Each pole is a single value (the area mean of the
        row nearest to it), so the field is single-valued there; between
        the outermost row and the pole the interpolation runs towards that
        polar value.
        """
        lat_q = np.asarray(lat_deg, dtype=np.float64)
        lon_q = np.asarray(lon_deg, dtype=np.float64)
        values = self.elevation_m
        lat = self.lat_deg
        if self.registration == "cell":
            lat = np.r_[-90.0, lat, 90.0]
            values = np.vstack([np.full(values.shape[1], values[0].mean()),
                                values,
                                np.full(values.shape[1], values[-1].mean())])
        else:
            values = values.copy()
            values[0] = values[0].mean()
            values[-1] = values[-1].mean()
        nlon = values.shape[1]
        x = ((lon_q - self.lon_deg[0]) % 360.0) / self._dlon
        i0 = np.floor(x).astype(int) % nlon
        i1 = (i0 + 1) % nlon
        fx = x - np.floor(x)
        y = np.interp(lat_q, lat, np.arange(lat.size, dtype=np.float64))
        j0 = np.clip(np.floor(y).astype(int), 0, lat.size - 2)
        fy = y - j0
        bottom = values[j0, i0] * (1 - fx) + values[j0, i1] * fx
        top = values[j0 + 1, i0] * (1 - fx) + values[j0 + 1, i1] * fx
        return bottom * (1 - fy) + top * fy

    def resample_to_points(self, lat_rad, lon_rad, *,
                           working_resolution: float | None = None
                           ) -> tuple[np.ndarray, dict]:
        """Values at state-grid points (radians) for spectral analysis.

        A source finer than ``working_resolution`` (degrees) is first
        area-averaged to it (:meth:`coarsened`), then sampled bilinearly
        (:meth:`sample`). Returns ``(values, settings)`` where ``settings``
        records what was done.
        """
        source = self
        if working_resolution is not None \
                and working_resolution > self.resolution_deg * (1 + 1e-9):
            source = self.coarsened(working_resolution)
        values = source.sample(np.degrees(np.asarray(lat_rad)),
                               np.degrees(np.asarray(lon_rad)))
        return values, {
            "resampling": ("area-weighted block mean, then bilinear"
                           if source is not self else "bilinear"),
            "working_resolution_deg": float(source.resolution_deg),
        }


# ---------------------------------------------------------------------------
# Normalization shared by every reader
# ---------------------------------------------------------------------------

def _normalize_axes(values: np.ndarray, lat: np.ndarray, lon: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Order rows south->north and columns by ascending wrapped longitude."""
    notes: dict = {}
    lat, lon = lat.ravel(), lon.ravel()
    if values.shape != (lat.size, lon.size):
        raise ElevationImportError(
            f"elevation shape {values.shape} does not match the coordinates "
            f"(nlat, nlon) = ({lat.size}, {lon.size})")
    if lat.size >= 2 and lat[0] > lat[-1]:
        lat, values = lat[::-1], values[::-1]
        notes["rows_flipped_to_south_first"] = True
    if np.any(np.abs(lat) > 90.0 + 1e-6):
        raise ElevationImportError(
            f"latitudes outside [-90, 90] (range {lat.min():g}..{lat.max():g}"
            "): coordinates must be in degrees")
    if lon.size >= 2:
        _uniform_step(lon, "longitudes")
        span = lon[-1] - lon[0]
        step = (lon[-1] - lon[0]) / (lon.size - 1)
        if math.isclose(span, 360.0, rel_tol=_COORD_RTOL):
            if not np.allclose(values[:, 0], values[:, -1], rtol=1e-6,
                               atol=1e-6 * (np.abs(values).max() + 1.0)):
                raise ElevationImportError(
                    "the first and last longitude columns are the same "
                    "meridian but hold different values")
            lon, values = lon[:-1], values[:, :-1]
            notes["duplicate_seam_column_dropped"] = True
        elif not math.isclose(step * lon.size, 360.0, rel_tol=_COORD_RTOL):
            raise ElevationImportError(
                f"longitudes cover {step * lon.size:g} degrees; a global "
                "raster must cover 360 degrees (regional rasters are not "
                "supported)")
        wrapped = (lon + 180.0) % 360.0 - 180.0
        order = np.argsort(wrapped, kind="stable")
        if np.any(order != np.arange(lon.size)):
            notes["longitudes_rewrapped"] = "[-180, 180)"
        lon, values = wrapped[order], values[:, order]
    notes["lon_first_deg"] = float(lon[0]) if lon.size else None
    return values, lat, lon, notes


def _apply_missing(values: np.ndarray, missing: np.ndarray,
                   fill_value: float | None, where: str, prov: dict
                   ) -> np.ndarray:
    count = int(np.count_nonzero(missing))
    prov["missing_count"] = count
    if count == 0:
        return values
    if fill_value is None:
        raise ElevationImportError(
            f"{where}: {count} missing values ({100.0 * count / missing.size:.3g}"
            "% of the raster: NaN, fill or nodata values); pass "
            "fill_value=<metres> to replace them explicitly")
    out = values.copy()
    out[missing] = float(fill_value)
    prov["fill_value_m"] = float(fill_value)
    return out


def _scale_offset(scale, offset, height_range, max_dn, where: str
                  ) -> tuple[float, float]:
    if height_range is not None:
        if scale is not None or offset is not None:
            raise ElevationImportError(
                f"{where}: give either height_range or scale/offset, not both")
        if max_dn is None:
            raise ElevationImportError(
                f"{where}: height_range needs an integer bit depth (8- or "
                "16-bit grey); give scale and offset instead")
        lo, hi = (float(v) for v in height_range)
        if not hi > lo:
            raise ElevationImportError(
                f"{where}: height_range must be (low, high) with high > low")
        return (hi - lo) / max_dn, lo
    if scale is None:
        raise ElevationImportError(
            f"{where}: pixel values carry no units; give scale= (units per "
            "pixel value) and offset=, or height_range=(low, high) for the "
            "full grey range")
    return float(scale), float(offset or 0.0)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_elevation(path, *, format: str | None = None,
                   variable: str | None = None, units: str | None = None,
                   scale: float | None = None, offset: float | None = None,
                   height_range: tuple[float, float] | None = None,
                   lon_west_deg: float | None = None,
                   nodata: float | None = None,
                   fill_value: float | None = None) -> ElevationRaster:
    """Read a global elevation raster from ``path`` (see the module docstring).

    Parameters
    ----------
    format : {"netcdf", "image", "tiff"}, optional
        Overrides detection by file suffix.
    variable : str, optional
        NetCDF variable (required when several 2-D variables exist).
    units : {"m", "km"}, optional
        Units of the (scaled) values. Required for TIFF without a metre
        vertical-units key and for NetCDF variables without ``units``;
        images default to metres (``scale`` is in these units).
    scale, offset, height_range :
        ``value = raw * scale + offset``; ``height_range=(low, high)`` maps
        the full 8/16-bit grey range linearly. Required for bitmaps.
    lon_west_deg : float, optional
        Longitude of the western edge of the first column, for rasters
        without georeferencing (bitmaps, plain TIFF). ``-180`` for maps
        centred on the prime meridian, ``0`` for maps starting at it.
    nodata : float, optional
        Raw value marking missing data.
    fill_value : float, optional
        Elevation (m) to put where data are missing; without it missing
        data raise.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise ElevationImportError(f"no such file: {path}")
    kind = format or _detect_format(path)
    if kind not in ("netcdf", "image", "tiff"):
        raise ElevationImportError(
            f"unknown format {kind!r}; choose netcdf, image or tiff")
    prov: dict = {"source_name": path.name,
                  "source_sha256": _file_sha256(path), "format": kind}
    if kind == "netcdf":
        values, lat, lon, missing = _read_netcdf(
            path, variable, units, scale, offset, nodata, prov)
    else:
        values, lat, lon, missing = _read_raster_image(
            path, kind, units, scale, offset, height_range, lon_west_deg,
            nodata, prov)
    values = _apply_missing(values, missing, fill_value, path.name, prov)
    values, lat, lon, notes = _normalize_axes(values, lat, lon)
    prov.update(notes)
    if np.abs(values).max() > 1.0e5:
        raise ElevationImportError(
            f"{path.name}: |elevation| reaches {np.abs(values).max():.3g} m, "
            "beyond any planetary relief; check units, scale and offset")
    return ElevationRaster(values, lat, lon, provenance=prov)


def _detect_format(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _NETCDF_SUFFIXES:
        return "netcdf"
    if suffix in _TIFF_SUFFIXES:
        return "tiff"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    raise ElevationImportError(
        f"cannot tell the format of {path.name!r} from its suffix; pass "
        "format='netcdf', 'image' or 'tiff'")


def _check_size(shape, where: str) -> None:
    n = int(np.prod(shape))
    if n > MAX_RASTER_PIXELS:
        raise ElevationImportError(
            f"{where}: {shape[0]} x {shape[1]} = {n} values exceeds the "
            f"{MAX_RASTER_PIXELS} limit; downsample first (for example "
            "gdal_translate -outsize 25% 25% in.tif out.tif)")


# -- NetCDF -------------------------------------------------------------------

_LAT_NAMES = ("lat", "latitude", "y", "nav_lat")
_LON_NAMES = ("lon", "longitude", "x", "nav_lon")


def _coordinate_kind(var) -> str | None:
    units = str(getattr(var, "units", "")).strip().lower()
    std = str(getattr(var, "standard_name", "")).strip().lower()
    name = var.name.lower()
    if std == "latitude" or units in ("degrees_north", "degree_north",
                                      "degree_n", "degrees_n"):
        return "lat"
    if std == "longitude" or units in ("degrees_east", "degree_east",
                                       "degree_e", "degrees_e"):
        return "lon"
    if units in ("degrees", "degree", "deg"):
        if name in _LAT_NAMES:
            return "lat"
        if name in _LON_NAMES:
            return "lon"
    return None


def _read_netcdf(path, variable, units, scale, offset, nodata, prov):
    try:
        import netCDF4
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ElevationImportError(
            "reading NetCDF needs the netCDF4 package (pip install netCDF4)"
        ) from err
    where = path.name
    with netCDF4.Dataset(path) as ds:
        coords = {}
        for dim in ds.dimensions:
            if dim in ds.variables and ds.variables[dim].ndim == 1:
                kind = _coordinate_kind(ds.variables[dim])
                if kind:
                    coords[dim] = kind
        if "lat" not in coords.values() or "lon" not in coords.values():
            raise ElevationImportError(
                f"{where}: needs 1-D latitude and longitude coordinate "
                "variables in degrees (units degrees_north / degrees_east "
                "or standard_name latitude / longitude)")
        candidates = []
        for name, var in ds.variables.items():
            if name in coords:
                continue
            spatial = [d for d in var.dimensions if d in coords]
            others = [d for d in var.dimensions if d not in coords]
            if len(spatial) == 2 and {coords[d] for d in spatial} \
                    == {"lat", "lon"} and all(ds.dimensions[d].size == 1
                                              for d in others):
                candidates.append(name)
        if variable is None:
            if len(candidates) != 1:
                raise ElevationImportError(
                    f"{where}: found {len(candidates)} latitude-longitude "
                    f"variables {candidates}; pass variable=<name>")
            variable = candidates[0]
        elif variable not in candidates:
            raise ElevationImportError(
                f"{where}: variable {variable!r} is not a latitude-longitude "
                f"field; candidates are {candidates}")
        var = ds.variables[variable]
        _check_size([ds.dimensions[d].size for d in var.dimensions
                     if d in coords], where)
        var.set_auto_maskandscale(True)
        data = var[...]
        dims = list(var.dimensions)
        keep = [i for i, d in enumerate(dims) if d in coords]
        data = np.ma.asarray(data)
        data = data.reshape([data.shape[i] for i in keep])
        dims = [dims[i] for i in keep]
        lat_dim = next(d for d in dims if coords[d] == "lat")
        lon_dim = next(d for d in dims if coords[d] == "lon")
        if dims == [lon_dim, lat_dim]:
            data = data.T
        lat = np.asarray(ds.variables[lat_dim][:], dtype=np.float64)
        lon = np.asarray(ds.variables[lon_dim][:], dtype=np.float64)
        file_units = getattr(var, "units", None)
        for attr in ("scale_factor", "add_offset", "_FillValue",
                     "missing_value"):
            if hasattr(var, attr):
                prov[f"cf_{attr.strip('_')}"] = float(
                    np.ravel(getattr(var, attr))[0])
    prov["variable"] = variable
    if file_units is not None and units is not None \
            and _unit_factor(file_units, where) != _unit_factor(units, where):
        raise ElevationImportError(
            f"{where}: units={units!r} contradicts the variable's units "
            f"attribute {file_units!r}")
    chosen = units if units is not None else file_units
    if chosen is None:
        raise ElevationImportError(
            f"{where}: variable {variable!r} has no units attribute; pass "
            "units='m' or units='km'")
    factor = _unit_factor(chosen, where)
    raw = np.ma.getdata(data).astype(np.float64)
    missing = np.ma.getmaskarray(data) | ~np.isfinite(raw)
    if nodata is not None:
        missing |= raw == float(nodata)
        prov["nodata"] = float(nodata)
    a, b = float(1.0 if scale is None else scale), float(offset or 0.0)
    prov.update(units=str(chosen), unit_factor_to_m=factor, scale=a,
                offset=b)
    return (raw * a + b) * factor, lat, lon, missing


# -- Bitmaps and TIFF ---------------------------------------------------------

def _read_raster_image(path, kind, units, scale, offset, height_range,
                       lon_west_deg, nodata, prov):
    from PIL import Image

    where = path.name
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as img:
            _check_size((img.height, img.width), where)
            mode = img.mode
            if mode not in _GREY_MODES:
                raise ElevationImportError(
                    f"{where}: image mode {mode!r} is not single-channel "
                    "grey; elevation must be one value per pixel (colour "
                    "renderings cannot be converted back to heights)")
            raw = np.asarray(img).astype(np.float64)
            tags = dict(img.tag_v2) if kind == "tiff" and hasattr(
                img, "tag_v2") else {}
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    prov["image_mode"] = mode
    nrow, ncol = raw.shape
    geo = _geotiff_extent(tags, nrow, ncol, where) if tags else None
    if geo is not None:
        west, north, dlon, dlat, note = geo
        if lon_west_deg is not None and not math.isclose(
                (float(lon_west_deg) - west + 180.0) % 360.0 - 180.0, 0.0,
                abs_tol=1e-6):
            raise ElevationImportError(
                f"{where}: lon_west_deg={lon_west_deg} contradicts the "
                f"GeoTIFF georeferencing (west edge {west:g})")
        prov["georeferencing"] = note
    else:
        if lon_west_deg is None:
            raise ElevationImportError(
                f"{where}: the raster carries no georeferencing; pass "
                "lon_west_deg (longitude of the west edge of the first "
                "column: -180 for maps centred on 0 E, 0 for maps starting "
                "at the prime meridian). Rows are read north-up.")
        west, north = float(lon_west_deg), 90.0
        dlon, dlat = 360.0 / ncol, 180.0 / nrow
        prov["georeferencing"] = ("none: global plate carree, north-up, "
                                  "pixel-is-area")
        prov["lon_west_deg"] = west
    lat = north - (np.arange(nrow) + 0.5) * dlat
    lon = west + (np.arange(ncol) + 0.5) * dlon

    missing = ~np.isfinite(raw)
    gdal_nodata = tags.get(42113)
    if gdal_nodata not in (None, ""):
        value = float(str(gdal_nodata).strip("\x00 "))
        if nodata is not None and float(nodata) != value:
            raise ElevationImportError(
                f"{where}: nodata={nodata} contradicts the file's GDAL "
                f"nodata value {value:g}")
        nodata = value
    if nodata is not None:
        missing |= raw == float(nodata)
        prov["nodata"] = float(nodata)

    file_scale = _gdal_scale_offset(tags.get(42112))
    if file_scale is not None:
        if scale is not None or offset is not None or height_range is not None:
            raise ElevationImportError(
                f"{where}: the file declares scale/offset {file_scale}; do "
                "not pass scale, offset or height_range as well")
        a, b = file_scale
        prov["scale_source"] = "GDAL metadata"
    elif kind == "tiff" and scale is None and height_range is None:
        a, b = 1.0, float(offset or 0.0)
    else:
        a, b = _scale_offset(scale, offset, height_range,
                             _MODE_MAX_DN.get(mode), where)

    vertical = _geokeys(tags, where).get(4099) if tags else None
    if kind == "tiff":
        if units is None and vertical == 9001:
            units = "m"
        if units is None:
            raise ElevationImportError(
                f"{where}: TIFF elevation units are not declared; pass "
                "units='m' or units='km'")
    else:
        units = units or "m"
    factor = _unit_factor(units, where)
    prov.update(units=str(units), unit_factor_to_m=factor, scale=a, offset=b)
    return (raw * a + b) * factor, lat, lon, missing


def _gdal_scale_offset(xml) -> tuple[float, float] | None:
    if not xml:
        return None
    text = str(xml)
    scale = re.search(r'role="scale"[^>]*>\s*([-+0-9.eE]+)', text)
    offset = re.search(r'role="offset"[^>]*>\s*([-+0-9.eE]+)', text)
    if not scale and not offset:
        return None
    return (float(scale.group(1)) if scale else 1.0,
            float(offset.group(1)) if offset else 0.0)


def _geokeys(tags: Mapping, where: str) -> dict:
    directory = tags.get(34735)
    if not directory:
        return {}
    directory = [int(v) for v in np.atleast_1d(directory)]
    raw_doubles = tags.get(34736)
    doubles = ([] if raw_doubles is None
               else [float(v) for v in np.atleast_1d(raw_doubles)])
    keys: dict = {}
    for k in range(directory[3]):
        key, location, count, value = directory[4 + 4 * k: 8 + 4 * k]
        if location == 0:
            keys[key] = value
        elif location == 34736:
            vals = doubles[value:value + count]
            keys[key] = vals[0] if count == 1 else tuple(vals)
        else:
            keys[key] = None  # ASCII citations: not needed here
    return keys


def _geotiff_extent(tags: Mapping, nrow: int, ncol: int, where: str):
    """(west, north, dlon, dlat, note) in degrees from GeoTIFF tags, or None."""
    pixel_scale = tags.get(33550)
    tiepoint = tags.get(33922)
    if not pixel_scale or not tiepoint:
        return None
    keys = _geokeys(tags, where)
    sx, sy = float(pixel_scale[0]), float(pixel_scale[1])
    i, j, _, x, y, _ = (float(v) for v in tiepoint[:6])
    if keys.get(1025) == 2:          # PixelIsPoint: tiepoint at a centre
        i, j = i + 0.5, j + 0.5
    west_x, north_y = x - i * sx, y + j * sy
    model = keys.get(1024)
    if model == 2:                   # geographic
        unit = keys.get(2054)
        if unit not in (None, 9102):
            raise ElevationImportError(
                f"{where}: geographic GeoTIFF angular unit {unit} is not "
                "degrees")
        return west_x, north_y, sx, sy, "GeoTIFF geographic (degrees)"
    if model == 1:                   # projected
        transform = keys.get(3075)
        if transform != 17:
            raise ElevationImportError(
                f"{where}: projected GeoTIFF (coordinate transformation "
                f"{transform}) is not equirectangular; reproject to a "
                "global latitude-longitude grid first (for example gdalwarp "
                "-t_srs '+proj=longlat +R=<radius>')")
        parallel = float(keys.get(3078) or 0.0)
        if abs(parallel) > 1e-9:
            raise ElevationImportError(
                f"{where}: equirectangular standard parallel {parallel:g} "
                "is not 0; reproject to a plain latitude-longitude grid")
        linear = keys.get(3076)
        if linear not in (None, 9001):
            raise ElevationImportError(
                f"{where}: projected linear unit {linear} is not metres")
        radius = keys.get(2057) or keys.get(2056)
        if not radius:
            raise ElevationImportError(
                f"{where}: equirectangular GeoTIFF without the sphere "
                "radius (GeogSemiMajorAxisGeoKey); cannot convert metres "
                "to degrees")
        radius = float(radius)
        centre = float(keys.get(3088) or keys.get(3080) or 0.0)
        false_e = float(keys.get(3082) or 0.0)
        false_n = float(keys.get(3083) or 0.0)
        deg = 180.0 / (math.pi * radius)
        return (centre + (west_x - false_e) * deg, (north_y - false_n) * deg,
                sx * deg, sy * deg,
                f"GeoTIFF equirectangular (sphere radius {radius:g} m)")
    raise ElevationImportError(
        f"{where}: GeoTIFF model type {model} is not supported (only "
        "geographic or equirectangular global rasters)")
