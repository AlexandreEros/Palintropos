"""What every elevation import path shares: its error type and size limit.

Both :mod:`tropoi.spatial.terrain.raster` (files and arrays) and
:mod:`tropoi.spatial.terrain.height_function` (callables and expressions)
raise :class:`ElevationImportError` and respect :data:`MAX_RASTER_PIXELS`.
Keeping them here lets ``raster`` use ``height_function`` without
``height_function`` depending back on ``raster``. ``raster`` re-exports
both names, which remain its public API.

Standard library only.
"""
from __future__ import annotations

__all__ = ["ElevationImportError", "MAX_RASTER_PIXELS"]

#: Largest raster (pixels) read into memory (float64: 512 MB). Larger
#: global DEMs must be downsampled first, e.g. ``gdal_translate -outsize``.
MAX_RASTER_PIXELS = 1 << 26


class ElevationImportError(ValueError):
    """An elevation file is unsupported, ambiguous or inconsistent."""
