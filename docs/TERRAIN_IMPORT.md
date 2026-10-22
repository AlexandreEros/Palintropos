# Terrain from elevation files

Load a planet's bottom topography from a NetCDF file, a grey bitmap
(16-bit PNG and similar) or a global TIFF. Look at the terrain the model will
actually use before running anything:

```python
from tropoi.run.custom import (DiscretizationSpec, PlanetSpec,
                               prepare_planet, read_elevation)

raster = read_elevation("mars_mola_4ppd.tif", units="m")       # CPU
world = prepare_planet(PlanetSpec.oblate(3.3962e6, 88642.66),
                       DiscretizationSpec.gauss_latlon(42, 66, 132),
                       raster, smoothing=0.3)                    # CUDA
world.plot()          # static Matplotlib figure
world.view()          # interactive 3D globe, one self-contained HTML file
```

A runnable version of this workflow, with generated reference terrain, is in
[examples/terrain_planet.py](../examples/terrain_planet.py):

```bash
python examples/terrain_planet.py generate terrain               # synthetic Mars-like files (CPU)
python examples/terrain_planet.py show terrain/mars_like.nc      # resolve; figure + globe (CUDA)
python examples/terrain_planet.py state terrain/mars_like.nc states/mars_like
python examples/terrain_planet.py preview states/mars_like       # figure + globe, CPU only
tropoi run swe --initial-state states/mars_like --days 1         # a run over that terrain
```

The generated terrain is **synthetic**. It has a dichotomy, a Tharsis-like
rise with four volcanoes and a Hellas-like basin, so the whole file path can
be exercised without a download. It is not MOLA data. For real Mars, see
[Mars from MOLA](#mars-from-mola).

## The pipeline

| Step | Where | What happens |
| --- | --- | --- |
| Read | `read_elevation` (CPU) | A format reader interprets the file. Every reader returns the same normalized `ElevationRaster`: metres, rows south to north, uniform global longitudes. The source's name, SHA-256 and every interpretation are recorded in `raster.provenance`. |
| Resample | `ElevationRaster.resample_to_points` (CPU) | A source finer than the working resolution `180° / (2 (l_max + 1))` is area-averaged to it, then sampled bilinearly at the model's state-grid points. Longitudes wrap, and each pole is single-valued. |
| Clip (optional) | `clip_below_m=` | For example `0` puts an ocean floor at sea level. |
| Project | `Topography.from_raster` (CUDA) | Analysis with the backend's own transform at the full truncation, as for the presets. The truncation residual is measured. |
| Smooth (optional) | `smoothing=s` | Hoskins-type filter `exp(-k (l(l+1)/(L(L+1)))²)`, `k = -ln s`. It keeps amplitude `s` at `l_max` and leaves `l = 0` untouched. Off by default. |
| Record | `TopographySpec.imported(...)` | A provenance record: the file name and SHA-256, the reader settings, the preprocessing settings, the measured residuals and extrema. It is not a recipe. The resolved coefficients always travel with it. |

Real terrain is not band-limited. As for the Williamson-5 cone, the residual
is **reported**, not used to reject. The preview shows the consequences:
Gibbs ripples, and peaks lowered by truncation. On the synthetic Mars-like
terrain at `l_max = 42`, the 18.8 km volcano is resolved as about 10 km.

What smoothing does, measured on the Williamson-5 cone at `l_max = 42`:

| `smoothing` | RMS ringing > 45° from the cone | undershoot at the cone's base | peak |
| --- | --- | --- | --- |
| none | 0.37 m | −12.5 m | 1801 m |
| 0.5 | 0.18 m | −12.5 m | 1782 m |
| 0.1 | 0.03 m | −15.1 m | 1744 m |

It removes the far-field ripples but not the undershoot next to steep
relief, and it lowers peaks. That is why it is opt-in, and why the figure
shows "model − source".

## Readers

Inputs that cannot be read unambiguously are rejected, and the error names
the argument to pass. Nothing is guessed.

| Format | Coordinates | Units and values | You must say |
| --- | --- | --- | --- |
| NetCDF (`.nc`, `.nc4`, `.cdf`; needs `netCDF4`, the `terrain` extra) | 1-D latitude/longitude coordinate variables in degrees (`degrees_north`/`degrees_east`, or `standard_name`). Either dimension order. Extra singleton dimensions are allowed. | `units` attribute (`m` or `km`). CF `scale_factor`, `add_offset`, `_FillValue` and `missing_value` are honoured. | `variable=` when several lat-lon fields exist; `units=` when the variable has none |
| Bitmaps (`.png`, `.pgm`, `.bmp`, …; Pillow) | none: a global plate carrée, **row 0 = north** | single-channel grey only (8- or 16-bit, int, float). Colour renderings are refused. | `lon_west_deg` (west edge of column 0: `-180` or `0`); `scale`/`offset`, or `height_range=(low, high)` for the full grey range |
| TIFF (`.tif`, `.tiff`; Pillow) | GeoTIFF tags if present: geographic degrees, or equirectangular metres with standard parallel 0 (converted with the file's sphere radius). Otherwise, as a bitmap. | GDAL nodata and scale/offset metadata are honoured | `units=` (unless the file declares metres); `lon_west_deg` without georeferencing |

Common to all readers:

- Longitudes may use either the `[-180, 180)` or the `[0, 360)` convention.
  A repeated seam column is dropped, but only if it really repeats the
  first one.
- Latitudes may run either way. Cell-centred and node grids (including the
  poles) are both accepted.
- Only global, uniformly spaced latitude–longitude grids are accepted.
  Regional tiles, mosaics and other projections are refused. Reproject
  first, for example with `gdalwarp`.
- Missing values (NaN, fill or nodata) raise an error unless
  `fill_value=<metres>` says what to put there. The count is recorded.
- `|elevation| > 100 km` is refused, which catches metres read as km.
- At most 2²⁶ pixels are read. Downsample larger DEMs first, for example
  with `gdal_translate -outsize`.

## Looking at it: `plot()` and `view()`

Both methods draw the model terrain the same way: the resolved coefficients
evaluated exactly on a 1° mesh (0.5° above `l_max` 44) with
`tropoi.spatial.transforms.host_synthesis`, a NumPy evaluation of the
backends' own basis. It is not interpolated from the grid and not
re-projected. GPU parity is asserted in `tests/test_terrain_import_gpu.py`.
When the terrain was imported in this session, the source raster is shown
too. It is area-averaged to the mesh spacing, labelled "before truncation",
and drawn on the same colour scale.

- **`plot()`** returns a Matplotlib figure with the source, the model, the
  model − source difference, and the provenance and preprocessing record.
- **`view(path=None, open_browser=True)`** writes one HTML file (system temp
  directory by default) and opens it. The page is self-contained: plain
  WebGL with the data embedded, and no network, server, Python or CUDA
  needed to open it. It has:
  - drag to rotate, wheel or ± to zoom, and arrow keys;
  - a model/source layer switch;
  - an elevation colour bar with units;
  - 30° graticules;
  - radial relief with a labelled exaggeration slider ("display only"; the
    model terrain is unchanged);
  - a matching flat map with hover readout. Click the map to centre the
    globe there.

  The colours are the repository's terrain colours (`YlOrBr`, light low to
  dark high), as in the saved-run figures. The interaction design follows
  the Maurer GEM Lab
  [Spherical Harmonics Explorer](https://maurer-gemlab.github.io/Potential_Field_Theory/demos/spherical_harmonics_explorer.html)
  (CC BY 4.0). No code is reused.

Previewing never needs CUDA once the terrain is resolved.
`PreparedPlanet.from_initial_state(path)` accepts a saved SWE state or a
custom run directory (its `initial_state/` is used) and draws its terrain on
any machine. Built-in presets work too:
`prepare_planet(planet, disc, TopographySpec.williamson5_cone())`.

## Into a run

Pass the `PreparedPlanet` as the topography of an SWE state. The run then
uses exactly these coefficients:

```python
from tropoi.run.custom import FluidLayerSpec, prepare_swe_state, run_initial_state

prepared = prepare_swe_state(planet, disc, FluidLayerSpec(3.71, 25000.0),
                             world, free_surface_anomaly=lambda lat, lon: 0 * lat)
prepared.save("states/mars")                 # topography_elevation_lm.npy + provenance
run_dir = run_initial_state("states/mars", duration_days=2.0)
```

The saved state stores the resolved coefficients
(`topography_elevation_lm.npy`) and the `imported` specification with its
provenance. A custom run copies both into the capsule's `initial_state/`.
`tropoi inspect` names the source file and its hash. Saved-run plots draw
the terrain from those coefficients, and nothing is ever re-read from the
source file. `free_surface_anomaly = 0` is the exact lake at rest over any
terrain, imported or not. The layer's mean depth must exceed the highest
resolved peak.

## Mars from MOLA

The USGS Astrogeology "Mars MGS MOLA DEM 463m" global mosaic
(<https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m>, file
`Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`) is a signed 16-bit raster:
elevation in metres relative to the areoid, about 46080 × 23040 pixels,
about 2 GB. It is far finer than any truncation used here. Average it down
to a global latitude–longitude grid with GDAL, for example 4 pixels per
degree:

```bash
gdalwarp -t_srs "+proj=longlat +R=3396190 +no_defs" -te -180 -90 180 90 -ts 1440 720 -r average -ot Float32 Mars_MGS_MOLA_DEM_mosaic_global_463m.tif mola_4ppd.tif
```

```bash
python examples/terrain_planet.py show mola_4ppd.tif --units m --smoothing 0.3
```

The result is a geographic GeoTIFF, which the TIFF reader places from its
tags. Use elevation relative to the areoid, not planetary radius: radius
includes the ~20 km polar flattening and would look like enormous terrain.
The MOLA download and conversion have not been run in this repository's
tests. The geographic and equirectangular GeoTIFF paths are tested with
generated files.

## Limits

- **SWE only.** Imported terrain reaches runs through SWE initial states.
  The PE core needs terrain band-limited at its dealiased cut (see
  `Topography.mountain(l_cut=...)`) and has no custom-state route, so PE
  runs cannot use imported terrain yet. The `tropoi run` CLI has no terrain
  file flag either; the Python API is the interface.
- **Grids.** Global regular latitude–longitude rasters only. No
  reprojection, no regional tiles, and no spherical-harmonic coefficient
  files (for example SHTOOLS models).
- **Not persisted.** The source raster itself is not stored. Saved states
  keep its provenance and the resolved coefficients, so a reloaded preview
  shows the model layer only.
- **Picking.** Hovering over the globe locates points on the undisplaced
  sphere. With strong relief exaggeration, the readout may refer to a point
  slightly offset from the one under the cursor. The flat map is exact.
