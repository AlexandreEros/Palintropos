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

The same terrain then feeds shallow-water states, or primitive-equation
runs (`solver="pe"`, then `tropoi run pe --terrain`); see
[Into a run](#into-a-run). [Real data: Mars from MOLA](#real-data-mars-from-mola)
records what the path does with actual MOLA topography.

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
[Real data: Mars from MOLA](#real-data-mars-from-mola).

## The pipeline

| Step | Where | What happens |
| --- | --- | --- |
| Read | `read_elevation` (CPU) | A format reader interprets the file. Every reader returns the same normalized `ElevationRaster`: metres, rows south to north, uniform global longitudes. The source's name, SHA-256 and every interpretation are recorded in `raster.provenance`. |
| Resample | `ElevationRaster.resample_to_points` (CPU) | A source finer than the working resolution `180° / (2 (l_max + 1))` is area-averaged to it, then sampled bilinearly at the model's state-grid points. Longitudes wrap, and each pole is single-valued. |
| Clip (optional) | `clip_below_m=` | For example `0` puts an ocean floor at sea level. |
| Project | `Topography.from_raster` (CUDA) | Analysis with the backend's own transform, as for the presets: at the full truncation `l_max`, or band-limited at `l_cut = floor(2 l_max / 3)` with `solver="pe"`. The truncation residual is measured on the coefficients kept. |
| Smooth (optional) | `smoothing=s` | Hoskins-type filter `exp(-k (l(l+1)/(L(L+1)))²)`, `k = -ln s`, where `L` is the band limit: `l_cut` for the PE core, `l_max` without a cut. It keeps amplitude `s` at `L` and leaves `l = 0` untouched. Off by default. Recorded as `smoothing_retained_at_band_limit` with the degree in `smoothing_band_limit`; terrain saved earlier carries the same value as `smoothing_retained_at_lmax`, which is still read. |
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

### Shallow water

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

A terrain prepared for the PE core (`solver="pe"`) is also accepted here: it
is simply terrain with fewer modes, zero above `l_cut`. Its coefficients are
reused unchanged, and the preparation report states the band limit
(`band_limit_l_cut`, and a line in `report.lines()`). The report is kept in
the saved state's provenance, so the cut stays traceable after saving and
reloading.

### Primitive equations

The PE core needs terrain band-limited at its dealiased product cut
`floor(2 l_max / 3)`, so resolve it for that core, save it, and run over the
saved directory:

```python
world = prepare_planet(PlanetSpec.earth_like(day_hours=24.6229, radius_earth_units=0.53307),
                       DiscretizationSpec.gauss_latlon(42, 66, 132),
                       read_elevation("megt_4ppd.nc"), solver="pe")
world.view()                       # the band-limited terrain the PE core will use
world.save("terrain/mola_pe")      # terrain.json + topography_elevation_lm.npy
```

```bash
tropoi run pe --terrain terrain/mola_pe --scenario orographic_isothermal_rest --radius-earth-units 0.53307 --day-hours 24.6229 --gravity 3.71 --r-dry 191.8 --cp-dry 735 --temperature 210 --surface-pressure 610 --levels 4 --dt-seconds 300 --days 0.25
```

How `--terrain` behaves:

- **Discretization.** The run adopts the terrain's `l_max` and grid unless
  they are given on the command line, in which case they must agree. A
  terrain resolved for the SWE (full truncation) is refused with a pointer
  to `solver="pe"`.
- **No re-projection.** The stored coefficients are used exactly. They
  depend only on the discretization, not on the planet's size or rotation:
  measured bit-identical on the Gauss grid and within 3e-11 m on the
  geodesic grid for an Earth-sized and a Mars-sized planet. The run's
  planet comes from its own options as usual (`--radius-earth-units`,
  `--day-hours`), and `--gravity` converts elevation into
  `Phi_s = g h_s`.
- **Identity.** The run configuration records `topography` (the terrain's
  preset, for example `imported`), `topography_parameters` (its
  provenance), `terrain_digest`, `terrain_l_cut` and `gravity`. All are
  part of the scientific hash. The directory path is not: the same terrain
  saved in two places gives the same run id.
- **Capsule.** The run copies the directory into `terrain/` and checks the
  copy's digest. `tropoi inspect` reports the prepared terrain, and
  `tropoi plot` rebuilds the model from `terrain/`. A capsule whose
  `terrain/` was altered is refused for plotting.
- **Same path as the presets.** A Gaussian mountain prepared this way gives
  a run bit-identical to `--topography mountain` with the same parameters
  (`tests/test_pe_prepared_terrain.py`).

`PreparedPlanet.save(path)` / `PreparedPlanet.load(path)` store and read
any resolved terrain (format: `representation/archive/prepared_terrain.py`).
`load` also accepts a PE run directory, so `PreparedPlanet.load(run).view()`
shows the terrain a past run used, on any machine.

## Real data: Mars from MOLA

This section is **real-data validation**. It is separate from the
synthetic Mars-like example above, whose terrain is generated and
resembles Mars only in its shapes.

**Data.** The Mars Global Surveyor MOLA Mission Experiment Gridded Data
Records (dataset `MGS-M-MOLA-5-MEGDR-L3-V1.0`) from the NASA PDS
Geosciences Node,
<https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/>:

| File | Grid | Size | SHA-256 of the `.img` |
| --- | --- | --- | --- |
| `meg004/megt90n000cb.img` + `.lbl` | 4 px/deg, 1440 × 720 | 2.1 MB | `25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c` |
| `meg016/megt90n000eb.img` + `.lbl` | 16 px/deg, 5760 × 2880 | 33 MB | `d18d9b9ab8c5516d02e157dd2cde0f1d0d160c21940e953ba22391269a545e7b` |

Each sample is the median MOLA topography in its bin: planetary radius
minus areoid radius, in metres, planetocentric, east longitude. The files
are raw PDS3 images (big-endian int16) that `read_elevation` does not read
directly. [examples/mola_mars.py](../examples/mola_mars.py) `convert`
parses the PDS3 label, checks every property it relies on (sample type and
bits, unit, simple-cylindrical projection, east-positive planetocentric
longitude, global extent, size), and writes CF NetCDF. From there it is an
ordinary import.

```bash
python examples/mola_mars.py convert megt90n000cb.lbl mola/megt_4ppd.nc
python examples/mola_mars.py validate mola/megt_4ppd.nc --fine mola/megt_16ppd.nc
python examples/mola_mars.py show mola/megt_4ppd.nc --lmax 42
python examples/mola_mars.py prepare-pe mola/megt_4ppd.nc terrain/mola_pe
```

**Measured (2026-10-21, Windows, GeForce MX110, Gauss lat-lon grids, Mars-sized planet).**

- **Extremes.** The imported extremes equal the label's stated extremes for
  the whole data set: −8068 / 21134 m at 4 px/deg and −8177 / 21171 m at
  16 px/deg. So nothing is lost or rescaled in conversion, units or
  reading.
- **Placement.** The highest sample, 21134 m, is at (17.4° N, 133.1° W) on
  Olympus Mons. The lowest, −8068 m, is at (32.9° S, 62.1° E) on the
  Hellas floor. The local summits of Ascraeus, Arsia and Elysium Mons lie
  0.1°, 1.8° and 0.3° from their nominal positions; Olympus Mons is 1.4°
  away. A north–south flip or a 180° longitude shift would move each by
  tens of degrees.
- **Dichotomy.** Area means are −2180 m in the north and +1077 m in the
  south (global −552 m).
- **Resolution.** The model terrain is the same from either source grid:
  coefficients from 16 px/deg and 4 px/deg differ by 3.0e-3 to 4.6e-3
  relative L2. Area-averaging to the working resolution makes the result
  independent of the source's sampling.

Per truncation:

| `l_max` | truncation residual | model range (m) | Olympus Mons (m) | Hellas (m) | degree-1 north–south term (m) |
| --- | --- | --- | --- | --- | --- |
| 21 | 0.123 | −7495 .. 9019 | 7425 | −7322 | −3005 |
| 31 | 0.090 | −7393 .. 11606 | 11544 | −7038 | −3001 |
| 42 | 0.074 | −7326 .. 16500 | 16500 | −7140 | −3004 |

The degree-1 term `a_10 · sqrt(3/4π)` is the amplitude of the
`sin(latitude)` component of the elevation. At −3.0 km it agrees with the
roughly 3 km southward offset of Mars's centre of figure from its centre of
mass (Smith et al. 1999, *Science* 284, 1495). That is an independent
physical check on the whole import and projection chain. Olympus Mons
needs high degrees: it reaches 16.5 of 21.1 km at `l_max` 42.
`l_max` 63 did not fit on the 2 GB MX110: the dense Gauss transform matrix
alone is 0.64 GB, and its working copies exceed the device memory.

**PE run over MOLA terrain.** The terrain was prepared with `solver="pe"`
(`l_max` 42, cut 28) and run with the command above: orographic isothermal
rest with Mars-like constants (CO₂ gas constants, 210 K, 610 Pa reference,
g = 3.71 m/s²), 4 levels, 72 steps of 300 s. After 6 h:

- maximum wind 1.1e-10 m/s and mass drift 3.3e-16;
- temperature unchanged at 210 K (|T − 210 K| ≤ 1.1e-11 K at every level);
- surface pressure 210 Pa (over the resolved Olympus Mons) to 1230 Pa
  (Hellas), as the hydrostatic balance requires.

The capsule plotted from its `terrain/` copy. This is a balance check of
the terrain coupling on real relief, not a Mars climate simulation.

**The 463 m USGS mosaic (alternative, untested here).** The USGS
Astrogeology "Mars MGS MOLA DEM 463m"
(<https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m>, about
2 GB, 46080 × 23040) exceeds the reader's 2²⁶-pixel limit. Average it to a
geographic grid first, for example with
`gdalwarp -t_srs "+proj=longlat +R=3396190 +no_defs" -te -180 -90 180 90 -ts 1440 720 -r average -ot Float32 <in>.tif mola_4ppd.tif`,
then read it with `units="m"`. This route was not run: GDAL is not
installed here and the MEGDR grids above cover the need.

## Limits

- **Interfaces.** The `tropoi run` CLI has no elevation-file flag: import
  and inspect terrain in Python, then pass the saved prepared terrain to
  `tropoi run pe --terrain`, or the saved SWE state to `--initial-state`.
- **Grids.** Global regular latitude–longitude rasters only. No
  reprojection, no regional tiles, no spherical-harmonic coefficient files
  (for example SHTOOLS models), and no raw PDS3 images (convert them, as
  `examples/mola_mars.py` does).
- **Source raster not persisted.** The source raster itself is not stored.
  Saved terrain keeps its provenance and the resolved coefficients, so a
  reloaded preview shows the model layer only.
- **Truncation on small GPUs.** Resolution uses the model's dense
  transforms. On a 2 GB GPU, Gauss grids up to `l_max` 42 work and 63 does
  not.
- **Picking.** Hovering over the globe locates points on the undisplaced
  sphere. With strong relief exaggeration, the readout may refer to a point
  slightly offset from the one under the cursor. The flat map is exact.
