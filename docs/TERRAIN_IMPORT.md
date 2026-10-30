# Terrain from elevation files

## Start with Mars, without downloading a DEM

From a repository checkout:

```bash
python examples/mola_mars.py quickstart --out out/mars --open
```

This uses the small prepared MOLA terrain already included with the
[published Mars example](runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a/terrain/).
It needs no CUDA, network access, PDS conversion or NetCDF dependency. It writes:

| Output | Use |
| --- | --- |
| `out/mars/mars_globe.html` | Open the interactive terrain globe offline. |
| `out/mars/mars_terrain.png` | A coloured preview; this is not an elevation input. |
| `out/mars/mars_heightmap.png` | An editable 16-bit grayscale elevation map, 180 × 90 pixels (2° cells). Black = −10000 m, white = +10000 m; row 0 is north, west edge −180°. |
| `out/mars/mars_heightmap.json` | The heightmap's source and decoding settings. |
| `out/mars/terrain/` | Prepared terrain, ready for `tropoi run pe --terrain out/mars/terrain`. |

The prepared terrain is `l_max` 21 with elevation through degree 14, exactly
as used in the example run. The grayscale image samples that **already
truncated relief**, not the original MOLA measurements. Its height quantization
is about 0.31 m per grey step. It is useful for learning and editing at this
resolution; it cannot recover individual mountains removed by truncation.

A short atmosphere run over the prepared terrain (CUDA):

```bash
tropoi run pe --terrain out/mars/terrain --scenario orographic_thermal_wave --radius-earth-units 0.53307 --day-hours 24.6229 --gravity 3.71 --r-dry 191.8 --cp-dry 735 --temperature 210 --surface-pressure 610 --levels 4 --dt-seconds 60 --days 0.02 --n-snapshots 3 --no-plots
```

This is a usage example. It demonstrates supplying terrain and evolving a
small perturbation with the stated settings; it does not establish accuracy
for arbitrary topographies. The longer run and its checks are documented
[separately](validation/mars_evolving_2026-10-27/README.md).
To choose a different terrain resolution from the source measurements, use
the [raw MOLA workflow below](#real-data-mars-from-mola).

## Other ways to make your planet's terrain

### A height expression or a Python function

`tropoi terrain prepare` turns a formula into prepared terrain without
editing any repository file. Give exactly one source:

```bash
tropoi terrain prepare --expression "1200*cos(radians(lat))**2*cos(2*radians(lon)) + 600*sin(radians(lat))" --resolution-deg 2 --lmax 21 --solver pe --out terrain/my_planet
tropoi terrain prepare --function "./my_terrain.py:height" --resolution-deg 2 --lmax 21 --solver pe --out terrain/my_planet
```

where `my_terrain.py` is your own file, anywhere on disk:

```python
import numpy as np

def height(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return 1200 * np.cos(lat)**2 * np.cos(2 * lon)
```

Both routes, and the Python API below, go through the same steps.
The source is sampled on a global grid (CPU) by `ElevationRaster.from_function`.
That gives an ordinary `ElevationRaster`, validated and traced like an imported file.
`prepare_planet` then resolves it on the model's discretization (CUDA).
`PreparedPlanet.save` writes it to `--out`.

**The height-function contract.**

- **Inputs.** The function is called **once** as `height(lat_deg, lon_deg)`
  with two float64 NumPy arrays of the same shape `(nlat, nlon)`:
  - latitude in degrees, north positive, at cell centres, south to north;
  - east longitude in degrees, at cell centres in [−180, 180)
    (`--lon-start-deg 0` gives [0, 360)).

  Write it with NumPy (`np.cos`, `np.where`), not `math`. A function that
  only handles scalars can use `--pointwise`, which calls it once per point.
  These are **degrees**: the initial-field callables of
  [custom initial states](CUSTOM_INITIAL_STATES.md) take radians.
- **Output.** Elevation in **metres**: an array of that shape, or anything NumPy broadcasts to it.
  - A scalar gives a constant height; `--expression "1000"` works.
  - An `(nlat, 1)` column gives a function of latitude only.
  - Integer and float results are accepted.
  - Complex, boolean, NaN/Inf, masked and wrongly shaped results are refused, with the reason.
- **Sampling is not truncation.** `--resolution-deg` (default 1, must divide
  180) is the spacing of the sampled grid. `--lmax` is the spectral
  truncation of the model terrain.
  - Preparation area-averages a sample finer than the truncation's working
    resolution, `180 / (2 (l_max + 1))` degrees (4.1° at `l_max` 21), then
    interpolates it to the model grid.
  - Sampling finer than that adds nothing the model keeps.
  - The command notes when the sample is coarser than the working resolution.
- **Solver.** `--solver pe` band-limits the terrain at the dealiased cut
  `floor(2 l_max / 3)` that `tropoi run pe --terrain` requires.
  `--solver swe` keeps the full truncation.
- **Other settings.**
  - Grid: `--nlat`/`--nlon` default to the Gauss grid
    `3(l_max+1)/2 + 1` (rounded up to even) × twice that, which is 34 × 68 at `l_max` 21.
  - Planet: `--radius-earth-units` (default 1) and `--day-hours` (default 24)
    are stored with the terrain and size the previews. The coefficients do not
    depend on them; a PE run takes its planet from its own options.
  - Optional `--smoothing` and `--clip-below-m` work as for imported files.

**Expressions** are checked against a narrow allowlist and evaluated by a
small interpreter; they are never run as Python. An expression may use:
- `lat` and `lon` in degrees, the constant `pi`, and numbers;
- `+ - * / % **`, comparisons (chained too, `-30 < lat < 30`), and `& | ~` to combine conditions;
- the NumPy functions `radians degrees sin cos tan arcsin arccos arctan
  arctan2 sinh cosh tanh exp log log10 sqrt abs minimum maximum where clip
  floor ceil sign hypot`.

Piecewise terrain uses `where`, for example
`"where((lat > 0) & (lon < 90), 800, -400)"`.

Anything else is rejected before evaluation, with a message naming it:
- statements and imports;
- attribute access such as `np.sin`;
- subscripts, comprehensions and lambdas;
- `and`/`or`/`not` and `if`-expressions;
- strings, keyword arguments and unknown names.

An expression that starts with a minus sign must be attached to the option,
`--expression="-500*sin(radians(lat))"`, so the shell parser does not take
it for an option.

**Python files** (`--function FILE.py:NAME`):
- **This executes the file, as `python FILE.py` would.** Use only code you
  trust: this route is not sandboxed.
- A relative path is resolved from the current working directory. Paths with
  spaces and Windows drive letters work: the text is split at its **final**
  colon (`"C:\terrain\my hills.py:height"`).
- `NAME` may be dotted (`Shapes.plateau`).
- Errors name the problem: a missing file, an exception while executing it,
  a missing name (listing the callables the file does define), something
  not callable, or an exception raised by the function.

**Without CUDA.** Sampling and validation run on the CPU.
`--sample-only` checks the input, prints the sampled range and stops,
writing nothing. The spectral preparation projects with the model's own
transform and needs CUDA, like every `prepare_planet` call. Without CUDA the
command says so.

**What is saved.** The output directory holds:
- the resolved coefficients;
- a record with the source's provenance:
  - the expression text and its SHA-256,
  - or the function file's name (not its path), the SHA-256 of the bytes
    executed, and the callable's name;
- the coordinate convention, the sampling spacing and longitude range, and
  the hash of the sampled values.

The function is never stored or pickled, and saved terrain never re-runs it.
After you delete or change the file, the terrain still reloads, previews and
feeds runs, exactly as prepared. An existing `--out` is never replaced:
`--overwrite` replaces only a previous prepared terrain, nothing else.

**Look at it and use it.**

- **Preview while preparing.** `--figure terrain.png` writes the source,
  resolved and difference maps; `--globe globe.html` writes the offline 3-D
  globe.
- **Preview later, without CUDA.**

  ```python
  from tropoi.run.custom import PreparedPlanet

  world = PreparedPlanet.load("terrain/my_planet")
  world.view()          # offline globe
  world.plot()          # resolved terrain and its provenance
  ```
- **Primitive equations (CUDA).** Run over the terrain with
  `tropoi run pe --terrain terrain/my_planet --scenario orographic_isothermal_rest`,
  plus the planet's options (the command prints a matching line).
- **Shallow water.** Prepare with `--solver swe`, then pass the loaded terrain
  to `prepare_swe_state(planet, disc, fluid, world, ...)` as shown
  [below](#shallow-water).

**From Python**, without the command line:

```python
import numpy as np
from tropoi.run.custom import (DiscretizationSpec, ElevationRaster,
                               PlanetSpec, prepare_planet)

def height(lat_deg, lon_deg):                  # degrees in, metres out
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return 1200 * np.cos(lat)**2 * np.cos(2 * lon) + 600 * np.sin(lat)

raster = ElevationRaster.from_function(height, resolution_deg=2)      # CPU
same = ElevationRaster.from_expression(
    "1200*cos(radians(lat))**2*cos(2*radians(lon)) + 600*sin(radians(lat))",
    resolution_deg=2)                                                  # CPU
world = prepare_planet(PlanetSpec.earth_like(),
                       DiscretizationSpec.gauss_latlon(21, 34, 68),
                       raster, solver="pe")                            # CUDA
world.save("terrain/my_planet")
```

`load_height_function("path/to/file.py:height")` returns a file's callable
and its provenance for `from_function(..., provenance=...)`; this is what
`--function` does.

### Runnable examples: function, image, NetCDF

[examples/custom_topographies.py](../examples/custom_topographies.py) has
three runnable routes to the same broad synthetic hills and basins. No
downloads are needed. Preparation uses CUDA; `--inputs-only` creates or reads
the source on CPU without preparing a model.

| Route | Command | What to change |
| --- | --- | --- |
| A height function | `python examples/custom_topographies.py function --out out/function` | Nothing to edit: add `--expression "..."` or `--function FILE.py:NAME` (as above) to replace the built-in hills, and `--resolution-deg` to change the sampling. |
| A painted grayscale heightmap | `python examples/custom_topographies.py image --out out/image` | Generates a small 16-bit PNG. Paint your own global map and supply `--file`, its height range, and longitude origin. |
| A scientific elevation file | `python examples/custom_topographies.py netcdf --out out/netcdf` | Generates a tiny CF NetCDF with coordinates and units. Supply `--file` and, if needed, `--variable` for your own file. Requires `netCDF4` (`pip install "palintropos[terrain]"`). |

For example, after editing the Mars heightmap above:

```bash
python examples/custom_topographies.py image --file out/mars/mars_heightmap.png --height-range -10000 10000 --lon-west -180 --out out/edited-mars
```

The image supplies elevation only. This script chooses a custom planet with
0.8 Earth radii and a 30 h rotation; edit those choices independently of the
terrain. Each route writes three things, then prints a short PE run command
with matching planetary parameters:
- `terrain/`;
- `terrain.png`, showing the source, the resolved relief and their difference;
- `globe.html`.

Use a new output directory for each preparation. Saved terrain can also feed
an SWE state as shown [below](#shallow-water).

When writing a grayscale heightmap yourself, encode metres with
`encode_grey16(heights, (black_m, white_m))`. It refuses heights outside the
range instead of letting a `uint16` cast wrap them, and it is the inverse of
`read_elevation(png, height_range=(black_m, white_m))`.

Use broad features when learning at coarse truncations. Increasing image
size adds no model detail once the source is adequately sampled: the retained
spherical-harmonic degrees determine the relief used by the solver. Inspect
the resolved preview, particularly after drawing narrow peaks or sharp edges.
A coloured or shaded map does not encode elevation unambiguously; supply
grayscale heights with an explicit scale, or an elevation file with units.

For a single idealized mountain, a parameterized preset is shorter still:

```python
from tropoi.run.custom import TopographySpec

world = prepare_planet(planet, disc,
                       TopographySpec.mountain(height_m=1000, lat_deg=25,
                                                lon_deg=60, width_deg=30),
                       solver="pe")
```

## Import your own elevation file

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

The same page, with a translucent atmospheric layer over the terrain, is what
`Simulation.view()` writes for a saved PE run; see
[ATMOSPHERE_VIEW.md](ATMOSPHERE_VIEW.md). Terrain and atmosphere share the
radius, the datum and the exaggeration.

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

**In the viewer.** A browser capture of `view()` on this terrain
(`l_max` 42, ×20 relief, colour bar in metres), with its exact settings and
the command that reproduces it, is in
[docs/validation/mola_terrain_2026-10-22](validation/mola_terrain_2026-10-22/README.md);
the README shows it.

**An evolving atmosphere over MOLA.** The orographic-rest check above starts
and stays at rest. [examples/mola_mars_evolving.py](../examples/mola_mars_evolving.py)
adds a small temperature perturbation (`--scenario orographic_thermal_wave`)
to the same balanced base, at `l_max` 21. It integrates one rotation period
alongside a resting control and a 30 s-step twin. This is an idealized dry-CO₂
perturbation experiment, not a climate simulation. Its protocol, results and
offline page are in
[docs/validation/mars_evolving_2026-10-27](validation/mars_evolving_2026-10-27/README.md),
and the run is published under `docs/runs/`.

**The 463 m USGS mosaic (alternative, untested here).** The USGS
Astrogeology "Mars MGS MOLA DEM 463m"
(<https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m>, about
2 GB, 46080 × 23040) exceeds the reader's 2²⁶-pixel limit. Average it to a
geographic grid first, for example with
`gdalwarp -t_srs "+proj=longlat +R=3396190 +no_defs" -te -180 -90 180 90 -ts 1440 720 -r average -ot Float32 <in>.tif mola_4ppd.tif`,
then read it with `units="m"`. This route was not run: GDAL is not
installed here and the MEGDR grids above cover the need.

## Limits

- **Interfaces.** `tropoi terrain prepare` takes a height expression or a
  Python function, not an elevation file. Import and inspect files in Python.
  Either way, pass the saved prepared terrain to `tropoi run pe --terrain`,
  or a saved SWE state to `--initial-state`.
- **Height functions.**
  - Sampling is on global cell centres only; there is no node grid that
    includes the poles.
  - A `--function` file runs with your permissions; it is not sandboxed.
  - Expressions are limited to the allowlist above; anything longer belongs
    in a function file.
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
