# Terrain import and planet preview — design

Date: 2026-10-21. Status: approved direction (user, in chat); concise by request.

## Intent

Load a planet's bottom topography from an elevation file (NetCDF, 16-bit PNG
and other grey bitmaps, global lat-lon TIFF) and *look at it* before running
anything: a static figure (`planet.plot()`) and an interactive 3D globe in the
browser (`planet.view()`). Mars is the showcase. The preview must show the
terrain the model actually uses: after resampling, optional smoothing and
spectral truncation, taken from the resolved coefficients, never rebuilt
separately.

## Goals

- One shared import path: format readers produce a normalized
  `ElevationRaster` (metres, latitude ascending, uniform global longitudes);
  one resampling/projection path turns it into a `Topography`.
- Explicit handling of orientation, coordinates, units, scale/offset,
  missing values and longitude wrapping; ambiguous or unsupported inputs are
  rejected with errors that say what to pass.
- `prepare_planet(planet, discretization, terrain)` returns a
  `PreparedPlanet` for imported rasters *and* built-in presets, with
  `.plot()` (Matplotlib) and `.view()` (self-contained HTML globe: rotate,
  zoom, elevation colour bar with units, graticules, labelled display-only
  relief exaggeration, flat map, model/source layers).
- Provenance (file name, SHA-256, reader settings, preprocessing settings,
  measured residuals) recorded in the `TopographySpec`; the resolved
  coefficients persist through the existing SWE initial-state convention
  (`topography_elevation_lm.npy`), so custom runs keep them in
  `initial_state/`.
- Viewing a saved state / prepared planet is CPU-only and needs no solver; the
  HTML opens from disk with no network, CUDA or server.

## Non-goals (now)

General GIS reprojection, regional tiles/mosaics, spherical-harmonic
coefficient files, PE wiring of imported terrain (PE needs the dealiased
`l_cut` path and has no custom-state route), a `tropoi` CLI flag, volume
rendering, time animation, an SH editor.

## Decisions

- **Layers.** `spatial/terrain/raster.py` (NumPy; Pillow / netCDF4 imported
  lazily) reads and resamples. `spatial/transforms/host_synthesis.py` is a
  NumPy evaluation of the repository basis (orthonormal, Condon–Shortley,
  `m >= 0`) for CPU previews. `Topography.from_raster` (CUDA) projects with the
  backend's own transform. `representation/visual/terrain.py` builds the
  preview and renders figures/HTML (template is package data). `run/custom/
  planet.py` holds the thin `PreparedPlanet`. Representation never imports run.
- **Normalization.** Rows are stored south→north; images are read north-up
  (row 0 = north), the universal raster convention. Bitmaps carry no
  georeferencing, so `lon_west_deg` (west edge of column 0) is required.
  Image values have no units, so `scale`/`offset` or `height_range` is
  required. NetCDF units come from the `units` attribute (m/km), CF
  `scale_factor`/`add_offset`/`_FillValue`/`missing_value` are honoured.
  TIFF georeferencing comes from GeoTIFF tags when present (geographic
  degrees, or equirectangular metres with standard parallel 0); vertical
  units must be given (`units=`). A duplicated seam column is dropped.
  Coverage must be global and spacing uniform. Missing values raise unless
  `fill_value=` is given.
- **Resampling.** Area-weighted (cos φ) block mean to a working grid of
  ≈ 180°/(2(l_max+1)) when the source is finer (anti-aliasing), then bilinear
  sampling at the state-grid points with longitude wrap. Recorded.
- **Smoothing.** Off by default (presets apply none; the preview makes Gibbs
  ripples visible). Opt-in `smoothing=s`: Hoskins-type filter
  `exp(-k (l(l+1)/(L(L+1)))^2)`, `k = -ln s`, i.e. amplitude `s` retained at
  `l_max`. `clip_below_m` opt-in (e.g. 0 for sea level).
- **Residual policy.** Report, not reject (as the W5 cone): truncation
  residual and model extrema are recorded and shown. All-zero terrain is
  rejected (use `flat`).
- **Viewer.** Hand-written WebGL (no three.js, no CDN), inspired by the
  Maurer GEM Lab spherical-harmonics explorer (CC BY 4.0; ideas only, no code
  copied). Both layers on one 1° (finer for high `l_max`) node grid; shared
  colour scale; source layer labelled "before truncation".

## Validation

Analytic Williamson-5 cone written to NetCDF / 16-bit PNG / TIFF, read back,
projected, compared with `Topography.williamson5_cone` (tolerance for raster
sampling and bilinear interpolation); peak location (30°N, 90°W) and height
checked so flips, 180° shifts and km/m mistakes fail. CPU tests for readers,
rejections, resampling, host synthesis (analytic Y_lm) and HTML/figure
generation; GPU parity of host synthesis; an SWE initial state over imported
terrain saved, reloaded and run briefly; the HTML exercised in a browser.

## Addendum, 2026-10-22: PE terrain and `tropoi run pe --terrain`

The non-goals above listed "PE wiring of imported terrain" and "a `tropoi`
CLI flag". At the user's request (same PR, commit `2a70164`), PE support was
added. It does not contradict the second non-goal:

- **What was added.** `prepare_planet(..., solver="pe")` band-limits the
  resolved terrain at the PE dealiased cut `floor(2 l_max / 3)`.
  `PreparedPlanet.save` writes a *prepared terrain* (resolved coefficients
  plus their record: `representation/archive/prepared_terrain.py`), and
  `tropoi run pe --terrain DIR` runs over it. The run adopts the terrain's
  discretization, records its digest in the scientific identity, uses the
  coefficients exactly, and keeps a verified copy in the capsule's
  `terrain/`.
- **What `--terrain` is not.** It does not read a raster. It takes a terrain
  that was already imported, resolved and inspected in Python
  (`read_elevation` → `prepare_planet` → `plot()`/`view()` → `save`). A
  flag that imported an elevation file directly would have to expose every
  reader option on the command line (units, scale/offset, georeferencing,
  missing values, smoothing) and would re-resolve terrain nobody looked at.
  That remains out of scope; the inspect-then-save route is the design.
- **Band limit.** Smoothing's retained amplitude applies at the band limit
  (`l_cut` for PE, `l_max` otherwise), recorded as
  `smoothing_retained_at_band_limit` / `smoothing_band_limit`; the earlier
  `smoothing_retained_at_lmax` is still read. PE-prepared terrain may also
  be used in SWE states; the preparation report records its cut.
