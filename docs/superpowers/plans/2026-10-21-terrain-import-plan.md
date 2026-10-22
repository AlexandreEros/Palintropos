# Terrain import and planet preview — plan

Spec: `docs/superpowers/specs/2026-10-21-terrain-import-design.md`.
Branch `feat/terrain-import` from `origin/main` (7991fc7).

1. `spatial/terrain/raster.py`: `ElevationRaster`, `read_elevation` (NetCDF,
   PNG/bitmap, TIFF), normalization, `coarsened`, `sample`,
   `resample_to_points`. CPU tests with synthetic files.
2. `spatial/transforms/host_synthesis.py`: NumPy synthesis on lat/lon nodes.
   CPU test vs analytic Y_lm; GPU parity vs `PointSetSphericalHarmonics`.
3. `Topography.from_raster` + `"imported"` preset (+ `TopographySpec`
   `imported`, run-config / inspect / presets branches). Sync tests updated.
4. `representation/visual/terrain.py` + `terrain_globe.html`: preview data,
   `plot_terrain`, `write_globe_html`. Package data in `pyproject.toml`.
5. `run/custom/planet.py`: `prepare_planet`, `PreparedPlanet` (`plot`,
   `view`, `from_initial_state`); `prepare_swe_state` accepts it.
6. Tests: W5 round trip (GPU), SWE state save/load/run (GPU), CPU-boundary
   check for previewing a saved state.
7. `examples/terrain_planet.py` (generated reference terrain; Mars recipe),
   docs (`docs/TERRAIN_IMPORT.md`, CUSTOM_INITIAL_STATES, README pointer),
   architecture snapshot refresh, full test run, browser check.
