# Terrain

Fixed model topography is a resolved band-limited elevation field, owned by
`Topography` in `topography.py`; SWE and PE use its surface geopotential.
`raster.py` reads elevation data on CPU, `height_function.py` samples
functions/expressions, and `elevation_import.py` resolves them on the model
grid with recorded smoothing and projection residuals. `terrain_spectral.py`
and `tectonics.py` also supply decorative terrain, which does not drive
the fluid cores.

[IMPORT.md](IMPORT.md) contains the import/prepare/inspect/save workflow,
file/units/missing-data contracts and Mars MOLA recipe.
[Custom preparation](../../run/custom/README.md) manages the prepared object;
[archive](../../representation/archive/README.md) persists it;
[visualization](../../representation/visual/README.md) renders it.
PE terrain is limited at the nonlinear cut; SWE can retain full L.
The [MOLA balance evidence](../../../../docs/validation/mola_terrain_2026-10-22/README.md)
and [Mars perturbation](../../../../docs/validation/mars_evolving_2026-10-27/README.md)
are bounded worked examples, not general climate or convergence validation.
