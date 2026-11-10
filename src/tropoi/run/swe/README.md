# SWE command runner

`tropoi run swe` integrates the inviscid rotating shallow-water equations
in vector-invariant form, with prognostic vorticity `ζ`, divergence `δ` and
perturbation thickness geopotential `φ`, over an optional fixed bottom.
This directory resolves options, drives the integration and writes a run
capsule. The physics is `ShallowWaterModel` in
[`temporal/tendencies/shallow_water.py`](../../temporal/tendencies/shallow_water.py);
its formulation, terrain treatment, positivity rules and Williamson 5 setup
are in [SHALLOW_WATER.md](../../temporal/tendencies/SHALLOW_WATER.md).

| Module | Role |
| --- | --- |
| `config.py` | `SWERunConfig.resolve` layers explicit options over defaults and validates them. Defines the scenario and topography lists, and `w5_canonical()`, which is true only for the exact Williamson 5 planet and fluid constants. Standard library only, so `--help` and validation never load CUDA |
| `execution.py` | Runs inside the shared `run/lifecycle.py`: builds the model (planet, fluid layer, terrain), builds the named initial state, records backend and terrain-projection provenance in the manifest, then calls the runner |
| `runner.py` | `run_swe`: RK4 on the packed state (`temporal.integration.rk4_step_array`), validation after every accepted step, diagnostics, snapshots, figures |
| `model.py` | Re-exports the model builder from `temporal.models` under its historical path |

There are no presets, forcing, expression-based initial states or terrain
files here. Custom planets, fields and imported terrain go through
`--initial-state` and [run/custom](../custom/README.md).

## One step

1. RK4 advances the `(3, L+1, L+1)` state; nonlinear terms are formed on
   the backend's `fine` product sampling (always on for SWE) and cut to
   `floor(2L/3)`.
2. `validate_state` rejects non-finite coefficients, nonzero monopoles of
   `ζ`, `δ` or `φ`, and any non-positive depth on the state or product
   grids. A violation stops the run and the capsule is marked `failed`.
3. The recorder writes a row, and the next ceiling is
   `0.5 · cfl_length_scale / s_max`, with `s_max = max|u| + sqrt(max(Φ0 + φ))`:
   flow speed plus the gravity-wave speed of the deepest fluid.

## Options

| Option | Default | Choices |
| --- | --- | --- |
| `--scenario` | `williamson2` | `rest`, `gravity_wave`, `williamson2`, `williamson5` ([initialization](../../spatial/initialization/README.md)) |
| `--topography` | `flat` | `mountain`: one Gaussian mountain, band-limited at `L` (default 2000 m at 30°N 90°E, 20° wide; `--mountain-*` to change). Williamson 5 builds its own cone |
| Planet and fluid | 23.9345 h day, Earth radius, `g = 9.80616`, mean depth 3000 m | `--day-hours`, `--radius-earth-units`, `--gravity`, `--mean-depth` |
| Backend and truncation | geodesic resolution 4, `L = 21` | `--backend latlon` / `gauss-latlon` with `--nlat`, `--nlon` |
| Snapshots and figures | as for [BVE](../bve/README.md) | |

A flat-bottom run writes exactly the historical config dict, so its run
identity is unchanged by the terrain feature; non-flat terrain becomes part
of the identity. A Williamson 5 run with overridden constants is labelled
noncanonical in its manifest.

## Outputs

`config.json`, `manifest.json`, `swe_coeffs.npy`
`(n_snapshots, 3, L+1, L+1)` stacked `[ζ, δ, φ]`, `swe_snapshot_times.npy`,
`diagnostics/timeseries.csv`, and the requested figures
(`figures/`, `snapshots/{physical,spectral}/`, `swe_summary.png`).

```bash
tropoi run swe
tropoi run swe --topography mountain --mountain-height-m 2000 --days 2
tropoi run swe --scenario gravity_wave --day-hours inf --mean-depth 1000
```

## Limits

- The command runs inviscid. `ShallowWaterModel` accepts a `∇⁴`
  hyperdiffusion coefficient (`hyperdiffusion_nu4`, which damps the free
  surface rather than the thickness), but this command does not expose it,
  so a run has no small-scale dissipation.
- The 2/3 cut and geodesic quadrature give no general conservation
  guarantee; mass is exact in spectral space, while energy and potential
  enstrophy are measured
  ([diagnostics](../../representation/diagnostics/README.md)).
- Lake at rest over terrain is exact. The Williamson 5 comparison with
  MRI-JMA and the terrain tests are bounded worked examples, not a
  convergence study ([validation](../../../../docs/VALIDATION.md)).
