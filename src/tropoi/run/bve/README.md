# BVE command runner

`tropoi run bve` integrates the barotropic vorticity equation

```
∂ζ/∂t = -J(ψ, ζ + f) + ν∇²ζ,    ∇²ψ = ζ
```

for a single layer of incompressible flow on a sphere, with the state held
as vorticity coefficients. This directory resolves options, runs the
integration and writes a run capsule. The tendency itself is
`BarotropicVorticity` in
[`temporal/tendencies/barotropic.py`](../../temporal/tendencies/barotropic.py);
signs and conventions are in
[CONVENTIONS.md](../../spatial/CONVENTIONS.md#bve-equation-and-signs).

| Module | Role |
| --- | --- |
| `config.py` | `BVERunConfig.resolve` layers explicit options over a preset over defaults, validates them, resolves the snapshot schedule and emits the config dict that fixes the run's scientific identity. Standard library only, so `--help` and validation never load CUDA |
| `presets.py` | `rh4` (the [RH4 validation](../../../../docs/VALIDATION.md#rh4-benchmark): `L = 21`, geodesic resolution 4, 24 h day, fine products, one day) and `two-vortices-quick` (the README smoke run) |
| `execution.py` | Creates the run directory and provenance through the shared `run/lifecycle.py`, builds the planet, samples the preset on the grid and analyzes it, then calls the runner |
| `runner.py` | `run_bve`: RK4 under a state-adaptive CFL ceiling, diagnostics after every accepted step, snapshots, figures |
| `model.py` | Re-exports the planet builder from `temporal.models` under its historical path |

## One step

1. The model synthesizes `u = k × ∇ψ` and `∇(ζ + f)` on the backend's
   product sampling, multiplies them pointwise, analyzes once and keeps
   degrees up to `floor(2L/3)` ([operators](../../spatial/operators/README.md)).
   `f` is added in spectral space, and the degree-0 tendency is set to zero,
   so total circulation cannot drift.
2. `rk4_step` combines four such tendencies.
3. The [diagnostics recorder](../../representation/diagnostics/README.md)
   writes a row and returns the maximum speed.
4. The next timestep ceiling is `0.5 · cfl_length_scale / max|u|`, from
   that accepted state (600 s if the grid defines no length scale). The
   scheduler shortens steps to land exactly on snapshot times.

## Defaults worth knowing

| Option | Default | Note |
| --- | --- | --- |
| Backend | geodesic, resolution 4 | `--backend latlon` (or `gauss-latlon`) uses Gauss nodes, default 128×256 |
| Truncation | `L = 21` | |
| Rotation | `day_hours = inf` | **Non-rotating** unless set; the `rh4` preset uses 24 h |
| Viscosity | 0 | `ν∇²` diffusion; the CFL ceiling does not account for its stability |
| Products | `fine` | `coarse` reproduces the historical state-grid products |
| Snapshots | 5, evenly spaced, including start and end | `--n-snapshots N` (exact target times) or `--dt-snapshots` (legacy interval mode) |
| Figures | none | `--plot diagnostics|snapshots|summary`; images never change run identity |

Presets are listed in [initialization](../../spatial/initialization/README.md).
`--initial-state PATH` runs a state prepared with
[run/custom](../custom/README.md) instead; that state fixes the planet and
the initial vorticity, so options that would redefine them are refused.

## Outputs

Inside `RUN/` (layout and provenance in
[ARCHITECTURE.md](../../../../docs/ARCHITECTURE.md#output-capsules-and-provenance)):
`config.json`, `manifest.json`, `vorticity_coeffs.npy`
`(n_snapshots, L+1, L+1)`, `vorticity_grid.npy`, `bve_snapshot_times.npy`,
`diagnostics/timeseries.csv`, `diagnostics/spectra.npz`, and any requested
figures. An empty schedule still writes arrays with a zero-length time
axis. Saved runs reopen through the [Simulation API](../../temporal/SAVED_RUNS.md).

```bash
tropoi run bve --preset rh4
tropoi run bve --backend gauss-latlon --nlat 12 --nlon 24 --l-max 8 --days 0.02
```

## Limits

- Grid-sampled presets carry the analysis error of the grid's quadrature,
  and `random_low_l` is unseeded, so it is not reproducible
  ([initialization](../../spatial/initialization/README.md)).
- Conservation is measured per configuration. The 2/3 cut and geodesic
  quadrature give no general Galerkin conservation guarantee; on a rotating
  planet the invariant to watch is absolute enstrophy.
- Explicit RK4 has no implicit treatment of viscosity, so large `ν` can be
  unstable below the advective ceiling.
