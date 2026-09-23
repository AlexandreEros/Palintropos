# Architecture

This document describes how Palintropos is put together today: package layout, the
two grid backends, the spectral transform flow, the run-capsule/provenance
model, and how to add or compare a backend.

> This file reflects the current backend/product-space design. The mermaid
> [class diagram](class-structure.md) and [call diagram](call-structure.md) are
> also current; equation conventions and normalizations live in
> [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md). Dated audit records
> ([KNOWN_RISKS.md](KNOWN_RISKS.md), [VALIDATION_PLAN.md](VALIDATION_PLAN.md))
> carry their own snapshot commits — prefer this file and the current tests
> where they disagree.

## Package layout

The implementation is organized by responsibility into three layers, plus
the orchestration that wires them together:

```text
src/tropoi/
├── spatial/             what a state is and how it is discretized
│   ├── environment.py       PlanetaryParameters: the prescribed environment (CPU-only)
│   ├── williamson5.py       Williamson (1992) case-5 prescribed world constants (stdlib)
│   ├── planet.py            Planet facade: environment + discretization resources + decorative elevation
│   ├── terrain/             model topography (Topography) and decorative terrain generation
│   ├── grids/               geodesic, uniform and Gauss lat-lon geometries, quadrature, interpolation
│   ├── transforms/          spherical-harmonic transforms; cuda/ holds the .cu kernel sources
│   ├── spherical_backend.py backends and the ProductSpace nonlinear-product sampling
│   ├── operators/           SpectralOperators (Laplacian, derivatives, velocity, Jacobian)
│   ├── sigma_coordinate.py  PE vertical coordinate and column operators
│   ├── states/              BVE / SWE / PE state dataclasses over packed coefficient stacks
│   ├── modes.py             immutable field specifications and read-only host coefficient views
│   ├── initialization/      named preset initial states (bve, swe, pe)
│   └── truncation.py        the 2/3 product cut and the preset support guards (stdlib)
├── temporal/            how states evolve, and how a saved evolution is read
│   ├── integration.py       IntegrationScheduler, advective CFL arithmetic, RK4, driver loop (stdlib)
│   ├── tendencies/          BVE, SWE and PE tendency implementations (coupled dissipation included)
│   └── simulation.py        read-only Simulation / Snapshot over a small storage protocol
├── representation/      what is recorded and shown
│   ├── diagnostics/         per-core recorders, CSV/NPZ records and diagnostic figures
│   ├── archive/             capsule reader (open_simulation), versioned schema, run-capsule writer
│   └── visual/              figure fields/specs/timelines/renderers, per-core compositions, Snapshot.plot adapter
├── run/{bve,swe,pe}/    orchestration: resolved run configuration and the runners
├── cli/                 tropoi (main.py); aeolus/psx-bve/psx-gen/psx-recompile compatibility entry points
└── numerics/ physics/ planet/ viz/ support.py
                         compatibility import paths only (see "Compatibility paths")
tests/                   asserting GPU tests plus standalone audit scripts
docs/                    architecture, validation records, and tracked README assets
```

The saved-run interface (`spatial.modes`, `temporal.simulation`,
`representation.archive`) is CPU-only; `representation.visual` is imported
only by an explicit `Snapshot.plot` or by a runner rendering its products.
It is documented in [SAVED_RUNS.md](SAVED_RUNS.md).

### Dependency direction

`spatial` imports only `spatial`; `temporal` imports `spatial` and
`temporal`; `representation` imports all three layers. Orchestration
(`run`, `cli`) sits above them. Mathematical helpers stay below their
consumers: the product cut lives in `spatial.truncation`, which the
operators, the tendencies, the diagnostics and the configuration layer all
consume, so no spatial operation depends on a tendency implementation. The
initializers receive a model object but reference its class only for type
annotations. `tests/test_layout_compat.py` enforces this direction by
parsing every import, with exactly two documented exceptions:

- `representation.archive.writer` (the run-capsule writer) imports
  `scientific_config_subset` from the stdlib-only `run.bve.config`: the
  run-id hash is defined by the configuration layer's scientific subset.
- `representation.visual.snapshot` lazily imports the solver modules'
  model builders (`cli.bve.build_planet`, `cli.swe.build_swe_model`,
  `cli.pe.build_pe_model`) when `Snapshot.plot` must reconstruct model
  resources from a persisted configuration.

Prescribed environment and discretization are separate objects.
`PlanetaryParameters` (radius, rotation, mass) is importable without CuPy
from `spatial.environment` (the legacy `tropoi.planet` paths still load the
CUDA facade through that package's initializer, as they always did); grids, transforms, backends and their cached
`ProductSpace`s are discretization resources built from it and shared by
every field and snapshot. `Planet` remains the facade that bundles them
for the cores. Radius, gravity and rotation defaults are unchanged.

### Canonical imports

New code imports from the layer modules, for example:

```python
from tropoi.spatial.environment import PlanetaryParameters
from tropoi.spatial.planet import Planet
from tropoi.spatial.truncation import product_truncation_cut
from tropoi.spatial.states.shallow_water import ShallowWaterState
from tropoi.spatial.initialization.swe import make_swe_ic
from tropoi.temporal.tendencies.shallow_water import ShallowWaterModel
from tropoi.temporal.integration import rk4_step_array
from tropoi.representation.archive import open_simulation
from tropoi.representation.diagnostics.bve import plot_diagnostics
```

The initializers of `spatial`, `temporal`, `representation` and their
subpackages import nothing, so importing one module never drags in CuPy or
Matplotlib through a parent package. The one exception is
`representation.archive`, whose initializer exposes the CPU-only reader
API (`open_simulation`, the schema types).

### Compatibility paths

Every pre-reorganization module path still imports, for the remainder of
the 0.1 series; removing one requires an explicit release decision. Each
old module file calls `tropoi._compat.alias_module`, which binds the old
name to the canonical module object in `sys.modules`: there is one
implementation, `old.Class is new.Class`, module state is shared, and
monkeypatching either name patches both. Canonical modules never import
these paths. The legacy package initializers `tropoi.numerics`,
`tropoi.planet` and `tropoi.viz` keep their historical re-exports (the same
objects), and names that moved out of a module remain importable from it
(the state classes from the tendency modules, the preset guards and
Williamson-5 constants from `run.swe.config` / `run.pe.config`).

| Legacy path | Canonical module |
| --- | --- |
| `tropoi.support` | `tropoi.spatial.truncation` |
| `tropoi.planet.planetary_parameters` | `tropoi.spatial.environment` |
| `tropoi.planet.planet` | `tropoi.spatial.planet` |
| `tropoi.planet.{elevation_data,terrain_spectral,tectonics}` | `tropoi.spatial.terrain.*` |
| `tropoi.physics.topography` | `tropoi.spatial.terrain.topography` |
| `tropoi.physics.sigma_coordinate` | `tropoi.spatial.sigma_coordinate` |
| `tropoi.numerics.{grid_base,grid,geodesic_grid,latlon_grid,cartesian_to_spherical,integration,grid_interpolation}` | `tropoi.spatial.grids.*` |
| `tropoi.numerics.{spherical_harmonics,fast_geodesic_sh,optimized_geodesic_sh,compute_optimal_weights}` | `tropoi.spatial.transforms.*` |
| `tropoi.numerics.cuda.cuda_utils` | `tropoi.spatial.transforms.cuda.cuda_utils` |
| `tropoi.numerics.spherical_backend` | `tropoi.spatial.spherical_backend` |
| `tropoi.numerics.{spectral_operators,differential_operators_spherical}` | `tropoi.spatial.operators.*` |
| `tropoi.run.{bve,swe,pe}.initial_conditions` | `tropoi.spatial.initialization.{bve,swe,pe}` |
| `tropoi.run.engine` | `tropoi.temporal.integration` |
| `tropoi.physics.{barotropic,shallow_water,primitive_equations}` | `tropoi.temporal.tendencies.*` (states in `tropoi.spatial.states.*`) |
| `tropoi.run.{bve,swe,pe}.diagnostics` | `tropoi.representation.diagnostics.{bve,swe,pe}` |
| `tropoi.run.bve.io` | `tropoi.representation.archive.writer` |
| `tropoi.run.{bve,swe,pe}.visualization` | `tropoi.representation.visual.{bve,swe,pe}` |
| `tropoi.run.pe.snapshot_visualization` | `tropoi.representation.visual.pe_snapshots` |
| `tropoi.viz.*` | `tropoi.representation.visual.*` |

`tropoi.run.bve.barotropic_vorticity` remains the older re-export module
for the BVE core and state. Dated records (audits, handoffs, validation
reports) and descriptive strings already written into manifests keep the
paths of their time; read them through this table. The package, the
distribution and every command (`tropoi`, `aeolus`, `psx-bve`, `psx-gen`,
`psx-recompile`) are unchanged.

### CPU and GPU requirements

- **CPU only** (no CuPy import, no CUDA initialization, no Matplotlib):
  CLI help, `list`, configuration resolution and validation,
  `tropoi inspect` with or without `--snapshot/--field`, `open_simulation`
  and host coefficient views, `spatial.truncation`,
  `spatial.environment`, `spatial.williamson5`, `spatial.modes`,
  `temporal.integration`, `temporal.simulation`, and coefficient-space
  (`"spectral"`) BVE/SWE snapshot plots. Fresh-interpreter tests enforce
  these boundaries.
- **CUDA required**: running any solver; anything that builds grids,
  transforms, backends, operators, states, initial conditions or
  tendencies; the per-step diagnostics recorders; and every physical-space
  plot (fields are synthesized through the run's own transform).
- **Packaged CUDA sources**: the kernels are package data of
  `tropoi.spatial.transforms.cuda` and are read through
  `importlib.resources`, so an installed wheel compiles them without a
  source checkout.

Numerical conventions matter more than style (there is no configured formatter
or linter yet): use SI units; keep live arrays on the GPU as CuPy arrays;
preserve the `m >= 0` coefficient convention; put grid-specific decisions behind
`GridGeometry`/`SphericalGridBackend`; reject unsupported product modes instead
of silently falling back; and add asserting backend-parity tests for numerical
changes.

## Spectral transform flow

The dependency direction is deliberately narrow:

```text
GridGeometry
    +
SphericalHarmonicTransform
    +
SphericalGridBackend
    +
ProductSpace
    →
SpectralOperators
    →
BarotropicVorticity
    →
Runner / Diagnostics
```

- `GridGeometry` owns points, areas/weights, shape, and the geometry-specific
  CFL length proxy.
- The transform maps between backend sampling and the shared dense `(l,m)`
  coefficient layout.
- `SphericalGridBackend` pairs geometry and transform, caches a `ProductSpace`,
  and is the sole authority on nonlinear-product sampling.
- `SpectralOperators` contains Laplacian, derivative recurrence, velocity, and
  pseudospectral Jacobian operations without branching on grid family.
- `BarotropicVorticity` (`temporal/tendencies/barotropic.py`) owns the equation
  and exact spectral Coriolis mode; the runner (`run/bve/runner.py`) drives
  `temporal/integration.py` and owns snapshot storage, capsules, and
  diagnostics recording.
- `run/bve/config.py` (`BVERunConfig`) owns configuration resolution for the
  CLI: preset layering (explicit flag > preset > ordinary default), snapshot
  scheduling, and plot selection — all validated before CUDA initialization.
  It also owns the two CPU-testable time-stepping seams: `advective_cfl_timestep`
  (the sole advective-CFL arithmetic) and `IntegrationScheduler` (time + snapshot
  bookkeeping, independent of CuPy/physics), which the runner steps one event at
  a time so a ceiling recomputed from each accepted state governs the next step.
  The runner consumes the resolved explicit snapshot schedule and clips
  integration steps to land exactly on the requested output times.

The runner selects its execution semantics from an explicit `snapshot_mode`
passed at its boundary (never inferred from whether a schedule was supplied):

- **Count mode** (`--n-snapshots`, canonical) consumes the authoritative
  `snapshot_times` with **exact target-time** semantics. Every scheduled time,
  including `t_end` for `N >= 1`, is landed on exactly by clipping the final
  step and snapping off floating-point drift; every representably positive
  pre-target residual is integrated, never absorbed by a tolerance. If a
  positive CFL-limited step is too small to advance the floating-point clock,
  the run aborts rather than enlarging the step past the CFL ceiling. Thus the
  stored snapshot times and final diagnostic time equal the requested targets
  exactly, including deliberately non-aligned durations such as
  `t_end = 600.0000003 s`.
- **Interval mode** (`--snapshot-interval-seconds` / legacy `psx-bve`)
  reproduces the historical countdown bit-for-bit: it stores `t = 0` and each
  interval boundary, and stops within the legacy `1e-6 * dt_snapshots`
  tolerance — so the final state is stored only when the duration is a multiple
  of the interval. This historical stopping behavior is preserved intentionally
  and is *not* replaced by the exact-`t_end` guarantee.

The prognostic variable is relative vorticity, represented by complex
spherical-harmonic coefficients for `m >= 0`:

```text
∂ζ/∂t + J(ψ, ζ + f) = ν∇²ζ + F
∇²ψ = ζ
f = 2Ω sin φ
u = k × ∇ψ
```

The CLI uses classical explicit RK4, exact spectral Laplacian eigenvalues,
and no forcing (`F=0`). The timestep policy is a **state-adaptive advective
CFL ceiling** (`0.5 · cfl_length_scale / max|u|`): formed from the initial
velocity for the first step and recomputed from every accepted state
thereafter, reusing the max speed the per-step diagnostics record already
produces. Individual steps are shortened only to land exactly on a requested
snapshot time or on `t_end`, never lengthened past the ceiling. This governs
only the advective condition — not explicit-viscosity (`ν∇²`) stability, and
it is not embedded-error RK adaptivity. SI units and a perfect spherical
surface are used throughout. Equation conventions and normalizations are
detailed in [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md).

User-facing configuration is fully validated *before* CuPy is imported or
CUDA is initialized: the CLI parses with all defaults set to `None`, and
[`BVERunConfig.resolve`](../src/tropoi/run/bve/config.py) layers
explicit values over the selected preset over ordinary defaults, checks
finiteness/domains, and resolves the snapshot schedule and plot selection.
Ordinary user errors (invalid backend, negative viscosity, NaN duration,
misaligned `--plot summary --n-snapshots 0`) therefore fail with a clear
parser message instead of a mid-run CUDA traceback.

## Backends

Both backends produce the same coefficient layout and enter the BVE through
`SphericalGridBackend`. Keeping both exposes grid-orientation and quadrature
errors that a single implementation could hide.

- The **geodesic backend** exercises arbitrary point-set transforms on a
  quasi-uniform icosahedral mesh. It is the path toward geometry-independent
  spherical numerics and avoids the conceptual pole concentration of a
  structured grid.
- The **Gauss lat–lon backend** provides a mathematically controlled reference:
  tensor-product Gauss–Legendre × periodic-longitude quadrature gives
  floating-point-exact analysis for adequately band-limited fields and exact
  quadratic-product projection at the documented dimensions. It is also much
  faster and smaller at the present dense-transform resolutions.

The measured quality gap between the two is documented in
[VALIDATION.md](VALIDATION.md); the Gauss backend is the stronger quadrature
reference and the geodesic backend is experimental.

### Nonlinear product quadrature

Backend-owned `ProductSpace` sampling handles the nonlinear term. The
distinction between the backends is scientifically important:

- **Geodesic `fine`:** synthesize derivatives on a resolution-`(r+1)` geodesic
  co-grid, form the pointwise Jacobian, analyze with Voronoi weights, and then
  apply the spectral cut. This is an empirically useful **overresolved product
  quadrature**, not mathematically exact dealiasing.
- **Gauss lat–lon `fine`:** a product of two degree-`L` fields has degree at
  most `2L`; projecting it against harmonics through degree `L` requires
  integrating degree `3L`. The periodic longitude rule therefore needs
  `nlon >= 3L + 1`, while Gauss–Legendre exactness through degree
  `2*nlat - 1` requires `nlat >= ceil((3L + 1)/2)`. The code uses at least
  those sizes (or retains a larger state grid), making quadratic-product
  projection exact up to floating-point error for band-limited inputs.

For the state transform alone, the corresponding requirements are
`nlat >= L + 1` and `nlon >= 2L + 1`. The code warns rather than aborts when a
state grid is under-resolved.

## Output capsules and provenance

Visualization data and rendering are separated under `representation/visual/`: `fields.py`
represents latitude-longitude scalar fields and unpacked triangular
spherical-harmonic fields; `normalization.py` and `specs.py` describe numeric
scaling, scalar maps, streamline maps, complex-coefficient maps, and generic
panel groups without plotting objects; `renderers.py` is the
small backend protocol; `timeline.py` owns timestamped figure sequences,
cross-frame normalization, representative-frame selection, deterministic
filenames, and transactional whole-product publication; and
`matplotlib_renderer.py` is the initial backend. The per-core
adapters (`bve.py`, `swe.py`, `pe.py`, `pe_snapshots.py`) choose their own physical fields, labels, panels, and layouts and
reconstruct frames from persisted run arrays. PNGs are written to
same-directory temporary siblings and atomically replaced, so only complete
images can become run artifacts.

Physical snapshot frames use backend-neutral panel-group metadata to display
`Prognostic state` and `Diagnostic fields` headings with a subtle separator;
the renderer knows only group geometry and styling. BVE places relative
vorticity in the prognostic group and derives streamfunction and velocity
streamlines from the same state. SWE places relative vorticity, horizontal
divergence, and `h' = Phi'/g` in the prognostic group and derives velocity
streamlines instantaneously. History-dependent diagnostics remain in the
diagnostic time-series products rather than snapshot frames.

Complex spectral panels use the shared `SpectralCoefficientMapSpec` with an
explicit encoding. The default `phase-magnitude` encoding maps the fixed
coefficient phase interval `[-pi, pi)` to cyclic hue and maps amplitude dB
relative to the timeline-wide valid-coefficient maximum onto saturation at
value 1 (default floor `-60 dB`), so weak coefficients fade
toward white without losing phase information in strong coefficients. The
optional `magnitude` encoding retains the scalar magnitude view. Both use the
complete persisted `0 <= l,m <= l_max` array extent in every frame; no
cumulative-power crop is applied, and the invalid `m > l` triangle is neutral
gray rather than an ordinary zero. `FigureTimeline` freezes one magnitude
normalization from the complete sequence before rendering either individual
frames or the representative overview.

A current run capsule looks like this:

```text
runs/
├── latest_run.txt
└── <run-id>/
    ├── manifest.json
    ├── config.json
    ├── diagnostics/
    │   ├── timeseries.csv
    │   └── spectra.npz
    ├── figures/
    │   ├── invariant_drift.png
    │   ├── cfl_history.png
    │   ├── spectral_health.png
    │   └── spectra.png
    ├── vorticity_coeffs.npy
    ├── vorticity_grid.npy
    ├── bve_snapshot_times.npy
    ├── bve_summary.png
    └── snapshots/
        ├── physical/
        │   ├── timeline.png
        │   └── t<seconds>s.png
        └── spectral/
            ├── timeline.png
            └── t<seconds>s.png
```

An SWE capsule uses `swe_coeffs.npy` (shape `(N, 3, L+1, L+1)`, ordered as
relative vorticity, horizontal divergence, and perturbation geopotential),
`swe_snapshot_times.npy`, and `swe_summary.png` in place of the BVE state and
summary products. Its physical and spectral snapshot products and final
summary are regenerated from those persisted arrays.

The intended semantic categories are:

```text
runs/
└── <run-id>/
    ├── manifest.json
    ├── config.json
    ├── diagnostics/
    ├── states/       # currently BVE's three state/time .npy files at capsule root
    ├── figures/      # diagnostic figures; summaries are currently at root
    ├── snapshots/    # physical/spectral frames and representative overviews
    └── logs/         # reserved; CLI stdout is not persisted yet
```

Which figures exist depends on the run's plot selection (`--plot` /
`--no-plots`): `figures/` comes from the `diagnostics` plot product,
`snapshots/{physical,spectral}/` comes from `snapshots`, and the model summary
comes from `summary`. Each snapshot representation has one deterministic
frame per stored state and a vertically assembled overview of at most five
physical-time representatives. The `.npy` state files and `diagnostics/`
data are written regardless of plot selection (and are empty-but-present when
`--n-snapshots 0` stores no field states).

`config.json` is the authoritative model/CLI configuration: the historical
psx-bve key set plus the additive keys `snapshot_mode`, `n_snapshots`,
`snapshot_times` (the resolved schedule in seconds), and `plots`.
`dt_snapshots` remains the uniform interval where one exists (interval mode,
or count mode with `N >= 2`) and is `null` for `N` in {0, 1}. `manifest.json`
adds the exact command, UTC creation time, run **status**
(`running` → `completed` or `failed`, with a concise error record on
failure), Git commit/branch/dirty flag, Python and library versions, GPU,
transform, state sampling, and actual product sampling.

Run ids encode UTC timestamp, scenario, rotation state, resolution, l_max,
and a snapshot tag (`dtNh` where a uniform interval exists, `snapN` for
count modes with `N` in {0, 1}). Legacy interval-mode runs keep the
historical run-id form byte-for-byte. New canonical (count-mode) runs also
append a short 8-hex-character deterministic hash of the resolved
scientific configuration (backend, dimensions, duration, viscosity,
quadrature, snapshot schedule); purely locational fields (`out`,
`experiment`, `overwrite`) and derived artifacts (`plots`) are excluded
from the hash. Two runs that differ only in output location share a hash;
two runs that differ scientifically are strongly disambiguated by it (the
digest is 32-bit, so a same-second collision between distinct scientific
configurations is very unlikely, not impossible). Figures embed
run metadata; `--overwrite` reuses a directory but first removes known
generated artifacts so a stale plot selection cannot survive alongside a
new run configuration.

The authoritative scientific diagnostics are `diagnostics/timeseries.csv` (one
flushed row per accepted step, including energy, relative and absolute
enstrophy, circulation, CFL, high-degree content, and periodic transform
residuals) and `diagnostics/spectra.npz` (degree spectra). The viewer summary is
useful for visual inspection but is not the scientific invariant record.
`vorticity_coeffs.npy` is the saved spectral state; `vorticity_grid.npy`
contains plotting snapshots; and `bve_snapshot_times.npy` is their
authoritative time axis in seconds. BVE and SWE timeline frames can be
regenerated from these persisted arrays without rerunning either model.

New manifests also carry two additive, versioned blocks derived from the
resolved configuration: `state_schema` (the stored fields' row layout,
dimensions, units, conventions, support, geometry, vertical coordinates and
environment provenance) and `diagnostic_definitions` (the CSV column
meanings, including the structured `high_l_enstrophy_frac` definition). They
are descriptive only — never part of `run_config` or the scientific hash —
and `open_simulation` reads them, inferring the known historical layouts for
older capsules and saying so ([SAVED_RUNS.md](SAVED_RUNS.md)).

A capsule is reproducible only to the extent recorded by its manifest. Runs from
a dirty tree require the uncommitted patch as well as the commit, and the
current `random_low_l` scenario does not record an RNG seed. Treat those as
explicit exceptions, not bitwise-reproducible experiments.

Diagnostic plots can be regenerated from the saved authoritative CSV/NPZ data
without rerunning the model:

```powershell
python -c "from pathlib import Path; from tropoi.representation.diagnostics.bve import plot_diagnostics; r=Path('runs'); plot_diagnostics(r/(r/'latest_run.txt').read_text().strip())"
```

## How to add or compare a backend

1. Implement `GridGeometry` for coordinates, point count, areas/weights,
   optional structured shape, and a documented CFL length scale.
2. Provide a transform with `transform`, `inv_transform`, `weights`, and
   `l_max`, producing the shared dense coefficient layout.
3. Subclass `SphericalGridBackend`; define supported quadrature names and
   construct/cache each `ProductSpace` with a provenance label.
4. Register the pairing in `make_backend` (`spatial/spherical_backend.py`) and in
   `Planet.generate` (`spatial/planet.py`)/the CLI if it
   is a user-facing grid.
5. Run the same transform, Jacobian, velocity, diagnostics, RH4, provenance, and
   end-to-end tests used for both current backends.

### Tests

```powershell
pytest
```

The suite requires a working CUDA GPU. On a Windows machine whose global pytest
temp directory has stale ACLs, use a workspace-local directory:

```powershell
pytest --basetemp .pytest-tmp
```

`requirements-dev.txt` includes the runtime pins, an editable package install,
pytest, and ipykernel.

### Benchmarks and audits

These scripts are intentionally outside normal pytest collection and write their
artifacts beneath ignored `runs/` directories:

```powershell
python tests/audit_r3_product.py res4
python tests/audit_r3_product.py res5
python tests/audit_r4_convergence.py
python tests/audit_r5_mechanism.py
```

`audit_r3_product.py res5` and fine-product configurations can require much more
GPU memory than the default run. Read each script's header before running it.
The README figures can be reproduced from run capsules with
[`readme_figures.py`](readme_figures.py); their portable scientific provenance is
tracked separately from the ignored raw runs.
