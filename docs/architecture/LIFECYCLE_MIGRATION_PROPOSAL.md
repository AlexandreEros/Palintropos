# Palintropos lifecycle boundaries: evidence-based proposal

**For review. Nothing here moves production code.** It records what the
interactive atlas and the runtime captures show about the current code, tests
the proposed lifecycle grouping against that evidence, and turns the result
into a sequence of small, independently reviewable migration issues.

| Evidence | Identity |
|---|---|
| Static source | `src/tropoi` at [`08e6f22`](https://github.com/AlexandreEros/Palintropos/commit/08e6f223923851fb738fbd837581b36f9826a47d), clean; content SHA-256 `363d3660…` (`docs/architecture/current/metrics.json`) |
| Atlas | [`current/interactive.html`](current/interactive.html), `tools.architecture` 2.3.0, 2 059 nodes, 441 import edges, 2 182 static call edges |
| Overlay | [`proposed-packages.json`](proposed-packages.json) revision `2026-11-13.2` (schema 2) |
| Runtime | six traces in [`evidence/runtime/`](evidence/runtime/), same commit and same source content hash as the static snapshot; GPU: NVIDIA GeForce MX110 (CC 5.0, 2 GiB), CUDA 11.8, CuPy 13.4.0, NumPy 2.2.6, Python 3.12.12, Windows 11 |
| Timing | one synchronized profile in [`evidence/profile/`](evidence/profile/), separate process, no tracer |

Every number below can be reproduced from those files; the atlas shows each
cited edge with its source sites (Group by: *Proposed owner* → *Boundaries*).

## 0. Conclusions

1. **The lifecycle grouping holds, after nine source-grounded amendments.**
   With the revised overlay, all 260 executed import edges and all 422 static
   calls that cross a proposed boundary point down the declared order, and
   there is no group cycle once the nine declared exceptions are set aside
   (§4). Three of those exceptions are temporary and disappear with specific
   migration issues; three disappear with the saved-run protocol inversion;
   two are lateral by design; one is explicitly unresolved.
2. **`world` is necessary, but it is reference data as much as planets.** The
   import graph forced fundamental constants, species identity and stated
   compositions into it (stars and orbits import them at module scope). It
   stays the lowest layer, CPU-only and immutable.
3. **The planet has five descriptions in production, not three** (§3). The
   canonical physical descriptor should be the existing, already frozen and
   versioned `tropoi.spatial.specification.PlanetSpec`, extended, not a new
   class and not `PlanetaryBody`. `PlanetaryBody`/`PLANETS` has **no consumer
   in `src/`**; the radiative and Held–Suarez paths carry their own paper
   constants, which must stay distinct.
4. **`Planet` is a numerical-resource bundle**: the cores import it at module
   scope and read `sh`, `so`, `grid` and `params`. It belongs with `spectral`;
   only `Planet.generate` is construction. Every HS/radiative/PE construction,
   including resume, also generates decorative GPU terrain that only the `gen`
   viewer reads (runtime evidence, §6).
5. **Static analysis sees about half of the runtime call structure.** In the
   radiative lifecycle 171 of 413 observed relations (41 %) have no static
   edge: attribute-receiver dispatch (`self.so.d_lambda_coeffs`, 16 200 calls),
   property getters and transform delegation. Boundaries must be judged with
   both kinds of evidence; neither alone suffices.
6. **Exact resume works, bitwise**, and fresh construction and reconstruction
   are distinct paths (§6). The saved-run CPU read path never imports CuPy;
   the default saved PE *figure* does.

## 1. Proposed structure

```text
tropoi/
    cli/          public command routing and compatibility
    run/          orchestration, output schedules, checkpoint coordination
    setup/        configuration, presets, terrain/IC preparation, component construction
    analysis/     saved-run access, diagnostics, reductions, visualization
    archive/      persistence formats, reconstruction, run identity, checkpoint encoding
    integration/  steppers, stage/history management, numerical filters, coupling schedules
    dynamics/     prognostic state schemas, equations -> tendencies, term/damping contract
    physics/      radiation, convection, surface exchange, thermodynamic/optical relations
    spectral/     grids, quadrature, truncation, transforms, operators, kernels, resource bundle
    world/        prescribed physical descriptions and the reference data they are stated in
```

Declared dependency order (a group may depend on any group below it; same-tier
groups are independent):

```text
cli → run → {setup, analysis} → {archive, integration} → {dynamics, physics} → spectral → world
```

Rationale, in the terms the evidence supports:

* **Construction is not evaluation.** `run.radiative.model.build_*` and
  `initial_state` run once per construction (runtime: phase `construction`
  and again in `resume-construction`, never during stepping); the resulting
  `ColumnPhysics.compute` runs once per step. The former is `setup`, the
  latter `physics`, and `temporal.column_coupling` (which schedules the call
  and holds the coupling state that checkpoints persist) is `integration`.
* **Kernel ownership follows the operation.** Transform kernels
  (`spatial.transforms.cuda`) stay with `spectral`; nothing in `physics`
  needs `spectral` today (no `physics → spectral` edge exists), and the one
  per-step device transfer in physics is the insolation upload in
  `Radiation._stellar` (§6), not a transform.
* **`archive` sits below `analysis` and `setup`** because both decode what it
  stores, and it must decode the persisted specifications; those therefore
  live in `world` (physical) and `spectral` (`DiscretizationSpec`).
* **`world` evaluated.** Necessary: without it `setup → physics` and
  `physics → setup` would both be needed for the planet. Name: `world` reads
  well next to `setup`/`run`; `reference` would describe the constants and
  species better. Either is acceptable; the boundary matters, not the word.
  Placement: lowest. It must never import `physics` evaluation code; today
  two temporary exceptions (`PlanetaryBody.cp`/`regime()` and its stored
  `StellarIllumination`) do, and issue M4 removes them.

## 2. Mapping of existing modules and classes

The authoritative, machine-readable mapping is `proposed-packages.json`
(module, class or function prefixes; most specific wins). Rendered from it:

| Proposed owner | Modules | Current modules |
|---|---|---|
| `cli` | 7 | `tropoi` (empty root), `cli`, `cli.capsule`, `cli.clear_cache`, `cli.generate_planet`, `cli.main`, `cli.terrain` |
| `run` | 19 | `run`, `run.bve`, `run.bve.execution`, `run.bve.runner`, `run.custom`, `run.custom.execute`, `run.held_suarez`, `run.held_suarez.__main__`, `run.held_suarez.experiment`, `run.lifecycle`, `run.pe`, `run.pe.execution`, `run.pe.runner`, `run.radiative`, `run.radiative.__main__`, `run.radiative.experiment`, `run.swe`, `run.swe.execution`, `run.swe.runner` |
| `setup` | 28 | `run.{bve,swe,pe,radiative,held_suarez}.config`, `run.{bve,swe,pe}.model` (re-export shims), `run.bve.presets`, `run.radiative.model`, `run.held_suarez.model`, `run.custom.{config,planet,prepare,presets,resources}`, `spatial.initialization{,.bve,.pe,.swe}`, `spatial.terrain{,.elevation_import,.height_function,.raster,.tectonics,.terrain_spectral,.topography}`, `temporal.models` |
| `analysis` | 38 | `representation`, `representation.diagnostics.*` (8), `representation.visual.*` (26), `run.held_suarez.{report,statistics}`, `temporal.simulation` |
| `archive` | 11 | `representation.archive.*` (9), `representation.identity`, `run.held_suarez.checkpoint` |
| `integration` | 5 | `temporal`, `temporal.column_coupling`, `temporal.integration`, `temporal.semi_implicit`, `temporal.steppers` |
| `dynamics` | 11 | `spatial.states{,.barotropic,.primitive_equations,.shallow_water}`, `temporal.hooks`, `temporal.tendencies{,.barotropic,.drag,.held_suarez,.primitive_equations,.shallow_water}` |
| `physics` | 13 | `physics`, `physics.{column,column_model,convection,insolation,rayleigh,surface,thermodynamics}`, `physics.radiation{,.gray,.kdist,.scheme,.twostream}` |
| `spectral` | 25 | `spatial`, `spatial.grids.*` (8), `spatial.laplacian`, `spatial.modes`, `spatial.operators{,.spectral_operators}`, `spatial.planet`, `spatial.sigma_coordinate`, `spatial.spherical_backend`, `spatial.terrain.elevation_data`, `spatial.transforms.*` (7), `spatial.truncation` |
| `world` | 10 | `physics.{composition,constants,orbit,planets,rotation,species,star}`, `spatial.environment`, `spatial.specification`, `spatial.williamson5` |

Symbol-level assignments that differ from their module (each is a split the
migration has to make):

| Symbol | Module's owner → symbol's owner | Why |
|---|---|---|
| `spatial.planet.Planet.generate` | spectral → setup | construction of the resource bundle |
| `spatial.specification.DiscretizationSpec` | world → spectral | numerical description, decoded by archive |
| `physics.planets.VerticalGridSpec` | world → setup | a numerical vertical-grid choice carried by a physical body |
| `representation.archive.capsule.open_simulation` | archive → analysis | constructs the analysis facade |
| `temporal.tendencies.held_suarez.hyperdiffusion_{damping,coefficient}` | dynamics → integration | numerical filter, not an equation term |

Amendments to the first draft, each forced by edges the atlas shows:

| Amendment | Edges that motivated it |
|---|---|
| `physics.{constants,species,composition}` → `world` | `orbit`, `star` import `constants`; `star._load_catalogue` calls `species.load_dataset`; `planets` imports `composition` (6 `world → physics` imports) |
| `spatial.specification` → `world`, `DiscretizationSpec` → `spectral` | `archive.initial_state`/`prepared_terrain` decode the specs (5 `archive → setup` calls); spec validators called from `PlanetSpec` (11 `world → setup` calls) |
| `spatial.williamson5` → `world` | `specification` imports `W5_CONE_HEIGHT_M`; the module calls itself "the benchmark's prescribed world" |
| `spatial.planet` → `spectral` (bundle), `.generate` → `setup` | `tendencies.{barotropic,shallow_water,primitive_equations}` import `Planet` at module scope (4 `dynamics → setup`) |
| `spatial.terrain.elevation_data` → `spectral` | a field of the bundle |
| `temporal.hooks` → `dynamics` | `tendencies.held_suarez` instantiates `DiagonalDamping` (2 `dynamics → integration`) |
| `open_simulation` → `analysis` | it instantiates `temporal.simulation.Simulation` |

## 3. The planet data model

### Five descriptions in production

| Description | What it holds | Who uses it (source) | Runtime evidence |
|---|---|---|---|
| `spatial.specification.PlanetSpec` | frozen; `construction` (`ideal_sphere` exact / `oblate_volumetric_mean`), reference radius, sidereal day (`None` = non-rotating), mass; versioned `to_dict`/`from_dict` that recomputes derived values and refuses mismatches | custom BVE/SWE runs (`run.custom.*`), `temporal.models`, archive (`initial_state`, `prepared_terrain`, `schema`), `cli.terrain` | `setup-cpu`: W5 ideal sphere round-trips; dynamical radius exactly 6 371 220 m, oblateness exactly 0 |
| `spatial.environment.PlanetaryParameters` | **mutable**; mass, equatorial radius, day; derives ω, oblateness, volumetric radius, density; `ideal_sphere` overrides shape fields | every core via `Planet.params` (`radius`, `angular_velocity`) | built twice per radiative run (construction and resume) |
| `RadiativeConfig` / `HeldSuarezConfig` fields | explicit `radius`, `omega`, `gravity`, `gas_constant`, `cp`, `reference_pressure`; part of `config_sha256` | `run.radiative.model.build_model`, `run.held_suarez.model.build_model` | config identity `a5bed515…` recorded in trace and checkpoint |
| BVE/SWE/PE preset keys (`day_hours`, `radius_earth_units`, W5 constants) | resolved run-config mapping | `temporal.models.build_*` via `from_earth_like` / W5 `ideal_sphere` | (not captured: see §6) |
| `physics.planets.PlanetaryBody` / `PLANETS` | frozen; radius, gravity, `Rotation`, `Composition`, `Star`, `KeplerOrbit`, a stored `StellarIllumination`, reference pressure/temperature/range, albedo, interior flux, `VerticalGridSpec`, notes; derived `cp`, `kappa`, `regime()` | **no module in `src/`**; `tools/radiation/{regimes,adiabatic_energy,energy_attribution}.py` | not touched by the radiative lifecycle; inspected directly in `setup-cpu` |

`spatial.planet.Planet` is not a physical description: it bundles
`params`, grid, transform, operators and an `ElevationData`, and is a
numerical resource (§2).

The preliminary assessment is confirmed for `Planet` and
`PlanetaryParameters` and corrected for `PlanetaryBody`: it is the richest
*physical* description, but it is off every production path, mixes physics
evaluation and numerical configuration into the descriptor, and its values are
not the experiments' constants. Radiative Earth uses FHZ06's `6.376e6 m`,
`9.8 m s⁻²`; `PLANETS["earth"]` has `6.371e6 m`, `9.80665 m s⁻²`
(`setup-cpu` records both). Jupiter: `69.86e6 m`, `26.0` vs `6.9911e7`, `24.79`.

### Decisions

* **D1 — One canonical physical planet descriptor: `PlanetSpec`**, the
  existing class, moved to `tropoi.world` and re-exported from its current
  paths (`spatial.specification`, `run.custom`). It already is immutable,
  validated, versioned and persisted in capsules; no second type is created.
* **D2 — Rotation is stated, signed, and may be exact.** HS and radiative
  build `ideal_sphere(radius, 2π/ω)` and then *mutate*
  `params.angular_velocity = cfg.omega`. For the preset values the period
  round trip is already exact, so the mutation's real job is the
  **sign** (radiative builds the period from `|ω|`) and **ω = 0** (period
  `1e30`). `PlanetSpec` gains an optional signed `angular_velocity_rad_s`
  that, when given, is used verbatim; `PlanetaryParameters.ideal_sphere`
  gains the same keyword so no caller mutates the object after construction.
  Sidereal-day documents keep today's arithmetic bit for bit.
* **D3 — Gravity is prescribed, never derived.** An optional explicit
  `gravity_m_s2` in `PlanetSpec`; `FluidLayerSpec.gravity` stays the SWE
  layer's own explicit value. Nothing computes `GM/R²` in place of a stated
  constant, and no flattening model replaces `ideal_sphere` for benchmarks.
* **D4 — Experiment constants stay with the experiment.** `RadiativeConfig`
  and `HeldSuarezConfig` keep their flat fields and their `config_sha256`.
  Adapters `planet_spec_from(cfg)` derive a `PlanetSpec` for construction;
  the configuration schema and hash do not change. `PLANETS` is a catalogue
  of *reference bodies*, never silently substituted into a preset.
* **D5 — `PlanetaryBody` becomes a composition**: a `PlanetSpec` (geometry,
  rotation, gravity) plus `Composition`, `Star`, `KeplerOrbit`, `Rotation`
  axis, and an explicit `ReferenceSettings` record (reference pressure and
  temperature, temperature range, albedo, interior heat flux, notes). Its
  `cp`, `kappa`, `regime()` and `illumination` become `physics` functions of
  the body, kept as compatibility properties; `VerticalGridSpec` moves to
  `setup`. `PLANETS` keys and every stated number stay identical.
* **D6 — `Star` owns stellar properties; orbits own elements only.**
  `KeplerOrbit` already holds only elements (semimajor axis, eccentricity,
  inclination, node, periapsis, epoch anomaly); the gravitational parameter
  comes from the star at `StellarIllumination.from_components` time unless
  stated. Keep it that way; do not add stellar mass or luminosity to orbits.
* **D7 — `PlanetaryParameters` is the derived-geometry adapter**, produced
  by `PlanetSpec.to_parameters()`. It becomes frozen once D2 removes the two
  mutation sites (deprecation first: assignment warns, then fails).
* **D8 — `Planet` becomes `PlanetResources`** (alias `Planet` kept), built
  by a `setup` factory; `Planet.generate` stays as a facade. Decorative
  procedural elevation becomes an explicit option of that factory (§5, M6b).

Old interfaces versus the eventual public contract:

| Kept (compatibility) | Eventual public contract |
|---|---|
| `tropoi.spatial.specification.PlanetSpec`, `tropoi.run.custom.PlanetSpec` | `tropoi.world.PlanetSpec` |
| `PlanetaryParameters(...)`, `.from_earth_like`, `.from_si`, `.ideal_sphere` | `PlanetSpec.ideal_sphere/oblate/earth_like(...).to_parameters()` |
| `params.angular_velocity = ...` (deprecated) | `angular_velocity_rad_s=` at construction |
| `PlanetaryBody.cp`, `.kappa`, `.regime()`, `.illumination` | `physics.thermodynamics.reference_cp(body)`, `physics.insolation.illumination_of(body)`, … |
| `PlanetaryBody.vertical_grid` | `setup` vertical-grid configuration |
| `spatial.planet.Planet`, `Planet.generate` | `spectral.PlanetResources`, `setup.build_planet_resources(...)` |
| `PLANETS[name]` | unchanged keys and values |

## 4. Dependency directions and intentional exceptions

Executed imports crossing proposed owners (module, function scope; TYPE_CHECKING
excluded). Cell = edges, *(n exc.)* = covered by a declared exception:

| from \ to | cli | run | setup | analysis | archive | integration | dynamics | physics | spectral | world |
|---|---|---|---|---|---|---|---|---|---|---|
| **cli** |  | 5 | 8 | 6 | 4 |  |  |  | 3 | 1 |
| **run** |  |  | 28 | 11 | 16 | 5 | 9 |  | 1 | 3 |
| **setup** |  |  |  | 1 (1 exc.) | 7 | 6 | 16 | 6 | 20 | 13 |
| **analysis** |  |  | 1 (1 exc.) |  | 5 |  | 5 | 2 | 24 | 4 |
| **archive** |  |  |  | 9 (9 exc.) |  |  |  |  | 7 | 3 |
| **integration** |  |  |  |  |  |  | 2 |  | 3 |  |
| **dynamics** |  |  | 1 (1 exc.) |  |  |  |  |  | 10 |  |
| **physics** |  |  |  |  |  |  |  |  |  | 10 |
| **spectral** |  |  | 2 (2 exc.) |  |  |  |  |  |  | 1 |
| **world** |  |  |  |  |  |  |  | 2 (2 exc.) |  |  |

260 crossing edges: 244 downward, 0 lateral, 0 upward outside exceptions.
Static calls: 422 crossing, 402 downward, none upward outside exceptions.
Group cycles with exceptions included: `{analysis, archive, dynamics,
integration, setup, spectral}` and `{physics, world}`; **with them removed:
none**, for imports and calls alike.

High crossing counts are expected where orchestration lives (`run → setup`
28, `setup → spectral` 20): `setup` constructs from everything below it by
definition. They are not failures; the red (upward) cells are.

Declared exceptions (`exceptions` in the overlay; the page shows each one's
edges). The count in brackets is the import and call edges each one covers:

| # | Source → target | Kind | Removed by |
|---|---|---|---|
| 1 | `representation.archive.capsule` → `representation.visual` (11) | the `SnapshotStorage` protocol delegates `plot`/`view`/`sky`/`quantities`/`resources` to storage | M8 |
| 2 | `representation.archive.held_suarez` → `representation.visual` (6) | same | M8 |
| 3 | `representation.archive.capsule` → `temporal.simulation` (1) | `open_simulation` builds the facade | M8 |
| 4 | `representation.visual.snapshot` → `temporal.models` (4) | figures rebuild the run's model with the runners' own builders (one construction for both callers, per `temporal.models`) | lateral by design |
| 5 | `run.custom.planet` → `representation.visual.terrain` (4) | `PreparedPlanet.plot/view` previews, imported on use | lateral by design |
| 6 | `physics.planets` → `physics.insolation` (2) | stored `StellarIllumination` | M4 |
| 7 | `physics.planets` → `physics.thermodynamics` (3) | `cp`, `regime()` evaluate heat capacity | M4 |
| 8 | `temporal.tendencies.shallow_water` → `spatial.terrain.topography` (2) | SWE core consumes a prepared `Topography` | unresolved (§5) |
| 9 | `spatial.planet` → `spatial.terrain` (3) | bundle module imports `generate`'s terrain dependencies | M6a |

## 5. Unresolved or ambiguous ownership

* **Prepared boundary fields** (`spatial.terrain.topography.Topography`):
  built by `setup`, consumed by the SWE core (`Topography.flat` default
  inside `ShallowWaterModel.__init__`); rebuilt three times in one custom
  workflow (§6). Options: a `dynamics` schema
  (`BottomTopography`) built by `setup`, or keep it in `setup` and inject it.
  Decide before M10; until then exception 8.
* **`sigma_coordinate`**: vertical grid metadata *and* hydrostatic column
  operators used by the PE core. Placed in `spectral` ("numerical
  operators"); `dynamics` is defensible. No edge forces either.
* **Held–Suarez forcing** (`tendencies.held_suarez.NewtonianRelaxation`,
  Rayleigh drag): a prescribed forcing that behaves like a parameterization
  but is written as a tendency term. Kept in `dynamics`; `physics` is
  defensible. Hyperdiffusion is assigned to `integration` (a filter).
* **`run.held_suarez.checkpoint`** is the checkpoint codec of *both* HS and
  radiative runs (`RadiativeRun` imports it). Proposed `archive`; the
  radiative run's dependency on a sibling experiment package disappears.
* **`run.held_suarez.experiment` helpers** (`code_identity`,
  `environment_identity`, `daily_sample`, `RunError`) are imported by
  `RadiativeRun` too: shared run provenance (`archive`) and a shared sample
  (`analysis`/`run`) live in an experiment module today.
* **Saved-run figures rebuild a GPU model.** `saved-run-view` shows the
  default `Snapshot.plot()` of a PE capsule calling `temporal.models.build_pe_model`,
  initializing CuPy and making ≥ 830 device→host and 76 host→device
  transfers, 756 of them from
  `spatial.grids.grid_interpolation.geodesic_to_latlon_grid`; the 3-D
  `view()` of the same capsule, run first in the same fresh process, is built
  without importing CuPy. Ownership is clear
  (`analysis`); the dependency on GPU construction is a design question
  (M12), not a move.
* **Static blind spots.** Relations that matter for boundaries
  (`PrimitiveEquationsModel._stack_derivs → SpectralOperators.*`, stepper →
  `model.tendency`, coupling → `physics.compute`) are runtime-only. Any
  boundary rule enforced in CI from static edges alone would under-report
  `dynamics → spectral` and `integration → physics` coupling.

## 6. What the runtime evidence shows

Captured with `python -m tools.architecture.runtime_capture --scenario <name>`
(registered drivers in `tools/architecture/scenarios.py`); `--measure-overhead`
for the three scenarios whose claims do not depend on import state; one capture
(`cli-swe-williamson2-tiny`) runs the public CLI as a module. All six ran
at `08e6f22`, clean source; tools dirty at capture time (this change).
Overhead = traced / warm uninstrumented run in the same process. The saved-run
scenarios make claims about which modules get imported, which only a fresh
process can support (the overhead protocol's earlier runs would import them
first), so their overhead is not measured.

| Scenario | Lifecycle covered | Verified | Overhead |
|---|---|---|---|
| `setup-cpu` | preset resolution and validation, CPU component construction, `PlanetSpec` persistence, terrain sampling, `PLANETS` inspection | config round trip keeps `config_sha256`; invalid `radiation_interval` rejected; W5 ideal sphere exact; CuPy not imported | 6.9× (tiny absolute times) |
| `radiative-tiny-lifecycle` | construction, RK4 startup, 71 leapfrog steps + daily output + checkpoint through `RadiativeRun.run(until_day=1)`, resume, 3 resumed steps, 3 reference steps | **resumed == uninterrupted, bitwise**: `x_curr`, `x_prev` (76×11×11 complex128), step 75, `t` 90 000 s, surface temperature (512), coupling budget | 1.03× (GPU-bound) |
| `custom-swe-lifecycle` | `PlanetSpec`/`DiscretizationSpec`/`FluidLayerSpec`, sampled terrain raster, GPU terrain projection (`prepare_planet`), SWE state from a streamfunction over it, save, reload (same process), 6 h custom run at l_max 10, open the capsule | capsule opens with 2 frames, solver `swe` | 1.5× |
| `cli-swe-williamson2-tiny` | `tropoi run swe --scenario williamson2 --lmax 10 --resolution 3 --days 0.25 --n-snapshots 2 --no-plots`: CLI routing, `SWERunConfig.resolve`, run lifecycle, 12 steps, capsule writing | completed, capsule written (run id in the trace) | not measured |
| `saved-run-cpu` | open published BVE capsule, snapshot, spectral figure; HS fixture climate reduction and figure | CuPy never imported (SciPy neither), fresh process | not measured |
| `saved-run-view` | PE capsule 3-D page and default snapshot figure | page built without CuPy; the figure then imports CuPy and SciPy, fresh process | not measured |

Lifecycle facts from `radiative-tiny-lifecycle` (calls per phase):

* `initial_state` and `SemiImplicitLeapfrogStepper.initialize`: construction
  only. `_resume` and `load_state_dict`: resume only, never `initial_state`.
* `_startup` (RK4): exactly once, in the first step; every later step is a
  leapfrog step. No resumed step re-runs the startup.
* `ColumnPhysicsCoupling.__call__` and `ColumnPhysics.compute`: once per
  step including the startup step (`radiation_interval = 1`).
* `Planet.generate` and `generate_spectral_terrain_gpu`: once per
  construction **and once per resume**, for a flat-surface run.
* `_write_checkpoint`: at creation (step 0) and at the end of day 1.
* Device events (lower bound): one host→device upload per step from
  `physics.radiation.scheme.Radiation._stellar`; daily output and checkpoint
  are device→host. The collector itself inserted no synchronization and read
  no device memory (unit-tested against a real CuPy array).

Separate synchronized profile (`radiative_pilot.py profile --mode sync`, same
configuration `a5bed515…`, no tracer): 240 ms per step at `tiny` on the
MX110. Exclusive per step: frictional heating (`MomentumSinkHeating`) 40.6 ms,
dynamics transforms 35.5 ms, column physics coupling 4.8 ms + its transforms
7.3 ms. A tiny grid on a laptop GPU is not representative of production
resolution; these numbers locate costs, they are not throughput.

Custom-run facts from `custom-swe-lifecycle`: `Planet.generate` (the full
resource build, with its decorative terrain) runs **three times** in one
workflow: terrain projection, initial-condition preparation and the run each
build their own; `ShallowWaterModel` is constructed twice (validation of the
prepared state, then the run) and `Topography` three times. Prepared inputs
are re-resolved per phase rather than handed on: a construction cost and an
argument for M6a/M6b and for settling the ownership of prepared boundary
fields (§5).

**Not observed** (commands in [`evidence/README.md`](evidence/README.md)):
BVE and PE *preset* runs through the CLI, the Jupiter configuration, the PE prepared-terrain run path, the `gen` CLI viewer, and
anything in other threads or processes. Unobserved is not unused.

## 7. Migration issues

Each issue is mechanical *or* behavioural, never both; each keeps old import
paths working through re-export shims until a separate deprecation. Labels
are from `.github/ISSUE_LABELS.md`; kind (`enhancement`, …) is GitHub's.

**M1 — Explicit signed angular velocity at planet construction** ·
`engine:primitive-equations`, `area:reproducibility`
*Depends on:* nothing. *Scope:* add `angular_velocity_rad_s=None` to
`PlanetaryParameters.ideal_sphere` (and the `__init__` path) used verbatim
when given; replace the two post-construction assignments in
`run.held_suarez.model.build_model` and `run.radiative.model.build_model`.
*Acceptance:* `params` fields bitwise equal before/after for HS
`development_config`, both radiative presets, a retrograde ω (Venus sign),
and ω = 0; model `Omega` identical; `config_sha256` and checkpoints
unchanged; existing HS/radiative tests pass.

**M2 — `tropoi.world` package (mechanical)** · `area:reproducibility`
*Depends on:* nothing. *Scope:* move `spatial.specification` (minus
`DiscretizationSpec`), `spatial.williamson5`, `spatial.environment`,
`physics.{constants,species,composition,star,orbit,rotation,planets}` to
`tropoi.world`; shims at every old path; `DiscretizationSpec` stays importable
from `spatial.specification`. *Acceptance:* every old import works
(`test_entry_points`, a shim test per moved module); `to_dict()` documents of
every spec byte-identical; capsules under `docs/runs/` and
`tests/fixtures/` open; run ids and config hashes unchanged; `world` imports
nothing above it except exceptions 6–7.

**M3 — `PlanetSpec` document version 2** · `area:reproducibility`
*Depends on:* M1, M2. *Scope:* optional signed `angular_velocity_rad_s` and
explicit `gravity_m_s2`; v2 writer, v1 reader (no rewrite of stored
documents); `planet_spec_from(RadiativeConfig | HeldSuarezConfig)` adapters
used by the two `build_model`s. *Acceptance:* v1 documents load to identical
objects; v2 round trip exact; `to_parameters()` bitwise equal to M1's output;
W5 radius/oblateness exact; no derived gravity anywhere (grep test).

**M4 — Split `PlanetaryBody` into identity, references and evaluation** ·
`area:reproducibility`, `area:validation`
*Depends on:* M3. *Scope:* D5; `cp`/`kappa`/`regime()`/`illumination` as
physics functions behind properties; `VerticalGridSpec` to setup; remove
exceptions 6–7. *Acceptance:* every `PLANETS` field and property value
identical (table test); `tools/radiation/regimes.py` output byte-identical;
`docs/radiation/REGIMES.md` numbers unchanged; `world` has no `physics` import.

**M5 — Hooks contract and filters (mechanical)** · `numerics:time-integration`
*Depends on:* nothing. *Scope:* `temporal.hooks` with dynamics; hyperdiffusion
helpers with integration; shims. *Acceptance:* stepper `signature()` dicts and
checkpoint `stepper` metadata identical; HS and radiative short-run states
bitwise equal.

**M6a — Resource bundle versus factory (mechanical)** ·
`numerics:spectral-transforms`
*Depends on:* nothing. *Scope:* `PlanetResources` (alias `Planet`) in
spectral; `build_planet_resources` in setup; `Planet.generate` calls it;
remove exception 9. *Acceptance:* `Planet.generate` outputs (grid, `sh`,
`so`, elevation coefficients) bitwise equal for geodesic and lat-lon; all
callers unchanged.

**M6b — Decorative terrain only when asked** · `area:performance`, `area:cuda`
*Depends on:* M6a. *Scope:* solver builders (HS, radiative, PE, SWE, BVE)
request no procedural elevation; `gen` keeps it. *Acceptance:* first verify
that no consumer reads CuPy's global RNG after `generate_spectral_terrain_gpu`
(else the change alters later random draws, and the issue stops); then
bitwise-equal short runs for every preset family; construction time and
device memory recorded before/after; `planet_viewer` unchanged.

**M7 — `tropoi.spectral` package (mechanical)** · `numerics:spectral-transforms`, `area:cuda`
*Depends on:* M6a. *Scope:* `spatial.{grids,transforms,operators,modes,
truncation,laplacian,spherical_backend,sigma_coordinate,planet}` and
`DiscretizationSpec`; CUDA sources move with the transforms; shims.
*Acceptance:* CPU reference and backend-parity tests; geodesic and Gauss
regressions; kernel cache/compilation unchanged (same `.cu` bytes);
`clear_cache` CLI works; GPU timing within noise of the M6a baseline.

**M8 — Invert the saved-run storage protocol** · `area:visualization`, `area:reproducibility`
*Depends on:* nothing (before M10). *Scope:* `open_simulation` and the
`SnapshotStorage` protocol owned by analysis; `Simulation.plot/view/sky` call
visualization with storage-provided data; storage methods kept as deprecated
delegates; remove exceptions 1–3. *Acceptance:* every saved-run test;
`saved-run-cpu` scenario re-captured shows no CuPy import; legacy capsules open;
figures/pages byte- or pixel-identical where already deterministic.

**M9 — `tropoi.setup` package (mechanical)** · `area:cli`, `area:reproducibility`
*Depends on:* M2, M6a. *Scope:* configs, presets, builders
(`run.*.config`, `run.*.model`, `run.radiative.model`, `run.held_suarez.model`,
`temporal.models`), `spatial.initialization`, terrain preparation,
`run.custom.{prepare,presets,planet,resources,config}`; shims.
*Acceptance:* preset IDs, `list-presets` output, config canonicalization,
`config_sha256`/run ids, manifests identical; fixed-seed initial states
bitwise; CLI help and flags unchanged.

**M10 — `tropoi.archive` package (mechanical)** · `area:reproducibility`
*Depends on:* M8. *Scope:* `representation.archive`, `representation.identity`,
`run.held_suarez.checkpoint`, shared run provenance from
`run.held_suarez.experiment`; resolve exception 8 first or keep it.
*Acceptance:* golden save/open/resume (the `radiative-tiny-lifecycle`
verification, re-captured, still bitwise); old checkpoints resume; atomic
writes and pruning unchanged; capsule receipts verify.

**M11 — `dynamics`, `integration`, `physics`, `analysis` package moves
(mechanical, one issue each)** · `engine:*`, `numerics:time-integration`,
`area:visualization` as applicable
*Depends on:* M5, M7 (dynamics/integration), M4 (physics), M8 (analysis).
*Acceptance:* per package: shims, Williamson 2/5, HS and radiative energy-budget
regressions bitwise on fixed configurations, and the atlas overlay shows no new
upward edge.

**M12 — Saved PE figures without CUDA (behavioural)** · `area:visualization`, `area:cuda`, `area:performance`
*Depends on:* M8. *Scope:* host synthesis for the default snapshot figure, or
an explicit `allow_cuda` like `view()`; investigate the per-element transfers
in `geodesic_to_latlon_grid`. *Acceptance:* `saved-run-view` re-captured shows
no CuPy import for the figure; figure numerically equal to the GPU path within
the stated synthesis budget.

Outside these issues: changes of truncation policy, new radiation schemes,
GPU optimization (including the per-step insolation upload and the cost of
`MomentumSinkHeating`), exoplanet presets and PR #29 follow-ups. Each needs its
own validation issue and acceptance criteria.

## 8. Compatibility and numerical-regression gates (every issue)

1. **Public surface**: CLI commands, flags, help text and exit codes; every
   pre-move import path (shim test); `__all__` unchanged.
2. **Identity**: `config_sha256`, run ids, preset IDs, `PLANETS` keys and
   values, benchmark constants (W5 radius, explicit gravities) unchanged.
3. **Persistence**: every capsule in `docs/runs/` and `tests/fixtures/` opens
   on a CPU without CuPy; old checkpoints resume; the exact-resume check of
   `radiative-tiny-lifecycle` stays bitwise.
4. **Numerics**: fixed-configuration short runs bitwise equal for mechanical
   issues; behavioural issues state their tolerance and why before running.
5. **Read-only inspection** stays import-light (no CuPy on open/inspect).
6. **Visualization**: existing machinery reused, not rewritten.
7. **Validation limits** documented in `docs/KNOWN_LIMITATIONS.md` and
   `docs/radiation/REGIMES.md` are restated, never strengthened by a move.
8. **Architecture**: `python -m tools.architecture.generate --check` passes;
   the overlay's residual upward edges and cycles stay at zero, and an issue
   that removes an exception deletes it from the overlay.

## 9. First recommended issue

**M1 — explicit signed angular velocity at planet construction.** It is the
smallest change that removes the only hidden numerical convention on the
planet path (post-construction mutation restoring sign and zero rotation),
it is testable bitwise with existing configurations, and M3 (the canonical
`PlanetSpec` v2) depends on it. M2 (the mechanical `world` move) can proceed
in parallel.
