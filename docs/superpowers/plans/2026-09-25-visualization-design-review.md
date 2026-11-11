# Saved-run visualization: design review and implementation brief

Date: 2026-09-25. Base: `main` at `9e5d0eb`. Status: design only. No code
has changed.

Inputs read: `README.md`, `docs/SAVED_RUNS.md`, the plotting stack under
`src/tropoi/representation/visual/`, the diagnostics modules, the saved-run
API (`temporal/simulation.py`, `representation/archive/`), the sigma-coordinate
operators, the CLI, the current hero script
`docs/validation/williamson_5/plot_swe_holistic.py`, and the canonical T63
capsule.

There is no `AGENTS.md` at `HEAD` or anywhere in the git history. Repository
conventions were taken from the README, `docs/ARCHITECTURE.md` (layer rules),
`tests/test_layout_compat.py`, and the CI workflows.

---

## 0. Verdict in brief

- The proposal is sound in direction. About two-thirds of what it needs already
  exists: declarative specs, run-wide normalization groups, representative-time
  selection, a lazy saved-run API that builds the model once, and PE level
  selection. What is missing is small and specific:
  1. a **layered map** panel (scalar background, contours and a vector overlay
     on one axis);
  2. a **quantity catalogue** with an evaluator that caches fields;
  3. **cadence-aware** time-series panels;
  4. the **spectral-complexity** code, which today lives only in a docs script;
  5. snapshot-cadence **potential enstrophy**;
  6. one public entry point (`Simulation.plot`, `Snapshot.plot(view=...)`,
     `tropoi plot`).
- **Do not build a general plotting framework.** Two composition shapes cover
  every example in the proposal. `Grid` is several panels at one time.
  `Overview` is a static row, then one map template repeated over the selected
  times, then a diagnostics row. Anything beyond that goes through a TOML
  layout file mapped 1:1 onto the same objects.
- **Real blockers for the README figure:**
  1. the canonical T63 coefficients are not in git (a decision is needed, §8);
  2. potential enstrophy is in neither the capsule nor its CSV;
  3. the spectral-complexity definitions exist only in
     `plot_swe_holistic.py`;
  4. no layered map exists;
  5. the current scalar-map renderer misregisters cells by up to half a cell,
     which would show as soon as vectors are overlaid (§2.4).
- **Readability is the main defect of the current hero.** It is 15 × 21 in at
  300 dpi (4500 × 6300 px). Shown at an 880 px README column it is scaled by
  0.196, so its 7.8 pt guide text renders at **≈ 6.4 px**, the 10 pt tick labels
  at **≈ 8.2 px**, and the panel titles at ≈ 9.8 px. About half of its
  information is prose in a side column. A 7.2 in wide figure with 7.5 pt or
  larger text, with the prose moved into the caption, fixes this. The rule can
  be checked automatically (§6, V1).

---

## 1. Inventory: what exists

### 1.1 Rendering infrastructure (reuse as-is)

| Component | Path | What it gives us |
|---|---|---|
| Backend-neutral fields | `representation/visual/fields.py` | `ScalarGridField` (north-to-south, `[0, 2π)` endpoint-exclusive longitudes, optional time axis), `SphericalHarmonicField` (unpacked `(l, m≥0)`) |
| Declarative panels | `representation/visual/specs.py` | `ScalarMapSpec`, `StreamlineMapSpec`, `SpectralCoefficientMapSpec`, `LinePanelSpec`/`LineSeriesSpec`, `TextPanelSpec`, `PanelPlacement` (row/col spans), `PanelGroupSpec`, `FigureSpec` (width/height ratios, dpi) |
| Normalization | `representation/visual/normalization.py` | `NormalizationPolicy` (auto, symmetric, fixed, log-magnitude), `from_resolved` freeze |
| Cross-panel limits | `representation/visual/timeline.py` `FigureTimeline.resolve_normalizations` | Groups panels by `normalization_group` **across and within frames** and resolves one limit per group. This is exactly the "consistent colour scales" requirement. |
| Representative times | `timeline.select_representative_frame_indices` | DP selection that is uniform in physical time and always keeps the first and last snapshot. Reuse it for sparse or long schedules. |
| Renderer boundary | `representation/visual/renderers.py`, `matplotlib_renderer.py` | `Renderer.render_figure(FigureSpec)`, atomic PNG write, embedded metadata |
| Grid adapter | `representation/visual/grid_adapter.py` | `map_to_uniform_latlon`: geodesic or Gauss samples → 91 × 181 view grid (linear interpolation, nearest-neighbour fill) |
| Streamline geometry | `matplotlib_renderer._render_streamline_panel` | Correct equirectangular geometry, `dλ/dt = u/(R cos φ)`, `dφ/dt = v/R`, speed colouring |

### 1.2 Per-core compositions (the in-run products; keep untouched)

| Core | Path | Notes |
|---|---|---|
| BVE | `representation/visual/bve.py` | ζ, ψ, streamlines; winds derived spectrally through `planet.so` |
| SWE | `representation/visual/swe.py` | ζ, δ, `h′ = φ/g`, streamlines; with terrain adds `η′ = (φ + φ_s′)/g` and a static `h_s` panel. Winds come from `model.wind_on_state_grid`. |
| PE | `representation/visual/pe.py`, `pe_snapshots.py` | Upper and lower σ levels via `select_snapshot_levels` / `_nearest_full_level`; `T′` via the zeroed monopole; `p_s′` de-meaned from `exp(ln p_s)` with quadrature weights. The header says "sigma = p/p_s", never "altitude". |
| Runner products | `run/{bve,swe}/config.py` `PLOT_TYPES = (diagnostics, snapshots, summary)`; PE `(diagnostics, summary)` | `--plot`/`--no-plots`. Pixel-identity with `Snapshot.plot` is a tested contract (`tests/test_saved_run_plotting.py`). |

### 1.3 Saved-run access

| Component | Path | Notes |
|---|---|---|
| `open_simulation`, `CapsuleStorage` | `representation/archive/capsule.py` | Memory-mapped, read-only; `resources()` builds the model **once per capsule** (`resource_builds`); `stored_array()` |
| `Simulation`/`Snapshot` | `temporal/simulation.py` | `Snapshot.plot` delegates through the `SnapshotStorage` protocol. By the layer rule, temporal code must not import representation, so `Simulation.plot` has to follow the same delegation route. |
| Schema | `representation/archive/schema.py` | `FieldSpec` per prognostic (name, units, description, rows, levels, monopole rule); `diagnostic_definitions_for` (column meanings per solver); inferred for legacy capsules |
| Discovery CLI | `cli/main.py` `tropoi inspect RUN [--snapshot I --field F]` | CUDA-free |
| Lazy render adapter | `representation/visual/snapshot.py` | `PlotUnavailableError`; CPU-only for `representation="spectral"` |

### 1.4 Diagnostics and what saved artifacts contain

| Quantity | Stored? | Cadence | Where |
|---|---|---|---|
| SWE `total_mass`, `total_energy`, `h_min_m`, `eta_min/max_m`, `terrain_max_m`, `max_wind_ms`, `cfl`, `*_l2` | yes | every accepted step | `diagnostics/timeseries.csv` (`SWE_DIAGNOSTIC_COLUMNS`) |
| BVE `energy`, `enstrophy_*`, `circulation`, `high_l_enstrophy_frac` | yes | every step (`roundtrip_residual` every 10th) | CSV |
| BVE degree spectra `energy_l`, `enstrophy_l` | yes | every 10th recorded step (`spectra_every=10`) | `diagnostics/spectra.npz` (**BVE only**) |
| PE `total_mass`, `mass_rel_drift`, `t_min/max`, `ps_min/max`, `max_wind_ms` | yes | every step | CSV. **No energy column**, and the PE core makes no energy claim. |
| SWE potential enstrophy `Z = ∫(ζ+f)²/(2h) dA` | **no** | snapshot only | `representation/diagnostics/swe.py::potential_enstrophy(model, state)`. CUDA, needs the model; deliberately not a CSV column. |
| Spectral complexity `⟨ℓ⟩`, `⟨\|m\|⟩`, `S`, `N_eff` | **no** | snapshot only | Only in `docs/validation/williamson_5/plot_swe_holistic.py` |
| Terrain `h_s` | **no** (not persisted) | static | Reconstructed through the model (`model.topography.elevation_on(model.sh)`), so it needs CUDA |
| PE `σ̇`, `ω/p`, hydrostatic `Φ` | **no** | reconstructible at snapshots | `spatial/sigma_coordinate.py`: `interface_sigma_dot`, `omega_over_p`, `hydrostatic_geopotential`. These accept NumPy or CuPy, so they work on the host. |

### 1.5 The current hero and its inputs

`plot_swe_holistic.py` draws on three inputs:
- `aeolus_w5_t63.npz` (tracked, 1.8 MB): Colab-synthesized `u`, `v`,
  `layer_depth`, free surface, band-limited terrain, Gauss weights, and `Z` at
  four times;
- `swe_coeffs.npy` from a **gitignored** local capsule;
- the per-step CSV. It is tracked as `aeolus_t63_timeseries.csv` and is
  byte-identical (sha `3d8c8cb4…`) to the capsule copy.

The PNG metadata pins `CoefficientsSHA256 = dd28eff9…`. That matches the local
capsule. `CHECKSUMS.txt` does not list it. The capsule opens through the
generic API: `open_simulation` reports schema `inferred` ("manifest has no
state_schema block"), 4 snapshots at days 0, 5, 10 and 15, and `plots: []`, so
there is no in-run product to compare against.

---

## 2. Challenges to the proposal

### 2.1 Scientific distinctions the catalogue must encode, not just document

- **Streamfunction.** BVE flow is non-divergent: `ψ` describes it completely,
  and `χ ≡ 0` by construction. Requesting `velocity_potential` for BVE should
  be an *unsupported quantity* with that reason, not a zero map.
- **SWE and PE winds.** `ψ` is the rotational part only. Contours of `ψ` are
  not streamlines of the full wind. The catalogue offers `wind` (full),
  `wind_rotational` and `wind_divergent` as separate vector quantities, and
  `ψ`/`χ` as scalars with the `l = 0` gauge set to zero.
- **Streamlines vs trajectories.** Streamlines are instantaneous curves tangent
  to one saved wind. Trajectories need the wind at sub-snapshot cadence. With
  snapshots 5 days apart they **cannot be computed honestly** from a capsule,
  so they are out of scope. They would need in-run particle output.
- **Kinetic energy.** Three different quantities go by this name:
  - `ke_specific = ½|u|²` [m² s⁻²], pointwise.
  - SWE `ke_column = ½ h |u|²` [m³ s⁻²] **per unit density**. The model has no
    ρ; do not invent one. PE `ke_column = Σ_k ½|u_k|² Δσ_k p_s/g` [J m⁻²] has
    real units, because `p_s/g` is mass per area.
  - The hero's modal energy `E(ℓ,m)` decomposes `½∫|u|² dA` [m⁴ s⁻²], the
    **area-integrated specific** KE. It is *not* weighted by depth. The legend
    must say "unweighted ½|u|²".
- **SWE `total_energy` in the CSV** is `∫[Φ|u|²/2 + Φ²/2 + Φφ_s] dA`, which is
  energy × g/ρ [m⁶ s⁻⁴]. Drift ratios are unaffected. Label the units as they
  are.
- **Potential enstrophy units.** By dimensional analysis they are m s⁻² per
  unit density. The helper's docstring unit string should be checked when the
  catalogue entry is written.
- **Prognostic vs derived.** Prognostic fields are `ζ`, `δ`, `φ`, `T`,
  `ln p_s`. Everything else is derived and names its recipe. For example,
  `free_surface_height = (Φ₀ + φ)/g + h_s`, where `h_s` is the band-limited
  terrain.
- **Free surface.** The existing SWE panel shows the *anomaly*
  `η′ = (φ + φ_s′)/g`, with `φ_s` mean-removed. The hero shows the *full*
  `H = h + h_s` (≈ 5000–6000 m). Both belong in the catalogue as separate
  entries.
- **Statistics come from the state grid.** Peak speed, area means and ranges
  must be computed on the **state grid with its quadrature weights**, never on
  the interpolated view grid. The hero already does this with the Gauss
  weights; the generic path must too.

### 2.2 Cadence

- Per-step CSV series are lines.
- Snapshot-cadence quantities are **markers only**, with the legend suffix
  "at N saved snapshots". This covers `Z`, complexity, and point or region time
  series.
- BVE spectra and `roundtrip_residual` keep their every-10th-record cadence.
  Their NaN rows are not interpolated.
- Snapshot times are matched to CSV rows by **exact** equality, as the hero
  asserts. If a snapshot has no exact row, the panel states the nearest row's
  time. It never matches silently.

### 2.3 Unnecessary complexity to avoid

- **Faceting grammar.** No general faceting grammar, no expression language for
  derived fields, no user-registered quantities in v1. `Grid` and `Overview`
  plus TOML layouts are enough.
- **Run subtraction and runner changes.** No run-subtraction API. No change to
  the runner's plot products in this work.
- **Williamson-specific wording.** The hero's per-panel sentences (such as
  "wave train building downstream of the mountain") are Williamson-specific
  interpretation. They go in the README caption, not the generic renderer.
- **Prose columns.** The explanatory side column goes too. Definitions live in
  the caption and the linked validation report (§4.1 there already holds
  them).

### 2.4 Existing rendering defect that layering will expose

`MatplotlibRenderer._render_scalar_panel` draws with `imshow`:
- **Longitude.** The extent `[lon₀, lon_last + Δλ]` makes pixel *i* span
  `[iΔλ, (i+1)Δλ]` while its value was sampled at `iΔλ`. That is a
  **half-cell eastward shift** (≈ 1° on the 91 × 181 view).
- **Latitude.** The view grid includes both poles (91 nodes, 2° step), but
  `imshow` spreads 91 rows over 180°. Row centres are misplaced by up to
  ≈ ±1° near the poles.

Alone this is cosmetic. Under a streamline or contour overlay drawn at true
coordinates it becomes a visible registration error. The new layered panel must
use `pcolormesh` with cell edges centred on the samples, which also works for
non-uniform Gauss latitudes. Fixing the legacy panels changes in-run product
pixels, so that is optional follow-up work, recorded with a deliberate
pixel-change note.

### 2.5 Vector geometry on the equirectangular map

- **Streamlines.** Existing code is correct (`(u/cos φ, v)` in map
  coordinates). Near the poles it clamps `cos φ ≥ 1e-4`. Default to masking
  streamline seeds poleward of ±85°, and state that on the key.
- **Arrows.** Direction must follow `(u/cos φ, v)` in map coordinates, while
  length follows the physical `|V|`. Pass an explicit `angles` array to
  `quiver`. Plotting raw `(u, v)` misstates direction at high latitude.
- **Longitude seam.** Close it with one extra column for contours and
  pcolormesh, as the hero does.

---

## 3. Recommended API and CLI

### 3.1 Placement (respects the layer rules)

| New module | Layer | Role |
|---|---|---|
| `representation/diagnostics/spectral.py` | representation | Host NumPy: `kinetic_energy_modes(zeta_lm, delta_lm=None, *, radius)`, and `spectral_complexity(E)` returning `⟨ℓ⟩`, `⟨\|m\|⟩`, `S`, `N_eff`. Definitions moved verbatim from the hero script. |
| `representation/visual/quantities.py` | representation | The catalogue: frozen `Quantity` records. Pure data; imports neither CuPy nor Matplotlib. |
| `representation/visual/evaluate.py` | representation | `FieldEvaluator(storage)`: computes a quantity at `(snapshot, level)` on the **state grid** (for statistics) and on the **view sampling** (for display). It caches per storage, keyed by `(quantity, snapshot, level, sampling)`, and uses `storage.resources()` (one model build). |
| `representation/visual/views.py` | representation | Public, backend-free view objects (§3.3), serializable to and from TOML |
| `representation/visual/overview.py` | representation | Solver defaults: view objects → `FigureSpec` |
| `specs.py` (additive) | representation | `VectorGridField`, `LayeredMapSpec`, `ColorKeySpec`, and `LineSeriesSpec` gains `marker`, `draw_line` and `y_scale` |
| `temporal/simulation.py` (additive) | temporal | `Simulation.plot`, `Simulation.quantities`, and `Snapshot.plot(view=...)`, all delegating through `SnapshotStorage` (`plot_simulation`, `quantities`) |
| `cli/main.py` | cli | `tropoi plot RUN`, `tropoi inspect RUN --quantities` |

### 3.2 Signatures

These are illustrative, as the proposal requested.

```python
# temporal/simulation.py
Simulation.plot(output_path, view=None, *, snapshots=None,
                normalization="shown", renderer=None, metadata=None) -> Path
Simulation.quantities() -> QuantityTable      # host-only; what THIS run provides
Snapshot.plot(output_path, view=None, *, representation=None,
              normalization="timeline", renderer=None, metadata=None) -> Path
```

**`Snapshot.plot` stays compatible.**
- With `view=None` it renders the in-run product exactly as today. That keeps
  the pixel-identity contract; `representation` defaults to `"physical"` in that
  mode.
- With a `view` it uses the new path. Passing `representation` then is an error.

**`snapshots=`** accepts indices, or exact stored times (`"5d"`, `432000.0`).
There is no interpolation in time. An unknown time raises and lists the stored
times. The default is `Overview`'s representative selection with at most 4
maps.

**`normalization`**
- `"shown"`: limits come from the displayed snapshots. This is the overview
  default.
- `"all-saved"`: limits come from every stored snapshot, costing one synthesis
  per stored time per quantity.
- `"frame"`: limits come from each panel alone.
- PNG metadata records which mode was used.

**Output path.** A path that resolves *inside* the run directory is refused.
Saved runs stay immutable after the run.

### 3.3 View objects

```python
Map(background="free_surface_height" | None, *, level=None,
    contours=None, vectors=None, cmap=None, norm=None, title=None)
Streamlines(vector="wind", color="speed" | None, width="speed" | None, density=1.0)
Arrows(vector="wind", stride=None)
Contours(field="terrain", levels=None | [..] | interval)
Series(["total_energy", "potential_enstrophy"], transform="relative_drift" | None)
Complexity(measures=("mean_degree", "mean_order", "effective_modes"))
Spectrum(quantity="kinetic_energy_modes", by="degree")
Grid(rows=[[panel, ...], ...], at=snapshot)            # several panels, one time
Overview(map=Map(...), static=Map("terrain") | None,  # time as small multiples
         diagnostics=[Series(...), Complexity()] | None, max_maps=4)
Level(k) | Sigma(0.75)        # Pressure/Height rejected until §4.3 exists
```

- **Levels.** `Sigma(s)` resolves to the nearest full level through the existing
  `_nearest_full_level`. The label always shows the actual value, e.g.
  "σ = 0.750 (full level 5 of 6)".
- **Omitted entries.** In defaults, missing data drops the panel and the reason
  is recorded in metadata. In explicit requests, missing data raises
  `QuantityUnavailableError`.

### 3.4 The three illustrations

**Default overview.** For W5, the solver default resolves to: static terrain
map; four free-surface maps with streamlines; conservation drift; spectral
complexity.

```python
sim = open_simulation("docs/validation/williamson_5/runs/t63/w5-mri/20260730T011700Z_…")
sim.plot("w5_overview.png")
```
```bash
tropoi plot docs/validation/williamson_5/runs/t63/w5-mri/20260730T011700Z_… -o w5_overview.png
```

**Velocity-only streamlines** (a first-class product):

```python
sim[-1].plot("day15_flow.png", Map(None, vectors=Streamlines("wind", color="speed")))
```
```bash
tropoi plot RUN --snapshot -1 --map none --vectors streamlines -o day15_flow.png
```

**Custom multipanel:**

```python
sim.plot("custom.png", Grid(at=-1, rows=[
    [Map("free_surface_height", vectors=Streamlines("wind", color=None, width="speed")),
     Map("vorticity", contours=Contours("terrain", levels=[500, 1000, 1500]))],
    [Map("velocity_potential", vectors=Arrows("wind_divergent")),
     Map("ke_specific")],
    [Series(["total_energy", "potential_enstrophy"], transform="relative_drift"),
     Complexity()],
]))
# PE: Map("temperature_anomaly", level=Sigma(0.75), vectors=Streamlines("wind"))
```
```bash
tropoi plot RUN --layout custom.toml -o custom.png
```
```toml
# custom.toml, mapped 1:1 onto the view objects above
kind = "grid"
at = -1
[[rows]]
panels = [
  { map = "free_surface_height", vectors = { streamlines = "wind", width = "speed" } },
  { map = "vorticity", contours = { field = "terrain", levels = [500, 1000, 1500] } },
]
[[rows]]
panels = [ { series = ["total_energy", "potential_enstrophy"], transform = "relative_drift" },
           { complexity = true } ]
```

**Full CLI surface** (one new command):

```
tropoi plot RUN_PATH [-o PATH]
    [--snapshot I ...] [--times 0d,5d,...]
    [--map QUANTITY|none] [--contours QUANTITY[:LEVELS]]
    [--vectors streamlines|arrows|none] [--vector-quantity wind|wind_rotational|wind_divergent]
    [--level K | --sigma S]
    [--no-static] [--no-diagnostics] [--layout FILE.toml]
    [--normalization shown|all-saved|frame]
tropoi inspect RUN_PATH --quantities         # discovery; never initializes CUDA
```

With no `--map`/`--layout`, the command draws the solver default `Overview`.
With `--snapshot` and one map option, it draws a single `Map`. The `aeolus`
compatibility entry point picks the command up automatically.

### 3.5 Relation to existing pieces

- **Runner `--plot` products.** Unchanged. `overview` could later become a
  fourth runner product that calls the same builder with the in-memory model
  (optional).
- **Shared colour scales.** Map panels of one quantity share a
  `normalization_group`. `FigureTimeline.resolve_normalizations` already groups
  by name *within* one frame. Factor the resolver into
  `resolve_figure_normalizations(FigureSpec)`, keeping behaviour identical, and
  reuse it.
- **Shared colorbars.** A `ColorKeySpec(group=...)` panel draws one colorbar
  bound to a resolved group. This replaces a colorbar on every panel.
- **Time labels.** `_physical_time_label` shows seconds and hours. Add a
  formatter that chooses days for runs of a day or more: "day 5
  (t = 432 000 s)". A day is 86 400 s, not the rotation period (23.93 h for
  W5); say so on the axis.
- **Renderer protocol.** Unchanged. New panel types go through
  `_render_panel`'s dispatch.

### 3.6 Catalogue and discovery

Each `Quantity` record carries:
- `id`, `long_name` and `units`;
- `kind` ∈ {prognostic, derived, recorded, static};
- `solvers`, `dims` (level axis or not) and `cadence` ∈ {snapshot, every step,
  every 10th record, static};
- `needs` ∈ {host, cuda};
- `definition` (formula text) and `source` (a `FieldSpec` name, a CSV column
  from `diagnostic_definitions`, or a function path).

Prognostic entries are **generated from `FieldSpec`**, and recorded entries are
**generated from `diagnostic_definitions_for`**, so nothing is restated by
hand.

`sim.quantities()` / `tropoi inspect RUN --quantities` evaluates availability
**for that run**, host-only. Examples of the answers it gives:
- `terrain`: "flat bottom, not applicable"
- `kinetic_energy_spectrum_l` (recorded): "BVE-only artifact"
- `potential_enstrophy`: "needs CUDA to render; available"
- a missing CSV: "diagnostics/timeseries.csv not found"

Initial entries (★ = needed for the README figure):

| id | kind | solvers | units | cadence | needs |
|---|---|---|---|---|---|
| `vorticity` (ζ), `divergence` (δ) | prognostic | all (δ: SWE, PE) | s⁻¹ | snapshot | cuda (map); host (coeffs) |
| `streamfunction` ψ, `velocity_potential` χ | derived | ψ all; χ SWE, PE | m² s⁻¹ | snapshot | cuda |
| ★ `wind`, `wind_rotational`, `wind_divergent` (vector) | derived | all (BVE: `wind` only) | m s⁻¹ | snapshot | cuda |
| ★ `wind_speed` | derived | all | m s⁻¹ | snapshot | cuda |
| `ke_specific` | derived | all | m² s⁻² | snapshot | cuda |
| `ke_column` | derived | SWE (per unit density), PE | m³ s⁻² / J m⁻² | snapshot | cuda |
| `layer_depth` h, `thickness_anomaly` φ/g | derived | SWE | m | snapshot | cuda |
| ★ `free_surface_height` H = h + h_s; `free_surface_anomaly` η′ | derived | SWE | m | snapshot | cuda |
| ★ `terrain` h_s (band-limited) | static | SWE, PE with topography | m | static | cuda |
| `potential_vorticity` (ζ+f)/h | derived | SWE | m⁻¹ s⁻¹ | snapshot | cuda |
| `temperature`, `temperature_anomaly` | prognostic/derived | PE, per level | K | snapshot | cuda |
| `surface_pressure`, `surface_pressure_anomaly` | derived | PE | hPa | snapshot | cuda |
| ★ `kinetic_energy_modes` E(ℓ,m) and `spectral_complexity` | derived | all | m⁴ s⁻² / 1 | snapshot | **host** |
| ★ `potential_enstrophy` Z | derived | SWE | m s⁻² (per unit density) | snapshot | cuda |
| ★ `total_mass`, `total_energy` (and the other CSV columns) | recorded | per solver | as documented | every step | host |

---

## 4. Physical-field selection and future slices

### 4.1 Coefficients are not coordinates

- Slicing the stored `(l, m)` axes selects **modes**, not places.
- One exact exception is useful. For a *linear* field, the zonal mean is the
  `m = 0` part, `[f](φ) = Σ_l a_l0 Y_l0(φ)`, so it is exact and host-cheap.
- For products (`[p_s v]`, `[u v]`, `½|u|²`), the product must be formed in
  **grid space** and then averaged over uniformly spaced longitudes. That is
  exact for band-limited products when `nlon > 2 × degree`.

### 4.2 Where each step belongs

| Step | Home | Existing support |
|---|---|---|
| **Sampling geometry** (`LatLonMesh`, `Meridian(λ)`, `Parallel(φ)`, `Column(φ, λ)`, `Region(...)`) | `spatial` (pure data, later `spatial/sampling.py`) | `LatLonGridGeometry.create`, Gauss geometry |
| **Synthesis** (coefficients → values at points), exact, no interpolation | `spatial` | Device: the model transform, and `PointSetSphericalHarmonics(lats, lons, l_max)`, which evaluates at *arbitrary* points. It builds a dense `n_points × n_modes` complex matrix: fine for a column or section, ≈ 0.5 GB for a 91 × 181 view at T63, so avoid it for full maps on the MX110. Host (optional, §5 stage E): a separable Legendre-times-FFT synthesizer, NumPy, exact, pinned to the device basis by a test. |
| **Horizontal interpolation** (only for geodesic state grids) | `representation.visual.grid_adapter` | `map_to_uniform_latlon` |
| **Vertical coordinate** | `spatial.sigma_coordinate` | `SigmaGrid` (`full_levels`, `interfaces`, `thickness`); `p_k = σ_k p_s` per column; `hydrostatic_geopotential(T, Φ_s)` gives `Φ_k`, and geopotential height `Φ/g₀` |
| **Averaging** (zonal, area, region) | `representation.diagnostics` | The state-grid quadrature weights (`model.sh.weights`) |
| **Plot coordinate choice** | `representation.visual` | Level index / σ / p / Z labels (§4.3) |

### 4.3 The vertical axis stays honest

- **Axis choice.** A section's vertical axis is **level index** or **σ** by
  default, and it is labelled that way.
- **Pressure.**
  - A σ-section can be drawn against `p = σ p_s(φ, λ)` without interpolation,
    because each column gets its own vertical coordinate and `pcolormesh`
    accepts a 2-D `y`. The label then reads "pressure at σ levels".
  - Interpolation *onto* fixed pressure surfaces is a separate, explicit step.
    Values below ground are masked, never extrapolated.
- **Height.** Geopotential height needs `T` and `Φ_s`, and `Φ_s` needs the
  terrain, which means the model. Label it "geopotential height Φ/g₀". It is
  **not** geometric height; the model uses constant g.
- **Selection objects.** `Pressure(...)` and `Height(...)` selectors stay
  rejected, with a message, until that interpolation exists.

### 4.4 Future views as selections (none are prerequisites)

| View | Selection | Notes |
|---|---|---|
| Vertical profile at (φ, λ) | `Column(φ, λ)` × all levels | Point synthesis per level; exact |
| Latitude–vertical at λ | `Meridian(λ)` × levels | Exact point synthesis |
| Longitude–vertical at φ | `Parallel(φ)` × levels | Exact point synthesis |
| Zonal-mean latitude–vertical | `m = 0` for linear fields; grid-average for products | See §4.1 |
| Point or region time series | `Column` / `Region` × snapshots | **Snapshot cadence only.** States are not saved per step; markers only. Region means use the state-grid quadrature restricted to the region. |

### 4.5 Overturning and circulation cells

**Reconstructible at every saved snapshot** (nothing extra is stored):
- `V_k` comes from `ζ_k`, `δ_k`. `∇ln p_s` needs spectral derivatives (the
  model).
- `G_k = δ_k + V_k·∇ln p_s`.
- `σ̇` at interfaces comes from `interface_sigma_dot(G)`, and `ω/p` from
  `omega_over_p(G, V·∇ln p_s)`. These are the energy-consistent discrete forms
  the core itself uses, with structural top and bottom impermeability.

**Zonal-mean mass streamfunction in σ:**

    Ψ(φ, σ_{k+½}) = (2π a cos φ / g) Σ_{j ≤ k} [p_s v_j] Δσ_j

It needs the grid-space product `[p_s v]`. For a *single snapshot*,
`∂[p_s]/∂t ≠ 0`, so the zonal-mean mass flux is not non-divergent. Integrating
from the top and from the bottom then disagree, and `Ψ(σ = 1) ≠ 0` measures the
imbalance. Report that residual next to the plot.

A closed-cell reading needs a time mean over a quasi-steady state. The current
PE runs are short and unforced, with no forcing or dissipation, so no Hadley-like
cell should be expected. The plot must say "instantaneous" or "mean of N saved
snapshots".

**Fixed-longitude section.** Show `v` and `ω` (or `σ̇`) as colour, or as
direction glyphs with a stated scaling, since m s⁻¹ and s⁻¹ are incommensurate.
**Never draw a streamfunction or "cells" there**, because the flow crosses the
plane zonally.

**Physics caveat.** The hydrostatic PE core diagnoses `ω` from continuity. It
resolves large-scale ascent and descent only. It has no convection scheme and
does not resolve convection. Horizontal winds at several levels alone do not
give an overturning circulation; `σ̇`/`ω` and a mass-weighted zonal mean are
required. BVE and SWE have no vertical structure, so these views do not apply
to them.

---

## 5. Staged implementation plan

For every commit:
- CPU CI must pass (`tests.yml`), and `tools.architecture.generate --check`
  must pass. Regenerate `docs/architecture/current/` in the **same commit**
  whenever modules are added.
- Run the local GPU suite (MX110, about 8 minutes) before merging each stage.
- Import only canonical paths.

**Stage A: host diagnostics and discovery** (CPU, no rendering)
1. `feat(diagnostics): spectral kinetic-energy modes and complexity measures`.
   Adds `representation/diagnostics/spectral.py`, with the definitions moved
   verbatim from the hero script: m-doubling, `E(0,0) = 0`, zero-power modes
   contribute 0 to `S`. Tests:
   - analytic solid-body flow;
   - the tracked smoke capsule (`docs/validation/williamson_5/smoke/…`, l42,
     2 snapshots) as a CPU fixture;
   - under a CUDA marker, the T63 modal sum against the tracked npz winds at
     rtol 1e-11.
2. `feat(archive): quantity catalogue and inspect --quantities`. Adds
   `quantities.py`, generated from `FieldSpec` and `diagnostic_definitions_for`;
   `CapsuleStorage.quantities()`; `Simulation.quantities()` via the protocol;
   the CLI flag. The CPU laziness probe must show that neither CuPy nor
   Matplotlib is imported.

**Stage B: evaluation and layered rendering**

3. `refactor(visual): resolve normalization groups for a single figure`. Factors
   the resolver out of `FigureTimeline`. The existing tests must pass unchanged.
4. `feat(visual): layered map, colour key and cadence-aware series panels`. Adds
   `VectorGridField`, `LayeredMapSpec` (pcolormesh with centred edges, seam
   closure, contours, streamlines or arrows with correct angles, polar seed
   mask), `ColorKeySpec`, an "identically zero" annotation, and the
   `LineSeriesSpec` marker/line/symlog options. CPU renderer tests use synthetic
   fields; Agg is available in CI.
5. `feat(visual): field evaluator with per-run cache`. Adds `evaluate.py`: state
   grid for statistics, view grid for display; potential enstrophy through the
   existing helper; exact matching of CSV rows to snapshot times. GPU tests:
   `resource_builds == 1` and one synthesis per (quantity, snapshot, level).
   Legacy builders stay untouched.

**Stage C: public API and CLI**

6. `feat(api): Simulation.plot, Snapshot.plot(view=...) and solver overviews`.
   Adds `views.py` and `overview.py` (BVE, SWE flat, SWE with terrain, PE).
   Tests cover refusal of output paths inside the run and a capsule-unchanged
   check (tree, sizes, SHA-256).
7. `feat(cli): tropoi plot`, including `--layout` TOML. Argument and error paths
   are tested on the CPU; rendering is tested on the GPU.

**Stage D: README figure** (after a user decision, §8)

8. `docs(readme): replace the Williamson-5 hero with the generic overview`.
   - Render it with `tropoi plot`, or through a thin `docs/readme_figures.py w5`
     subcommand that calls only the public API.
   - Write `docs/assets/williamson5_t63_overview.png` plus a sidecar
     `…overview.json` holding every number shown.
   - Regenerate the caption numbers from the sidecar; drop the "Aeolus" note;
     correct the README claim that every run writes `spectra.npz` (only BVE
     does).
   - Leave `docs/validation/williamson_5/overview.png` and
     `plot_swe_holistic.py` **unchanged**. `williamson5_mri_2026-07-30.md:171`
     cites them as dated evidence.

**Stage E: optional, later, each on its own**
- a host separable synthesizer, giving CPU rendering and CI-rendered figures;
- the sampling module and `Column`/`Meridian`/`Parallel` selections;
- zonal mean;
- PE `σ̇`/`ω`/`Ψ` with its residual;
- a runner `--plot overview` product;
- the legacy `imshow` registration fix;
- persisting terrain coefficients in *new* capsules;
- pressure-surface interpolation.

---

## 6. Acceptance checks

### Numerical meaning

| # | Check | Tolerance / expectation |
|---|---|---|
| N1 | Σ E(ℓ,m) = ½∫\|u\|² dA by state-grid quadrature of synthesized winds (T63, 4 snapshots) | rtol 1e-11 (the hero's own bound) |
| N2 | Complexity reproduces the hero: ⟨ℓ⟩ 1.000 / 1.563 / 2.149 / 2.663; ⟨\|m\|⟩ 0.000 / 0.467 / 0.884 / 1.208; N_eff 1.00 / 1.89 / 3.04 / 4.42 | Same formula and inputs; ≤ 1e-12 absolute on unrounded values |
| N3 | Recomputed Z vs the tracked npz `potential_enstrophy` | Record the measured rtol; drifts must equal +4.91e-8 / −1.46e-6 / −8.52e-6 to 3 significant figures |
| N4 | Mass drift exactly 0.0 (and `swe_coeffs[:, PHI, 0, 0]` bit-constant); energy drift from CSV rows −4.85e-9 / −1.81e-7 / −7.92e-7 | Exact reads |
| N5 | Synthesized `u`, `v`, `H` on the state grid vs the npz; reconstructed `max h_s` vs CSV `terrain_max_m` = 1885.842884527992 | Record measured max abs diff; expect ≤ 1e-9 m |
| N6 | CSV row ↔ snapshot matching is exact; a forced mismatch surfaces "nearest step" text | Test with a doctored copy of the fixture in `tmp_path` |
| N7 | Every catalogue quantity has units, which appear in its key label; the KE entries carry "per unit density" where applicable | CPU test |
| N8 | Streamline and arrow geometry: solid-body `u = u₀cos φ` gives zero map-`y` component; arrow angles use `(u/cos φ, v)` | CPU test on renderer inputs |
| N9 | Exactly zero fields (W5 δ at day 0; BVE χ): annotated "≡ 0" or refused as unsupported, never given a ±1 scale | CPU + GPU |
| N10 | Cadence: per-step line length equals the CSV row count (3537 for T63); snapshot markers = 4, not joined; legend says "saved snapshots" | CPU |

### Behaviour per core

- **BVE** (`two-vortices-quick`, `rh4`):
  - The default overview is ζ with streamlines, energy and enstrophy drift per
    step, and complexity at snapshots.
  - Requesting `velocity_potential` raises with the reason "non-divergent core".
- **SWE flat** (W2): there is no terrain panel, the provenance line says "flat
  bottom", and `free_surface_height` equals `layer_depth`.
- **SWE with terrain**: W5 as above.
- **PE** (thermal-wave demo and orographic rest):
  - Level labels read "σ = … (full level k of K)", never "height" or
    "pressure".
  - `Pressure(500)` is rejected with a message.
  - No energy series is shown.
  - The terrain panel appears only with topography.
- **Sparse schedules**:
  - N = 0: diagnostics only, plus a note.
  - N = 1: one map, with single complexity markers.
  - N = 13: four maps, including the first and last.
- **Missing inputs**: a default overview with no CSV omits the diagnostics row
  and records why in metadata. An explicit `Series(...)` raises.
- **CPU/CUDA**: `sim.quantities()` and `inspect --quantities` work without
  CuPy. A CUDA-needing plot raises `PlotUnavailableError` naming the host
  alternatives.
- **Immutability**: the capsule's tree and hashes are identical before and after
  every plot. An output path inside the run is refused.

### Visual readability

- **V1** (automated): every Matplotlib `Text` artist must satisfy
  `fontsize_pt × 880 / (72 × fig_width_in) ≥ 11` CSS px. At a 7.2 in width that
  means ≥ 6.5 pt; use 7.5 pt as the floor.
- **V2** (automated): no two text bounding boxes overlap (`get_window_extent`
  after draw). The PNG is 1760–2000 px wide, height ≤ 1.35 × width, and has at
  most 10 data panels.
- **V3** (human, recorded in the PR): inspect copies downscaled to 880 px and
  440 px. Check that:
  - timestamps, units and colour-key ticks are legible;
  - the cone is visible;
  - streamline direction arrows are visible;
  - the colour-key limits are identical across the four maps.
- **V4**: perceptually uniform, colour-blind-safe maps (sequential for `H`,
  diverging only for signed fields) on a white background, which reads in
  GitHub's dark mode too.
- **V5**: deterministic output. Two renders give identical pixels. No wall-clock
  timestamps go into PNG metadata.

### README reproduction

- **R1.** One documented public-API command regenerates the figure from the
  canonical capsule. PNG metadata records:
  - run id, solver, grid and `l_max`;
  - capsule path;
  - `CoefficientsSHA256 = dd28eff9…` and `DiagnosticsSHA256 = 3d8c8cb4…`;
  - renderer commit and catalogue version.
- **R2.** The sidecar JSON values equal N2–N4. The caption numbers are copied
  from the sidecar.
- **R3.** Every item of the current figure's information survives:
  - four times with timestamps;
  - `H` with a shared key, velocity, and terrain;
  - mass, energy and Z drift;
  - ⟨ℓ⟩, ⟨|m|⟩ and N_eff;
  - grid, `l_max` and provenance.

  Area-mean speed, peak speed and the per-snapshot `H` range move to the sidecar
  (peak speed also appears as the speed-key maximum). That is a deliberate
  trade; veto it if you want a stats table.
- **R4.** The proposed README layout (7.2 in wide):
  - title and a one-line provenance;
  - a static terrain map beside the shared `H` key and speed key;
  - a 2 × 2 grid of the four time maps;
  - a bottom strip: a symlog drift panel (mass as an exact 0, annotated
    "bit-identical"; energy as a per-step line; Z as markers) and a complexity
    panel (three dimensionless measures, all within 0–5, on one axis, as
    markers).

---

## 7. Legacy and external-artifact incompatibilities

- **Canonical capsule (668e6c9a):**
  - It has no `state_schema` or `diagnostic_definitions` block. Inference works
    (verified).
  - `argv` and `out` name `/content/...` and `planetary_sandbox`. Never read
    paths from `argv`.
  - `plots: []`, so there is no in-run product to diff against.
- **Hero physics source.** The hero's physical fields come from the
  Colab-produced npz. The generic path re-synthesizes from coefficients with
  current code on a different GPU. Expect roundoff-level differences, and
  measure them (N5); do not assume they are zero.
- **Terrain.** Terrain is not persisted. Rendering the static panel rebuilds the
  model, which runs the W5 cone path with its 0.25 projection gate; that is fine
  at l63. On the MX110 the T63 model must build **alone on the GPU**; a shared
  CuPy process has caused TDR/OOM before.
- **Traps:**
  - `artifacts/w5-mri-validation/` is the superseded v2 pipeline (42bcc38c) and
    must never be used as input.
  - The tracked `smoke/` capsule is a different run (l42, `08f7b901`, 6 h). It
    is a fixture, not evidence.
- **Old names.** The old figure says "Aeolus". Old manifests keep old module
  paths in `diagnostic_definitions.source`. Those strings are frozen history, so
  display them without rewriting.

## 8. Blockers vs optional, and decisions needed

**Blockers for the README replacement:**
- the capsule-availability decision (below);
- §5 commits 1 and 3–8;
- the layered-map registration fix inside the new panel.

**Not blockers:**
- commit 2's CLI flag (the catalogue itself is needed);
- everything in stage E;
- runner integration;
- run subtraction;
- the MRI comparison figures, which stay as they are.

**Decisions for you:**
1. **Capsule in git?**
   - (a) Commit the canonical T63 capsule (≈ 1.9 MB: manifest, config,
     coefficients, times, CSV) with checksums. This makes README regeneration
     and N1–N5 reproducible from a clone, but it reverses the Drive-only policy
     for this one capsule.
   - (b) Keep it on Drive, add `swe_coeffs.npy`'s hash to `CHECKSUMS.txt`, and
     have the figure command verify it.

   *Recommendation: (a). It is smaller than the npz already tracked.*
2. **Public import spelling** for the view objects: the long canonical path, or a
   short re-export (for example from `tropoi.representation.archive` next to
   `open_simulation`).
3. **`Snapshot.plot()` without a view** keeps the in-run product style
   (recommended, since it preserves the pixel contract) rather than switching to
   the new style.
4. **New figure path.** Use a new file under `docs/assets/` (recommended) rather
   than overwriting the cited `overview.png`.
5. **Potential enstrophy** computed on the GPU through the existing helper now
   (recommended), with a host Gauss-quadrature version deferred to stage E.
