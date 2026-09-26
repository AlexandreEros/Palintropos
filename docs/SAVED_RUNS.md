# Saved runs: the read-only simulation interface

A finished BVE, shallow-water, or primitive-equation run capsule (see
[ARCHITECTURE.md](ARCHITECTURE.md), "Output capsules and provenance") can be
opened on any host, inspected without a GPU, read through named coefficient
views, and rendered on explicit request with the same scientific
compositions the run itself produced.

```python
from tropoi.representation.archive import open_simulation

sim = open_simulation(run_path)          # a run directory, or a base dir with latest_run.txt
sim.times                                # saved times in seconds; read-only host array
snapshot = sim[137]                      # saved output index (negative indices count back)
snapshot.time                            # seconds
snapshot.metadata                        # schema, units, environment, provenance, run config
snapshot.state["temperature"].coeffs     # PE host view, shape (nlev, l_max+1, l_max+1)
snapshot.plot(output_path=path)          # explicit, lazy; returns pathlib.Path
sim.plot(path)                           # overview of the whole run (see "Drawing views")
```

The same information is available from the shell without Python:

```bash
tropoi inspect RUN_PATH --snapshot 137 --field temperature
tropoi plot RUN_PATH                     # the overview; --list-quantities needs no GPU
```

## What a snapshot exposes

`snapshot.state` is an immutable mapping from field name to a read-only
`SpectralModes` view. Each view carries `coeffs`, `units`, `description`,
`dimensions`, `l_max`, and (for levelled fields) `nlev` and `level_values`
(full-level sigma coordinates, top to bottom). The stored coefficient
conventions are the repository's: complex orthonormal spherical harmonics for
`m >= 0`, axis 0 = degree, axis 1 = order, `m > l` entries zero, `Im a_l0 = 0`.

| Solver | Fields | Shape of `coeffs` | Meaning |
| --- | --- | --- | --- |
| BVE | `zeta` | `(L+1, L+1)` | relative vorticity (s^-1); circulation monopole conserved, not pinned |
| SWE | `zeta`, `delta`, `phi` | `(L+1, L+1)` each | vorticity, divergence (s^-1); `phi` is the **perturbation thickness geopotential** (m^2 s^-2): `Phi = g h = Phi0 + phi`, `Phi0 = gravity * mean_depth_m`; the free surface adds the fixed `phi_s` over topography |
| PE | `zeta`, `delta`, `temperature` | `(nlev, L+1, L+1)` | per full level, top to bottom; `temperature` is the **full** temperature (K), its monopole is the horizontal mean |
| PE | `ln_ps` | `(L+1, L+1)` | natural logarithm of surface pressure (ln Pa); a single surface row, never padded with a level axis |

Views are lightweight slices of the packed storage: `snapshot.state[name].coeffs`
shares memory with the capsule's memory-mapped array and rejects writes. A
snapshot keeps its own reference to its frame, so it stays valid after
another snapshot is selected or the `Simulation` is released. No solver
arithmetic lives here; the views describe stored states, not rates.

Opening validates the metadata, the `.npy` headers, the array layout/dtype
implied by the schema, and the time axis (finite, strictly increasing, one
entry per stored frame). Coefficient values are not scanned at open; the
convention checks (finite, triangular padding, real `a_l0`, monopole rule)
run through `SpectralModes.validate()` and are reported by `tropoi inspect
--snapshot`. Nothing is ever repaired. An empty saved sequence
(`--n-snapshots 0`) opens; indexing it fails clearly.

## Where the interpretation comes from

New manifests carry two additive, versioned blocks derived from the resolved
run configuration: `state_schema` (field names and row layout, dimensions,
units, normalization and reality conventions, support with the
product-truncation cut, geometry, vertical sigma coordinates, time units,
prescribed environment and initialization provenance) and
`diagnostic_definitions` (the meaning of every `diagnostics/timeseries.csv`
column; for the BVE the structured `high_l_enstrophy_frac` definition:
`l > cut` band, triangular domain, positive-order multiplicity, denominator
and zero-power convention, definition version). They describe; they never
enter `run_config` or the scientific run-id hash, and existing CSV values
keep their meaning.

Capsules written before these blocks existed are interpreted by inference,
limited to the **known historical formats**, and the result says so in
`snapshot.metadata["provenance"]`:

- a manifest with `run_config.solver` but no `state_schema` uses the
  solver's documented storage layout (`provenance.schema.source ==
  "inferred"`, with the convention source named);
- a capsule with **no** `solver` key is recognised as the historical
  `psx-bve` capsule only when it holds `vorticity_coeffs.npy` and a
  `viscosity` key; missing `grid` / `product_quadrature` keys take the only
  backend and the `Planet.generate` default that existed then, recorded
  under `inferred_defaults`;
- a legacy BVE capsule without `bve_snapshot_times.npy` gets the interval
  schedule `t_k = k * dt_snapshots` (`provenance.time_axis.source ==
  "inferred"`).

Anything else is refused rather than guessed: an unknown layout fails at
open, and a manifest whose `state_schema.version` this reader does not know
keeps ordinary metadata inspection working but refuses typed field access
with `UnknownSchemaVersionError`. Legacy capsules are never re-validated
through newer preset guards and their arrays are never projected on load.

## CPU / GPU boundary

Opening, `sim.times`, `snapshot.metadata`, `snapshot.state[...]`, and
`tropoi inspect` (with or without `--snapshot`) never import CuPy,
Matplotlib, the numerical cores (`tropoi.temporal.tendencies`), the
discretization (`tropoi.spatial` grids, transforms, operators), or the
visualization package (`tropoi.representation.visual`); this is enforced by
fresh-interpreter tests. Opening memory-maps the coefficient
array read-only, so its cost does not scale with the total payload
(`open_simulation(path, mmap=False)` loads it into host memory instead,
still read-only).

`snapshot.plot(...)` and `sim.plot(...)` are the only operations that may
need a GPU, and they import what they need on the call. Without a view,
`snapshot.plot` needs:

| Representation | BVE | SWE | PE |
| --- | --- | --- | --- |
| `"physical"` (default) | CUDA | CUDA | CUDA |
| `"spectral"` | CPU | CPU | not provided |

Every physical composition synthesizes fields through the run's own
spherical-harmonic transform (the BVE frame derives streamfunction and winds
spectrally even though a grid field was saved), so it is CUDA-gated. Without
CUDA the call raises `PlotUnavailableError` naming the alternative;
metadata and host coefficient access are unaffected. The coefficient-space
(`"spectral"`) frames need no synthesis and render on the host.

The model/backend resources a plot needs (planet, grid, transform, terrain,
sigma grid) are reconstructed once per opened capsule from its persisted
configuration through the solver modules' own builders, then shared by every
snapshot, representation, and field; they are never built per field.

## Plotting the run's own snapshot figure

Without a view, `snapshot.plot` reuses the existing per-core figure builders
and the
existing atomic PNG renderer; no second renderer exists. The image is the
frame the run's `snapshots/<representation>/` product would contain for that
time: same panels, interpolation, palettes, and layout.

- `normalization="timeline"` (default) composes **every** stored frame so
  the color limits are the run-wide limits of the persisted product. On CUDA
  the resulting image is pixel-identical to the frame the in-run product
  renders (tested for representative BVE, Williamson-2, Williamson-5, and PE
  frames). This costs one synthesis per stored frame.
- `normalization="frame"` composes only the selected frame; its limits come
  from that single state and are not comparable across times.

The PNG metadata records `Representation` and `Normalization`, so an image's
provenance is inspectable after the fact.

## Drawing views: `Simulation.plot` and `tropoi plot`

`sim.plot(path, view=None)` draws a view of the whole run, and
`snapshot.plot(path, view)` draws one at a single saved state. The same
views are available from the shell as `tropoi plot RUN_PATH` (see
`tropoi plot --help`). A view says what to draw and holds no data:

```python
from tropoi.representation.visual.views import (
    Complexity, Contours, Drift, Grid, Map, Overview, Sigma, Streamlines)

sim.plot("overview.png")                                  # solver default
sim.plot("h.png", Overview(map=Map("free_surface_height",
                                   contours=(Contours("terrain", (500.0, 1000.0)),),
                                   vectors=Streamlines()),
                           snapshots=(0, "5d", -1)))
sim[-1].plot("flow.png", Map(None, vectors=Streamlines()))   # streamlines alone
sim[2].plot("grid.png", Grid(((Map("vorticity"), Map("divergence")),
                              (Drift(("total_energy",)), Complexity()))))
sim.plot("pe.png", Overview(map=Map("temperature_anomaly", level=Sigma(0.75),
                                    vectors=Streamlines())))
```

- **Default overviews.** BVE: vorticity with streamlines. SWE: free-surface
  height with streamlines, plus terrain contours and a static terrain map when
  the run has terrain. PE: temperature anomaly with streamlines at the full
  level nearest σ = 0.75. Each overview adds a conservation panel (per-step
  columns of `diagnostics/timeseries.csv`, and for SWE the potential
  enstrophy at the saved states) and a kinetic-energy spectral-complexity
  panel. Parts a run cannot provide are omitted from a default and listed in
  the sidecar. The same parts requested explicitly raise
  `QuantityUnavailableError` with the reason.
- **Quantities.** `sim.quantities()` (or `tropoi plot RUN --list-quantities`)
  lists each quantity's meaning, units, kind (prognostic, derived, static,
  recorded), cadence, and whether it needs CUDA, with the reason when this run
  cannot provide it. For example, BVE has no divergence or velocity potential:
  both are identically zero by construction. This listing never imports CuPy
  or Matplotlib. In SWE and PE the streamfunction is the rotational part of
  the flow only; `wind` is the full flow.
- **Levels.** PE maps take `level=K` (0-based from the top) or `Sigma(s)`, the
  full level nearest `s`. Labels always show the level's actual σ = p/p_s,
  never pressure or height.
- **Shared scales.** The maps of an overview share one colour scale, and their
  winds share one speed scale. A single colour key and a single line-width key
  show exactly those scales. Grid panels each have their own.
- **Where numbers come from.** Fields are evaluated once per (quantity, saved
  state, level) through the run's model, which is built once per opened
  capsule. Statistics (ranges, area means, peak speeds, potential enstrophy)
  are computed on the state grid with its quadrature weights. Gauss-grid runs
  are drawn on their own samples with no interpolation. Geodesic runs are
  interpolated onto the shared 91 × 181 view grid for drawing only.
- **Cadence.** Per-step columns are drawn as lines. Values that exist only at
  saved states (potential enstrophy, spectral complexity) are drawn as
  unconnected markers, and their labels say so. Snapshot times are matched to
  CSV rows exactly, and never by nearest time.
- **Rest.** Winds whose largest speed (or, for the spectrum, whose rms speed)
  is below 1e-9 m/s are treated as rest. They are not drawn, and no
  kinetic-energy distribution is reported for them: a state at rest carries
  roundoff of about 1e-15 m/s, whose "spectrum" means nothing. Fields that are
  identically zero are labelled as such.
- **Streamlines** are instantaneous curves tangent to one saved wind; they are
  not particle trajectories, which sparse saved states cannot provide. On the
  map, the direction of motion is (u / cos φ, v), for streamlines and arrows
  alike.
- **Provenance.** PNG metadata records the run id, solver, commit, the
  SHA-256 of the coefficient file and of `diagnostics/timeseries.csv`, and the
  full view. With `sidecar=True` (or `--sidecar`), every number shown is also
  written to `<output>.json`. The sidecar is reproducible byte for byte. The
  image is not guaranteed to be: with streamlines, Matplotlib's
  rasterization varies between processes by a few antialiased pixels,
  although the streamline geometry itself is identical.
- **Where figures go.** `sim.plot()` and `tropoi plot RUN` write
  `RUN/assets/overview.png` (with `--sidecar`, also `overview.json`); other
  views from the command get a name spelling out their options, and `-o`
  chooses any path. `RUN/assets/` holds derived, reproducible products. It is
  the only place inside a run that plotting writes, it never enters the run
  id, the completion status, or a published run's `SHA256SUMS`, and
  `--overwrite` sweeps it with the replaced run's other generated results.

Figures meant to stay fixed should spell out every field of every view
object, as the README figure's recipe does
(`docs/figures/williamson5_t63_overview.py`, enforced by
`tests/test_readme_figure_recipe.py`), so that a later change of a default
cannot alter them silently.

## Where the implementation lives

The interface follows the package's spatial / temporal / representation
layout ([ARCHITECTURE.md](ARCHITECTURE.md), "Package layout"):

| Piece | Module | Needs |
| --- | --- | --- |
| `open_simulation`, capsule layout, legacy inference | `tropoi.representation.archive` (`capsule.py`) | CPU |
| `state_schema` / `diagnostic_definitions` (read and written) | `tropoi.representation.archive.schema` | CPU |
| run ids, run directories, manifests (the writer) | `tropoi.representation.archive.writer` | CPU |
| `Simulation`, `Snapshot`, the `SnapshotStorage` protocol | `tropoi.temporal.simulation` | CPU |
| `FieldSpec`, `SpectralModes`, `SpectralState` host views | `tropoi.spatial.modes` | CPU |
| `Snapshot.plot` adapter (no view) | `tropoi.representation.visual.snapshot` | per the table above |
| view objects | `tropoi.representation.visual.views` | CPU |
| quantity catalogue | `tropoi.representation.visual.quantities` | CPU |
| field evaluation and caching | `tropoi.representation.visual.evaluate` | CUDA for fields |
| view composition, `render_view` | `tropoi.representation.visual.compose` | CUDA for fields |
| layered maps, keys, renderer | `tropoi.representation.visual.{specs,timeline,matplotlib_renderer}` | CPU |
| kinetic-energy spectral complexity | `tropoi.representation.diagnostics.spectral` | CPU |
| per-core figure compositions | `tropoi.representation.visual.{bve,swe,pe_snapshots}` | per the table above |
| model resources a physical plot rebuilds | `tropoi.spatial` (planet, grids, transforms, terrain, sigma grid), via the CLI solver modules' builders | CUDA |

`Simulation` depends only on the storage protocol; the capsule format is one
implementation of it, outside the temporal layer. Capsules written before
this layout (including those from earlier 0.1 builds and the historical
`psx-bve` format) are read unchanged; the reorganization did not alter any stored
array, schema block, or file name. Descriptive paths that older manifests
record (for example `run/bve/diagnostics.py` in `diagnostic_definitions`)
name the modules of their time, which remain importable as compatibility
aliases.

## What this interface does not do

No interpolation in time (`sim[i]` is a saved output index, never a time),
no arbitrary initial conditions, no temporal spectra, no restart
checkpoints, and no codecs. Plotting draws horizontal maps and diagnostic
series only: no particle trajectories, and no vertical profiles or sections,
zonal means, or interpolation to pressure or height yet. These would be new
samplings of the same per-level state-grid fields in
`tropoi.representation.visual.evaluate`. Diagnostic time series remain in
`diagnostics/timeseries.csv` with their documented meanings.
