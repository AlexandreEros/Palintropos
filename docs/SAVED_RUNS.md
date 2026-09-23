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
```

The same information is available from the shell without Python:

```bash
tropoi inspect RUN_PATH --snapshot 137 --field temperature
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

`snapshot.plot(...)` is the only operation that may need a GPU, and it
imports what it needs on the call:

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

## Plotting: what is rendered and how it is normalized

`snapshot.plot` reuses the existing per-core figure builders and the
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
| `Snapshot.plot` adapter | `tropoi.representation.visual.snapshot` | per the table above |
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
checkpoints, no codecs, and no new plotting framework. Diagnostic time
series remain in `diagnostics/timeseries.csv` with their documented meanings.
