# Airless-body thermal experiments

`tropoi.run.airless` runs independent conductive columns
(`tropoi.physics.substrate.ConductiveSubstrate`) under the shared orbit and
rotation illumination geometry (`tropoi.physics.insolation`). Runs execute on
CUDA (CuPy); the canonical C++ step is the same on the CPU, which is used for
verification only and only when requested (`--device cpu`, hidden from
`--help`, or `TROPOI_EXECUTION=cpu-verification`; see
[`run/execution.py`](../../../src/tropoi/run/execution.py)). Without a usable
GPU a run fails at once instead of falling back to the CPU. Presets are
configurations of that machinery; Mercury with the Vasavada, Paige & Wood
(1999) regolith is the primary one. Tier-2 output is streamed to disk in
per-block files and read back with `load_tier2`.

* entry path: [`calls/airless-mercury.mmd`](../current/calls/airless-mercury.mmd)
* acceptance register: [`../../validation/PLANET-ONE-GATES.md`](../../validation/PLANET-ONE-GATES.md)
* status and handoff: [`../../validation/PLANET-ONE-STATUS.md`](../../validation/PLANET-ONE-STATUS.md)

## Global discretizations and diagnostic points

A `latlon` grid (regular, cell-centred) or a `gauss` grid (the
Gauss–Legendre grid of the spectral atmospheric models, with exactly their
nodes, ordering and weights, and no spectral transform; see
[`spatial/grids/quadrature.py`](../../../src/tropoi/spatial/grids/quadrature.py))
is a **global** discretization. Its columns carry their exact
cell areas (m², summing to 4πR²), and the run publishes area integrals
(`series/*.npz`, in W) and area means (`mean_absorbed_W_m2`,
`mean_emitted_W_m2`, `global_residual_over_tolerance` in `cycles.json`).

A `points` collection is **diagnostic**. Its columns are independent sample
locations with no physical area, so the run publishes nothing global:

* `grid.npz` has no `weights`;
* `series/*.npz` holds `per_point_<quantity>_W_m2` with shape (steps,
  points);
* each `cycles.json` entry lists `points` (column index, latitude,
  longitude, absorbed, emitted, energy imbalance and residual/tolerance per
  point).

The vertical mesh is spatial discretization. `build_substrate` builds it
with `tropoi.spatial.vertical.stratified_interfaces`, and the substrate
receives explicit faces. `grid.npz` stores the column coordinates in
radians (`lat_rad`, `lon_rad`), which is what the runner uses, as well as in
degrees.

`column_kind` (`global` or `diagnostic`) is recorded in `config.json`,
`grid.npz` and every cycle entry.

Point runs written before this distinction stored unit weights. Their
`series` values are plain sums over the points, and their `mean_*` and
`global_*` values are unweighted point averages, not global quantities. Read
them per point from `integrals.npz` instead.

## Running a saved configuration

Any body the configuration schema can describe runs from its saved JSON
document, with no preset and no Python:

```
python -m tropoi.run.airless show-config --config FILE [--steps N | --duration SECONDS]
python -m tropoi.run.airless run --config FILE --out DIR [--steps N | --duration SECONDS]
```

`FILE` holds either the configuration itself (the `AirlessConfig.to_dict`
form) or an envelope with it under `"config"`, such as a run's own
`config.json`. Other envelope keys are ignored, except `config_sha256`,
which must match the loaded configuration if it is present. The
configuration is used exactly as saved. The options that build or adjust a
preset (`--points`, `--grid`, `--depth`, `--cycle-steps` and so on) are
refused with `--config`. The run's `config.json` records the document's
path, its file SHA-256 and whether it was an envelope, as `config_source`.

## Two kinds of run

**Periodic** (no `--steps` or `--duration`). The run iterates illumination
cycles of length `cycle_s`: an optional coarse spin-up, then production
cycles, until the G-P2 periodicity criteria hold in consecutive production
cycles or a cycle limit is reached. The retained status is `PERIODIC` or
`TRANSIENT`. Because successive cycles are compared phase by phase, `cycle_s`
must be an exact recurrence period of the illumination.

**Finite transient** (`--steps N` or `--duration SECONDS`). The run starts
from the configured initial state and takes exactly `N` steps of
`dt = cycle_s / cycle_steps`, then stops. A duration is met by the first
step that reaches it (`ceil(T/dt)`, with a time within 1e-9 relative of a
whole number of steps taken as that number). The status records the actual
final time `N * dt` beside the request. There is no spin-up, acceleration
or periodicity test, and `cycle_s` is not assumed to be a recurrence
period. It only sets the timestep and the segment length. The steps are
written in segments of `cycle_steps` steps (the last one shorter if
needed). Each segment has the same tier-1 and tier-2 output and the same
numerical diagnostics as a cycle of a periodic run. Step `m` (from 0) is
forced at `(m + 1) * dt`, the same arithmetic as the periodic run at the
production timestep. The retained status is `COMPLETED_TRANSIENT`, and its
`claims` keep four things apart:

| Claim | What a transient run states |
|---|---|
| solver diagnostics | reported per segment in `cycles.json`: the largest Newton residual ratio \|r\|/τ (≤ 1 at every step, or the step raises) and the energy-identity residuals |
| numerical reproduction | not evaluated; it needs a comparison with a reference, which the run does not make |
| periodic equilibrium | not evaluated |
| physical validation | none |

Each segment also records the SHA-256 of the configuration the worker
received (`worker_config_sha256`), its timestep and its first global step.
The run refuses to continue if a worker ran a different configuration.

With more than one worker, the pool uses `fork` where the platform has it
and `spawn` otherwise (Windows, and always with CuPy). A single worker
starts no pool.

## Example: an equatorial column on an idealized Ceres

[`tests/fixtures/airless/ceres-equatorial-column.airless.json`](../../../tests/fixtures/airless/ceres-equatorial-column.airless.json)
is an envelope document for one flat equatorial column on a smooth Ceres in
a Keplerian orbit. Its `provenance` block separates measured, derived,
inferred and assumed inputs. Its `cycle_s` is the mean solar day, which is
not an exact recurrence for this eccentric, oblique orbit, so it is run as a
transient:

```
python -m tropoi.run.airless run --config tests/fixtures/airless/ceres-equatorial-column.airless.json --steps 8640 --out runs/ceres-transient
```

This takes two mean solar days (t = 65348.734 s) of 7.5635 s steps from the
160 K isothermal start. When this command was accepted, its final thermal
state was compared with an independent driver that called the same worker
cycle by cycle, and the two agreed bit for bit. That comparison, not the run
itself, is what established numerical reproduction. The result is not a
physical model of Ceres. Its vertical grid (1 mm top cell) was not refined
for Ceres's 1.6 mm diurnal skin depth, and its material inputs are partly
assumed.
