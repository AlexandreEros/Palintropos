# Held–Suarez climate figure (issue #17, CPU first pass)

`Simulation.plot()` on a Held–Suarez experiment with saved states draws the six
conventional latitude–σ climate sections as one PNG plus a PDF with the same
stem, computed on the CPU from the run's saved states:

```python
from tropoi.representation.archive import open_simulation

sim = open_simulation("runs/hs-T42L20-prod-002")
png = sim.plot()                    # RUN/assets/overview.png + overview.pdf
png = sim.plot("hs_climate.png")    # explicit PNG path + hs_climate.pdf
png = sim.plot("hs_late.png", interval=(1000, 1200))   # 1000 < day <= 1200
png = sim.plot("hs.png", scales={"u": range(-50, 55, 5)}, sidecar=True)
```

The return value is the PNG path. The PDF's file name is in the PNG metadata
(`PDF`), and with `sidecar=True` it is also in `<png>.json`. The sidecar holds every
number drawn: the six arrays, the grid, the sample list with each state file's
SHA-256, the measured bandlimit and the scales. Outputs follow the assets rule:
inside the run directory only `assets/` is written, and nothing there is evidence.

Code: reduction in `tropoi.representation.diagnostics.held_suarez_sections`
(NumPy only), drawing in `tropoi.representation.visual.held_suarez_climate`,
delegation in `HeldSuarezStorage.plot_simulation`. Neither module imports
`tropoi.run`, CuPy or a model.

## What each panel is

`[ ]` is the plain mean over the `nlon` equally spaced longitudes at one latitude
on one native full σ level. `*` is the departure from that state's instantaneous
`[ ]`. Products are formed for each state, then averaged over time.

| key | panel | statistic | units |
| --- | --- | --- | --- |
| `u` | Mean zonal wind | time mean of [u] | m s⁻¹ |
| `T` | Mean temperature | time mean of [T] | K |
| `usvs` | Eddy momentum flux | time mean of [u* v*] | m² s⁻² |
| `vsTs` | Eddy heat flux | time mean of [v* T*] | K m s⁻¹ |
| `eke` | Eddy kinetic energy | ½ time mean of [u*² + v*²] | m² s⁻² |
| `TsTs` | Eddy temperature variance | time mean of [T*²] | K² |

This is the run-side accumulator's spatial-anomaly definition
(`run/held_suarez/experiment.py: daily_sample`). It is not a temporal-anomaly
("transient eddy") definition. The accumulator has no [v*²], so full EKE exists
only from saved states. These are σ-level means, not pressure-level means.

## Sampling

* Window: `start < day <= end`. The default is the accumulator's window,
  `spinup_days < day <= spinup_days + n_blocks * block_days`. An explicit
  `interval=(start, end)` uses the same rule.
* Every saved state inside the window is used, with equal weight `1/N`. There is no
  trapezoidal weighting, interpolation or gap filling.
* The header prints the rule, the requested interval, the actual first and last
  sample days, the count and the cadence. Irregular spacing is printed as a
  histogram of intervals.
* Coverage is checked against the nominal save days of the window (multiples of
  `state_every_days`). Missing days are labelled `INCOMPLETE` and listed, never
  filled.
* Defined failures (`SectionError`, a `ValueError`):
  * an empty, reversed, negative or non-finite interval;
  * fewer than `min_states` (default 2) states in the window;
  * an experiment with no `states/`. The message says the figure needs saved
    states and that plotting from statistics alone is not implemented.
* Explicit views and per-snapshot figures still refuse
  (`HeldSuarezPlotError`). `Snapshot.plot(path)` is unchanged.

## Grid, bandlimit, orientation

* Reduction runs on the run's own Gaussian state grid. Latitudes come from
  `statistics.npz['lat']`, checked to 1e-9° against the configured `nlat`
  Gauss–Legendre nodes. If that key is absent, the same nodes are rebuilt. A
  different grid is refused, never substituted. Longitudes are `360 j / nlon`
  degrees, the model's convention.
* Only u, v and T are synthesized, one frame at a time, by exact host synthesis
  (`synthesize_latlon`, `synthesize_wind_latlon`). Vorticity, divergence,
  geopotential and pressure are never synthesized. Memory is one frame plus the
  running sums.
* The zonal bandlimit `m_max` is measured: it is the highest order with any nonzero
  ζ, δ or T coefficient in any selected state. A populated buffer degree of the
  stored-`l_max = L + 1` / retained-`L` layout therefore counts. `nlon > 2 m_max`
  is required, otherwise the reduction refuses, because longitude means of
  products would alias. This makes the longitude quadrature exact. It says nothing
  about any GPU versus host synthesis difference. The sidecar records the stored
  `l_max`, the nominal truncation, the measured maximum degree and order, and the
  largest coefficient above the nominal truncation.
* Arrays are `(nlat, nlev)`, latitude north → south as stored, level 0 = top, the
  `statistics.npz` layout. The figure draws latitude south → north and σ linear,
  increasing downward, over the native full levels. Each level is a minor tick.

## Fixed comparison scales (`HS_SECTION_SCALES`)

These are a repository recipe. They are not copied from a published figure.
Each range covers the measured climate of `hs-T42L20-prod-002` (days 210–1200,
table below) with headroom, so runs drawn with the defaults compare panel by panel.
Signed fields use symmetric diverging maps with a bold zero contour. T, EKE and T*²
use sequential maps. Values beyond a scale take the colour bar's extension colours,
are cross-hatched, and the panel note gives the data range and the share of points
beyond the scale. Override any panel with `scales={key: levels}` or a
`SectionScale`; the footer then names the overridden panels.

| key | range | colour interval | labelled lines | map |
| --- | --- | --- | --- | --- |
| `u` | -40 to 40 m s⁻¹ | 5 | every 10 | RdBu_r |
| `T` | 185 to 315 K | 5 | every 10 | YlOrRd |
| `usvs` | -80 to 80 m² s⁻² | 10 | every 20 | PuOr_r |
| `vsTs` | -24 to 24 K m s⁻¹ | 2 | every 4 | RdBu_r |
| `eke` | 0 to 450 m² s⁻² | 25 | every 50 | YlGnBu |
| `TsTs` | 0 to 50 K² | 2.5 | every 5 | PuRd |

The panel set follows Held & Suarez (1994) and Kawai & Tomita (2025, Figure 10).
The white, print-friendly style follows polyclim, except that this figure uses a
linear σ axis and has no tropopause or season labels.

## Real-run demonstration (CPU)

`runs/hs-T42L20-prod-002` (local, gitignored): config `d1fa645d…`, commit
`a52ea205`, T42 retained (stored `l_max` 43), L20, 64 × 128, dt 720 s, centred
damping. Default window `200 < day <= 1200`, with 100 states (days 210–1200 every
10 days), coverage complete. Measured `m_max` = 42 < 64 = nlon/2. The buffer
degree 43 is exactly zero in every selected state. The stored latitudes match the
rebuilt Gauss nodes to 7e-15°.

| panel | data min | data max |
| --- | --- | --- |
| [u] | -23.9 | 32.6 m s⁻¹ |
| [T] | 189.3 | 305.9 K |
| [u*v*] | -73.1 | 71.8 m² s⁻² |
| [v*T*] | -21.8 | 22.2 K m s⁻¹ |
| ½[u*²+v*²] | 1.96 | 393.9 m² s⁻² |
| [T*²] | 0.04 | 43.1 K² |

Performance (Windows, the development laptop's CPU, one process, final code):
50.6 s wall for `sim.plot(sidecar=True)`. Of that, 46.8 s is reduction: reading
and hashing 100 state files of 1.9 MB each and synthesizing u, v, T on 20 levels,
about 0.5 s per state. Drawing and saving both files takes about 4 s. The process
peak working set is 202 MiB, against 60 MiB after the imports. The reduction itself
is bounded by one frame plus the sums: an earlier run without the PDF band
rasterization peaked at 109 MiB (62.7 s wall), so most of the difference is the
rasterized PDF layers. Wall time varied between 50 and 63 s across runs. Running
under `tracemalloc` inflates the time about sevenfold (444 s), so do not time with
it.

### Descriptive comparison with the run's own accumulators (different cadence)

`statistics.npz` holds daily samples (5 blocks of 200 days, days 201–1200). The
figure uses every tenth day (210–1200). The numbers below are measured differences
between two different samples of the same climate. They are not a tolerance, and
the block spread is not a standard error. Both are on the same grid; the
area- and Δσ-weighted RMS uses the stored Gauss weights.

| field | max abs diff | weighted RMS diff | accumulator peak | block-to-block spread (RMS) |
| --- | --- | --- | --- | --- |
| [u] | 0.54 | 0.16 | 32.8 | 1.50 m s⁻¹ |
| [T] | 0.24 | 0.04 | 305.9 | 0.36 K |
| [u*v*] | 5.50 | 1.24 | 72.9 | 2.89 m² s⁻² |
| [v*T*] | 1.70 | 0.31 | 21.5 | 0.50 K m s⁻¹ |
| [T*²] | 1.76 | 0.33 | 41.7 | 1.47 K² |

Daily block means cannot recover the every-tenth-day subset, so this comparison
cannot isolate a GPU versus host synthesis difference.

## Verification in this pass (CPU only)

`tests/test_held_suarez_climate.py` uses manufactured degree-1 spectral states with
closed-form statistics:

* zonal means, the signs and magnitudes of the momentum and heat fluxes, full EKE
  including the [v*²] the accumulator lacks, and [T*²];
* products formed before averaging (alternating-sign eddies) and equal weights
  under irregular spacing;
* the open-left, closed-right window, incomplete-coverage disclosure, invalid
  intervals, insufficient states, no states and zero fields;
* latitude and level ordering;
* the retained/buffer layout: a populated degree `l_max` is measured, reduced and
  refused when `nlon <= 2 m_max`;
* one frame at a time, and only u, v and T synthesized;
* the stored-versus-rebuilt grid, and refusal of a different grid;
* the public API: default and explicit paths, the PDF beside the PNG, metadata,
  sidecar, overrides, the assets rule, refused views, and unchanged snapshot
  semantics.

These tests do **not** stand in for the production-fixture comparison below.

## Outstanding acceptance items of #17 (deferred: need a GPU run)

1. Commit a small archive from a real GPU run of exactly
   `development_config(days=4, retained=True)` (T21L10, 32 × 64, stored `l_max` 22,
   retained 21, `spinup_days=0`, `state_every_days=1`, two 2-day blocks). It needs
   asserted layout metadata, generation instructions, the resolved config and
   source provenance, and a `statistics.npz` from the production
   `daily_sample`/`OnlineStatistics` path. Then add a CPU CI test that reduces saved
   days 1–4 and compares every corresponding accumulator statistic on the same grid.
   No such fixture exists in the repository, and none was fabricated.
2. Measure and document the GPU-transform versus host-synthesis discrepancy, to set
   that test's tolerances. Optionally add a CUDA-marked regeneration test.
3. Production comparisons at different cadences beyond the descriptive table above,
   if wanted.
4. Publish the real figure and README usage. The figure above lives in the gitignored
   `runs/` tree. Committing it (for example under a published `docs/runs/` bundle)
   is a separate publication decision.

`tropoi plot` on a Held–Suarez directory still builds an ordinary overview view, so
it refuses. The CLI route is not part of this pass.
