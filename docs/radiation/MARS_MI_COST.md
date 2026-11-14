# Mars M-I CV-1: what it would cost, measured (2026-11-14)

**Scope.** This report measures what CV-1 would cost
(`MARS_MI_PLAN.md` §4.3, §4.4). CV-1 is a T42L25 run of at least 1920 sols
from L_s 0, whose L_s 225°–315° zonal-mean u is compared against Thomson &
Vallis (2019) Fig. 4b. The question is whether three levers make that run
affordable:

1. a low-resolution seasonal spin-up followed by a T42 continuation;
2. a longer time step;
3. optimization guided by profiling.

**Nothing here is implemented, and no long run was launched.** Every number
comes from short instrumented runs and micro-benchmarks made with
[`tools/radiation/mars_cost_profile.py`](../../tools/radiation/mars_cost_profile.py).
The tool changes no production behaviour: it wraps callables of its own
in-process model only. The raw records are in
[`evidence/mars-mi-cost-mx110/`](evidence/mars-mi-cost-mx110/). **Any time
step other than 110 s is a labelled departure from IR-10, not conformance**
(§3.4).

- Code: `cfebd16`. This is the `dev-mars-gray-flat-bl` preset of `06f5356`
  plus documentation only.
- Local GPU: NVIDIA GeForce MX110 (2 SMs, 2 GB, Windows WDDM), with an
  i7-8550U host, CuPy 13.4.0 and CUDA 11.8.
- Colab numbers: the user's committed L4 and T4 runs (§1.4).

## The answer in brief

1. **At T42 the step is spectral-transform-bound, and the transforms do much
   more arithmetic than they need to.**
   - On the MX110 a T42 step takes 8.86 ± 0.03 s. 93 % of the synced time is
     in the dense-matrix spherical-harmonic GEMMs, about 109 GFLOP per step
     (counted from the code). GPU utilization is 99 % at full clock.
   - At T21 the transforms are 66 % of the synced step. At T10 the loop is
     host-bound: 27 % GPU utilization and 24 600 Python calls per step.
   - A low resolution is therefore **not proportionally cheaper**. A T10
     step costs 0.40–0.46 s, against 0.875 s at T21.
2. **CV-1 as written (T42, 1920 sols) is out of reach locally.**
   - MX110: 29 days at dt 600 s and 161 days at Isca's 110 s, at full
     clock. Sustained load throttles the laptop GPU by about 23 %, so
     expect 36 and 198 days.
   - The MX110 cannot run T42 unmodified at all: one 608-row synthesis GEMM
     outlives the Windows 2 s watchdog (TDR, `CUDA_ERROR_LAUNCH_TIMEOUT`).
     The local T42 numbers were measured with the benchmark-only GEMM
     chunking of §1.1.
   - Estimated from the measured FLOP count and the T4 rate implied by
     Held–Suarez T42 (not measured for this preset): an L4 needs about
     1.5 days at 600 s and 8 days at 110 s; a T4 about 2.5 days at 600 s and
     13 days at 110 s.
3. **The time step can grow about 2× at T21 and T10; T42 is at the edge at
   600 s.**
   - Short probes from a developed L_s 29° state are stable up to 1200 s at
     T21 (unstable at 1440 s) and up to 2400 s at T10 (unstable at 3600 s).
   - The binding constraint is horizontal advection. The estimate
     N·|V|max·dt/a is about 1.3 at both thresholds, while the vertical
     Courant number at the threshold differs between the two truncations.
   - At T42, 600 s survived a 0.25-sol probe at that estimate's 1.1–1.26,
     just under the T10/T21 threshold. 450, 360 and 300 s also survived;
     no T42 step failed, so its limit is inferred.
   - The northern-winter jets of L_s 225°–315° are stronger than the probe
     state's 150–170 m/s. The usable T42 step for CV-1 is therefore about
     **300–450 s**, not 600 s (§3.3).
4. **A low-resolution spin-up mainly buys the slab.**
   - The global-mean slab temperature of the 60-day T21 run relaxes from its
     uniform 285 K start with τ = 34.9 sols (rms fit residual 0.11 K). The
     atmosphere's own memory is short: 1.3–5 sols radiatively, 5–22 sols for
     surface spin-down.
   - After a T(low)→T42 transfer, about 3τ ≈ 105 sols of T42 time should
     forget the slab. Spectral small scales refill within hours of a
     zero-padded transfer (measured).
   - T10 spin-up to sol ~1010, then T42 to the end of the year-2 window
     (sol 1256), is about 246 T42 sols: **7.8× less T42 time** than 1920
     sols. It is a dated protocol amendment, not a measurement shortcut
     (§4).
5. **The biggest measured lever is the transform itself (§5).**
   - The Gauss grid allows FFT plus Legendre transforms, about 100× fewer
     flops than the dense matrices.
   - Short of that, the real-output and m = 0 waste in
     `inv_transform_batch` (about 4×) and the per-level scalar transforms of
     the boundary layer are measured hotspots.
   - Combined with a 300–450 s step and the spin-up, CV-1 at T42 plausibly
     drops to about a day of Colab time. That is a projection to be
     re-measured, not a result.

## 1. Baseline cost

### 1.1 Method

`profile` builds the preset at a given truncation and runs it from its own
initial state, in a scratch directory. The cost of a step does not depend on
the flow: every array has a fixed shape and nothing iterates. Five
measurements are taken:

1. **Uninstrumented s/step.** `cupyx.profiler.benchmark(stepper.step)` runs
   3 trial blocks of 10 steps (3 × 3 at T42). The RK4 start-up step and 5
   warm-up steps (2 at T42) are excluded. The reported value is the
   CUDA-event time per step. During these steps `nvidia-smi` samples GPU
   utilization and the SM clock every 100 ms.
2. **Operator breakdown.** A hierarchical timer synchronizes the device at
   the entry and exit of each wrapped operator, so GPU work is charged to
   the operator that launched it. It records inclusive and exclusive time
   per operator and per transform primitive. The synchronizations add time:
   the synced step is 1.1–1.5× the uninstrumented one. **The breakdown is
   an attribution; the cost is the uninstrumented number.**
   - `--nvtx-steps` emits the same ranges unsynced, so Nsight Systems can
     attribute them independently. `nsys` is not installed on this machine;
     the Colab commands are in §6.
3. **I/O.** The daily sample, the daily state write and a checkpoint are
   timed with `RadiativeRun`'s own methods and amortized at the preset's
   cadence: one sample and one state per model day, one checkpoint per 10
   days.
4. **cProfile** of uninstrumented steps, for the host-side view.
5. **Micro-benchmarks** of each transform primitive, plus the floor cost of
   one CuPy launch and of one device-to-host scalar read.

Resolutions:

| truncation | `l_max` / retained | state grid | product grid |
| --- | --- | --- | --- |
| T10 | 11 / 10 | 16 × 32 | 17 × 34 |
| T15 | 16 / 15 | 24 × 48 | 25 × 49 |
| T21 | 22 / 21 (the preset) | 32 × 64 | 34 × 67 |
| T42 | 43 / 42 | 64 × 128 | 65 × 130 |

Each keeps the preset's convention of storing one degree above the retained
truncation. The T42 grid is the one `tools/colab/radiative_pilot.py` uses.

**T42 on the MX110 needs a benchmark-only change.** Unmodified, it dies in
the RK4 start-up with `CUDA_ERROR_LAUNCH_TIMEOUT`
(`profile-T42-dt600.json` was never written; the failure is recorded in the
[evidence README](evidence/mars-mi-cost-mx110/README.md)). The dynamics synthesizes 12K+4 = 304 fields in one
`inv_transform_batch`. That becomes a 608 × 990 by 990 × 8450 complex GEMM:
41 GFLOP, about 1.6 s or more at the MX110's measured rate, in one cuBLAS
kernel. This is at the Windows 2 s display-driver watchdog.

`--chunk-rows 64` splits each batched GEMM into chunks of at most 64 fields.
The arithmetic per output element is the same, and cuBLAS may pick different
kernels. The change is recorded in each JSON
(`benchmark_only_gemm_chunk_rows`). Nothing in `src/` changes.

### 1.2 MX110 results (dt 600 s)

| | T10 | T15 | T21 | T42 |
| --- | --- | --- | --- | --- |
| **s/step, uninstrumented** (mean ± std over all steps; median) | 0.463 ± 0.084; 0.474 | 0.403 ± 0.029; 0.393 | **0.875 ± 0.023**; 0.868 | **8.856 ± 0.031**; 8.856 |
| trial-block means | 0.446, 0.470, 0.472 | 0.397, 0.416, 0.396 | 0.865, 0.883, 0.878 | three blocks of 3 steps, 8.81–8.91 |
| second pass, about 2 h later (mean ± std; median) | 0.414 ± 0.175; 0.341 (clock 783 MHz) | 0.413 ± 0.045; 0.400 | 0.862 ± 0.012; 0.859 | **10.93 ± 0.46**; 10.64, **throttled**: SM clock 764 MHz mean, 731 min |
| GPU utilization (mean); SM clock | 27 %; 821 MHz | 70 %; 1006 MHz | 88 %; 1006 MHz | 99 %; 1006 MHz |
| synced step (attribution run) | 0.71 | 0.84 | 1.20 | 9.85 |
| RK4 start-up step (excluded) | 1.8 s | 1.9 s | 3.5 s | 29.4 s |
| CuPy pool in use | 18 MB | 43 MB | 96 MB | 696 MB (422 used) |
| Python calls per step (cProfile) | 24 642 | 24 642 | 24 642 | 24 836 |
| I/O amortized per step | 0.0016 s | 0.0020 s | 0.0025 s | 0.025 s |

Floors measured on this host: one CuPy elementwise launch costs 45–63 µs of
host time, and one device-to-host scalar read 108–152 µs (WDDM).

**Sustained load throttles the MX110.** The second T42 pass came after
about 1.5 h of continuous GPU work (the probes). It ran at 764 MHz instead
of 1006 MHz and took 10.6–11.7 s per step. A GEMM-bound step scales with
the clock: 8.86 × 1006 / 764 = 11.7 s. A multi-day local run would see the
throttled rate, so the full-clock 8.86 s is a lower bound for one.

**Variance.** The laptop's CPU turbo and the MX110's clock management make
T10 noisy: steps range 0.30–0.71 s, and the mean clock was 821 MHz against
1006 MHz in the other runs. A host-bound loop feels CPU turbo directly.
From T21 up the GPU sits at full clock and the spread is 1–3 %. T10 is not
cheaper than T15 here. Both are at the host floor, roughly 24 600 Python
calls and several thousand launches of about 50 µs each.

**Earlier figure.** The 0.61 s/step quoted for T21 on the MX110 was the
pilot preset, before the boundary layer. The boundary layer is now about
40 % of the synced T21 step (§1.3). That is consistent with the L4's step
going from 0.089 to 0.183 s once it was added (§1.4).

### 1.3 Where the time goes

Exclusive synced seconds per step, from the per-operator table of the
second pass (`instrumented.by_operator`). "Transforms" means the
`PointSetSphericalHarmonics` primitives the operator called; "other" is
everything else it ran.

| operator (top level of a step) | T21: transforms | T21: other | **T21: total** | T42: transforms | T42: other | **T42: total** |
| --- | --- | --- | --- | --- | --- | --- |
| boundary layer (`BoundaryLayerDiffusion`, lagged) | 0.300 | 0.171 | **0.472** (40 %) | 2.60 | 0.15 | **2.75** (27 %) |
| dynamics (`PrimitiveEquationsModel.tendency`) | 0.291 | 0.107 | **0.398** (33 %) | 4.13 | 0.09 | **4.22** (41 %) |
| frictional heating (`MomentumSinkHeating`, explicit) | 0.161 | 0.046 | **0.207** (17 %) | 2.51 | 0.04 | **2.55** (25 %) |
| column coupling (radiation, sensible exchange, slab) | 0.042 | 0.068 (radiation 0.058) | **0.109** (9 %) | 0.55 | 0.18 (radiation 0.16) | **0.73** (7 %) |
| semi-implicit solve plus damping, operator, RAW | — | 0.010 | **0.010** | — | 0.02 | **0.02** |
| **step (synced)** | 0.794 | 0.400 | **1.194** | 9.79 | 0.48 | **10.27** |

The T42 column comes from the throttled second pass (764 MHz). Its shares
are what matter. Uninstrumented, the same steps took 10.93 s. I/O is not
in a step: 0.0025 s per step at T21 and 0.025 s at T42 (below).

At T21 the boundary layer is the most expensive operator: 40 % of the step,
for one column scheme. Its per-level loop runs 26 `_wind_from` calls and 26
scalar analyses, each through the dense matrices. At T42 every operator
that synthesizes on the full grid scales with the GEMM.

The transform primitives across all operators (first pass,
`instrumented.rows`):

| primitive | T21 calls/step | T21 s/step | T42 s/step | what it is |
| --- | --- | --- | --- | --- |
| `inv_transform_batch[product]` | 2 | 0.233 | 3.244 | dynamics synthesis (304 fields → a 608-row GEMM) plus the boundary layer's T |
| `inv_transform_batch[state]` | 2 | 0.168 | 2.524 | frictional-heating synthesis (9K+1 = 226 fields) plus the coupling's T |
| `inv_transform[product]` (scalar) | 102 | 0.202 | 1.440 | the boundary layer's per-level winds (`_wind_from`, 4 synthesis mat-vecs per level) |
| `transform_batch[product_ext]` | 4 | 0.061 | 0.706 | the weak-form vector analysis (dynamics and boundary layer) |
| `transform_batch[product]` | 1 | 0.050 | 0.627 | dynamics analysis (4K+1 fields) |
| `transform[product]` (scalar) | 26 | 0.040 | 0.374 | the boundary layer's per-level T rows and KE sink; each call re-copies `Y.conj().T` |
| `transform_batch[state]` | 2 | 0.025 | 0.304 | the coupling's heating analysis |
| `inv_transform[state]` (scalar) | 5 | 0.010 | 0.069 | the coupling's ln p_s and lowest winds |
| **all transforms** | | **0.789 of 1.196** | **9.30 of 9.85** | |
| radiation (`radiation.fluxes`) | 1 | 0.058 | 0.161 | gray two-stream, every step |
| everything else | | 0.35 | 0.39 | Python glue, per-level loops, tridiagonal solves, semi-implicit solve (0.008 / 0.017) |

The micro-benchmarks give the cost of one K = 25 batch (GPU median):

| | T21 analysis | T21 synthesis | T42 analysis | T42 synthesis |
| --- | --- | --- | --- | --- |
| state grid | 11.8 ms | 20.3 ms | 144 ms | 282 ms |
| product grid | 13.0 ms | 22.7 ms | 149 ms | 295 ms |

Synthesis costs twice analysis because of the m = 0 GEMM (§5, item 2). At
T42, a K = 25 synthesis on the state grid is 3.2 GFLOP of complex GEMM in
0.28 s, about 11.5 GFLOPS. The full step's 109 GFLOP in 9.3 s is about
11.7 GFLOPS. That is roughly half the MX110's fp64 peak, so the whole T42
step runs at the fp64 GEMM rate.

**I/O and checkpointing are not a factor.** At T42 the daily sample costs
3.4 s, the state write 0.21 s and a checkpoint 0.41 s. Amortized, that is
0.025 s per 600 s step (0.3 %), and 0.005 s per step at 110 s.

### 1.4 Colab L4 and T4

These step rates come from the committed production event logs (wall time
between 10-day checkpoints, so they include the daily samples). No Colab
profile of this tool exists yet; §6 gives the commands.

| run | GPU | s/step (10-day blocks after the first) |
| --- | --- | --- |
| `dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4` (**this preset**) | L4 | **0.1826 ± 0.0012** (n = 5) |
| `dev-mars-gray-flat-bl-T21L25-60d-06f5356-l4` (unheated) | L4 | 0.1604 |
| `dev-mars-gray-flat-heated-T21L25-30d-b3aaef8-l4` (pilot, before the BL) | L4 | 0.0885 |
| `dev-mars-gray-flat-sponge-implicit-heated-T21L25-60d-63d05a2-t4` (before the BL) | T4 | 0.0967 |
| `hs-T42L20-prod-001` (Held–Suarez, `docs/held_suarez/STATUS.md`) | T4 | 0.253 |

At T21 the L4 is only 4.8× faster than the MX110 (0.183 against 0.875 s). The
L4 cannot be GEMM-bound at T21: the transforms are 8 GFLOP per step, about
0.02 s at the L4's fp64 rate. Most of its 0.18 s is the host-side floor:
launches, per-level loops and scalar reads.

**T42 on Colab: an estimate, not a measurement.** The model is
s/step ≈ F / R + H, where:

- F = 109 GFLOP, the per-step transform work counted from the code. It
  matches the MX110 timing (9.3 s at 11.7 GFLOPS).
- R is the effective fp64 GEMM rate. For the T4, the HS run implies
  R ≥ 170 GFLOPS. That rests on about 43 GFLOP of T42L20 transforms per HS
  step (counted the same way, assuming HS ran the same batched transform
  path, which was not checked) done within its 0.253 s step. The
  spec-sheet fp64 peak is 254 GFLOPS (third-party listings), so this is
  about 67 % of peak. The L4's listed fp64 peak is 473–490 GFLOPS (1:64);
  at a similar efficiency R is about 350.
- H is the non-transform host floor, about 0.15 s, the L4's T21 step minus
  its transforms.

That gives **roughly 0.45 s/step on an L4 and 0.75 s/step on a T4**, each
with an uncertainty of about 30 %. These figures need the Colab
measurement of §6 before any launch is planned on them.

## 2. CV-1 wall-clock projections

Steps = sols × 88 440 s / dt. Wall = steps × (s/step + amortized I/O). The
table comes from `project` (`projection-mx110-run1.json` and
`projection-mx110.json`); the Colab rows use the §1.4 estimates.

| GPU, T42 | s/step | dt 110 s (Isca, IR-10): 1 543 680 steps | dt 300 s: 566 016 | dt 450 s: 377 344 | dt 600 s (current): 283 008 |
| --- | --- | --- | --- | --- | --- |
| MX110 (measured, chunked GEMMs, full clock) | 8.856 | 3 857 h = **161 d** | 1 400 h = 58 d | 932 h = 39 d | 698 h = **29 d** |
| MX110 (measured, throttled second pass) | 10.93 | 4 744 h = 198 d | 1 726 h = 72 d | 1 149 h = 48 d | 861 h = 36 d |
| L4 (estimate) | ~0.45 | ~193 h = **8 d** | ~71 h = 3 d | ~47 h | ~35 h = **1.5 d** |
| T4 (estimate) | ~0.75 | ~322 h = **13 d** | ~118 h = 5 d | ~79 h | ~59 h = **2.5 d** |

For comparison, all 1920 sols at **T21** take 69 h on the MX110 and 14.4 h on
the L4 (0.183 s/step measured) at dt 600 s, or 7.2 h on the L4 at 1200 s.

**Timing facts that shape the protocol** (computed by the tool from the
preset's orbit):

- the year is 669.0 sols;
- the L_s 225°–315° window spans sols 445.5–587.2 of each year (141.7 sols);
- the year-3 window would close at sol 1925.2, after a 1920-sol run ends.
  So "the last complete year" of CV-1 is **year 2** (sols 1114.5–1256.2),
  and sols 1256–1920 (34 % of the run) feed no window. Year 3's window
  needs 1926 sols.

A Colab session is limited to hours, not days. Every option above therefore
runs as many checkpointed segments. Exact resume is already a tested NI-6
property.

## 3. Time-step stability

### 3.1 Probe design

`probe` starts each probe from the same developed state: day 60 of the
heated T21 BL run (`checkpoint-s000008640.npz`, L4, `06f5356`), at
L_s ≈ 29°, with max |u| 143–150 m/s and speeds up to 170 m/s. It
re-truncates or zero-pads that state to the probe's truncation (§4.1) and
restarts the leapfrog with the stepper's own RK4 start-up, at the
checkpoint's time t = 60 × 86 400 s.

At fixed intervals each probe records:

- finiteness;
- grid max |u| and speed;
- T extremes;
- the share of vorticity variance at degrees ≥ 0.8 N;
- the horizontal estimate C_h = N·|V|max·dt/a;
- the vertical Courant number C_v = max |σ̇|·dt/Δσ (interior interfaces,
  the thinner layer on either side).

A probe is "unstable" if the state turns non-finite, the speed passes
500 m/s, or T leaves Isca's `valid_range_T` [0, 700] K.

Probes run for 2 sols at T21 and T10. At T42 they run for 0.25 sol (37–74
steps), because each step costs 11 s on the MX110 with the checks.

### 3.2 Results

| truncation | dt (s) | result | steps | C_h (start → max) | C_v |
| --- | --- | --- | --- | --- | --- |
| T10 | 1200 | stable, 2 sols | 147 | 0.48 | 0.17 |
| T10 | 1800 | stable, 2 sols | 98 | 0.72 | 0.26 |
| T10 | **2400** | **stable, 2 sols** | 74 | 0.96 | 0.34 |
| T10 | 3600 | **unstable** at step 16 (0.65 sol) | 16/49 | 1.44 → 1.8 before the runaway | 0.51 |
| T21 | 600 | stable, 2 sols | 295 | 0.55–0.62 | 0.31 |
| T21 | 900 | stable, 2 sols | 197 | 0.83–0.94 | 0.47 |
| T21 | **1200** | **stable, 2 sols** | 147 | 1.11–1.26 | 0.64 |
| T21 | 1440 | **unstable** at step 70 (1.14 sols) | 70/123 | 1.33 → 1.85 before the runaway | 0.76 |
| T21 | 1800 | **unstable** at step 16 (0.33 sol) | 16/98 | 1.66 → 2.0 | 0.95 |
| T42 | 600 | stable, 0.25 sol | 37 | 1.08–1.26 | 0.36–0.75 |
| T42 | 450 | stable, 0.25 sol | 49 | 0.79–0.95 | 0.27–0.59 |
| T42 | 360 | stable, 0.25 sol | 61 | 0.63–0.75 | 0.22–0.47 |
| T42 | 300 | stable, 0.25 sol | 74 | 0.52–0.63 | 0.18–0.39 |

No T42 step failed. Its threshold is inferred from the T10/T21 scaling, not
observed. A T42 probe at 720–900 s would locate it directly, at about 11 s
per step on the MX110, or on Colab.

A failure follows the same pattern each time. Growth starts where the jet
is fastest: speed goes from 144 to 183 m/s in 15 steps at T21 / 1440 s.
Then the high-degree share jumps (0.24 → 0.77) and T goes negative within
a few steps.

### 3.3 What binds

**Horizontal advection binds.** At both measured thresholds C_h lies near
1.3: T10 is stable at 0.96 and unstable at 1.44; T21 is stable at up to
1.26 and unstable from 1.33. That is a limit scaling as a/(N·V), as the
leapfrog advective limit of a spectral model should.

C_h is a conservative estimate. The maximum speed and the highest degrees
are not co-located, so the true threshold of the scheme is 1 when C_h is
about 1.3. The vertical Courant number at the threshold differs between the
truncations (T10 0.34–0.51, T21 0.64–0.76), so it does not bind.

The other candidates are implicit in this code, so the probes cannot make
them bind:

- the stress and diffusion of the boundary layer, whose stiffness
  k·w/C_a ≈ 8 enters a backward-Euler column solve;
- the sponge and the ∇⁸ hyperdiffusion (centred implicit dampers);
- the sensible exchange (§2d backward Euler).

The gray-radiation relaxation, which is lagged and explicit, is 1.3–5 sols
(§4.3), far from any step considered. Gravity waves are semi-implicit about
T_ref 300 K, which is above every Mars temperature.

**The time step usable for CV-1 at T42.** Apply C_h ≲ 1.3, measured at T10
and T21, to T42 and to the winds CV-1 will meet:

- dt ≲ 1.3 a / (42 V) gives 700 s at V = 150 m/s, 526 s at 200 m/s and
  420 s at 250 m/s.
- The probe state is near L_s 29°. Fig. 4b's northern-winter jet has a
  zonal mean above the 140 m/s cap, so its grid maximum is higher; this
  report does not know by how much.
- With a 20 % margin, the step to plan on is **300–450 s at T42**: 2.7–4.1×
  Isca's 110 s, and 1.3–2× below the current 600 s.
- The T42 / 600 s survival above is short and in a weak-jet season. It does
  not make 600 s safe for CV-1.
- The same scaling gives about 1500–2000 s for T10 and 800–1000 s for T21 in
  the winter season.

Two-sol and 0.25-sol probes cannot exclude slow instabilities. The 1440 s
T21 failure needed 1.14 sols to develop. A chosen step must be confirmed by
the first CV-1 segment's own NI-1 checks. Before T42 is committed to a
larger step, one segment that crosses L_s 270° is the real test.

### 3.4 What any dt ≠ 110 s means for IR-10

IR-10 sets Δt 110 s. Any other step, in a run called M-I, is a **labelled
departure**, listed under §4.5 with its argument. It is never conformance.
The stability probes show that larger steps run. They say nothing about
whether the climate is unchanged. Three effects of the step are known:

- the stiffness of the lagged column terms scales with it;
- the diurnal sampling of the instantaneous zenith angle is coarser;
- the RAW filter's damping per unit time changes.

Measuring the step's effect on the CV-1 observable needs a pair of runs at
two steps, which is beyond this report.

There is also a bookkeeping gap. `RadiativeConfig` requires dt to divide
86 400 s, so **110 s cannot be configured**: 86 400 / 110 = 785.45. It
divides the sol: 88 440 / 110 = 804 exactly, which is why Isca counts sols.
IR-10 conformance therefore needs the day bookkeeping to move to sols, or
to accept steps that divide the sol (§4.5, "time unit"). The nearest
configurable steps are 108 s (800 per day) and 112.5 s (768 per day).

## 4. Low-resolution spin-up

### 4.1 What a T(low) → T42 transfer needs

All of this was exercised in-process by `probe` (`transfer_state`).

- **The spectral state** is (3K+1) × (l_max+1)² coefficients on an
  orthonormal basis, so the same (l, m) value is the same function at any
  l_max.
  - Up (T21 → T42): zero-pad. Exact; nothing is dropped
    (`dropped_norm_fraction` 0).
  - Down (T21 → T10): truncate at the destination's retained degree. This
    dropped 1.17 % of the state norm.
  - The source carries nothing above its own retained cut (measured 0), so
    no aliased content travels.
- **The leapfrog's second level** (x_prev, RAW-filtered at the old step) is
  not transferable. Restart from x_curr alone with the stepper's RK4
  start-up (one substep at these steps, two at 1800 s and above). This also
  makes the step free to change at the transfer.
- **The slab T_s on the new grid.** Analyse it on the source Gauss grid at
  the source l_max, then synthesize on the destination grid.
  - Round-trip error on the source grid: 0.04 K.
  - T21 → T42: the destination range [193.41, 230.55] K stays inside the
    source range [193.58, 230.57] K, so there is no overshoot.
  - T21 → T10: 0.7 K of smoothing at the minimum.
  - Bilinear interpolation would be the conservative alternative if a
    sharper T_s field ever overshoots.
- **The sigma grid is unchanged** (25 levels, Isca `bk`), so no vertical
  remapping is needed.
- **The time.** The stepper must start at the source's t, which drives
  insolation, L_s and the diurnal phase.
- **Accumulators start fresh.** That covers the coupling budget, the
  boundary layer's KE sink, the series and the means. The first day's
  energy rate has no predecessor.

**What does not exist yet** (needed before a hybrid run; not implemented
here, by design):

- an initial-state kind that loads a transferred state and records its
  provenance: source run, checkpoint SHA-256, method, truncations;
- day bookkeeping that is not "step × dt from 0". `RadiativeRun` derives
  days from the step count, so a transferred run would number its days
  from the transfer while L_s follows t.

### 4.2 The transfer shock, measured

After zero-padding T21 → T42 the degrees above 21 start empty. The
high-degree share of vorticity variance then climbs: 0 → 0.014 (4 steps),
0.11 (12 steps), 0.20 (20 steps), 0.29 (37 steps = 0.26 sol). By then it
is at the T21 run's level of 0.23–0.26.

The small scales therefore refill within hours. The large-scale flow and
temperature carry over as they were (max |u| 152 → 161 m/s over the
probe, comparable to the T21 control). The adjustment that takes time is
the one of the T42 climate itself: jets, eddy fluxes and their slab
response. That sets the "forget" time of §4.4.

### 4.3 The memory of the M-I system

`timescales` → `timescales.json`:

| process | timescale | basis |
| --- | --- | --- |
| slab, C/(4σT_s³ + h), at the day-60 T_s (193.6–230.6 K) | 34–57 sols (h = 0); 28–44 sols (h = 0.5 W m⁻² K⁻¹) | C = 8.26 × 10⁶ J m⁻² K⁻¹ |
| slab at 285 / 240 / 210 / 180 / 150 K | 18 / 30 / 45 / 71 / 121 sols | the same formula |
| **global-mean T_s, fitted, days 10–60** | **τ = 34.9 sols**; asymptote 209.1 K; amplitude 72.8 K; rms 0.11 K | exponential fit to the L4 run's `series.json` (the forcing changes over L_s 0–29°, so the asymptote is not an equilibrium) |
| gray longwave, optically thin, 8σT³(τ₀/p₀)g/c_p | 1.3 (240 K) – 5.1 (150 K) sols | τ = 0.2 (p/610 Pa) |
| surface spin-down, H/(C_D V), z₁ 6.2 m, C_D 1.08 × 10⁻³ | 22 / 11 / 5 sols at 5 / 10 / 20 m s⁻¹ | neutral drag at Isca's roughness |

So the slab is the system's memory, about a month. There is no deep soil,
no CO₂ cycle and no dust, so nothing slower exists in M-I.

The polar-night slab relaxes slowly (121 sols at 150 K), but it is forced
toward a cold state rather than remembering its start. What a transfer
leaves behind is the gap between the low-resolution and the T42 slab
states. That gap decays at the local rate.

The 60-day run shows how far from equilibrium a uniform 285 K start is:
mean T_s falls 285.0 → 264.5 → 250.6 → 240.6 → 233.0 → 227.1 → 222.5 K
at 10-day intervals, and max |u| is still growing at day 60
(116 → 137 → 143 m/s). With τ = 34.9 sols, the 72.8 K initial-condition
error falls below 1 K after τ·ln 72.8 ≈ 150 sols and below 0.1 K after
≈ 230 sols.

### 4.4 How much T42 time after a transfer

- **The physics bound.** The T42 continuation must outlast the slab's
  adjustment to the T42 climate. With τ ≈ 35 sols, **3τ ≈ 105 sols**
  removes 95 % of any slab difference left by the spin-up; 2τ (70 sols)
  removes 86 %. The atmosphere follows within its 1–20-sol memory.
- **The unknown.** How large that slab difference is has not been measured.
  The T10 jets are broader and weaker: the ∇⁸ hyperdiffusion's 0.1-day
  e-folding sits at degree 10. A cheap test is offered in §5.
- **The arithmetic, for the year-2 window** (sols 1114.5–1256.2):

| plan | T(low) sols | T42 sols | T42 share of 1920 |
| --- | --- | --- | --- |
| CV-1 as written | 0 | 1920 | 100 % |
| T42 from L_s 0 to the end of the year-2 window | 0 | 1256 | 65 % |
| T10 to sol 1010, then T42 to sol 1256 (105-sol forget plus the 141.7-sol window) | 1010 | 246 | **12.8 % (7.8×)** |
| T10 to sol 340, then T42 for the year-1 window (sols 445.5–587.2) | 340 | 247 | 12.9 %. T42 starts 3τ before the window; the 285 K start is 9.7τ old by then |

- **Spin-up cost.** T10 for 1010 sols at a 1800 s step (inside the
  measured 2400 s limit, with a winter margin) is 49 600 steps. That is
  about 5.5 h on the MX110 at 0.40 s/step, and much less on an L4.
- **Protocol status.** Every row below the first changes CV-1's protocol
  item 1 ("T42L25 … 1920 sols from L_s 0"). That item has origin D (ours),
  not P: the paper states no integration length or spin-up (§4.4). A
  hybrid would be a **dated amendment**, made before the run. It costs the
  ability to report years 1–3 separately at T42 (protocol item 2), unless
  the T42 continuation covers more than one window.

## 5. Optimization candidates, ranked by measured hotspot

None of these is implemented. "Gain" is against the uninstrumented T42 step
on the MX110 (8.86 s) unless stated.

Every candidate that changes arithmetic must be **opt-in**. The scalar
transform paths are frozen because pinned run hashes depend on their exact
arithmetic (`fast_geodesic_sh.py`). An opt-in must also be validated against
the dense path to round-off (≤ 10⁻¹² relative), as `batched_transforms` was.

1. **Replace the dense-matrix transform with FFT plus Legendre on the Gauss
   grid** (transforms: 9.3 of 9.85 s synced at T42; 0.79 of 1.20 at T21).
   - Per field, the dense method is n_points × n_basis complex multiply-adds
     (8.4 M at T42). FFT plus Legendre is about nlat·(nlon log nlon) +
     nlat·n_basis/2 (about 0.1 M). That is about 100× fewer flops, and the
     memory traffic of the Y matrices (134–140 MB each at T42) disappears.
   - Expected: T42 transforms to a few tenths of a second on the MX110, and
     the step to about 0.6–1 s, bound by the non-transform remainder
     (0.55 s synced). On an L4 the step would fall to its host floor,
     perhaps about 0.1–0.2 s. It would also remove the T42 TDR failure.
   - Risk: high. It is a new transform that must agree with the frozen
     dense path to round-off. The Legendre recursion needs care for
     stability at high degree, though T42 is easy. The product-grid and
     extended-degree vector analysis need the same treatment.
2. **Remove the arithmetic waste in the batched transforms** (part of the
   same 9.3 s).
   - `inv_transform_batch` computes a complex GEMM and keeps only the real
     part (2×). It adds a second full GEMM for the m = 0 correction (2×),
     when scaling coefficients by (2 − δ_m0) gives the same field from one
     product. Real-output synthesis is then two real GEMMs on the
     precomputed Re Y and Im Y: **4× fewer flops**.
   - `transform_batch` promotes the real weighted field to complex: 2×.
   - Expected at T42: synthesis about 5.8 → 1.5 s, analysis about 1.6 →
     0.8 s, the step about 8.9 → 3.5–4 s. Small at T21, which is not
     GEMM-bound on an L4.
   - Risk: low to medium. The change is a few lines and opt-in, but the
     rounding differs from the dense path (round-off, not bitwise).
3. **Batch the boundary layer's per-level transforms.**
   - At T21 the boundary layer is 0.47 s of 1.20 synced: the largest single
     operator. 0.25 s of that is 26 per-level `_wind_from` calls doing 102
     scalar synthesis mat-vecs; 0.04 s is 26 scalar analyses, each
     re-copying `Y.conj().T`.
   - At T42 these scalar paths are 1.8 s. The dynamics already has the
     batched equivalents (`_stack_derivs` plus one `inv_transform_batch`).
   - Expected: about −0.25 s at T21 (about 20–25 %), about −1.3 s at T42,
     and about 130 fewer launches per step on every GPU.
   - Risk: low. It reuses tested batched paths; the boundary layer's
     results change at round-off only, behind the boundary layer's own
     opt-in.
4. **Reuse the dynamics' wind synthesis in the frictional heating.**
   - `MomentumSinkHeating` re-synthesizes the state's winds on the state
     grid (226 fields; 2.5 s at T42 with the coupling's share; 0.21 s
     inclusive at T21), alongside its sink.
   - Expected: about half of that.
   - Risk: medium. The heating is part of the energy closure that NI-7
     audits, so it needs a test that the dissipation is unchanged to
     round-off.
5. **Host floor: launches, per-level Python loops, scalar reads** (T10/T15
   at 27–70 % GPU utilization; the L4's T21 step).
   - Per step: 24 600 Python calls; 353 `d_lambda_coeffs` and 253
     `sin_theta_d_theta_coeffs` calls; 100 `_adjoint_ext` calls; 75
     `_truncate` calls; at least 10 `float()` device reads in the coupling
     and the boundary layer budgets.
   - Measured floors: about 50 µs per launch and about 130 µs per read
     here; smaller on Linux Colab hosts.
   - Remedies: vectorize the per-level loops across levels; accumulate the
     budget on the device and read it once per day; and later, CUDA Graphs
     over a whole step (fixed shapes make this possible).
   - Expected: at least 2× at T10 and T15, which is what makes a
     low-resolution spin-up as cheap as its flop count suggests, and a
     large share of the L4's 0.18 s at T21.
   - Risk: low per change, but many changes; each needs the bitwise-resume
     test (NI-6).
6. **The time step** (§3): at T42, 300–450 s against 110 s is 2.7–4.1×.
   - Risk: an IR-10 departure, labelled. The climate effect is untested.
7. **Radiation** (0.06 s at T21, 0.16 s at T42; 1.6–5 %). `radiation_interval`
   > 1 would save most of it, but it changes the physics against M-I, which
   calls radiation every step. Not recommended.
8. **I/O and checkpoints**: 0.3 % at T42 / 600 s. Nothing to gain.

**Not recommended for CV-1: fp32 transforms.** The GeForce, T4 and L4 run
fp32 32–64× faster than fp64. But spectral coefficients and the energy
tripwires (NI-7 at 10⁻³ F) are fp64-level quantities, and an fp32 GEMM
carries about 10⁻⁷ relative error per transform. It would need its own
validation campaign.

**What the levers multiply to (projection, to be re-measured).** Apply
items 1–3 at an L4 floor of about 0.15–0.25 s/step, a T42 step of 300 s and
the T10 spin-up's 246 T42 sols (§4.4): 246 × 88 440 / 300 ≈ 72 500 steps,
or **about 3–5 h of T42 on an L4**, plus about 1–2 h of T10 spin-up. That is
against about 8 days for CV-1 as written at 110 s. Each factor is separately
measurable, and none should be planned on before it is.

## 6. Reproducing these numbers

From a checkout of this branch, with the project venv and `PYTHONPATH=src`:

```bash
python tools/radiation/mars_cost_profile.py profile --truncation T21 --dt 600 --out profile-T21-dt600.json
```

```bash
python tools/radiation/mars_cost_profile.py profile --truncation T42 --dt 600 --steps 3 --trials 3 --warmup 2 --cprofile-steps 2 --io-repeat 2 --micro-repeat 3 --chunk-rows 64 --out profile-T42-dt600-chunk64.json
```

```bash
python tools/radiation/mars_cost_profile.py probe --truncation T21 --dt 1200 --sols 2 --check-every 6 --source <run>/checkpoints/checkpoint-s000008640.npz --source-truncation T21 --out probe-T21-dt1200.json
```

```bash
python tools/radiation/mars_cost_profile.py timescales --checkpoint <run>/checkpoints/checkpoint-s000008640.npz --series docs/radiation/evidence/dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/series.json --out timescales.json
```

```bash
python tools/radiation/mars_cost_profile.py project profile-*.json --dt 110 300 450 600 --out projection.json
```

`--chunk-rows` is needed only on a GPU that drives a Windows display (TDR).
**On Colab L4/T4, run without it**, for T10, T21 and T42. Add
`--nvtx-steps 5` and wrap the command in
`nsys profile --trace=cuda,nvtx -o mars-T42 …` to get an independent
attribution from Nsight Systems. Three gaps remain:

- **the Colab profiles** of this preset at T21 and T42, which replace the
  §1.4 estimate;
- **an `nsys` trace**, which checks the synced attribution;
- a T42 probe at 450 s over more than 1 sol.

The probe's source checkpoint is the day-60 checkpoint of
`dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4`. It is local
(`docs/runs/…`, untracked) and on the Drive folder recorded in the plan's
§2e. Its SHA-256 is recorded in the checkpoint's own metadata.

## 7. Limits of this report

- The T42 numbers on the MX110 use benchmark-only GEMM chunking (§1.1).
  Unmodified T42 does not run there.
- The Colab T42 figures are estimates (§1.4), and the L4 and T4 fp64 peaks
  come from third-party spec listings.
- Stability probes are short (2 sols; 0.25 sol at T42) and start from one
  state at L_s 29°. They bound the step from above, for that state.
- The attribution is synced, so overlap between operators is not
  represented; the uninstrumented totals are the cost.
- The forget time of §4.4 rests on the slab's measured relaxation, not on a
  hybrid run. The size of the T10/T42 slab difference is unmeasured.
- Whether any departure (step, hybrid spin-up) changes the CV-1 observable
  is a climate question, outside a cost report.
