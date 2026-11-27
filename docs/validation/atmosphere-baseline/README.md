# Atmospheric T4 baseline after the substrate consolidation (2026-11-27)

The complete primitive-equation step of the heated radiative runs, measured
on a Colab **Tesla T4** (driver 580.82.07, CuPy 14.0.1, 2 vCPU Xeon @
2.00 GHz) at commit `7622a74`, the consolidated `main` of PR #83. The
atmosphere's code is unchanged by the consolidation. These are measurements
and proposals; nothing here changes the model.

**Cases.**

| Case | Grid | Levels | dt | Radiation |
| --- | --- | --- | --- | --- |
| Earth gray dry, heated | T21, 32 × 64 | 25 | 1200 s | gray two-stream, every step |
| Earth gray dry, heated | T42, 64 × 128 | 25 | 600 s | gray two-stream, every step |
| THAI Ben 2 correlated-k (`dev-thai-ben2-ck`) | T21, 32 × 64 | 40 | 1200 s | 17 LW + 42 SW k-terms, every third step; boundary-layer diffusion |

**Method** (`tools/colab/atmosphere_baseline.py`, which runs the existing
`tools/colab/radiative_pilot.py` and adds the THAI case and labels for the
boundary layer and the radiative fluxes):

1. **Throughput**, uninstrumented: 72 production steps after 6 warm ones,
   synchronized only at the ends.
2. **Component times**: every labelled call is bracketed by a device
   synchronization (inclusive and exclusive milliseconds per step). This
   serializes the stream, so the components sum to more than the
   throughput step.
3. **cProfile** of the uninstrumented loop (host time per Python function).
4. **Nsight Systems** with NVTX ranges: kernels per range, and GPU time
   projected onto the ranges.

## Throughput

| Case | s/step (uninstrumented) | Synchronized step | Kernels per step | Memory pool |
| --- | --- | --- | --- | --- |
| Earth T21L25 | **0.0935** | 0.123 | 4,713 | 68 MB |
| Earth T42L25 | **0.113** | 0.107 | (not traced) | 386 MB |
| THAI Ben 2 ck T21L40 | **0.556** | 0.551 | 24,640 | 459 MB |

**Caveats.**
- Each throughput figure is one stretch of 72 steps (n = 1).
- The first case ran with a cold CuPy cache.
- The synchronized T42 step came out faster than T21's, which is
  unexplained.
- The ranking below does not depend on these figures, but they are not
  precise timings.

**The step is bound by kernel launches, not by arithmetic.**
- Earth T42 has 4 times the columns of T21 but takes only 21% longer.
- The THAI step launches 24,640 kernels, and 41,497 in each radiation call.
- The average kernel lasts about 10-12 µs, roughly a launch's own cost.
- In the THAI trace, the GPU's kernel time is roughly half of the
  uninstrumented wall time. This is an estimate: the trace's kernel total
  includes model setup and the warm steps.

## Where the THAI step goes (T21L40, synchronized component times)

| Component | ms/step | Share | What dominates (cProfile, Nsight) |
| --- | --- | --- | --- |
| Radiative transfer (`Radiation.fluxes`, one call every 3 steps, ~950 ms each) | **317** | 57% | `twostream.adding` (7.2 s host over 456 calls), `kdist._layer_k` (2.3 s), `stellar_fluxes` and `direct_layer`: Python loops over layers and k-terms, each launching small elementwise kernels |
| Boundary-layer diffusion term | **133** | 24% | 162 per-level scalar syntheses (35 ms); 3 implicit column solves (31 ms; **10.4 ms per solve**, P6 baseline 9.5 ms); vector analysis (12.6 ms); boundary-layer depth (7.8 ms); its own elementwise work (40 ms) |
| Dynamics tendency | 68 | 12% | per-level spectral operators (`sin_theta_d_theta_coeffs` 14,670 calls), `vector_curl_div` (12 ms), vertical advection |
| Momentum sink heating | 26 | 5% | elementwise |
| Implicit (semi-implicit) solve | 1.9 | <1% | |
| Convection, daily sample, checkpoint | ~4 | <1% | |

**Earth T21L25:**
- dynamics 55 ms (45%);
- gray radiation 34 ms (28%);
- momentum sink heating 21 ms (17%);
- convection 3 ms.

**CUB segmented reductions.** In both traces, CUB's segmented sum
(`DeviceSegmentedReduce`, from CuPy reductions along an axis) is the single
largest kernel by GPU time: 34% of the Earth trace and 20% of the THAI
trace, at about 1.8 ms per call. Which reductions make these calls has not
yet been identified.

## D3 equivalence study of the boundary-layer solve (stage H)

`tools/colab/bl_thomas_equivalence.py` captured 30 of the boundary layer's
implicit diffusion systems from the THAI case after 216 steps (u, v and dry
static energy; 20 momentum and 10 energy systems; 40 levels and 2,278
product-grid columns each). It then solved each system in
three ways:
- with the production elimination (`implicit_column_diffusion`);
- with a kernel that assembles the same system with the same arithmetic and
  eliminates it with the canonical `pt::thomas`;
- in extended precision (NumPy `longdouble`).

| | Production | Canonical `pt::thomas` |
| --- | --- | --- |
| Largest difference between the two (column scale) | 1.3e-16 | |
| 99th percentile of the difference | 0 | |
| Values bitwise equal to the other (all rows; see the caveat below) | 99.56% | |
| Largest distance to the rounded extended solution (column scale; see below) | 2.10e-16 | 2.10e-16 |
| Mean distance to the rounded extended solution | 9.558e-18 | 9.541e-18 |
| Largest column-integral conservation error (zero-flux systems) | 4.8e-17 | 4.7e-17 |
| One solve (T4) | **9.14 ms** | **0.20 ms** |
| The u/v pair | 18.2 ms (two calls) | **0.29 ms** (one launch; see below) |

**Caveat on the bitwise fraction.** It counts every row. Above the
boundary layer the conductances are zero, the system is diagonal, and
both eliminations return `D/B` exactly, so these rows inflate the
fraction. The fraction over rows with non-zero coupling has not been
measured; the tool now reports it for the next run. The bound that
matters is the largest difference, 1.3e-16 of the column scale.

**What the accuracy rows measured.** The recorded run (`bdd3fcd`) rounded
the extended-precision solution to float64 before subtracting, so these rows
are distances to the correctly rounded reference, not errors against the
extended solution itself. Each can differ from the true error by up to half a
unit in the last place of the value, which is the same order as the figures.
The tool now subtracts in `longdouble`; the true errors have not been
re-measured.

**What the pair timing measured.** The canonical kernel assembles the matrix
once and runs `pt::thomas` for each right-hand side in the same launch.
`pt::thomas` recomputes the elimination every time, so the 0.29 ms shares
assembly and the launch, not a factorization. Reusing the factorization is
not measured. The JSON key `canonical_uv_pair_one_factorization_s` is the
old, inaccurate name; the tool now writes `canonical_uv_pair_one_launch_s`.

The two eliminations agree to round-off. Within this metric's resolution
they are equally accurate, and they conserve equally well. The recorded
means differ by less than that resolution, so they do not show that either
elimination is more accurate. The prototype meets
P6 (≤ 1.0 ms per solve at T21L40) by a factor of five per solve, and the
u/v pair by more.

Where values differ, the largest difference is 162 units in the last
place. That happens for values far below their column's largest
magnitude, where 162 units in the last place is still 1.3e-16 of the column
scale.

## Proposed priorities for the next native migration (for the owner's decision)

1. **The radiative-transfer column solver** (two-stream adding,
   per-layer optical properties and k-term sums) as canonical portable C++:
   one thread per column and k-term, following the substrate's pattern
   (`__dmul_rn`, exact specialization, a bitwise gate against the current
   CuPy path where the operation order is kept).
   - It is 57% of the THAI step, and about 41,500 launches per call would
     become a few.
   - A first estimate is one or two orders of magnitude on that 57%, to be
     measured by a prototype, as the substrate's was.
2. **The boundary layer**, in two parts:
   - (a) **stage H**, the implicit solves through a canonical `numerics`
     tridiagonal solver, assembling the u/v matrix once (measured above;
     factorization reuse not measured). Migration needs the decisions below.
   - (b) the 162 per-level syntheses and the per-level elementwise work.
     Batching them may change rounding, so each needs its own equivalence
     check.
3. **The segmented reductions:** identify their callers. A different
   reduction layout may be a cheap win, but it changes summation order, so
   the change is not bitwise.
4. **Dynamics:** batch the per-level spectral operators, which are the
   largest host cost after radiation in both cases. Like 2(b), this needs
   an equivalence check where it reorders arithmetic.

The insolation and the substrate are not on this list: the substrate step
now takes 17-19 ms at 16,200 columns, and the forcing costs no measurable
wall-clock time.

**Owner rulings (2026-11-27):**
- Shared native numerical infrastructure is the agreed direction. Reusable
  numerical kernels have canonical ownership below the physics layer.
- The proposed ordering of the atmospheric optimizations above is agreed.
- Stage H's arithmetic-equivalence criteria must be established explicitly
  before production migration. The agreement measured here is evidence, not
  an acceptance threshold.
- Neither stage H's production migration nor further atmospheric
  optimization is started on that date.

**Decisions that were requested before stage H's migration:**
- whether the measured differences above (≤ 1.3e-16 of column scale, equal
  accuracy and conservation) are acceptable, since no
  tolerance has been agreed for atmospheric arithmetic;
- where the CPU path of a canonical `numerics` solve lives. Today the native
  CPU library belongs to the substrate (`temporal`); the options are a
  `numerics` extension of its own, or one shared native library at the
  bottom layer.

## Files

| File | Content |
| --- | --- |
| `t4-throughput-7622a74.json` | throughput, memory pool and configuration hash of each case |
| `profile-<case>-sync.{txt,json}` | synchronized component times, transforms by caller |
| `profile-<case>-cprofile-{tottime,cumulative}.txt` | host profiles |
| `nsys-<case>-nvtx-gpu-projection.csv`, `nsys-<case>-kernels.csv` | Nsight Systems: GPU time and kernels per NVTX range; kernel summary |
| `t4-bl-thomas-equivalence-bdd3fcd.json` | the D3 study, per captured system |
