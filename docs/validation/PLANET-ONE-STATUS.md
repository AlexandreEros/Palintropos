# Operation Planet One: status and handoff

Updated 2026-11-25. The acceptance register is
[PLANET-ONE-GATES.md](PLANET-ONE-GATES.md), frozen before any result. This
file records what exists, what is verified, what is outstanding and how to
resume. It never changes a criterion.

## Branches and PRs

- `feature/planet-one-geometry`: unified orbit/rotation illumination
  (PR #66, against `main`).
- `feature/planet-one-thermo`: stacked on the geometry branch. It holds the
  generalized thermophysics, the airless driver and the Mercury presets.
  Draft PR.
- Worktree: `/home/user/Projects/Palintropos-planet-one`, created from tag
  `substrate-cpu-verified-v1`.
- Merging to `main` requires explicit owner approval.

## Environment

- Tests and runs use the CPU venv `/home/user/.venvs/planet-one` (numpy
  2.2.6, scipy 1.16.3, pytest, pytest-xdist), run as
  `PYTHONPATH=src:. python ...`.
- The architecture snapshot needs Python 3.12 with grimp 3.17 and
  networkx 3.7 (`/home/user/.venvs/arch312`):
  `PYTHONPATH=src:. python -m tools.architecture.generate [--check]`.
- The independent reference heat1d is pinned at
  `phayne/heat1d@5b73a6c1ba5047f59f6210185b37539e46e68136` in its own venv,
  `/home/user/.venvs/heat1d`. It is not a project dependency.
- Three CPU tests also fail at the tag, for environment reasons:
  `test_custom_examples`, `test_terrain_examples` and
  `test_terrain_height_functions`.

## Primary sources (read from the rendered pages)

- **VPW99** (Vasavada, Paige & Wood 1999, Icarus 141, 179), sec. II.A, p. 181:
  - upper layer: 2 cm thick, 1300 kg m⁻³, k_c = 9.22e-4 W m⁻¹ K⁻¹, χ = 1.48;
  - lower layer: 1800 kg m⁻³, k_c = 9.3e-3 W m⁻¹ K⁻¹, χ = 0.073;
  - conductivity law: k = k_c[1 + χ(T/350)³];
  - albedo 0.10, emissivity 0.95, Mercury internal flux 0.020 W m⁻². The
    flat-surface results neglect this flux.
  - Sec. II.B, p. 182, deep temperatures at 0°N 0°W: Model BOT 365 K, TOP
    463 K, TWO 427 K.
  - VPW99 uses a finite solar disk with limb darkening and extrapolates the
    surface temperature from the top three layers. We use a point-source Sun
    and a diagnosed Kirchhoff skin.
- **Ledlow et al. 1992** (ApJ 384, 640), sec. 5.2, p. 650, eqs. (6) and (7):
  - eq. (6), for T ≤ 350 K: a polynomial in (T − 300)/300;
  - eq. (7), for T > 350 K: 0.2029 + 0.0383[1 − exp(−(T − 350)/100)];
  - units are cal g⁻¹ K⁻¹, converted with 4184 J (thermochemical calorie;
    the paper does not say which calorie);
  - as printed, the two branches step by −2.1e-4 relative at 350 K, from the
    rounding of 0.2029;
  - eq. (6) is negative below 10.13 K.
- **Hayne et al. 2017** (arXiv:1711.00977), Appendix A and Table A1. This is
  the lunar analogue, used for numerical testing only. Its c_p fit is stated
  valid "from < 90 K to > 400 K", so Mercury daytime temperatures would be
  extrapolated.
- **Orbit and planet constants:** JPL SSD approximate elements, Table 1
  (a = 0.38709927 au, e = 0.20563593); IAU 2015 B3 nominal GM☉ and L☉; NASA
  NSSDC fact sheet (Bond albedo 0.068, used only by the Hayne analogue).

## Implementation (done)

| Capability | Where |
|---|---|
| Orbit/rotation illumination, multiple sources | `physics/insolation.py`, `rotation.py`, `orbit.py` (geometry PR) |
| Strata, k(T, z), ρ(z), heat-capacity laws (polynomial, saturating exponential, piecewise; Ledlow 1992) | `physics/materials.py` |
| Enthalpy-form Newton backward Euler, Kirchhoff skin, radiative top, geothermal bottom, sources, energy identity in every step report | `physics/substrate.py` |
| Airless experiment driver: worker processes, tiered output, periodicity monitor, guarded Aitken spin-up | `run/airless/` (`python -m tropoi.run.airless`) |
| Presets: `mercury-vpw99` (models TWO/TOP/BOT, any G) is primary; `mercury-hayne2017-analogue` | `run/airless/config.py` |
| G-P ladder tool | `tools/planet_one/convergence.py` |

## Verification status

| Gate | Status | Evidence |
|---|---|---|
| G-A geometry | **PASS** (26 tests) | `tests/test_illumination_geometry.py` |
| G-B nonlinear conduction | **PASS**: Jacobian vs FD; manufactured two-stratum transient with space order 1.99 (cells) and 2.00 (skin), time order 0.95 (cells) and 1.00 (skin); steady Kirchhoff order 1.97; zero-k limit; linear one-step limit; stress; energy closure; extents | `tests/test_substrate_nonlinear.py` |
| Ledlow law | **PASS**: printed values, 350 K step, positivity above 10.13 K, enthalpy vs quadrature, increments down to 1e-11 K including across 350 K | `tests/test_heat_capacity_laws.py` |
| G-R original substrate gates | **PASS** at commit 7090c86; `src/` is unchanged since then, so this is the final kernel: G1M, G2, G3-D, G4 and G5 pass at the original tolerances; G1C and G5b reported | `planet-one-evidence/gr-7090c86/` |
| G-P Mercury columns | **PASS** (VPW99 TWO, G = 0.02; all six columns; every ladder run PERIODIC). Uses the owner-approved amendment A1 (depth criterion at the same physical depth, 1 m) and estimator correction E1 (flux-consistent face reconstruction). The original frozen depth criterion failed (1.84–2.15 K, a physical geothermal offset) and is preserved | `planet-one-evidence/gp-vpw99-two-G0.02-{pre-amendment,amended}.json` |
| G-P2 periodicity | Per run, reported in each run's `cycles.json` and `retained/status.json` | |
| G-I heat1d (L2) | **FAIL at 0°N 0°E, PASS at 0°N 90°E (strict evaluation; owner decision pending)**. At 0°E the finest-level max \|ΔTs\| is 2.40 K at the first sunrise step, against a bound of 2.26 K, and the native maxima run 3.75/4.10/2.40 K (not monotone). At 90°E: 1.36 K against 1.84 K. RMS halves per level at both columns. An earlier PASS used coarse-time sampling chosen after the results were seen; it is withdrawn (register R1) | `planet-one-evidence/gi-heat1d-vpw99-two-G0.02.json` |
| G-M L3 (VPW99) | **EVALUATED**, G = 0, production settings, temperature at depth at 0°N 0°: BOT 367.85 vs 365 K (+2.85), TOP 470.69 vs 463 K (+7.69), TWO 425.78 vs 427 K (−1.22). All within the 10 K investigation threshold; no parameter changed | `planet-one-evidence/l3-vpw99.json` |
| G-G global | **PASS** (2°, 90×180 = 16 200 columns, CuPy FP64 on a Colab T4, commit 3d05287, 1.63 h): PERIODIC, with G-P2 met in 2 consecutive production cycles; the standalone replay of six columns is bitwise identical (0.0 K); global emitted − absorbed = 0.0202 W m⁻² = G; extents within the c_p law's domain. A supplementary 4° CPU run exists | `planet-one-evidence/global2-gpu-t4.json`; local download `runs/planet-one/global2-gpu-vpw99-two-G0.02-download/` |
| G-GPU parity (A2) | **PASS** on a T4: max \|T_GPU − T_CPU\| = 1.9e-11 K over a production cycle of the six columns (bound 1e-6 K), with no Newton-iteration differences; energy closure, G4/G5 and manufactured parity on the GPU; 78 tests pass on the T4 | `planet-one-evidence/gpu-t4-2026-11-24.json` |
| L4 observational | **NOT EVALUATED** | |

**Exploratory runs, not evidence:** coarse settings, dirty tree. BOT at
0°N 0° gave a deep T of about 368 K; hot and warm-meridian maxima were about
700 K and 568 K. These only show that the pipeline works.

## Measured costs

- Nonlinear step: about 64 µs per column at about 56 layers, with 3–4
  Newton iterations.
- 2° global grid (16 200 columns) on 8 processes: about 0.13 s per step.
- Deep relaxation at 2 m depth has an e-fold of about 21 cycles; the slowest
  mode is predicted at about 17. Guarded Aitken extrapolation in the coarse
  spin-up stage cuts this sharply. The periodic claim is still checked at
  the production Δt over ≥ 2 consecutive cycles, never across a coarse/fine
  switch.

## Known limitations

- A point-source Sun; VPW99 uses a finite disk with limb darkening.
- Idealized orbit: fixed elements, exact 3:2 rotation, zero obliquity, no
  SPICE.
- Uniform flat material, with no topography, shadowing or lateral
  conduction.
- The calorie convention is ambiguous (6.9e-4 relative effect).
- The Kepler solve is scalar Python, evaluated once per step. That is
  adequate here.
- No GPU path has been exercised for the new kernel. The kernel is
  xp-generic, but CPU/GPU parity is untested.
- Ben 2 G8, the bitwise rerun of an archived THAI run, cannot hold at
  current commits (see `docs/radiation/SHARED_PHYSICS.md`). Verification
  criteria for when atmospheric coupling resumes are to be set then.

## How to resume

1. **Pending owner decision: G-I at 0°N 0°E** (register R1). The likely
   next step, if chosen, is gathering more evidence rather than changing the
   criterion. Add a fourth joint refinement level to
   `tools/planet_one/heat1d_compare.py` (LEVELS), rerun it, and call
   `verdict()`. heat1d's CFL substeps make that level slow on the CPU, about
   hours.
2. **GPU runs on Colab:** `colab new -s NAME --gpu T4`, then the bootstrap
   and launch scripts pattern described in the memory note. Use
   `python -m tropoi.run.airless run ... --backend cupy --workers 1`. Then
   run `tools/planet_one/global_check.py RUN` and
   `tools/planet_one/build_view.py RUN OUT.html --gp/--gi/--l3` on the VM,
   download files of 19 MB or less, and `colab stop`.
3. **Drive mirror** (main account): folder `planet-one-global2-gpu-evidence`
   under the Do-Mercury folder holds the view, the run bundle and the
   profile subset (5 MiB parts with `MANIFEST-v2.txt`). The connector limit
   is about 7 MB.
4. **Before any merge:** owner review of #66 and #67. Merging needs explicit
   approval.

## Results so far (2026-11-25)

- **Interactive view:** one full solar day from the 2° GPU run, built on
  the VM from the retained cycle, published privately at
  https://claude.ai/artifact/CtuVELqWVUt4UaDqzWdL7Y. The local copy is
  `runs/planet-one/global2-gpu-vpw99-two-G0.02-download/mercury-view.html`
  (sha256 6568a84a…).
- **Gate summary:**
  - pass: G-A, G-B, G-R (at the final kernel), G-P (with A1/E1), G-M L3 (within 10 K; BOT checked on a 2 m column), G-GPU (A2) and G-G;
  - **G-I fails at 0°N 0°E** under the strict evaluation (owner decision pending);
  - L4 is NOT EVALUATED.
- **PR #66 GPU coverage:** the CUDA radiative tests pass on a T4 at 3bce170 (154 passed, 9 THAI k-table skips; `planet-one-evidence/geometry-pr66-cuda-tests-t4.json`).
- **4° CPU run:** finished PERIODIC (supplementary).
- **View:** published (version 3) and rendered in headless Chromium: no console errors, slider end to end, geometry readouts checked (subsolar point 180°E at the second perihelion, retrograde loop). It states the G-I verdict.
- **4° CPU run:** supplementary evidence only. The 2° acceptance run is the
  GPU run.
- **Remaining before review:** owner review of #66/#67; optionally rerun the
  full 2° tier-2 retention to a durable store (Drive mount needs interactive
  authentication).


## Reproducibility of the 2° run (wording corrected 2026-11-25)

Demonstrated: (1) G-G – replaying the retained cycle of six columns standalone from the saved start state reproduced the global run's tier-2 samples and end state bitwise on the same T4 and software stack; (2) G-GPU – CPU and GPU agree to 1.9e-11 K over a production cycle. Not demonstrated: that rerunning the whole experiment from scratch reproduces the discarded ~2 GB of full-resolution profiles bitwise. That is an expectation (deterministic code, fixed configuration and commit 3d05287, same hardware and software stack), not a tested result; on a different stack only agreement within the G-GPU tolerance is expected.
