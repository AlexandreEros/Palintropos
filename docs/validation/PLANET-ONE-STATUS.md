# Operation Planet One: status and handoff

Updated 2026-11-25. The acceptance register is
[PLANET-ONE-GATES.md](PLANET-ONE-GATES.md), frozen before any result. This
file records what exists, what is verified, what is outstanding and how to
resume. It never changes a criterion.

## Branches and PRs

- [PR #66](https://github.com/AlexandreEros/Palintropos/pull/66)
  (`feature/planet-one-geometry`) contains the unified orbit/rotation
  illumination. PR #67 incorporates its review fixes.
- [PR #67](https://github.com/AlexandreEros/Palintropos/pull/67)
  (`feature/planet-one-thermo`) contains generalized thermophysics, the
  airless driver and the Mercury presets.
- The original validation worktree was
  `/home/user/Projects/Palintropos-planet-one`, created from tag
  `substrate-cpu-verified-v1`.

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
| G-A geometry | **PASS** (original G-A tests plus seven review regression cases) | `tests/test_illumination_geometry.py` |
| G-B nonlinear conduction | **PASS**: Jacobian vs FD; manufactured two-stratum transient with space order 1.99 (cells) and 2.00 (skin), time order 0.95 (cells) and 1.00 (skin); steady Kirchhoff order 1.97; zero-k limit; linear one-step limit; stress; energy closure; extents | `tests/test_substrate_nonlinear.py` |
| Ledlow law | **PASS**: printed values, 350 K step, positivity above 10.13 K, enthalpy vs quadrature, increments down to 1e-11 K including across 350 K | `tests/test_heat_capacity_laws.py` |
| G-R original substrate gates | **PASS** at commit `7090c86`: G1M, G2, G3-D, G4 and G5 pass at the original tolerances. G1C and G5b were also reported | `planet-one-evidence/gr-7090c86/` |
| G-P Mercury columns | **PASS** (VPW99 TWO, G = 0.02; all six columns; every ladder run PERIODIC). Uses the owner-approved amendment A1 (depth criterion at the same physical depth, 1 m) and estimator correction E1 (flux-consistent face reconstruction). The original frozen depth criterion failed (1.84–2.15 K, a physical geothermal offset) and is preserved | `planet-one-evidence/gp-vpw99-two-G0.02-{pre-amendment,amended}.json` |
| G-P2 periodicity | Per run, reported in each run's `cycles.json` and `retained/status.json` | |
| G-I heat1d (L2) | **Satisfied by the owner-approved four-level evidence (R2)**: 0°E 1.33 K against a 2.03 K bound, 90°E 0.69 K against 1.11 K; RMS and maxima decrease at every level after L1. R1's three-level strict FAIL (0°E 2.40 K against 2.26 K) remains recorded. Sunrise attribution: our backward-Euler time error (0.58 K at L3 dt) and heat1d's lag (0.55 K) and spatial error (0.54 K); refined codes agree to 0.15–0.19 K | `planet-one-evidence/gi-heat1d-vpw99-two-G0.02.json`, `gi-sunrise-attribution.json` |
| G-M L3 (VPW99) | **EVALUATED**, G = 0, production settings, temperature at depth at 0°N 0°: BOT 367.85 vs 365 K (+2.85), TOP 470.69 vs 463 K (+7.69), TWO 425.78 vs 427 K (−1.22). All within the 10 K investigation threshold; no parameter changed | `planet-one-evidence/l3-vpw99.json` |
| G-G global | **PASS** (2°, 90×180 = 16 200 columns, CuPy FP64 on a Colab T4, commit 3d05287, 1.63 h): PERIODIC, with G-P2 met in 2 consecutive production cycles; the standalone replay of six columns is bitwise identical (0.0 K); global emitted − absorbed = 0.0202 W m⁻² = G. Archived material-range coverage is described below. A supplementary 4° CPU run exists | `planet-one-evidence/global2-gpu-t4.json`; local download `runs/planet-one/global2-gpu-vpw99-two-G0.02-download/` |
| G-GPU parity (A2) | **PASS** on a T4: max \|T_GPU − T_CPU\| = 1.9e-11 K over a production cycle of the six columns (bound 1e-6 K), with no Newton-iteration differences; energy closure, G4/G5 and manufactured parity on the GPU; 78 tests pass on the T4 | `planet-one-evidence/gpu-t4-2026-11-24.json` |
| L4 observational | **NOT EVALUATED** | |

**Exploratory runs, not evidence:** coarse settings, dirty tree. BOT at
0°N 0° gave a deep T of about 368 K; hot and warm-meridian maxima were about
700 K and 568 K. These only show that the pipeline works.

## Effect of the #67 review corrections on archived evidence

- **G-I:** All eight archived heat1d column/level records report a nonzero
  measured last-cycle difference of 0.00623–0.00958 K. Each is below the
  frozen 0.01 K limit. The old cycle-limit self-comparison produced exactly
  zero, so it does not account for these records. The R1 and R2 verdicts stand.
- **Grid and retained replay:** The archived VPW99 ladder and production
  grid settings produce identical cell faces with stretch enforcement.
  The 2° replay used increasing column indices, so the retained-column
  ordering defect does not affect its comparison.
- **Material temperature range:** The archived 2° run’s `extent_K` covers
  only the final state. Its sampled surface range was 62.44–699.96 K.
  Those samples and the final-state extents are within the Ledlow law’s
  positive domain. The archive does not establish the complete cycle’s
  per-stratum extrema. This limits the material-range claim. The separate
  G-G replay, energy closure, and periodicity evidence still stand.
  New runs accumulate temperatures from the cycle start through every
  step, including the skin. The rebuilt viewer labels legacy extents
  as final-state values and flags unverified full-cycle coverage.
- **Viewer:** New exports render run and evidence text literally. The
  published artifact and checksums remain historical records.

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
- The archived CPU/GPU parity result covers the six production columns
  for one cycle on a T4. The archived geometry GPU tests cover flat Mars.
- Ben 2 G8, the bitwise rerun of an archived THAI run, cannot hold at
  current commits (see `docs/radiation/SHARED_PHYSICS.md`). Verification
  criteria for when atmospheric coupling resumes are to be set then.

## How to resume

1. **GPU runs on Colab:** `colab new -s NAME --gpu T4`, then the bootstrap
   and launch scripts pattern described in the memory note. Use
   `python -m tropoi.run.airless run ... --backend cupy --workers 1`. Then
   run `tools/planet_one/global_check.py RUN` and
   `tools/planet_one/build_view.py RUN OUT.html --gp/--gi/--l3` on the VM,
   download files of 19 MB or less, and `colab stop`.
2. **Drive mirror** (main account): folder `planet-one-global2-gpu-evidence`
   under the Do-Mercury folder holds the view, the run bundle and the
   profile subset (5 MiB parts with `MANIFEST-v2.txt`). The connector limit
   is about 7 MB.

## Results so far (2026-11-25)

- **Interactive view:** one full solar day from the 2° GPU run, built on
  the VM from the retained cycle, published privately at
  https://claude.ai/artifact/CtuVELqWVUt4UaDqzWdL7Y. The local copy is
  `runs/planet-one/global2-gpu-vpw99-two-G0.02-download/mercury-view.html`
  (sha256 6568a84a…).
- **Gate summary:**
  - pass: G-A, G-B, G-R (at `7090c86`), G-P (with A1/E1), G-M L3 (within 10 K; BOT checked on a 2 m column), G-GPU (A2) and G-G;
  - G-I is satisfied by the four-level evidence (R2); R1's three-level FAIL is kept on record;
  - L4 is NOT EVALUATED.
- **PR #66 GPU coverage:** the CUDA radiative tests pass on a T4 at 3bce170 (154 passed, 9 THAI k-table skips; `planet-one-evidence/geometry-pr66-cuda-tests-t4.json`).
- **4° CPU run:** finished PERIODIC (supplementary).
- **View:** published (version 3) and rendered in headless Chromium: no console errors, slider end to end, geometry readouts checked (subsolar point 180°E at the second perihelion, retrograde loop). It displays the verdict from the supplied G-I evidence.
- The archived numerical results apply to their recorded commits and
  run configurations. The #66 review regressions verify the illumination
  corrections separately. The full 2° tier-2 profiles were discarded.


## Reproducibility of the 2° run (wording corrected 2026-11-25)

Demonstrated: (1) G-G – replaying the retained cycle of six columns standalone from the saved start state reproduced the global run's tier-2 samples and end state bitwise on the same T4 and software stack; (2) G-GPU – CPU and GPU agree to 1.9e-11 K over a production cycle. Not demonstrated: that rerunning the whole experiment from scratch reproduces the discarded ~2 GB of full-resolution profiles bitwise. That is an expectation (deterministic code, fixed configuration and commit 3d05287, same hardware and software stack), not a tested result; on a different stack only agreement within the G-GPU tolerance is expected.

- **Known limitation (from R2):** at the production timestep the surface temperature in the first hours after sunrise has a backward-Euler error of about 1 K. A higher-order or adaptive time integrator would reduce it.
