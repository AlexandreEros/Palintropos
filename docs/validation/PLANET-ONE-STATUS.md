# Operation Planet One: status and handoff

Updated 2026-11-24. The acceptance register is
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
| G-R original substrate gates | **PASS** (G1M, G2, G3-D, G4, G5; G1C and G5b reported). Run on commit fe74bb7^, before the heat-capacity-law refactor | Must be **rerun at the final commit**: `python tests/audit_substrate_gates.py OUT` |
| G-P Mercury columns | **RUNNING**: VPW99 TWO, G = 0.02 | `runs/planet-one/gp-vpw99-two-G0.02/` (git-ignored; `gp.json` when done) |
| G-P2 periodicity | Per run, reported in each run's `cycles.json` and `retained/status.json` | |
| G-I heat1d (L2) | **NOT YET RUN**. Plan: drive `heat1d.profile.Profile.update_T` with our absorbed-flux series; Ledlow c_p injected into heat1d; custom layers for the 2 cm step; `DTSURF` 1e-9; property cache threshold 0; joint refinement over 3 levels | |
| G-M L3 (VPW99) | **NOT YET RUN**. Needs production G = 0 BOT/TOP/TWO column runs; compare deep T at 0°N 0° with 365/463/427 K; investigate differences > 10 K; change no parameters | |
| G-G global | **NOT YET RUN** | |
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

1. Wait for, or rerun, the ladder: `PYTHONPATH=src:. python
   tools/planet_one/convergence.py runs/planet-one/gp-vpw99-two-G0.02 --model
   TWO --geothermal-flux 0.02 --workers 8`. Completed run directories are
   reused; `--tables-only` recomputes `gp.json` only.
2. If G-P passes at the base settings (4320 steps per cycle, top 1 mm,
   stretch 1.0488, depth 1 m), run the L3 reproductions with the same
   settings: `python -m tropoi.run.airless run --preset mercury-vpw99 --model
   {TWO,TOP,BOT} --geothermal-flux 0 --points 0,0 --out ...`.
3. G-I: write `tools/planet_one/heat1d_compare.py` as planned above.
4. Global: `python -m tropoi.run.airless run --preset mercury-vpw99 --grid
   90x180 --workers 8 --out runs/planet-one/global-vpw99-two`, then compare
   selected columns with standalone runs at identical coordinates (G-G).
5. Visualization: a self-contained HTML built from `retained/`
   (surface.npy: samples × [T_s, SW, absorbed, emitted, q_s] × columns;
   profiles.npy: samples × L × columns; grid.npz).
6. Rerun G-R at the final commit, regenerate the architecture snapshot, and
   update this file and the PR.
