# Held–Suarez status

Plan: docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md (r3). T0 = 2026-09-27 04:25 −03:00
(07:25 UTC), the approval commit f52ae5b. Budget is elapsed time. **Handoff point: S0–S3 done; S4 not
started** (S3 was run in a separate session per the user's decision; this file is the handoff).

| stage | state | evidence |
|---|---|---|
| S0 protocol + reference | DONE | PROTOCOL.md, REFERENCE_CONFIG.json, DEVLOG.md; HS94 text obtained (GFDL copy, sha256 in PROTOCOL); Dinosaur T42L20 float64 CPU: 112 ms/step at Δt = 600 s (≈ 16–17 s/day with diagnostics) → 1200 d ≈ 5.5 h; reference run **COMPLETE, 1200 days** (2026-09-27 09:32 local, ≈4.9 h) in `runs/hs-reference-dinosaur-T42L20-001/` (`series.npz` daily zonal means + scalars, `checkpoint.pkl` final state, `meta.json`, `run.log`; finite throughout; mean ln p_s drift 3.7e-4 over the run, i.e. the reference passes its own B2 gate; resumable driver: rerun the same command in `tools/held_suarez/dino_reference.py` docstring; scratch venv `…/scratchpad/dino-venv` of session 7b549ee2 — recreate with `pip install git+https://github.com/neuralgcm/dinosaur.git@be5409da…` if lost) |
| S1 stepper interface | DONE (commit c9817a1) | `temporal/steppers.py` (`TimeStepper`, `RK4Stepper`); tests/test_steppers.py 5 PASS; PE runner adopts it; bitwise characterization test PASS; BVE/SWE runners untouched |
| S2 batched transforms | DONE (this commit) | `PointSetSphericalHarmonics.transform_batch/inv_transform_batch`, `SpectralOperators.vector_curl_div_spectral_batch`, `PrimitiveEquationsModel(batched_transforms=True)` → `_tendency_batched`; tests/test_batched_transforms.py 5 PASS (batched vs per-level ≤ 1e-13 for transforms, ≤ 1e-12 for the full tendency; scalar path bitwise unchanged; rest state exactly zero). Local MX110: T21 L20 949 → 249 ms/tendency (3.8×), T31 L20 1582 → 962 ms |
| S3 SI stepper | DONE (this commit) | `temporal/semi_implicit.py` (`SemiImplicitOperator`, `SemiImplicitSolver`, `SemiImplicitLeapfrogStepper`), `SEMI_IMPLICIT.md`; tests/test_semi_implicit.py **21 PASS** (15 CPU incl. CI, 6 GPU). Evidence: L = fast terms from the column operators to < 1e−13 (NumPy and CuPy); reduced K×K solution satisfies the unreduced (2K+1)×(2K+1) system to 1.1e−15 / 9.5e−16 / 1.3e−15 at Δt = 300/1200/3600 s (criterion 1e−12) and matches the balanced dense solve to 1e−12; linearization residual about isothermal rest on Ω = 0: ratios ε→ε/2 ζ 4.001, δ 4.007, T 3.999, q 4.000 (O(ε²)); on Ω ≠ 0: ζ/δ residual ratios 1.998/2.003 and residual − Coriolis linearization ratios 4.001/4.007 (1.4–1.6 % of the Coriolis term); Cayley eigenvalues |λ| = 1 ± 1e−12 with phases 2 atan(ωΔt); real-model gravity waves (T21 L6, Ω = 0, δ′ = 1e−10) follow C_l to 1.6e−7 with modal amplitude drift 4.6e−11 over 8 × 1800 s; isothermal rest bit-identical through RK4 startup + SI + RAW (both time levels); **2-day T21 L10 SI vs RK4 (Δt 2400/1200/600 s, RK4 ref 300 s, RAW off): errors 1.06e−1, 2.45e−2, 5.70e−3, ratios 4.35, 4.30**; RAW recurrence verified; state dict carries both levels, resume bit-identical, refuses scheme/Δt/filter/T_ref mismatch and a missing X^{n−1}. MX110: SI T21 L10 ≈ 0.13 s/step (Δt-independent); RK4 ≈ 4× |
| S4 forcing + ∇⁸ | NOT STARTED | |
| S4b complete-scheme checks | NOT STARTED | |
| S5 experiment module | NOT STARTED | |
| S6 notebook | NOT STARTED | |

## Open issues for S4+

1. **TDR on the local GPU.** T42 L20 batched tendency raises `CUDA_ERROR_LAUNCH_TIMEOUT` on
   the MX110 (Windows display driver, ≈ 2 s kernel limit); the per-level path (4.86 s, many
   short kernels) survives. Either chunk the batched GEMMs along K when a `TDR-safe` flag is
   set, or accept that T42 batched timing is only measurable on Colab (S8 gate). Local
   development of S4–S5 must use T21 L10/L20, where batched works (S3 did; T42 was not retried).
2. The PE runner re-`initialize`s the RK4Stepper every step (count mode may clip the last
   step). The HS experiment module (S5) must drive `SemiImplicitLeapfrogStepper` continuously
   at fixed Δt (its first `step()` is the RK4 startup; its `state_dict()` holds `x_prev`,
   `x_curr`); do not reuse `run_pe` for HS.
3. S4 hooks are not yet on the stepper: the implicit ∇⁸ and Rayleigh factors go on X^{n+1}
   after `SemiImplicitSolver.advance` and before the RAW filter (place marked in
   `_leapfrog`); Newtonian relaxation is added to the explicit tendency passed in.
4. **T_ref vs HS temperatures.** T_ref = 300 K while HS T_eq reaches 315 K in the tropical
   lower troposphere; classical SI theory tolerates a modest excess, but the S4b 5-day run at
   Δt ≥ 900 s is the measurement (stop rule §6.3 if unstable after the timebox).
5. **Dealiasing cut.** L is the exact linearization only for perturbations inside the product
   cut (2/3 rule); content above it is unreachable from band-limited data and its weak-form
   curl is not linear (measured 1.4e−10 vs 2.8e−24). The S5 seed (degrees 1–8) is inside the
   cut; keep it so, and seed ζ only (δ seeds excite gravity waves the SI slows but keeps).
6. **S3 convergence ratios 4.35/4.30, not 4.00**, at Δt = 2400/1200/600 s: the SI phase error
   2 atan(ωΔt) is not asymptotic for the fastest excited modes at 2400 s. Consistent with
   second order; if the S4b convergence check (with forcing) drifts further from 4, halve Δt.
7. Regression gate at S3: `pytest tests -q` → **1052 passed, 5 skipped** (1031 + 5 pre-existing-skip in the untouched files, 13 min on the MX110 with a concurrent GPU job; plus tests/test_semi_implicit.py 21 passed in 5:55, of which the 2-day convergence run is 5:30); `git diff main --stat --
   tests/` shows additions only (208 lines in test_batched_transforms.py, test_pe_runner.py,
   test_steppers.py, plus the new test_semi_implicit.py); no pre-existing test modified. Architecture graph regenerated for the new module
   (`docs/architecture/current`, `--check` passes).
8. Reference run finished (see S0 row). Its `series.npz` holds the daily zonal means `u`, `v`,
   `T`, `TsTs`, `usus`, `vsTs`, `usvs` (shape (1200 days, 64 lat, 20 levels)), the axes `lat`,
   `sigma`, `day`, and the daily scalars `mean_lnps`, `mean_T`, `ke_mass_weighted`, `max_abs_u`;
   the driver rewrote it atomically after every 10-day chunk.

Forecast: S3 took ≈ 3 h of working time (budget 4 h). Remaining S4, S4b, S5, S6 + buffer ≈ 8 h
of budget; M2 unchanged at ~T0 + 16 h of working time.
