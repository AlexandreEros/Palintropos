# Held–Suarez status

Plan: docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md (r3). T0 = 2026-09-27 ~04:00 UTC
(approval). Budget is elapsed time. **Handoff point: S0–S2 done; S3 not started** (user
decision: S3–S6 to be run under a different model/session).

| stage | state | evidence |
|---|---|---|
| S0 protocol + reference | DONE | PROTOCOL.md, REFERENCE_CONFIG.json, DEVLOG.md; HS94 text obtained (GFDL copy, sha256 in PROTOCOL); Dinosaur T42L20 float64 CPU: 112 ms/step at Δt = 600 s (≈ 16–17 s/day with diagnostics) → 1200 d ≈ 5.5 h; reference run **COMPLETE, 1200 days** (2026-09-27 09:32 local, ≈4.9 h) in `runs/hs-reference-dinosaur-T42L20-001/` (`series.npz` daily zonal means + scalars, `checkpoint.pkl` final state, `meta.json`, `run.log`; finite throughout; mean ln p_s drift 3.7e-4 over the run, i.e. the reference passes its own B2 gate; resumable driver: rerun the same command in `tools/held_suarez/dino_reference.py` docstring; scratch venv `…/scratchpad/dino-venv` of session 7b549ee2 — recreate with `pip install git+https://github.com/neuralgcm/dinosaur.git@be5409da…` if lost) |
| S1 stepper interface | DONE (commit c9817a1) | `temporal/steppers.py` (`TimeStepper`, `RK4Stepper`); tests/test_steppers.py 5 PASS; PE runner adopts it; bitwise characterization test PASS; BVE/SWE runners untouched |
| S2 batched transforms | DONE (this commit) | `PointSetSphericalHarmonics.transform_batch/inv_transform_batch`, `SpectralOperators.vector_curl_div_spectral_batch`, `PrimitiveEquationsModel(batched_transforms=True)` → `_tendency_batched`; tests/test_batched_transforms.py 5 PASS (batched vs per-level ≤ 1e-13 for transforms, ≤ 1e-12 for the full tendency; scalar path bitwise unchanged; rest state exactly zero). Local MX110: T21 L20 949 → 249 ms/tendency (3.8×), T31 L20 1582 → 962 ms |
| S3 SI stepper | NOT STARTED | formulation fixed in plan §3 (with the `1·νᵀ` correction) |
| S4 forcing + ∇⁸ | NOT STARTED | |
| S4b complete-scheme checks | NOT STARTED | |
| S5 experiment module | NOT STARTED | |
| S6 notebook | NOT STARTED | |

## Open issues for S3+

1. **TDR on the local GPU.** T42 L20 batched tendency raises `CUDA_ERROR_LAUNCH_TIMEOUT` on
   the MX110 (Windows display driver, ≈ 2 s kernel limit); the per-level path (4.86 s, many
   short kernels) survives. Either chunk the batched GEMMs along K when a `TDR-safe` flag is
   set, or accept that T42 batched timing is only measurable on Colab (S8 gate). Local
   development of S3–S5 should use T21 L10/L20, where batched works.
2. The PE runner re-`initialize`s the RK4Stepper every step (count mode may clip the last
   step). The HS experiment module (S5) must drive a stepper continuously at fixed Δt so a
   multi-level SI scheme keeps its time levels; do not reuse `run_pe` for HS.
3. `sin_theta_d_theta_coeffs` / `d_lambda_coeffs` are 2-D only; the batched path loops them
   per level (cheap). The SI solve (S3) needs per-degree K×K matrices assembled from
   `hydrostatic_geopotential`, `omega_over_p` (with A=0) and `Δσ` — all CPU-capable.
4. Regression gate at handoff: `pytest tests -q` → **1031 passed, 5 skipped** (1036 collected = 1025 pre-existing + 11 new); the 5 skips are the pre-existing "real capsule not present" skips in untouched files; `git diff main --stat -- tests/` shows additions only. Re-run after S3.
5. Reference run finished (see S0 row); its `series.npz` holds `series.npz` (daily zonal means (day, lat, level)
   of u, v, T, T*², u*², v*T*, u*v* and scalars) atomically after every 10-day chunk.

Forecast: unchanged (M2 at ~T0 + 16 h of working time) — S3–S6 remain ≈ 11.5 h of budget.
