# Held–Suarez status

Plan: docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md (r3). T0 = 2026-09-27 04:25 −03:00
(07:25 UTC), the approval commit f52ae5b. Budget is elapsed time. Branch:
`feat/semi-implicit-integration` (the plan's `feat/held-suarez`; renamed by the user).
**Handoff point: S0–S6 done locally (not pushed); stop at the human S7/S8 launch decision.**

| stage | state | evidence |
|---|---|---|
| S0 protocol + reference | DONE | PROTOCOL.md, REFERENCE_CONFIG.json, DEVLOG.md; Dinosaur T42L20 float64 reference **COMPLETE, 1200 days** in `runs/hs-reference-dinosaur-T42L20-001/` (`series.npz` sha256 `baa7cffd…c4663`). **B2 corrected in S5**: the global mean of p_s = exp(ln p_s) changes by 3.23e−4 between day 0 (rebuilt initial state) and day 1200 (stored state) → reference B2 PASS *at the endpoints only*; daily ⟨p_s⟩ was not retained and cannot be established (DEVLOG, `REFERENCE_B2.json`). The earlier claim from mean ln p_s is withdrawn. |
| S1 stepper interface | DONE (c9817a1) | tests/test_steppers.py |
| S2 batched transforms | DONE (7125ddd) | tests/test_batched_transforms.py |
| S3 SI stepper | DONE (7dc40e2, f443efd) | tests/test_semi_implicit.py, SEMI_IMPLICIT.md §§1–7 |
| S4 forcing + ∇⁸ | DONE | §S4 below |
| S4b complete-scheme checks | DONE, with a finding | §S4b below |
| S5 experiment module | DONE | §S5 below |
| S6 notebook | DONE | §S6 below |

## S4 — HS forcing, Rayleigh drag, ∇⁸ (generic stepper hooks)

`tropoi.temporal.hooks` (explicit terms in N; `DiagonalDamping` 1/(1 + 2Δt r) on X^{n+1} after
the SI solve, before RAW; startup on the complete RHS), `tropoi.temporal.tendencies.held_suarez`
(HS94 formulas, Newtonian relaxation on the product grid truncated at the cut, drag and ∇⁸
dampers). `tests/test_held_suarez_forcing.py` 13 tests: hand values at 11 points (worst 2.1e−16),
exact zero at equilibrium, k_T decay (2e−14 pointwise; 4e−15 spectral), independent product-grid
check (1.5e−15), factors per degree (0 ulp), bitwise rest, uniform-T_eq rest (2.6e−21 s⁻¹ after
a day), no-hook path bitwise S3. SEMI_IMPLICIT.md §8.

## S4b — complete scheme (forcing + drag + ∇⁸ + SI + RAW), `tests/test_held_suarez_scheme.py`

- (a) scalar recurrence: |λ| = 0.99739925 / 0.8024 at Δt = 900 s, k_s (no RAW: 1.0026, growing);
  max |λ| over the (k, r, ω_SI) grid 1.000000000000. RAW limits explicit oscillations to
  ωΔt ≤ 0.4371 → with ∇⁸, jet winds ≤ 109.5 m/s (900 s) / 80.4 m/s (1200 s) at T42 are
  non-amplifying; Dinosaur reaches 94.5 m/s → **Δt = 900 s selected**.
- (b) convergence vs RK4, 2 days T21 L10: see "S4b (b)" in DEVLOG. **Finding:** the complete
  scheme is asymptotically *first order* in time; the cause is the plan's damping placement
  (factor on X^{n+1} ⇒ damping evaluated at n+1, local error −rΔt Ẋ), shown in isolation on the
  real stepper (r = k_f: ratios → 1.97; r = 0: 4.000). A second, small first-order term is RAW with α = 0.53 (α = ½
  is second order). Forcing + SI without dampers and RAW is second order at 300/150/75 s (ratios
  3.75–4.12). Not changed: the user mandated
  these factors; see DECISION list.
- (c) 5-day stability at 900 and 1200 s: |X^n − X^{n−1}| peaks on day 2 and decays; computational
  indicator flat after the day-1 spin-up (≤ 0.065 at 900 s).
- (d) startup: n_sub = 1 from L + damping + k_T; with the complete tendency X^1 converges at
  fourth order in n_sub (δ ratios 16.5); no early growth.

## S5 — experiment, checkpoints, statistics, report

`tropoi.run.held_suarez` (`config`, `model`, `statistics`, `checkpoint`, `experiment`, `report`,
`__main__`). Stepper initialized once, fixed Δt, both leapfrog levels in every checkpoint.
Checkpoint = npz without pickle + JSON metadata (stepper state, RNG state, accumulators, config and
sha256, protocol sha256, commit/dirty, CuPy/CUDA/device) + SHA-256 of every array and of the
metadata; tmp + fsync + rename. Resume refuses config, commit, scheme/Δt/operator/hooks,
schema, protocol changes, orphaned checkpoints, foreign backup dirs; hash failures stop.
Evidence: forced stop at day 1 and at mid-day step 100, resumed in a fresh driver, final state
**bit-identical** (sha256 `486752c9…`); one RK4 startup; online block means = offline to 0.0
(2 samples per block); report on synthetic fixtures produces every verdict, never PASS on missing
or incomplete data, byte-deterministic.

## S6 — Colab notebook

`notebooks/held_suarez_colab.ipynb`: pinned SOLVER_COMMIT, GPU/float64 hard check, Drive mount,
reference hash check, 200-step benchmark gate with projection and automatic stop, restore from
backup, resumable run loop (backup every 10 days, keep 3), report. SOLVER_COMMIT = e47a582 (S5 + review fix;
must be pushed). Smoke path executed locally through the same entrypoint
(`tests/test_held_suarez_notebook.py`): benchmark 0.147 s/step (T21 L10, MX110), forced stop at day
1 + resume in a fresh process, backups, report. T42 timing is Colab-only (MX110 TDR).

## Regression gate (final code, HEAD of the S6 follow-up)

`pytest tests -q` on the MX110: **1109 passed, 5 skipped, 0 failed** in 65 min (the 5 skips are
pre-existing: 4 run-through-subprocess placeholders in test_saved_run_overview.py and the
TROPOI_W5_ACCEPTANCE gate). `git diff main -- tests/`: additions only (2702 lines; the one
pre-existing file touched, test_pe_runner.py, gained 21 lines in S1). New skips: none on CUDA; the
GPU tests skip cleanly without CuPy (CI emulation: 50 passed, 17 skipped). Slowest: the two S4b
convergence tests (24 and 19.5 min), S3 convergence (5.5 min). Architecture snapshot regenerated
per commit; `--check` passes; no import cycle changed.

## DECISION NEEDED before launch (plan §6.1, §6.3-type numerics questions)

1. **Effective truncation.** The core truncates analyzed products at 2/3, so l_max = 42 evolves
   l ≤ 28 (effective T28) while Dinosaur T42 evolves l ≤ 42. As configured, tier C is
   INCONCLUSIVE by construction (configuration mismatch) and ∇⁸ has 2.45-day e-folding at the
   smallest evolving wave. Options: (a) l_max = 63 (cut 42), 64×128 state grid, ∇⁸ at l = 42 —
   matches Dinosaur (`config_mismatches` = []), ≈ 4–5× cost per step (A100 needed); (b) keep
   l_max = 42 and accept tier C INCONCLUSIVE (or rerun Dinosaur at T28); (c) ∇⁸ reference at 28.
2. **First-order complete scheme.** Accept the mandated backward-Euler damping on X^{n+1} and RAW
   α = 0.53 (complete scheme first order; lag term ≈ 1–1.5 % of the 2-day ζ change at 900 s), or
   authorize a centred (Crank–Nicolson) damping inside the SI solve (changes the S3 reduced system)
   and/or α = 0.5 (second order, but slightly amplifies explicit oscillations).
3. **Δt = 900 s** (proposed from S4b); 1200 s is below the reference's typical peak jet speeds by
   the RAW/advection bound.
4. Push `feat/semi-implicit-integration` so SOLVER_COMMIT is fetchable by Colab (A4).
