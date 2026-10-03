# Held–Suarez status

Plan: docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md (r3). T0 = 2026-09-27 04:25 −03:00
(07:25 UTC), the approval commit f52ae5b. Budget is elapsed time. Branch:
`feat/semi-implicit-integration` (the plan's `feat/held-suarez`; renamed by the user).
**Handoff point (2026-10-02): production run `hs-T42L20-prod-001` completed (S7–S10); its
verdicts stand as reported: A PASS (with the local tier-A evidence), B FAIL (B2 only), C PASS. The
B2 cause is measured; run 002 is a user decision (plan §5, §6.4).**

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
| S7–S9 launch, gate, production run | DONE (2026-09-30/10-01, Colab T4) | §Production run 001 |
| S10 report | DONE; B FAIL (B2), C PASS | §Production run 001 |

## Production run 001 — `hs-T42L20-prod-001` (verdicts as reported; never edited)

- **Provenance.** Solver commit `206ef25` (clean; pushed, = `origin/feat/held-suarez`), config
  `4cd2eeae…` (production preset: retained T42, L20, lagged damping, RAW(0.1, 0.53), Δt 900 s),
  frozen PROTOCOL.md sha256 `bdc77bbe427d8d1b87a643b0115d189a8a03a09bd205d23038998e901c539574`
  (the committed LF bytes at 206ef25, recorded in every checkpoint). Colab, Tesla T4, CuPy 14.0.1,
  CUDA runtime 12.9; created 2026-09-30 20:15:59Z, day 1200 at 2026-10-01 04:22:29Z, one segment,
  no resumes: **29 190 s wall (8.1 h), 0.253 s/step** (plan S9 estimate for a T4: 10–14 h). The
  committed notebook still pins `21203ff`; the run's checkout assertion means the Colab copy was
  edited to `206ef25` before Run all.
- **Local copy.** `runs/hs-T42L20-prod-001/` (gitignored), the Drive folder downloaded as-is:
  `verify` passes on all three kept checkpoints (days 1180–1200), all 131 zip members byte-equal
  to the extracted files, day-1200 state file = checkpoint `x_curr` (sha256 `c6a1c493…`). The zip
  and the Colab server log are kept beside it in `runs/hs-T42L20-prod-001.download/`. The local
  `report` command reproduces `report/` exactly (modulo CRLF on Windows, DEVLOG 2026-10-02).
- **Verdicts** (`runs/hs-T42L20-prod-001/report/REPORT.md`):

  | tier | verdict | detail |
  |---|---|---|
  | A | INCOMPLETE → **PASS** | Colab report had no tier-A evidence JSON; PASS with the local evidence (below) |
  | B | **FAIL** | B2 only: max \|Δ⟨p_s⟩\|/⟨p_s⟩ = 1.548e−3 > 1e−3. B1 (day 1200, 0 aborts), B3 (KE trend 1.8 %, T trend 0.12 K), B4 (slope −10.6 over (28, 42]), B5 (one config, one clean commit) PASS |
  | C | **PASS** | C1 corr 0.994 / RMSE 1.07 m/s; C2 jets NH 32.97 vs 32.82 m/s, SH 32.75 vs 32.63 m/s, latitude within one Gaussian row, level equal; C3 0.33 K; C4 [T′²] and [v′T′] peaks within 2 %, corr 0.993 / 0.998; C5 PASS; C6 jet 33.0 m/s at 43.3°, surface easterlies |

- **B2 cause, measured (DEVLOG 2026-10-02): the lagged damping placement.** The SI solve
  advances ln p_s with the *undamped* D\*^{n+1}; the dampers then divide D by (1 + 2Δt r). The
  ln p_s update carries −Δt Σ ν_j (D\* − D)_j per step, and divergence under high p_s (Ekman
  pumping in the drag layer) makes its mass effect one-signed. The leak is steady (−1.36e−6/day,
  every day of 200–1200 negative; 1e−3 crossed on day 790), the spatial discretisation's own mass
  tendency is ≤ 3e−10/day, and the lag term predicts −1.32e−6 ± 0.05e−6/day from 11 saved states
  (Rayleigh drag 99.5 %, ∇⁸ 0.5 %). Two-day restarts from the day-1200 checkpoint
  (`tools/held_suarez/mass_budget.py`): lagged 900 s −2.616e−6 (predicted −2.636e−6), lagged
  720 s −2.104e−6 (0.804 × the 900 s leak: first order in Δt), **centred 720 s −2.36e−9**. The
  centred placement (6a447d8, opt-in) solves the damped system, so ln p_s sees the same
  D^{n+1}. This is the mass-budget face of the S4b first-order lag term; climate impact on tier C:
  none detected (C PASS with margins).
- **Stability margin.** Daily max|u| peaked at **100.3 m/s on day 1047** (mean of the daily maxima
  over days 200–1200: 74.3 m/s), against the lagged RAW/advection bound of 105.3 m/s at 900 s (5 %
  margin) and the reference's 94.5 m/s. The centred bound at 900 s (94.9 m/s) is below this
  climate's peak; at 720 s it is 121.1 m/s (21 % margin).
- **Tier A evidence → A PASS.** Full suite at 206ef25 (src/ and tests/ verified identical to the
  commit before and after) on the MX110, 2026-10-02 23:22–00:49Z: **1131 passed, 5 skipped
  (pre-existing), 0 failed** in 87 min. `tools/held_suarez/tier_a_evidence.py` turns its JUnit XML
  into the report's `--tier-a` JSON: tests/ diff vs main additions only (3491 lines), 111 new test
  cases in 11 files all PASS, no new skips, A3 smoke stop/resume bit-identical, solver 206ef25 an
  ancestor of `origin/feat/held-suarez` (fetched). The regenerated report
  (`runs/hs-T42L20-prod-001/assets/report-with-tier-a/`) differs from the Colab `report/` (kept
  unchanged) only in the tier-A block: **A PASS, B FAIL, C PASS**, B and C byte-for-byte the same
  content. Evidence files in `runs/hs-T42L20-prod-001/assets/tier_a/`.

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
backup, resumable run loop (backup every 10 days, keep 3), report. SOLVER_COMMIT = 21203ff (retained-T42 solver;
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

## Decisions before launch (closed by the launch of run 001; kept as the record)

Run 001 launched with item 1 as resolved, item 2's recommendation (lagged, RAW(0.1, 0.53)),
item 3 (Δt 900 s), and item 4 done (the branch is pushed to 206ef25).

1. **Effective truncation — RESOLVED by the opt-in patch (user decision 2026-09-27).** The
   production preset now stores l_max = 43 and retains 42 (DEALIASING_AUDIT.md): evolved T42, ∇⁸ at
   l = 42, `config_mismatches` = [] against REFERENCE_CONFIG.json; product grid 65×130 vs Dinosaur's
   64×128 (a sampling difference, not a resolution one); +4.6 % tendency cost (per-level, MX110).
   The legacy layout (T28) remains as preset `legacy-production` and as the core default.
2. **First-order complete scheme — evidence complete (2026-09-28), decision open.** The
   centred (Crank–Nicolson) damping is implemented as an **opt-in** (`damping_scheme="centred"`,
   SEMI_IMPLICIT.md §9, `tests/test_centred_damping.py`, commit 6a447d8); the default, the
   production preset, RAW α = 0.53 and the notebook pin are unchanged. Measured (DEVLOG
   2026-09-28): reduced solve exact to 1e−15 vs the unreduced damped system; centred + RAW off is
   second order on T21 L10 (ratios 3.77–4.13, equal to the damper-free campaign); RAW α = 0.53
   leaves a first-order term ¼ the size of the lag term. At **Δt = 900 s** the lag term is
   ≈ 1.5 % of the 2-day ζ change (δ 2.3 %, T 0.07 %, q 0.45 %), the steady forced–dissipative
   balance is exact in both placements, and the centred option **does not lower the 900 s error
   vs RK4** (the SI/explicit transient error dominates: ζ 2.4 %, δ/q ≈ 9 %). Stability: the
   RAW/advection non-amplification bound at the retained T42 layout is **94.9 m/s** with centred
   damping vs **105.3 m/s** lagged (reference peak 94.5 m/s), and growth beyond it is fast
   (gain 1.12/step at 100 m/s); the lagged factor extends the explicit leapfrog limit by
   √(1 + 2Δt r), the centred one shortens it to √(1 − (Δt r)²). Cost identical (11.7 ms/step
   non-tendency part at T42 L20 on the MX110, < 0.3 % of a step). Contracts touched by the
   option: `HeldSuarezConfig.damping_scheme` (production `config_sha256` dc5901db… →
   4cd2eeae… with the default; 692fadd1… centred), stepper `state_dict["damping_scheme"]`
   (checkpoints without the key load as lagged; mismatch refuses), notebook pin unchanged
   (21203ff does not contain the option; a centred production run would need a re-pin).
   **Recommendation:** keep the lagged scheme with RAW(0.1, 0.53) at Δt = 900 s for the
   production run, accepting the documented first-order term (≈ 1 % transient-response bias of
   the damped fields, exact steady balance, climate impact unmeasured), because the centred
   option buys no accuracy at 900 s and removes the 10 % advective stability margin over the
   reference's peak winds; keep the centred option as a verified sensitivity switch (it is
   bound 146.9 m/s at 600 s and 121.1 m/s at 720 s, vs lagged 157.5 / 131.8). Not recommended: α = 0.5 (0.4 %
   effect, slightly less damping of explicit oscillations). Short-run numerical evidence only.
   **Update 2026-10-02 (run 001):** the lag term's climate-run consequence is now measured: it
   leaks mass at −1.3e−6/day and fails B2; the centred placement removes it (−1e−9/day). The
   900 s centred bound (94.9 m/s) is below run 001's peak (100.3 m/s), so the stability argument
   above stands at 900 s and moves the centred option to Δt ≤ 720 s. See the run-002 decision.
3. **Δt = 900 s** (proposed from S4b); 1200 s is below the reference's typical peak jet speeds by
   the RAW/advection bound.
4. Push `feat/semi-implicit-integration` so SOLVER_COMMIT is fetchable by Colab (A4).

## DECISION NEEDED: run 002 (plan §5: run 001's verdicts stand; a fix starts `prod-002`)

Options for the B2 failure (cost from run 001's 0.253 s/step on a T4):

1. **Centred damping at Δt = 720 s** (recommended): no new code (opt-in in 206ef25, tested;
   SEMI_IMPLICIT.md §9), leak measured −1e−9/day (B2 margin ≈ 400×), second order without RAW,
   stability bound 121.1 m/s vs run 001's 100.3 m/s peak; ≈ 10.1 h on a T4. Needs: a production
   preset or `--config` with `damping_scheme="centred"`, `dt=720` (a new `config_sha256`), the
   notebook re-pinned to the solver commit that carries it, and `EXPERIMENT_ID = hs-T42L20-prod-002`.
2. Backward-Euler damping *inside* the SI solve ("consistent lagged"): keeps 900 s and the lagged
   stability extension, removes the leak by construction; new code, tests and a stability
   analysis; saves ≈ 2 T4-hours per run.
3. Global mass fixer: hides a measured cause; not recommended.
4. Stop at run 001 (C PASS, B FAIL documented).
