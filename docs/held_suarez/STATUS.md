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
  ωΔt ≤ 0.4371 → with ∇⁸, peak winds ≤ 125.1 m/s (900 s) / 85.5 m/s (1200 s) at T42 (either
  layout) are non-amplifying; Dinosaur reaches 94.5 m/s → **Δt = 900 s selected**. *Corrected
  2026-09-28* (the bound first used ω = U l/(a cos 45°) with k = k_s: 109.5 / 80.4 m/s; see
  "Stability-bound correction" below).
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

## Stability-bound correction (2026-09-28)

The RAW/advection jet bound (S4b (a); centred option, DECISION 2) used the explicit frequency
ω_l = U l/(a cos 45°) + 2Ω with the explicit damping k = k_s on every row. Both were wrong for the
wind at jet level:

- **Wavenumber.** A degree-l harmonic with zonal wavenumber m has its amplitude where
  cos φ ≳ m/√(l(l+1)) and is evanescent poleward, so m/(a cos φ) ≤ √(l(l+1))/a wherever it lives:
  putting m = l at 45° overstates the frequency by √2. Checked on the T42 truncation
  (`test_advective_frequency_bound_holds_on_the_truncation`: Galerkin zonal-advection eigenvalues
  for solid-body rotation and jets at 30/45/60°): the largest frequency is 0.75–0.99 of
  U√(L(L+1))/a, and the former form overstated it 1.46–2.62×; the fastest mode of a 45° jet has
  m ≈ 28, not 42.
- **Damping.** Above σ_b = 0.7 the wind has no Rayleigh drag and the Newtonian relaxation acts on
  T only, so the jet-level wind rows have k = 0 (∇⁸ aside).

Corrected bound, ω_l = U√(l(l+1))/a + 2Ω, k = 0, ∇⁸ per degree (`jet_wind_bound`;
`test_raw_limits_explicit_oscillations_and_the_selected_dt_covers_the_jet`,
`test_centred_raw_explicit_oscillation_bound`), retained T42 (cut 42), lagged / centred, m/s:

| Δt | non-amplifying (was) | first degree to amplify | fast growth, > 1.01/step |
|---|---|---|---|
| 600 s | 198.8 / 198.7 (157.5 / 146.9) | 23 | — |
| 720 s | 162.5 / 162.5 (131.8 / 121.1) | 22 | — |
| 900 s | **125.1 / 125.0** (105.3 / 94.9) | 21 | **147.5 / 132.8** |
| 1200 s | 85.5 / 85.5 (77.8 / 68.1) | 19 | 108.6 / 95.4 |

The legacy layout (cut 28) gives the same non-amplifying bounds (125.1 / 85.5 m/s; was
109.5 / 80.4). What changes: the 900 s margin over the reference's 94.5 m/s peak is ≈ 30 %, not
≈ 11 % (lagged) or ≈ 0.4 % (centred); the non-amplification bound no longer depends on the damping
placement, because the binding mode now sits at l ≈ 21 (Δt r∇⁸ < 1e−3), limited by RAW's α = 0.53
growth above ωΔt = 0.437, not at the leapfrog limit at l ≈ 42. Unchanged: Δt = 900 s, every
scheme, preset, config hash and the notebook pin (tests and documents only); the S4b (c) and T42
one-day runs; the accuracy figures. The analysis is still the frozen-coefficient scalar
recurrence; long-run T42 behaviour is measured by the S8 gate and run.

## DECISION NEEDED before launch (plan §6.1, §6.3-type numerics questions)

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
   vs RK4** (the SI/explicit transient error dominates: ζ 2.4 %, δ/q ≈ 9 %). Stability
   (*corrected 2026-09-28*, see "Stability-bound correction"): the RAW/advection
   non-amplification bound at the retained T42 layout is **125.1 m/s lagged vs 125.0 m/s
   centred** at 900 s (reference peak 94.5 m/s). The first degree to amplify is l ≈ 21, where
   RAW's α = 0.53 growth beats a negligible ∇⁸ and the placement does not matter. Above the bound
   growth is slow (e-folding ≈ 63 d at 130 m/s) until the leapfrog limit at l ≈ 42, where the
   placement does matter (the lagged factor extends the explicit leapfrog limit to
   √(1 + 2Δt r), the centred one shortens it to √(1 − (Δt r)²)): growth turns fast
   (> 1.01/step, e-folding < 1 d) above **147.5 m/s lagged vs 132.8 m/s centred** at 900 s
   (108.6 / 95.4 at 1200 s). The earlier 94.9 / 105.3 m/s at 900 s (146.9 / 157.5 at 600 s,
   121.1 / 131.8 at 720 s) came from an overstated advective frequency and are withdrawn;
   corrected 198.7 / 198.8 (600 s), 162.5 / 162.5 (720 s). Cost identical (11.7 ms/step
   non-tendency part at T42 L20 on the MX110, < 0.3 % of a step). Contracts touched by the
   option: `HeldSuarezConfig.damping_scheme` (production `config_sha256` dc5901db… →
   4cd2eeae… with the default; 692fadd1… centred), stepper `state_dict["damping_scheme"]`
   (checkpoints without the key load as lagged; mismatch refuses), notebook pin unchanged
   (21203ff does not contain the option; a centred production run would need a re-pin).
   **Recommendation:** keep the lagged scheme with RAW(0.1, 0.53) at Δt = 900 s for the
   production run, accepting the documented first-order term (≈ 1 % transient-response bias of
   the damped fields, exact steady balance, climate impact unmeasured), because the centred
   option buys no accuracy at 900 s, needs a re-pin, and has the lower fast-growth threshold
   (132.8 vs 147.5 m/s; both ≥ 40 % above the reference's peak, so a margin, not a blocker).
   *Withdrawn 2026-09-28:* the earlier reason that the centred option "removes the 10 %
   advective stability margin" — the non-amplification bounds are equal. Keep the centred option
   as a verified sensitivity switch; at 900 s it is stability-safe. Not recommended: α = 0.5 (0.4 %
   effect, slightly less damping of explicit oscillations). Short-run numerical evidence only.
3. **Δt = 900 s** (proposed from S4b): corrected bound 125.1 m/s, ≈ 30 % above the reference's
   94.5 m/s peak. 1200 s: bound 85.5 m/s, below that peak; a 94.5 m/s jet grows weakly there
   (e-folding 34 d, RAW-limited, not a leapfrog blow-up), but the fast-growth threshold (108.6 m/s
   lagged, 95.4 centred) sits close to the peak, so 900 s remains the proposal.
4. Push `feat/semi-implicit-integration` so SOLVER_COMMIT is fetchable by Colab (A4).
