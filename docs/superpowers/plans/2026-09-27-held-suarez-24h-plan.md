# Held–Suarez in 24 hours: plan, revision 3 (2026-09-27)

Status: **APPROVED r3 (user, 2026-09-27) with the five corrections of r3 incorporated.** Branch (to create):
`feat/held-suarez` off `main` @ 23f6652. Baseline suite: 1025 tests collected.

## 0. Measured starting point

- PE core: dry hydrostatic, Lorenz sigma (Simmons–Burridge 1981, "SB81"), vorticity–
  divergence, fully explicit, full T, no dissipation, no forcing. Fixed-step RK4 runner, all
  snapshots in RAM, no checkpoints. No stepper abstraction; BVE `step_leapfrog` is dead (R-6).
- **Performance (measured, MX110):** T42 L20 Gauss 64x128 = **4.55 s per tendency**, 365 dense
  `Y_matrix` mat-vecs per call (one level at a time, `Y.conj().T` re-copied per forward
  transform). Fix (S2): batch the K levels into one GEMM, cache the conjugate. After that the
  cost is fp64 GEMM. **All T4/A100 numbers below are estimates until S2 (local) and S8
  (Colab) measure complete timesteps.**
- Google Drive for desktop syncs `G:\O meu disco`; Colab outputs are readable locally.

## 1. Verdict vocabulary (one set, used everywhere)

| Verdict | Meaning |
|---|---|
| PASS | every criterion of the tier holds on the frozen experiment |
| FAIL | at least one criterion violated, and the violation exceeds sampling uncertainty (tier C) |
| INCONCLUSIVE | the criterion is violated but the inter-core difference lies within 2σ of block sampling noise; OR the reference is unavailable, unsuitable (fails its own tier B), or configuration-mismatched |
| INCOMPLETE | run shorter than the protocol, or a stage not reached; never upgraded to PASS |

Tier verdicts are per criterion, then the tier takes the worst per-criterion verdict
(FAIL > INCONCLUSIVE > INCOMPLETE > PASS).

## 2. Tiers, fixed before any HS output exists

### A. Implementation complete
- A1 All pre-existing tests pass; no pre-existing test is modified, deleted, or weakened; any new
  skip is listed with its reason. Evidence: `pytest tests -q` summary + `git diff main -- tests/`
  showing additions only.
- A2 Every new test listed in S1–S6 passes.
- A3 Local smoke (T21 L10, 2 days, SI, forced stop + resume at day 1) finishes; the resumed
  final state is bit-identical to the uninterrupted run **on the same commit, GPU, CuPy/CUDA
  versions** (hashlib over the coefficient arrays).
- A4 `feat/held-suarez` pushed; the notebook pins `SOLVER_COMMIT`, an immutable pushed commit
  containing the solver it runs. The commit that contains the notebook itself is recorded in
  STATUS.md (it necessarily differs from the pin). The pin must be an ancestor of HEAD.

### B. Stable execution (frozen protocol, T42 L20, 1200 days)
- B1 Reaches day 1200 with no validation abort/NaN; every resume hash-verified against the
  checkpoint it loaded.
- B2 |Δ global-mean p_s| / p_s ≤ 1e-3 over the run (dry core; mass should be conserved to
  round-off + truncation).
- B3 Stationarity, days 200–1200, using five 200-day blocks: the linear trend across block means
  of global KE ≤ 10 % of the mean, and of global-mean T ≤ 0.5 K. Basis: HS94 discards 200 days
  of spin-up and averages 1000 days; these are project tolerances chosen so a drift comparable
  to the paper's inter-core spread would be flagged.
- B4 No spectral pile-up: the time-mean KE spectrum (days 200–1200) has a negative least-squares
  log-log slope over (2L/3, L] **and** KE(L) < KE(2L/3). (Revised from the single-degree 2×
  ratio, which a single noisy degree could trip either way.)
- B5 No parameter or code change mid-run (checkpoint carries config hash + commit; a mismatch
  refuses to resume).

### C. Climate validation (days 200–1200, zonal/time means, vs the matched Dinosaur run)
Sampling uncertainty: every scalar metric is computed per 200-day block (5 blocks) for both
cores; σ_block is the pooled block standard deviation; "within sampling noise" means the
inter-core difference ≤ 2·σ_block·√(2/5).
- C1 Zonal-mean u: pattern correlation ≥ 0.98; mass-weighted RMSE ≤ 2.0 m/s.
- C2 Jet maximum: magnitude within ±10 %, latitude within ±4°, level within one σ level.
- C3 Zonal-mean T: mass-weighted RMSE ≤ 1.5 K.
- C4 [T′²] and [v′T′]: peak magnitude within ±20 %, peak latitude within ±5°, pattern corr ≥ 0.9.
- C5 Hemispheric symmetry (forcing is symmetric, so any NH/SH difference is sampling + numerics):
  report |NH−SH| jet maximum and latitude per block; FAIL only if the difference exceeds both
  10 % and 3·σ_block. (Revised from the fixed 10 %.)
- C6 Secondary: paper figures 1–3 read off in S0 (jet ≈ 30 m/s near 45°/σ 0.25, surface easterlies
  in tropics, eddy heat flux peak mid-latitudes near σ 0.85), reported as consistent/not, no
  verdict weight.

**Basis of the thresholds.** They are provisional project tolerances, not Held–Suarez pass
criteria (HS94 defines no numeric acceptance; its two cores differ by roughly 1–3 m/s in the
jet and a few K in polar T). C1–C4 are set at about half the inter-core spread visible in HS94
Fig. 1–3 and in later intercomparisons (Wan et al. 2008, Jablonowski's test suite), so passing
means "closer to Dinosaur than the classic cores are to each other". They can be changed only by
user decision, only before the production launch.

**What must match Dinosaur** (checked by a script, written into `REFERENCE_CONFIG.json`):
truncation T42 on the 64×128 Gaussian grid; 20 equally spaced σ levels; p0 = 1000 hPa; R, c_p,
Ω, a, g; HS94 constants (k_a, k_s, k_f, σ_b, ΔT_y, Δθ_z, T_eq floor 200 K); ∇⁸ hyperdiffusion on ζ,
δ, T with the same e-folding time at l = 42; 1200 days with days 200–1200 averaged; float64
(`jax_enable_x64` — Dinosaur defaults to float32, which would be a mismatch); output cadence.
**Unavoidable differences, documented not hidden:** time integrator (Dinosaur: IMEX RK
"sil3"; ours: SI leapfrog + RAW), Δt, initial perturbation, dealiasing grid details, transform
implementation, SI reference temperature, hyperdiffusion time discretisation. Δt and RAW
coefficient are core choices, so they are frozen in the protocol, not matched.

## 3. Semi-implicit formulation (documented here; `docs/held_suarez/SEMI_IMPLICIT.md` in S3 expands it)

State per (l,m): ζ_k, δ_k, T_k (k = 1..K), q = ln p_s. Existing operators reused: hydrostatic
matrix **G** (`hydrostatic_geopotential`, Φ = Φ_s + G T, exact/linear), column (ω/p) operator
(`omega_over_p`, linear in the mass divergence G_k when V·∇q is dropped), thickness weights ν_k.

- **Reference state:** isothermal T_ref = 300 K (Hoskins & Simmons 1975, SB81), resting, flat
  reference p_s. With T_ref constant, vertical advection of T_ref vanishes and the linear
  thermodynamic term is purely adiabatic conversion.
- **Fast linear operator L** (per horizontal mode, diagonal in l via ∇² → −l(l+1)/a²):
  - δ̇ |_L = −∇²( G T + R T_ref q )          (K×K matrix G, plus the R T_ref q column)
  - Ṫ |_L = −**τ** δ, with τ_k δ = −κ T_ref (ω/p)_k^lin(δ), (ω/p)^lin from `omega_over_p`
    with G_k = δ_k and A_k = 0
  - q̇ |_L = −ν·δ, ν_k = Δσ_k
  - ζ is not in L (no fast linear term).
- **Explicit remainder:** N(X) = tendency(X) − L X, evaluated at time level n. L is only the
  selected fast operator; the full tendency also has linear terms left explicit (Coriolis
  coupling in ζ and δ), so the full-Jacobian residual after subtracting εLY is O(ε) in general.
  Tests: (i) L applied to random Y equals the fast-term function (−∇²(GT + R T_ref q), −τδ,
  −ν·δ) built from the same discrete operators, to round-off; (ii) on a **non-rotating** planet
  (Ω = 0), where every linear term about isothermal rest is in L,
  ‖tendency(X_rest+εY) − tendency(X_rest) − εLY‖ = O(ε²) (ratio ≈ 4 between ε and ε/2);
  (iii) on the rotating planet the same residual is O(ε) and equals the Coriolis linearization
  (f_lm coupling of εY) to O(ε²).
- **Leapfrog SI step** (Δt, three time levels n−1, n, n+1; overbar = ½(·^{n+1} + ·^{n−1})):
  X^{n+1} = X^{n−1} + 2Δt [ N(X^n) + L X̄ ]. Eliminating T̄, q̄ into the δ equation gives per
  degree l a K×K system
  (I + Δt² c_l [ G τ + R T_ref **1** νᵀ ]) δ^{n+1} = rhs_l,   c_l = l(l+1)/a²,
  where **1** is the all-ones column (q is one scalar per mode and enters every level's
  pressure gradient identically; the earlier "ν νᵀ" was wrong). The matrix is not hand-derived
  in code: it is assembled by applying the discrete operators (G, τ from `omega_over_p`, ν) to
  unit vectors, and the reduced solve is verified against a direct solve of the unreduced
  (2K+1)×(2K+1) coupled system in (δ, T, q) per degree. Precomputed and LU-factored once per
  (l, Δt). "Exact per degree" means: no iteration, direct dense solve per l, residual ≤ 1e-12
  relative against the unreduced system.
- **Hyperdiffusion (∇⁸), implicit and exact per degree:** after the SI solve, multiply ζ^{n+1},
  δ^{n+1}, T′^{n+1} (T minus its global mean, so the mean is not damped) by
  1 / (1 + 2Δt K₈ c_l⁴), with K₈ set by the e-folding time τ₈ at l = L: K₈ = 1/(τ₈ c_L⁴). This
  replaces the earlier "dissipation at n−1" wording (the explicit leapfrog-stable alternative
  evaluated at n−1); the implicit factor is unconditionally stable and exactly the analytic
  per-degree damping of the backward step, so no explicit variant is implemented.
- **Rayleigh drag:** k_v depends on σ_k only, so it is applied exactly and implicitly per
  level: ζ^{n+1}, δ^{n+1} ×= 1/(1 + 2Δt k_v,k).
- **Newtonian relaxation:** k_T(φ,σ)(T − T_eq(φ,p)) is evaluated on the grid at level n and
  added to N (explicit). Stable since 2Δt·k_s ≈ 0.007 ≪ 1 at Δt = 1200 s.
- **RAW filter** (Williams 2009): after computing X^{n+1}, d = ν(X^{n−1}_f − 2X^n + X^{n+1});
  X^n_f = X^n + α d; X^{n+1} −= (1−α) d. ν = 0.1, α = 0.53 (frozen in PROTOCOL).
  **Reconciliation with the "no filters absent from HS94" rule:** HS94 prescribes the *forcing*
  and requires each core to document its own numerics (time scheme, filters, dissipation).
  The rule is therefore restated: forcing terms exactly as HS94; numerical dissipation and
  time filtering are core choices that must be documented and frozen before launch, and may not
  be changed during the production run. Adding a filter that is not in the frozen protocol
  (e.g. divergence damping) is a stop condition.
- **Startup:** X^1 from RK4 (the reference stepper) over the interval Δt using n_sub substeps,
  n_sub = ceil(Δt / Δt_explicit) with Δt_explicit the RK4 gravity-wave limit
  (≈ 2.8 / (c_gw · √(L(L+1))/a), c_gw = √(R T_ref)·(fastest vertical mode), measured rather than
  assumed by a 1-step growth check at startup). Then leapfrog from n = 1. Restart from a
  checkpoint needs no startup.
- **Damping factors are backward-Euler, not exponential:** 1/(1+2Δt k) vs e^{−2Δt k}; the
  difference is O((Δt k)²) per step and is documented, not claimed exact in time. "Exact per
  degree" refers only to the spatial (per-l) treatment.
- **Stepper interface** (S1): `TimeStepper.initialize(X0, t0) -> None`,
  `step(t) -> t + dt`, `state -> X^n`, `state_dict()` / `load_state_dict()`. For SI the state
  dict holds `X_prev` (X^{n−1}_f), `X_curr` (X^n, unfiltered until the next step), `t`, `step`,
  `dt`. RK4's holds `X_curr`, `t`, `step`, `dt`.
- **Checkpoint schema** (S5): stepper state dict + `rng_state` + statistics accumulators
  (sums, sums of squares, cross sums, count, block index) + `config_sha256` + `git_commit` +
  `cupy/cuda version` + `sha256` of every array. Atomic write (tmp + rename); resume refuses on
  config/commit mismatch. This is the full set needed for A3 bit-identity.
- **Verification (S3, unforced):** rest isothermal state exact; the three L tests above;
  reduced vs unreduced solve; linear gravity-wave test (initial δ perturbation on the resting
  isothermal state, compare with the analytic SI-leapfrog amplification factor of L per degree
  — modulus 1 for the undamped scheme); 2-day T21 L10 SI vs RK4 with Δt, Δt/2, Δt/4: error
  ratio ≈ 4 (second order).
- **Verification (S4b, complete scheme):** with forcing, ∇⁸, drag, and RAW all enabled on T21
  L10 HS: (a) the linear scalar recurrence of leapfrog + explicit relaxation + RAW is checked
  numerically for |amplification| ≤ 1 at the chosen Δt·k_s and ν, α; (b) 2-day convergence SI
  vs RK4 (both with the same forcing/diffusion) second order; (c) 5-day stability at Δt ≥ 900 s
  with the computational mode monitored (‖X^n − X^{n−1}‖ not growing); (d) RK4 startup at the
  SI Δt shows no growth in the first leapfrog steps.

## 4. Stages, budgets, dependencies (critical path *)

Budgets are my working time excluding usage-limit pauses. T0 = approval.

| # | Stage | Budget | Depends | Acceptance (printed in transcript) |
|---|---|---|---|---|
| S0 | Protocol + reference: transcribe HS94 constants/protocol into `docs/held_suarez/PROTOCOL.md` (with the tiers of §2 and the numerics of §3 marked "frozen at launch"); pin Dinosaur in a scratch venv; write `REFERENCE_CONFIG.json`; 5-day T42 L20 Dinosaur smoke on CPU; measure its throughput | 1.5 h | approval | PROTOCOL committed before any HS code; Dinosaur smoke finite; config match list printed; if 1200 d fit in ≤ 12 h on CPU, start the reference locally in the background now |
| S1* | `TimeStepper` protocol + `RK4Stepper` | 1 h | — | existing BVE/SWE/PE outputs byte-identical; 4th-order convergence on thermal_wave (3 short runs) |
| S2* | Level-batched transforms + cached conjugate (opt-in, PE only) | 1.5 h | S1 | ≤ 1e-13 relative vs per-level path; local ms/tendency before/after printed; old path default for existing runs |
| S3* | SI stepper per §3 + SEMI_IMPLICIT.md | 4 h (3 h debug timebox) | S1, S2 | §3 verification list |
| S4* | HS forcing + ∇⁸ hyperdiffusion (as pluggable "after-step" and "explicit forcing" hooks on the stepper) | 2 h | S1 | T_eq, k_T, k_v vs hand values; forcing = 0 at (T_eq, V=0); pure relaxation decays at k_T; per-degree damping factor equals 1/(1+2Δt K₈ c_l⁴); rest stays at rest |
| S4b* | Complete-scheme verification (§3 "S4b" list) | 1 h | S3, S4 | all four checks printed with numbers |
| S5* | Experiment module: seeded perturbation, atomic checkpoints (schema above), online block statistics, per-day scalar series, `report` command implementing §1–§2 | 3 h | S3, S4 | restart bit-identical (A3); accumulators = offline means to 1e-12; report deterministic on synthetic PASS/FAIL/INCONCLUSIVE/INCOMPLETE fixtures |
| S6* | Colab notebook: pinned commit, GPU check, Drive mount, **benchmark gate (200 complete steps incl. statistics + one checkpoint write, projected 1200-day wall time)**, run/resume loop, Drive backup every 10 days (keep 3), Dinosaur reference cell, report cell; push branch | 1.5 h | S5 | notebook's smoke path executed locally through the same entrypoint |
| — | Buffer | 1.5 h | | |
| **M2** | **Tier A / launch-ready** — target T0 + 16 h; goal 1 ends with DECISION NEEDED: launch | | | |
| S7* | [HUMAN] open notebook, mount Drive, Run all. Dinosaur reference in a second Colab session (or local CPU if S0 showed it fits), so the two do not compete for one GPU | | M2 | |
| S8* | Benchmark gate; also fixes Δt for production if S3 showed the stability margin | auto | S7 | projection printed; over the session/deadline limit → notebook stops, DECISION NEEDED |
| S9* | Frozen production run `hs-T42L20-prod-001` (+ reference) | A100 est. 1–2 h; T4 est. 10–14 h | S8 | tier B |
| S10* | Automatic report (in notebook, and locally from synced files) | 0.5 h | S9 | REPORT.md + report.json, verdicts per criterion and tier |

**Budget is elapsed time** (T0 = approval, 2026-09-27), including usage-limit pauses; the
forecast in STATUS.md is revised as each stage's measurement lands. Reference-run compute
(second Colab session, or local CPU) is verified in S0 before concurrent runs are assumed.
**Latest viable launch times.** A100: launch by T0 + 20 h for tier C by T0 + 24 h. T4: tier C
cannot finish inside 24 h (run alone is ≥ 10 h estimated); the attainable milestone is tier A
plus an INCOMPLETE tier B/C report with the run continuing; the full validation plan stays in
force and the report is regenerated when the run finishes.
**Dropped:** the local 30-day SI-vs-RK4 comparison (RK4 at T21 L10 for 30 days is hours of local
GPU time and adds nothing over the convergence test + the 5-day stability run).
**Reordered:** performance (S2) now precedes all expensive tests; S3 (SI) precedes forcing (S4)
so the stepper/checkpoint interface is settled before the pipeline is built around it.

## 5. Development vs frozen production experiment

- During development (before M2), Δt, τ₈, RAW coefficients, and debugging changes are free,
  each recorded in `docs/held_suarez/DEVLOG.md` with the reason.
- `PROTOCOL.md` is frozen at launch: its sha256 goes into the checkpoint. Experiment IDs:
  `hs-T42L20-prod-NNN`.
- If a bug is found after launch: the running experiment is stopped or completed, its verdicts
  are reported as they stand (never edited), the bug fix is committed with a test, and a new
  experiment `prod-(N+1)` starts. The old run's directory and report are kept.
- Thresholds never change to convert FAIL → PASS. A threshold change requires a user decision
  and applies only to experiments launched after it.

## 6. Stop and consult (print `DECISION NEEDED:`, commit STATUS.md, push notification)

1. HS94's protocol or Dinosaur's capabilities conflict with the plan (resolution, levels, ∇⁸
   timescale, float64).
2. A change would alter a pinned hash, published run, frozen string, or an existing test.
3. SI unstable at Δt ≥ 900 s after the 3 h timebox, or would need a filter absent from the frozen
   numerics list.
4. Any change to numerics, thresholds, or code of a **production** experiment after its
   launch (development tuning before launch is allowed and logged in DEVLOG.md, §5).
5. S8 projects past the session or deadline limit (options: bigger GPU, FFT+Legendre transforms
   ≈ 4 h more work, accept finishing later). Resolution is never lowered unilaterally.
6. Tier B or C FAIL: report only.
7. Any outward-facing action beyond pushing `feat/held-suarez`.
8. M2 reached (human launch).

## 7. `/goal` mechanics verified in this session (2026-09-27)

- Engine: Claude Code 2.1.281 (desktop); `/goal`, check-ins, and pause/continue on usage limits
  are all ≥ their minimum versions. No `disableAllHooks`/`allowManagedHooksOnly` in user or project
  settings; no managed settings directory exists.
- Auto mode active; `/goal` inherits it.
- Push notification tested: **desktop notification only — "Mobile push not sent (Remote Control
  inactive)".** To get phone alerts, turn on Remote Control in the app before sleeping.
- Not verifiable by me: that the desktop app actually auto-continues after a usage limit (docs
  say yes for claude.ai-subscription interactive sessions; cannot be triggered on purpose), and
  that the evaluator reads long transcripts reliably — hence every acceptance item is printed
  compactly in a final "LAUNCH-READY REPORT"/"DECISION NEEDED" block, and STATUS.md is the
  resume point if the goal is lost.

## 8. Goal commands

**Goal 1** (run after approval; long `/goal` text is allowed up to 4000 chars):

/goal Execute docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md (revision 3) stages S0–S6 in order on branch feat/held-suarez, obeying section 6 stop rules and section 5 development rules (development tuning allowed and logged in docs/held_suarez/DEVLOG.md; the production protocol freezes only at launch). Make routine implementation decisions yourself; stop only for a scientific or scope decision or the Colab launch. Done when EITHER (a) you print a block titled "LAUNCH-READY REPORT" containing: the `pytest tests -q` summary with 0 failures; `git diff --stat main -- tests/` showing only additions plus a list of any new skips with reasons; the name and PASS result of every new test listed for S1–S6 including the S4b complete-scheme checks with their numbers; the S2 before/after ms-per-tendency numbers; the local T21 L10 2-day SI smoke output including "forced stop/resume bit-identical: PASS"; `git status` clean; `git rev-parse HEAD` equal to `git rev-parse origin/feat/held-suarez`; the notebook's SOLVER_COMMIT pin, shown to be a pushed ancestor of HEAD; the revised completion forecast; and the line "DECISION NEEDED: launch Colab" — OR (b) you print "DECISION NEEDED:" followed by the question and options, after committing the same text to docs/held_suarez/STATUS.md and calling PushNotification. Update STATUS.md (with the elapsed-time forecast) and commit at the end of every stage. Never modify or delete existing tests, pinned hashes, published runs, or frozen strings; never push to main; never open or merge PRs. If neither (a) nor (b) holds after 18 hours of elapsed time since the goal was set, print STATUS.md and stop.

**Goal 2** (after you launch Colab):

/goal Monitor experiment hs-T42L20-prod-001 per docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md S8–S10 using the Drive-synced files under G:\O meu disco\Palintropos-HS\ (read-only; never edit them). Check no more often than every 20 minutes. Done when EITHER you print the full REPORT.md produced by the committed `report` command with a verdict (PASS/FAIL/INCONCLUSIVE/INCOMPLETE) for every criterion of tiers B and C, commit it under docs/held_suarez/, and push feat/held-suarez — OR you print "DECISION NEEDED:" per section 6 after committing STATUS.md and calling PushNotification. Never change thresholds, parameters, or code to alter a verdict; a bug found now is reported, not fixed in place. If the run is still incomplete 22 hours after this goal was set, produce the INCOMPLETE report and stop.
