# Held–Suarez acceptance criteria

The following verdict vocabulary and tiers are reproduced verbatim from
sections 1–2 of the approved revision-3 plan at `f52ae5b`, before the
production runs. They are the criteria implemented by
[`report.py`](../../src/tropoi/run/held_suarez/report.py).
Historical branch names in these frozen criteria identify the launch workflow.
The experiment parameters and amendments remain in [PROTOCOL.md](PROTOCOL.md);
the completed runs and their verdicts are in [STATUS.md](STATUS.md).

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
