# Completed Held–Suarez baseline

Two 1200-day T42L20 production runs completed. Run 001 (lagged damping,
900 s) returned **A PASS, B FAIL (B2 only), C PASS**. Run 002 (centred
damping, 720 s; protocol amendment 1) returned **A PASS, B PASS, C PASS**.
Verdicts and measured results below remain as reported. Local tier-A
evidence completed the initially INCOMPLETE Colab A reports.

Start with the [experiment README](README.md), [acceptance criteria](ACCEPTANCE.md)
and [frozen protocol](PROTOCOL.md). Raw production/reference runs are local
or externally backed up, not all bundled in Git; this record preserves
their identities, measurements and evidence locations. Current code lives
in [`run/held_suarez`](../../src/tropoi/run/held_suarez/README.md).

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

- **B2 cause, measured ([DEVLOG](DEVLOG.md), 2026-10-02): the lagged damping placement.** The SI solve
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
- **Stability.** Daily max|u| peaked at **100.3 m/s on day 1047** (mean of the
  daily maxima over days 200–1200: 74.3 m/s; reference peak 94.5 m/s).
  The corrected lagged frozen-coefficient non-amplification bound is
  125.1 m/s at 900 s. See the stability interpretation below.
- **Tier A evidence → A PASS.** Full suite at 206ef25 (src/ and tests/ verified identical to the
  commit before and after) on the MX110, 2026-10-02 23:22–00:49Z: **1131 passed, 5 skipped
  (pre-existing), 0 failed** in 87 min. `tools/held_suarez/tier_a_evidence.py` turns its JUnit XML
  into the report's `--tier-a` JSON: tests/ diff vs main additions only (3491 lines), 111 new test
  cases in 11 files all PASS, no new skips, A3 smoke stop/resume bit-identical, solver 206ef25 an
  ancestor of `origin/feat/held-suarez` (fetched). The regenerated report
  (`runs/hs-T42L20-prod-001/assets/report-with-tier-a/`) differs from the Colab `report/` (kept
  unchanged) only in the tier-A block: **A PASS, B FAIL, C PASS**, B and C byte-for-byte the same
  content. Evidence files in `runs/hs-T42L20-prod-001/assets/tier_a/`.

## Production run 002 — `hs-T42L20-prod-002` (verdicts as reported; never edited)

- **Provenance.** Solver commit `a52ea20` (clean; pushed), config `d1fa645d…` (production preset,
  `--dt 720 --damping-scheme centred`), PROTOCOL.md amendment 1, protocol sha256 `26a8ac31…`.
  Colab, Tesla T4, CuPy 14.0.1, CUDA runtime 13.0; created 2026-10-03 17:35:10Z, day 1200 at
  2026-10-04 03:27:35Z, one segment, no resumes: **35 545 s wall (9.9 h), 0.247 s/step**. Initial
  state sha256 `54ba7845…`, identical to run 001's (same seed and preset layout).
- **Local copy.** `runs/hs-T42L20-prod-002/` (gitignored; the Drive backup folder, downloaded and
  extracted by the user), zip in `runs/hs-T42L20-prod-002.download/`: all 130 zip members
  byte-equal to the extracted files, `verify` passes on the three kept checkpoints (days
  1180–1200), day-1200 state = checkpoint `x_curr` (sha256 `469169aa…`); the local `report`
  reproduces `report/` exactly (modulo CRLF).
- **Verdicts** (`runs/hs-T42L20-prod-002/report/REPORT.md`):

  | tier | verdict | detail |
  |---|---|---|
  | A | INCOMPLETE → **PASS** | Colab report had no tier-A evidence JSON; PASS with the local evidence (below) |
  | B | **PASS** | B1 day 1200, 0 aborts; **B2 2.016e−6** (limit 1e−3; fit −1.74e−9/day over days 200–1200, vs run 001's −1.354e−6/day); B3 KE trend 0.45 %, T trend 0.043 K; B4 slope −10.45; B5 one config, one clean commit |
  | C | **PASS** | C1 corr 0.9955 / RMSE 0.90 m/s (sampling 1.05); C2 jets NH 32.56 vs 32.82 m/s, SH 32.80 vs 32.63 m/s, latitude within one Gaussian row, level equal; C3 0.29 K; C4 [T′²] peaks −4.5 % / +3.7 %, [v′T′] −0.4 % / +0.4 %, corr 0.994 / 0.999; C5 PASS; C6 jet 32.6 m/s at 43.3°, surface easterlies |

- **Mass.** The leak measured in run 001 is gone; a residual of −1.7e−9/day remains (still
  one-signed, matching the 2-day restart's −2.1e−9/day; RAW α ≠ ½ and SI time-centring are the
  candidates, unattributed; 500× under B2 over 1200 days).
- **Stability.** Daily max|u| peaked at **97.9 m/s on day 483** (mean of daily maxima, days
  200–1200: 74.8 m/s) at the selected 720 s timestep. The corrected frozen-coefficient bound is
  162.5 m/s; the earlier 121.1 m/s estimate was superseded (see below). Leapfrog
  computational-mode indicator 0.013 (run 001: 0.016).
- **Run 001 vs run 002 climate** (`tools/held_suarez/compare_runs.py`, the report's weights and
  sampling error): every difference is below sampling noise: zonal-mean u RMSE 0.34 m/s
  (0.34 × the sampling RMS of a 5-block difference), T 0.14 K (0.57 ×), [T′²] 0.58 K² (0.56 ×),
  [v′T′] 0.20 K m/s (0.57 ×); jet maxima NH −0.40, SH +0.06 m/s at the same latitude and level.
  The lagged placement's first-order term, its 0.15 % mass loss and the Δt change have **no
  detectable effect on the climate** at this sampling (DEVLOG 2026-09-28's "climate impact
  unmeasured" is now measured).
- **Tier A evidence → A PASS.** Full suite at a52ea20 in a clean scratch worktree (the main
  checkout's notebook was being edited; `runs/` joined in by a junction so the local-capsule
  tests run as before; `PYTHONPATH` = the worktree's `src`), MX110, 2026-10-04 03:58–05:25Z:
  **1132 passed, 5 skipped (the same pre-existing 5), 0 failed** in 87 min; tests/ diff vs main
  additions only, 112 new test cases all PASS, no new skips, A3 bit-identical, a52ea20 an ancestor
  of `origin/feat/held-suarez`. Regenerated report in
  `runs/hs-T42L20-prod-002/assets/report-with-tier-a/` (only the tier-A block differs from the
  Colab `report/`): **A PASS, B PASS, C PASS**. Evidence in `runs/hs-T42L20-prod-002/assets/tier_a/`.


## Reference and restart evidence

The matched Dinosaur reference completed 1200 days in
`runs/hs-reference-dinosaur-T42L20-001/`; its `series.npz` SHA-256 is
`baa7cffd…c4663`. The reference's reconstructed initial and stored final
mean surface pressures differ by 3.23e-4: B2 passes **at those endpoints
only**, because daily mean p_s was not retained. Mean ln p_s cannot establish
that condition; [REFERENCE_B2.json](REFERENCE_B2.json) and [DEVLOG](DEVLOG.md)
preserve the correction and evidence.

Checkpoint/restart retains both leapfrog levels, RNG/accumulators, config
and protocol hashes, solver/environment metadata and array checksums.
The completed smoke tests stopped at day 1 and at mid-day step 100 and
resumed through a fresh driver: final states were bit-identical, startup
ran once, and online/offline block means agreed. Resume refuses incompatible
schema/config/commit/operator/hooks/protocol and corrupted checkpoints.
[`test_held_suarez_experiment.py`](../../tests/test_held_suarez_experiment.py)
and [driver documentation](../../src/tropoi/run/held_suarez/README.md) define
that supported path. These are restart tests, not climate validation.

## Stability interpretation

The later 2026-09-28 frequency-bound correction supersedes the earlier
94.9/105.3 m/s estimates at 900 s and 121.1 m/s at 720 s. The corrected
frequency is `U sqrt(L(L+1))/a + 2 Omega`, with no Rayleigh drag on the
jet-level wind rows. Non-amplification bounds for retained T42 (lagged /
centred) are 198.8/198.7 m/s at 600 s, 162.5/162.5 at 720 s,
125.1/125.0 at 900 s and 85.5/85.5 at 1200 s. Fast growth above 1.01/step
at 900 s begins at 147.5/132.8 m/s. These are frozen-coefficient scalar
recurrence bounds, not nonlinear stability guarantees.

The correction changes no run, timestep, configuration, protocol hash or
acceptance verdict. In particular, the archived protocol's amendment
retains the historical rationale for 720 s. Tests
`test_advective_frequency_bound_holds_on_the_truncation`,
`test_raw_limits_explicit_oscillations_and_the_selected_dt_covers_the_jet`
and `test_centred_raw_explicit_oscillation_bound` check the corrected bounds;
the [development evidence](DEVLOG.md) records their derivation and the
measured damping/time-order campaign.
