# Held–Suarez development log

Development-phase decisions and tuning (allowed before the production freeze; see
PROTOCOL.md). One entry per decision, newest last.

- 2026-09-27 S0: HS94 text obtained (GFDL copy). The paper shows T63; we run T42 (cost).
  Comparison target is therefore the T42 Dinosaur run (tier C1–C5); paper values are C6 only.
- 2026-09-27 S0: SI reference temperature 300 K chosen to equal the initial isothermal state
  (Dinosaur does the same to avoid the Simmons–Hoskins–Burridge instability).
- 2026-09-27 S0: hyperdiffusion applied to T′ = T − global mean; ln p_s not diffused
  (GFDL practice). Dinosaur diffuses every field — documented difference.
- 2026-09-27 S2: level-batched transforms are opt-in (`batched_transforms=True` on the PE
  model, not a persisted run-config field) so every existing capsule hash stays on the
  untouched per-level path. Batched forward uses conj((f·w)·Y) — no conjugate copy of Y.
  Local MX110 (Windows TDR ≈ 2 s per kernel): T21 L20 949 → 249 ms/tendency (3.8×);
  T31 L20 1582 → 962 ms; T42 L20 batched hits CUDA_ERROR_LAUNCH_TIMEOUT locally (per-level
  4.86 s survives because its kernels are short). The T42 batched path is therefore measurable
  only on Colab or after a TDR-safe GEMM chunking (open issue, see STATUS.md).
- 2026-09-27 S3: the fast operator L carries the tendency's dealiasing mask (τ, ν and the
  R T_ref q column are zero above the product cut; −∇²(G T) is kept everywhere), so it is the
  exact linearization of the discrete tendency for perturbations inside the cut. Perturbations
  above the cut are not linearizable through the weak-form curl (measured: q′ content above the
  cut alone gives |ζ̇| = 1.4e−10 vs 2.8e−24 inside) and are unreachable from band-limited initial
  data; the S5 perturbation must be seeded inside the cut (PROTOCOL degrees 1–8 already are).
- 2026-09-27 S3: the reduced K×K system is inverted once per (l, Δt) with LAPACK LU and applied
  as one batched device product (no host round trip per step); "exact per degree" is measured as
  the residual of the reduced solution in the unreduced (2K+1)×(2K+1) system, ≤ 1.3e−15.
- 2026-09-27 S3: the S3 convergence test runs with the RAW filter off (ν_raw = 0) so it isolates
  the SI leapfrog's second order (ratios 4.35, 4.30 at 2400/1200/600 s vs RK4 at 300 s, 2 days,
  T21 L10); the filtered scheme is checked in S4b with the forcing. Convergence-test Δt = 2400 s
  is a development choice; the production Δt is still fixed at S8.
- 2026-09-27 S3: RK4 startup substeps are chosen from the RK4 stability polynomial on the
  assembled operator's largest gravity-wave frequency (l = cut, because L has no gravity waves
  above it): n_sub = 1 at T42 L20 for Δt ≤ 1850 s.
- 2026-09-27 S4: forcing and dissipation enter through generic stepper hooks
  (`tropoi.temporal.hooks`): explicit terms are added to N(X^n); `DiagonalDamping` multiplies
  X^{n+1} by 1/(1 + 2Δt r) after the SI solve and before RAW; the RK4 startup integrates the
  complete right-hand side (tendency + terms + −r X). With no hooks the S3 step is unchanged
  bit for bit (tested). The stepper stays independent of HS; `tropoi.temporal.tendencies.
  held_suarez` builds the HS instances and `tropoi.run.held_suarez.config` holds every experiment
  parameter.
- 2026-09-27 S4: Newtonian relaxation is evaluated on the backend product grid with the
  instantaneous p_s (p = σ_k p_s) and truncated at the product cut, exactly like every analyzed
  product of the PE tendency, so the state stays band-limited at the cut and the SI mask stays
  the exact linearization. σ in k_T, k_v and T_eq is the full-level σ_k.
- 2026-09-27 S4: the HS planet is `PlanetaryParameters.ideal_sphere(6.371e6 m)` with
  Ω = 7.292e−5 s⁻¹ set exactly; `from_earth_like` would shrink the radius by the oblateness
  model (~0.06 %). R = κ c_p = 286.857 J kg⁻¹ K⁻¹ (c_p = 1004, κ = 2/7), not the core default
  287.04.
- 2026-09-27 S4 (FINDING, needs a user decision before the freeze): the PE core zeroes every
  analyzed product above the 2/3 cut, so a state with l_max = 42 evolves only l ≤ 28 (T and q have
  identically zero tendency above the cut; ζ, δ only receive −∇²(G T + kinetic) of cut-limited
  fields). The effective truncation of "T42" is T28. Consequences: (i) PROTOCOL's ∇⁸ rule
  "0.1 day at l = 42" damps the smallest *evolving* wave (l = 28) with e-folding 2.45 days, ~24×
  weaker than HS94's "smallest resolved wave 0.1 day"; (ii) Dinosaur T42 evolves l ≤ 42, so the
  tier-C comparison is T28 vs T42, a configuration mismatch that the report marks INCONCLUSIVE;
  (iii) plan B4's range (2L/3, L] with L = 42 would be identically zero; the report uses the
  effective L = 28. Options are listed in STATUS.md; nothing was changed unilaterally.
- 2026-09-27 S4b (a): scalar recurrence of leapfrog + explicit relaxation + backward-Euler damping
  + RAW(ν = 0.1, α = 0.53), built independently in the test and tied to the stepper (a ζ row under
  an explicit (−k + iω) term and a damper follows the 2×2 map to 1e−13). At Δt k_s: |λ| =
  0.99739925 / 0.8024 (physical / computational, Δt = 900 s) and 0.99653385 / 0.8032 (1200 s);
  without RAW the computational mode grows (1.0026 / 1.0035 per step). max |λ| over k ∈ [0, k_s],
  r ∈ {0, k_f, ∇⁸ at l = 1, 14, 28, 42}, ω_SI Δt up to 10: 1.000000000000.
- 2026-09-27 S4b (a, extension): RAW(0.1, 0.53) keeps an *explicit* oscillation (advection,
  Coriolis) non-amplifying only for ωΔt ≤ 0.4371 (gain 1.00036/step at 0.5, 1.0056 at 0.7). With
  the per-degree ∇⁸ damping, the largest 45° jet wind for which every retained degree is
  non-amplifying is 80.4 m/s at Δt = 1200 s and 109.5 m/s at 900 s (T42 as configured: cut 28,
  ∇⁸ at l = 42); 116.7 / 158.0 m/s with ∇⁸ at l = 28; 77.8 / 105.3 m/s for l_max = 63 (cut 42,
  ∇⁸ at 42). The Dinosaur T42 reference's instantaneous max|u| over days 200–1200: median 74.5,
  > 80 m/s on 20 % of days, max 94.5 m/s. **Development Δt selected: 900 s** (1200 s would sit
  below the reference's typical peak winds by this bound). Cost: 96 steps/day, 115 200 steps.
  [Bound values superseded: see 2026-09-28 stability-bound correction.]
- 2026-09-27 S4b (c): 5-day T21 L10 complete-scheme runs at Δt = 900 and 1200 s stay valid; the
  daily max of |X^n − X^{n−1}_f| per block peaks on day 2 and decreases afterwards (ζ 7.05e−7 →
  4.66e−7 at 900 s); the computational-mode indicator |D^n − D^{n−1}|/|D^n + D^{n−1}| drops after
  the day-1 spin-up (ζ 0.171 → 0.058) and stays flat (≤ 0.065 on days 3–5).
- 2026-09-27 S4b (d): the startup substep count (1 at 900 and 1200 s) is sized from L plus the
  damping rates and the Newtonian k_T; with the complete HS tendency, X^1 from 1/2/4/8 substeps
  converges to a 32-substep reference at fourth order (δ ratios 16.5, 16.5, 16.3; ζ, T, q 6–7,
  12, 14.4 — rising toward 16),
  i.e. RK4 is in its stable asymptotic range; the default count is within 8e−5 (relative to
  X^1 − X^0) of the reference. No startup change was needed.
- 2026-09-27 S4b (b), convergence (2 days, T21 L10, HS initial state, relative 2-norm error per
  block vs RK4 of the same right-hand side). Complete scheme vs RK4 300 s at 2400/1200/600/300 s:
  ζ 1.65e−1, 4.19e−2, 1.34e−2, 5.32e−3 (ratios 3.94, 3.12, 2.53); δ ratios 2.43, 1.80, 3.05;
  T 2.51, 2.23, 3.23; q 2.35, 1.74, 3.34 — irregular. Diagnosis, each variable changed alone:
  RAW off → same pattern; exact exponential damping factors → errors identical to 3 digits;
  dampers off in both paths → same δ/T/q pattern; RK4 300 s vs RK4 37.5 s differ by ≤ 8.5e−6
  (reference adequate). Cause 1: the forcing switches on at t = 0 and excites gravity waves up to
  the fastest discrete mode at the T21 cut (ω = 7.7e−4 s⁻¹); their accumulated SI phase error
  ω³Δt²t/3 is O(1) over 2 days for Δt ≳ 200 s, so 2400–600 s is not asymptotic. Halved to
  300/150/75 s (RK4 37.5 s reference): complete scheme ζ 5.32e−3, 2.46e−3, 1.23e−3 (ratios 2.17,
  **2.00**), δ 3.22, 2.53, T 3.37, 2.95, q 3.71, 3.76 → asymptotically **first order**. Cause 2
  (dominant): the plan's damping placement — the factor 1/(1 + 2Δt r) on X^{n+1} evaluates the
  damping at n+1 instead of centred at n (local error −rΔt Ẋ; an exponential factor has the same
  leading term). On the real stepper, x' = −r x + sin(Wt): r = k_f gives ratios 1.88, 1.94, 1.97;
  r = 0 gives 4.002, 4.000, 4.000. Cause 3 (small): RAW with α = 0.53 ≠ ½ has a first-order
  component (scalar oscillator on the real stepper: α = 0.53 → 4.01, 3.85, 3.44, 2.76; α = 0.5 →
  4.001; Robert–Asselin α = 1 → 1.99); with dampers off but RAW on, ζ ratios 3.29, 2.79 at
  300/150/75 s. Forcing + SI with RAW off and no dampers (RK4 150 s reference, ≤ 5.4e−7 from RK4 37.5 s): ζ 3.36e−3,
  8.89e−4, 2.25e−4 (ratios 3.78, 3.94), δ 4.12, 4.06, T 3.79, 3.94, q 3.75, 3.93 → **second order**. Neither frozen choice was
  changed (the user mandated the backward-Euler factors; α = 0.53 is the protocol value, chosen by
  Williams for amplitude accuracy/stability — α = 0.5 slightly amplifies explicit oscillations):
  DECISION NEEDED item in STATUS.md. Size at the production Δt: the lag term is ≈ k_v Δt ≈ 1 %
  of the damped response at 900 s (ζ error vs RK4 at 900 s ≈ 2–2.5 % of the 2-day change, of which
  ≈ 1.4 % first-order), small against the tier-C tolerances but not second order.
- 2026-09-27 S5 (B2 correction): STATUS previously cited the Dinosaur reference's mean ln p_s
  drift (−3.69e−4) as B2 evidence. That is not a mass diagnostic (⟨ln p_s⟩ ≤ ln⟨p_s⟩, and the gap
  grows with eddy p_s variance). `tools/held_suarez/dino_b2.py` (Dinosaur venv) computes the
  Gaussian-weighted global mean of p_s = exp(ln p_s): day 0 (initial state rebuilt with the same
  code, config and seed; not stored) 100000.000 Pa, day 1200 (stored final state) 99967.740 Pa,
  relative change 3.23e−4 ≤ 1e−3 → reference B2 PASS **at the endpoints only**; daily ⟨p_s⟩ cannot
  be established for this run (only daily ⟨ln p_s⟩ and zonal means were kept). Repair: the
  driver now records daily `mean_ps` for future runs; the evidence is
  `docs/held_suarez/REFERENCE_B2.json`. Our own driver records daily ⟨p_s⟩ from exp(ln p_s).
- 2026-09-27 S5: the HS driver (`tropoi.run.held_suarez.experiment`) initializes the SI stepper
  once and only calls `step()` (the PE runner's per-step re-initialization is not used), so the
  RK4 startup runs exactly once per experiment (tested with a counting patch across a mid-day
  stop/resume). Checkpoints are uncompressed npz read with `allow_pickle=False` plus a JSON
  metadata block; every array and the metadata itself carry SHA-256; writes are tmp (unique per
  pid) + fsync + `os.replace` (+ directory fsync on POSIX). The daily statistics follow the
  Dinosaur driver's definitions (instantaneous zonal-deviation eddy products, σ-weighted KE and
  T, Gaussian area weights) so tier C compares like with like; ⟨p_s⟩ is computed from
  exp(ln p_s) daily. The report's computable readings of B2–B4 and C are stated in the
  `report` module docstring (B4 uses the effective truncation L = 28 at l_max = 42).
- 2026-09-27 S5: an independent read-only review of the driver/report found two defects, fixed and
  tested: (1) any exception (e.g. a Drive I/O error while backing up) was logged as "aborted",
  which would FAIL B1 permanently — now only NaN/validation failures are "aborted", other errors
  "interrupted", and backup failures are logged and the run continues; (2) the configuration
  check compared the ∇⁸ reference degree only with the reference's truncation — it now also
  requires it to equal our effective truncation. Also added: metadata SHA-256, refusal to start
  over orphaned checkpoints or into another experiment's backup directory, refusal on PROTOCOL.md
  changes and on unknown commits, untracked files under src/ count as dirty, a missing event log
  or unverified reference B2 can never yield PASS, the grid is part of the reference match.
- 2026-09-27 S6: `notebooks/held_suarez_colab.ipynb` pins SOLVER_COMMIT = f529c6b (the S5 commit;
  it must be pushed before launch — A4). Every numerical action is a call to
  `python -m tropoi.run.held_suarez` at that commit; the notebook adds only the GPU/float64 hard
  check, Drive paths, the reference hash check and control flow. Default production parameters:
  preset `production` (T42 L20, 64×128, 1200 days), Δt = 900 s, MAX_RUN_HOURS = 24,
  DEADLINE = T0 + 24 h (2026-09-28T07:25Z; None accepts finishing later), SESSION_HOURS = 23.5.
  The benchmark gate exits with code 3 ("DECISION NEEDED") when the projection exceeds the budget
  and never lowers the resolution. The local smoke executes the notebook's own code cells
  (`HS_NOTEBOOK_LOCAL_SMOKE=1`: smoke preset, no clone, no Drive) through the same entrypoint:
  benchmark 0.147 s/step over 200 steps with 2 statistics samples and one 0.023 s checkpoint;
  forced stop at day 1, resume in a fresh process with all 22 array hashes verified, completion,
  backups, report.
- 2026-09-27 S6 review: an independent read-only review of the notebook found the pin not yet
  pushed (launch blocker; user decision) and robustness gaps, fixed: the backup-directory check
  now sits inside the backup's error handling (a dropped Drive mount is logged as
  `backup_failed`, never stops the run — tested with an injected OSError); the 10-day state
  snapshots are copied to the backup; the notebook restores from the newest *verifiable* Drive
  backup, falling back to older ones; the committed `REFERENCE_B2.json` is used (after checking
  its `series_sha256` against the pinned reference hash) when Drive holds no b2.json; the notebook
  checks Python ≥ 3.12 and prints a resume instruction when a session-limit stop leaves the run
  unfinished. SOLVER_COMMIT re-pinned to the commit with the solver fix.
- 2026-09-27 dealiasing audit → opt-in retained truncation (user decision after the audit;
  DEALIASING_AUDIT.md). The 2/3 product cut is not needed for dealiasing on the 3/2-rule Gauss
  grid (every quadratic term exact at every degree; cubic V·∇ln p_s terms ~1e−6 for an HS-like
  spectrum) but guarded a representational defect: sinθ∂θ drops its degree-(l_max+1) part, so
  content at l_max produces O(ε) spurious ζ/δ on all degrees (S3's "above-cut" anomaly). Patch:
  `PrimitiveEquationsModel(retained_truncation=L)`, L ≤ l_max − 1 (degree l_max refused); default
  None = the 2/3 cut, bitwise (tested), so BVE/SWE, the PE runner/CLI and published capsules are
  untouched. `SemiImplicitOperator.from_model` defaults `fast_cut` to the model's retained cut.
  HS: `retained_truncation` config field; production = l_max 43 / retained 42 / 64×128 / Δt 900 s
  (config sha256 dc5901db…); ∇⁸ reference defaults to the retained cut; `legacy-production`
  preset keeps the T28 layout; smoke (A3) uses the retained layout at T21 (store 22 / retain 21);
  the S4/S4b development config stays on the legacy layout, so the S4b evidence is unchanged.
  Fresh reference comparison: `config_mismatches(production, REFERENCE_CONFIG)` = []; legacy: the
  two known truncation findings. "Matches Dinosaur T42" = the retained spectral resolution and the
  checked parameters; the product sampling differs (65×130 vs 64×128). RAW/advection bound at the
  retained layout: 105.3 m/s at 900 s (77.8 at 1200 s) vs the reference's 94.5 m/s peak.
  [Superseded: 125.1 / 85.5 m/s, see 2026-09-28 stability-bound correction.]
- 2026-09-27 T42 check of the production layout (store 43 / retain 42, L20, Δt 900 s, per-level
  path — the TDR-safe equivalent of the batched path, agreement ≤ 1e−12): SI fast_cut 42;
  ω_max Δt = 2.045 < 2√2, startup substeps 1 (L only and with the hooks); X^1 with 1/2/4 substeps
  vs 8: n_sub = 1 within 1.0e−4 of X^1 − X^0 (q), δ ratios 15.9, 17.2 (ζ, T, q 8–9, 13), i.e. RK4
  stable and near fourth order. One day (96 SI steps): state valid, the storage-only degree 43
  stays exactly 0, computational-mode indicator 0.267 (startup quarter) then 0.161, 0.190, 0.190
  (flat), max|X^n − X^{n−1}_f| in ζ 7.7e−7 → 1.0e−6 (forced spin-up; at T21 this peaked on day 2
  and decayed). A one-day check only: long-run T42 behaviour is measured by the S8 gate and run.
- 2026-09-28 centred damping option (opt-in; user authorized implementing and testing it, not
  adopting it). `SemiImplicitLeapfrogStepper(damping_scheme="centred")` /
  `HeldSuarezConfig.damping_scheme` take −R X trapezoidally inside the SI solve (L → L − R;
  SEMI_IMPLICIT.md §9); the default `"lagged"` is the S4 placement, unchanged bit for bit
  (tested against the sequential-factor path). `tests/test_centred_damping.py` (14 tests).
  CPU evidence: reduced solve vs the unreduced damped (2K+1)² system — residual 1.04e−15 /
  1.24e−15 / 1.60e−15 at 300 / 1200 / 3600 s, dense-solve agreement ≤ 1e−12 per block; the ζ
  row follows the centred 2×2 map to 1e−13; x′ = −k_f x + sin(Wt) ratios 4.0017, 4.0004, 4.0001
  (lagged 1.875, 1.939, 1.970; 1200 s error 9.3e−4 vs 7.5e−3); scalar recurrence max |λ| over
  (k ≤ k_s, r ∈ {0, k_f, ∇⁸ at l = 1, 14, 28, 42, k_f + ∇⁸₄₂}, ω_SI) = 1.000000000000 at 900 and
  1200 s, pure damping |λ| = √((1 − Δt r)/(1 + Δt r)) (0.9007 at l = 42, 900 s); startup substep
  count identical (the rule reads the rates, not the placement); rest bitwise (T21 L10, 6 steps,
  drag + ∇⁸ + centred SI + RAW). **Stability finding** [withdrawn: see 2026-09-28
  stability-bound correction — the non-amplification bounds are equal]: the RAW-filtered explicit-oscillation
  bound (largest non-amplifying 45° jet wind, every degree ≤ 42, ∇⁸ at 42, k = k_s) is
  **lower** with centred damping: 94.9 m/s at 900 s (lagged 105.3) and 68.1 m/s at 1200 s
  (lagged 77.8), against the reference's 94.5 m/s peak — the lagged factor damps the
  computational/explicit-growth mode of the whole X^{n+1} by 1/(1 + 2Δt r) per step, the
  centred form only through (1 − Δt r)/(1 + Δt r) on X^{n−1}: for an explicit oscillation ωΔt
  the lagged recurrence λ²(1 + 2a) − 2iωΔt λ − 1 = 0 (a = Δt r) turns unstable only above
  ωΔt = √(1 + 2a), the centred (1 + a)λ² − 2iωΔt λ − (1 − a) = 0 already above √(1 − a²), so the
  strong ∇⁸ at l = 42 (a = 0.104) *extends* the leapfrog limit by 10 % in the lagged scheme and
  *shortens* it by 0.5 % in the centred one. Beyond the bound growth is fast for both: worst
  per-step gain over l ≤ 42 at 900 s — lagged 0.9973 (94.5 m/s), 0.9982 (105), 1.078 (110,
  e-folding 0.1 d); centred 0.9973 (94.5), 1.116 (100 m/s, e-folding 0.1 d), 1.238 (105).
  RAW α = 0.5 instead of 0.53 (centred, 94.5 m/s): gain 0.99826 vs 0.99730, still < 1.
  Config hash: adding the field changes the production
  `config_sha256` from dc5901db… to 4cd2eeae9b89ba095609f2675f1951a0e33f1025ee9e67e0b52bc1546e155802
  (lagged) / 692fadd1e78ba945eb8468c501d82c869e241bbf16355bbb882d08eaf89ba2ad (centred); no
  checkpoint or published run carries the old hash (production not launched).
- 2026-09-28 sizing at the production Δt (T21 L10, HS initial state, 2 days, RK4 150 s
  reference of the complete right-hand side, relative 2-norm error per block of the 2-day
  change; scratch script, numbers reproduced by hand in `tests/test_centred_damping.py`'s
  harness where marked). Complete scheme at **900 s** vs RK4: lagged RAW(0.1, 0.53)
  [production] ζ 2.44e−2, δ 8.76e−2, T 4.03e−3, q 9.04e−2; centred RAW(0.1, 0.53) ζ 2.65e−2,
  δ 8.81e−2, T 3.92e−3, q 9.00e−2; centred α = 0.5: ζ 2.62e−2; centred RAW off ζ 2.32e−2; lagged
  RAW off ζ 2.20e−2. Pairwise (the size of each first-order term at 900 s): **damping lag**
  (lagged − centred, same RAW) ζ 1.48e−2, δ 2.27e−2, T 7.4e−4, q 4.5e−3 — i.e. ≈ 1.5 % of the
  2-day ζ change, 60 % of the ζ error vs RK4 but 25 % of δ's, 18 % of T's, 5 % of q's; **RAW
  α = 0.53 vs 0.5** ζ 3.8e−3, δ 7.0e−3, T 3.5e−4, q 5.1e−3 (¼ of the lag term); RAW(0.1, 0.5)
  vs off ζ 3.7e−3, q 1.3e−2. Reading: at 900 s the time-discretisation error is dominated by
  the second-order SI/explicit dynamics of the switch-on transient (δ, q ≈ 9 %, ζ ≈ 2.4 %),
  and **the centred option does not reduce the 900 s error against RK4** (ζ even 8 % larger:
  the lag term partly cancels the SI phase error in this transient); its benefit is the
  asymptotic order, visible below ≈ 300 s (T21 L10 campaign, RAW off: ζ 3.45e−3 / 9.15e−4 /
  2.32e−4 at 300/150/75 s, ratios 3.766, 3.940; δ 4.131, 4.062; T 3.792, 3.944; q 3.756, 3.934
  — identical to 3 digits to the damper-free forcing + SI campaign of S4b; with RAW(0.1, 0.53)
  ζ 3.368, 2.915, δ 3.863, 3.517, T 3.719, 3.687, q 3.773, 3.920: RAW's first-order term is
  what remains). Both placements have the **exact steady balance** x = N/r for constant
  forcing (fixed point of either recurrence), so the lag term is a bias in the response to
  time-varying forcing of the damped fields (≈ rΔt ≈ 1 % for k_f, ≈ 10 % of the ∇⁸ response
  at l = 42 where it is 0.1 d), not a bias of the forced–dissipative balance. Against tier C
  (C1 RMSE 2 m/s ≈ 7 % of the jet, C2 ±10 %, C3 1.5 K): a 1–1.5 % transient-response
  discrepancy is well inside, but tier C measures 1000-day climate means, and the climate
  impact of the placement is **unmeasured** (short-run numerical evidence only; the S4b
  5-day runs and this 2-day campaign are not climate). 5-day centred stability at 900 /
  1200 s (S4b (c) monitor): valid states, max|X^n − X^{n−1}_f| peaks on day 2 and decays
  (ζ 7.07e−7 → 4.67e−7 at 900 s), indicator 0.171 → 0.058–0.065 (900 s), same as lagged.
  **Cost:** the non-tendency part of a step at the production stack shape (61, 44, 44) on the
  MX110 is 11.68 ms (lagged) vs 11.73 ms (centred), i.e. identical and < 0.3 % of the 4.55 s
  T42 L20 tendency; T21 L10 full steps 135.4–135.8 ms for every variant (Colab numbers remain
  estimates). Recommendation recorded in STATUS.md.
- 2026-09-28 stability-bound correction (tests and documents only; no scheme, preset, config
  hash or pin changed). The RAW/advection jet bound of S4b (a) and of the centred-damping entry
  used ω_l = U l/(a cos 45°) + 2Ω with k = k_s on every row. (1) A degree-l harmonic with zonal
  wavenumber m is evanescent poleward of cos φ ≈ m/√(l(l+1)), so m/(a cos φ) ≤ √(l(l+1))/a
  wherever it has amplitude; m = l at 45° overstates the frequency by √2. Measured on the T42
  truncation (`test_advective_frequency_bound_holds_on_the_truncation`: symmetric Galerkin matrix
  of u/(a cos φ) on degrees m..42, 400 Gauss nodes, orthonormality to 1e−12): largest zonal
  advection frequency / (U√(L(L+1))/a) = 0.9883 for solid-body rotation (exactly L/√(L(L+1)),
  m = 42), 0.8789 / 0.8115 for jets at 45° (sin²2φ; Gaussian, 10° wide; m = 28 / 27), 0.7838 at
  30° (m = 34), 0.7539 at 60° (m = 18); the former form overstates these by 1.59, 1.72, 1.46,
  2.62×. (2) Above σ_b the wind has no drag and k_T acts on T only: the jet-level wind rows have
  k = 0. Corrected bound (`jet_wind_bound`: ω_l = U√(l(l+1))/a + 2Ω, k = 0, ∇⁸ at 42 per
  degree, every l ≤ cut non-amplifying): retained T42 lagged 198.8 / 162.5 / 125.1 / 85.5 m/s at
  600 / 720 / 900 / 1200 s, centred 198.7 / 162.5 / 125.0 / 85.5 (was lagged 157.5 / 131.8 /
  105.3 / 77.8, centred 146.9 / 121.1 / 94.9 / 68.1); legacy cut 28 the same 125.1 / 85.5 at
  900 / 1200 s (was 109.5 / 80.4). The first degree to amplify is 23 / 22 / 21 / 19, with
  Δt r∇⁸ < 1e−3 there: the binding constraint is RAW's α = 0.53 growth above ωΔt = 0.4371, not
  the leapfrog limit at l ≈ 42 where the former frequency had put it (ωΔt ≈ 1.0 at l = 40–42),
  which is why the placement no longer matters. Beyond the bound (900 s): gain 1.00016/step at
  130 m/s (e-folding 62–64 d, both placements), 1.00094 lagged / 1.1247 centred at 140 m/s; fast
  growth (> 1.01/step) above 147.5 m/s lagged, 132.8 m/s centred (1200 s: 108.6 / 95.4). A
  94.5 m/s jet at 1200 s: worst gain 1.0004121/step (l = 22), e-folding 33.7 d. Consequences:
  Δt = 900 s stands, with a ≈ 30 % margin over the reference's 94.5 m/s peak (not ≈ 11 %
  lagged / 0.4 % centred); the stability argument against the centred option is withdrawn
  (equal non-amplification bounds; the remaining stability difference is the fast-growth
  threshold, both ≥ 40 % above the peak at 900 s); STATUS updated.
- 2026-10-02 run 001 (`hs-T42L20-prod-001`, post-freeze; nothing here changes its verdicts):
  B2 FAIL attributed. Observed: ⟨p_s⟩ = Gaussian mean of exp(ln p_s) falls steadily from day ≈ 50
  (onset of eddies), −1.354e−6/day fitted over days 200–1200 (200-day segments −1.30 to
  −1.41e−6/day), the daily increment negative on all 1000 days (mean −1.361e−6, sd 1.4e−7, corr
  with KE −0.55); |Δ| crosses 1e−3 on day 790, 1.548e−3 at day 1200. Not spin-up, not round-off.
  **Spatial term:** ⟨e^q q_t⟩/⟨e^q⟩ from the model's ln p_s tendency on 11 saved states (days
  210–1200) is +1e−10 (max 3.1e−10)/day — the semi-discrete scheme conserves mass to ≈ 1e−4 of
  the leak (ln p_s is neither forced nor diffused). **Time scheme:** in the lagged placement
  `SemiImplicitSolver.advance` computes q^{n+1} = q^{n−1} + 2Δt(N_q − ν·d̄) with
  d̄ = (D\*^{n+1} + D^{n−1})/2 from the *undamped* D\*, and `_leapfrog` then divides D^{n+1} by
  (1 + 2Δt r) (Rayleigh drag × ∇⁸). The stored state therefore carries an ln p_s increment
  −Δt Σ_j ν_j (D\* − D)_j ≈ −2Δt² Σ_j ν_j r_j D_j per step that its own divergence does not
  support; in the drag layer D > 0 under high p_s (Ekman), so ⟨e^q ·⟩ is negative definite in
  practice. Each leapfrog chain advances 2Δt per update, so the predicted rate is
  (steps/day ÷ 2)·⟨e^q dq⟩/⟨e^q⟩ = **−1.32e−6 ± 0.05e−6/day** (mean ± s.e., 11 states; drag 99.5 %,
  ∇⁸ 0.5 %) vs −1.36e−6 observed. **Restarts** from the day-1200 checkpoint (`x_curr` sha256
  `c6a1c493…`, RK4 startup n_sub = 1, per-level transforms, MX110), 2 days each: lagged 900 s
  −2.616e−6 (predicted along the trajectory −2.636e−6), lagged 720 s −2.104e−6 (−2.116e−6; ratio
  to 900 s 0.804 = first order in Δt), **centred 720 s −2.36e−9** (fit −2.1e−9/day, ≈ 1100×
  smaller; the centred solve takes ln p_s with the same damped d̄). Evidence:
  `tools/held_suarez/mass_budget.py` (`states`, `restart`), outputs in
  `runs/hs-T42L20-prod-001/assets/mass_budget/`. Reading: the S4b first-order lag term
  (2026-09-27/28, "climate impact unmeasured") is invisible in tier C (PASS, margins above) but
  is a secular mass sink of 0.05 %/year at Δt 900 s. Also measured: daily max|u| peaked at
  100.3 m/s on day 1047, 5 % under the lagged 900 s bound (105.3) and above the centred 900 s
  bound (94.9). Run-002 options in STATUS.md.
- 2026-10-02 run 001 housekeeping findings (no code changed): (1) the run used solver
  `206ef25` while the committed notebook pins `21203ff` — the notebook's own checkout assertion
  shows its Colab copy was edited before Run all; a run-002 launch should commit the pin it uses.
  (2) `report.write_report` writes with the platform newline, so a report regenerated on
  Windows differs from the Colab bytes only by CRLF (parsed JSON equal; byte-equal after
  CRLF→LF). (3) `experiment.protocol_sha256` hashes the working-tree bytes of PROTOCOL.md; a
  Windows checkout with `core.autocrlf=true` gives `185f5898…` instead of the frozen
  `bdc77bbe…` (the committed LF bytes), so resuming a Colab checkpoint locally would be refused
  as a protocol change.
- 2026-10-04 run 002 (`hs-T42L20-prod-002`, PROTOCOL amendment 1: centred damping, Δt 720 s;
  solver a52ea20, config `d1fa645d…`): B PASS, C PASS (STATUS.md). **B2 fix confirmed at climate
  length:** ⟨p_s⟩ changes by −2.016e−6 over 1200 days (fit −1.74e−9/day over days 200–1200, vs
  run 001's −1.354e−6/day: ≈ 780× smaller; the 2-day restart from run 001's day-1200 state
  predicted −2.1e−9/day). The residual is still negative on every day; it is not the damping
  lag (the centred solve feeds the damped D to ln p_s) and is left unattributed (RAW α = 0.53 ≠ ½
  and the SI time-centring are second-order candidates). **Climate impact of the damping
  placement, now measured** (`tools/held_suarez/compare_runs.py`, the report's Gaussian × Δσ
  weights and 5-block sampling error; output in `runs/hs-T42L20-prod-002/assets/compare/`):
  run 001 vs run 002 zonal-mean u RMSE 0.34 m/s = 0.34 × sampling RMS, T 0.14 K (0.57 ×),
  [T′²] 0.58 K² (0.56 ×), [v′T′] 0.20 K m/s (0.57 ×), correlations ≥ 0.998, jet maxima
  NH −0.40 / SH +0.06 m/s at the same row and level. The first-order lag term, the 0.15 % mass
  loss and the Δt change are invisible in the 1000-day climate. Stability: daily max|u| peak
  97.9 m/s (day 483) under the centred 720 s bound 121.1 m/s; 3 days above the 900 s centred
  bound 94.9 m/s. Wall 9.9 h on a T4 (0.247 s/step; run 001 0.253). Note: the Drive backup's
  `events.jsonl` ends with the day-1200 checkpoint event in both runs (the backup is taken at
  the checkpoint, before the driver logs `completed`); B1 reads the day-1200 entry of the
  daily series in `statistics.npz`, not that event.
