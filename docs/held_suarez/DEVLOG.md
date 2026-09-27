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
