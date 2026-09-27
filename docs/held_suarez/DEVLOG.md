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
