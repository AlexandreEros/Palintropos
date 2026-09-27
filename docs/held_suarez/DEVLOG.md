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
