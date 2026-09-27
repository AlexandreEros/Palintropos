# Held–Suarez development log

Development-phase decisions and tuning (allowed before the production freeze; see
PROTOCOL.md). One entry per decision, newest last.

- 2026-09-27 S0: HS94 text obtained (GFDL copy). The paper shows T63; we run T42 (cost).
  Comparison target is therefore the T42 Dinosaur run (tier C1–C5); paper values are C6 only.
- 2026-09-27 S0: SI reference temperature 300 K chosen to equal the initial isothermal state
  (Dinosaur does the same to avoid the Simmons–Hoskins–Burridge instability).
- 2026-09-27 S0: hyperdiffusion applied to T′ = T − global mean; ln p_s not diffused
  (GFDL practice). Dinosaur diffuses every field — documented difference.
