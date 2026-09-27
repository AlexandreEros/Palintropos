# Held–Suarez status

Plan: docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md (r3). T0 = 2026-09-27 ~04:00 UTC
(approval). Budget is elapsed time.

| stage | state | evidence |
|---|---|---|
| S0 protocol + reference | DONE | PROTOCOL.md, REFERENCE_CONFIG.json; HS94 text obtained (GFDL copy); Dinosaur T42L20 float64 CPU: 112 ms/step at Δt = 600 s → 1200 d ≈ 5.4 h; reference run **launched locally** in `runs/hs-reference-dinosaur-T42L20-001/` (resumable driver `tools/held_suarez/dino_reference.py`, resume tested) |
| S1 stepper interface | DONE | `temporal/steppers.py` (`TimeStepper`, `RK4Stepper`); tests/test_steppers.py 5 PASS; PE runner adopts it with a bitwise characterization test (test_pe_runner: `test_run_pe_matches_hand_rolled_rk4_step_array_bitwise` PASS); BVE/SWE runners untouched |
| S2 batched transforms | next | |
| S3 SI stepper | | |
| S4 forcing + ∇⁸ | | |
| S4b complete-scheme checks | | |
| S5 experiment module | | |
| S6 notebook | | |

Forecast (revised as measurements land): M2 (launch-ready) at T0 + ~16 h if no usage-limit
pauses. Reference run finishes at ~T0 + 7 h on the local CPU, so no GPU contention on Colab.

Open decisions: none.
