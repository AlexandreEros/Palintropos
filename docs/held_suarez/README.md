# Held–Suarez dry-dynamics baseline

The implemented experiment compares Palintropos's dry hydrostatic PE core
with a pinned T42L20 Dinosaur reference under Held–Suarez relaxation and
drag. Production run 002 (centred damping, 720 s) completed 1200 days and
passed implementation, execution and climate tiers. Run 001 (lagged, 900 s)
passed A/C but failed mass-drift criterion B2. This is a controlled dry
benchmark, not a general planetary-climate validation.

| Read | Role |
| --- | --- |
| [STATUS.md](STATUS.md) | Completed results, provenance, measured mass leak/correction and stability interpretation |
| [ACCEPTANCE.md](ACCEPTANCE.md) | Verbatim pre-run verdict vocabulary and tiers extracted from the completed plan |
| [PROTOCOL.md](PROTOCOL.md) | Frozen source specification, experiment choices and amendment 1 |
| [REFERENCE_CONFIG.json](REFERENCE_CONFIG.json), [REFERENCE_B2.json](REFERENCE_B2.json) | Machine-checked reference parameters and endpoint-only reference mass evidence |
| [DEVLOG.md](DEVLOG.md) | Dated measured development evidence, including time-order, damping, reference corrections and performance |
| [Driver](../../src/tropoi/run/held_suarez/README.md) | Implementation, CLI, checkpoints and statistics |
| [Semi-implicit scheme](../../src/tropoi/temporal/SEMI_IMPLICIT.md), [retained truncation](../../src/tropoi/temporal/tendencies/RETAINED_TRUNCATION.md) | Numerical derivations and bounds |
| [Climate figure](../../src/tropoi/representation/visual/HELD_SUAREZ_CLIMATE.md) | Saved-state sampling, definitions and reproducible rendering |

Protocol bytes are preserved because runs/checkpoints record their hash.
Its original development banner, references to the old plan and historical
stability estimates describe the freeze process, not today's completion
state. The criteria are now readable in ACCEPTANCE.md; STATUS.md states the
corrected stability bounds. DEVLOG entries describe the configuration at
their date; later decisions and completed-run measurements supersede earlier
proposals. Raw production/reference capsules are not all committed; their
identities and external/local evidence locations remain explicit.
