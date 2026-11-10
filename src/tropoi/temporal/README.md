# Time evolution and saved simulations

[`tendencies/`](tendencies/README.md) assembles BVE, SWE and dry PE right-hand
sides using spatial resources. `integration.py` owns RK4 and event/snapshot
scheduling; `steppers.py`, `semi_implicit.py` and `hooks.py` provide PE
semi-implicit integration, RAW and opt-in forcing/damping.
[SEMI_IMPLICIT.md](SEMI_IMPLICIT.md) derives that scheme and its limits.
`models.py` builds resources from resolved configuration mappings; it does
not own CLI validation.

`simulation.py` defines the CPU read-only Simulation/Snapshot interface
over a storage protocol. [SAVED_RUNS.md](SAVED_RUNS.md) describes coefficient
views, provenance and explicit plot/view requests. Archive implementations
live in [representation](../representation/README.md); execution belongs to
[run](../run/README.md). A saved Simulation is not a restart interface.

Explicit BVE/SWE ceilings are state-dependent; the PE command is fixed-step.
The [Held–Suarez driver](../run/held_suarez/README.md) uses the semi-implicit
stepper with experiment-specific limits and checkpointing. See
[validation](../../../docs/VALIDATION.md) before interpreting stability or
conservation claims.
