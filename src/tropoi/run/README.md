# Run orchestration

This layer resolves configuration and coordinates execution, output and
provenance. `lifecycle.py` owns the shared lifecycle; solver subdirectories
contain configuration, execution and runner modules. Model/resource builders
are owned by `temporal.models`, re-exported through `run.*.model`.

| Entry path | Purpose |
| --- | --- |
| [bve](bve/README.md) | Explicit RK4 BVE, state-adaptive advective ceiling; presets in `bve/presets.py` |
| [swe](swe/README.md) | Explicit RK4 SWE with characteristic-speed ceiling; [formulation/CLI](../temporal/tendencies/SHALLOW_WATER.md) |
| [pe](pe/README.md) | Fixed-step dry PE demonstrations, terrain and saved capsules |
| [held_suarez](held_suarez/README.md) | Semi-implicit forced dry benchmark, checkpoints, statistics and acceptance report |
| [radiative](radiative/README.md) | Semi-implicit PE with column physics (gray radiation, convection, surface), energy budget, checkpoints |
| [custom](custom/README.md) | User planets, resolved BVE/SWE initial states and prepared terrain |

[`cli`](../cli/README.md) parses commands and dispatches here. Configuration
validation precedes CUDA initialization; numerical execution needs CUDA.
[Architecture](../../../docs/ARCHITECTURE.md) defines scientific identity,
snapshot schedules and failure semantics. Saved runs are opened through
[representation.archive](../representation/archive/README.md).
