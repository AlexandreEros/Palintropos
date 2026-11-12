# Technical reference

Start at the [repository README](../README.md) for setup and implemented
capabilities, or the [source index](../src/tropoi/README.md) to follow the
software by responsibility. Directory READMEs explain their node, including
when the repository is imported into a wiki.

| Reference | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | Layer contracts, CPU/GPU boundaries, configuration, provenance and tests |
| [Validation](VALIDATION.md) | Accepted results and their scientific limits |
| [Current limitations](KNOWN_LIMITATIONS.md) | Unresolved restrictions with source/evidence links |
| [Radiation status](radiation/README.md) | Radiative–dynamical foundation: what is implemented, tested, measured and unvalidated |
| [Held–Suarez](held_suarez/README.md) | Frozen protocol, acceptance criteria, reference and completed production evidence |
| [Validation records](validation/README.md) | Dated measurements, reproducibility records and associated assets |
| [Generated architecture](architecture/README.md) | Current diagrams, frozen comparison snapshots and generation tooling |
| [Published runs](runs/README.md) | Small committed capsules used by examples/figures |
| [Assets](assets/README.md), [figure recipes](figures/README.md) | Source data, hashes and reproducible illustrations |

Subsystem guides live beside their owners: [numerics](../src/tropoi/spatial/README.md),
[tendencies](../src/tropoi/temporal/tendencies/README.md),
[saved runs](../src/tropoi/temporal/SAVED_RUNS.md),
[visualization](../src/tropoi/representation/visual/README.md),
[run orchestration](../src/tropoi/run/README.md) and
[terrain import](../src/tropoi/spatial/terrain/README.md).
The [notebooks](../notebooks/README.md) retain pinned reproduction workflows.

Old audit prose, completed implementation plans and handoffs are recoverable
from Git history. Frozen protocols, manifests, notebook pins and generated
historical diagrams keep names and paths from their recorded commit; those
strings are provenance. Current prose links use today's hierarchy.
