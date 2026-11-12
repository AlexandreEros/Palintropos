# tropoi package

Palintropos implements BVE, single-layer SWE and dry hydrostatic PE dynamics
on a sphere. The runtime dependency direction is
`cli → run → representation → temporal → spatial`; lower layers do not
depend on orchestration. [`tests/test_layout.py`](../../tests/test_layout.py)
enforces it (type-only annotations are discussed in the architecture).

| Layer | Responsibility and interfaces |
| --- | --- |
| [physics](physics/README.md) | Column physics independent of any dynamical core: gas thermodynamics, planet presets, radiation (two-stream solver, gray and correlated-k optics), insolation, dry convection, slab surface |
| [spatial](spatial/README.md) | Environment/specifications, grids, transforms, operators, states and initial conditions |
| [temporal](temporal/README.md) | Tendencies, integration/steppers, model assembly and read-only Simulation/Snapshot |
| [representation](representation/README.md) | Archive/schema, diagnostics and explicit visualization |
| [run](run/README.md) | Resolved configuration, lifecycle, execution and Held–Suarez experiments |
| [cli](cli/README.md) | Argument parsing and dispatch through the single `tropoi` command |

[Architecture](../../docs/ARCHITECTURE.md) documents their relationships;
[generated graphs](../../docs/architecture/README.md) show static structure.
Solvers require CUDA; many reading, configuration and viewing operations
are CPU-only. [Validation](../../docs/VALIDATION.md) and
[limitations](../../docs/KNOWN_LIMITATIONS.md) delimit scientific claims.
These Markdown pages are repository references; wheels package Python,
CUDA sources and HTML templates, without promising to install this wiki.
