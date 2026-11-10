# Spatial representation and discretization

This layer defines the prescribed environment, spherical sampling and
state representation. It does not advance time or manage runs.

| Contents | Contract |
| --- | --- |
| `environment.py`, `specification.py`, `planet.py` | CPU environment/versioned specifications; Planet assembles GPU discretization resources |
| [grids](grids/README.md), [transforms](transforms/README.md), `spherical_backend.py` | Geometry, SH analysis/synthesis and backend-owned ProductSpace sampling |
| [operators](operators/README.md), `laplacian.py` | Shared spectral calculus and the single zero-monopole Laplacian inversion policy |
| `sigma_coordinate.py` | Backend-independent PE column hydrostatics, continuity and exchange identities |
| [states](states/README.md), `modes.py`, [initialization](initialization/README.md), `truncation.py` | Packed states, host views, preset construction and spectral-support gates |
| [terrain](terrain/README.md) | Fixed model topography, elevation import and decorative terrain |

[CONVENTIONS.md](CONVENTIONS.md) defines basis normalization, signs,
quadrature envelopes, product policy and numerical constraints.
[Temporal tendencies](../temporal/tendencies/README.md) consume these
resources; [custom preparation](../run/custom/README.md) resolves user
specifications. [Validation](../../../docs/VALIDATION.md) separates measured
Gauss/geodesic behavior from general guarantees.
