# Shared physics

The physical world every consumer shares — the radiation schemes
(`temporal.column_coupling` hosts them), the sky page
(`representation.visual.sky`) and the experiment drivers — plus the column
physics itself. This package imports nothing else from `tropoi` and runs on
NumPy or CuPy arrays (level axis first, index 0 at the top). Its scientific
data are packaged files under [`data/`](data) (TOML; loaded once through
`importlib.resources`, SHA-256-identified for run provenance); equations,
fits, validation and loading stay in Python. See
[docs/radiation/SHARED_PHYSICS.md](../../../docs/radiation/SHARED_PHYSICS.md).

| Module | Contents |
| --- | --- |
| `constants.py` | Exact SI constants, derived σ/R*/c₂, CODATA G, IAU nominal solar values |
| `species.py` | Species identities and molar masses (from `data/species.toml`); the dataset loader and identities |
| `composition.py` | `Composition`: strict stated fractions, explicit normalization, mole↔mass conversion, property-coverage checks; presets from `data/mixtures.toml` |
| `thermodynamics.py` | `c_p(T)` of a composition (rigid rotor / harmonic oscillator; explicit ortho/para H2) from `data/thermodynamics.toml` |
| `rayleigh.py` | Rayleigh cross sections from sourced optical records (`data/rayleigh.toml`) |
| `column.py` | `AtmosphericColumn`: a 1-D hydrostatic (z, ln p, T) column for optics |
| `star.py` | `Star` (intrinsic: mass, radius, luminosity, T_eff; blackbody consistency) and `ApparentStar` (viewing geometry); catalogue from `data/stars.toml` |
| `orbit.py` | `KeplerOrbit`: six elements, SI, explicit frame/epoch conventions, propagation with an explicit gravitational parameter |
| `rotation.py` | `Rotation`: spin axis, angular velocity, phase |
| `planets.py` | Bulk presets with their star/orbit/rotation and stretched vertical grids (idealized configurations, not validated planets) |
| `radiation/twostream.py` | Two-stream layers and adding, thermal and stellar, absorbing and scattering |
| `radiation/scheme.py` | Column contract, lower boundaries, spectral loop, heating from flux divergence |
| `radiation/gray.py` | Gray optics of published idealized models |
| `radiation/kdist.py` | Correlated-k tables and optics. Method verified on a synthetic spectrum; **no real opacity data** |
| `insolation.py` | `StellarIllumination` (star + orbit + rotation combined, or prescribed); diurnal-mean, diurnal-cycle, tidally locked and prescribed stellar flux |
| `convection.py` | Enthalpy-conserving dry convective adjustment in closed form |
| `surface.py` | Slab surface and bulk sensible heat flux |
| `column_model.py` | `ColumnPhysics` (shared by all hosts) and the single-column model |

The PE host is [`temporal/column_coupling.py`](../temporal/column_coupling.py).
Conventions, measurements and limits: [status page](../../../docs/radiation/README.md).
