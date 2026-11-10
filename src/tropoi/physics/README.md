# Column physics

Physics of a single atmospheric column, independent of any dynamical core:
this package imports nothing else from `tropoi` and runs on NumPy or CuPy
arrays (level axis first, index 0 at the top).

| Module | Contents |
| --- | --- |
| `thermodynamics.py` | Species, mixtures, `R`, `c_p(T)` (rigid rotor / harmonic oscillator; explicit ortho/para H2) |
| `planets.py` | Bulk presets and their stretched vertical grids (idealized configurations, not validated planets) |
| `radiation/twostream.py` | Two-stream layers and adding, thermal and stellar, absorbing and scattering |
| `radiation/scheme.py` | Column contract, lower boundaries, spectral loop, heating from flux divergence |
| `radiation/gray.py` | Gray optics of published idealized models |
| `radiation/kdist.py` | Correlated-k tables and optics. Method verified on a synthetic spectrum; **no real opacity data** |
| `insolation.py` | Orbit, diurnal-mean, diurnal-cycle, tidally locked and prescribed stellar flux |
| `convection.py` | Enthalpy-conserving dry convective adjustment in closed form |
| `surface.py` | Slab surface and bulk sensible heat flux |
| `column_model.py` | `ColumnPhysics` (shared by all hosts) and the single-column model |

The PE host is [`temporal/column_coupling.py`](../temporal/column_coupling.py).
Conventions, measurements and limits: [status page](../../../docs/radiation/README.md).
