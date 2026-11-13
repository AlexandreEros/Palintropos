# Column radiation

This package computes plane-parallel thermal and stellar fluxes and their
heating rates. The level axis comes first, interface 0 is at the top, and
net downward flux is downward minus upward flux.

| Module | Responsibility |
| --- | --- |
| [scheme.py](scheme.py) | `Column`, `LowerBoundary`, and `Radiation` define the column inputs, combine spectral points in chunks, and return interface fluxes and heating from flux divergence. |
| [twostream.py](twostream.py) | Two-stream layer reflection, transmission, sources, and adding for thermal emission and stellar scattering. |
| [gray.py](gray.py) | Prescribed optical-depth profiles and gray optics for the idealized presets. |
| [kdist.py](kdist.py) | Correlated-k tables, quadrature, and longwave and shortwave optics providers. The repository tests the method with a synthetic spectrum and ships no real gas opacity tables. |

The [single-column model](../column_model.py) and the
[primitive-equation coupling](../../temporal/column_coupling.py) share these
routines. See the [radiation status and evidence](../../../../docs/radiation/README.md)
and the [test record](../../../../docs/radiation/TEST_RESULTS.md) for the
conditions under which each path was exercised.
