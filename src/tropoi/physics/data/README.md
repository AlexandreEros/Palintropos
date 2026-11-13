# Shared physics datasets

These TOML resources ship with `tropoi.physics`. Python loaders validate
the records and expose SHA-256 identities for run provenance.

| Resource | Contents and consumer |
| --- | --- |
| [species.toml](species.toml) | Atomic weights and species formulae for molar masses in [species.py](../species.py). |
| [mixtures.toml](mixtures.toml) | Prescribed mole fractions for idealized climate presets and separate optics presets, loaded by [composition.py](../composition.py). |
| [thermodynamics.toml](thermodynamics.toml) | Rotational and vibrational parameters for temperature-dependent heat capacity in [thermodynamics.py](../thermodynamics.py). |
| [rayleigh.toml](rayleigh.toml) | Sourced refractive-index and depolarization records with validity ranges for [rayleigh.py](../rayleigh.py). |
| [stars.toml](stars.toml) | Stellar radius, effective temperature or luminosity, and provenance for the blackbody catalogue in [star.py](../star.py). |

A species identity alone does not imply heat-capacity or optical coverage.
Keep units, source references, and applicability ranges with each record.
Equations and validation live in the loaders. See the
[shared physics reference](../../../../docs/radiation/SHARED_PHYSICS.md).
