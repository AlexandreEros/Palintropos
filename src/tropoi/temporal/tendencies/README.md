# Dynamical tendencies

These models consume spatial states/operators and return coefficient
tendencies; runners choose the time integration and persistence policy.

| Model | Equation set and reference |
| --- | --- |
| `barotropic.py` | Single-layer non-divergent BVE; [equations and conventions](../../spatial/CONVENTIONS.md#bve-equation-and-signs) |
| `shallow_water.py` | Inviscid rotating single-layer SWE with fixed topography; [formulation](SHALLOW_WATER.md) |
| `primitive_equations.py` | Dry hydrostatic multi-level PE in sigma; [formulation and discrete identities](PRIMITIVE_EQUATIONS.md) |
| `held_suarez.py` | Newtonian temperature relaxation, Rayleigh drag and degree-dependent hyperdiffusion hooks for the dry PE model |

[RETAINED_TRUNCATION.md](RETAINED_TRUNCATION.md) distinguishes PE storage,
retained degrees, derivative buffer and quadratic/cubic product bounds.
The [parent layer](../README.md) owns steppers; [run](../../run/README.md)
owns experiments. [Validation](../../../../docs/VALIDATION.md) includes
RH4, Williamson-5 and the completed Held–Suarez comparison, each with its
configuration and limits. No moisture or radiative heating is implemented.
