# Current scope and limitations

Palintropos's v0.3.0 baseline comprises GPU BVE, single-layer SWE and a dry
hydrostatic PE core, including completed Held–Suarez validation. The
[validation overview](VALIDATION.md) identifies tested configurations and
accepted results. It does not establish a general planetary climate model.

| Area | Present limitation | Supporting reference |
| --- | --- | --- |
| Physics | No moisture, clouds or aerosols. Gray two-stream radiation, dry convective adjustment and a slab surface are coupled to the PE core and tested at low resolution; real-gas (correlated-k) optics are verified on a synthetic spectrum only and ship with no opacity data. No boundary-layer diffusion or surface stress; constant c_p. No published radiative–dynamical benchmark has been reproduced in 3-D | [Radiation status](radiation/README.md), [issue #26](https://github.com/AlexandreEros/Palintropos/issues/26) |
| Energy | del^8 hyperdiffusion deletes kinetic energy without returning heat (measured up to 89 % of the kinetic energy per day for the Jupiter preset at T9); no frictional heating or energy fixer | [Measurements](radiation/README.md#6-energy-coupling-closure-and-the-hyperdiffusion-finding) |
| Terrain | Fixed band-limited bottom topography; imported data and analytic/function terrain are supported. Isothermal rest and the Mars worked experiment do not validate arbitrary terrain flows or climates | [Terrain guide](../src/tropoi/spatial/terrain/IMPORT.md), [Mars evidence](validation/mars_evolving_2026-10-27/README.md) |
| Quadrature | Geodesic Voronoi quadrature is approximate and orientation-dependent. Fine products reduce error but do not give exact dealiasing | [RH4 measurements](VALIDATION.md#rh4-benchmark), [factorized experiment](validation/r3_characterization.md) |
| Spectral bands | Default BVE/SWE and legacy PE store beyond the nonlinear tendency cut; general Galerkin conservation is not guaranteed. HS production retains T42 with a derivative buffer; cubic aliasing and non-polynomial forcing remain | [Conventions](../src/tropoi/spatial/CONVENTIONS.md), [PE retained truncation](../src/tropoi/temporal/tendencies/RETAINED_TRUNCATION.md) |
| Timestepping | BVE/SWE recompute CFL ceilings after accepted states; BVE explicit-viscosity stability has no separate controller. `tropoi run pe` is fixed-step and only records Courant number | [Integration](../src/tropoi/temporal/integration.py), [PE runner](../src/tropoi/run/pe/GUIDE.md#timestep-policy), [controller tests](../tests/integration/test_bve_controller.py) |
| PE accuracy | Lorenz vertical computational mode, sigma pressure-gradient error and nonzero ln-p_s mass drift remain. Column energy-exchange identities do not prove global energy conservation; no general joint spatial/vertical convergence claim | [PE assumptions](../src/tropoi/temporal/tendencies/PRIMITIVE_EQUATIONS.md#12-validation-boundaries-and-remaining-numerical-limitations), [run 002](held_suarez/STATUS.md#production-run-002--hs-t42l20-prod-002-verdicts-as-reported-never-edited) |
| HS scheme | Lagged damping has a measured mass leak (run 001 failed B2). Centred damping at 720 s passed run 002. RAW alpha=0.53 retains a small first-order term; no blanket second-order accuracy claim for the complete scheme | [Scheme algebra](../src/tropoi/temporal/SEMI_IMPLICIT.md), [measured decisions](held_suarez/DEVLOG.md), [verdicts](held_suarez/STATUS.md) |
| Execution | Solvers require CuPy/CUDA; no independent CPU dynamical core. CPU interfaces, saved-state synthesis and CI tests exist, but CPU CI does not test CUDA numerics | [CPU/GPU boundaries](ARCHITECTURE.md#cpu-and-gpu-requirements), [CI](../.github/workflows/tests.yml) |
| Scale | Dense basis memory/work limits small GPUs; Windows display-GPU kernels may hit TDR. Published MX110 and Colab T4 environments are measurements, not universal hardware support guarantees | [Conventions](../src/tropoi/spatial/CONVENTIONS.md#5-products-timestepping-and-limits), [HS evidence](held_suarez/STATUS.md) |
| Saved views | Display interpolation/quantization and masking are not solver states; no time interpolation or restart through the read-only Simulation API | [Saved runs](../src/tropoi/temporal/SAVED_RUNS.md), [atmosphere views](../src/tropoi/representation/visual/ATMOSPHERE_VIEW.md) |

Lower-level constraints remain in the owning reference: optimized-weight caches
do not hash point coordinates; the legacy Simpson transform is not the Gauss
reference; metric divisions assume supported non-polar samplings. BVE's
`random_low_l` preset has no recorded RNG seed (unlike the deterministic
Held–Suarez perturbation). See [spherical numerics](../src/tropoi/spatial/CONVENTIONS.md)
and [`initialization/bve.py`](../src/tropoi/spatial/initialization/bve.py).

Validation still needed includes broader analytic/independent dynamical-core
benchmarks, systematic convergence and general-terrain studies. An old plan's
unexecuted target is not an accepted result. Optional architecture follow-ups
are tracked separately in [issue #19](https://github.com/AlexandreEros/Palintropos/issues/19).

The legacy `VorticityViewer` summary still labels RMS vorticity as
"Enstrophy" and uses an energy-unit label inappropriate for its area
integral ([source](../src/tropoi/representation/visual/vorticity_viewer.py)).
Use run-scoped [`DiagnosticsRecorder`](../src/tropoi/representation/diagnostics/bve.py)
and the capsule's diagnostic records for invariant measurements. This
presentation debt is not a defect in the current BVE tendency.
