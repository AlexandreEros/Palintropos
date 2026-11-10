# Test record for the radiative–dynamical foundation

Environment: Windows 11, NVIDIA GeForce MX110 (2 GB), Python 3.12.12,
NumPy 2.2.6, SciPy 1.16.3, CuPy 13.4.0 (pins in `requirements.txt`).
Date: 2026-11-10. Source: commit `2d5ca13` (the commit that adds this file
changes documentation only).

## Passed

```bash
python -m pytest tests/test_radiative_dynamical.py tests/test_radiation_twostream.py tests/test_physics_columns.py tests/test_radiation_kdist.py tests/test_layout.py tests/test_no_unreferenced_functions.py -q
```

52 passed in 119 s. This covers the solver, gray optics, correlated-k on the
synthetic spectrum, column equilibria, rest states on four planet presets,
adiabatic energy, the lagged-term hook (bitwise no-op by default), coupled
energy closure, the hyperdiffusion sink, checkpoint/resume identity, slab
bookkeeping, the layer rule and the dead-code guard.

`python -m tools.architecture.generate --check`: up to date at `2d5ca13`.
Architecture-tool tests (`tests/architecture_tool`, 68 tests): passed before
the snapshot was regenerated for the final source; not rerun after.

## Failed on first run, and what was done

| Test | First result | Cause | Change |
| --- | --- | --- | --- |
| coupled Jupiter energy closure | residual 0.058 vs bound 0.005 W m^-2 | del^8 deletes kinetic energy (0.059 W m^-2) | bound kept; the test now runs with del^8 off (0.003), and a new test asserts residual = kinetic-energy loss with del^8 on |
| slab/atmosphere exchange | 120.41 vs 122.11 W m^-2 | the test divided by 86400 s; the accumulator covers 85200 s (startup step excluded) | test corrected; the uncounted startup step is documented as a gap |
| adiabatic energy (first version) | −52 W m^-2 on Jupiter | del^8 sink (see README §6) | rewritten to measure with and without del^8; bounds are regression envelopes around the measurements, stated as such |

## Routine regression suite

```bash
python -m pytest -m "not slow" -q
```

Run on `2d5ca13` (existing BVE/SWE/PE/Held–Suarez tests plus the new ones):
**1633 passed, 22 skipped, 1 failed, 4 deselected (slow)** in 1269 s.

The failure was `test_laplacian_eigenvalues.py::
test_no_second_definition_of_the_eigenvalue_in_src`, a textual guard against
spelling `l (l + 1)` outside `spatial/laplacian.py`. It flagged two new
files. `run/radiative/model.py` did re-spell the eigenvalue in the seeded
perturbation and now calls `laplacian_eigenvalues()`. `physics/
thermodynamics.py` uses `J (J + 1)` for the rotational levels of H2, which is
not the Laplacian, and was added to the guard's allow-list with that reason.

After the fix, the guard, the radiative–dynamical tests, the layer rule and
the dead-code guard were rerun: 64 passed in 91 s. **The full suite was not
rerun after the fix**; the change touches only the initial-state seed of
`tropoi.run.radiative`. The four `slow` convergence tests and the browser
tests were not run.

## Not tested at all

- `temporal/tendencies/drag.py` has no dedicated test (it runs inside the
  coupled Jupiter tests).
- The correlated-k path on the GPU, with more than one absorber, or coupled
  to the dynamics.
- `InstantaneousInsolation` and `TidallyLockedInsolation` in a 3-D run.
- Any run longer than 30 model days; any run at T42.
- CPU-only CI (Linux, no CuPy): the new host tests are expected to run and
  the GPU ones to skip, but this was not executed.
