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

## Full suite at the final code revision (2026-11-11)

Run by the repository owner from the main checkout at `b2b8ea2` (clean tree;
Windows, CUDA, Python 3.12.12, pytest 9.1.1), recorded on
[PR #29](https://github.com/AlexandreEros/Palintropos/pull/29#issuecomment-6787079627):

| Command | Passed | Skipped | Failed | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| `python -m pytest -m "not slow" -q --durations=20` | 1651 | 5 | 0 | 22:09 |
| `python -m pytest -m slow -vv -s --durations=10` | 4 | 0 | 0 | 1:11:33 |

This supersedes the statements above that the full suite and the four slow
convergence tests were not run after the fix. CPU-only CI (Ubuntu, Python
3.12) also passed at `b2b8ea2`, with the CUDA tests skipped.

The 300-day Earth and 200-day Jupiter runs completed at the same commit;
their results are in README §6.

## Startup accounting and frictional heating (2026-11-11, commit `d38d2a3`)

```bash
python -m pytest tests/test_radiative_dynamical.py tests/test_physics_columns.py tests/test_no_unreferenced_functions.py tests/test_layout.py tests/test_laplacian_eigenvalues.py -q
```

82 passed in 170 s (Windows, MX110). New or changed tests:

| Test | First result | Final |
| --- | --- | --- |
| slab/atmosphere exchange over a full day (startup step now counted) | passed | passed; slab heat equals accumulated flux x 86400 s to 1e-9 |
| preset hashes unchanged by the new option | passed | passed |
| adiabatic Jupiter, del^8 + heating, within 5 % of the del^8 loss | **failed** (lagged term, momentum only) | passed after the term was made time-centred and the temperature-diffusion enthalpy was returned; criterion unchanged |
| coupled Jupiter, del^8 + heating, residual < 5e-3 W m^-2 | **failed** at 0.016 | passed at 5.7e-4; criterion unchanged |

The full suite was **not** rerun at `d38d2a3`. The change touches
`temporal/column_coupling.py`, `temporal/tendencies/drag.py` and
`run/radiative/`; with the option off the stepper receives the same terms as
before, and the only behavioural change is the extra slab step at startup.

## Full suite on Linux/CUDA at `341acfe` (2026-11-12, Colab Tesla T4)

Run by the repository owner with the [Colab CUDA regression notebook](../../notebooks/Palintropos-Colab-CUDA-Regression.ipynb)
on a clone of `341acfeb9e9331d3a9a53e0eecceb966177cebf4`. Counts below were
re-read from the run's JUnit report and `result.json`.

| | |
| --- | --- |
| Command | `python -m pytest -q -ra --durations=20` (no marker filter: routine and slow together) |
| Result | **1624 passed, 39 skipped, 0 failed, 0 errors, 0 deselected** in 23 min 55 s |
| GPU | Tesla T4 (15 GB), driver 580.82.07, CUDA runtime 13.0 |
| Software | Linux (glibc 2.39), Python 3.13.16, CuPy 14.0.1, NumPy 2.2.6, SciPy 1.16.3, pytest 8.4.2 |

- All 16 tests of `tests/test_radiative_dynamical.py` passed, including the
  coupled budget with resume and both frictional-heating criteria.
- The four `slow` convergence tests passed (336, 227, 227 and 66 s).
- No test skipped for want of CUDA. The 39 skips:

| Skips | Reason |
| ---: | --- |
| 17 | browser tests: no Chrome, Chromium or Edge on the machine |
| 17 | need run directories or capsules that exist only on the development machine (saved-run compatibility, overview, plotting and provenance tests; one Held–Suarez view test) |
| 4 | executed inside `test_canonical_t63_checks_run_in_a_fresh_cuda_context`, which passed; pytest reports the four in-process copies as skipped |
| 1 | the 15-day Williamson 5 acceptance run, opt-in through `TROPOI_W5_ACCEPTANCE=1` |

`341acfe` is the head of the branch at the time of writing, and the only
commits after the batching change `814bcf8` are documentation and evidence,
so the solver and the tests run here are the ones on the branch.

What this is and is not: Linux/CUDA evidence with CuPy 14.0.1 and
Python 3.13, the repository's NumPy and SciPy pins, and Colab's CuPy. It is
not a run of the pinned `cupy-cuda11x==13.4.0` / Python 3.12 environment. On
Windows in that environment the full suite was last run at `b2b8ea2`, the
radiative files at `d38d2a3` (82 targeted tests), and after `814bcf8` only
the three frictional-heating tests and the guards. The browser tests and the
17 data-dependent tests have not run at this head on any machine.

## Not tested at all

- `temporal/tendencies/drag.py` has no dedicated test (it runs inside the
  coupled Jupiter tests).
- The correlated-k path on the GPU, with more than one absorber, or coupled
  to the dynamics.
- `InstantaneousInsolation` and `TidallyLockedInsolation` in a 3-D run.
- Any run at T42 or finer; any run longer than 300 model days.
- `tools/radiation/summarize_run.py` has no test; it was run on the two
  completed runs only.
