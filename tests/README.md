# Verification

Tests combine CPU interface/data checks and CUDA numerical checks. The CPU
substrate tests need the native CPU library (`pip install -e .` or
`python setup.py build_ext --inplace`; see
[native substrate code](../docs/ARCHITECTURE.md#native-substrate-code-c)): without it they fail,
they do not skip. CPU CI
installs dependencies without CuPy and runs `python -m pytest -q`; GPU tests
skip when CUDA dependencies are unavailable. A CPU pass does not run or
validate the GPU solvers.

**CPU verification runs.** Production runs execute on CUDA, and a run
without a usable GPU fails instead of falling back to the CPU
(`tropoi.run.execution`). Tests and verification scripts select the CPU
explicitly: `backend="numpy"` in a runner's Python API, the hidden
`--device cpu` option of `python -m tropoi.run.airless run`, or
`TROPOI_EXECUTION=cpu-verification` in the environment of a subprocess. An
explicit CUDA request while that variable is set is an error. The run's
`config.json` records the backend, the CPU model and the precision, and a CPU
run never imports CuPy (`tests/test_run_execution.py`).

The [architecture test instructions](../docs/ARCHITECTURE.md#tests) define
routine GPU regression (`-m "not slow"`), long convergence (`-m slow`) and
release selections. [`architecture_tool/`](architecture_tool/README.md) tests source extraction, graph
generation, layers, provenance and export. Browser tests verify offline
saved-state pages when a supported browser is available. [`fixtures/`](fixtures/README.md)
contains small data contracts, not substitutes for production acceptance.
See [`cli/`](cli/README.md) for command tests and
[`integration/`](integration/README.md) for runner and controller tests.

[Validation](../docs/VALIDATION.md) links measured numerical evidence;
[Held–Suarez criteria](../docs/held_suarez/ACCEPTANCE.md) define its experiment
acceptance. Standalone `audit_r*.py` scripts preserve factorized numerical
experiments and write ignored artifacts; they are not normal pytest tests.
