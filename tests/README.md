# Verification

Tests combine CPU interface/data checks and CUDA numerical checks. CPU CI
installs dependencies without CuPy and runs `python -m pytest -q`; GPU tests
skip when CUDA dependencies are unavailable. A CPU pass does not run or
validate the GPU solvers.

The [architecture test instructions](../docs/ARCHITECTURE.md#tests) define
routine GPU regression (`-m "not slow"`), long convergence (`-m slow`) and
release selections. `architecture_tool/` tests source extraction, graph
generation, layers, provenance and export. Browser tests verify offline
saved-state pages when a supported browser is available. `fixtures/`
contains small data contracts, not substitutes for production acceptance.

[Validation](../docs/VALIDATION.md) links measured numerical evidence;
[Held–Suarez criteria](../docs/held_suarez/ACCEPTANCE.md) define its experiment
acceptance. Standalone `audit_r*.py` scripts preserve factorized numerical
experiments and write ignored artifacts; they are not normal pytest tests.
