# Architecture tool tests

CPU tests for source and graph extraction, layer declarations, metrics,
rendering, package and entry-path coverage, provenance, export, the
interactive atlas payload (`test_arch_interactive.py`) and runtime capture
(`test_arch_runtime_capture.py`). `test_arch_atlas_browser.py` drives the
generated `interactive.html` in a headless Chrome, Chromium or Edge with
networking blocked (`tools/browser_cdp.py`) and is skipped when none is
installed; one CuPy check is skipped without a CUDA device. Run
`python -m pytest tests/architecture_tool -q` with the development
requirements installed. See the [architecture tool guide](../../tools/architecture/README.md).
