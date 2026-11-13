# Architecture generator

`python -m tools.architecture.generate` extracts source structure and
writes the current snapshot. Add `--check` to compare the generated files
with the checkout. `python -m tools.architecture.export --out PATH` bundles
the architecture report and supporting files.

The snapshot includes `interactive.html`, an offline atlas built by
`interactive.py` from the same graphs: imports, static calls, possible
dispatch, call sites with argument bindings, runtime observations and a
proposed-package overlay. Two optional inputs feed it, both under version
control and recorded by SHA-256 in `metrics.json`:

* `--proposal docs/architecture/proposed-packages.json` (default; `none` to
  omit): responsibility groups by id prefix, intended order, exceptions;
* `--evidence docs/architecture/evidence/runtime` (default; `none` to omit):
  traces from `python -m tools.architecture.runtime_capture`.

`runtime_capture.py` records Python call evidence of a named scenario in its
own process (registered drivers in `scenarios.py`; `--list` shows them). It
never synchronizes the device or reads device memory; see its module
docstring and [the evidence guide](../../docs/architecture/evidence/README.md).

Start with the [architecture documentation](../../docs/architecture/README.md)
for the pinned development environment, entry-path declarations, and frozen
snapshot commands. The [architecture tests](../../tests/architecture_tool/README.md)
cover the generator, the atlas (including headless-browser checks), runtime
capture and export.
