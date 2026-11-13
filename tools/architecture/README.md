# Architecture generator

`python -m tools.architecture.generate` extracts source structure and
writes the current snapshot. Add `--check` to compare the generated files
with the checkout. `python -m tools.architecture.export --out PATH` bundles
the architecture report and supporting files.

Start with the [architecture documentation](../../docs/architecture/README.md)
for the pinned development environment, entry-path declarations, and frozen
snapshot commands. The [architecture tests](../../tests/architecture_tool/README.md)
cover the generator and export.
