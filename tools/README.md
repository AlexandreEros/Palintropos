# Repository tooling

`architecture/` reads source text to generate the
[architecture snapshots](../docs/architecture/README.md); run
`python -m tools.architecture.generate --check` for freshness and
`python -m tools.architecture.export --out PATH` to export a combined report
and checksum-bearing ZIP. It needs the pinned Python/grimp/networkx toolchain,
not CUDA. Generated paths remain tool-owned; frozen snapshots are not
regenerated against current source.

`held_suarez/` contains evidence/recovery/reference tools for the
[HS experiment](../docs/held_suarez/README.md): reference extraction, run
comparison, mass budgets, tier-A evidence and compact climate fixtures.
`browser_cdp.py` supports offline browser verification. These are developer
tools, separate from the installed `tropoi` CLI. Read their module usage
before writing output; preserve frozen inputs and production evidence.
