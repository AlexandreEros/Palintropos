# Generated architecture documentation

Everything below this directory, except this file, is written by
`tools/architecture` (`python -m tools.architecture.generate`). The generator
reads the source of `src/tropoi` and records generic structural facts: module
imports (grimp), classes, inheritance, and methods (`ast`), and statically
resolvable calls (`ast`). It also records descriptive graph metrics (networkx).
It encodes no expectation about how the packages should depend on each other.

The hand-authored `docs/class-structure.md` and `docs/call-structure.md` are
separate and are not replaced by these files.

## Layout

| Path | Status |
|---|---|
| `history/pre-sprint3/` | **Frozen.** Source commit `f879851`, immediately before the spatial/temporal/representation refactor. Never regenerated; `--check` verifies it (see its README). |
| `current/` | Not created yet. Default output of the generator for the checked-out tree. Intended for a CI staleness check (see below). |

Each snapshot's `README.md` states its provenance, the inclusion and exclusion
rules, the extraction rules, the limits of static call analysis, and the
definition of every metric. `metrics.md` has the values.

## Commands

```bash
pip install -r requirements-dev.txt
```

```bash
python -m tools.architecture.generate
```

```bash
python -m tools.architecture.generate --check
```

The first writes `docs/architecture/current/`. `--check` renders in memory and
exits 1 if any file on disk differs, is missing, or is unexpected, which is
the same check as

```bash
python -m tools.architecture.generate && git diff --exit-code docs/architecture/current
```

Output is byte-for-byte deterministic for a given source tree, generator
version, and pinned toolchain (Python minor version, grimp, networkx; see
`requirements-dev.txt`).
