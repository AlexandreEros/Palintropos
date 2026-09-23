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
| `history/numerics-physics-layout/` | **Frozen.** Source commit `f879851`: the original layout, with implementation modules in `tropoi.numerics`, `tropoi.physics`, `tropoi.planet`, and `tropoi.viz`. Never regenerated; `--check` verifies it (see its README). |
| `history/spatial-temporal-representation-layout/` | **Frozen.** Source commit `47dec90`: the same code reorganized into `tropoi.spatial`, `tropoi.temporal`, and `tropoi.representation`, with the old paths kept as aliases (see *Compatibility paths* in `docs/ARCHITECTURE.md`). Never regenerated; `--check` verifies it (see its README). |
| `current/` | **Live.** Default output of the generator; always describes the committed `src/tropoi`. CI fails any push where it is stale (see *Automated generation*). |

`current/` and `history/spatial-temporal-representation-layout/` hold identical
graphs when `current/` is introduced, because `src/` has not changed since
`47dec90`. They differ only in the README and metrics provenance. The first
source change after that updates `current/` and leaves the frozen snapshot as
it is.

Snapshots are named after what they capture, here the package layout, rather
than after the work that produced them.

### Adding a frozen snapshot

To record a milestone, render a committed source commit into a new
`history/` directory. Name it after what the snapshot captures:

```bash
python -m tools.architecture.generate --out docs/architecture/history/<name> --title <Title> --source-commit <rev> --freeze
```

`--source-commit` refuses unless `src/tropoi` in the working tree equals `<rev>`.
`--freeze` records the generator's commit and needs the generator to be
committed. After that, the generator refuses to overwrite the directory. Don't
copy `current/` instead: the copy would carry no provenance.

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

## Automated generation

`.github/workflows/architecture.yml` runs on every push to any branch and can
also be started by hand from the Actions tab. It checks the graphs; it does
not commit them.

1. It checks out the pushed commit and sets up Python 3.12, the minor version
   recorded in the toolchain.
2. It installs only the `grimp` and `networkx` pins, which it reads from
   `requirements-dev.txt`. The generator reads `src/` as text and never imports
   `tropoi`, so it needs neither CuPy nor a GPU.
3. It runs `python -m tools.architecture.generate --check`. The job fails if
   any file in `current/` differs from what the pushed source produces.
4. On failure, it regenerates `current/` and attaches it to the run as the
   `architecture-current` artifact.

When a change to `src/` alters the structure, regenerate before pushing and
commit the result with the change:

```bash
python -m tools.architecture.generate
```

```bash
git add docs/architecture/current
```

A red `architecture` check on a push therefore means the committed graphs don't
match the committed source. To fix it, run the command above, or unpack the
artifact over `docs/architecture/current/`, and commit.

CI doesn't commit the graphs itself. A bot commit on every push would put a
second commit after each source change, fail on protected branches, and
trigger the workflow again. Committing the graphs with the change keeps each
commit's source and its graphs together.

The frozen `history/` snapshots are not checked in CI. Their `--check` needs
a checkout at their own source commit, and nothing regenerates them anyway.
