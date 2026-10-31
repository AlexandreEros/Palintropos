# Architecture reform review

The reform is reconciled with main at `6a03708` (PR #14) on branch
`reform/reconcile-main`. The analyzed source and generator are based on the commit recorded in
`../current/metrics.json`; source and generator trees
were clean when generated. Full provenance and per-file content hashes are
in `../current/metrics.json`. The original reform baseline was
`5c8d7e8539be07e3e6c1f808380510f18d84a965`.

The merge preserves the new terrain expressions/functions/file loading,
terrain CLI, orographic thermal-wave scenario, evolving Mars example and
published capsule, and atmosphere playback. The CLI conflict was resolved
by retaining both the run-owned preset registry and the new terrain command.
The added CPU-import coverage was migrated from the retired compatibility
test into `tests/test_layout.py`. Architecture outputs were regenerated
from the combined source.

## Ownership findings and changes

`run.custom` imported model builders, manifest notes, overwrite cleanup,
output-directory resolution, and the shared execution/provenance lifecycle
from the command modules. These serve Python execution as well as commands.
They now live in `run.{bve,swe,pe}.model`, `run.{bve,swe,pe}.execution`,
and `run.lifecycle`. `cli.main` parses arguments and dispatches to execution.

`representation.visual.snapshot` imported the three model builders from
CLI modules to reconstruct resources from saved configurations. It now
imports the run-layer model builders. This remains an explicit architectural
exception: orchestration owns construction, while plotting requests resources
lazily. Moving the import inside a function controls loading; relocation
establishes ownership. `snapshot.plot()` retains this behavior.

## What the cyclic groups did

All three paths existed before the compatibility cleanup. The cleanup's
updated graph exposed them; it did not introduce their functionality.

* **cli.main → run.custom → presets → cli.main:** the CLI executes saved
  states through the public custom API; that package re-exports preset
  export helpers, which obtained named settings from the command registry.
  The shared registry now belongs to `run.bve.presets`. The public re-exports
  remain; a Python preset export no longer depends on argument parsing.
* **custom.planet ↔ custom.prepare:** terrain preparation borrowed
  specification checks, planet construction and topography resolution from
  fluid-state preparation; SWE preparation accepts a `PreparedPlanet` to
  reuse resolved terrain coefficients. Shared helpers and `PreparationError`
  now live in `custom.resources`. The SWE-to-prepared-terrain relationship
  remains one-way. `PreparedPlanet.save/load`, preview, plot and view remain.
* **archive.held_suarez ↔ visual.atmosphere:** storage's explicit `view()`
  method calls the atmosphere renderer; the renderer imported the storage
  module for `NoSnapshotsError` and the statistics-only explanation. Those
  two shared availability definitions now belong to `archive.availability`.
  Their former imports from `archive.held_suarez` still work with the same
  object identity. `simulation.view()` continues to delegate through storage.

These are bounded ownership fixes. No physics, integration algorithm, stored
schema, library/application split, or namespace rename is part of them.

## Snapshot comparison

| Measurement | Original base | After cleanup | After ownership | Updated main | Reconciled |
|---|---:|---:|---:|---:|---:|
| Discovered modules | 177 | 122 | 128 | 180 | 131 |
| Modules included in import graph | 156 | 102 | 108 | 159 | 111 |
| Module import edges | 411 | 339 | 350 | 420 | 359 |
| Nontrivial module cyclic groups | 2 | 3 | 0 | 2 | 0 |
| Modules in cyclic groups | 14 | 7 | 0 | 15 | 0 |

The extra modules after ownership review represent explicit owners for shared
code. These descriptive numbers are not quality scores. The base snapshot
includes compatibility packages and re-export paths; its grouping differs.
Full baseline measurements are `base-commit.metrics.json` and
`after-cleanup.metrics.json`, `after-ownership.metrics.json`, and
`updated-main.metrics.json`; current results are in `../current/metrics.md`
and `../current/metrics.json`. Static graphs omit some runtime dispatch;
zero static cycles is not a claim that every runtime interaction is acyclic.

## Validation and compatibility

* Reconciled CPU suite: **971 passed, 435 skipped**. This includes saved-run,
  atmosphere export, snapshot plotting, custom-state and terrain persistence,
  lifecycle, overwrite cleanup, CPU-import, and architecture-tool checks.
* Built a distribution wheel, installed it into a separate environment,
  reused the existing CPU dependencies, and ran outside the checkout.
  Verified the imported package came from that installation.
* Installed `tropoi list presets` and `tropoi inspect ... --snapshot -1
  --field zeta` passed; inspection reported the expected read-only 22×22
  coefficients and valid conventions. All three committed published runs opened, including the new nine-state
  Mars PE run; installed `tropoi inspect` read its final temperature field.
  The installed `tropoi terrain prepare --help` command also passed.
* Installed-package initial-state preparation/save/load and
  `PreparedPlanet.save/load` passed using synthetic coefficients on the CPU.
  This validates persistence and delegation, not numerical terrain projection.
* Wheel contents include the new owners and CUDA sources and exclude retired
  execution modules. Generated reports pass `--check`; export includes actual
  metrics, provenance, diagram sources, baseline results, and checksums.
* CUDA numerical integration, physical-grid projection and GPU-dependent plot
  reconstruction could not run on this CPU-only host. The two new browser tests were attempted: Chromium navigation returned
  `net::ERR_BLOCKED_BY_ADMINISTRATOR` for the exported local HTML, so the
  viewer never loaded. The initial suite had 971 passes, 433 skips and these
  two failures. The final run explicitly disabled browser discovery with
  `TROPOI_BROWSER=/tmp/no-browser-in-this-managed-environment`, yielding the
  435 skips above. Browser tests and viewer code remain unchanged from main;
  offline playback still needs verification in an environment allowing local
  HTML navigation. This is an environment limitation, not evidence of a
  reform regression.

Intentional breaks from the earlier cleanup remain: retired legacy Python
paths, aeolus/psx-* commands, unused aliases and standalone parsers. The
internal `cli.bve`, `cli.swe`, `cli.pe`, and `cli.run_lifecycle` modules are
also relocated; repository callers use the run-layer owners. Supported saved
formats and the named convenience methods remain available.

No remaining code blocker was found in these paths. GPU verification and
browser playback are outstanding validation limitations. The reform and
main reconciliation are committed locally; nothing has been pushed or merged
into remote main. The branch is ready for diff review. A library/application
split and namespace rename belong in a separate proposal and PR.

## CI provenance correction

The first pushed branch passed the CPU tests, but failed the architecture
freshness job: generation-time HEAD, Git status, and the explicit source
commit differed when CI regenerated after the report commit. Only README.md
and metrics.json differed; the graphs were unchanged.

The check now retains the original provenance when both source and generator
content hashes and per-file inventories still match, and compares every
generated file. Regression coverage includes later documentation commits,
dirty generation followed by a commit, comment-only source/generator edits,
and modified diagram output. All 49 architecture-tool tests pass.
