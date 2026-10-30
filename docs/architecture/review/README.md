# Architecture reform review

The current working tree is based on commit
`5c8d7e8539be07e3e6c1f808380510f18d84a965`. The changes are uncommitted;
current source/generator status and per-file content hashes are in
`../current/metrics.json`. A base commit does not identify the dirty source.

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

| Measurement | Base commit snapshot | After compatibility cleanup | After ownership review |
|---|---:|---:|---:|
| Discovered modules | 177 | 122 | 128 |
| Modules included in import graph | 156 | 102 | 108 |
| Module import edges | 411 | 339 | 350 |
| Nontrivial module cyclic groups | 2 | 3 | 0 |
| Modules in cyclic groups | 14 | 7 | 0 |

The extra modules after ownership review represent explicit owners for shared
code. These descriptive numbers are not quality scores. The base snapshot
includes compatibility packages and re-export paths; its grouping differs.
Full baseline measurements are `base-commit.metrics.json` and
`after-cleanup.metrics.json`; current results are in `../current/metrics.md`
and `../current/metrics.json`. Static graphs omit some runtime dispatch;
zero static cycles is not a claim that every runtime interaction is acyclic.

## Validation and compatibility

* Full CPU suite: **844 passed, 428 skipped**. This includes saved-run,
  atmosphere export, snapshot plotting, custom-state and terrain persistence,
  lifecycle, overwrite cleanup, CPU-import, and architecture-tool checks.
* Built a distribution wheel, installed it into a separate environment,
  reused the existing CPU dependencies, and ran outside the checkout.
  Verified the imported package came from that installation.
* Installed `tropoi list presets` and `tropoi inspect ... --snapshot -1
  --field zeta` passed; inspection reported the expected read-only 22×22
  coefficients and valid conventions. Both committed published runs opened.
* Installed-package initial-state preparation/save/load and
  `PreparedPlanet.save/load` passed using synthetic coefficients on the CPU.
  This validates persistence and delegation, not numerical terrain projection.
* Wheel contents include the new owners and CUDA sources and exclude retired
  execution modules. Generated reports pass `--check`; export includes actual
  metrics, provenance, diagram sources, baseline results, and checksums.
* CUDA numerical integration, physical-grid projection and GPU-dependent plot
  reconstruction could not run on this CPU-only host. A browser was not used
  for manual interaction with the exported globe in this review.

Intentional breaks from the earlier cleanup remain: retired legacy Python
paths, aeolus/psx-* commands, unused aliases and standalone parsers. The
internal `cli.bve`, `cli.swe`, `cli.pe`, and `cli.run_lifecycle` modules are
also relocated; repository callers use the run-layer owners. Supported saved
formats and the named convenience methods remain available.

No remaining code blocker was found in these paths. GPU verification is the
outstanding validation limitation. The changes remain uncommitted and have
not been pushed or merged; they are ready for diff review. A library/application
split and namespace rename belong in a separate proposal and PR.
