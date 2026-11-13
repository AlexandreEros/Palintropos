# Architecture evidence

Observations of the running code that the [interactive atlas](../current/interactive.html)
joins with the static graphs, and the measurements the
[lifecycle proposal](../LIFECYCLE_MIGRATION_PROPOSAL.md) cites. Two kinds,
kept apart on purpose:

| Directory | What | Made by | Joined into the atlas |
|---|---|---|---|
| `runtime/` | Call relations, boundary data and state changes of one named scenario (trace schema 2) | `python -m tools.architecture.runtime_capture` | yes, every `*.json`, each with its own revision |
| `profile/` | Synchronized GPU timing of the radiative pilot | `tools/colab/radiative_pilot.py profile --mode sync` in a separate process, no tracer | no (cited by the proposal) |

A trace's Python wall times include tracing overhead and asynchronous GPU
launches; they are never GPU execution times. The profile's times are
device-synchronized around each instrumented call (which itself perturbs the
pipeline) and are not throughput either. Never mix the two.

## Current captures

All at source commit `08e6f22` with a clean `src/tropoi` and the same
LF-normalized content hash as `../current/metrics.json` (the atlas says so per
trace); `tools/` was dirty (this extension). GPU: NVIDIA GeForce MX110, CUDA
11.8, CuPy 13.4.0; Python 3.12.12, NumPy 2.2.6; Windows 11.

| File | Lifecycle | Verification recorded in the trace |
|---|---|---|
| `runtime/setup-cpu.json` | configuration, validation, CPU construction, `PlanetSpec`, terrain sampling | config round trip keeps `config_sha256`; W5 ideal sphere exact; no CuPy |
| `runtime/radiative-tiny-lifecycle.json` | construction, RK4 startup, stepping, physics coupling, daily output, checkpoint, resume | resumed run equals the uninterrupted run **bitwise** after 3 steps |
| `runtime/custom-swe-lifecycle.json` | custom IC over GPU-projected terrain, save, reload, custom run, open | capsule opens, 2 frames |
| `runtime/cli-swe-williamson2-tiny.json` | public CLI preset run | run completed |
| `runtime/saved-run-cpu.json` | saved BVE capsule and HS fixture: inspect, reduce, figures | no CuPy import, fresh process |
| `runtime/saved-run-view.json` | saved PE capsule: 3-D page, snapshot figure | page without CuPy; figure imports CuPy (recorded, not a failure) |
| `profile/profile-earth-tiny-sync.{json,txt}` | 36 synchronized steps of the heated Earth at `tiny` | same `config_sha256` as the lifecycle trace |

## Reproducing

From the repository root with the development requirements, `PYTHONPATH=src`
(PowerShell: `$env:PYTHONPATH = "src"`). Scratch output goes to the system
temporary directory unless `--scratch` is given; nothing is written into
`runs/` or a production run directory.

```bash
python -m tools.architecture.runtime_capture --list
```

```bash
python -m tools.architecture.runtime_capture --scenario setup-cpu --measure-overhead --out docs/architecture/evidence/runtime/setup-cpu.json
```

```bash
python -m tools.architecture.runtime_capture --scenario radiative-tiny-lifecycle --measure-overhead --out docs/architecture/evidence/runtime/radiative-tiny-lifecycle.json
```

```bash
python -m tools.architecture.runtime_capture --scenario custom-swe-lifecycle --measure-overhead --out docs/architecture/evidence/runtime/custom-swe-lifecycle.json
```

The saved-run scenarios verify which modules get imported, so they must run in
a fresh process **without** `--measure-overhead` (its earlier runs would import
CuPy first; the driver then reports the claim as not determinable):

```bash
python -m tools.architecture.runtime_capture --scenario saved-run-cpu --out docs/architecture/evidence/runtime/saved-run-cpu.json
```

```bash
python -m tools.architecture.runtime_capture --scenario saved-run-view --out docs/architecture/evidence/runtime/saved-run-view.json
```

```bash
python -m tools.architecture.runtime_capture --scenario cli-swe-williamson2-tiny --out docs/architecture/evidence/runtime/cli-swe-williamson2-tiny.json --boundary "tropoi.run.swe.*" --boundary "tropoi.cli.main._cmd_run*" --boundary "tropoi.temporal.models.*" --module tropoi.cli.main -- run swe --scenario williamson2 --lmax 10 --resolution 3 --days 0.25 --n-snapshots 2 --no-plots --out /tmp/palintropos-atlas-cli-runs
```

The separate timing profile (GPU):

```bash
python tools/colab/radiative_pilot.py profile --case earth --resolution tiny --steps 36 --warm-steps 6 --mode sync --scratch /tmp/palintropos-atlas-scratch --out /tmp/palintropos-atlas-profile
```

Then regenerate the snapshot so the atlas embeds the new traces:

```bash
python -m tools.architecture.generate
```

## Not yet observed

Unobserved is not unused. These workflows have no trace yet; the commands are
the smallest known runs (not executed for this capture set):

* BVE preset through the CLI, e.g. `--module tropoi.cli.main -- run bve
  --preset rh4 --lmax 10 --days 0.25 --n-snapshots 2 --no-plots --out <scratch>`
  (check `tropoi run bve --help` for the flags of the checkout).
* PE preset with prepared terrain: `tropoi terrain …` to prepare, then
  `run pe --scenario orographic_thermal_wave --terrain <dir> --lmax 10
  --days 0.1 --no-plots --out <scratch>`.
* The Jupiter radiative configuration (do not run the full experiment): the
  pilot accepts `--case jupiter --resolution tiny`; wrap it with
  `--module tools.colab.radiative_pilot -- profile --case jupiter --resolution
  tiny --steps 2 --warm-steps 1 --mode cprofile --scratch <s> --out <o>`
  (`cprofile` replaces the tracer's hooks, so trace the `nvtx` or `sync` mode
  instead, and note that `sync` inserts synchronizations of its own).
* The `gen` CLI demo planet (`planet_viewer`), the only reader of the
  procedural elevation every `Planet.generate` builds.
* Anything in worker threads or subprocesses (the recorder sees only the
  scenario thread).

## Limits of a trace

Stated in every trace under `provenance.capture.limits`: one boundary record
per (function, phase), 300 in total, 24 attributes and 6 items per described
object, strings cut at 160 characters, CPU arrays fingerprinted up to 4096
bytes, device arrays never read. Mutations are reported as rebinding, value,
metadata, buffer or fingerprint changes; a device or large array that is the
same object before and after is listed as *unverified*. Device events count
only calls visible to the profiler hook and are a lower bound. Units and axis
meanings are never inferred.
