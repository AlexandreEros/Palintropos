"""Regenerate the committed Held–Suarez climate-figure fixture (needs CUDA).

The fixture is a real production run of exactly
``development_config(days=4, retained=True)`` (T21 retained, stored
``l_max`` 22, L10, 32 x 64, dt 1200 s, ``spinup_days`` 0, one state per day,
two 2-day statistics blocks), produced by :class:`HeldSuarezRun` from a clean
source tree, so its ``statistics.npz`` comes from the production
``daily_sample`` -> ``OnlineStatistics`` path on the GPU. The CPU CI test
``tests/test_held_suarez_climate_fixture.py`` reduces its saved days 1-4 with
the host synthesis and compares every statistic both paths compute.

Written to ``--out`` (default ``tests/fixtures/hs-T21L10-dev4``):

* the run's own files: ``config.json``, ``events.jsonl``, ``series.json``,
  ``progress.json``, ``statistics.npz``, ``states/state-d0000{1..4}.npy``
  (``checkpoints/`` are left out: they only serve resumption);
* ``gpu_daily_samples.npz``: the zonal fields of each day's production
  ``daily_sample`` (captured by the run's ``sample_hook``, days 0-4), the
  per-state inputs of the block sums;
* ``generation.json``: the commands, source and environment identity, the
  repeat-run determinism check, and the MEASURED gridpoint differences
  between the GPU synthesis (``model.wind_on_state_grid`` /
  ``temperature_on_state_grid``) and the host synthesis on the same states;
* ``SHA256SUMS`` over every file above.

    python -m tools.held_suarez.climate_fixture [--out DIR] [--repeat 2]

Run it from a clean checkout (the run refuses uncommitted changes under
``src/``). It takes a few minutes on a laptop GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "tests" / "fixtures" / "hs-T21L10-dev4"
RUN_FILES = ("config.json", "events.jsonl", "series.json", "progress.json",
             "statistics.npz")
ZONAL = ("u", "v", "T", "TsTs", "usus", "vsTs", "usvs")


def fixture_config():
    from tropoi.run.held_suarez.config import development_config
    return development_config(days=4, retained=True)


def run_once(work: pathlib.Path) -> tuple[dict, float]:
    """One production run into ``work``; the per-day zonal samples."""
    from tropoi.run.held_suarez.experiment import HeldSuarezRun
    captured: dict[int, dict] = {}

    def hook(day, sample):
        captured[int(day)] = {k: np.array(sample[k], dtype=np.float64)
                              for k in ZONAL}

    started = time.perf_counter()
    run = HeldSuarezRun(fixture_config(), work, allow_dirty=False,
                        sample_hook=hook, log=lambda message: None)
    result = run.run()
    if result["status"] != "completed":
        raise SystemExit(f"run did not complete: {result}")
    return captured, time.perf_counter() - started


def _states(run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted((run_dir / "states").glob("state-d*.npy"))


def measure_synthesis(run_dir: pathlib.Path) -> dict:
    """GPU (model) versus host synthesis of u, v, T on every saved state."""
    import cupy as cp

    from tropoi.run.held_suarez.model import build_model
    from tropoi.spatial.states.primitive_equations import \
        PrimitiveEquationsState
    from tropoi.spatial.transforms.host_synthesis import (
        synthesize_latlon, synthesize_wind_latlon)

    cfg = fixture_config()
    model = build_model(cfg)
    grid = model.grid
    lat = np.degrees(np.asarray(grid._latitudes_np))
    lon = np.degrees(np.asarray(grid._longitudes_np))
    K, L = cfg.nlev, cfg.l_max
    per_state = []
    for path in _states(run_dir):
        frame = np.load(path)
        state = PrimitiveEquationsState(cp.asarray(frame))
        gu, gv = (cp.asnumpy(a).reshape(K, lat.size, lon.size)
                  for a in model.wind_on_state_grid(state))
        gT = cp.asnumpy(model.temperature_on_state_grid(state)).reshape(
            K, lat.size, lon.size)
        hu = np.empty_like(gu)
        hv = np.empty_like(gv)
        hT = np.empty_like(gT)
        for k in range(K):
            hT[k] = synthesize_latlon(frame[2 * K + k], lat, lon)
            hu[k], hv[k] = synthesize_wind_latlon(frame[k], frame[K + k],
                                                  cfg.radius, lat, lon)
        rows = frame[:3 * K]
        nonzero = np.abs(rows) > 0.0
        entry = {"file": path.name,
                 "max_nonzero_degree": int(np.nonzero(
                     nonzero.any(axis=(0, 2)))[0][-1]),
                 "max_nonzero_order": int(np.nonzero(
                     nonzero.any(axis=(0, 1)))[0][-1]),
                 "max_abs_buffer_degree": float(np.abs(rows[:, L, :]).max())}
        for name, g, h in (("u", gu, hu), ("v", gv, hv), ("T", gT, hT)):
            diff = float(np.abs(g - h).max())
            scale = float(np.abs(h).max())
            entry[name] = {"max_abs_diff": diff, "max_abs_field": scale,
                           "relative": diff / scale if scale else 0.0}
        per_state.append(entry)
    summary = {name: max(e[name]["relative"] for e in per_state)
               for name in ("u", "v", "T")}
    return {"what": "max |GPU - host| over the 32 x 64 x 10 grid of each "
                    "saved state; relative = that / max |host field|",
            "per_state": per_state, "max_relative": summary}


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--repeat", type=int, default=2,
                        help="independent runs compared bit for bit (>= 1)")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing fixture directory")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to replace it")
        return 2

    from tropoi.run.held_suarez.experiment import (code_identity,
                                                   environment_identity)
    code = code_identity()
    if code.get("git_dirty") or code.get("git_commit") is None:
        print(f"refusing: source tree is not a clean commit ({code})")
        return 2

    with tempfile.TemporaryDirectory(prefix="hs-fixture-") as tmp:
        runs, samples, walls = [], [], []
        for n in range(max(1, args.repeat)):
            work = pathlib.Path(tmp) / f"run{n}"
            captured, wall = run_once(work)
            runs.append(work)
            samples.append(captured)
            walls.append(wall)
            print(f"run {n}: {wall:.1f} s, {len(_states(work))} states")
        first = runs[0]
        repeat = []
        for other in runs[1:]:
            same_states = all(
                np.array_equal(np.load(a), np.load(b))
                for a, b in zip(_states(first), _states(other)))
            with np.load(first / "statistics.npz") as sa, \
                    np.load(other / "statistics.npz") as sb:
                same_stats = all(np.array_equal(sa[k], sb[k], equal_nan=True)
                                 for k in sa.files)
            repeat.append({"states_bit_identical": bool(same_states),
                           "statistics_bit_identical": bool(same_stats)})
        synthesis = measure_synthesis(first)

        if out.exists():
            shutil.rmtree(out)
        (out / "states").mkdir(parents=True)
        for name in RUN_FILES:
            shutil.copy2(first / name, out / name)
        for path in _states(first):
            shutil.copy2(path, out / "states" / path.name)
    days = sorted(samples[0])
    np.savez(out / "gpu_daily_samples.npz", day=np.asarray(days),
             **{k: np.stack([samples[0][d][k] for d in days]) for k in ZONAL})
    cfg = fixture_config()
    generation = {
        "fixture": "Held–Suarez climate-figure comparison fixture (#17)",
        "config_call": "development_config(days=4, retained=True)",
        "config": cfg.to_dict(),
        "config_sha256": cfg.sha256(),
        "command": "python -m tools.held_suarez.climate_fixture "
                   f"--repeat {max(1, args.repeat)}",
        "code": code,
        "environment": environment_identity(),
        "wall_seconds_per_run": walls,
        "repeat_runs": repeat,
        "gpu_daily_samples": "zonal fields of each day's production "
                             "daily_sample (sample_hook), days "
                             f"{days[0]}-{days[-1]}, shape (day, nlat, nlev)",
        "gpu_vs_host_synthesis": synthesis,
        "excluded": "checkpoints/ (resumption only)",
    }
    (out / "generation.json").write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    files = sorted(p for p in out.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(out).as_posix()}\n"
                for p in files), encoding="utf-8", newline="\n")
    print(json.dumps({"out": str(out), "repeat": repeat,
                      "max_relative": synthesis["max_relative"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
