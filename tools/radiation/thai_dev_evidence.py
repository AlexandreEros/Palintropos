"""Record a short THAI development run as evidence (CPU; no solver import).

    python tools/radiation/thai_dev_evidence.py runs/radiative/<run>
        [--uninterrupted runs/radiative/<control run>]
        [--evidence-dir docs/radiation/evidence]

Copies the small records of the run (``config.json``, ``events.jsonl``,
``series.json``) to ``<evidence-dir>/<run name>/`` and writes
``summary.json`` beside them: the daily series, the checkpoint and resume
events, and - with ``--uninterrupted`` - a comparison with a second run of
the same configuration and commit that was never stopped. For every day
both runs saved, the state and surface arrays are compared bit for bit, and
so are the daily series over the common days. An interrupted run that
resumes exactly is indistinguishable from the control.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

RECORDS = ("config.json", "events.jsonl", "series.json")
KEYS = ("t_min", "t_max", "max_abs_u", "max_wind_bottom", "mean_surface_temperature", "olr",
        "absorbed_stellar", "toa_net_down", "bottom_net_down", "sensible", "surface_net",
        "column_energy_tendency", "energy_rate", "energy_residual", "energy_total", "mean_ps")


def _events(run: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _table(series: dict) -> dict:
    days = series["day"]
    rows = {}
    for i, day in enumerate(days):
        row = {}
        for k in KEYS:
            if k not in series:
                continue
            values = series[k]
            j = i - (len(days) - len(values))          # rates have no day-0 entry
            if j >= 0:
                row[k] = values[j]
        rows[str(day)] = row
    return rows


def compare(run: pathlib.Path, control: pathlib.Path) -> dict:
    cfg_a = json.loads((run / "config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((control / "config.json").read_text(encoding="utf-8"))
    same = (cfg_a["config_sha256"] == cfg_b["config_sha256"]
            and cfg_a["code"].get("git_commit") == cfg_b["code"].get("git_commit"))
    out = {"control": control.name, "same_configuration_and_commit": same, "states": {}}
    worst = 0.0
    for path in sorted((control / "states").glob("*.npy")):
        other = run / "states" / path.name
        if not other.exists():
            continue
        a, b = np.load(other), np.load(path)
        identical = bool(a.shape == b.shape and np.array_equal(a, b))
        diff = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
        worst = max(worst, diff)
        out["states"][path.name] = {"bit_identical": identical, "max_abs_difference": diff}
    sa = json.loads((run / "series.json").read_text(encoding="utf-8"))
    sb = json.loads((control / "series.json").read_text(encoding="utf-8"))
    n = min(len(sa["day"]), len(sb["day"]))
    mismatched = []
    for k in sorted(set(sa) & set(sb)):
        offset_a, offset_b = len(sa["day"]) - len(sa[k]), len(sb["day"]) - len(sb[k])
        if sa[k][:n - offset_a] != sb[k][:n - offset_b]:
            mismatched.append(k)
    out["series_days_compared"] = n - 1
    out["series_keys_not_identical"] = mismatched
    out["all_bit_identical"] = bool(same and out["states"] and not mismatched and all(
        s["bit_identical"] for s in out["states"].values()))
    out["max_abs_state_difference"] = worst
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--uninterrupted")
    ap.add_argument("--evidence-dir", default="docs/radiation/evidence")
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    out_dir = pathlib.Path(args.evidence_dir) / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RECORDS:
        shutil.copyfile(run / name, out_dir / name)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    series = json.loads((run / "series.json").read_text(encoding="utf-8"))
    events = _events(run)
    summary = {
        "schema": "palintropos.radiation.thai_dev_run/1",
        "run": run.name, "experiment_id": cfg["config"]["experiment_id"],
        "config_sha256": cfg["config_sha256"], "git_commit": cfg["code"].get("git_commit"),
        "git_dirty": cfg["code"].get("git_dirty"),
        "longwave": cfg["config"]["longwave"], "shortwave": cfg["config"]["shortwave"],
        "radiation_interval_steps": cfg["config"]["radiation_interval"],
        "dt_seconds": cfg["config"]["dt"],
        "gas_constant": cfg["config"]["gas_constant"], "cp": cfg["config"]["cp"],
        "last_day": series["day"][-1],
        "events": [{k: e[k] for k in ("event", "day", "step", "checkpoint", "file", "time",
                                      "error") if k in e} for e in events],
        "resumed_at_days": [e["day"] for e in events if e["event"] == "resumed"],
        "completed": any(e["event"] == "completed" for e in events),
        "aborted": [e.get("error") for e in events if e["event"] == "aborted"],
        "daily": _table(series),
        "max_abs_energy_residual_w_m2": float(np.abs(series.get("energy_residual",
                                                                 [0.0])).max()),
    }
    if args.uninterrupted:
        summary["resume_check"] = compare(run, pathlib.Path(args.uninterrupted))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(f"{run.name}: day {summary['last_day']}, completed {summary['completed']}, "
          f"resumed at {summary['resumed_at_days']}, aborted {summary['aborted']}")
    for day, row in summary["daily"].items():
        print(f"  day {day:>2}: " + "  ".join(
            f"{k} {row[k]:.4g}" for k in ("t_min", "t_max", "max_abs_u", "olr",
                                          "absorbed_stellar", "mean_surface_temperature",
                                          "energy_residual") if k in row))
    if args.uninterrupted:
        rc = summary["resume_check"]
        print(f"  resume check vs {rc['control']}: all bit-identical "
              f"{rc['all_bit_identical']} ({len(rc['states'])} arrays, "
              f"{rc['series_days_compared']} days of series, max |difference| "
              f"{rc['max_abs_state_difference']:.3g})")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
