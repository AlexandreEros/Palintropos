"""Command line for the Held–Suarez experiment (the Colab notebook calls this).

    python -m tropoi.run.held_suarez run       --preset production --run-dir DIR [--until-day N]
                                               [--max-wall-hours H] [--backup-dir D] [--dt S]
    python -m tropoi.run.held_suarez benchmark --preset production --work-dir DIR
                                               [--steps 200] [--max-hours H] [--deadline ISO]
    python -m tropoi.run.held_suarez report    --run-dir DIR --out DIR [--reference series.npz]
                                               [--reference-config JSON] [--reference-b2 JSON]
                                               [--tier-a JSON]
    python -m tropoi.run.held_suarez verify    --run-dir DIR
    python -m tropoi.run.held_suarez show-config --preset production [--dt S]

Exit codes: 0 ok; 2 usage/refusal (config or code mismatch, corrupt
checkpoint); 3 benchmark over budget (DECISION NEEDED); 4 run aborted.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from tropoi.run.held_suarez.config import PRESETS, HeldSuarezConfig

REPO = pathlib.Path(__file__).resolve().parents[4]


def _config(args) -> HeldSuarezConfig:
    if getattr(args, "config", None):
        cfg = HeldSuarezConfig.from_dict(json.loads(pathlib.Path(args.config).read_text()))
    else:
        kw = {}
        if args.experiment_id:
            kw["experiment_id"] = args.experiment_id
        cfg = PRESETS[args.preset](**kw)
    if getattr(args, "dt", None):
        cfg = cfg.with_(dt=float(args.dt))
    return cfg


def _add_cfg(p):
    p.add_argument("--preset", choices=sorted(PRESETS), default="production")
    p.add_argument("--config", help="JSON configuration (overrides --preset)")
    p.add_argument("--experiment-id")
    p.add_argument("--dt", type=float, help="time step (s); must divide one day")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tropoi.run.held_suarez")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    _add_cfg(p)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--until-day", type=int)
    p.add_argument("--max-wall-hours", type=float)
    p.add_argument("--backup-dir")
    p.add_argument("--backup-every-days", type=int, default=10)
    p.add_argument("--allow-dirty", action="store_true")
    p = sub.add_parser("benchmark")
    _add_cfg(p)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-hours", type=float)
    p.add_argument("--deadline")
    p.add_argument("--allow-dirty", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference")
    p.add_argument("--reference-config",
                   default=str(REPO / "docs" / "held_suarez" / "REFERENCE_CONFIG.json"))
    p.add_argument("--reference-b2", help="default: b2.json beside --reference")
    p.add_argument("--tier-a")
    p = sub.add_parser("verify")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("show-config")
    _add_cfg(p)
    args = ap.parse_args(argv)

    if args.cmd == "show-config":
        cfg = _config(args)
        print(json.dumps({"config": cfg.to_dict(), "config_sha256": cfg.sha256(),
                          "total_steps": cfg.total_steps, "effective_truncation": cfg.retained_cut},
                         indent=2, sort_keys=True))
        return 0

    if args.cmd == "report":
        from tropoi.run.held_suarez.report import build_report, write_report
        b2 = args.reference_b2
        if b2 is None and args.reference:
            b2 = str(pathlib.Path(args.reference).with_name("b2.json"))
        rep = build_report(pathlib.Path(args.run_dir), reference=args.reference and pathlib.Path(args.reference),
                           reference_config=pathlib.Path(args.reference_config),
                           reference_b2=b2 and pathlib.Path(b2),
                           tier_a_evidence=args.tier_a and pathlib.Path(args.tier_a))
        j, m = write_report(rep, pathlib.Path(args.out))
        print(m.read_text(encoding="utf-8"))
        print(f"wrote {j} and {m}")
        return 0

    if args.cmd == "verify":
        from tropoi.run.held_suarez.checkpoint import (CheckpointError, list_checkpoints,
                                                       read_checkpoint)
        found = list_checkpoints(pathlib.Path(args.run_dir) / "checkpoints")
        if not found:
            print("no checkpoints")
            return 2
        for step, path in found:
            try:
                _, meta = read_checkpoint(path)
            except CheckpointError as exc:
                print(f"CORRUPT {path.name}: {exc}")
                return 2
            print(f"OK {path.name}: day {meta['day']}, {len(meta['arrays'])} arrays verified, "
                  f"config {meta['config_sha256'][:12]}, commit {meta['code']['git_commit']}")
        return 0

    from tropoi.run.held_suarez.experiment import HeldSuarezRun, RunError, benchmark
    cfg = _config(args)
    if args.cmd == "benchmark":
        res = benchmark(cfg, pathlib.Path(args.work_dir), steps=args.steps,
                        max_hours=args.max_hours, deadline=args.deadline,
                        allow_dirty=args.allow_dirty)
        print(json.dumps(res, indent=2, sort_keys=True))
        print(f"BENCHMARK {cfg.experiment_id}: {res['t_step_mean_s']:.4f} s/step, "
              f"projected {res['projected_hours']:.2f} h for {cfg.days} days -> {res['verdict']}")
        if res["verdict"] != "within_budget":
            print("DECISION NEEDED: the projected wall time exceeds the allowed budget "
                  f"({'; '.join(res['reasons'])}). Options: a larger GPU, a later deadline, "
                  "or faster transforms. Resolution is never lowered automatically.")
            return 3
        return 0

    from tropoi.run.held_suarez.checkpoint import CheckpointError
    try:
        run = HeldSuarezRun(cfg, pathlib.Path(args.run_dir), allow_dirty=args.allow_dirty,
                            backup_dir=args.backup_dir and pathlib.Path(args.backup_dir))
    except (RunError, CheckpointError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    wall = None if args.max_wall_hours is None else args.max_wall_hours * 3600.0
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsStateError
    try:
        res = run.run(args.until_day, max_wall_seconds=wall,
                      backup_every_days=args.backup_every_days)
    except PrimitiveEquationsStateError as exc:
        print(f"ABORTED (validation) at day {run.day:.3f}: {exc!r}")
        return 4
    except RunError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps(res, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
