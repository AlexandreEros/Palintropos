"""Command line for radiative–dynamical experiments.

    python -m tropoi.run.radiative run    --preset jupiter-schneider-liu --run-dir DIR
                                          [--days N] [--dt S] [--until-day N]
                                          [--max-wall-hours H] [--adiabatic] [--allow-dirty]
    python -m tropoi.run.radiative column --preset jupiter-schneider-liu
    python -m tropoi.run.radiative show-config --preset earth-gray-dry

``column`` solves the single-column radiative(-convective) equilibrium of a
preset on the CPU and prints it; ``--adiabatic`` runs the same dynamical
configuration with the column physics switched off.

Exit codes: 0 ok; 2 refusal (configuration or code mismatch); 4 run aborted.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from tropoi.run.radiative.config import PRESETS, RadiativeConfig


def _config(args) -> RadiativeConfig:
    if getattr(args, "config", None):
        cfg = RadiativeConfig.from_dict(json.loads(pathlib.Path(args.config).read_text()))
    else:
        cfg = PRESETS[args.preset]()
    changes = {k: getattr(args, k) for k in ("days", "dt", "experiment_id")
               if getattr(args, k, None) is not None}
    return cfg.with_(**changes) if changes else cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tropoi.run.radiative")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "column", "show-config"):
        p = sub.add_parser(name)
        p.add_argument("--preset", choices=sorted(PRESETS), default="jupiter-schneider-liu")
        p.add_argument("--config", help="JSON configuration (overrides --preset)")
        p.add_argument("--experiment-id")
        p.add_argument("--days", type=int)
        p.add_argument("--dt", type=float)
        if name == "run":
            p.add_argument("--run-dir", required=True)
            p.add_argument("--until-day", type=int)
            p.add_argument("--max-wall-hours", type=float)
            p.add_argument("--adiabatic", action="store_true")
            p.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    cfg = _config(args)

    if args.cmd == "show-config":
        print(json.dumps({"config": cfg.to_dict(), "config_sha256": cfg.sha256(),
                          "total_steps": cfg.total_steps}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "column":
        from tropoi.run.radiative.model import reference_column
        ref = reference_column(cfg)
        print(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                          for k, v in ref.items()}, indent=2, sort_keys=True))
        return 0

    from tropoi.run.radiative.experiment import CheckpointError, RadiativeRun, RunError
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsStateError
    try:
        run = RadiativeRun(cfg, pathlib.Path(args.run_dir), physics=not args.adiabatic,
                           allow_dirty=args.allow_dirty)
    except (RunError, CheckpointError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    wall = None if args.max_wall_hours is None else args.max_wall_hours * 3600.0
    try:
        res = run.run(args.until_day, max_wall_seconds=wall)
    except PrimitiveEquationsStateError as exc:
        print(f"ABORTED (validation) at day {run.day:.3f}: {exc!r}")
        return 4
    print(json.dumps(res, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
