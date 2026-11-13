"""Command line for radiative–dynamical experiments.

    python -m tropoi.run.radiative run    --preset jupiter-schneider-liu --run-dir DIR
                                          [--days N] [--dt S] [--until-day N]
                                          [--max-wall-hours H] [--adiabatic] [--allow-dirty]
                                          [--frictional-heating | --no-frictional-heating]
    python -m tropoi.run.radiative column --preset jupiter-schneider-liu
    python -m tropoi.run.radiative show-config --preset earth-gray-dry

Frictional heating returns the kinetic energy removed by drag and
hyperdiffusion as heat. The presets turn it on (an eddying run without it
fails the energy criteria, docs/radiation/README.md §6);
``--no-frictional-heating`` gives the unheated configuration of the
recorded runs. A ``--config`` file means what it says unless one of the two
flags overrides it. Each choice is a different configuration hash.

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
    heating = getattr(args, "frictional_heating", None)
    changes = {k: getattr(args, k) for k in ("days", "dt", "experiment_id")
               if getattr(args, k, None) is not None}
    if getattr(args, "config", None):
        cfg = RadiativeConfig.from_dict(json.loads(pathlib.Path(args.config).read_text()))
        if heating is not None:
            changes["frictional_heating"] = heating
    else:
        # through the preset, so that its default experiment id follows the
        # heating (the ids of the recorded heated and unheated runs)
        cfg = PRESETS[args.preset](**({} if heating is None
                                      else {"frictional_heating": heating}))
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
        p.add_argument("--frictional-heating", action=argparse.BooleanOptionalAction,
                       default=None)
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
