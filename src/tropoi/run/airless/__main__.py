"""``python -m tropoi.run.airless {show-config,run} --preset NAME [...]``."""
from __future__ import annotations

import argparse
import json
import sys

from tropoi.run.airless.config import PRESETS


def _config(args):
    kw = {}
    if args.points:
        lat, lon = zip(*(tuple(float(x) for x in p.split(",")) for p in args.points))
        kw["columns"] = {"kind": "points", "lat_deg": list(lat), "lon_deg": list(lon)}
    elif args.grid:
        nlat, nlon = (int(x) for x in args.grid.split("x"))
        kw["columns"] = {"kind": "latlon", "nlat": nlat, "nlon": nlon}
    for name in ("depth", "top_thickness", "max_stretch", "cycle_steps"):
        value = getattr(args, name)
        if value is not None:
            kw[name] = value
    if args.no_spinup:
        kw["spinup_cycle_steps"] = None
    elif args.spinup_cycle_steps is not None:
        kw["spinup_cycle_steps"] = args.spinup_cycle_steps
    if args.model:
        kw["model"] = args.model
    if args.geothermal_flux is not None:
        kw["geothermal_flux"] = args.geothermal_flux
    if args.experiment_id:
        kw["experiment_id"] = args.experiment_id
    return PRESETS[args.preset](**kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tropoi.run.airless")
    ap.add_argument("command", choices=("show-config", "run"))
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out")
    ap.add_argument("--experiment-id")
    ap.add_argument("--points", nargs="*", help="LAT,LON pairs in degrees")
    ap.add_argument("--grid", help="NLATxNLON regular cell-centred grid")
    ap.add_argument("--model", choices=("TWO", "TOP", "BOT"), help="VPW99 regolith model")
    ap.add_argument("--geothermal-flux", type=float)
    ap.add_argument("--depth", type=float)
    ap.add_argument("--top-thickness", type=float)
    ap.add_argument("--max-stretch", type=float)
    ap.add_argument("--cycle-steps", type=int)
    ap.add_argument("--spinup-cycle-steps", type=int)
    ap.add_argument("--no-spinup", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes (blocks of columns); with --backend cupy, 1 per GPU")
    ap.add_argument("--backend", choices=("numpy", "cupy"), default="numpy",
                    help="execution backend; the mathematics is identical")
    ap.add_argument("--max-wall-s", type=float)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    cfg = _config(args)
    if args.command == "show-config":
        print(json.dumps({"config": cfg.to_dict(), "config_sha256": cfg.sha256()}, indent=1))
        return 0
    if not args.out:
        ap.error("run needs --out")
    from tropoi.run.airless.experiment import AirlessRun
    run = AirlessRun(cfg, args.out, workers=args.workers, backend=args.backend,
                     allow_dirty=args.allow_dirty)
    result = run.run(max_wall_s=args.max_wall_s)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
