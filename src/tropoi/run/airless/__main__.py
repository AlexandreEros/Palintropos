"""``python -m tropoi.run.airless {show-config,run} (--preset NAME [...] | --config FILE)
[--steps N | --duration SECONDS]``, also ``tropoi run airless [--show-config] ...``.

Runs execute on CUDA (CuPy); there is no CPU fallback. CPU execution is for
verification: the hidden ``--device cpu`` or
``TROPOI_EXECUTION=cpu-verification`` (:mod:`tropoi.run.execution`).

Without ``--steps``/``--duration`` a run iterates cycles until periodic or a
cycle limit (``AirlessRun.run``). With one of them it is a finite transient
at the production timestep that stops at that point (``AirlessRun.run_transient``).
"""
from __future__ import annotations

import argparse
import json
import sys

from tropoi.run.airless.config import PRESETS, load_config_document

#: options that build or adjust a preset; a saved document is used as saved
_PRESET_OPTIONS = ("points", "grid", "model", "geothermal_flux", "depth", "top_thickness",
                   "max_stretch", "cycle_steps", "spinup_cycle_steps", "experiment_id")


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


def add_arguments(ap):
    """The options shared by ``python -m tropoi.run.airless`` and ``tropoi run airless``."""
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=sorted(PRESETS))
    source.add_argument("--config", metavar="FILE",
                        help="saved configuration JSON: the configuration itself or an "
                             "envelope holding it under \"config\" (e.g. a run's config.json); "
                             "used exactly as saved")
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
    stop = ap.add_mutually_exclusive_group()
    stop.add_argument("--steps", type=int,
                      help="finite transient: this many steps at dt = cycle_s/cycle_steps "
                           "from the initial state, then stop (no periodicity test)")
    stop.add_argument("--duration", type=float, metavar="SECONDS",
                      help="finite transient: run until this simulated time (the first step "
                           "reaching it); the actual final time is reported")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes (blocks of columns); 1 per GPU")
    # Runs execute on CUDA. CPU execution is for verification (tests/README.md):
    # --device cpu or TROPOI_EXECUTION=cpu-verification; hidden from --help.
    ap.add_argument("--device", choices=("cuda", "cpu"), help=argparse.SUPPRESS)
    ap.add_argument("--max-wall-s", type=float)
    ap.add_argument("--allow-dirty", action="store_true")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tropoi.run.airless")
    ap.add_argument("command", choices=("show-config", "run"))
    add_arguments(ap)
    return execute(ap.parse_args(argv), ap)


def execute(args, ap) -> int:
    """Show or run the configured experiment (``args.command``)."""
    source_info = None
    if args.config is not None:
        if not args.config.strip():
            ap.error("--config needs a file path")
        given = [n for n in _PRESET_OPTIONS if getattr(args, n) not in (None, [])]
        if args.no_spinup:
            given.append("no_spinup")
        if given:
            ap.error("--config runs the saved configuration as saved; these options only "
                     "adjust presets: " + ", ".join("--" + g.replace("_", "-") for g in given))
        try:
            cfg, source_info = load_config_document(args.config)
        except (OSError, ValueError) as exc:
            ap.error(f"--config {args.config}: {exc}")
    else:
        cfg = _config(args)
    transient = args.steps is not None or args.duration is not None
    if transient and args.max_wall_s is not None:
        ap.error("a transient stops at --steps/--duration; --max-wall-s applies to periodic runs")
    plan = None
    if transient:
        from tropoi.run.airless.experiment import transient_steps
        try:
            steps, request = transient_steps(cfg, steps=args.steps, duration_s=args.duration)
        except ValueError as exc:
            ap.error(str(exc))
        plan = {"mode": "transient", **request}
    if args.command == "show-config":
        out = {"config": cfg.to_dict(), "config_sha256": cfg.sha256()}
        if source_info is not None:
            out["config_source"] = source_info
        if plan is not None:
            out["run_plan"] = plan
        print(json.dumps(out, indent=1))
        return 0
    if not args.out:
        ap.error("run needs --out")
    from tropoi.run.airless.experiment import AirlessRun
    from tropoi.run.execution import ExecutionError, resolve_backend
    try:
        backend = resolve_backend(device=args.device)
    except ExecutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run = AirlessRun(cfg, args.out, workers=args.workers, backend=backend,
                     allow_dirty=args.allow_dirty, config_source=source_info)
    if transient:
        result = run.run_transient(steps, request=request)
    else:
        result = run.run(max_wall_s=args.max_wall_s)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
