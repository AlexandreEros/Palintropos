"""G-P: discretization ladders for the Mercury columns (Planet One).

Runs the six canonical columns of a preset at refined timesteps, grids and
depths, each to its own periodic state. It then estimates the error of every
G-P quantity with Richardson extrapolation and writes ``gp.json``. The
tolerances are those frozen in docs/validation/PLANET-ONE-GATES.md; this
script only applies them.

    python tools/planet_one/convergence.py OUTDIR [--preset mercury-vpw99]
        [--model TWO] [--geothermal-flux 0.02] [--workers 8] [--tables-only]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np

from tropoi.physics.constants import SIGMA_SB
from tropoi.run.airless.config import PRESETS
from tropoi.run.airless.experiment import AirlessRun

POINTS = {"lat_deg": [0.0, 0.0, 0.0, 0.0, 45.0, 85.0],
          "lon_deg": [0.0, 90.0, 180.0, 270.0, 0.0, 0.0]}
NAMES = ["0N 0E", "0N 90E", "0N 180E", "0N 270E", "45N 0E", "85N 0E"]
#: frozen G-P tolerances
TOL = {"T_max": 0.5, "T_min": 0.5, "t_max_frac": 2e-3, "T_2cm_mean": 0.2, "T_2cm_amp": 0.2,
       "T_10cm_mean": 0.2, "T_10cm_amp": 0.2, "T_30cm_mean": 0.2, "T_30cm_amp": 0.2,
       "T_bottom_mean": 0.2, "emitted_rel": 1e-3, "abs_conducted_rel": 1e-3}
# Ladders: (cycle_steps), (top_thickness, max_stretch), depth
DT_LADDER = [2160, 4320, 8640, 17280]
GRID_LADDER = [(2e-3, 1.1), (1e-3, 1.0488), (5e-4, 1.0241), (2.5e-4, 1.012)]
DEPTH_LADDER = [1.0, 2.0]
BASE = {"cycle_steps": 4320, "grid": (1e-3, 1.0488), "depth": 1.0}


def _cfg(preset, model, g, cycle_steps, grid, depth, tag):
    kw = dict(columns={"kind": "points", **POINTS}, cycle_steps=cycle_steps,
              top_thickness=grid[0], max_stretch=grid[1], depth=depth,
              spinup_cycle_steps=360, geothermal_flux=g)
    if model is not None:
        kw["model"] = model
    cfg = PRESETS[preset](**kw)
    return cfg.with_(experiment_id=f"{cfg.experiment_id}-gp-{tag}")


def _run(args):
    cfg_dict, outdir = args
    from tropoi.run.airless.config import AirlessConfig
    cfg = AirlessConfig.from_dict(cfg_dict)
    run = AirlessRun(cfg, outdir, workers=1, log=lambda s: None)
    return run.run()


def quantities(run_dir: pathlib.Path) -> dict:
    """The G-P quantities of a finished point run's retained cycle."""
    cfgd = json.loads((run_dir / "config.json").read_text())["config"]
    P = cfgd["cycle_s"]
    grid = np.load(run_dir / "grid.npz")
    zc = grid["centres"]
    ev = np.load(run_dir / "retained" / "every_step.npy")     # (steps, 1 + L, ncol)
    ints = np.load(run_dir / "retained" / "integrals.npz")
    steps = ev.shape[0]
    t = (np.arange(steps) + 1) / steps                         # cycle fraction
    out = {}
    for c, name in enumerate(NAMES):
        ts = ev[:, 0, c]
        i = int(np.argmax(ts))
        # parabolic refinement of the time of maximum
        if 0 < i < steps - 1:
            y0, y1, y2 = ts[i - 1], ts[i], ts[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        else:
            off = 0.0
        q = {"T_max": float(ts.max()), "T_min": float(ts.min()),
             "t_max_frac": float((t[i] + off / steps) % 1.0)}
        prof = ev[:, 1:, c]                                     # (steps, L)
        for depth, key in ((0.02, "2cm"), (0.10, "10cm"), (0.30, "30cm")):
            series = np.array([np.interp(depth, zc, row) for row in prof])
            q[f"T_{key}_mean"] = float(series.mean())
            q[f"T_{key}_amp"] = float(0.5 * (series.max() - series.min()))
        q["T_bottom_mean"] = float(prof[:, -1].mean())
        q["emitted"] = float(ints["emitted"][c] / P)
        q["abs_conducted"] = float(ints["abs_conducted"][c] / P)
        out[name] = q
    return out


def richardson(values):
    """Error estimate of values[-2] (the second finest) from the last three
    levels of a refinement ladder. With e1 = |Q1 - Q2| and e2 = |Q2 - Q3|,
    the observed order is p = log2(e1/e2) and the estimate is e2 + e2/(2^p - 1)
    (the difference to the finest level plus the finest level's own
    remaining error). When the order cannot be estimated (p <= 0.5, non-
    monotone, or e2 == 0) the estimate is the conservative 2 * e1."""
    q1, q2, q3 = values[-3:]
    e1, e2 = abs(q1 - q2), abs(q2 - q3)
    if e2 == 0.0:
        return {"error": 0.0 if e1 == 0.0 else 2 * e1, "order": None}
    p = math.log2(e1 / e2) if e1 > 0 else 0.0
    if p <= 0.5 or math.copysign(1, q1 - q2) != math.copysign(1, q2 - q3):
        return {"error": 2 * e1, "order": p}
    return {"error": e2 + e2 / (2 ** p - 1), "order": p}


def _scaled(key, q, name, err):
    if key == "emitted_rel":
        return err / q[name]["emitted"]
    if key == "abs_conducted_rel":
        return err / q[name]["abs_conducted"]
    return err


def analyse(ladder_quantities, production_index):
    """For a ladder whose ``production_index`` level is the candidate setting,
    use that level and the two finer ones."""
    sel = ladder_quantities[production_index:production_index + 3]
    if len(sel) < 3:
        raise ValueError("need two levels finer than the production level")
    table = {}
    for name in NAMES:
        rows = {}
        for key in TOL:
            base = key.replace("_rel", "")
            vals = [q[name][base] for q in sel]
            r = richardson([vals[0], vals[1], vals[2]])
            # Richardson estimates the error of the middle level; the
            # production level is the coarsest: use the difference to the
            # middle level plus the middle level's estimate.
            err_prod = abs(vals[0] - vals[1]) + r["error"]
            scaled = _scaled(key, sel[0], name, err_prod)
            rows[key] = {"values": vals, "estimated_error": scaled, "order": r["order"],
                         "tolerance": TOL[key], "pass": scaled <= TOL[key]}
        table[name] = rows
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--preset", default="mercury-vpw99")
    ap.add_argument("--model", default="TWO")
    ap.add_argument("--geothermal-flux", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.outdir)
    model = None if args.preset != "mercury-vpw99" else args.model
    runs = {}
    for i, steps in enumerate(DT_LADDER):
        runs[f"dt{i}"] = _cfg(args.preset, model, args.geothermal_flux, steps,
                              BASE["grid"], BASE["depth"], f"dt{i}")
    for i, grid in enumerate(GRID_LADDER):
        runs[f"grid{i}"] = _cfg(args.preset, model, args.geothermal_flux, BASE["cycle_steps"],
                                grid, BASE["depth"], f"grid{i}")
    for i, depth in enumerate(DEPTH_LADDER):
        runs[f"depth{i}"] = _cfg(args.preset, model, args.geothermal_flux, BASE["cycle_steps"],
                                 BASE["grid"], depth, f"depth{i}")
    # identical configurations are run once
    unique = {}
    for tag, cfg in runs.items():
        unique.setdefault(cfg.with_(experiment_id="x").sha256(), (tag, cfg))
    alias = {tag: unique[cfg.with_(experiment_id="x").sha256()][0] for tag, cfg in runs.items()}
    if not args.tables_only:
        todo = [(cfg.to_dict(), str(out / tag)) for tag, cfg in unique.values()
                if not (out / tag / "retained" / "status.json").exists()]
        with mp.get_context("fork").Pool(args.workers) as pool:
            for res in pool.imap_unordered(_run, todo):
                print(res, flush=True)
    status = {tag: json.loads((out / alias[tag] / "retained" / "status.json").read_text())
              for tag in runs}
    q = {tag: quantities(out / alias[tag]) for tag in runs}
    dt_q = [q[f"dt{i}"] for i in range(len(DT_LADDER))]
    grid_q = [q[f"grid{i}"] for i in range(len(GRID_LADDER))]
    prod_dt = DT_LADDER.index(BASE["cycle_steps"])
    prod_grid = GRID_LADDER.index(BASE["grid"])
    report = {
        "preset": args.preset, "model": model, "geothermal_flux": args.geothermal_flux,
        "production": BASE, "status": status, "aliases": alias,
        "tolerances": TOL,
        "depth": {},
        "quantities": q,
    }
    # The production level's error uses it and the two finer levels.
    report["timestep"] = analyse(dt_q, prod_dt)
    report["grid"] = analyse(grid_q, prod_grid)
    d0, d1 = q["depth0"], q["depth1"]
    for name in NAMES:
        report["depth"][name] = {}
        for key in TOL:
            base = key.replace("_rel", "")
            diff = abs(d0[name][base] - d1[name][base])
            scaled = _scaled(key, d0, name, diff)
            report["depth"][name][key] = {"change": scaled, "tolerance": TOL[key],
                                          "pass": scaled <= TOL[key]}
    every = [r["pass"] for part in ("timestep", "grid", "depth")
             for col in report[part].values() for r in col.values()]
    report["all_periodic"] = all(s["status"] == "PERIODIC" for s in status.values())
    report["pass"] = bool(all(every) and report["all_periodic"])
    (out / "gp.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"pass": report["pass"], "all_periodic": report["all_periodic"],
                      "failures": [(part, col, key) for part in ("timestep", "grid", "depth")
                                   for col, rows in report[part].items()
                                   for key, r in rows.items() if not r["pass"]]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
