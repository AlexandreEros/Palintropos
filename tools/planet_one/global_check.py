"""G-G: global columns against standalone runs of the same columns.

    python tools/planet_one/global_check.py RUN_DIR [--columns i,j,...] [--backend cupy]

The retained cycle of a global run is replayed for selected columns as an
independent standalone run. The replay uses the same float coordinates
(read from the run's grid.npz), the same start state (retained/state.npz),
the same cycle index, timestep and settings, and the same backend. A
retained transient segment is replayed with its recorded timestep, first
global step and sampling cadence; runs that predate those records use the
cycle form (timestep cycle_s/steps, first step cycle*steps). Every
tier-2 sample of the replay (surface fields and full T(z)) is compared with
the global run's. The frozen bound is 1e-12 K. Writes RUN_DIR/gg.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

from tropoi.run.airless.experiment import load_tier2, run_block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--columns")
    ap.add_argument("--backend")
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    meta = json.loads((run / "config.json").read_text())
    cfg = meta["config"]
    backend = args.backend or meta.get("backend", "numpy")
    grid = np.load(run / "grid.npz")
    lat, lon, w = grid["lat_deg"], grid["lon_deg"], grid["weights"]
    if args.columns:
        cols = np.array([int(c) for c in args.columns.split(",")])
    else:                                   # canonical-like points of the grid
        nlat, nlon = cfg["columns"]["nlat"], cfg["columns"]["nlon"]
        la, lo = lat.reshape(nlat, nlon)[:, 0], lon.reshape(nlat, nlon)[0]
        pick = [(0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, 270.0), (45.0, 0.0), (85.0, 0.0)]
        cols = np.array([int(np.argmin(np.abs(la - a))) * nlon
                         + int(np.argmin(np.abs(((lo - b + 180) % 360) - 180))) for a, b in pick])
    ret = run / "retained"
    state = np.load(ret / "state.npz")
    steps = int(state["steps"])
    cycle = int(state["cycle"])
    # the retained cycle's own timing and cadence when recorded (transient
    # segments need them); older runs fall back to run_block's cycle form
    timing = {}
    if "dt" in state.files:
        timing["dt"] = float(state["dt"])
        timing["first_step"] = int(state["first_step"])
    if "sample_every" in state.files:
        timing["sample_every"] = tuple(int(v) for v in state["sample_every"])
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp)
        (out / "blocks.json").write_text(json.dumps({"ranges": [[0, cols.size]], "steps": steps}))
        res = run_block(cfg, lat[cols], lon[cols], w[cols], state["start_anomaly"][:, cols],
                        state["start_surface_anomaly"][cols], cycle, steps, False,
                        str(out), None, 0, backend, **timing)
        s_rep = load_tier2(out, "surface")
        p_rep = load_tier2(out, "profiles")
    s_glob = load_tier2(ret, "surface", columns=cols)
    p_glob = load_tier2(ret, "profiles", columns=cols)
    d_s = float(np.max(np.abs(s_rep - s_glob)))
    d_p = float(np.max(np.abs(p_rep - p_glob)))
    d_end = float(np.max(np.abs(res["anomaly"] - state["end_anomaly"][:, cols])))
    report = {"run": str(run), "backend": backend, "cycle": cycle, "steps": steps,
              "timing": {k: list(v) if isinstance(v, tuple) else v for k, v in timing.items()},
              "columns": cols.tolist(), "lat_deg": lat[cols].tolist(), "lon_deg": lon[cols].tolist(),
              "max_abs_surface_fields_diff": d_s, "max_abs_profile_diff_K": d_p,
              "max_abs_end_state_diff_K": d_end, "bound_K": 1e-12,
              "pass": max(d_s, d_p, d_end) <= 1e-12}
    (run / "gg.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
