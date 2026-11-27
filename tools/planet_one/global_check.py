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
the global run's. The frozen bound is 1e-12 K. A kind of sample the
retained segment does not hold (a transient tail shorter than the sampling
cadence) is reported as not compared, never as a zero difference, and the
verdict rests on the quantities that were compared (the end state always
is). Writes RUN_DIR/gg.json; exits 0 on a pass and 1 otherwise.
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
    elif cfg["columns"]["kind"] != "latlon":
        ap.error("--columns is required for a run whose columns are not a latlon grid")
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
    # A segment can hold no samples of a kind (a transient tail shorter than
    # the cadence). Such a quantity is reported as not compared (None), never
    # as a zero difference, and the verdict rests only on what was compared.
    diffs = {"surface_fields": _max_abs_diff(s_rep, s_glob),
             "profiles": _max_abs_diff(p_rep, p_glob),
             # cells and the diagnosed skin both: a sample-free segment is judged on these alone
             "end_state": max(
                 float(np.max(np.abs(res["anomaly"] - state["end_anomaly"][:, cols]))),
                 float(np.max(np.abs(res["surface_anomaly"]
                                     - state["end_surface_anomaly"][cols]))))}
    compared = [k for k, v in diffs.items() if v is not None]
    bound = 1e-12
    report = {"run": str(run), "backend": backend, "cycle": cycle, "steps": steps,
              "timing": {k: list(v) if isinstance(v, tuple) else v for k, v in timing.items()},
              "columns": cols.tolist(), "lat_deg": lat[cols].tolist(), "lon_deg": lon[cols].tolist(),
              "samples": {"surface": int(s_glob.shape[0]), "profiles": int(p_glob.shape[0])},
              "max_abs_surface_fields_diff": diffs["surface_fields"],
              "max_abs_profile_diff_K": diffs["profiles"],
              "max_abs_end_state_diff_K": diffs["end_state"], "bound_K": bound,
              "compared": compared,
              "not_compared": {k: "no samples in the retained segment"
                               for k, v in diffs.items() if v is None},
              "pass": all(diffs[k] <= bound for k in compared)}
    (run / "gg.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


def _max_abs_diff(replayed, retained):
    """max |replayed - retained|, or None when there is nothing to compare.
    The two must have the same shape (same sample count) to be compared."""
    if replayed.shape != retained.shape:
        raise ValueError(f"replay produced {replayed.shape} samples, the run holds "
                         f"{retained.shape}")
    if replayed.size == 0:
        return None
    return float(np.max(np.abs(replayed - retained)))


if __name__ == "__main__":
    sys.exit(main())
