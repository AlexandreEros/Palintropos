"""Kernel-launch and synchronization census of production substrate steps
under Nsight Systems: the P3 protocol of the consolidation audit (section 15).

    PYTHONPATH=src:. python tools/colab/substrate_nsys.py --out census.json
        [--steps 24 96] [--state-file STATE.npz]

Each traced run advances one ``run_block`` call (Mercury VPW99 TWO, 90 x 180,
51 layers) by N warm production steps, with tier-2 output sampled only at
the end of the call. Per-step figures are the **difference** between two
lengths, (count(N2) - count(N1)) / (N2 - N1), so the call's setup and
teardown (initial energy, final samples, returned arrays) cancel and only
the steps between outputs remain. Output events are counted separately
from a run at the production sampling cadence, by the same differencing.

Counted per step:

* kernel launches: cuLaunchKernel + cudaLaunchKernel;
* host synchronizations: cudaStreamSynchronize + cudaDeviceSynchronize +
  device-to-host copies (each CuPy device-to-host copy synchronizes; they
  are listed separately so that no synchronization is counted twice);
* launches by phase, from NVTX ranges around the forcing, the step
  (validation and preparation), the native kernel launch, the report and
  the stratum extents; the remainder is the runner's accumulation.

A report that fails or cannot be parsed raises: it is never read as zero
activity. Tracing adds overhead, so wall times here are not throughput
figures (``substrate_gate.py`` measures those).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time


def child(args) -> None:
    import numpy as np
    import cupy as cp
    from cupy.cuda import nvtx
    import tropoi.physics.insolation as insolation
    import tropoi.physics.substrate as physics_substrate
    import tropoi.temporal.substrate as temporal_substrate
    import tropoi.temporal.substrate_native as native
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_columns, build_substrate, run_block

    def ranged(owner, name, label):
        fn = getattr(owner, name)

        def wrapper(*a, **k):
            nvtx.RangePush(label)
            try:
                return fn(*a, **k)
            finally:
                nvtx.RangePop()
        setattr(owner, name, wrapper)

    ranged(insolation.OrbitalInsolation, "incident", "forcing")
    ranged(temporal_substrate.BackwardEulerSubstrate, "step", "step")
    ranged(native, "run_step", "native")
    ranged(temporal_substrate.BackwardEulerSubstrate, "_report", "report")
    ranged(physics_substrate.ConductiveSubstrate, "_temperature_extent", "extent")

    cfg = mercury_vpw99(columns={"kind": "latlon", "nlat": 90, "nlon": 180})
    c = build_columns(cfg)
    if args.state_file and pathlib.Path(args.state_file).exists():
        z = np.load(args.state_file)
        a, s, first = z["anomaly"], z["surface_anomaly"], int(z["next_step"])
    else:
        st = build_substrate(cfg).initial_state(c.size)
        a, s, first = np.asarray(st.anomaly), np.asarray(st.surface_anomaly), 0
    n = args.steps[0]
    cadence = ((cfg.cycle_steps // cfg.output["surface_samples"],
                cfg.cycle_steps // cfg.output["profile_samples"])
               if args.cadence == "production" else (n, n))
    with tempfile.TemporaryDirectory() as tmp:
        for label, k in (("warm", 6), ("measured", n)):
            d = pathlib.Path(tmp) / label
            d.mkdir()
            cp.cuda.Device().synchronize()
            if label == "measured":
                cp.cuda.runtime.profilerStart()
            run_block(cfg.to_dict(), c.lat, c.lon, c.weights, a, s, 0, k, False, str(d), None, 0,
                      "cupy", dt=cfg.dt, first_step=first,
                      sample_every=(cadence if label == "measured" else (k, k)))
            cp.cuda.Device().synchronize()
            if label == "measured":
                cp.cuda.runtime.profilerStop()
    print(json.dumps({"ok": True}))


def _stats(report: str, rep: pathlib.Path) -> list:
    out = subprocess.run(["nsys", "stats", "-q", "--report", report, "--format", "csv",
                          "--output", "-", str(rep)], capture_output=True, text=True)
    lines = out.stdout.splitlines()
    head = next((i for i, l in enumerate(lines) if "," in l and ("Time" in l or "Range" in l)),
                None)
    if out.returncode or head is None:
        raise RuntimeError(f"nsys stats --report {report} failed (exit {out.returncode}): "
                           f"{out.stderr[:800]} {out.stdout[:400]}")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[head:]))))
    if not rows:
        raise RuntimeError(f"nsys stats --report {report} returned no rows")
    return rows


def _num(r, *keys):
    for k in keys:
        if k in r and r[k] not in ("", None):
            return float(r[k])
    return 0.0


def trace(steps: int, cadence: str, state_file) -> dict:
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit("nsys not found")
    with tempfile.TemporaryDirectory() as tmp:
        rep = pathlib.Path(tmp) / "trace"
        cmd = [nsys, "profile", "-t", "cuda,nvtx", "--capture-range=cudaProfilerApi",
               "--capture-range-end=stop", "-o", str(rep), "-f", "true", sys.executable,
               __file__, "--child", "--steps", str(steps), "--cadence", cadence]
        if state_file:
            cmd += ["--state-file", state_file]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "src:.")})
        if p.returncode:
            raise RuntimeError(f"traced run failed: {p.stderr[-2000:]}")
        r = pathlib.Path(str(rep) + ".nsys-rep")
        api = _stats("cuda_api_sum", r)
        mem = _stats("cuda_gpu_mem_time_sum", r)
        kern = _stats("cuda_gpu_kern_sum", r)
        nvk = _stats("nvtx_kern_sum", r)
    calls = {row.get("Name", "?"): _num(row, "Num Calls") for row in api}
    launches = sum(v for k, v in calls.items() if k.startswith(("cuLaunchKernel",
                                                                 "cudaLaunchKernel")))
    syncs = sum(v for k, v in calls.items() if "Synchronize" in k)
    d2h = sum(_num(row, "Count") for row in mem if "Device-to-Host" in row.get("Operation", ""))
    phase = {}
    for row in nvk:
        rng = row.get("NVTX Range", row.get("Range", "?")).lstrip(":")
        phase[rng] = phase.get(rng, 0.0) + _num(row, "Kern Inst", "Instances")
    return {"steps": steps, "cadence": cadence, "launches": launches, "synchronizations": syncs,
            "device_to_host_copies": d2h, "launches_in_ranges": phase,
            "kernel_ms": sum(_num(row, "Total Time (ns)") for row in kern) / 1e6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, nargs="+", default=[24, 96])
    ap.add_argument("--state-file")
    ap.add_argument("--cadence", choices=("production", "none"), default="none")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    if args.child:
        child(args)
        return 0
    n1, n2 = args.steps
    runs = {f"{cad}-{n}": trace(n, cad, args.state_file)
            for cad in ("none", "production") for n in (n1, n2)}

    def per_step(cad, key):
        return (runs[f"{cad}-{n2}"][key] - runs[f"{cad}-{n1}"][key]) / (n2 - n1)

    doc = {"runs": runs, "protocol": "per step = (count(N2) - count(N1)) / (N2 - N1)"}
    for cad in ("none", "production"):
        ph1, ph2 = runs[f"{cad}-{n1}"]["launches_in_ranges"], runs[f"{cad}-{n2}"]["launches_in_ranges"]
        doc[f"per_step_{cad}"] = {
            "launches": per_step(cad, "launches"),
            "synchronizations": per_step(cad, "synchronizations"),
            "device_to_host_copies": per_step(cad, "device_to_host_copies"),
            "launches_by_range": {k: (ph2.get(k, 0.0) - ph1.get(k, 0.0)) / (n2 - n1)
                                  for k in sorted(set(ph1) | set(ph2))}}
    nb, pb = doc["per_step_none"], doc["per_step_production"]
    doc["output_events_per_step"] = {k: pb[k] - nb[k] for k in ("launches", "synchronizations",
                                                                 "device_to_host_copies")}
    pathlib.Path(args.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "runs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
