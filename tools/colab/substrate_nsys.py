"""Kernel-launch census of production substrate steps under Nsight Systems.

    PYTHONPATH=src:. python tools/colab/substrate_nsys.py --out census.json [--steps 20]
        [--state-file STATE.npz] [--thomas-prototype]

Runs ``--steps`` warm production steps (Mercury VPW99 TWO, 90 x 180, 51
layers) of ``run_block`` inside ``nsys profile -t cuda,nvtx`` and reports, per
step: kernel launches, summed kernel execution time, CUDA API time, memcpy
and synchronization calls, and the GPU-busy fraction (summed kernel time /
traced wall time). Tracing adds overhead, so the wall time here is not a
throughput figure; ``substrate_benchmark.py end_to_end`` is.
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

HERE = pathlib.Path(__file__).resolve().parent


def child(args) -> None:
    import numpy as np
    import cupy as cp
    from cupy.cuda import nvtx
    sys.path.insert(0, str(HERE))
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_columns, build_substrate, run_block
    cfg = mercury_vpw99(columns={"kind": "latlon", "nlat": 90, "nlon": 180})
    if args.thomas_prototype:
        import thomas_prototype as tp
        import tropoi.physics.substrate as m
        m._tridiagonal = tp.make_solver(fmad=False)
    lat, lon, w = build_columns(cfg)
    if args.state_file and pathlib.Path(args.state_file).exists():
        z = np.load(args.state_file)
        a, s, first = z["anomaly"], z["surface_anomaly"], int(z["next_step"])
    else:
        st = build_substrate(cfg).initial_state(lat.size)
        a, s, first = np.asarray(st.anomaly), np.asarray(st.surface_anomaly), 0
    cadence = (cfg.cycle_steps // cfg.output["surface_samples"],
               cfg.cycle_steps // cfg.output["profile_samples"])
    with tempfile.TemporaryDirectory() as tmp:
        for label, n in (("warm", 5), ("measured", args.steps)):
            d = pathlib.Path(tmp) / label
            d.mkdir()
            cp.cuda.Device().synchronize()
            if label == "measured":
                cp.cuda.runtime.profilerStart()
                nvtx.RangePush("measured")
            t = time.perf_counter()
            r = run_block(cfg.to_dict(), lat, lon, w, a, s, 0, n, False, str(d), None, 0, "cupy",
                          dt=cfg.dt, first_step=first, sample_every=cadence)
            cp.cuda.Device().synchronize()
            wall = time.perf_counter() - t
            if label == "measured":
                nvtx.RangePop()
                cp.cuda.runtime.profilerStop()
    print(json.dumps({"traced_wall_s": wall, "iterations_mean": float(np.mean(r["iterations"]))}))


_RAW: dict = {}


def _csv(report: str, rep: pathlib.Path) -> list:
    out = subprocess.run(["nsys", "stats", "-q", "--report", report, "--format", "csv",
                          "--output", "-", str(rep)], capture_output=True, text=True)
    text = out.stdout
    # the CSV starts at its header line; anything before it is nsys chatter
    lines = text.splitlines()
    head = next((i for i, l in enumerate(lines) if "Time" in l and "," in l), None)
    rows = (list(csv.DictReader(io.StringIO("\n".join(lines[head:]))))
            if head is not None else [])
    if not rows:
        _RAW[report] = {"stdout_head": text[:1500], "stderr_head": out.stderr[:1500]}
    return rows


def parent(args) -> int:
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit("nsys not found")
    with tempfile.TemporaryDirectory() as tmp:
        rep = pathlib.Path(tmp) / "trace"
        cmd = [nsys, "profile", "-t", "cuda,nvtx", "--capture-range=cudaProfilerApi",
               "--capture-range-end=stop", "-o", str(rep), "-f", "true",
               sys.executable, __file__, "--child", "--steps", str(args.steps)]
        if args.state_file:
            cmd += ["--state-file", args.state_file]
        if args.thomas_prototype:
            cmd.append("--thomas-prototype")
        p = subprocess.run(cmd, capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "src:.")})
        child_out = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
        report = pathlib.Path(str(rep) + ".nsys-rep")
        kern = _csv("cuda_gpu_kern_sum", report)
        api = _csv("cuda_api_sum", report)
        mem = _csv("cuda_gpu_mem_time_sum", report)

    def num(r, *keys):
        for k in keys:
            if k in r and r[k] != "":
                return float(r[k])
        return 0.0
    steps = args.steps
    launches = sum(num(r, "Instances", "Num Calls") for r in kern)
    kernel_ns = sum(num(r, "Total Time (ns)") for r in kern)
    api_calls = {r.get("Name", "?"): num(r, "Num Calls") / steps for r in api}
    top = sorted(kern, key=lambda r: -num(r, "Total Time (ns)"))[:15]
    doc = {"steps": steps, "thomas_prototype": args.thomas_prototype, **child_out,
           "kernel_launches_per_step": launches / steps,
           "kernel_time_ms_per_step": kernel_ns / 1e6 / steps,
           "traced_wall_ms_per_step": 1e3 * child_out["traced_wall_s"] / steps,
           "gpu_busy_fraction_of_traced_wall": kernel_ns / 1e9 / child_out["traced_wall_s"],
           "api_calls_per_step": api_calls,
           "memcpy_per_step": {r.get("Operation", "?"): num(r, "Count") / steps for r in mem},
           "top_kernels": [{"name": r.get("Name", "")[:120],
                            "instances_per_step": num(r, "Instances") / steps,
                            "ms_per_step": num(r, "Total Time (ns)") / 1e6 / steps,
                            "avg_us": num(r, "Avg (ns)") / 1e3} for r in top],
           "unparsed_reports": _RAW}
    pathlib.Path(args.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "top_kernels"}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--state-file")
    ap.add_argument("--thomas-prototype", action="store_true")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a)
    else:
        raise SystemExit(parent(a))
