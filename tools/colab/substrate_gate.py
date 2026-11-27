"""Regression gate of the native substrate step on the production workload.

    PYTHONPATH=src:. python tools/colab/substrate_gate.py --out gate.json [--steps 200]

On the reference GPU (Tesla T4):

1. regenerates the representative state of the consolidation audit (Mercury
   VPW99 TWO, 90 x 180 x 51; 3 coarse cycles of 720 steps, then 432
   production steps from the 350 K isothermal start) with the production
   runner, and reports its SHA-256 against the audit's
   ef413855553df37ba5aa1d23121526375bcbe38b1727407de8cb62b814964998
   (bitwise regression of the whole runner, step and forcing);
2. advances ``--steps`` production steps from it, recording the final
   state's SHA-256, then loads the kernel in a fresh process from the
   populated cache (audit P5, warm);
3. times the complete production runner (``run_block``: forcing, step,
   accumulation, sampling) and the bare step, warm, synchronized, median of
   ``--repeats``; records the memory pool, the kernel's registers and local
   memory, and the cold compile time; and the insolation forcing's share of
   the runner (the same run with the forcing replayed from arrays computed
   beforehand, outputs compared bit for bit);
4. checks the CPU build against the CUDA build on a subset of columns
   (G-GPU criteria, and bitwise status).

The bitwise migration gates against the former Python implementation (N1:
trajectory, terminator batches, the Ceres fixture) are recorded in
docs/validation/consolidation-evidence/ at the commits that ran them; that
implementation has since been removed.

The GPU model, driver and CPU are recorded; results from another GPU model
are not comparable with the T4 figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time

import numpy as np

#: run in a fresh process after the cold compile, with the same CUPY_CACHE_DIR:
#: the kernel's load from CuPy's disk cache (audit P5, warm), and the first step
#: against a later one
WARM_LOAD = """
import json, time
t0 = time.perf_counter()
import cupy as cp
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import build_substrate
from tropoi.temporal import substrate_native
from tropoi.temporal.substrate import BackwardEulerSubstrate
cfg = mercury_vpw99(columns={"kind": "latlon", "nlat": 90, "nlon": 180})
sub = build_substrate(cfg)
cp.cuda.Device().synchronize()
t1 = time.perf_counter()
substrate_native._device_kernel(sub)
cp.cuda.Device().synchronize()
t2 = time.perf_counter()
stepper = BackwardEulerSubstrate(sub)
st = sub.initial_state(16200, xp=cp)
sw = cp.full(16200, 2000.0)
times = []
for _ in range(3):
    cp.cuda.Device().synchronize(); a = time.perf_counter()
    st, _ = stepper.step(st, cfg.dt, shortwave_down=sw)
    cp.cuda.Device().synchronize(); times.append(time.perf_counter() - a)
print(json.dumps({"kernel_load_from_cache_s": t2 - t1, "imports_and_setup_s": t1 - t0,
                  "first_step_s": times[0], "third_step_s": times[2],
                  "first_step_excess_s": times[0] - times[2]}))
"""
AUDIT_STATE = "ef413855553df37ba5aa1d23121526375bcbe38b1727407de8cb62b814964998"


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _same_results(a: dict, b: dict) -> dict:
    """Bitwise comparison of two run_block results (every array)."""
    def flat(r, pre=""):
        for k, v in r.items():
            if isinstance(v, dict):
                yield from flat(v, f"{pre}{k}/")
            elif v is not None and not isinstance(v, str):
                yield f"{pre}{k}", np.asarray(v, dtype=float if isinstance(v, (list, tuple))
                                              else None)
    fa, fb = dict(flat(a)), dict(flat(b))
    differ = sorted(k for k in fa if k not in fb or fa[k].shape != fb[k].shape
                    or fa[k].tobytes() != fb[k].tobytes())
    return {"compared": len(fa), "differing": differ, "identical": not differ}


def main(argv=None) -> int:
    import cupy as cp
    from tropoi.physics.insolation import insolation_from_spec
    from tropoi.physics.substrate import SubstrateState
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_columns, build_substrate, run_block
    from tropoi.temporal import substrate_native
    from tropoi.temporal.substrate import BackwardEulerSubstrate
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timed-steps", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    sync = lambda: cp.cuda.Device().synchronize()                       # noqa: E731
    props = cp.cuda.runtime.getDeviceProperties(0)
    doc = {"environment": {
        "gpu": props["name"].decode(), "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                     if l.startswith("model name")), "?"), "vcpus": os.cpu_count(),
        "python": platform.python_version(), "numpy": np.__version__, "cupy": cp.__version__,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()}}
    out_path = pathlib.Path(args.out)
    save = lambda: out_path.write_text(json.dumps(doc, indent=1, default=float))  # noqa: E731

    cfg = mercury_vpw99(columns={"kind": "latlon", "nlat": 90, "nlon": 180})
    sub = build_substrate(cfg)
    cols = build_columns(cfg)
    doc["implementation"] = BackwardEulerSubstrate(sub).describe("cupy")

    # cold compile of the native kernel (the launcher sets an empty CUPY_CACHE_DIR)
    t = time.perf_counter()
    substrate_native._device_kernel(sub)
    sync()
    doc["native_kernel_cold_compile_s"] = time.perf_counter() - t
    k, _ = substrate_native._device_kernel(sub)
    doc["native_kernel"] = {"registers": k.num_regs, "local_bytes": k.local_size_bytes}

    # 1. the representative state, regenerated by the production runner
    t = time.perf_counter()
    a, s = (np.asarray(v) for v in sub.initial_state(cols.size))
    with tempfile.TemporaryDirectory() as tmp:
        for c in range(3):
            d = pathlib.Path(tmp) / f"s{c}"
            d.mkdir()
            r = run_block(cfg.to_dict(), cols.lat, cols.lon, cols.weights, a, s, c,
                          cfg.spinup_cycle_steps, False, str(d), None, 0, "cupy")
            a, s = r["anomaly"], r["surface_anomaly"]
        d = pathlib.Path(tmp) / "p"
        d.mkdir()
        first = 3 * cfg.cycle_steps
        r = run_block(cfg.to_dict(), cols.lat, cols.lon, cols.weights, a, s, 0, 432, False, str(d),
                      None, 0, "cupy", dt=cfg.dt, first_step=first, sample_every=(432, 432))
        a, s = r["anomaly"], r["surface_anomaly"]
    first += 432
    doc["state"] = {"sha256": sha(a, s), "audit_sha256": AUDIT_STATE,
                    "identical_to_audit_state": sha(a, s) == AUDIT_STATE,
                    "spinup_wall_s": time.perf_counter() - t}
    save()

    ins = insolation_from_spec(cfg.illumination)
    la, lo = cp.asarray(cols.lat), cp.asarray(cols.lon)
    nv = (cp.cos(la) * cp.cos(lo), cp.cos(la) * cp.sin(lo), cp.sin(la))
    forcing = lambda k: ins.incident(*nv, (first + k + 1) * cfg.dt)[0]   # noqa: E731

    # 2. a production trajectory from it (warms every kernel of a step)
    native = BackwardEulerSubstrate(sub)
    x = SubstrateState(cp.asarray(a), cp.asarray(s))
    for k in range(args.steps):
        x, _ = native.step(x, cfg.dt, shortwave_down=forcing(k))
    doc["trajectory"] = {"steps": args.steps, "sha256": sha(x.anomaly.get(),
                                                            x.surface_anomaly.get())}
    save()

    # P5 warm: a fresh process, with the cache now holding every kernel of a
    # step, loading the native kernel and taking its first steps
    import sys
    warm = subprocess.run([sys.executable, "-c", WARM_LOAD], capture_output=True, text=True,
                          env=dict(os.environ))
    if warm.returncode:
        raise RuntimeError(f"warm-cache load failed: {warm.stderr[-800:]}")
    doc["native_kernel_warm_cache_load"] = json.loads(warm.stdout.strip().splitlines()[-1])
    save()

    # 3. timings: the complete production runner and the bare step
    cadence = (cfg.cycle_steps // cfg.output["surface_samples"],
               cfg.cycle_steps // cfg.output["profile_samples"])
    import tropoi.run.airless.experiment as experiment

    results = {}

    def time_run_block(stepper_cls, label="native", incident=None):
        original = experiment.BackwardEulerSubstrate
        experiment.BackwardEulerSubstrate = stepper_cls
        original_ins = experiment.insolation_from_spec
        if incident is not None:
            experiment.insolation_from_spec = lambda spec: incident
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        ts, iters = [], []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for rep in range(args.repeats + 1):
                    d = pathlib.Path(tmp) / f"r{rep}"
                    d.mkdir()
                    sync(); t0 = time.perf_counter()
                    r = run_block(cfg.to_dict(), cols.lat, cols.lon, cols.weights, a, s, 0,
                                  args.timed_steps, False, str(d), None, 0, "cupy", dt=cfg.dt,
                                  first_step=first, sample_every=cadence)
                    sync()
                    if rep:
                        ts.append((time.perf_counter() - t0) / args.timed_steps)
                        iters.append(float(np.mean(r["iterations"])))
                    results[label] = r
        finally:
            experiment.BackwardEulerSubstrate = original
            experiment.insolation_from_spec = original_ins
        return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts),
                "n": len(ts), "newton_iterations_mean": statistics.median(iters),
                "memory_pool_total_bytes": int(pool.total_bytes()),
                "column_steps_per_second": cols.size / statistics.median(ts)}

    def time_step(stepper):
        ts = []
        for rep in range(args.repeats + 1):
            st = SubstrateState(cp.asarray(a), cp.asarray(s))
            sync(); t0 = time.perf_counter()
            for k in range(args.timed_steps):
                st, _ = stepper.step(st, cfg.dt, shortwave_down=forcing(k))
            sync()
            if rep:
                ts.append((time.perf_counter() - t0) / args.timed_steps)
        return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts)}

    doc["run_block"] = {"native": time_run_block(BackwardEulerSubstrate)}
    save()

    # 3b. the wall-clock cost of the insolation forcing inside the runner: the
    # same run_block with the forcing replayed from arrays computed beforehand
    class Replayed:
        def __init__(self):
            self.cache = {}

        def incident(self, nx, ny, nz, t):
            key = float(t)
            if key not in self.cache:
                self.cache[key] = ins.incident(nx, ny, nz, t)
            return self.cache[key]

    replay = Replayed()
    time_run_block(BackwardEulerSubstrate, "replay-fill", replay)            # fills the cache
    replayed = time_run_block(BackwardEulerSubstrate, "replayed", replay)
    ts = []
    for rep in range(args.repeats + 1):
        sync(); t0 = time.perf_counter()
        for k in range(args.timed_steps):
            ins.incident(*nv, (first + k + 1) * cfg.dt)
        sync()
        if rep:
            ts.append((time.perf_counter() - t0) / args.timed_steps)
    doc["insolation"] = {
        "run_block_with_forcing_replayed": replayed,
        "runner_wall_clock_share_s": (doc["run_block"]["native"]["median_s"]
                                      - replayed["median_s"]),
        "incident_alone_synchronized_median_s": statistics.median(ts),
        "replayed_outputs_bitwise": _same_results(results["native"], results["replayed"])}
    doc["step"] = {"native": time_step(native)}
    save()

    # 4. the CPU build against the CUDA build on a subset of columns
    lib = substrate_native.host_library()
    pick = np.arange(0, cols.size, 37)
    xg = SubstrateState(cp.asarray(a[:, pick]), cp.asarray(s[pick]))
    xc = SubstrateState(np.array(a[:, pick]), np.array(s[pick]))
    nv_h = tuple(v.get()[pick] for v in nv)
    worst, bit, same_it = 0.0, True, True
    for k in range(10):
        sw = ins.incident(*nv_h, (first + k + 1) * cfg.dt)[0]
        xg, rg = native.step(xg, cfg.dt, shortwave_down=cp.asarray(sw))
        xc, rc = native.step(xc, cfg.dt, shortwave_down=sw)
        worst = max(worst, float(np.max(np.abs(xg.anomaly.get() - xc.anomaly))),
                    float(np.max(np.abs(xg.surface_anomaly.get() - xc.surface_anomaly))))
        bit &= bool(np.array_equal(xg.anomaly.get(), xc.anomaly))
        same_it &= rg.iterations == rc.iterations
    doc["cpu_against_cuda"] = {"columns": int(pick.size), "steps": 10,
                               "max_abs_difference_K": worst, "g_gpu_bound_K": 1e-6,
                               "within_bound": worst <= 1e-6,
                               "newton_counts_identical": bool(same_it), "bitwise": bit,
                               "host_build": lib.flags}
    save()
    print(json.dumps(doc, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
