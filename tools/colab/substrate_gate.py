"""Migration gate of the native substrate step on the production workload.

    PYTHONPATH=src:. python tools/colab/substrate_gate.py --out gate.json [--steps 200]

On the reference GPU (Tesla T4):

1. regenerates the representative state of the consolidation audit (Mercury
   VPW99 TWO, 90 x 180 x 51; 3 coarse cycles of 720 steps, then 432
   production steps from the 350 K isothermal start) with the production
   runner, and reports its SHA-256 against the audit's
   ef413855553df37ba5aa1d23121526375bcbe38b1727407de8cb62b814964998;
2. advances ``--steps`` production steps from it with the native step and
   with the CuPy migration reference, comparing the state and every
   StepReport field bit for bit, and the Newton counts;
3. times the complete production runner (``run_block``: forcing, step,
   accumulation, sampling) and the bare step, native and reference, warm,
   synchronized, median of ``--repeats``; records the memory pool, the
   kernel's registers and local memory, and the cold compile time;
4. checks the CPU library against the NumPy reference (G-GPU criteria, and
   bitwise status) on a subset of columns;
5. the rest of the N1 set: terminator batches of the VPW99 and of a
   constant-property (THAI-like) substrate at 16200 columns, 200 steps each,
   bitwise against the reference; and the Ceres finite-transient fixture run
   end to end by AirlessRun with each stepper, every output array compared.

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
FIELDS = ("energy_tendency", "absorbed", "emitted", "conducted", "bottom", "source",
          "residual_sum", "tolerance_sum", "max_residual_ratio")


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class _DescribedReference:
    """The migration reference as AirlessRun uses a stepper (it records the
    implementation's provenance)."""

    def __new__(cls, sub):
        from tropoi.temporal._substrate_reference import ReferenceBackwardEulerSubstrate
        ref = ReferenceBackwardEulerSubstrate(sub)
        ref.describe = lambda backend: {"implementation": "Python migration reference",
                                        "backend": backend}
        return ref


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


def ceres_check() -> dict:
    """The Ceres finite-transient fixture, end to end through AirlessRun on
    the GPU, with the native and the reference step: every output array and
    every cycles.json value compared."""
    import tropoi.run.airless.experiment as experiment
    from tropoi.run.airless.config import load_config_document
    from tropoi.run.airless.experiment import AirlessRun
    from tropoi.temporal.substrate import BackwardEulerSubstrate
    ceres, _ = load_config_document("tests/fixtures/airless/ceres-equatorial-column.airless.json")
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for label, cls in (("native", BackwardEulerSubstrate), ("reference", _DescribedReference)):
            original = experiment.BackwardEulerSubstrate
            experiment.BackwardEulerSubstrate = cls
            try:
                d = pathlib.Path(tmp) / label
                AirlessRun(ceres, d, backend="cupy", allow_dirty=True,
                           log=lambda m: None).run_transient(2 * ceres.cycle_steps)
            finally:
                experiment.BackwardEulerSubstrate = original
            dirs[label] = d
        compared = differing = 0
        for f in sorted(dirs["reference"].rglob("*.np[yz]")):
            g = dirs["native"] / f.relative_to(dirs["reference"])
            A, B = np.load(f), np.load(g)
            for key in (A.files if f.suffix == ".npz" else [None]):
                compared += 1
                differing += 0 if np.array_equal(A[key] if key else A, B[key] if key else B) else 1
        ca = json.loads((dirs["reference"] / "cycles.json").read_text())
        cb = json.loads((dirs["native"] / "cycles.json").read_text())
        for u, v in zip(ca, cb):
            for key in u:
                if key != "wall_s":
                    compared += 1
                    differing += 0 if u[key] == v[key] else 1
    return {"steps": 2 * ceres.cycle_steps, "arrays_and_values_compared": compared,
            "differing": differing}


def main(argv=None) -> int:
    import cupy as cp
    from tropoi.physics.insolation import insolation_from_spec
    from tropoi.physics.substrate import SubstrateState
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_columns, build_substrate, run_block
    from tropoi.temporal import substrate_native
    from tropoi.temporal._substrate_reference import ReferenceBackwardEulerSubstrate
    from tropoi.temporal.substrate import BackwardEulerSubstrate
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timed-steps", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", choices=("all", "ceres"), default="all")
    args = ap.parse_args(argv)
    if args.only == "ceres":
        result = {"ceres_fixture": ceres_check()}
        pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return 0
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
    # P5 warm: a fresh process loading the kernel from the now populated cache
    import sys
    warm = subprocess.run([sys.executable, "-c", WARM_LOAD], capture_output=True, text=True,
                          env=dict(os.environ))
    if warm.returncode:
        raise RuntimeError(f"warm-cache load failed: {warm.stderr[-800:]}")
    doc["native_kernel_warm_cache_load"] = json.loads(warm.stdout.strip().splitlines()[-1])
    save()

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

    # 2. bitwise trajectory, native against the CuPy reference
    native, ref = BackwardEulerSubstrate(sub), ReferenceBackwardEulerSubstrate(sub)
    x = SubstrateState(cp.asarray(a), cp.asarray(s))
    y = SubstrateState(cp.asarray(a), cp.asarray(s))
    mism = {"state": 0, "report": 0, "iterations": 0, "extent": 0}
    for k in range(args.steps):
        sw = forcing(k)
        x, rx = native.step(x, cfg.dt, shortwave_down=sw)
        y, ry = ref.step(y, cfg.dt, shortwave_down=sw)
        mism["state"] += 0 if bool(cp.array_equal(x.anomaly, y.anomaly)
                                   & cp.array_equal(x.surface_anomaly, y.surface_anomaly)) else 1
        mism["report"] += 0 if all(bool(cp.array_equal(getattr(rx, f), getattr(ry, f)))
                                   for f in FIELDS) else 1
        mism["iterations"] += 0 if rx.iterations == ry.iterations else 1
        mism["extent"] += 0 if ([tuple(map(float, e)) for e in rx.extent]
                                == [tuple(map(float, e)) for e in ry.extent]) else 1
        if mism["state"]:
            y = SubstrateState(x.anomaly.copy(), x.surface_anomaly.copy())
    doc["gpu_bitwise_against_reference"] = {"steps": args.steps, **{f"{k}_mismatched_steps": v
                                                                     for k, v in mism.items()}}
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

    doc["run_block"] = {"native": time_run_block(BackwardEulerSubstrate),
                        "reference": time_run_block(ReferenceBackwardEulerSubstrate,
                                                    "reference")}
    doc["run_block"]["speedup"] = (doc["run_block"]["reference"]["median_s"]
                                   / doc["run_block"]["native"]["median_s"])
    doc["run_block"]["outputs_bitwise_native_vs_reference"] = _same_results(
        results["native"], results["reference"])
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
    doc["step"] = {"native": time_step(native), "reference": time_step(ref)}
    save()

    # 4. the CPU library against the NumPy reference on a subset of columns
    lib = substrate_native.host_library()
    pick = np.arange(0, cols.size, 37)
    xc = SubstrateState(np.array(a[:, pick]), np.array(s[pick]))
    yc = SubstrateState(np.array(a[:, pick]), np.array(s[pick]))
    nv_h = tuple(v.get()[pick] for v in nv)
    worst, bit, same_it = 0.0, True, True
    for k in range(10):
        sw = ins.incident(*nv_h, (first + k + 1) * cfg.dt)[0]
        xc, rx = native.step(xc, cfg.dt, shortwave_down=sw)
        yc, ry = ref.step(yc, cfg.dt, shortwave_down=sw)
        worst = max(worst, float(np.max(np.abs(xc.anomaly - yc.anomaly))),
                    float(np.max(np.abs(xc.surface_anomaly - yc.surface_anomaly))))
        bit &= bool(np.array_equal(xc.anomaly, yc.anomaly))
        same_it &= rx.iterations == ry.iterations
    doc["cpu_against_reference"] = {"columns": int(pick.size), "steps": 10,
                                    "max_abs_difference_K": worst, "g_gpu_bound_K": 1e-6,
                                    "within_bound": worst <= 1e-6,
                                    "newton_counts_identical": bool(same_it), "bitwise": bit,
                                    "host_build": lib.flags}
    save()

    # 5a. terminator batches, VPW99 and constant properties, bitwise
    from tropoi.physics.materials import Stratum
    from tropoi.physics.substrate import ConductiveSubstrate
    from tropoi.spatial.vertical import geometric_interfaces
    rng = np.random.default_rng(16200)
    tlat = np.radians(rng.uniform(-60, 60, 16200))
    tlon = np.radians(np.where(rng.random(16200) < 0.5, 90.0, 270.0) + rng.uniform(-3, 3, 16200))
    tn = (cp.cos(cp.asarray(tlat)) * cp.cos(cp.asarray(tlon)),
          cp.cos(cp.asarray(tlat)) * cp.sin(cp.asarray(tlon)), cp.sin(cp.asarray(tlat)))
    thai = ConductiveSubstrate([Stratum.volumetric(3.5, 2.0e6, 2.0)],
                               interfaces=geometric_interfaces(0.02, 3.5, layers=30), albedo=0.1,
                               emissivity=0.95, bottom_heat_flux=0.0, initial_temperature=300.0)
    doc["terminator_bitwise"] = {}
    for label, tsub in (("vpw99_two", sub), ("constant_thai_like", thai)):
        nat, rf = BackwardEulerSubstrate(tsub), ReferenceBackwardEulerSubstrate(tsub)
        x = tsub.initial_state(16200, xp=cp)
        y = tsub.initial_state(16200, xp=cp)
        bad = {"state": 0, "report": 0, "iterations": 0, "extent": 0}
        for k in range(args.steps):
            sw = ins.incident(*tn, (k + 1) * cfg.dt)[0]
            x, rx = nat.step(x, cfg.dt, shortwave_down=sw)
            y, ry = rf.step(y, cfg.dt, shortwave_down=sw)
            bad["state"] += 0 if bool(cp.array_equal(x.anomaly, y.anomaly)
                                      & cp.array_equal(x.surface_anomaly, y.surface_anomaly)) else 1
            bad["report"] += 0 if all(bool(cp.array_equal(getattr(rx, f), getattr(ry, f)))
                                      for f in FIELDS) else 1
            bad["iterations"] += 0 if rx.iterations == ry.iterations else 1
            bad["extent"] += 0 if ([tuple(map(float, e)) for e in rx.extent]
                                   == [tuple(map(float, e)) for e in ry.extent]) else 1
            if bad["state"]:
                y = SubstrateState(x.anomaly.copy(), x.surface_anomaly.copy())
        doc["terminator_bitwise"][label] = {"steps": args.steps, "layers": tsub.layers,
                                            **{f"{k}_mismatched_steps": v for k, v in bad.items()}}
        save()

    doc['ceres_fixture'] = ceres_check()
    save()
    print(json.dumps(doc, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
