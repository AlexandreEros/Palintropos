"""Conductive-substrate GPU benchmark: baseline, call counts, scaling,
compilation, and the compiled-Thomas feasibility prototype.

Run from a clean checkout with ``PYTHONPATH=src:.``. It changes nothing in
the package: the prototype is patched in for the duration of a measurement.

    python tools/colab/substrate_benchmark.py all --out results.json [--quick]
    python tools/colab/substrate_benchmark.py SECTION [SECTION ...] --out results.json

Sections:

* ``environment``: GPU model, driver, CPU model, vCPU count, library versions, commit.
* ``state``: the representative start state. The Mercury VPW99 TWO production
  configuration on its 90 x 180 grid (16200 columns, 51 layers) is spun up
  from the configured isothermal start: ``--spinup-cycles`` coarse cycles
  (720 steps each), then ``--production-steps`` steps at the production
  timestep. The state is saved and its SHA-256 recorded. Every timed
  section starts from it.
* ``end_to_end``: **the headline throughput**. The production per-step path
  ``run_block`` (insolation, Newton step, tier-1 accumulators, the four
  weighted global sums, tier-2 sampling at the production cadence), with no
  instrumentation, warm, synchronized before and after, repeated. It also
  times bare ``ConductiveSubstrate.step`` with the same forcing.
* ``counts``: calls per step of the residual-with-Jacobian assembly, the
  trial-residual assembly and the tridiagonal solve, and the implied host
  synchronizations. Counting adds no device synchronization.
* ``components``: synchronized inclusive wall time per component, and CUDA
  event spans (no synchronization inside the step). These are component
  costs. They are **not** additive shares of the uninstrumented step.
* ``scaling``: uninstrumented ``ConductiveSubstrate.step`` over column counts
  (including the T21/T42/T63 Gauss grids and the Mercury grid), layer counts,
  constant and temperature-dependent materials, one and two strata, and
  smooth versus terminator forcing; a least-squares fit
  t = a + b L + c n L separates the per-launch floor from the per-element cost.
* ``compile``: a fresh process with an empty ``CUPY_CACHE_DIR`` (cold
  compilation), then a fresh process with that cache (warm on-disk cache).
* ``thomas``: the compiled one-thread-per-column Thomas prototype
  (``thomas_prototype.py``): exact agreement with the CuPy and NumPy solves on
  Jacobians captured from real steps (sunrise included), isolated solve
  timing, and the full step and ``run_block`` with the prototype patched in.
  FMA contraction off (expected bitwise) and on (rounding changes; needs
  approval) are both reported.
* ``diffusion``: ``implicit_column_diffusion`` at atmospheric shapes.

Timings are medians over ``--repeats`` with min and max. A launch-bound step
depends on the host CPU as much as on the GPU, which is why the CPU model and
vCPU count are recorded. Run nothing else on the machine at the same time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

MERCURY_GRID = {"kind": "latlon", "nlat": 90, "nlon": 180}
#: Gauss-grid column counts of the atmospheric spectral models (product grids).
GAUSS_GRIDS = {"T21 32x64": 2048, "T42 64x128": 8192, "T63 96x192": 18432}


# ------------------------------------------------------------------ helpers
def cp_module():
    import cupy as cp
    return cp


def sync():
    cp_module().cuda.Device().synchronize()


def stats(samples) -> dict:
    s = sorted(samples)
    return {"median": statistics.median(s), "min": s[0], "max": s[-1], "n": len(s)}


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    except Exception as exc:                                   # noqa: BLE001
        return f"unavailable: {exc!r}"


def production_config(**kw):
    from tropoi.run.airless.config import mercury_vpw99
    return mercury_vpw99(model=kw.pop("model", "TWO"), columns=kw.pop("columns", MERCURY_GRID), **kw)


# ------------------------------------------------------------------ environment
def section_environment(args, ctx) -> dict:
    cp = cp_module()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    cpu = "unknown"
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
             "--format=csv,noheader"], text=True).strip()
    except Exception as exc:                                   # noqa: BLE001
        smi = f"unavailable: {exc!r}"
    return {
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "multiprocessors": props["multiProcessorCount"],
        "gpu_memory_MiB": props["totalGlobalMem"] // 2 ** 20,
        "nvidia_smi": smi,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cpu_model": cpu, "vcpus": os.cpu_count(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "cupy": cp.__version__,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "precision": "FP64 throughout",
    }


# ------------------------------------------------------------------ state
def section_state(args, ctx) -> dict:
    """Spin the production grid up and keep the state (host arrays)."""
    from tropoi.run.airless.experiment import build_columns, build_substrate, run_block
    cfg = production_config()
    sub = build_substrate(cfg)
    lat, lon, w = build_columns(cfg)
    path = pathlib.Path(args.state_file)
    if path.exists():
        z = np.load(path)
        a, s = z["anomaly"], z["surface_anomaly"]
        how = f"loaded from {path}"
        first_step = int(z["next_step"])
    else:
        st = sub.initial_state(lat.size)
        a, s = np.asarray(st.anomaly), np.asarray(st.surface_anomaly)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            for c in range(args.spinup_cycles):
                d = pathlib.Path(tmp) / f"s{c}"
                d.mkdir()
                r = run_block(cfg.to_dict(), lat, lon, w, a, s, c, cfg.spinup_cycle_steps, False,
                              str(d), None, 0, "cupy")
                a, s = r["anomaly"], r["surface_anomaly"]
            d = pathlib.Path(tmp) / "p"
            d.mkdir()
            # production timestep, starting where the coarse cycles ended in time
            first_step = args.spinup_cycles * cfg.cycle_steps
            r = run_block(cfg.to_dict(), lat, lon, w, a, s, 0, args.production_steps, False,
                          str(d), None, 0, "cupy", dt=cfg.dt, first_step=first_step,
                          sample_every=(args.production_steps, args.production_steps))
            a, s = r["anomaly"], r["surface_anomaly"]
        first_step += args.production_steps
        how = (f"{args.spinup_cycles} coarse cycles of {cfg.spinup_cycle_steps} steps, then "
               f"{args.production_steps} production steps, from the configured "
               f"{cfg.initial_temperature} K isothermal start ({time.perf_counter() - t0:.0f} s)")
        np.savez(path, anomaly=a, surface_anomaly=s, next_step=first_step)
    ctx["state"] = (a, s, first_step)
    T_s = 300.0 + s
    return {"config_sha256": cfg.sha256(), "columns": int(lat.size), "layers": sub.layers,
            "origin": how, "next_step": first_step, "sha256": sha256_arrays(a, s),
            "surface_T_K": {"min": float(T_s.min()), "max": float(T_s.max())}}


def _forcing(cfg, cols, xp):
    from tropoi.physics.insolation import insolation_from_spec
    from tropoi.run.airless.experiment import build_columns
    ins = insolation_from_spec(cfg.illumination)
    lat, lon, _ = build_columns(cfg) if cols is None else cols
    la, lo = xp.asarray(np.radians(lat)), xp.asarray(np.radians(lon))
    n = (xp.cos(la) * xp.cos(lo), xp.cos(la) * xp.sin(lo), xp.sin(la))
    return lambda t: ins.incident(*n, t)[0]


# ------------------------------------------------------------------ end to end
def _time_run_block(cfg, ctx, steps, repeats, label_dir):
    from tropoi.run.airless.experiment import build_columns, run_block
    a, s, first = ctx["state"]
    lat, lon, w = build_columns(cfg)
    cadence = (cfg.cycle_steps // cfg.output["surface_samples"],
               cfg.cycle_steps // cfg.output["profile_samples"])
    times, iters = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(repeats + 1):                # repeat 0 warms (compilation)
            d = pathlib.Path(tmp) / f"{label_dir}{rep}"
            d.mkdir()
            sync()
            t = time.perf_counter()
            r = run_block(cfg.to_dict(), lat, lon, w, a, s, 0, steps, False, str(d), None, 0,
                          "cupy", dt=cfg.dt, first_step=first, sample_every=cadence)
            sync()
            if rep:
                times.append((time.perf_counter() - t) / steps)
                iters.append(float(np.mean(r["iterations"])))
    return times, iters, r


def _time_step(sub, ctx, forcing, dt, steps, repeats):
    cp = cp_module()
    a, s, first = ctx["state"]
    from tropoi.physics.substrate import SubstrateState
    times, iters = [], []
    for rep in range(repeats + 1):
        st = SubstrateState(cp.asarray(a), cp.asarray(s))
        it = []
        sync()
        t = time.perf_counter()
        for k in range(steps):
            st, r = sub.step(st, dt, shortwave_down=forcing((first + k + 1) * dt))
            it.append(r.iterations)
        sync()
        if rep:
            times.append((time.perf_counter() - t) / steps)
            iters.append(float(np.mean(it)))
    return times, iters, st


def section_end_to_end(args, ctx) -> dict:
    from tropoi.run.airless.experiment import build_substrate
    cp = cp_module()
    cfg = production_config()
    sub = build_substrate(cfg)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    times, iters, _ = _time_run_block(cfg, ctx, args.steps, args.repeats, "rb")
    rb_peak = pool.total_bytes()
    stimes, siters, _ = _time_step(sub, ctx, _forcing(cfg, None, cp), cfg.dt, args.steps,
                                   args.repeats)
    n = 16200
    med = statistics.median(times)
    return {"configuration": "Mercury VPW99 TWO production: 90x180 (16200 columns), 51 layers, "
                             "dt = cycle_s/4320, from the representative state",
            "steps_per_repeat": args.steps,
            "run_block_seconds_per_step": stats(times),
            "run_block_newton_iterations_mean": stats(iters),
            "run_block_column_steps_per_second": n / med,
            "run_block_production_cycle_minutes": med * cfg.cycle_steps / 60.0,
            "substrate_step_seconds_per_step": stats(stimes),
            "substrate_step_newton_iterations_mean": stats(siters),
            "outside_step_fraction_of_run_block": 1.0 - statistics.median(stimes) / med,
            "memory_pool_total_bytes_after_run_block": int(rb_peak),
            "device_memory_used_bytes": int(cp.cuda.runtime.memGetInfo()[1]
                                            - cp.cuda.runtime.memGetInfo()[0])}


# ------------------------------------------------------------------ counts
class _Counter:
    def __init__(self):
        self.calls = {"system_jacobian": 0, "system_trial": 0, "tridiagonal": 0, "steps": 0}


def _patch_counts(counter):
    import tropoi.physics.substrate as m
    orig_sys, orig_tri, orig_step = m.ConductiveSubstrate._system, m._tridiagonal, \
        m.ConductiveSubstrate.step

    def sys_(self, xp, th, ths, th_old, dt, forcing, jacobian):
        counter.calls["system_jacobian" if jacobian else "system_trial"] += 1
        return orig_sys(self, xp, th, ths, th_old, dt, forcing, jacobian)

    def tri(*a, **k):
        counter.calls["tridiagonal"] += 1
        return orig_tri(*a, **k)

    def step(self, *a, **k):
        counter.calls["steps"] += 1
        return orig_step(self, *a, **k)

    m.ConductiveSubstrate._system, m._tridiagonal, m.ConductiveSubstrate.step = sys_, tri, step

    def undo():
        m.ConductiveSubstrate._system, m._tridiagonal, m.ConductiveSubstrate.step = \
            orig_sys, orig_tri, orig_step
    return undo


def section_counts(args, ctx) -> dict:
    """Call counts per step. Per step the code synchronizes once to validate
    the forcing, once per Newton pass (the convergence test, including the
    final pass whose Jacobian is never used) and once per line-search trial,
    so syncs = 1 + jacobian_calls + trial_calls."""
    from tropoi.run.airless.experiment import build_substrate
    cp = cp_module()
    out = {}
    cases = {"production_state": None}
    for label in cases:
        cfg = production_config()
        sub = build_substrate(cfg)
        c = _Counter()
        undo = _patch_counts(c)
        try:
            _, iters, _ = _time_step(sub, ctx, _forcing(cfg, None, cp), cfg.dt, args.steps, 1)
        finally:
            undo()
        k = {name: v / c.calls["steps"] for name, v in c.calls.items() if name != "steps"}
        k["newton_iterations"] = iters[0]
        k["host_synchronizations"] = 1 + k["system_jacobian"] + k["system_trial"]
        k["unused_final_jacobians"] = 1.0
        k["trial_residuals_recomputed_with_jacobian"] = k["system_jacobian"] - 1.0
        out[label] = k
    out["reading"] = ("Each Newton pass assembles residual + Jacobian at the accepted trial "
                      "point, whose residual the line search has just computed; the last pass "
                      "assembles a Jacobian that is never used. Reusing them is "
                      "arithmetic-preserving (same function of the same arrays).")
    return out


# ------------------------------------------------------------------ components
def section_components(args, ctx) -> dict:
    import tropoi.physics.substrate as m
    from tropoi.run.airless.experiment import build_substrate
    cp = cp_module()
    cfg = production_config()
    sub = build_substrate(cfg)
    forcing = _forcing(cfg, None, cp)
    wall = {"system_jacobian": 0.0, "system_trial": 0.0, "tridiagonal": 0.0, "forcing": 0.0}
    events = {k: [] for k in wall}
    orig_sys, orig_tri = m.ConductiveSubstrate._system, m._tridiagonal

    def timed(label, fn, mode):
        def wrapper(*a, **k):
            if mode == "sync":
                sync(); t = time.perf_counter()
                r = fn(*a, **k)
                sync(); wall[label] += time.perf_counter() - t
                return r
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            r = fn(*a, **k)
            e1.record()
            events[label].append((e0, e1))
            return r
        return wrapper

    out = {}
    for mode in ("sync", "events"):
        def sys_(self, xp, th, ths, th_old, dt, forc, jacobian, _m=mode):
            lab = "system_jacobian" if jacobian else "system_trial"
            return timed(lab, orig_sys, _m)(self, xp, th, ths, th_old, dt, forc, jacobian)
        m.ConductiveSubstrate._system = sys_
        m._tridiagonal = timed("tridiagonal", orig_tri, mode)
        f = timed("forcing", forcing, mode)
        try:
            # _time_step runs a warm repeat and one measured repeat, each of
            # args.steps identical steps from the same state, and the wrappers
            # accumulate over both: hence the division by 2 * args.steps below.
            ts, _, _ = _time_step(sub, ctx, f, cfg.dt, args.steps, 1)
        finally:
            m.ConductiveSubstrate._system, m._tridiagonal = orig_sys, orig_tri
        if mode == "sync":
            out["synchronized_inclusive_ms_per_step"] = {
                k: 1e3 * v / (2 * args.steps) for k, v in wall.items()}
            out["synchronized_step_ms"] = 1e3 * ts[0]
        else:
            sync()
            out["cuda_event_span_ms_per_step"] = {
                k: sum(cp.cuda.get_elapsed_time(a, b) for a, b in v) / (2 * args.steps)
                for k, v in events.items()}
            out["event_instrumented_step_ms"] = 1e3 * ts[0]
    out["note"] = ("Synchronized inclusive times include the drain each synchronization "
                   "forces; event spans include device idle time while the host launches. "
                   "Neither set is an additive decomposition of the uninstrumented step.")
    return out


# ------------------------------------------------------------------ scaling
def _scaling_case(material, layers_stretch, ncol, forcing_kind):
    """A configuration and columns for one scaling point."""
    from tropoi.physics.materials import Stratum
    from tropoi.physics.substrate import ConductiveSubstrate
    from tropoi.run.airless.experiment import build_substrate
    rng = np.random.default_rng(ncol)
    if forcing_kind == "smooth":       # dayside, sun high: no terminator in the batch
        lat = rng.uniform(-30, 30, ncol)
        lon = rng.uniform(-30, 30, ncol) % 360
    else:                              # across the terminator (sunrise and sunset)
        lat = rng.uniform(-60, 60, ncol)
        lon = np.where(rng.random(ncol) < 0.5, 90.0, 270.0) + rng.uniform(-3, 3, ncol)
    cfg = production_config(columns={"kind": "points", "lat_deg": lat.tolist(),
                                     "lon_deg": lon.tolist()},
                            model="TWO" if material == "vpw99_two" else "TOP",
                            max_stretch=layers_stretch)
    if material == "constant":
        # THAI-like constant properties (C = 2e6, k = 2), test geometry only
        sub = ConductiveSubstrate([Stratum.volumetric(3.5, 2.0e6, 2.0)], top_thickness=0.02,
                                  max_stretch=layers_stretch, albedo=0.1, emissivity=0.95,
                                  initial_temperature=300.0)
    else:
        sub = build_substrate(cfg)
    return cfg, sub, (lat, lon, None)


def section_scaling(args, ctx) -> dict:
    cp = cp_module()
    cols = [256, 1024, 2048, 4096, 8192, 16200, 18432, 32768]
    if args.quick:
        cols = [256, 2048, 16200]
    stretches = {"vpw99_two": [1.2, 1.1, 1.05], "vpw99_top": [1.1], "constant": [1.1, 1.05]}
    rows = []
    for material, ss in stretches.items():
        for stretch in ss:
            for kind in ("smooth", "terminator"):
                for n in cols:
                    if material != "vpw99_two" and n not in (2048, 16200):
                        continue
                    cfg, sub, c = _scaling_case(material, stretch, n, kind)
                    st = sub.initial_state(n, xp=cp)
                    f = _forcing(cfg, c, cp)
                    # 24 warm steps from isothermal, then timed steps (state evolves)
                    for k in range(24):
                        st, _ = sub.step(st, cfg.dt, shortwave_down=f((k + 1) * cfg.dt))
                    ts, it = [], []
                    for rep in range(args.repeats):
                        sync(); t = time.perf_counter()
                        its = []
                        for k in range(args.scaling_steps):
                            st, r = sub.step(st, cfg.dt, shortwave_down=f((k + 25) * cfg.dt))
                            its.append(r.iterations)
                        sync()
                        ts.append((time.perf_counter() - t) / args.scaling_steps)
                        it.append(float(np.mean(its)))
                    med = statistics.median(ts)
                    rows.append({"material": material, "strata": len(sub.strata),
                                 "layers": sub.layers, "columns": n, "forcing": kind,
                                 "seconds_per_step": stats(ts),
                                 "newton_iterations_mean": statistics.median(it),
                                 "seconds_per_newton_iteration": med / max(statistics.median(it), 1e-9),
                                 "column_steps_per_second": n / med})
    # t/iteration = a + b L + c n L over the VPW99 TWO rows
    sel = [r for r in rows if r["material"] == "vpw99_two"]
    A = np.array([[1.0, r["layers"], r["columns"] * r["layers"]] for r in sel])
    y = np.array([r["seconds_per_newton_iteration"] for r in sel])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return {"rows": rows, "gauss_grid_columns": GAUSS_GRIDS,
            "fit_seconds_per_newton_iteration": {
                "model": "a + b*L + c*n*L (VPW99 TWO rows)",
                "a_s": coef[0], "b_s_per_layer": coef[1], "c_s_per_cell": coef[2],
                "max_relative_misfit": float(np.max(np.abs(pred - y) / y))}}


# ------------------------------------------------------------------ compile
_COMPILE_CHILD = r"""
import json, sys, time
sys.path[:0] = [{src!r}, {repo!r}]
t0 = time.perf_counter()
import cupy as cp
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import build_substrate
from tropoi.physics.insolation import insolation_from_spec
import numpy as np
cfg = mercury_vpw99(columns={{"kind": "latlon", "nlat": 90, "nlon": 180}})
sub = build_substrate(cfg)
ins = insolation_from_spec(cfg.illumination)
rng = np.random.default_rng(0)
la = cp.asarray(np.radians(rng.uniform(-89, 89, 16200))); lo = cp.asarray(np.radians(rng.uniform(0, 360, 16200)))
n = (cp.cos(la) * cp.cos(lo), cp.cos(la) * cp.sin(lo), cp.sin(la))
st = sub.initial_state(16200, xp=cp)
t_import = time.perf_counter() - t0
cp.cuda.Device().synchronize(); t = time.perf_counter()
st, _ = sub.step(st, cfg.dt, shortwave_down=ins.incident(*n, cfg.dt)[0]); cp.cuda.Device().synchronize()
first = time.perf_counter() - t
ws = []
for k in range(5):
    cp.cuda.Device().synchronize(); t = time.perf_counter()
    st, _ = sub.step(st, cfg.dt, shortwave_down=ins.incident(*n, (k + 2) * cfg.dt)[0]); cp.cuda.Device().synchronize()
    ws.append(time.perf_counter() - t)
print(json.dumps({{"import_and_setup_s": t_import, "first_step_s": first, "warm_step_s": sorted(ws)[2]}}))
"""


def section_compile(args, ctx) -> dict:
    code = _COMPILE_CHILD.format(src=str(REPO / "src"), repo=str(REPO))
    out = {}
    with tempfile.TemporaryDirectory() as cache:
        env = {**os.environ, "CUPY_CACHE_DIR": cache}
        for label in ("cold_empty_cache", "warm_disk_cache"):
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, timeout=1800)
            if r.returncode:
                out[label] = {"error": r.stderr[-2000:]}
                continue
            d = json.loads(r.stdout.strip().splitlines()[-1])
            d["first_step_overhead_s"] = d["first_step_s"] - d["warm_step_s"]
            out[label] = d
        out["cache_files"] = sum(len(f) for _, _, f in os.walk(cache))
    return out


# ------------------------------------------------------------------ thomas
def _capture_systems(sub, ctx, cfg, steps, forcing, start=None):
    """Newton systems (lower, diag, upper, rhs) captured from real steps."""
    import tropoi.physics.substrate as m
    cp = cp_module()
    captured = []
    orig = m._tridiagonal

    def grab(lower, diag, upper, rhs, xp):
        captured.append(tuple(x.copy() for x in (lower, diag, upper, rhs)))
        return orig(lower, diag, upper, rhs, xp)
    a, s, first = ctx["state"] if start is None else start
    from tropoi.physics.substrate import SubstrateState
    st = SubstrateState(cp.asarray(a), cp.asarray(s))
    m._tridiagonal = grab
    try:
        for k in range(steps):
            st, _ = sub.step(st, cfg.dt, shortwave_down=forcing((first + k + 1) * cfg.dt))
    finally:
        m._tridiagonal = orig
    return captured


def section_thomas(args, ctx) -> dict:
    import tropoi.physics.substrate as m
    import thomas_prototype as tp
    from tropoi.run.airless.experiment import build_substrate
    cp = cp_module()
    cfg = production_config()
    sub = build_substrate(cfg)
    forcing = _forcing(cfg, None, cp)
    systems = _capture_systems(sub, ctx, cfg, 4, forcing)
    # sunrise systems: terminator columns from an isothermal-free evolved state
    tcfg, tsub, tc = _scaling_case("vpw99_two", 1.1, 4096, "terminator")
    tst = tsub.initial_state(4096, xp=cp)
    tf = _forcing(tcfg, tc, cp)
    for k in range(48):
        tst, _ = tsub.step(tst, tcfg.dt, shortwave_down=tf((k + 1) * tcfg.dt))
    systems += _capture_systems(tsub, {"state": (tst.anomaly.get(), tst.surface_anomaly.get(), 48)},
                                tcfg, 4, tf)
    out = {"systems_compared": len(systems)}
    for fmad in (False, True):
        solve = tp.make_solver(fmad=fmad)
        worst_cupy, worst_numpy, equal_cupy, equal_numpy = 0.0, 0.0, 0, 0
        for (lo, d, up, r) in systems:
            x_k = solve(lo, d, up, r, cp).get()
            x_c = m._tridiagonal(lo, d, up, r, cp).get()
            x_n = m._tridiagonal(lo.get(), d.get(), up.get(), r.get(), np)
            equal_cupy += int(np.array_equal(x_k, x_c))
            equal_numpy += int(np.array_equal(x_k, x_n))
            scale = np.maximum(np.abs(x_c), 1e-300)
            worst_cupy = max(worst_cupy, float(np.max(np.abs(x_k - x_c) / scale)))
            worst_numpy = max(worst_numpy, float(np.max(np.abs(x_k - x_n) / scale)))
        out[f"fmad_{str(fmad).lower()}"] = {
            "bitwise_equal_to_cupy_systems": equal_cupy, "bitwise_equal_to_numpy_systems": equal_numpy,
            "max_relative_difference_vs_cupy": worst_cupy,
            "max_relative_difference_vs_numpy": worst_numpy}
    # isolated solve timing on a production system
    lo, d, up, r = systems[0]
    def bench(fn, reps=50):
        fn(); sync()
        ts = []
        for _ in range(args.repeats):
            sync(); t = time.perf_counter()
            for _ in range(reps):
                fn()
            sync(); ts.append((time.perf_counter() - t) / reps)
        return stats(ts)
    k_off = tp.make_solver(fmad=False)
    k_on = tp.make_solver(fmad=True)
    out["isolated_solve_seconds"] = {
        "cupy_python_thomas": bench(lambda: m._tridiagonal(lo, d, up, r, cp)),
        "compiled_fmad_false": bench(lambda: k_off(lo, d, up, r, cp)),
        "compiled_fmad_true": bench(lambda: k_on(lo, d, up, r, cp))}
    # full step and run_block with the prototype patched in
    orig = m._tridiagonal
    res = {}
    for label, fn in (("baseline", orig), ("compiled_fmad_false", k_off)):
        m._tridiagonal = fn
        try:
            st_t, _, st_end = _time_step(sub, ctx, forcing, cfg.dt, args.steps, args.repeats)
            rb_t, _, rb = _time_run_block(cfg, ctx, args.steps, args.repeats, label)
        finally:
            m._tridiagonal = orig
        res[label] = {"substrate_step_seconds_per_step": stats(st_t),
                      "run_block_seconds_per_step": stats(rb_t),
                      "end_state_sha256": sha256_arrays(rb["anomaly"], rb["surface_anomaly"])}
    res["end_states_bitwise_equal"] = (res["baseline"]["end_state_sha256"]
                                       == res["compiled_fmad_false"]["end_state_sha256"])
    res["run_block_speedup"] = (res["baseline"]["run_block_seconds_per_step"]["median"]
                                / res["compiled_fmad_false"]["run_block_seconds_per_step"]["median"])
    out["patched"] = res
    return out


# ------------------------------------------------------------------ diffusion
def section_diffusion(args, ctx) -> dict:
    from tropoi.physics.boundary_layer import implicit_column_diffusion
    cp = cp_module()
    out = {}
    for label, (nlev, ncol) in {"T21L40": (40, 2048), "T42L40": (40, 8192),
                                "T21L25": (25, 2048)}.items():
        rng = np.random.default_rng(1)
        q = cp.asarray(rng.normal(size=(nlev, ncol)))
        mass = cp.asarray(rng.uniform(100, 2000, (nlev, ncol)))
        g = cp.asarray(rng.uniform(0, 1, (nlev - 1, ncol)))
        gb = cp.asarray(rng.uniform(0, 0.1, ncol))
        fn = lambda: implicit_column_diffusion(q, mass, g, 1800.0, bottom_conductance=gb)
        fn(); sync()
        ts = []
        for _ in range(args.repeats):
            sync(); t = time.perf_counter()
            for _ in range(20):
                fn()
            sync(); ts.append((time.perf_counter() - t) / 20)
        out[label] = {"seconds_per_solve": stats(ts), "solves_per_boundary_layer_call": 3,
                      "note": "u and v share one matrix; dse uses the heat conductance"}
    return out


SECTIONS = {"environment": section_environment, "state": section_state,
            "end_to_end": section_end_to_end, "counts": section_counts,
            "components": section_components, "scaling": section_scaling,
            "compile": section_compile, "thomas": section_thomas,
            "diffusion": section_diffusion}
NEEDS_STATE = {"end_to_end", "counts", "components", "thomas"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sections", nargs="+", choices=[*SECTIONS, "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-file", default="substrate-benchmark-state.npz")
    ap.add_argument("--spinup-cycles", type=int, default=3)
    ap.add_argument("--production-steps", type=int, default=432)
    ap.add_argument("--steps", type=int, default=60, help="timed steps per repeat")
    ap.add_argument("--scaling-steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="smaller scaling sweep (development)")
    args = ap.parse_args(argv)
    names = list(SECTIONS) if "all" in args.sections else list(args.sections)
    if NEEDS_STATE & set(names) and "state" not in names:
        names.insert(0, "state")
    if "environment" not in names:
        names.insert(0, "environment")
    out_path = pathlib.Path(args.out)
    doc = {"tool": "tools/colab/substrate_benchmark.py", "argv": sys.argv[1:],
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ctx: dict = {}
    for name in names:
        t = time.perf_counter()
        try:
            doc[name] = SECTIONS[name](args, ctx)
        except Exception as exc:                               # noqa: BLE001
            import traceback
            doc[name] = {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]}
        doc.setdefault("section_wall_s", {})[name] = time.perf_counter() - t
        out_path.write_text(json.dumps(doc, indent=1, default=float))
        print(f"[{name}] {doc['section_wall_s'][name]:.0f} s", flush=True)
    doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.write_text(json.dumps(doc, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
