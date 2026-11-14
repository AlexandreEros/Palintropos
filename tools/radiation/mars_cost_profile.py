"""Measured cost and time-step probes for the Mars M-I CV-1 question.

Evidence tool for ``docs/radiation/MARS_MI_COST.md``. READ-ONLY with respect
to production behaviour: it builds the ``dev-mars-gray-flat-bl`` preset at a
chosen truncation and time step in a scratch directory, and its
instrumentation wraps callables of that one in-process model only. Nothing
it does changes a configuration hash, a preset or a module default.

Subcommands
-----------

``profile``
    1. Uninstrumented s/step: ``cupyx.profiler.benchmark`` over the
       stepper's ``step`` (warm-up excluded, RK4 start-up excluded), repeated
       in several trial blocks so block-to-block variance is visible.
    2. Operator breakdown: the same steps with a hierarchical timer that
       synchronizes the device at entry and exit of each wrapped callable
       (inclusive and exclusive times, call counts per step). The synced
       total is larger than the uninstrumented one; the breakdown is an
       attribution, the uninstrumented number is the cost.
    3. I/O: the daily sample (``RadiativeRun._sample``), the daily state
       write and a full checkpoint (``RadiativeRun._write_checkpoint``),
       each timed several times, amortized per step at the preset cadence.
    4. ``cProfile`` of uninstrumented steps: the host-side view (Python
       call counts per step, the hottest functions by own time).
    5. Micro-benchmarks of the transform primitives at this resolution.

``probe``
    A short run at one (truncation, dt) from a transferred developed state
    (a checkpoint of an earlier run, spectrally re-truncated or zero-padded;
    the slab T_s carried by spectral interpolation; the sigma grid
    unchanged). Re-starts the leapfrog with the stepper's RK4 start-up.
    Records, every ``--check-every`` steps: finiteness, grid max |u| and
    speed, T extremes, the high-degree share of vorticity variance, the
    advective Courant number N |V|max dt / a and the vertical Courant number
    max |sigma_dot| dt / d_sigma. "Unstable" = non-finite, or max speed
    above 500 m/s, or T outside Isca's valid_range_T [0, 700] K.

Usage (project venv, PYTHONPATH=src)::

    python tools/radiation/mars_cost_profile.py profile --truncation T21 \
        --dt 600 --out cost-T21.json
    python tools/radiation/mars_cost_profile.py probe --truncation T42 \
        --dt 450 --sols 0.5 --source docs/runs/<run>/checkpoints/<ckpt>.npz \
        --source-truncation T21 --out probe-T42-450.json

Truncations: T10, T15, T21 (the preset's own grid), T42 (64 x 128 Gauss,
l_max 43, retained 42: "store 43 / retain 42" like T21's 22 / 21).
"""
from __future__ import annotations

import argparse
import collections
import cProfile
import functools
import io
import json
import math
import pathlib
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

# (l_max, retained_truncation, nlat, nlon); T21 is the preset's own grid.
RESOLUTIONS = {
    "T10": (11, 10, 16, 32),
    "T15": (16, 15, 24, 48),
    "T21": (22, 21, 32, 64),
    "T42": (43, 42, 64, 128),
}
PRESET = "dev-mars-gray-flat-bl"
SOL = 88440.0


# ---------------------------------------------------------------------------
# configuration and environment
# ---------------------------------------------------------------------------

def build_config(truncation: str, dt: float, days: int = 1):
    from tropoi.run.radiative.config import PRESETS
    l_max, rt, nlat, nlon = RESOLUTIONS[truncation]
    cfg = PRESETS[PRESET]()
    return cfg.with_(experiment_id=f"cost-{truncation}-dt{dt:g}", l_max=l_max,
                     retained_truncation=rt, nlat=nlat, nlon=nlon, dt=float(dt), days=days)


def environment() -> dict[str, Any]:
    import cupy as cp
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:                                   # noqa: BLE001
        commit, dirty = None, None
    free, total = dev.mem_info
    return {"gpu": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "sm_count": int(props["multiProcessorCount"]),
            "gpu_memory_total_bytes": int(total), "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "numpy": np.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "git_commit": commit, "git_dirty_tracked": dirty}


def sync() -> None:
    import cupy as cp
    cp.cuda.Device().synchronize()


# ---------------------------------------------------------------------------
# hierarchical synced timer
# ---------------------------------------------------------------------------

class Timer:
    """Inclusive/exclusive wall time per label, device-synchronized at the
    entry and exit of every wrapped call (so GPU work is charged to the
    callable that launched it)."""

    def __init__(self, nvtx: bool = False):
        self.enabled = False
        self.nvtx = bool(nvtx)
        self.stack: list[list] = []                 # frames [label, child seconds]
        self.inclusive: dict[str, float] = collections.defaultdict(float)
        self.children: dict[str, float] = collections.defaultdict(float)
        self.calls: collections.Counter = collections.Counter()
        self.parents: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        # exclusive seconds keyed by (top-level operator under "step", label)
        self.by_operator: dict[tuple, float] = collections.defaultdict(float)

    def reset(self) -> None:
        self.inclusive.clear()
        self.children.clear()
        self.calls.clear()
        self.parents.clear()
        self.by_operator.clear()

    def run(self, label: str, fn: Callable, *args, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        if self.nvtx:
            # NVTX ranges only, no synchronization: for Nsight Systems
            # (``nsys profile``), which attributes GPU time per range itself
            from cupy.cuda import nvtx
            nvtx.RangePush(label)
            try:
                return fn(*args, **kwargs)
            finally:
                nvtx.RangePop()
        sync()
        t0 = time.perf_counter()
        parent = self.stack[-1][0] if self.stack else None
        frame = [label, 0.0]
        self.stack.append(frame)
        try:
            return fn(*args, **kwargs)
        finally:
            sync()
            elapsed = time.perf_counter() - t0
            self.stack.pop()
            self.inclusive[label] += elapsed
            self.calls[label] += 1
            self.parents[label][parent or "-"] += 1
            if parent is not None:
                self.children[parent] += elapsed
                self.stack[-1][1] += elapsed
            if len(self.stack) >= 2:
                top = self.stack[1][0]
            elif len(self.stack) == 1:
                top = label                     # a direct child of "step"
            else:
                top = "step"
            self.by_operator[(top, label)] += elapsed - frame[1]

    def wrap(self, label: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return self.run(label, fn, *args, **kwargs)
        return inner

    def report(self, steps: int) -> list[dict[str, Any]]:
        rows = []
        for label, inc in self.inclusive.items():
            exc = inc - self.children.get(label, 0.0)
            rows.append({"label": label, "calls_per_step": self.calls[label] / steps,
                         "inclusive_s_per_step": inc / steps,
                         "exclusive_s_per_step": exc / steps,
                         "parents": dict(self.parents[label])})
        rows.sort(key=lambda r: -r["exclusive_s_per_step"])
        return rows

    def operator_table(self, steps: int) -> dict[str, dict[str, float]]:
        """Per top-level operator: exclusive s/step in transform primitives
        (``sh.*``), in everything else, and the total."""
        table: dict[str, dict[str, float]] = {}
        for (top, label), sec in self.by_operator.items():
            row = table.setdefault(top, {"transforms": 0.0, "other": 0.0, "total": 0.0})
            row["transforms" if label.startswith("sh.") else "other"] += sec / steps
            row["total"] += sec / steps
        return table


class _Wrapped:
    """A timed stand-in for a stepper hook that still answers attribute
    lookups (signature(), state_dict(), max_rate, ...)."""

    def __init__(self, timer: Timer, label: str, target):
        self._timer, self._label, self._target = timer, label, target

    def __call__(self, *args, **kwargs):
        return self._timer.run(self._label, self._target, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._target, name)


def instrument(run, timer: Timer) -> None:
    """Wrap the operators of one RadiativeRun (its stepper, model, coupling,
    boundary layer) and the transform primitives (class level)."""
    from tropoi.spatial.transforms import fast_geodesic_sh as fsh
    import tropoi.temporal.tendencies.boundary_layer as blmod
    import tropoi.spatial.sigma_coordinate as sigmod

    model, stepper = run.model, run.stepper
    n_state = int(model.grid.n_points)
    l_state = int(model.l_max) if hasattr(model, "l_max") else int(run.cfg.l_max)

    def grid_name(sh) -> str:
        if int(sh.n_points) == n_state:
            return "state"
        return "product_ext" if int(sh.l_max) == l_state + 1 else "product"

    cls = fsh.PointSetSphericalHarmonics
    if not getattr(cls, "_cost_profile_wrapped", False):
        for meth in ("transform", "inv_transform", "transform_batch", "inv_transform_batch"):
            orig = getattr(cls, meth)

            def make(orig=orig, meth=meth):
                @functools.wraps(orig)
                def inner(self, *a, **k):
                    tm = getattr(cls, "_cost_profile_timer", None)
                    if tm is None:
                        return orig(self, *a, **k)
                    return tm.run(f"sh.{meth}[{grid_name(self)}]", orig, self, *a, **k)
                return inner
            setattr(cls, meth, make())
        cls._cost_profile_wrapped = True
    cls._cost_profile_timer = timer

    stepper.step = timer.wrap("step", stepper.step)
    stepper.tendency = timer.wrap("dynamics.tendency", stepper.tendency)
    stepper.explicit_terms = tuple(_Wrapped(timer, f"explicit.{type(t).__name__}", t)
                                   for t in stepper.explicit_terms)
    names = {id(run.boundary_layer): "boundary_layer", id(run.coupling): "column_coupling"}
    stepper.lagged_terms = tuple(_Wrapped(timer, names.get(id(t), type(t).__name__), t)
                                 for t in stepper.lagged_terms)
    stepper.operator.apply = timer.wrap("si.operator_apply", stepper.operator.apply)
    stepper.solver.advance = timer.wrap("si.solve_and_damp", stepper.solver.advance)
    model.so.vector_curl_div_spectral_batch = timer.wrap(
        "vector_curl_div_batch", model.so.vector_curl_div_spectral_batch)
    model._wind_from = timer.wrap("wind_from", model._wind_from)
    if run.coupling is not None:
        phys = run.coupling.physics
        phys.radiation.fluxes = timer.wrap("radiation.fluxes", phys.radiation.fluxes)
        if phys.insolation is not None:
            phys.insolation = _Wrapped(timer, "insolation", phys.insolation)
        phys.compute = timer.wrap("column_physics.compute", phys.compute)
        run.coupling.columns = timer.wrap("coupling.columns", run.coupling.columns)
        run.coupling._lowest_wind_speed = timer.wrap("coupling.lowest_wind",
                                                     run.coupling._lowest_wind_speed)
    for name in ("implicit_column_diffusion", "boundary_layer_depth", "diffusivity_profile",
                 "exchange_coefficient"):
        if not hasattr(getattr(blmod, name), "__wrapped__"):
            setattr(blmod, name, timer.wrap(f"bl.{name}", getattr(blmod, name)))
    if not hasattr(sigmod.hydrostatic_geopotential, "__wrapped__"):
        sigmod.hydrostatic_geopotential = timer.wrap("hydrostatic_geopotential",
                                                     sigmod.hydrostatic_geopotential)


def chunk_batched_transforms(rows: int) -> None:
    """BENCHMARK-ONLY: split the level-batched transform GEMMs into chunks of
    at most ``rows`` fields, so no single cuBLAS kernel outlives the Windows
    display-driver watchdog (TDR, 2 s) on a small GPU. Same arithmetic per
    output element; cuBLAS may pick other kernels, so the last bits can
    differ. Applied in-process to this tool's run only, and recorded."""
    import cupy as cp
    from tropoi.spatial.transforms import fast_geodesic_sh as fsh
    cls = fsh.PointSetSphericalHarmonics
    if getattr(cls, "_cost_profile_chunked", None) is not None:
        cls._cost_profile_chunked = rows
        return
    inv_orig, fwd_orig = cls.inv_transform_batch, cls.transform_batch

    def chunked(orig):
        @functools.wraps(orig)
        def inner(self, values, *a, **k):
            n = cls._cost_profile_chunked
            if not n or values.shape[0] <= n:
                return orig(self, values, *a, **k)
            return cp.concatenate([orig(self, values[i:i + n], *a, **k)
                                   for i in range(0, values.shape[0], n)], axis=0)
        return inner
    cls.inv_transform_batch = chunked(inv_orig)
    cls.transform_batch = chunked(fwd_orig)
    cls._cost_profile_chunked = rows


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

def _stats(values) -> dict[str, float]:
    v = [float(x) for x in values]
    return {"n": len(v), "mean": statistics.fmean(v),
            "std": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "median": statistics.median(v), "max": max(v)}


def _new_run(cfg, scratch: pathlib.Path):
    from tropoi.run.radiative.experiment import RadiativeRun
    return RadiativeRun(cfg, scratch / cfg.experiment_id, allow_dirty=True,
                        log=lambda _msg: None)


def micro_transforms(model, repeat: int) -> dict[str, Any]:
    """Each transform primitive once per repeat, synced (cupyx benchmark)."""
    import cupy as cp
    from cupyx.profiler import benchmark
    K = model.nlev
    out = {}
    sh_s = model.sh.sh if hasattr(model.sh, "sh") else model.sh
    sh_p = model._ps.sh
    sh_ext, _ = model.so._vector_analysis_pieces()
    # a throw-away pass first: the GPU clock ramps after host-only work, and
    # the first case timed would otherwise carry it (seen on the MX110)
    for name, sh in (("warm", sh_s), ("state", sh_s), ("product", sh_p),
                     ("product_ext", sh_ext)):
        n = sh.l_max + 1
        rng = cp.random.default_rng(1)
        f = rng.standard_normal((K, sh.n_points))
        c = (rng.standard_normal((K, n, n)) + 1j * rng.standard_normal((K, n, n)))
        cases = {"transform_batch_K": lambda: sh.transform_batch(f),
                 "inv_transform_batch_K": lambda: sh.inv_transform_batch(c),
                 "transform_scalar": lambda: sh.transform(f[0]),
                 "inv_transform_scalar": lambda: sh.inv_transform(c[0])}
        rec = {"n_points": int(sh.n_points), "l_max": int(sh.l_max),
               "n_basis": int(sh.n_basis),
               "Y_matrix_bytes": int(sh.Y_matrix.nbytes)}
        for case, fn in cases.items():
            r = benchmark(fn, n_repeat=repeat, n_warmup=3)
            rec[case] = {"gpu_s": _stats(r.gpu_times[0]), "cpu_s": _stats(r.cpu_times)}
        if name != "warm":
            out[name] = rec
    # the floor of one CuPy elementwise launch and of one device->host scalar
    tiny = cp.zeros(1)
    r = benchmark(lambda: tiny + 1.0, n_repeat=200, n_warmup=20)
    out["one_elementwise_launch"] = {"gpu_s": _stats(r.gpu_times[0]), "cpu_s": _stats(r.cpu_times)}
    r = benchmark(lambda: float(tiny.sum()), n_repeat=200, n_warmup=20)
    out["one_scalar_readback"] = {"gpu_s": _stats(r.gpu_times[0]), "cpu_s": _stats(r.cpu_times)}
    return out


class GpuSampler:
    """``nvidia-smi`` utilization and SM clock sampled every 100 ms while
    the timed steps run (utilization = share of the interval in which any
    kernel was executing; a host-bound loop shows it well below 100 %)."""

    def __init__(self):
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm",
                 "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None
        time.sleep(0.3)
        return self

    def __exit__(self, *exc):
        self.samples = []
        if self.proc is None:
            return False
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        for line in out.splitlines():
            try:
                u, c = (float(x) for x in line.split(","))
                self.samples.append((u, c))
            except ValueError:
                continue
        return False

    def summary(self) -> dict[str, Any]:
        s = self.samples[3:]                   # drop the start-up samples
        if not s:
            return {"available": False}
        return {"available": True, "utilization_percent": _stats([u for u, _ in s]),
                "sm_clock_mhz": _stats([c for _, c in s])}


def profile(args) -> dict[str, Any]:
    import cupy as cp
    from cupyx.profiler import benchmark
    cfg = build_config(args.truncation, args.dt)
    if args.chunk_rows:
        chunk_batched_transforms(args.chunk_rows)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="mars-cost-", dir=args.scratch))
    doc: dict[str, Any] = {"tool": "mars_cost_profile.py profile", "preset": PRESET,
                           "benchmark_only_gemm_chunk_rows": args.chunk_rows,
                           "truncation": args.truncation, "dt": cfg.dt,
                           "l_max": cfg.l_max, "retained": cfg.retained_truncation,
                           "nlat": cfg.nlat, "nlon": cfg.nlon,
                           "environment": environment()}
    t_build = time.perf_counter()
    run = _new_run(cfg, scratch)
    sync()
    doc["build_seconds"] = time.perf_counter() - t_build
    doc["grid"] = {"state_points": int(run.model.grid.n_points),
                   "product_points": int(run.model._ps.sh.n_points),
                   "nlev": int(run.model.nlev)}
    doc["startup_substeps"] = int(run.stepper.startup_substeps)

    # warm-up: RK4 start-up step + kernel compilation
    t0 = time.perf_counter()
    run.stepper.step()
    sync()
    doc["startup_step_seconds"] = time.perf_counter() - t0
    for _ in range(args.warmup):
        run.stepper.step()
    sync()

    # 1. uninstrumented
    trials = []
    with GpuSampler() as sampler:
        for _ in range(args.trials):
            r = benchmark(run.stepper.step, n_repeat=args.steps, n_warmup=0)
            trials.append({"gpu_s": _stats(r.gpu_times[0]), "cpu_s": _stats(r.cpu_times),
                           "gpu_s_each": [float(x) for x in r.gpu_times[0]]})
    doc["uninstrumented"] = {
        "trials": trials,
        "trial_means_gpu_s": _stats([t["gpu_s"]["mean"] for t in trials]),
        "all_steps_gpu_s": _stats([x for t in trials for x in t["gpu_s_each"]]),
        "gpu_sampler": sampler.summary(),
        "note": "gpu_s = CUDA-event time from the call's start to its end on the null "
                "stream, cpu_s = host time of the call; per step, n = steps per trial"}
    mem = cp.get_default_memory_pool()
    doc["memory"] = {"pool_used_bytes": int(mem.used_bytes()),
                     "pool_total_bytes": int(mem.total_bytes()),
                     "device_free_bytes": int(cp.cuda.Device(0).mem_info[0])}

    # 4. cProfile (uninstrumented)
    pr = cProfile.Profile()
    sync()
    pr.enable()
    for _ in range(args.cprofile_steps):
        run.stepper.step()
    sync()
    pr.disable()
    st = pstats.Stats(pr)
    rows = []
    total_calls = 0
    for (fname, line, func), (cc, nc, tt, ct, _callers) in st.stats.items():
        total_calls += nc
        rows.append({"function": f"{pathlib.Path(fname).name}:{line}({func})",
                     "calls_per_step": nc / args.cprofile_steps,
                     "own_s_per_step": tt / args.cprofile_steps,
                     "cumulative_s_per_step": ct / args.cprofile_steps})
    rows.sort(key=lambda r: -r["own_s_per_step"])
    doc["cprofile"] = {"steps": args.cprofile_steps,
                       "python_calls_per_step": total_calls / args.cprofile_steps,
                       "top_by_own_time": rows[:30]}

    # 2. instrumented breakdown
    timer = Timer()
    instrument(run, timer)
    timer.enabled = True
    for _ in range(2):                     # warm the wrappers
        run.stepper.step()
    timer.reset()
    block_totals = []
    for _ in range(args.trials):
        before = timer.inclusive.get("step", 0.0)
        for _ in range(args.steps):
            run.stepper.step()
        block_totals.append((timer.inclusive["step"] - before) / args.steps)
    n_instr = args.trials * args.steps
    doc["instrumented"] = {"steps": n_instr, "synced_step_s": _stats(block_totals),
                           "rows": timer.report(n_instr),
                           "by_operator": timer.operator_table(n_instr)}
    if args.nvtx_steps:
        # unsynced NVTX ranges over the same operators, for an independent
        # attribution by Nsight Systems: run the whole command under
        #   nsys profile --trace=cuda,nvtx -o <report> python ... --nvtx-steps N
        timer.nvtx = True
        sync()
        t = time.perf_counter()
        from cupy.cuda import nvtx as _nvtx
        _nvtx.RangePush("nvtx_block")
        for _ in range(args.nvtx_steps):
            run.stepper.step()
        sync()
        _nvtx.RangePop()
        doc["nvtx_block"] = {"steps": args.nvtx_steps,
                             "wall_s_per_step": (time.perf_counter() - t) / args.nvtx_steps,
                             "note": "ranges named as the synced labels; read the "
                                     "attribution from the nsys report, not from here"}
        timer.nvtx = False
    timer.enabled = False
    from tropoi.spatial.transforms import fast_geodesic_sh as fsh
    fsh.PointSetSphericalHarmonics._cost_profile_timer = None

    # 3. I/O, sample and checkpoint (synced)
    spd = cfg.steps_per_day
    io_rec: dict[str, Any] = {"steps_per_day": spd,
                              "checkpoint_every_days": cfg.checkpoint_every_days,
                              "state_every_days": cfg.state_every_days}
    samples, writes, ckpts = [], [], []
    from tropoi.run.held_suarez.checkpoint import atomic_write_bytes
    sdir = run.run_dir / "states"
    sdir.mkdir(exist_ok=True)
    for i in range(args.io_repeat):
        sync()
        t = time.perf_counter()
        run._sample(i + 1)
        sync()
        samples.append(time.perf_counter() - t)
        t = time.perf_counter()
        for name, arr in (("state", run.stepper.state),
                          ("surface", run.coupling.surface_temperature)):
            b = io.BytesIO()
            np.save(b, cp.asnumpy(arr))
            atomic_write_bytes(sdir / f"{name}-d{i + 1:05d}.npy", b.getvalue())
        writes.append(time.perf_counter() - t)
        t = time.perf_counter()
        run._write_checkpoint()
        ckpts.append(time.perf_counter() - t)
    io_rec["daily_sample_s"] = _stats(samples)
    io_rec["daily_state_write_s"] = _stats(writes)
    io_rec["checkpoint_s"] = _stats(ckpts)
    io_rec["amortized_s_per_step"] = (
        (statistics.fmean(samples) + statistics.fmean(writes) / cfg.state_every_days) / spd
        + statistics.fmean(ckpts) / (spd * cfg.checkpoint_every_days))
    doc["io"] = io_rec

    # 5. transform micro-benchmarks
    doc["micro_transforms"] = micro_transforms(run.model, args.micro_repeat)
    doc["scratch_dir"] = str(scratch)
    return doc


# ---------------------------------------------------------------------------
# state transfer and dt probes
# ---------------------------------------------------------------------------

def transfer_state(x_src: np.ndarray, ts_src: np.ndarray, src_truncation: str,
                   dst_truncation: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Spectral re-truncation / zero-padding of a PE state between two
    truncations on the same sigma grid, and the slab T_s by spectral
    interpolation (analysis on the source Gauss grid, synthesis on the
    destination one). Orthonormal coefficients mean the same (l, m) value
    is the same function at any l_max, so padding is exact."""
    import cupy as cp
    from tropoi.spatial.grids.latlon_grid import (GaussLatLonGridGeometry,
                                                  GaussLatLonSphericalHarmonics)
    ls, rs, nlat_s, nlon_s = RESOLUTIONS[src_truncation]
    ld, rd, nlat_d, nlon_d = RESOLUTIONS[dst_truncation]
    ns, nd = ls + 1, ld + 1
    rows = x_src.shape[0]
    info: dict[str, Any] = {}
    above_src = np.zeros_like(x_src)
    above_src[:, rs + 1:, :] = x_src[:, rs + 1:, :]
    above_src[:, :, rs + 1:] = x_src[:, :, rs + 1:]
    info["source_norm_above_retained_cut"] = float(np.linalg.norm(above_src))
    x = np.zeros((rows, nd, nd), dtype=x_src.dtype)
    m = min(ns, nd, rs + 1, rd + 1)
    x[:, :m, :m] = x_src[:, :m, :m]
    info["dropped_norm_fraction"] = float(
        math.sqrt(max(np.linalg.norm(x_src) ** 2 - np.linalg.norm(x) ** 2, 0.0))
        / np.linalg.norm(x_src))

    g_s = GaussLatLonGridGeometry(nlat_s, nlon_s, radius=1.0)
    g_d = GaussLatLonGridGeometry(nlat_d, nlon_d, radius=1.0)
    sh_s = GaussLatLonSphericalHarmonics(g_s, ls)
    sh_d = GaussLatLonSphericalHarmonics(g_d, ld)
    a = cp.asnumpy(sh_s.transform(cp.asarray(ts_src)))
    back = cp.asnumpy(sh_s.inv_transform(cp.asarray(a))).ravel()
    info["ts_source_roundtrip_max_abs_K"] = float(np.abs(back - ts_src).max())
    b = np.zeros((nd, nd), dtype=a.dtype)
    k = min(ns, nd)
    b[:k, :k] = a[:k, :k]
    ts = cp.asnumpy(sh_d.inv_transform(cp.asarray(b))).ravel()
    info["ts_source_range_K"] = [float(ts_src.min()), float(ts_src.max())]
    info["ts_destination_range_K"] = [float(ts.min()), float(ts.max())]
    return x, ts, info


def _check(model, cfg, x, sigma_dt):
    """Stability observables of one state."""
    import cupy as cp
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
    finite = bool(cp.isfinite(x).all())
    if not finite:
        return {"finite": False}
    st = PrimitiveEquationsState(x)
    u, v = model.wind_on_state_grid(st)
    T = model.temperature_on_state_grid(st)
    speed = cp.sqrt(u * u + v * v)
    K = model.nlev
    zeta = x[0:K]
    n = zeta.shape[1]
    l = cp.arange(n)[:, None]
    m_ = cp.arange(n)[None, :]
    weight = cp.where(m_ == 0, 1.0, 2.0) * (m_ <= l)
    power = (weight * cp.abs(zeta) ** 2).sum(axis=0)        # (n, n) summed over levels
    cut = cfg.retained_cut
    tail = float(power[int(0.8 * cut):, :].sum() / power.sum())
    cont = model.continuity_diagnostics(st)
    sd = cp.abs(cont["sigma_dot"][1:-1])                      # interior interfaces (K-1, npts)
    dsig = cp.asarray(sigma_dt)                               # (K-1,) min thickness either side
    vmax = float(speed.max())
    return {"finite": True, "max_abs_u": float(cp.abs(u).max()), "max_speed": vmax,
            "t_min": float(T.min()), "t_max": float(T.max()),
            "vorticity_tail_fraction": tail,
            "courant_horizontal": cut * vmax * cfg.dt / cfg.radius,
            "courant_vertical": float((sd / dsig[:, None]).max()) * cfg.dt}


def probe(args) -> dict[str, Any]:
    import cupy as cp
    from tropoi.run.held_suarez.checkpoint import read_checkpoint
    cfg = build_config(args.truncation, args.dt)
    if args.chunk_rows:
        chunk_batched_transforms(args.chunk_rows)
    arrays, meta = read_checkpoint(pathlib.Path(args.source))
    x_src = arrays["x_curr"]
    ts_src = arrays["surface_temperature"]
    x, ts, tinfo = transfer_state(x_src, ts_src, args.source_truncation, args.truncation)
    t0 = float(meta["day"]) * 86400.0
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="mars-probe-", dir=args.scratch))
    run = _new_run(cfg, scratch)
    run.stepper.initialize(cp.asarray(x), t0)
    run.coupling.surface_temperature = cp.asarray(ts)
    sig = run.model.sigma.thickness_array()
    sigma_dt = np.minimum(sig[:-1], sig[1:])
    nsteps = int(round(args.sols * SOL / cfg.dt))
    records = [{"step": 0, **_check(run.model, cfg, run.stepper.state, sigma_dt)}]
    status = "stable"
    t_wall = time.perf_counter()
    for i in range(1, nsteps + 1):
        run.stepper.step()
        if i % args.check_every == 0 or i == nsteps:
            rec = {"step": i, "t_sols": i * cfg.dt / SOL,
                   **_check(run.model, cfg, run.stepper.state, sigma_dt)}
            records.append(rec)
            if not rec["finite"] or rec["max_speed"] > 500.0 \
                    or not (0.0 <= rec["t_min"] and rec["t_max"] <= 700.0):
                status = "unstable"
                break
    sync()
    wall = time.perf_counter() - t_wall
    steps_done = records[-1]["step"]
    return {"tool": "mars_cost_profile.py probe", "preset": PRESET,
            "truncation": args.truncation, "dt": cfg.dt, "sols_requested": args.sols,
            "steps_requested": nsteps, "steps_done": steps_done, "status": status,
            "source": str(args.source), "source_truncation": args.source_truncation,
            "source_day": meta["day"], "t0_seconds": t0, "transfer": tinfo,
            "benchmark_only_gemm_chunk_rows": args.chunk_rows,
            "seconds_per_step": wall / max(steps_done, 1),
            "startup_substeps": int(run.stepper.startup_substeps),
            "unstable_rule": "non-finite, max speed > 500 m/s, or T outside [0, 700] K",
            "environment": environment(), "records": records,
            "scratch_dir": str(scratch)}


# ---------------------------------------------------------------------------
# projections (arithmetic only, from profile records)
# ---------------------------------------------------------------------------

CV1_SOLS = 1920
YEAR_SOLS = 59166360.0 / SOL


def ls_time_sols(ls_deg: float) -> float:
    """Time of solar longitude ``ls_deg`` in the first year, in sols from
    L_s = 0 (the preset's own orbit: e 0.0935, perihelion L_s 250.85)."""
    from tropoi.run.radiative.config import _time_of_perihelion_for_equinox_start
    period, e, ls_peri = 59166360.0, 0.0935, 250.85
    t_peri = _time_of_perihelion_for_equinox_start(period, e, ls_peri)
    nu = math.radians(ls_deg - ls_peri)
    ecc = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(nu / 2.0))
    mean = ecc - e * math.sin(ecc)
    return ((t_peri + mean / (2.0 * math.pi) * period) % period) / SOL


def project(args) -> dict[str, Any]:
    rows = []
    for path in args.profiles:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        sps = d["uninstrumented"]["all_steps_gpu_s"]["mean"]
        io_ps = d["io"]["amortized_s_per_step"]
        for dt in args.dt:
            steps = args.sols * SOL / dt
            wall = steps * (sps + io_ps * 600.0 / dt)   # I/O is per model day, not per step
            rows.append({"profile": pathlib.Path(path).name, "gpu": d["environment"]["gpu"],
                         "truncation": d["truncation"], "dt": dt, "sols": args.sols,
                         "steps": steps, "s_per_step": sps, "io_s_per_step_at_600": io_ps,
                         "wall_hours": wall / 3600.0, "wall_days": wall / 86400.0})
    window = (ls_time_sols(225.0), ls_time_sols(315.0))
    return {"tool": "mars_cost_profile.py project", "rows": rows,
            "arithmetic": "steps = sols * 88440 / dt; wall = steps * (s/step + "
                          "amortized I/O per step at dt 600 * 600 / dt)",
            "year_sols": YEAR_SOLS,
            "ls_225_315_window_sols_year1": window,
            "cv1_last_complete_window_year": int((CV1_SOLS - window[1]) // YEAR_SOLS) + 1}


def timescales(args) -> dict[str, Any]:
    """Physical memory times of the M-I configuration (CPU only): the slab
    C / (4 sigma T_s^3 + h) from a checkpoint's T_s, the optically thin gray
    longwave relaxation 1 / (8 sigma T^3 (tau0/p0) g / c_p), the surface
    spin-down H / (C_D |V|), and an exponential fit of the series' global
    mean T_s (its approach to the seasonal state)."""
    from tropoi.run.held_suarez.checkpoint import read_checkpoint
    sig = 5.670374419e-8
    cfg = build_config("T21", 600.0)
    C = float(cfg.surface["heat_capacity"])
    g, cp_air, r = cfg.gravity, cfg.cp, cfg.gas_constant
    tau0, p0 = cfg.longwave["tau0"], cfg.longwave["p0"]
    arrays, meta = read_checkpoint(pathlib.Path(args.checkpoint))
    ts = arrays["surface_temperature"]
    out: dict[str, Any] = {"tool": "mars_cost_profile.py timescales",
                           "checkpoint": str(args.checkpoint), "checkpoint_day": meta["day"],
                           "slab_heat_capacity": C}
    slab = {}
    for h in (0.0, 0.5):
        tau = C / (4 * sig * ts ** 3 + h) / SOL
        slab[f"sensible_{h}"] = {"min": float(tau.min()), "median": float(np.median(tau)),
                                 "max": float(tau.max())}
    out["slab_relaxation_sols"] = slab
    out["slab_ts_range_K"] = [float(ts.min()), float(ts.max())]
    out["slab_relaxation_sols_at_K"] = {str(T): C / (4 * sig * T ** 3) / SOL
                                        for T in (140, 150, 180, 210, 240, 285)}
    out["gray_lw_thin_relaxation_sols_at_K"] = {
        str(T): 1.0 / (8 * sig * T ** 3 * (tau0 / p0) * g / cp_air) / SOL
        for T in (150, 180, 210, 240)}
    H = r * 200.0 / g
    sig_int = np.asarray(cfg.vertical["sigma"])
    z1 = 0.5 * (sig_int[-1] - sig_int[-2]) * H          # lowest full level, roughly
    cd = (0.4 / math.log(z1 / cfg.boundary_layer["roughness_length"])) ** 2
    out["surface_spin_down_sols"] = {"z1_m": z1, "C_D_neutral": cd,
                                     **{str(v): H / (cd * v) / SOL for v in (5, 10, 20)}}
    s = json.loads(pathlib.Path(args.series).read_text(encoding="utf-8"))
    days = np.asarray(s["day"], dtype=float)
    mts = np.asarray(s["mean_surface_temperature"], dtype=float)
    sel = days >= args.fit_from_day
    best = None
    for tau_d in np.arange(5.0, 400.0, 0.25):        # least squares in (T_eq, A) per tau
        e = np.exp(-days[sel] / tau_d)
        A = np.vstack([np.ones_like(e), e]).T
        coef, res, *_ = np.linalg.lstsq(A, mts[sel], rcond=None)
        rss = float(((A @ coef - mts[sel]) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, tau_d, coef)
    rss, tau_d, coef = best
    out["mean_ts_exponential_fit"] = {
        "series": str(args.series), "fit_days": [float(days[sel][0]), float(days[sel][-1])],
        "tau_days": float(tau_d), "tau_sols": float(tau_d * 86400.0 / SOL),
        "T_asymptote_K": float(coef[0]), "amplitude_K": float(coef[1]),
        "rms_residual_K": math.sqrt(rss / sel.sum()),
        "caveat": "the seasonal forcing changes over the fit (L_s 0-29), so the asymptote "
                  "is not an equilibrium; the fit dates the memory, not the climate"}
    return out


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("profile")
    a.add_argument("--truncation", choices=sorted(RESOLUTIONS), required=True)
    a.add_argument("--dt", type=float, default=600.0)
    a.add_argument("--warmup", type=int, default=5)
    a.add_argument("--steps", type=int, default=10, help="steps per trial block")
    a.add_argument("--trials", type=int, default=3)
    a.add_argument("--cprofile-steps", type=int, default=5)
    a.add_argument("--io-repeat", type=int, default=3)
    a.add_argument("--micro-repeat", type=int, default=10)
    a.add_argument("--scratch", default=None)
    a.add_argument("--chunk-rows", type=int, default=0,
                   help="benchmark-only: split batched transform GEMMs (TDR on small GPUs)")
    a.add_argument("--nvtx-steps", type=int, default=0,
                   help="also run N steps with unsynced NVTX ranges (for nsys)")
    a.add_argument("--out", required=True)
    b = sub.add_parser("probe")
    b.add_argument("--truncation", choices=sorted(RESOLUTIONS), required=True)
    b.add_argument("--dt", type=float, required=True)
    b.add_argument("--sols", type=float, default=1.0)
    b.add_argument("--source", required=True, help="checkpoint .npz of an earlier run")
    b.add_argument("--source-truncation", choices=sorted(RESOLUTIONS), default="T21")
    b.add_argument("--check-every", type=int, default=10)
    b.add_argument("--scratch", default=None)
    b.add_argument("--chunk-rows", type=int, default=0,
                   help="benchmark-only: split batched transform GEMMs (TDR on small GPUs)")
    b.add_argument("--out", required=True)
    c = sub.add_parser("project")
    c.add_argument("profiles", nargs="+", help="profile JSON records")
    c.add_argument("--dt", type=float, nargs="+", default=[110.0, 600.0])
    c.add_argument("--sols", type=float, default=float(CV1_SOLS))
    c.add_argument("--out", required=True)
    t = sub.add_parser("timescales")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--series", required=True)
    t.add_argument("--fit-from-day", type=float, default=10.0)
    t.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "timescales":
        doc = timescales(args)
        pathlib.Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True),
                                          encoding="utf-8")
        print(json.dumps({k: doc[k] for k in ("slab_relaxation_sols", "mean_ts_exponential_fit",
                                               "surface_spin_down_sols")}, indent=1))
        return 0
    if args.cmd == "project":
        doc = project(args)
        pathlib.Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True),
                                          encoding="utf-8")
        for r in doc["rows"]:
            print(f"{r['gpu']:<24} {r['truncation']} dt {r['dt']:>6g}: {r['steps']:>10.0f} "
                  f"steps x {r['s_per_step']:.4f} s = {r['wall_hours']:8.1f} h "
                  f"({r['wall_days']:.1f} d)")
        return 0
    doc = profile(args) if args.cmd == "profile" else probe(args)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    if args.cmd == "profile":
        u = doc["uninstrumented"]["trial_means_gpu_s"]
        print(f"{args.truncation} dt {args.dt:g}: {u['mean']:.4f} +- {u['std']:.4f} s/step "
              f"(uninstrumented, {args.trials} x {args.steps} steps)")
        for r in doc["instrumented"]["rows"][:15]:
            print(f"  {r['label']:<40} {r['exclusive_s_per_step']:.4f} s excl  "
                  f"{r['calls_per_step']:.1f} calls")
    else:
        last = doc["records"][-1]
        print(f"{args.truncation} dt {args.dt:g}: {doc['status']} after {doc['steps_done']}"
              f"/{doc['steps_requested']} steps; last {json.dumps(last)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
