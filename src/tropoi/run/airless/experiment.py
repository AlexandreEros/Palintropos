"""Run an airless-body thermal experiment: independent conductive columns,
cycle by cycle, until periodic or the cycle limit is reached
(:meth:`AirlessRun.run`), or for a fixed number of steps
(:meth:`AirlessRun.run_transient`).

A transient run uses the production timestep ``cycle_s / cycle_steps``,
starts from the configured initial state with no spin-up, and is written in
segments of ``cycle_steps`` steps (the last one shorter when the step count
is not a multiple). It asserts no forcing recurrence and makes no
periodicity claim: its retained status is ``COMPLETED_TRANSIENT``.

The columns are independent, so they are split into blocks and advanced by
worker processes. Each column's arithmetic is elementwise, so a column's
result does not depend on how the columns are split.

Run directory::

    config.json      configuration, sha256, code and environment identity
    events.jsonl     created / cycle / periodic / transient / completed
    cycles.json      per-cycle periodicity and energy diagnostics
    grid.npz         latitude/longitude (deg), area weights, depth faces
    series/cycle-KKKK.npz   per-step area-weighted global scalars (tier 1)
    cycles/cycle-KKKK/      tier-2 samples, streamed by each worker into
                            memory-mapped per-block files (surface-bNNN.npy,
                            profiles-bNNN.npy, every_step-bNNN.npy, blocks.json,
                            integrals.npz, state.npz with the cycle's start and
                            end states); the last three cycles are kept.
                            Read them with load_tier2.
    retained/        the cycle the result is quoted from (tier 2), with
                     its status (PERIODIC, TRANSIENT or, for a transient
                     run, COMPLETED_TRANSIENT) in status.json
    SHA256SUMS

Tier 1 runs every step but is never stored per column per step: it keeps
per-column cycle integrals of every surface flux term, the residual sums
and tolerances, and per-step global scalars. Tier 2 holds, per cycle, the
surface fields at ``surface_samples`` phases, full T(z) at
``profile_samples`` phases and, for point runs, T(z) at every step.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import pathlib
import shutil
import time
from typing import Callable, Optional

import numpy as np

from tropoi.physics.insolation import insolation_from_spec
from tropoi.physics.substrate import ConductiveSubstrate
from tropoi.run.airless.config import AirlessConfig, strata_from_dicts
from tropoi.run.held_suarez.experiment import RunError, code_identity, environment_identity

__all__ = ["AirlessRun", "build_columns", "build_substrate", "load_tier2", "run_block",
           "transient_steps"]


def build_columns(cfg: AirlessConfig):
    """(lat_deg, lon_deg, area weights): weights are R^2 dlambda (sin phi_N -
    sin phi_S) for the regular grid (exact cell areas), 1 for points."""
    c = cfg.columns
    if c["kind"] == "latlon":
        nlat, nlon = int(c["nlat"]), int(c["nlon"])
        edges = np.linspace(-90.0, 90.0, nlat + 1)
        lat = 0.5 * (edges[1:] + edges[:-1])
        lon = (np.arange(nlon) + 0.5) * (360.0 / nlon)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        band = np.sin(np.radians(edges[1:])) - np.sin(np.radians(edges[:-1]))
        w = cfg.radius_m ** 2 * math.radians(360.0 / nlon) * np.broadcast_to(
            band[:, None], la.shape)
        return la.ravel(), lo.ravel(), np.array(w).ravel()
    if c["kind"] == "points":
        lat = np.asarray(c["lat_deg"], dtype=float)
        lon = np.asarray(c["lon_deg"], dtype=float)
        return lat, lon, np.ones_like(lat)
    raise ValueError(f"unknown column kind {c['kind']!r}")


def build_substrate(cfg: AirlessConfig) -> ConductiveSubstrate:
    return ConductiveSubstrate(strata_from_dicts(cfg.strata),
                               top_thickness=cfg.vertical["top_thickness"],
                               max_stretch=cfg.vertical["max_stretch"],
                               albedo=cfg.albedo, emissivity=cfg.emissivity,
                               bottom_heat_flux=cfg.geothermal_flux,
                               initial_temperature=cfg.initial_temperature)


def run_block(cfg_dict: dict, lat_deg, lon_deg, weights, anomaly, surface_anomaly,
              cycle: int, steps: int, record_profiles_every_step: bool,
              out_dir: str, prev_dir: Optional[str], block: int,
              backend: str = "numpy", dt: Optional[float] = None,
              first_step: Optional[int] = None, sample_every: Optional[tuple] = None):
    """Advance one block of columns through illumination cycle ``cycle``
    with ``steps`` steps.

    By default the timestep is ``cfg.cycle_s / steps`` and step ``n`` is
    forced at ``(cycle * steps + n + 1) * dt``. A transient segment passes
    ``dt`` and the global index ``first_step`` of its first step instead;
    step ``n`` is then forced at ``(first_step + n + 1) * dt``. With
    ``dt = cfg.cycle_s / steps`` and ``first_step = cycle * steps`` the two
    forms are the same arithmetic.

    By default the samples are the ``gcd`` of the configured sample count
    and ``steps``, evenly spaced (a coarse spin-up cycle samples at the
    phases both cadences share). ``sample_every = (surface, profile)``
    fixes the cadence in steps instead: a sample after every that many
    steps, ``steps // cadence`` of them. A transient passes its production
    cadence, so a short final segment keeps it; steps after the last sample
    of a segment are then not sampled (the end state is still returned).

    Tier-2 samples are streamed into this block's memory-mapped files in
    ``out_dir`` (``surface-bNNN.npy`` (samples, 5, ncol), ``profiles-bNNN.npy``
    (samples, L, ncol) and, for point runs, ``every_step-bNNN.npy``), so no
    process holds a whole cycle in memory. When ``prev_dir`` holds the
    previous cycle's files for the same block, sampled the same way, the
    periodicity differences are computed here as the samples are written.
    The function returns the final state, the tier-1 records and those
    differences.

    ``backend`` is ``"numpy"`` or ``"cupy"``. With CuPy the state, the
    forcing, the Newton iterations, the tridiagonal solves and every tier-1
    accumulator stay on the GPU. The host sees only the tier-2 samples, at
    their sample times, and the final results."""
    from numpy.lib.format import open_memmap
    from tropoi.physics.substrate import SubstrateState
    xp = _backend(backend)
    cfg = AirlessConfig.from_dict(cfg_dict)
    sub = build_substrate(cfg)
    ins = insolation_from_spec(cfg.illumination)
    lat, lon = xp.asarray(np.radians(lat_deg)), xp.asarray(np.radians(lon_deg))
    nx, ny, nz = xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat)
    weights = xp.asarray(weights, dtype=xp.float64)
    st = SubstrateState(xp.asarray(anomaly, dtype=xp.float64),
                        xp.asarray(surface_anomaly, dtype=xp.float64))
    dt = cfg.cycle_s / steps if dt is None else float(dt)
    first = cycle * steps if first_step is None else int(first_step)
    if sample_every is None:
        # coarse spin-up cycles sample at the phases both cadences share
        n_surf = math.gcd(int(cfg.output["surface_samples"]), steps)
        n_prof = math.gcd(int(cfg.output["profile_samples"]), steps)
        every_s, every_p = steps // n_surf, steps // n_prof
    else:
        every_s, every_p = (int(v) for v in sample_every)
        if every_s < 1 or every_p < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every!r}")
        n_surf, n_prof = steps // every_s, steps // every_p
    ncol, L = lat.size, sub.layers
    out = pathlib.Path(out_dir)
    tag = f"b{block:03d}"
    surf = open_memmap(out / f"surface-{tag}.npy", "w+", np.float64, (n_surf, 5, ncol))
    prof = open_memmap(out / f"profiles-{tag}.npy", "w+", np.float64, (n_prof, L, ncol))
    every = (open_memmap(out / f"every_step-{tag}.npy", "w+", np.float64, (steps, 1 + L, ncol))
             if record_profiles_every_step else None)
    prev_s = prev_p = None
    if prev_dir is not None:
        ps, pp = (pathlib.Path(prev_dir) / f"surface-{tag}.npy",
                  pathlib.Path(prev_dir) / f"profiles-{tag}.npy")
        if ps.exists() and pp.exists():
            prev_s, prev_p = np.load(ps, mmap_mode="r"), np.load(pp, mmap_mode="r")
            if prev_s.shape != surf.shape or prev_p.shape != prof.shape:
                prev_s = prev_p = None
    d_s = d_p = 0.0
    E0 = _host(sub.energy(st)).copy()
    sums = {k: xp.zeros(ncol) for k in ("absorbed", "emitted", "conducted", "bottom",
                                        "residual", "tolerance", "abs_conducted")}
    max_ratio = xp.zeros(ncol)
    ts_lo, ts_hi = np.inf, -np.inf
    extent_min, extent_max = xp.asarray(sub.extent(st)).T
    g_series = {k: xp.zeros(steps) for k in ("energy_tendency", "net", "residual",
                                             "tolerance")}
    iters = np.zeros(steps, dtype=np.int32)
    for n in range(steps):
        t = (first + n + 1) * dt                      # backward Euler: forcing at t^{n+1}
        sw, _ = ins.incident(nx, ny, nz, t)
        st, rep = sub.step(st, dt, shortwave_down=sw)
        lows, highs = zip(*rep.extent)
        xp.minimum(extent_min, xp.stack(lows), out=extent_min)
        xp.maximum(extent_max, xp.stack(highs), out=extent_max)
        net = rep.absorbed - rep.emitted + rep.bottom + rep.source
        sums["absorbed"] += dt * rep.absorbed
        sums["emitted"] += dt * rep.emitted
        sums["conducted"] += dt * rep.conducted
        sums["abs_conducted"] += dt * xp.abs(rep.conducted)
        sums["bottom"] += dt * rep.bottom
        sums["residual"] += dt * rep.residual_sum
        sums["tolerance"] += dt * rep.tolerance_sum
        xp.maximum(max_ratio, rep.max_residual_ratio, out=max_ratio)
        # device-side writes: no synchronization in the step loop
        g_series["energy_tendency"][n] = weights @ rep.energy_tendency
        g_series["net"][n] = weights @ net
        g_series["residual"][n] = weights @ rep.residual_sum
        g_series["tolerance"][n] = weights @ rep.tolerance_sum
        iters[n] = rep.iterations
        if (n + 1) % every_s == 0:
            k = (n + 1) // every_s - 1
            sample = _host(xp.stack([sub.surface_temperature(st), sw, rep.absorbed,
                                     rep.emitted, rep.conducted]))
            surf[k] = sample
            ts = sample[0]
            ts_lo, ts_hi = min(ts_lo, float(ts.min())), max(ts_hi, float(ts.max()))
            if prev_s is not None:
                d_s = max(d_s, float(np.max(np.abs(ts - prev_s[k, 0]))))
        if (n + 1) % every_p == 0:
            k = (n + 1) // every_p - 1
            T = _host(sub.temperature(st))
            prof[k] = T
            if prev_p is not None:
                d_p = max(d_p, float(np.max(np.abs(T - prev_p[k]))))
        if every is not None:
            every[n, 0] = _host(sub.surface_temperature(st))
            every[n, 1:] = _host(sub.temperature(st))
    T_end = _host(sub.temperature(st))
    drift = (float(np.max(np.abs(T_end[-1] - prev_p[-1, -1]))) if prev_p is not None else None)
    for arr in (surf, prof, every):
        if arr is not None:
            arr.flush()
    del surf, prof, every, prev_s, prev_p
    return {
        "config_sha256": cfg.sha256(), "dt": dt, "first_step": first,
        "anomaly": _host(st.anomaly), "surface_anomaly": _host(st.surface_anomaly),
        "energy_start": E0, "energy_end": _host(sub.energy(st)),
        "sums": {k: _host(v) for k, v in sums.items()}, "max_ratio": _host(max_ratio),
        "global": {k: _host(v) for k, v in g_series.items()}, "iterations": iters,
        "compared": drift is not None, "d_s": d_s, "d_p": d_p, "drift": drift,
        # None when no surface sample fell in this call (a short transient tail)
        "surface_range": (ts_lo, ts_hi) if ts_lo <= ts_hi else None,
        "extent": _host(xp.stack((extent_min, extent_max), axis=1)).tolist(),
    }


def _backend(name: str):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy  # noqa: WPS433
        return cupy
    raise ValueError(f"backend must be 'numpy' or 'cupy', not {name!r}")


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def load_tier2(cycle_dir, name: str, columns=None, mmap: bool = True):
    """A tier-2 array of a cycle directory, joining the per-block files
    along the column axis. ``columns`` (global column indices) restricts the
    read so that large runs never load everything. The block layout is in
    ``blocks.json``."""
    d = pathlib.Path(cycle_dir)
    mode = "r" if mmap else None
    cols = None if columns is None else np.asarray(columns)
    if cols is not None:
        if cols.ndim != 1 or (cols.size and not np.issubdtype(cols.dtype, np.integer)):
            raise IndexError("columns must be a one-dimensional sequence of integer indices")
        if not cols.size:
            cols = cols.astype(np.intp)
    if not (d / "blocks.json").exists():         # runs written before streaming
        a = np.load(d / f"{name}.npy", mmap_mode=mode)
        return np.asarray(a if cols is None else a[..., cols])
    ranges = json.loads((d / "blocks.json").read_text())["ranges"]
    if cols is not None:
        total = ranges[-1][1]
        if np.any(cols >= total) or np.any(cols < -total):
            raise IndexError("column index out of range")
        cols = cols.astype(np.intp, copy=True)
        cols[cols < 0] += total
    parts, positions = [], []
    for b, (lo, hi) in enumerate(ranges):
        f = d / f"{name}-b{b:03d}.npy"
        if cols is None:
            if f.exists():
                parts.append(np.asarray(np.load(f, mmap_mode=mode)))
        elif not cols.size and f.exists():
            return np.asarray(np.load(f, mmap_mode=mode)[..., :0])
        else:
            selected = np.flatnonzero((cols >= lo) & (cols < hi))
            if selected.size:
                a = np.load(f, mmap_mode=mode)
                parts.append(np.asarray(a[..., cols[selected] - lo]))
                positions.append(selected)
    joined = np.concatenate(parts, axis=-1)
    if cols is not None:
        joined = joined[..., np.argsort(np.concatenate(positions))]
    return joined


def transient_steps(cfg: AirlessConfig, *, steps: Optional[int] = None,
                    duration_s: Optional[float] = None) -> tuple:
    """``(steps, request)``: the step count of a transient at the production
    timestep ``dt = cycle_s / cycle_steps``, from exactly one of a step count
    or an elapsed time.

    An elapsed time is met by the first step that reaches it: ``ceil(T/dt)``
    steps, except that a time within 1e-9 relative of a whole number of
    steps is that number (so ``n * dt`` written back out gives ``n``). The
    run then ends at ``steps * dt``, which ``request`` records beside what
    was asked."""
    if (steps is None) == (duration_s is None):
        raise ValueError("give exactly one of steps and duration_s")
    dt = cfg.cycle_s / cfg.cycle_steps
    if steps is not None:
        n = int(steps)
        if n != steps or n < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}")
        return n, {"steps": n, "dt_s": dt, "final_time_s": n * dt}
    T = float(duration_s)
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError(f"duration_s must be finite and > 0, got {duration_s!r}")
    q = T / dt
    n = int(round(q)) if abs(q - round(q)) <= 1e-9 * max(1.0, q) else math.ceil(q)
    n = max(n, 1)
    return n, {"duration_s": T, "steps": n, "dt_s": dt, "final_time_s": n * dt,
               "rule": "first step reaching the requested time (1e-9 relative snap)"}


def _join_ranges(ranges):
    """The (min, max) over the blocks' ranges, skipping None; None if all are."""
    found = [r for r in ranges if r is not None]
    if not found:
        return None
    return (min(r[0] for r in found), max(r[1] for r in found))


def _call(args):
    return run_block(*args)


#: Aitken acceleration is applied only in the asymptotic regime: every
#: cycle-to-cycle change of the triplet below this (K), and each
#: extrapolated correction at most this (K).
AITKEN_MAX_CHANGE_K = 2.0
AITKEN_MAX_CORRECTION_K = 20.0


def aitken(x0, x1, x2, max_ratio=0.97):
    """Componentwise Aitken extrapolation of a geometrically converging
    sequence of cycle-end states: x2 + d2 r / (1 - r) with r = d2 / d1.

    It is applied only where 0 < r < ``max_ratio`` (a clean monotone decay),
    where both changes are below :data:`AITKEN_MAX_CHANGE_K` and where the
    correction stays within :data:`AITKEN_MAX_CORRECTION_K`. Elsewhere x2 is
    kept. This only accelerates spin-up; periodicity is still checked
    afterwards at the production timestep."""
    d1, d2 = x1 - x0, x2 - x1
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(d1) > 1e-12, d2 / d1, 0.0)
        corr = np.where(r < 1.0, d2 * r / (1.0 - r), np.inf)
    ok = ((r > 0.0) & (r < max_ratio) & (np.abs(d1) <= AITKEN_MAX_CHANGE_K)
          & (np.abs(d2) <= AITKEN_MAX_CHANGE_K) & (np.abs(corr) <= AITKEN_MAX_CORRECTION_K))
    out = x2.copy()
    out[ok] = x2[ok] + corr[ok]
    return out, float(np.mean(ok))


class AirlessRun:
    """One experiment directory (fresh runs only; a finished directory is
    never overwritten)."""

    def __init__(self, cfg: AirlessConfig, run_dir, *, workers: int = 1,
                 backend: str = "numpy", allow_dirty: bool = False,
                 log: Callable[[str], None] = print,
                 initial_state=None, initial_state_source: Optional[str] = None,
                 config_source: Optional[dict] = None):
        """``initial_state`` = (anomaly (L, ncol), surface_anomaly (ncol,)), in
        K relative to 300 K, optionally replaces the configured isothermal
        start (for example a verified periodic solution on another grid,
        interpolated). Its provenance goes into config.json as
        ``initial_state_source``. The periodicity criteria are unchanged.

        ``config_source`` records where the configuration came from when it
        was loaded from a saved document (``load_config_document``); it goes
        into config.json unchanged."""
        _backend(backend)                       # validate early
        self.backend = backend
        self.cfg = cfg
        self.dir = pathlib.Path(run_dir)
        self.workers = max(1, int(workers))
        self.log = log
        self.code = code_identity()
        if not allow_dirty and (self.code.get("git_commit") is None or self.code.get("git_dirty")):
            raise RunError("refusing to run from an unidentified or modified source tree "
                           "(pass allow_dirty for development runs)")
        if self.dir.exists() and any(self.dir.iterdir()):
            raise RunError(f"{self.dir} is not empty; airless runs never overwrite")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sub = build_substrate(cfg)
        self.lat, self.lon, self.w = build_columns(cfg)
        self._write_json("config.json", {
            "config": cfg.to_dict(), "config_sha256": cfg.sha256(), "code": self.code,
            "environment": environment_identity(), "substrate": self.sub.signature(),
            "workers": self.workers, "backend": self.backend,
            "initial_state_source": initial_state_source if initial_state is not None else None,
            "config_source": config_source,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.initial_state = initial_state
        np.savez(self.dir / "grid.npz", lat_deg=self.lat, lon_deg=self.lon, weights=self.w,
                 interfaces=self.sub.interfaces, centres=self.sub.centres)
        self._event("created", columns=int(self.lat.size), layers=self.sub.layers)

    # -- io -----------------------------------------------------------------------------

    def _write_json(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=_json_default))

    def _event(self, event, **fields):
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps({"event": event, "time": time.time(), **fields},
                                default=_json_default) + "\n")

    # -- execution ------------------------------------------------------------------------

    def _pool(self):
        """A worker pool, or None for one worker. Fork where the platform
        has it (not with CuPy); spawn otherwise (Windows, CuPy)."""
        if self.workers <= 1:
            return None
        method = ("fork" if self.backend != "cupy" and "fork" in mp.get_all_start_methods()
                  else "spawn")
        return mp.get_context(method).Pool(self.workers)

    def _start_state(self):
        if self.initial_state is not None:
            return (np.array(self.initial_state[0], dtype=float),
                    np.array(self.initial_state[1], dtype=float))
        st = self.sub.initial_state(self.lat.size)
        return (np.asarray(st.anomaly), np.asarray(st.surface_anomaly))

    def _cycle(self, pool, state, cycle, steps, record_every, prev_dir, dt=None,
               first_step=None, sample_every=None):
        n = self.lat.size
        blocks = [b for b in np.array_split(np.arange(n), self.workers) if b.size]
        d = self._mkdir(f"cycles/cycle-{cycle:04d}")
        (d / "blocks.json").write_text(json.dumps(
            {"ranges": [[int(b[0]), int(b[-1]) + 1] for b in blocks], "steps": steps}))
        cfgd = self.cfg.to_dict()
        args = [(cfgd, self.lat[b], self.lon[b], self.w[b], state[0][:, b], state[1][b],
                 cycle, steps, record_every, str(d), None if prev_dir is None else str(prev_dir), k,
                 self.backend, dt, first_step, sample_every)
                for k, b in enumerate(blocks)]
        parts = pool.map(_call, args) if pool is not None else [_call(a) for a in args]
        expected = self.cfg.sha256()
        received = sorted({p["config_sha256"] for p in parts})
        if received != [expected]:
            raise RunError(f"workers ran configuration(s) {received}, not {expected}")
        out = {"worker_config_sha256": expected,
               "worker_dt_s": parts[0]["dt"], "worker_first_step": parts[0]["first_step"],
               "anomaly": np.concatenate([p["anomaly"] for p in parts], axis=1),
               "surface_anomaly": np.concatenate([p["surface_anomaly"] for p in parts]),
               "energy_start": np.concatenate([p["energy_start"] for p in parts]),
               "energy_end": np.concatenate([p["energy_end"] for p in parts]),
               "max_ratio": np.concatenate([p["max_ratio"] for p in parts]),
               "iterations": np.max([p["iterations"] for p in parts], axis=0),
               "dir": d, "compared": all(p["compared"] for p in parts),
               "d_s": max(p["d_s"] for p in parts), "d_p": max(p["d_p"] for p in parts),
               "drift": (max(p["drift"] for p in parts)
                         if all(p["compared"] for p in parts) else None),
               "surface_range": _join_ranges([p["surface_range"] for p in parts])}
        out["sums"] = {k: np.concatenate([p["sums"][k] for p in parts]) for k in parts[0]["sums"]}
        out["global"] = {k: np.sum([p["global"][k] for p in parts], axis=0)
                         for k in parts[0]["global"]}
        ext = [p["extent"] for p in parts]
        out["extent"] = [(min(e[j][0] for e in ext), max(e[j][1] for e in ext))
                         for j in range(len(ext[0]))]
        # restart states of this cycle (tier 2): the start state is also what
        # an exact standalone replay of selected columns starts from (G-G)
        # dt and first_step let a standalone replay (G-G) use this cycle's own
        # timing; sample_every is stored only when the cadence was fixed
        timing = {"dt": out["worker_dt_s"], "first_step": out["worker_first_step"]}
        if sample_every is not None:
            timing["sample_every"] = np.asarray(sample_every, dtype=np.int64)
        np.savez(d / "state.npz", start_anomaly=state[0], start_surface_anomaly=state[1],
                 end_anomaly=out["anomaly"], end_surface_anomaly=out["surface_anomaly"],
                 cycle=cycle, steps=steps, **timing)
        np.savez(d / "integrals.npz", energy_start=out["energy_start"],
                 energy_end=out["energy_end"], max_ratio=out["max_ratio"], **out["sums"])
        kept = sorted((self.dir / "cycles").iterdir())
        for old in kept[:-3]:
            shutil.rmtree(old)
        return out

    def run(self, max_wall_s: Optional[float] = None) -> dict:
        cfg = self.cfg
        per = cfg.periodicity
        state = self._start_state()
        record_every = bool(cfg.output.get("every_step_profiles"))
        history, previous = [], None
        stage = "spinup" if cfg.spinup_cycle_steps else "production"
        passes = 0
        cycle = 0
        ends = []                       # spin-up cycle-end states, for acceleration
        status = "TRANSIENT"
        t_start = time.time()
        pool = self._pool()
        try:
            while True:
                steps = cfg.spinup_cycle_steps if stage == "spinup" else cfg.cycle_steps
                t0 = time.time()
                res = self._cycle(pool, state, cycle, steps,
                                  record_every and stage == "production",
                                  None if previous is None else previous["dir"])
                state = (res["anomaly"], res["surface_anomaly"])
                diag = self._diagnostics(res, steps, stage, cycle)
                diag["wall_s"] = time.time() - t0
                history.append(diag)
                self._write_json("cycles.json", history)
                np.savez(self._mkdir("series") / f"cycle-{cycle:04d}.npz",
                         iterations=res["iterations"], **res["global"])
                self._event("cycle", **{k: v for k, v in diag.items()
                                        if not isinstance(v, (list, dict))})
                self.log(f"cycle {cycle} [{stage}] {diag['periodic']} "
                         f"dTs={diag.get('max_surface_change_K')} "
                         f"dT={diag.get('max_subsurface_change_K')} "
                         f"dE/P={diag['max_energy_imbalance_W_m2']:.3g} "
                         f"({diag['wall_s']:.0f} s)")
                def accelerate():
                    nonlocal state, ends, res
                    ends.append(state)
                    if per.get("spinup_acceleration") and len(ends) == 3:
                        a, fa = aitken(ends[0][0], ends[1][0], ends[2][0])
                        b, _ = aitken(ends[0][1], ends[1][1], ends[2][1])
                        state = (a, b)
                        self._event("aitken", cycle=cycle, stage=stage,
                                    fraction_extrapolated=fa)
                        ends = []
                        res = None          # never compare across an extrapolation
                        return True
                    return False

                if stage == "production":
                    passes = passes + 1 if diag["periodic"] else 0
                    if passes >= per["production_cycles_required"]:
                        status = "PERIODIC"
                        break
                    n_prod = sum(1 for h in history if h["stage"] == "production")
                    if n_prod >= per["max_production_cycles"]:
                        break
                    # Re-equilibration after the coarse-to-production switch may
                    # be accelerated too. Each extrapolation restarts the chain,
                    # so the claim always rests on unaccelerated consecutive
                    # production cycles.
                    if not diag["periodic"] and accelerate():
                        passes = 0
                else:
                    accelerate()
                    n_spin = sum(1 for h in history if h["stage"] == "spinup")
                    out_of_time = (max_wall_s is not None
                                   and time.time() - t_start > max_wall_s)
                    if diag["periodic"] or n_spin >= per["max_spinup_cycles"] or out_of_time:
                        # the first production cycle is never compared with a
                        # coarse one: the claim needs production-dt pairs
                        stage = "production"
                        res = None
                        ends = []
                if (stage == "production" and max_wall_s is not None
                        and time.time() - t_start > max_wall_s and res is not None):
                    self._event("wall_limit")
                    break
                previous = res
                cycle += 1
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        last = history[-1]
        retained = {"status": status, "cycle": last["cycle"],
                    "stage": last["stage"], "consecutive_periodic_production_cycles": passes,
                    "label": (f"PERIODIC (G-P2 met in {passes} consecutive production cycles)"
                              if status == "PERIODIC" else
                              f"TRANSIENT (cycle {last['cycle']}; G-P2 not met)")}
        src = self.dir / "cycles" / f"cycle-{last['cycle']:04d}"
        if src.exists():
            shutil.copytree(src, self.dir / "retained")
        else:
            self._mkdir("retained")
        (self.dir / "retained" / "status.json").write_text(json.dumps(retained, indent=1))
        self._event("completed", status=status)
        self._checksums()
        return retained

    def run_transient(self, steps: int, request: Optional[dict] = None) -> dict:
        """Advance the columns from the start state for exactly ``steps``
        production timesteps, ``dt = cycle_s / cycle_steps``, and stop.

        There is no spin-up stage, no acceleration and no periodicity test,
        and no forcing recurrence is assumed. The steps are written in
        segments of ``cycle_steps`` (the last one shorter if needed), each
        with the same tier-1 and tier-2 output and the same numerical
        diagnostics as a cycle of :meth:`run`. Step ``m`` (from 0) is forced
        at ``(m + 1) * dt``, which for whole segments is the same arithmetic
        as cycle ``c`` of :meth:`run` at the production timestep. Every
        segment samples at the production cadence (``cycle_steps /
        surface_samples`` and ``cycle_steps / profile_samples`` steps), so
        a short final segment keeps it; steps after its last sample are not
        sampled, and a segment shorter than the cadence has no samples
        (its surface range is then None).

        ``request`` records how the stopping point was asked for (see
        :func:`transient_steps`). The retained status is
        ``COMPLETED_TRANSIENT``. It reports the solver diagnostics and claims
        nothing beyond them: not numerical reproduction (no reference is
        compared), not periodic equilibrium, not physical validity."""
        cfg = self.cfg
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
            raise ValueError(f"a transient needs a positive whole number of steps, "
                             f"got {steps!r}")
        total = int(steps)
        seg_len = int(cfg.cycle_steps)
        dt = cfg.cycle_s / seg_len
        # the production sampling cadence, kept by every segment (a short final
        # segment included) instead of the spin-up gcd rule
        cadence = (seg_len // int(cfg.output["surface_samples"]),
                   seg_len // int(cfg.output["profile_samples"]))
        spec = {"steps": total, "dt_s": dt, "final_time_s": total * dt,
                "segment_steps": seg_len, "segments": -(-total // seg_len),
                "sample_every_steps": {"surface": cadence[0], "profile": cadence[1]},
                "start": "configured initial state" if self.initial_state is None
                         else "supplied initial state",
                "spinup": "none (spinup_cycle_steps is not used by a transient)",
                "request": request or {"steps": total}}
        self._event("transient", **spec)
        state = self._start_state()
        record_every = bool(cfg.output.get("every_step_profiles"))
        history, done, seg = [], 0, 0
        pool = self._pool()
        try:
            while done < total:
                k = min(seg_len, total - done)
                t0 = time.time()
                res = self._cycle(pool, state, seg, k, record_every, None, dt=dt,
                                  first_step=done, sample_every=cadence)
                state = (res["anomaly"], res["surface_anomaly"])
                diag = self._diagnostics(res, k, "transient", seg,
                                         duration=cfg.cycle_s * (k / seg_len))
                diag.update(periodic=None, first_step=done, t_start_s=done * dt,
                            t_end_s=(done + k) * dt,
                            worker_config_sha256=res["worker_config_sha256"],
                            worker_dt_s=res["worker_dt_s"],
                            worker_first_step=res["worker_first_step"],
                            wall_s=time.time() - t0)
                history.append(diag)
                self._write_json("cycles.json", history)
                np.savez(self._mkdir("series") / f"cycle-{seg:04d}.npz",
                         iterations=res["iterations"], **res["global"])
                self._event("segment", **{k_: v for k_, v in diag.items()
                                          if not isinstance(v, (list, dict))})
                self.log(f"segment {seg} steps {done}-{done + k - 1} "
                         f"t={(done + k) * dt:.6g} s "
                         f"dE/P={diag['max_energy_imbalance_W_m2']:.3g} "
                         f"r/tau={diag['max_cell_residual_ratio']:.3g} "
                         f"newton<={diag['max_newton_iterations']} ({diag['wall_s']:.0f} s)")
                done += k
                seg += 1
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        last = history[-1]
        retained = {
            "status": "COMPLETED_TRANSIENT", "mode": "transient", "cycle": last["cycle"],
            "stage": "transient", "steps": total, "dt_s": dt, "final_time_s": total * dt,
            "segments": seg, "request": spec["request"],
            "config_sha256": cfg.sha256(),
            "label": f"COMPLETED_TRANSIENT ({total} steps of {dt:.9g} s, "
                     f"t = {total * dt:.9g} s; periodicity not evaluated)",
            "claims": {
                "solver_diagnostics": "reported, not judged: per-step Newton convergence "
                                      "(max |r|/tau) and energy-identity residuals per "
                                      "segment in cycles.json",
                "numerical_reproduction": "not evaluated (this run makes no comparison "
                                          "with a reference)",
                "periodic_equilibrium": "not evaluated (a transient asserts no recurrence)",
                "physical_validation": "none"}}
        src = self.dir / "cycles" / f"cycle-{last['cycle']:04d}"
        shutil.copytree(src, self.dir / "retained")
        (self.dir / "retained" / "status.json").write_text(json.dumps(retained, indent=1))
        self._event("completed", status=retained["status"])
        self._checksums()
        return retained

    def _mkdir(self, name):
        p = self.dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _diagnostics(self, res, steps, stage, cycle, duration=None):
        """Per-cycle diagnostics. Rates are energy over ``duration`` (s),
        which defaults to ``cycle_s``; a short transient segment passes its
        own length."""
        cfg, per = self.cfg, self.cfg.periodicity
        P = cfg.cycle_s if duration is None else duration
        dE = (res["energy_end"] - res["energy_start"]) / P
        s = res["sums"]
        w = self.w / self.w.sum()
        diag = {"cycle": cycle, "stage": stage, "steps": steps, "dt_s": P / steps,
                "max_energy_imbalance_W_m2": float(np.max(np.abs(dE))),
                "mean_absorbed_W_m2": float(w @ (s["absorbed"] / P)),
                "mean_emitted_W_m2": float(w @ (s["emitted"] / P)),
                "global_residual_over_tolerance": float(
                    abs(w @ s["residual"]) / max(w @ s["tolerance"], 1e-300)),
                "max_cell_residual_ratio": float(np.max(res["max_ratio"])),
                "max_newton_iterations": int(np.max(res["iterations"])),
                "extent_K": res["extent"], "extent_scope": "cycle",
                "surface_T_range_K": (None if res["surface_range"] is None
                                      else list(res["surface_range"]))}
        periodic = False
        if res["compared"]:
            diag.update(max_surface_change_K=res["d_s"], max_subsurface_change_K=res["d_p"],
                        bottom_drift_K=res["drift"])
            periodic = (res["d_s"] <= per["surface_K"] and res["d_p"] <= per["subsurface_K"]
                        and res["drift"] <= per["bottom_drift_K"]
                        and diag["max_energy_imbalance_W_m2"] <= per["energy_W_m2"])
        diag["periodic"] = periodic
        return diag

    def _checksums(self):
        lines = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                lines.append(f"{h}  {p.relative_to(self.dir)}")
        (self.dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
