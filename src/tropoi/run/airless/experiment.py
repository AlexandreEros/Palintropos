"""Run an airless-body thermal experiment: independent conductive columns,
cycle by cycle, until periodic or the cycle limit is reached.

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
                            integrals.npz); the last three cycles are kept.
                            Read them with load_tier2.
    retained/        the cycle the result is quoted from (tier 2), with
                     its status (PERIODIC or TRANSIENT) in status.json
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

__all__ = ["AirlessRun", "build_columns", "build_substrate", "load_tier2", "run_block"]


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
              out_dir: str, prev_dir: Optional[str], block: int):
    """Advance one block of columns through illumination cycle ``cycle``
    with ``steps`` steps.

    Tier-2 samples are streamed into this block's memory-mapped files in
    ``out_dir`` (``surface-bNNN.npy`` (samples, 5, ncol), ``profiles-bNNN.npy``
    (samples, L, ncol) and, for point runs, ``every_step-bNNN.npy``), so no
    process holds a whole cycle in memory. When ``prev_dir`` holds the
    previous cycle's files for the same block, sampled the same way, the
    periodicity differences are computed here as the samples are written.
    The function returns the final state, the tier-1 records and those
    differences."""
    from numpy.lib.format import open_memmap
    from tropoi.physics.substrate import SubstrateState
    cfg = AirlessConfig.from_dict(cfg_dict)
    sub = build_substrate(cfg)
    ins = insolation_from_spec(cfg.illumination)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    nx, ny, nz = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    st = SubstrateState(np.array(anomaly, dtype=float), np.array(surface_anomaly, dtype=float))
    dt = cfg.cycle_s / steps
    # coarse spin-up cycles sample at the phases both cadences share
    n_surf = math.gcd(int(cfg.output["surface_samples"]), steps)
    n_prof = math.gcd(int(cfg.output["profile_samples"]), steps)
    every_s, every_p = steps // n_surf, steps // n_prof
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
    E0 = np.asarray(sub.energy(st)).copy()
    sums = {k: np.zeros(ncol) for k in ("absorbed", "emitted", "conducted", "bottom",
                                        "residual", "tolerance", "abs_conducted")}
    max_ratio = np.zeros(ncol)
    ts_lo, ts_hi = np.inf, -np.inf
    g_series = {k: np.zeros(steps) for k in ("energy_tendency", "net", "residual",
                                             "tolerance")}
    iters = np.zeros(steps, dtype=np.int32)
    for n in range(steps):
        t = (cycle * steps + n + 1) * dt              # backward Euler: forcing at t^{n+1}
        sw, _ = ins.incident(nx, ny, nz, t)
        st, rep = sub.step(st, dt, shortwave_down=sw)
        net = rep.absorbed - rep.emitted + rep.bottom + rep.source
        sums["absorbed"] += dt * rep.absorbed
        sums["emitted"] += dt * rep.emitted
        sums["conducted"] += dt * rep.conducted
        sums["abs_conducted"] += dt * np.abs(rep.conducted)
        sums["bottom"] += dt * rep.bottom
        sums["residual"] += dt * rep.residual_sum
        sums["tolerance"] += dt * rep.tolerance_sum
        np.maximum(max_ratio, rep.max_residual_ratio, out=max_ratio)
        g_series["energy_tendency"][n] = float(weights @ rep.energy_tendency)
        g_series["net"][n] = float(weights @ net)
        g_series["residual"][n] = float(weights @ rep.residual_sum)
        g_series["tolerance"][n] = float(weights @ rep.tolerance_sum)
        iters[n] = rep.iterations
        if (n + 1) % every_s == 0:
            k = (n + 1) // every_s - 1
            ts = sub.surface_temperature(st)
            surf[k] = np.stack([ts, sw, rep.absorbed, rep.emitted, rep.conducted])
            ts_lo, ts_hi = min(ts_lo, float(ts.min())), max(ts_hi, float(ts.max()))
            if prev_s is not None:
                d_s = max(d_s, float(np.max(np.abs(ts - prev_s[k, 0]))))
        if (n + 1) % every_p == 0:
            k = (n + 1) // every_p - 1
            T = sub.temperature(st)
            prof[k] = T
            if prev_p is not None:
                d_p = max(d_p, float(np.max(np.abs(T - prev_p[k]))))
        if every is not None:
            every[n, 0] = sub.surface_temperature(st)
            every[n, 1:] = sub.temperature(st)
    T_end = sub.temperature(st)
    drift = (float(np.max(np.abs(T_end[-1] - prev_p[-1, -1]))) if prev_p is not None else None)
    for arr in (surf, prof, every):
        if arr is not None:
            arr.flush()
    del surf, prof, every, prev_s, prev_p
    return {
        "anomaly": st.anomaly, "surface_anomaly": st.surface_anomaly,
        "energy_start": E0, "energy_end": np.asarray(sub.energy(st)),
        "sums": sums, "max_ratio": max_ratio, "global": g_series, "iterations": iters,
        "compared": drift is not None, "d_s": d_s, "d_p": d_p, "drift": drift,
        "surface_range": (ts_lo, ts_hi), "extent": sub.extent(st),
    }


def load_tier2(cycle_dir, name: str, columns=None, mmap: bool = True):
    """A tier-2 array of a cycle directory, joining the per-block files
    along the column axis. ``columns`` (global column indices) restricts the
    read so that large runs never load everything. The block layout is in
    ``blocks.json``."""
    d = pathlib.Path(cycle_dir)
    if not (d / "blocks.json").exists():         # runs written before streaming
        a = np.load(d / f"{name}.npy", mmap_mode="r" if mmap else None)
        return np.asarray(a if columns is None else a[..., np.asarray(columns)])
    layout = json.loads((d / "blocks.json").read_text())
    parts = []
    for b, (lo, hi) in enumerate(layout["ranges"]):
        f = d / f"{name}-b{b:03d}.npy"
        if not f.exists():
            continue
        a = np.load(f, mmap_mode="r" if mmap else None)
        if columns is None:
            parts.append(np.asarray(a))
        else:
            cols = np.asarray(columns)
            sel = cols[(cols >= lo) & (cols < hi)] - lo
            if sel.size:
                parts.append(np.asarray(a[..., sel]))
    if columns is not None:
        # keep the order of ``columns``
        joined = np.concatenate(parts, axis=-1)
        order = np.argsort(np.argsort(np.asarray(columns)))
        return joined[..., order]
    return np.concatenate(parts, axis=-1)


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
                 allow_dirty: bool = False, log: Callable[[str], None] = print):
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
            "workers": self.workers, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
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

    def _cycle(self, pool, state, cycle, steps, record_every, prev_dir):
        n = self.lat.size
        blocks = [b for b in np.array_split(np.arange(n), self.workers) if b.size]
        d = self._mkdir(f"cycles/cycle-{cycle:04d}")
        (d / "blocks.json").write_text(json.dumps(
            {"ranges": [[int(b[0]), int(b[-1]) + 1] for b in blocks], "steps": steps}))
        cfgd = self.cfg.to_dict()
        args = [(cfgd, self.lat[b], self.lon[b], self.w[b], state[0][:, b], state[1][b],
                 cycle, steps, record_every, str(d), None if prev_dir is None else str(prev_dir), k)
                for k, b in enumerate(blocks)]
        parts = pool.map(_call, args) if pool is not None else [_call(a) for a in args]
        out = {"anomaly": np.concatenate([p["anomaly"] for p in parts], axis=1),
               "surface_anomaly": np.concatenate([p["surface_anomaly"] for p in parts]),
               "energy_start": np.concatenate([p["energy_start"] for p in parts]),
               "energy_end": np.concatenate([p["energy_end"] for p in parts]),
               "max_ratio": np.concatenate([p["max_ratio"] for p in parts]),
               "iterations": np.max([p["iterations"] for p in parts], axis=0),
               "dir": d, "compared": all(p["compared"] for p in parts),
               "d_s": max(p["d_s"] for p in parts), "d_p": max(p["d_p"] for p in parts),
               "drift": (max(p["drift"] for p in parts)
                         if all(p["compared"] for p in parts) else None),
               "surface_range": (min(p["surface_range"][0] for p in parts),
                                 max(p["surface_range"][1] for p in parts))}
        out["sums"] = {k: np.concatenate([p["sums"][k] for p in parts]) for k in parts[0]["sums"]}
        out["global"] = {k: np.sum([p["global"][k] for p in parts], axis=0)
                         for k in parts[0]["global"]}
        ext = [p["extent"] for p in parts]
        out["extent"] = [(min(e[j][0] for e in ext), max(e[j][1] for e in ext))
                         for j in range(len(ext[0]))]
        np.savez(d / "integrals.npz", energy_start=out["energy_start"],
                 energy_end=out["energy_end"], max_ratio=out["max_ratio"], **out["sums"])
        kept = sorted((self.dir / "cycles").iterdir())
        for old in kept[:-3]:
            shutil.rmtree(old)
        return out

    def run(self, max_wall_s: Optional[float] = None) -> dict:
        cfg = self.cfg
        per = cfg.periodicity
        st = self.sub.initial_state(self.lat.size)
        state = (np.asarray(st.anomaly), np.asarray(st.surface_anomaly))
        record_every = bool(cfg.output.get("every_step_profiles"))
        history, previous = [], None
        stage = "spinup" if cfg.spinup_cycle_steps else "production"
        passes = 0
        cycle = 0
        ends = []                       # spin-up cycle-end states, for acceleration
        status = "TRANSIENT"
        t_start = time.time()
        ctx = mp.get_context("fork")
        pool = ctx.Pool(self.workers) if self.workers > 1 else None
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

    def _mkdir(self, name):
        p = self.dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _diagnostics(self, res, steps, stage, cycle):
        cfg, per = self.cfg, self.cfg.periodicity
        P = cfg.cycle_s
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
                "extent_K": res["extent"], "surface_T_range_K": list(res["surface_range"])}
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
