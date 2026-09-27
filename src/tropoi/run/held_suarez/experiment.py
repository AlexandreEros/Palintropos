"""Held–Suarez experiment driver: continuous fixed-dt SI run with atomic,
hash-verified checkpoints, exact resume, online statistics, backups and the
S8 benchmark gate.

Unlike :mod:`tropoi.run.pe.runner` (which re-initializes its stepper every
step), the driver initializes :class:`~tropoi.temporal.semi_implicit.
SemiImplicitLeapfrogStepper` ONCE and then only calls ``step()``: the RK4
startup happens exactly once per experiment (step 1), both leapfrog levels
travel through every checkpoint, and a resume continues the same recurrence.

Run directory layout::

    config.json            the frozen configuration (+ config_sha256)
    events.jsonl           append-only log: created / resumed (with the verified
                           checkpoint and hashes) / checkpoint / completed / aborted
    checkpoints/           checkpoint-s<step>.npz, newest ``keep_checkpoints`` kept
    states/                state-d<day>.npy full X^n every ``state_every_days``
    series.json            daily scalar series (rewritten atomically)
    statistics.npz         block means + series for the report (rewritten atomically)
    progress.json          day, step, wall-clock rate
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

from tropoi.run.held_suarez.checkpoint import (CheckpointError, array_sha256,
                                               atomic_write_bytes, backup_files,
                                               list_checkpoints, prune_checkpoints,
                                               read_checkpoint, write_checkpoint)
from tropoi.run.held_suarez.config import HeldSuarezConfig
from tropoi.run.held_suarez.statistics import OnlineStatistics

__all__ = ["RunError", "code_identity", "environment_identity", "protocol_sha256",
           "daily_sample", "HeldSuarezRun", "benchmark"]

REPO = pathlib.Path(__file__).resolve().parents[4]
PROTOCOL = REPO / "docs" / "held_suarez" / "PROTOCOL.md"


class RunError(RuntimeError):
    """The run cannot start, resume, or continue (details in the message)."""


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

def code_identity() -> dict[str, Any]:
    """git commit of the source tree this module runs from, and dirtiness."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        commit = git("rev-parse", "HEAD")
        # tracked or untracked changes to the solver or to the frozen protocol
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal", "--", "src",
                         "docs/held_suarez/PROTOCOL.md"))
    except Exception:
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def environment_identity() -> dict[str, Any]:
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "platform": platform.platform()}
    try:
        import cupy as cp
        env["cupy"] = cp.__version__
        env["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        env["cuda_driver"] = int(cp.cuda.runtime.driverGetVersion())
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        env["device"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception as exc:                           # pragma: no cover (CPU-only)
        env["cupy"] = None
        env["cupy_error"] = str(exc)
    return env


def protocol_sha256() -> Optional[str]:
    try:
        return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# daily diagnostics (GPU -> host)
# ---------------------------------------------------------------------------

def daily_sample(model, coeffs, x_prev=None) -> dict[str, Any]:
    """Zonal means, eddy products, KE spectrum and global scalars of one
    state on the model's state grid (see :mod:`.statistics`)."""
    import cupy as cp
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
    state = PrimitiveEquationsState(coeffs)
    grid = model.grid
    nlat, nlon = grid.nlat, grid.nlon
    K = model.nlev
    u, v = model.wind_on_state_grid(state)
    T = model.temperature_on_state_grid(state)
    lnps = model.sh.inv_transform(state.ln_ps).real
    u = cp.asnumpy(u).reshape(K, nlat, nlon)
    v = cp.asnumpy(v).reshape(K, nlat, nlon)
    T = cp.asnumpy(T).reshape(K, nlat, nlon)
    lnps = cp.asnumpy(lnps).reshape(nlat, nlon)
    ps = np.exp(lnps)
    w = np.asarray(grid._gl_weights, dtype=np.float64) / 2.0      # sums to 1
    dsig = model.sigma.thickness_array()

    def area(f2d):                                              # (nlat, nlon)
        return float((f2d.mean(axis=1) * w).sum())

    out: dict[str, Any] = {}
    zm = {"u": u.mean(axis=2), "v": v.mean(axis=2), "T": T.mean(axis=2)}
    us = u - zm["u"][:, :, None]
    vs = v - zm["v"][:, :, None]
    Ts = T - zm["T"][:, :, None]
    zm["TsTs"] = (Ts * Ts).mean(axis=2)
    zm["usus"] = (us * us).mean(axis=2)
    zm["vsTs"] = (vs * Ts).mean(axis=2)
    zm["usvs"] = (us * vs).mean(axis=2)
    for k, a in zm.items():
        out[k] = np.ascontiguousarray(a.T)                        # (nlat, K)
    out["mean_ps"] = area(ps)
    out["mean_lnps"] = area(lnps)
    out["mean_T"] = area((T * dsig[:, None, None]).sum(axis=0))
    out["ke"] = area(0.5 * ((u * u + v * v) * dsig[:, None, None]).sum(axis=0))
    out["max_abs_u"] = float(np.abs(u).max())
    out["t_min"] = float(T.min())
    out["t_max"] = float(T.max())
    # KE per degree: mean of |V|^2/2 = sum a^2 / (2 l(l+1)) (|zeta|^2 + |delta|^2) / (4 pi)
    zd = cp.asnumpy(coeffs[0:2 * K]).reshape(2, K, model.l_max + 1, model.l_max + 1)
    p = np.abs(zd) ** 2
    per_lm = p[..., 0].sum(axis=0) + 2.0 * p[..., 1:].sum(axis=(0, 3))  # (K, l)
    l = np.arange(model.l_max + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = np.where(l > 0, model.R ** 2 / (2.0 * l * (l + 1.0)), 0.0) / (4.0 * math.pi)
    out["ke_spectrum"] = (dsig[:, None] * per_lm * fac[None, :]).sum(axis=0)
    # |X^n - X^{n-1}_f| / |X^n| over zeta and delta (the T/ln p_s means would swamp it)
    if x_prev is None:
        out["comp_mode"] = 0.0
    else:
        dyn = slice(0, 2 * K)
        den = float(cp.linalg.norm(coeffs[dyn]))
        out["comp_mode"] = (float(cp.linalg.norm(coeffs[dyn] - x_prev[dyn])) / den
                            if den > 0 else 0.0)
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HeldSuarezRun:
    """One experiment directory. ``open`` creates or resumes; ``run`` advances."""

    def __init__(self, cfg: HeldSuarezConfig, run_dir: pathlib.Path, *,
                 code: Optional[dict] = None, backup_dir: Optional[pathlib.Path] = None,
                 allow_dirty: bool = False, log: Callable[[str], None] = print,
                 sample_hook: Optional[Callable[[int, dict], None]] = None):
        self.cfg = cfg
        self.run_dir = pathlib.Path(run_dir)
        self.code = code if code is not None else code_identity()
        self.backup_dir = None if backup_dir is None else pathlib.Path(backup_dir)
        self.log = log
        self.sample_hook = sample_hook
        self.config_sha256 = cfg.sha256()
        if not allow_dirty:
            if self.code.get("git_commit") is None:
                raise RunError("cannot identify the source commit (git unavailable); refusing "
                               "a run whose provenance cannot be recorded")
            if self.code.get("git_dirty"):
                raise RunError("refusing to run from a source tree with uncommitted changes "
                               "under src/ or to PROTOCOL.md (pass allow_dirty for development runs)")
        self._check_backup_dir()
        self._open()

    # -- setup ----------------------------------------------------------
    def _open(self) -> None:
        import cupy as cp
        from tropoi.run.held_suarez.model import build_model, build_stepper, initial_state
        from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState

        rd = self.run_dir
        rd.mkdir(parents=True, exist_ok=True)
        cfg_file = rd / "config.json"
        if cfg_file.exists():
            stored = json.loads(cfg_file.read_text(encoding="utf-8"))
            if stored.get("config_sha256") != self.config_sha256:
                raise RunError(
                    f"{rd} holds experiment {stored.get('config', {}).get('experiment_id')!r} "
                    f"with config_sha256 {stored.get('config_sha256')}; this configuration "
                    f"hashes to {self.config_sha256}. Refusing to mix configurations.")
        self.model = build_model(self.cfg)
        model = self.model

        def validator(y):
            model.validate_state(PrimitiveEquationsState(y), context="(RK4 startup stage)")

        self.stepper = build_stepper(self.cfg, model, stage_validator=validator)
        self.stats = OnlineStatistics.for_config(self.cfg)
        found = list_checkpoints(rd / "checkpoints")
        if found and not cfg_file.exists():
            raise RunError(f"{rd}: checkpoints exist but config.json is missing; refusing to "
                           "start a new experiment over them")
        if not cfg_file.exists() or not found:
            if cfg_file.exists() and not found:
                ev = [e for e in self.events() if e["event"] == "checkpoint"]
                if ev:
                    raise RunError(f"{rd}: events.jsonl lists checkpoints but none is present")
            x0, rng_state = initial_state(self.cfg)
            self.rng_state = rng_state
            self.stepper.initialize(x0, 0.0)
            self._sample(0)
            doc = {"config": self.cfg.to_dict(), "config_sha256": self.config_sha256,
                   "protocol_sha256": protocol_sha256(), "code": self.code,
                   "created": _now()}
            atomic_write_bytes(cfg_file, json.dumps(doc, indent=2, sort_keys=True).encode())
            self._event("created", day=0, step=0, x0_sha256=array_sha256(cp.asnumpy(x0)),
                        environment=environment_identity())
            self._write_checkpoint()
        else:
            self._resume(found[-1][1])

    def _resume(self, path: pathlib.Path) -> None:
        import cupy as cp
        arrays, meta = read_checkpoint(path)          # raises on any hash mismatch
        if meta.get("config_sha256") != self.config_sha256:
            raise RunError(f"{path}: checkpoint config_sha256 {meta.get('config_sha256')} "
                           f"!= {self.config_sha256}")
        if meta["code"].get("git_commit") != self.code.get("git_commit"):
            raise RunError(
                f"{path}: checkpoint was written by commit {meta['code'].get('git_commit')}, "
                f"this code is {self.code.get('git_commit')}; a production experiment may not "
                "change code mid-run (plan §2 B5). Start a new experiment instead.")
        if meta.get("protocol_sha256") != protocol_sha256():
            raise RunError(
                f"{path}: PROTOCOL.md changed since this experiment started "
                f"({meta.get('protocol_sha256')} -> {protocol_sha256()}); the protocol is frozen "
                "for a running experiment (plan §5)")
        env_now = environment_identity()
        if meta.get("environment") != env_now:
            self.log(f"WARNING: resuming on a different software/hardware environment "
                     f"({meta.get('environment')} -> {env_now}); the continuation is valid "
                     "but not bit-identical to an uninterrupted run on the original one")
        sd = dict(meta["stepper"])
        sd["x_curr"] = cp.asarray(arrays["x_curr"])
        sd["x_prev"] = cp.asarray(arrays["x_prev"]) if "x_prev" in arrays else None
        try:
            self.stepper.load_state_dict(sd)          # scheme, dt, operator, hooks
        except ValueError as exc:
            raise RunError(f"{path}: incompatible stepper state: {exc}") from exc
        self.stats = OnlineStatistics.from_arrays(meta["statistics"], arrays)
        self.rng_state = meta["rng_state"]
        self._event("resumed", day=meta["day"], step=sd["step"], checkpoint=path.name,
                    verified_arrays={k: v["sha256"] for k, v in meta["arrays"].items()},
                    verified_meta_sha256=meta["meta_sha256"],
                    config_sha256=meta["config_sha256"], git_commit=meta["code"]["git_commit"],
                    git_dirty=self.code.get("git_dirty"), environment=env_now)
        self.log(f"resumed {self.cfg.experiment_id} at day {meta['day']} step {sd['step']} "
                 f"from {path.name} (all {len(arrays)} array hashes verified)")

    # -- bookkeeping ----------------------------------------------------
    def events(self) -> list[dict]:
        f = self.run_dir / "events.jsonl"
        if not f.exists():
            return []
        return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]

    def _event(self, event: str, **fields) -> None:
        rec = {"event": event, "time": _now(), **fields}
        with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @property
    def step(self) -> int:
        return self.stepper.step_count

    @property
    def day(self) -> float:
        return self.step / self.cfg.steps_per_day

    def _sample(self, day: int) -> None:
        s = daily_sample(self.model, self.stepper.state, self.stepper.x_prev)
        self.stats.add(day, s)
        if self.sample_hook is not None:
            self.sample_hook(day, s)

    def _write_checkpoint(self) -> pathlib.Path:
        import cupy as cp
        sd = self.stepper.state_dict()
        arrays = {"x_curr": cp.asnumpy(sd.pop("x_curr"))}
        xp = sd.pop("x_prev")
        if xp is not None:
            arrays["x_prev"] = cp.asnumpy(xp)
        arrays.update(self.stats.to_arrays())
        meta = {"day": self.day, "stepper": sd, "rng_state": self.rng_state,
                "statistics": self.stats.meta(), "config": self.cfg.to_dict(),
                "config_sha256": self.config_sha256, "protocol_sha256": protocol_sha256(),
                "code": self.code, "environment": environment_identity()}
        path = write_checkpoint(self.run_dir / "checkpoints", self.step, arrays, meta)
        prune_checkpoints(self.run_dir / "checkpoints", self.cfg.keep_checkpoints)
        self._export()
        self._event("checkpoint", day=self.day, step=self.step, file=path.name,
                    x_curr_sha256=array_sha256(arrays["x_curr"]))
        return path

    def _export(self) -> None:
        ser = self.stats.series_arrays()
        atomic_write_bytes(self.run_dir / "series.json", json.dumps(
            {k: v.tolist() for k, v in ser.items()}, sort_keys=True).encode())
        import io
        buf = io.BytesIO()
        bm = self.stats.block_means()
        np.savez(buf, counts=self.stats.counts, **{f"block_{k}": v for k, v in bm.items()},
                 **{f"series_{k}": v for k, v in ser.items()},
                 lat=np.degrees(np.asarray(self.model.grid._latitudes_np)),
                 sigma=self.model.sigma.full_levels_array(),
                 dsigma=self.model.sigma.thickness_array(),
                 gauss_weights=np.asarray(self.model.grid._gl_weights) / 2.0)
        atomic_write_bytes(self.run_dir / "statistics.npz", buf.getvalue())

    def _check_backup_dir(self) -> None:
        if self.backup_dir is None:
            return
        other = self.backup_dir / "config.json"
        if other.exists():
            sha = json.loads(other.read_text(encoding="utf-8")).get("config_sha256")
            if sha != self.config_sha256:
                raise RunError(f"backup directory {self.backup_dir} belongs to another "
                               f"configuration ({sha}); refusing to overwrite it")

    def _backup(self, path: pathlib.Path) -> None:
        """Copy the checkpoint and run files to the backup directory. A backup
        failure (network drive, quota) is logged and does not stop the run."""
        if self.backup_dir is None:
            return
        self._check_backup_dir()
        rd = self.run_dir
        try:
            dst = backup_files(path, [rd / "config.json", rd / "events.jsonl", rd / "series.json",
                                      rd / "statistics.npz", rd / "progress.json"],
                               self.backup_dir, self.cfg.keep_checkpoints)
        except (OSError, CheckpointError) as exc:
            self._event("backup_failed", day=self.day, step=self.step, error=repr(exc))
            self.log(f"WARNING: backup of {path.name} failed ({exc!r}); the run continues")
            return
        self.log(f"backed up {path.name} -> {dst}")

    # -- the loop -------------------------------------------------------
    def run(self, until_day: Optional[int] = None, *, max_wall_seconds: Optional[float] = None,
            backup_every_days: int = 10, max_steps: Optional[int] = None) -> dict[str, Any]:
        """Advance to ``until_day`` (default: the configured length). Stops early,
        after a checkpoint, when ``max_wall_seconds`` would be exceeded or after
        ``max_steps`` steps of this segment (possibly mid-day).

        Event log: a NaN or a failed state validation is recorded as
        ``aborted`` (tier B1 FAIL); any other exception (I/O while writing a
        checkpoint, a crash) as ``interrupted`` — the run resumes from its
        last checkpoint and B1 is unaffected."""
        import cupy as cp
        cfg = self.cfg
        target = cfg.days if until_day is None else min(int(until_day), cfg.days)
        spd = cfg.steps_per_day
        end_step = target * spd
        t_start = time.perf_counter()
        last_ckpt_step = self.step
        first_step = self.step
        status = "reached"
        from tropoi.spatial.states.primitive_equations import (PrimitiveEquationsState,
                                                               PrimitiveEquationsStateError)
        try:
            while self.step < end_step:
                t_step = time.perf_counter()
                self.stepper.step()
                if self.step % spd == 0:
                    day = self.step // spd
                    if not bool(cp.isfinite(self.stepper.state).all()):
                        raise PrimitiveEquationsStateError(f"non-finite state at day {day}")
                    self.model.validate_state(PrimitiveEquationsState(self.stepper.state),
                                              context=f"(day {day})")
                    self._sample(day)
                    if day % cfg.state_every_days == 0:
                        sdir = self.run_dir / "states"
                        sdir.mkdir(exist_ok=True)
                        buf = cp.asnumpy(self.stepper.state)
                        import io
                        b = io.BytesIO()
                        np.save(b, buf)
                        atomic_write_bytes(sdir / f"state-d{day:05d}.npy", b.getvalue())
                    if day % cfg.checkpoint_every_days == 0 or self.step == end_step:
                        path = self._write_checkpoint()
                        last_ckpt_step = self.step
                        self._progress(t_start)
                        if day % backup_every_days == 0 or day == cfg.days:
                            self._backup(path)
                if max_wall_seconds is not None and self.step < end_step:
                    elapsed = time.perf_counter() - t_start
                    per = time.perf_counter() - t_step
                    if elapsed + 3.0 * per + 30.0 > max_wall_seconds:
                        status = "wall_limit"
                        break
                if max_steps is not None and self.step - first_step >= max_steps \
                        and self.step < end_step:
                    status = "step_limit"
                    break
        except PrimitiveEquationsStateError as exc:
            self._event("aborted", day=self.day, step=self.step, error=repr(exc))
            raise
        except Exception as exc:
            self._event("interrupted", day=self.day, step=self.step, error=repr(exc))
            raise
        if self.step != last_ckpt_step:
            path = self._write_checkpoint()
            self._backup(path)
        self._progress(t_start)
        if self.step >= cfg.total_steps:
            self._event("completed", day=self.day, step=self.step)
            status = "completed"
        return {"status": status, "day": self.day, "step": self.step,
                "wall_seconds": time.perf_counter() - t_start}

    def _progress(self, t_start: float) -> None:
        doc = {"experiment_id": self.cfg.experiment_id, "day": self.day, "step": self.step,
               "days_total": self.cfg.days, "segment_wall_seconds": time.perf_counter() - t_start,
               "time": _now()}
        atomic_write_bytes(self.run_dir / "progress.json",
                           json.dumps(doc, sort_keys=True).encode())


# ---------------------------------------------------------------------------
# S8 benchmark gate
# ---------------------------------------------------------------------------

def benchmark(cfg: HeldSuarezConfig, work_dir: pathlib.Path, *, steps: int = 200,
              max_hours: Optional[float] = None, deadline: Optional[str] = None,
              code: Optional[dict] = None, allow_dirty: bool = False,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Time ``steps`` complete leapfrog steps of ``cfg`` (after the one-off RK4
    startup), including the normal daily statistics work that falls inside
    them, one forced statistics sample and one atomic checkpoint write; then
    project the wall time of the whole experiment:

        T = t_startup + total_steps * t_step + days * t_sample + n_ckpt * t_ckpt

    ``verdict`` is ``"within_budget"`` when T fits in ``max_hours`` and (if
    given) ends before the ISO-8601 ``deadline``; otherwise ``"over_budget"``.
    The benchmark runs in ``work_dir/benchmark-<id>`` (a scratch experiment),
    never in the production run directory.
    """
    import cupy as cp
    bdir = pathlib.Path(work_dir) / f"benchmark-{cfg.experiment_id}"
    if bdir.exists():
        import shutil
        shutil.rmtree(bdir)
    run = HeldSuarezRun(cfg, bdir, code=code, allow_dirty=allow_dirty, log=log)
    t0 = time.perf_counter()
    run.stepper.step()                                   # RK4 startup, once
    cp.cuda.Stream.null.synchronize()
    t_startup = time.perf_counter() - t0
    spd = cfg.steps_per_day
    sample_times, step_times = [], []
    t_loop = time.perf_counter()
    for _ in range(steps):
        a = time.perf_counter()
        run.stepper.step()
        cp.cuda.Stream.null.synchronize()
        step_times.append(time.perf_counter() - a)
        if run.step % spd == 0:
            a = time.perf_counter()
            run._sample(run.step // spd)
            sample_times.append(time.perf_counter() - a)
    if not sample_times:                                 # force one statistics sample
        a = time.perf_counter()
        daily_sample(run.model, run.stepper.state, run.stepper.x_prev)
        sample_times.append(time.perf_counter() - a)
    a = time.perf_counter()
    run._write_checkpoint()
    t_ckpt = time.perf_counter() - a
    t_total = time.perf_counter() - t_loop
    t_step = float(np.mean(step_times))
    t_sample = float(np.mean(sample_times))
    n_ckpt = math.ceil(cfg.days / cfg.checkpoint_every_days)
    projected = t_startup + cfg.total_steps * t_step + cfg.days * t_sample + n_ckpt * t_ckpt
    finite = bool(cp.isfinite(run.stepper.state).all())
    result = {"experiment_id": cfg.experiment_id, "config_sha256": cfg.sha256(),
              "steps_timed": steps, "t_startup_s": t_startup, "t_step_mean_s": t_step,
              "t_step_median_s": float(np.median(step_times)),
              "t_sample_s": t_sample, "n_samples": len(sample_times), "t_checkpoint_s": t_ckpt,
              "t_window_s": t_total, "total_steps": cfg.total_steps, "days": cfg.days,
              "projected_hours": projected / 3600.0, "finite": finite,
              "environment": environment_identity(), "max_hours": max_hours,
              "deadline": deadline}
    over = []
    if max_hours is not None and projected / 3600.0 > max_hours:
        over.append(f"projected {projected / 3600.0:.2f} h > allowed {max_hours:.2f} h")
    if deadline is not None:
        end = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=projected)
        dl = _dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if dl.tzinfo is None:
            dl = dl.replace(tzinfo=_dt.timezone.utc)
        result["projected_end"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end > dl:
            over.append(f"projected end {result['projected_end']} is after the deadline {deadline}")
    if not finite:
        over.append("state not finite after the benchmark window")
    result["verdict"] = "over_budget" if over else "within_budget"
    result["reasons"] = over
    atomic_write_bytes(pathlib.Path(work_dir) / f"benchmark-{cfg.experiment_id}.json",
                       json.dumps(result, indent=2, sort_keys=True).encode())
    return result
