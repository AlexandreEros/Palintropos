"""Radiative–dynamical experiment driver: fixed-dt semi-implicit run with
column physics, hash-verified checkpoints, exact resume and an energy budget.

Run directory layout::

    config.json        frozen configuration (+ config_sha256), code identity,
                       the initial-state record
    events.jsonl       created / resumed / checkpoint / completed / aborted
    checkpoints/       stepper levels + surface temperature + accumulators
    states/            state-d<day>.npy (and surface-d<day>.npy)
    series.json        daily scalars: energy, budget, fluxes, extrema
    means.npz          time-mean zonal means over the averaging window

The daily energy record is what makes the coupling checkable. For each day

    residual = d(total energy)/dt - column_energy_tendency

where the first term is the change of :func:`tropoi.temporal.column_coupling.
global_energy` over the day and the second the time-mean flux-implied
tendency accumulated by the coupling. The residual is the energy the
numerics add or remove (hyperdiffusion, drag, time filter, spectral
truncation of the heating); it is reported, never corrected.
"""
from __future__ import annotations

import io
import json
import pathlib
import time
from typing import Any, Callable, Optional

import numpy as np

from tropoi.run.held_suarez.checkpoint import (CheckpointError, atomic_write_bytes,
                                               list_checkpoints, prune_checkpoints,
                                               read_checkpoint, write_checkpoint)
from tropoi.run.held_suarez.experiment import (RunError, _now, code_identity,
                                               daily_sample, environment_identity)
from tropoi.run.radiative.config import DAY_SECONDS, RadiativeConfig

__all__ = ["RadiativeRun", "RunError", "CheckpointError"]

_MEAN_FIELDS = ("u", "v", "T", "usvs", "vsTs")
_BUDGET_FIELDS = ("toa_net_down", "olr", "absorbed_stellar", "bottom_net_down", "sensible",
                  "interior", "column_energy_tendency", "surface_net")


class RadiativeRun:
    """One experiment directory. Construction creates or resumes; ``run``
    advances. ``physics=False`` runs the same numerics adiabatically (the
    control that measures the dynamical core's own energy error)."""

    def __init__(self, cfg: RadiativeConfig, run_dir, *, physics: bool = True,
                 allow_dirty: bool = False, log: Callable[[str], None] = print):
        import cupy as cp
        from tropoi.run.radiative.model import (build_model, build_physics, build_stepper,
                                                initial_state)
        from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
        from tropoi.temporal.column_coupling import ColumnPhysicsCoupling

        self.cfg = cfg
        self.run_dir = pathlib.Path(run_dir)
        self.log = log
        self.with_physics = bool(physics)
        self.code = code_identity()
        self.config_sha256 = cfg.sha256()
        if not allow_dirty and (self.code.get("git_commit") is None or self.code.get("git_dirty")):
            raise RunError("refusing to run from an unidentified or modified source tree "
                           "(pass allow_dirty for development runs)")
        rd = self.run_dir
        rd.mkdir(parents=True, exist_ok=True)
        cfg_file = rd / "config.json"
        if cfg_file.exists():
            stored = json.loads(cfg_file.read_text(encoding="utf-8"))
            if (stored.get("config_sha256") != self.config_sha256
                    or stored.get("physics") != self.with_physics):
                raise RunError(f"{rd} holds a different experiment "
                               f"({stored.get('config_sha256')}); refusing to mix")

        self.model = build_model(cfg)
        model = self.model

        def validator(y):
            model.validate_state(PrimitiveEquationsState(y), context="(RK4 startup stage)")

        found = list_checkpoints(rd / "checkpoints")
        x0, surface_t, info = (None, None, None)
        if not (cfg_file.exists() and found):
            x0, surface_t, info = initial_state(cfg)
        self.coupling = None
        if self.with_physics:
            physics_obj = build_physics(cfg)
            ts0 = None
            if physics_obj.surface is not None:
                npts = int(model.grid.n_points)
                ts0 = cp.full(npts, 280.0 if surface_t is None else float(surface_t))
            self.coupling = ColumnPhysicsCoupling(model, physics_obj, cfg.dt,
                                                  surface_temperature=ts0)
        self.stepper = build_stepper(cfg, model, self.coupling, stage_validator=validator)
        self.series: dict[str, list] = {}
        self._sums = None
        self._count = 0
        self._energy_prev: Optional[dict] = None
        if cfg_file.exists() and found:
            self._resume(found[-1][1])
        else:
            self.stepper.initialize(x0, 0.0)
            from tropoi.physics.species import dataset_identities
            doc = {"config": cfg.to_dict(), "config_sha256": self.config_sha256,
                   "physics": self.with_physics, "code": self.code, "created": _now(),
                   "initial_state": info,
                   # Which packaged physics datasets produced this run
                   # (descriptive provenance; never part of config_sha256).
                   "physics_datasets": dataset_identities()}
            atomic_write_bytes(cfg_file, json.dumps(doc, indent=2, sort_keys=True).encode())
            self._event("created", day=0, step=0, environment=environment_identity())
            self._sample(0)
            self._write_checkpoint()

    # -- bookkeeping ----------------------------------------------------
    def _event(self, event: str, **fields) -> None:
        with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "time": _now(), **fields}, sort_keys=True) + "\n")

    @property
    def step(self) -> int:
        return self.stepper.step_count

    @property
    def day(self) -> float:
        return self.step / self.cfg.steps_per_day

    def _mean_window_start(self) -> int:
        last = self.cfg.mean_last_days
        return self.cfg.days - last if last > 0 else self.cfg.days // 2

    def _sample(self, day: int) -> None:
        from tropoi.temporal.column_coupling import global_energy
        cfg = self.cfg
        s = daily_sample(self.model, self.stepper.state, self.stepper.x_prev)
        energy = global_energy(self.model, self.stepper.state, cfg.gravity)
        row: dict[str, Any] = {"day": day, "max_abs_u": s["max_abs_u"], "t_min": s["t_min"],
                               "t_max": s["t_max"], "mean_ps": s["mean_ps"],
                               "comp_mode": s["comp_mode"],
                               **{f"energy_{k}": v for k, v in energy.items()}}
        budget = None
        if self.coupling is not None:
            budget = self.coupling.budget_means(reset=True)
            for k in _BUDGET_FIELDS:
                row[k] = budget[k]
            ts = self.coupling.surface_temperature
            if ts is not None:
                row["mean_surface_temperature"] = float((self.coupling.weights * ts).sum())
        if self._energy_prev is not None:
            rate = (energy["total"] - self._energy_prev["total"]) / DAY_SECONDS
            row["energy_rate"] = rate
            row["energy_residual"] = rate - (budget["column_energy_tendency"] if budget else 0.0)
        self._energy_prev = energy
        for k, v in row.items():
            self.series.setdefault(k, []).append(v)
        if day > self._mean_window_start():
            if self._sums is None:
                self._sums = {k: np.zeros_like(s[k]) for k in _MEAN_FIELDS}
            for k in _MEAN_FIELDS:
                self._sums[k] += s[k]
            self._count += 1

    def _arrays(self) -> dict[str, np.ndarray]:
        import cupy as cp
        sd = self.stepper.state_dict()
        arrays = {"x_curr": cp.asnumpy(sd["x_curr"])}
        if sd["x_prev"] is not None:
            arrays["x_prev"] = cp.asnumpy(sd["x_prev"])
        if self.coupling is not None:
            ts = self.coupling.state_dict()["surface_temperature"]
            if ts is not None:
                arrays["surface_temperature"] = ts
        if self._sums is not None:
            for k, v in self._sums.items():
                arrays[f"sum_{k}"] = v
        return arrays

    def _write_checkpoint(self) -> pathlib.Path:
        sd = self.stepper.state_dict()
        sd.pop("x_curr")
        sd.pop("x_prev")
        coupling = None
        if self.coupling is not None:
            c = self.coupling.state_dict()
            coupling = {"budget_time": c["budget_time"], "budget": c["budget"]}
        meta = {"day": self.day, "stepper": sd, "coupling": coupling, "count": self._count,
                "series": self.series, "energy_prev": self._energy_prev,
                "config_sha256": self.config_sha256, "code": self.code,
                "environment": environment_identity()}
        path = write_checkpoint(self.run_dir / "checkpoints", self.step, self._arrays(), meta)
        prune_checkpoints(self.run_dir / "checkpoints", self.cfg.keep_checkpoints)
        atomic_write_bytes(self.run_dir / "series.json",
                           json.dumps(self.series, sort_keys=True).encode())
        self._write_means()
        self._event("checkpoint", day=self.day, step=self.step, file=path.name)
        return path

    def _write_means(self) -> None:
        grid, sigma = self.model.grid, self.model.sigma
        doc = {"lat": np.degrees(np.asarray(grid._latitudes_np)),
               "sigma": sigma.full_levels_array(), "sigma_interfaces": sigma.interfaces_array(),
               "gauss_weights": np.asarray(grid._gl_weights) / 2.0, "count": self._count,
               "reference_pressure": self.cfg.reference_pressure}
        if self._sums is not None and self._count:
            doc.update({k: v / self._count for k, v in self._sums.items()})
        buf = io.BytesIO()
        np.savez(buf, **doc)
        atomic_write_bytes(self.run_dir / "means.npz", buf.getvalue())

    def _resume(self, path: pathlib.Path) -> None:
        import cupy as cp
        arrays, meta = read_checkpoint(path)
        if meta.get("config_sha256") != self.config_sha256:
            raise RunError(f"{path}: checkpoint belongs to another configuration")
        if meta["code"].get("git_commit") != self.code.get("git_commit"):
            raise RunError(f"{path}: written by commit {meta['code'].get('git_commit')}, this "
                           f"code is {self.code.get('git_commit')}; start a new experiment")
        sd = dict(meta["stepper"])
        sd["x_curr"] = cp.asarray(arrays["x_curr"])
        sd["x_prev"] = cp.asarray(arrays["x_prev"]) if "x_prev" in arrays else None
        try:
            self.stepper.load_state_dict(sd)
        except ValueError as exc:
            raise RunError(f"{path}: incompatible stepper state: {exc}") from exc
        if self.coupling is not None:
            c = dict(meta["coupling"] or {})
            c["surface_temperature"] = arrays.get("surface_temperature")
            self.coupling.load_state_dict(c)
        self.series = {k: list(v) for k, v in meta["series"].items()}
        self._energy_prev = meta["energy_prev"]
        self._count = int(meta["count"])
        sums = {k[4:]: v for k, v in arrays.items() if k.startswith("sum_")}
        self._sums = sums or None
        self._event("resumed", day=meta["day"], step=sd["step"], checkpoint=path.name)
        self.log(f"resumed {self.cfg.experiment_id} at day {meta['day']} from {path.name}")

    # -- the loop -------------------------------------------------------
    def run(self, until_day: Optional[int] = None, *,
            max_wall_seconds: Optional[float] = None) -> dict[str, Any]:
        import cupy as cp
        from tropoi.spatial.states.primitive_equations import (PrimitiveEquationsState,
                                                               PrimitiveEquationsStateError)
        cfg = self.cfg
        target = cfg.days if until_day is None else min(int(until_day), cfg.days)
        spd = cfg.steps_per_day
        end_step = target * spd
        t_start = time.perf_counter()
        first = last_ckpt = self.step
        status = "reached"
        try:
            while self.step < end_step:
                self.stepper.step()
                if self.step % spd == 0:
                    day = self.step // spd
                    if not bool(cp.isfinite(self.stepper.state).all()):
                        raise PrimitiveEquationsStateError(f"non-finite state at day {day}")
                    self.model.validate_state(PrimitiveEquationsState(self.stepper.state),
                                              context=f"(day {day})")
                    self._sample(day)
                    if day % cfg.state_every_days == 0 or self.step == cfg.total_steps:
                        sdir = self.run_dir / "states"
                        sdir.mkdir(exist_ok=True)
                        for name, arr in (("state", self.stepper.state),
                                          ("surface", None if self.coupling is None
                                           else self.coupling.surface_temperature)):
                            if arr is not None:
                                b = io.BytesIO()
                                np.save(b, cp.asnumpy(arr))
                                atomic_write_bytes(sdir / f"{name}-d{day:05d}.npy", b.getvalue())
                    if day % cfg.checkpoint_every_days == 0 or self.step == end_step:
                        self._write_checkpoint()
                        last_ckpt = self.step
                        rate = (time.perf_counter() - t_start) / max(self.step - first, 1)
                        self.log(f"day {day}/{cfg.days}  {rate:.3f} s/step  "
                                 f"max|u| {self.series['max_abs_u'][-1]:.1f} m/s")
                    if max_wall_seconds is not None and self.step < end_step \
                            and time.perf_counter() - t_start > max_wall_seconds:
                        status = "wall_limit"
                        break
        except PrimitiveEquationsStateError as exc:
            self._event("aborted", day=self.day, step=self.step, error=repr(exc))
            raise
        if self.step != last_ckpt:
            self._write_checkpoint()
        wall = time.perf_counter() - t_start
        if self.step >= cfg.total_steps:
            self._event("completed", day=self.day, step=self.step)
            status = "completed"
        return {"status": status, "day": self.day, "step": self.step, "wall_seconds": wall,
                "seconds_per_step": wall / max(self.step - first, 1)}
