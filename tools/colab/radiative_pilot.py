"""Pilot timing and profiling of the heated radiative runs (GPU; Colab or local).

Run from a clean checkout of the pinned commit with ``PYTHONPATH=src``; this
file itself may live anywhere (it changes nothing in the package).

    python radiative_pilot.py time    --case earth --run-dir DIR --until-day 10
    python radiative_pilot.py profile --case earth --out DIR --mode sync|nvtx|cprofile
                                      [--resolution t21|t42] [--steps 36]

``time`` advances the *production* run directory with the production loop
(``RadiativeRun.run``), profiling off. It runs ``--warm-days`` first, then
synchronises the device, times the stretch to ``--until-day`` and
synchronises again. Only the daily sample and the checkpoint are timed
individually (both end in a host transfer, so they are synchronous anyway).
The run stops at a checkpoint and can be resumed by the same command with a
later ``--until-day`` or by ``python -m tropoi.run.radiative run``.

``profile`` uses a scratch run directory and never touches a production
run. ``sync`` brackets each instrumented call with a device synchronisation
and reports inclusive and exclusive time per label; ``nvtx`` pushes NVTX
ranges with no synchronisation (run it under ``nsys profile``);
``cprofile`` runs the Python profiler with no instrumentation. None of the
three is a throughput measurement.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pathlib
import pstats
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

CASES = {
    "earth": ("earth-gray-dry", "earth-gray-dry-heated-T21L25"),
    "jupiter": ("jupiter-schneider-liu", "jupiter-sl09-heated-T21L30"),
}
RESOLUTIONS = {
    "t21": {},
    "t42": {"l_max": 43, "retained_truncation": 42, "nlat": 64, "nlon": 128, "dt": 600.0},
    "tiny": {"l_max": 10, "retained_truncation": 9, "nlat": 16, "nlon": 32},
}


def config(case: str, resolution: str = "t21"):
    from tropoi.run.radiative.config import PRESETS
    preset, experiment_id = CASES[case]
    cfg = PRESETS[preset]().with_(frictional_heating=True, experiment_id=experiment_id)
    if resolution != "t21":
        cfg = cfg.with_(experiment_id=f"{experiment_id}-{resolution}", **RESOLUTIONS[resolution])
    return cfg


def sync() -> None:
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


def device() -> dict:
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    out = {"name": name.decode() if isinstance(name, bytes) else str(name),
           "total_memory_gb": props["totalGlobalMem"] / 2 ** 30, "cupy": cp.__version__}
    try:
        out["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception as exc:                                    # noqa: BLE001
        out["nvidia_smi"] = f"unavailable: {exc!r}"
    return out


class Utilisation:
    """Sample ``nvidia-smi`` GPU utilisation once a second in a thread."""

    def __init__(self):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                text = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=10)
                self.samples.append(float(text.split()[0]))
            except Exception:                                   # noqa: BLE001
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)

    def summary(self) -> dict:
        s = sorted(self.samples)
        if not s:
            return {"samples": 0}
        return {"samples": len(s), "mean_percent": sum(s) / len(s),
                "median_percent": s[len(s) // 2], "min_percent": s[0], "max_percent": s[-1]}


# ------------------------------------------------------------------ time
def cmd_time(args) -> dict:
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = config(args.case, args.resolution)
    run_dir = pathlib.Path(args.run_dir)
    run = RadiativeRun(cfg, run_dir, allow_dirty=args.allow_dirty)
    start_day = int(round(run.day))
    resumed_from = start_day if start_day > 0 else None
    spent = {"sample": [], "checkpoint": []}

    def timed(label, fn):
        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                spent[label].append(time.perf_counter() - t)
        return wrapper

    run._sample = timed("sample", run._sample)
    run._write_checkpoint = timed("checkpoint", run._write_checkpoint)

    warm_until = start_day + args.warm_days
    if args.until_day <= warm_until:
        raise SystemExit("--until-day must exceed the warm-up")
    sync()
    t = time.perf_counter()
    run.run(until_day=warm_until)
    sync()
    warm_seconds = time.perf_counter() - t
    for v in spent.values():
        v.clear()

    step0 = run.step
    with Utilisation() as util:
        sync()
        t0 = time.perf_counter()
        result = run.run(until_day=args.until_day)
        sync()
        wall = time.perf_counter() - t0
    steps = run.step - step0
    days = steps / cfg.steps_per_day
    t_sample = sum(spent["sample"]) / max(len(spent["sample"]), 1)
    t_ckpt = sum(spent["checkpoint"]) / max(len(spent["checkpoint"]), 1)
    stepping = wall - sum(spent["sample"]) - sum(spent["checkpoint"])
    per_step = stepping / steps
    full_seconds = (cfg.total_steps * per_step + cfg.days * t_sample
                    + (cfg.days / cfg.checkpoint_every_days) * t_ckpt)
    events = [json.loads(line) for line in
              (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    doc = {
        "kind": "pilot_timing", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case": args.case, "experiment_id": cfg.experiment_id, "config_sha256": cfg.sha256(),
        "code": run.code, "device": device(), "resumed_from_day": resumed_from,
        "resume_events": sum(e["event"] == "resumed" for e in events),
        "resolution": {"l_max": cfg.l_max, "retained": cfg.retained_truncation,
                       "nlat": cfg.nlat, "nlon": cfg.nlon, "nlev": run.model.nlev, "dt": cfg.dt},
        "target_days": cfg.days, "warm_days": args.warm_days, "warm_seconds": warm_seconds,
        "timed": {"from_day": warm_until, "to_day": result["day"], "steps": steps,
                  "wall_seconds": wall, "status": result["status"]},
        "seconds_per_step_all_inclusive": wall / steps,
        "seconds_per_step_stepping_only": per_step,
        "seconds_per_daily_sample": t_sample, "seconds_per_checkpoint": t_ckpt,
        "checkpoints_timed": len(spent["checkpoint"]),
        "simulated_days_per_wall_hour": days / wall * 3600.0,
        "estimated_full_run_hours": full_seconds / 3600.0,
        "estimated_remaining_hours": full_seconds / 3600.0 * (1.0 - result["day"] / cfg.days),
        "gpu_utilisation": util.summary(),
    }
    out = run_dir / f"pilot-timing-d{int(result['day']):05d}.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


# --------------------------------------------------------------- profile
class Recorder:
    """Inclusive/exclusive wall time per label, with optional device
    synchronisation at both ends of every instrumented call."""

    def __init__(self, mode: str):
        self.mode = mode
        self.stats: dict[str, list] = {}          # label -> [calls, inclusive, exclusive]
        self.stack: list[list] = []
        self.names: list[str] = []                # open non-transform labels
        self.enabled = False
        if mode == "nvtx":
            from cupy.cuda import nvtx
            self.nvtx = nvtx

    def wrap(self, label: str, fn, *, by_caller: bool = False):
        """Instrument ``fn``. With ``by_caller`` the label gets ``@<innermost
        open label>`` appended at call time (used for the transforms, so
        their calls and time are split by who asked for them)."""
        base = label

        def wrapper(*a, **k):
            if not self.enabled:
                return fn(*a, **k)
            if by_caller:
                label = f"{base}@{self.names[-1] if self.names else 'top'}"
            else:
                label = base
                self.names.append(base)
            try:
                return call(label, a, k)
            finally:
                if not by_caller:
                    self.names.pop()

        def call(label, a, k):
            if self.mode == "nvtx":
                self.nvtx.RangePush(label)
                try:
                    return fn(*a, **k)
                finally:
                    self.nvtx.RangePop()
            sync()
            frame = [time.perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return fn(*a, **k)
            finally:
                sync()
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                s = self.stats.setdefault(label, [0, 0.0, 0.0])
                s[0] += 1
                s[1] += elapsed
                s[2] += elapsed - frame[1]
        return wrapper

    def patch(self, obj, name: str, label: str, *, by_caller: bool = False) -> bool:
        fn = getattr(obj, name, None)
        if fn is None:
            return False
        try:
            setattr(obj, name, self.wrap(label, fn, by_caller=by_caller))
        except AttributeError:
            return False
        return True


class _Term:
    """A stepper term with its call instrumented (keeps ``signature``)."""

    def __init__(self, term, call):
        self._term = term
        self._call = call

    def __call__(self, *a, **k):
        return self._call(*a, **k)

    def __getattr__(self, name):
        return getattr(self._term, name)


def instrument(run, rec: Recorder) -> list[str]:
    model, stepper = run.model, run.stepper
    done = []
    spaces = {id(model.sh): model.sh}
    ps = getattr(model, "_ps", None)
    if ps is not None and getattr(ps, "sh", None) is not None:
        spaces[id(ps.sh)] = ps.sh
    for sh in spaces.values():
        for name, label in (("transform", "transform.analysis.scalar"),
                            ("transform_batch", "transform.analysis.batch"),
                            ("inv_transform", "transform.synthesis.scalar"),
                            ("inv_transform_batch", "transform.synthesis.batch")):
            if rec.patch(sh, name, label, by_caller=True):
                done.append(f"{type(sh).__name__}.{name}")
    if rec.patch(model.so, "vector_curl_div_spectral_batch", "transform.vector_curl_div",
                 by_caller=True):
        done.append("so.vector_curl_div_spectral_batch")
    rec.patch(stepper, "tendency", "dynamics.tendency")
    rec.patch(stepper.solver, "advance", "dynamics.implicit_solve")
    if run.coupling is not None:
        rec.patch(run.coupling.physics, "compute", "physics.radiation_and_boundary")
        rec.patch(run.coupling.physics, "adjust", "physics.convection")
    stepper.lagged_terms = tuple(
        _Term(t, rec.wrap("term." + t.signature()["kind"], t.__call__))
        for t in stepper.lagged_terms)
    stepper.explicit_terms = tuple(
        _Term(t, rec.wrap("term." + t.signature()["kind"], t.__call__))
        for t in stepper.explicit_terms)
    rec.patch(stepper, "step", "step")
    rec.patch(run, "_sample", "diagnostics.daily_sample")
    rec.patch(run, "_write_checkpoint", "output.checkpoint")
    return done


def transforms_by_caller(labels: dict) -> dict:
    """``{caller: {"synthesis.scalar": {calls_per_step, ms_per_step}, ...,
    "total_ms_per_step": ...}}`` from the ``transform.<kind>@<caller>`` labels."""
    out: dict = {}
    for label, row in labels.items():
        if not label.startswith("transform.") or "@" not in label:
            continue
        kind, caller = label[len("transform."):].split("@", 1)
        entry = out.setdefault(caller, {"total_ms_per_step": 0.0})
        entry[kind] = {"calls_per_step": row["calls_per_step"],
                       "ms_per_step": row["inclusive_ms_per_step"]}
        entry["total_ms_per_step"] += row["inclusive_ms_per_step"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["total_ms_per_step"]))


def table(doc: dict) -> str:
    """The sync profile as text: labels, then transforms split by caller."""
    lines = ["%-58s %10s %10s %10s" % ("label", "calls/step", "incl ms", "excl ms")]
    for label, r in doc["labels"].items():
        lines.append("%-58s %10.2f %10.2f %10.2f" % (label, r["calls_per_step"],
                                                     r["inclusive_ms_per_step"],
                                                     r["exclusive_ms_per_step"]))
    lines += ["", "transforms by caller (ms per step; calls per step in brackets)"]
    for caller, entry in doc["transforms_by_caller"].items():
        parts = ["%s %.1f [%.1f]" % (kind, v["ms_per_step"], v["calls_per_step"])
                 for kind, v in entry.items() if kind != "total_ms_per_step"]
        lines.append("  %-34s %7.1f   %s" % (caller, entry["total_ms_per_step"], "; ".join(parts)))
    lines.append("")
    lines.append("synchronised step: %.1f ms" % (1e3 * doc["synchronised_seconds_per_step"]))
    return "\n".join(lines)


def cmd_profile(args) -> dict:
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = config(args.case, args.resolution)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(args.scratch) / f"{cfg.experiment_id}-{args.mode}-{int(time.time())}"
    run = RadiativeRun(cfg, scratch, allow_dirty=True, log=lambda s: None)
    rec = Recorder(args.mode)
    patched = instrument(run, rec) if args.mode in ("sync", "nvtx") else []
    for _ in range(args.warm_steps):
        run.stepper.step()
    sync()

    def stretch():
        for _ in range(args.steps):
            run.stepper.step()
        run._sample(1)
        run._write_checkpoint()

    tag = f"{args.case}-{args.resolution}-{args.mode}"
    doc = {"kind": "profile", "mode": args.mode, "case": args.case,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "experiment_id": cfg.experiment_id, "config_sha256": cfg.sha256(), "code": run.code,
           "device": device(), "steps": args.steps, "warm_steps": args.warm_steps,
           "resolution": {"l_max": cfg.l_max, "retained": cfg.retained_truncation,
                          "nlat": cfg.nlat, "nlon": cfg.nlon, "nlev": run.model.nlev,
                          "dt": cfg.dt},
           "instrumented": patched}
    if args.mode == "cprofile":
        prof = cProfile.Profile()
        sync()
        t0 = time.perf_counter()
        prof.enable()
        stretch()
        sync()
        prof.disable()
        doc["wall_seconds"] = time.perf_counter() - t0
        for key in ("cumulative", "tottime"):
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(45)
            (out / f"profile-{tag}-{key}.txt").write_text(buf.getvalue(), encoding="utf-8")
        prof.dump_stats(str(out / f"profile-{tag}.pstats"))
    else:
        rec.enabled = True
        sync()
        t0 = time.perf_counter()
        stretch()
        sync()
        doc["wall_seconds"] = time.perf_counter() - t0
        rec.enabled = False
        if args.mode == "sync":
            step = rec.stats.get("step", [1, 1.0, 1.0])
            doc["labels"] = {
                label: {"calls": c, "calls_per_step": c / args.steps,
                        "inclusive_seconds": inc, "exclusive_seconds": exc,
                        "inclusive_ms_per_step": 1e3 * inc / args.steps,
                        "exclusive_ms_per_step": 1e3 * exc / args.steps,
                        "exclusive_fraction_of_step": exc / step[1]}
                for label, (c, inc, exc) in sorted(rec.stats.items(), key=lambda kv: -kv[1][2])}
            doc["synchronised_seconds_per_step"] = step[1] / step[0]
            doc["transforms_by_caller"] = transforms_by_caller(doc["labels"])
    (out / f"profile-{tag}.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    if args.mode == "sync":
        (out / f"profile-{tag}.txt").write_text(table(doc) + "\n", encoding="utf-8")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--until-day", type=int, default=10)
    t.add_argument("--warm-days", type=int, default=1)
    t.add_argument("--allow-dirty", action="store_true")
    p = sub.add_parser("profile")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("sync", "nvtx", "cprofile"), default="sync")
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--warm-steps", type=int, default=6)
    p.add_argument("--scratch", default="scratch-profile")
    for s in (t, p):
        s.add_argument("--case", choices=sorted(CASES), required=True)
        s.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="t21")
    args = ap.parse_args(argv)
    doc = cmd_time(args) if args.cmd == "time" else cmd_profile(args)
    print(json.dumps(doc, indent=1, sort_keys=True))
    if doc.get("mode") == "sync":
        print()
        print(table(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
