"""Profile one conduction step of the production Mercury configuration.

    python tools/planet_one/gpu_profile.py [--backend cupy] [--columns 16200] [--steps 20]

Times a full ``ConductiveSubstrate.step`` at the production grid (VPW99 TWO,
top cell 1 mm, stretch 1.0488, depth 1 m) with realistic forcing. It then
splits the time between residual/Jacobian assembly and the backend-generic
tridiagonal (Thomas) solve by wrapping both with device synchronization.
The split shows whether a specialized CUDA solver would pay off. Prints
JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import tropoi.temporal.substrate as substrate_mod
from tropoi.temporal.substrate import BackwardEulerSubstrate
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import _backend, build_substrate
from tropoi.physics.insolation import insolation_from_spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy")
    ap.add_argument("--columns", type=int, default=16200)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args(argv)
    xp = _backend(args.backend)
    sync = (lambda: xp.cuda.Device().synchronize()) if args.backend == "cupy" else (lambda: None)
    cfg = mercury_vpw99(model="TWO")
    sub = build_substrate(cfg)
    ins = insolation_from_spec(cfg.illumination)
    rng = np.random.default_rng(0)
    lat = xp.asarray(np.radians(rng.uniform(-89, 89, args.columns)))
    lon = xp.asarray(np.radians(rng.uniform(0, 360, args.columns)))
    n = (xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat))
    dt = cfg.cycle_s / cfg.cycle_steps
    st = sub.initial_state(args.columns, xp=xp)
    for k in range(5):                                   # warm-up (kernel compilation)
        st, _ = BackwardEulerSubstrate(sub).step(st, dt, shortwave_down=ins.incident(*n, (k + 1) * dt)[0])
    sync()
    timing = {"system": 0.0, "tridiagonal": 0.0, "calls_system": 0, "calls_tridiagonal": 0}
    orig_sys, orig_tri = substrate_mod.BackwardEulerSubstrate._system, substrate_mod._tridiagonal

    def timed_sys(self, *a, **k):
        sync(); t = time.perf_counter(); out = orig_sys(self, *a, **k); sync()
        timing["system"] += time.perf_counter() - t; timing["calls_system"] += 1
        return out

    def timed_tri(*a, **k):
        sync(); t = time.perf_counter(); out = orig_tri(*a, **k); sync()
        timing["tridiagonal"] += time.perf_counter() - t; timing["calls_tridiagonal"] += 1
        return out

    # untimed run first: total step time without the instrumentation's syncs
    t0 = time.perf_counter()
    iters = []
    for k in range(args.steps):
        st, rep = BackwardEulerSubstrate(sub).step(st, dt, shortwave_down=ins.incident(*n, (k + 6) * dt)[0])
        iters.append(rep.iterations)
    sync()
    total = (time.perf_counter() - t0) / args.steps
    substrate_mod.BackwardEulerSubstrate._system = timed_sys
    substrate_mod._tridiagonal = timed_tri
    t0 = time.perf_counter()
    for k in range(args.steps):
        st, rep = BackwardEulerSubstrate(sub).step(st, dt, shortwave_down=ins.incident(*n, (k + 6 + args.steps) * dt)[0])
    sync()
    instrumented = (time.perf_counter() - t0) / args.steps
    substrate_mod.BackwardEulerSubstrate._system, substrate_mod._tridiagonal = orig_sys, orig_tri
    out = {"backend": args.backend, "columns": args.columns, "layers": sub.layers,
           "seconds_per_step": total, "newton_iterations_mean": float(np.mean(iters)),
           "instrumented_seconds_per_step": instrumented,
           "system_seconds_per_step": timing["system"] / args.steps,
           "tridiagonal_seconds_per_step": timing["tridiagonal"] / args.steps,
           "system_calls_per_step": timing["calls_system"] / args.steps,
           "tridiagonal_calls_per_step": timing["calls_tridiagonal"] / args.steps,
           "production_cycle_minutes_estimate": total * cfg.cycle_steps / 60.0}
    if args.backend == "cupy":
        props = xp.cuda.runtime.getDeviceProperties(0)
        out["device"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        out["memory_used_bytes"] = int(xp.get_default_memory_pool().used_bytes())
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
