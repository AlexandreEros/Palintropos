"""Compare the device-resident fused Newton step (``fused_newton_prototype``)
with the production ``ConductiveSubstrate.step``: bitwise state after every
step, Newton iteration counts, and warm per-step time.

    PYTHONPATH=src:. python tools/colab/fused_check.py --out fused.json
        [--state-file STATE.npz] [--steps 40] [--repeats 5]

Cases: the production Mercury VPW99 TWO grid (from the benchmark's saved
state if given, otherwise from the isothermal start), a terminator batch,
and the constant-property (THAI-like) column.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    import cupy as cp
    import fused_newton_prototype as fp
    import substrate_benchmark as sb
    from tropoi.physics.substrate import SubstrateState
    from tropoi.run.airless.experiment import build_substrate
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-file")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--columns", type=int, default=16200, help="terminator/constant cases")
    args = ap.parse_args(argv)
    sync = lambda: cp.cuda.Device().synchronize()            # noqa: E731
    cases = {}
    cfg = sb.production_config()
    sub = build_substrate(cfg)
    if args.state_file and pathlib.Path(args.state_file).exists():
        z = np.load(args.state_file)
        start = (cp.asarray(z["anomaly"]), cp.asarray(z["surface_anomaly"]), int(z["next_step"]))
    else:
        st = sub.initial_state(16200, xp=cp)
        start = (st.anomaly, st.surface_anomaly, 0)
    cases["production_vpw99_two_90x180"] = (cfg, sub, sb._forcing(cfg, None, cp), start)
    for material in ("vpw99_two", "constant"):
        c2, s2, cols = sb._scaling_case(material, 1.1, args.columns, "terminator")
        f2 = sb._forcing(c2, cols, cp)
        st = s2.initial_state(args.columns, xp=cp)
        for k in range(24):
            st, _ = s2.step(st, c2.dt, shortwave_down=f2((k + 1) * c2.dt))
        cases[f"terminator_{material}_{args.columns}"] = (c2, s2, f2, (st.anomaly, st.surface_anomaly, 24))
    out = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "commit": sb.git("rev-parse", "HEAD"), "steps": args.steps, "cases": {}}
    for name, (c, s, forcing, (a0, s0, first)) in cases.items():
        # RawKernel compiles lazily; force it here (the launcher sets an empty
        # CUPY_CACHE_DIR, so this is a cold NVRTC compile)
        t0 = time.perf_counter()
        fused = fp.FusedStep(s)
        fused.kernel.compile()
        sync()
        compile_s = time.perf_counter() - t0
        # bitwise trajectory comparison
        ref = SubstrateState(a0.copy(), s0.copy())
        got = SubstrateState(a0.copy(), s0.copy())
        mismatched_steps, max_abs, it_ref, it_got, failures = 0, 0.0, [], [], 0
        for k in range(args.steps):
            sw = forcing((first + k + 1) * c.dt)
            ref, rep = s.step(ref, c.dt, shortwave_down=sw)
            got, iters, fail, _ = fused.step(got, c.dt, sw)
            same = bool(cp.array_equal(ref.anomaly, got.anomaly)
                        & cp.array_equal(ref.surface_anomaly, got.surface_anomaly))
            mismatched_steps += 0 if same else 1
            max_abs = max(max_abs, float(cp.max(cp.abs(ref.anomaly - got.anomaly))),
                          float(cp.max(cp.abs(ref.surface_anomaly - got.surface_anomaly))))
            it_ref.append(rep.iterations)
            it_got.append(int(iters.max()))
            failures += int(fail.sum())
            if not same:                                   # continue from the reference
                got = SubstrateState(ref.anomaly.copy(), ref.surface_anomaly.copy())

        def timed(fn):
            ts = []
            for rep_ in range(args.repeats + 1):
                st = SubstrateState(a0.copy(), s0.copy())
                sync(); t = time.perf_counter()
                for k in range(args.steps):
                    st = fn(st, forcing((first + k + 1) * c.dt))
                sync()
                if rep_:
                    ts.append((time.perf_counter() - t) / args.steps)
            return {"median": statistics.median(ts), "min": min(ts), "max": max(ts), "n": len(ts)}
        t_ref = timed(lambda st, sw: s.step(st, c.dt, shortwave_down=sw)[0])
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        t_fused = timed(lambda st, sw: fused.step(st, c.dt, sw)[0])
        fused_pool = int(pool.total_bytes())
        row = {"columns": int(a0.shape[1]), "layers": s.layers, "strata": len(s.strata),
               "compile_s": compile_s,
               "memory_pool_total_bytes_after_fused_steps": fused_pool,
               "steps_bitwise_identical": args.steps - mismatched_steps,
               "steps_compared": args.steps, "max_abs_state_difference_K": max_abs,
               "newton_iterations_production": it_ref, "newton_iterations_fused_max": it_got,
               "iteration_counts_identical": it_ref == it_got,
               "fused_failures": failures,
               "production_step_seconds": t_ref, "fused_step_seconds": t_fused,
               "speedup": t_ref["median"] / t_fused["median"],
               "fused_column_steps_per_second": a0.shape[1] / t_fused["median"]}
        out["cases"][name] = row
        print(name, json.dumps({k: v for k, v in row.items()
                                if not k.startswith("newton_iterations")}), flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
