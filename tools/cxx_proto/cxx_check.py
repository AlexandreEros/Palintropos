"""Compare the portable C++ column solver (``cxx_substrate``) with the
production Python implementation and with the tracer-generated kernel.

    PYTHONPATH=src:. python tools/cxx_proto/cxx_check.py --out cxx.json
        [--state-file STATE.npz] [--steps 200] [--host-steps 20] [--columns 2048]

Per case, each implementation advances the same trajectory step by step, and
after every step its state is compared bit for bit with its reference:

| implementation | reference |
| --- | --- |
| C++ on CUDA | CuPy production step |
| tracer kernel | CuPy production step |
| C++ on the host | NumPy production step |
| C++ host vs C++ CUDA | (reported as is) |

On a mismatch the trajectory continues from the reference state, so one
difference is not compounded. Newton iteration counts are compared, and warm
per-step times and cold compile times are reported.
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
sys.path[:0] = [str(HERE), str(HERE.parent / "colab")]


def main(argv=None) -> int:
    import cupy as cp
    import cxx_substrate as cx
    import fused_newton_prototype as fp
    import substrate_benchmark as sb
    from tropoi.physics.substrate import SubstrateState
    from tropoi.run.airless.experiment import build_substrate
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-file")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--host-steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--columns", type=int, default=2048)
    ap.add_argument("--host-cache")
    args = ap.parse_args(argv)
    sync = lambda: cp.cuda.Device().synchronize()                    # noqa: E731

    cases = {}
    cfg = sb.production_config()
    sub = build_substrate(cfg)
    if args.state_file and pathlib.Path(args.state_file).exists():
        z = np.load(args.state_file)
        start = (z["anomaly"], z["surface_anomaly"], int(z["next_step"]), "saved state")
    else:
        st = sub.initial_state(16200)
        start = (np.asarray(st.anomaly), np.asarray(st.surface_anomaly), 0, "isothermal start")
    cases["production_vpw99_two_90x180"] = (cfg, sub, None, start)
    for material in ("vpw99_two", "constant"):
        c2, s2, cols = sb._scaling_case(material, 1.1, args.columns, "terminator")
        f2 = sb._forcing(c2, cols, cp)
        st = s2.initial_state(args.columns, xp=cp)
        for k in range(24):
            st, _ = s2.step(st, c2.dt, shortwave_down=f2((k + 1) * c2.dt))
        cases[f"terminator_{material}_{args.columns}"] = (
            c2, s2, cols, (st.anomaly.get(), st.surface_anomaly.get(), 24, "24 steps from isothermal"))

    out = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "commit": sb.git("rev-parse", "HEAD"), "cases": {}}
    for name, (c, s, cols, (a0, s0, first, origin)) in cases.items():
        fdev = sb._forcing(c, cols, cp)
        row = {"columns": int(a0.shape[1]), "layers": s.layers, "strata": len(s.strata),
               "start": origin}
        t = time.perf_counter(); cud = cx.CudaStep(s); cud.compile(); sync()
        row["cxx_cuda_compile_s"] = time.perf_counter() - t
        row["cxx_cuda_specialization"] = cud.name
        t = time.perf_counter(); fused = fp.FusedStep(s); fused.kernel.compile(); sync()
        row["tracer_compile_s"] = time.perf_counter() - t
        t = time.perf_counter(); host = cx.HostStep(s, cache_dir=args.host_cache)
        row["cxx_host_build_s"] = time.perf_counter() - t
        row["cxx_host_compiled_now"] = host.compiled
        row["cxx_host_compiler"] = f"{host.compiler} ({host.compiler_identity})"

        def trajectory(step_ref, step_new, to_host, n_steps, xp):
            ref = SubstrateState(xp.asarray(a0).copy(), xp.asarray(s0).copy())
            new = SubstrateState(xp.asarray(a0).copy(), xp.asarray(s0).copy())
            mism, worst, it_same = 0, 0.0, True
            for k in range(n_steps):
                sw = fdev((first + k + 1) * c.dt)
                sw = sw if xp is cp else sw.get()
                ref, rit = step_ref(ref, sw)
                new, nit = step_new(new, sw)
                ra, rs_ = to_host(ref.anomaly), to_host(ref.surface_anomaly)
                na, ns_ = to_host(new.anomaly), to_host(new.surface_anomaly)
                same = np.array_equal(ra, na) and np.array_equal(rs_, ns_)
                if not same:
                    mism += 1
                    worst = max(worst, float(np.max(np.abs(ra - na))),
                                float(np.max(np.abs(rs_ - ns_))))
                    new = SubstrateState(ref.anomaly.copy(), ref.surface_anomaly.copy())
                it_same &= (rit == nit)
            return {"steps": n_steps, "bitwise_identical_steps": n_steps - mism,
                    "max_abs_difference_K": worst, "newton_counts_identical": bool(it_same)}

        def prod(st, sw):
            st, rep = s.step(st, c.dt, shortwave_down=sw)
            return st, rep.iterations

        def wrap(impl):
            def f(st, sw):
                st, iters, fail, _ = impl.step(st, c.dt, sw)
                if int(fail.max()):
                    raise RuntimeError("Newton failure")
                return st, int(iters.max())
            return f
        get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)     # noqa: E731
        row["cxx_cuda_vs_cupy"] = trajectory(prod, wrap(cud), get, args.steps, cp)
        row["tracer_vs_cupy"] = trajectory(prod, wrap(fused), get, args.steps, cp)
        row["cxx_host_vs_numpy"] = trajectory(prod, wrap(host), get, args.host_steps, np)

        def host_vs_cuda(st, sw):                     # C++ host trajectory, compared to C++ CUDA
            return wrap(host)(st, sw)
        row["cxx_host_vs_cxx_cuda"] = trajectory(
            lambda st, sw: (lambda r: (SubstrateState(r[0].anomaly.get(), r[0].surface_anomaly.get()), r[1]))(
                wrap(cud)(SubstrateState(cp.asarray(st.anomaly), cp.asarray(st.surface_anomaly)), cp.asarray(sw))),
            host_vs_cuda, get, args.host_steps, np)

        def timed(fn, xp, n_steps, reps):
            ts = []
            for rep in range(reps + 1):
                st = SubstrateState(xp.asarray(a0).copy(), xp.asarray(s0).copy())
                sws = [fdev((first + k + 1) * c.dt) for k in range(n_steps)]
                if xp is np:
                    sws = [x.get() for x in sws]
                sync(); t = time.perf_counter()
                for k in range(n_steps):
                    st = fn(st, sws[k])[0]
                sync()
                if rep:
                    ts.append((time.perf_counter() - t) / n_steps)
            return {"median": statistics.median(ts), "min": min(ts), "max": max(ts), "n": len(ts)}
        steps_t = min(args.steps, 60)
        row["seconds_per_step"] = {
            "cupy_production": timed(prod, cp, steps_t, args.repeats),
            "tracer_kernel": timed(wrap(fused), cp, steps_t, args.repeats),
            "cxx_cuda": timed(wrap(cud), cp, steps_t, args.repeats),
            "numpy_production": timed(prod, np, min(args.host_steps, 5), 1),
            "cxx_host_single_thread": timed(wrap(host), np, min(args.host_steps, 5), 1)}
        t = row["seconds_per_step"]
        row["cxx_cuda_over_tracer_time"] = t["cxx_cuda"]["median"] / t["tracer_kernel"]["median"]
        row["cxx_cuda_speedup_vs_cupy"] = t["cupy_production"]["median"] / t["cxx_cuda"]["median"]
        row["cxx_host_speedup_vs_numpy"] = (t["numpy_production"]["median"]
                                            / t["cxx_host_single_thread"]["median"])
        k = cud.kernel
        row["cxx_cuda_kernel"] = {"registers": k.num_regs, "local_bytes": k.local_size_bytes}
        row["tracer_kernel"] = {"registers": fused.kernel.num_regs,
                                "local_bytes": fused.kernel.local_size_bytes}
        out["cases"][name] = row
        print(name, json.dumps(row), flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
