"""Bitwise check of traced constitutive laws (``trace_prototype.py``) against
CuPy evaluating the same unmodified methods, and timing of one fused
evaluation against the CuPy ufunc chain.

    PYTHONPATH=src:. python tools/colab/trace_check.py --out trace.json [--n 826200]

The default n is 51 x 16200, the production substrate's cell count.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main(argv=None) -> int:
    import cupy as cp
    import trace_prototype as tp
    from tropoi.physics.materials import CellMaterials, PolynomialHeatCapacity, ledlow_1992
    from tropoi.run.airless.config import HAYNE_2017
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=51 * 16200)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(3)
    n = args.n
    # temperatures across both Ledlow branches and the 350 K breakpoint, plus
    # pairs that straddle it, nearly equal pairs and exactly equal pairs
    t1 = rng.uniform(20.0, 720.0, n)
    t0 = t1 + rng.choice([-1, 1], n) * 10.0 ** rng.uniform(-12, 2, n)
    t0[: n // 10] = t1[: n // 10]
    t0[n // 10: n // 5] = 700.0 - t1[n // 10: n // 5]
    t0 = np.clip(t0, 15.0, 730.0)
    delta = t1 - t0
    kc = rng.uniform(1e-4, 1e-2, n)
    chi = rng.uniform(0.0, 2.0, n) / 350.0 ** 3
    d = {k: cp.asarray(v) for k, v in dict(t1=t1, t0=t0, delta=delta, kc=kc, chi=chi).items()}
    ledlow = ledlow_1992()
    hayne = PolynomialHeatCapacity(HAYNE_2017["cp"])
    cases = {
        "ledlow_specific_heat": (ledlow.specific_heat, ["t1"]),
        "ledlow_enthalpy_difference": (ledlow.enthalpy_difference, ["t1", "t0", "delta"]),
        "hayne_specific_heat": (hayne.specific_heat, ["t1"]),
        "hayne_enthalpy_difference": (hayne.enthalpy_difference, ["t1", "t0", "delta"]),
        "conductivity": (CellMaterials.conductivity, ["kc", "chi", "t1"]),
        "conductivity_derivative": (CellMaterials.conductivity_derivative, ["kc", "chi", "t1"]),
        "kirchhoff_difference": (CellMaterials.kirchhoff_difference,
                                 ["kc", "chi", "t1", "t0", "delta"]),
    }
    out = {"n": n, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "cases": {}}

    def bench(fn, reps=20):
        fn(); cp.cuda.Device().synchronize()
        ts = []
        for _ in range(5):
            cp.cuda.Device().synchronize(); t = time.perf_counter()
            for _ in range(reps):
                fn()
            cp.cuda.Device().synchronize(); ts.append((time.perf_counter() - t) / reps)
        return sorted(ts)[2]

    for name, (fn, names) in cases.items():
        inputs = [d[k] for k in names]
        ref = fn(*inputs)                                    # the CuPy ufunc chain
        host = fn(*[x.get() for x in inputs])                # the NumPy reference
        row = {"statements": None}
        for fmad in (False, True):
            k, body = tp.elementwise(fn, names, f"traced_{name}_{int(fmad)}", fmad=fmad)
            got = k(*inputs)
            row["statements"] = body.count(";")
            row[f"fmad_{str(fmad).lower()}"] = {
                "differs_from_cupy": int((got != ref).sum()),
                "differs_from_numpy": int((got.get() != host).sum()),
                "max_relative_vs_cupy": float(cp.max(cp.abs(got - ref) / cp.maximum(cp.abs(ref), 1e-300)))}
        row["cupy_numpy_differ"] = int((ref.get() != host).sum())
        k_off, _ = tp.elementwise(fn, names, f"traced_{name}_t", fmad=False)
        row["seconds_cupy_chain"] = bench(lambda: fn(*inputs))
        row["seconds_traced_kernel"] = bench(lambda: k_off(*inputs))
        out["cases"][name] = row
        print(name, json.dumps(row))
    out["device_function_example"] = tp.device_function(ledlow.specific_heat, ["T"],
                                                        "ledlow_specific_heat")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
