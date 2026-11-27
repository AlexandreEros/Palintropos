"""D3 equivalence study: the boundary layer's implicit vertical diffusion
solved with its own elimination and with the canonical ``pt::thomas``.

    PYTHONPATH=src:. python tools/colab/bl_thomas_equivalence.py --out study.json \
        [--warm-steps 144] [--captures 30]

Measurement only: nothing in the package changes. The script

1. runs the heated radiative case on the GPU for ``--warm-steps`` steps,
   then records the inputs of ``--captures`` calls of
   ``implicit_column_diffusion`` made by the boundary-layer term (u, v and
   dry static energy, with their masses, conductances, windows and the
   implicit surface stress) together with the production result;
2. solves each captured system again with a CUDA kernel that assembles the
   same tridiagonal system with the same arithmetic (products as
   ``__dmul_rn``) and eliminates it with ``pt::thomas`` from
   ``tropoi/numerics/native/pt/numerics/thomas.h``, the canonical
   elimination;
3. compares both double-precision solutions with each other and with an
   extended-precision solution (NumPy ``longdouble``, assembled from the same
   float64 inputs), on every column, in units of the column's largest
   ``|q|``; and compares the conservation of the mass-weighted column
   integral (zero-flux systems);
4. times one solve both ways (synchronized, median of repeats), and the u/v
   pair in one launch that assembles the matrix once (P6: <= 1.0 ms per solve
   at T21L40). ``pt::thomas`` repeats the elimination for each right-hand
   side, so the pair shares assembly and the launch, not a factorization.

Errors against the extended-precision solution are differences taken in
``longdouble``: the double-precision results are promoted, not the reference
rounded.

The GPU model is recorded; the T4 is the reference device for the timings.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

KERNEL = r'''
#include "pt/numerics/thomas.h"
#define CAP 128
// Assemble the boundary layer's system with its arithmetic, then eliminate
// with pt::thomas: diag b_k = m_k + w (g_above + g_below), lower = upper =
// -w g, rhs d_k = m_k q_k. nrhs right-hand sides share the assembled matrix;
// pt::thomas eliminates it again for each one (no factorization reuse).
extern "C" __global__ void bl_thomas(const int K, const int n, const int nrhs,
                                     const double* q, const double* mass,
                                     const double* cond, const int has_bottom,
                                     const double* bottom, const double w, double* out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    using pt::real;
    real lo[CAP], di[CAP], up[CAP], rhs[CAP], c[CAP], x[CAP];
    const real ww(w);
    for (int k = 0; k < K; ++k) {
        const real ga = k > 0 ? real(cond[(long)(k - 1) * n + j]) : real(0.0);
        const real gb = k < K - 1 ? real(cond[(long)k * n + j])
                                  : (has_bottom ? real(bottom[j]) : real(0.0));
        di[k] = real(mass[(long)k * n + j]) + ww * (ga + gb);
        if (k < K - 1) {
            const real off = -(ww * real(cond[(long)k * n + j]));
            lo[k] = off;
            up[k] = off;
        }
    }
    for (int r = 0; r < nrhs; ++r) {
        const double* qr = q + (long)r * K * n;
        for (int k = 0; k < K; ++k) rhs[k] = real(mass[(long)k * n + j]) * real(qr[(long)k * n + j]);
        pt::thomas<CAP>(K, lo, di, up, rhs, c, x);
        for (int k = 0; k < K; ++k) out[(long)r * K * n + (long)k * n + j] = x[k].v;
    }
}
'''


def _kernel(cp):
    from tropoi.temporal.native import build_identity
    root = build_identity.TROPOI / "numerics" / "native"
    mod = cp.RawModule(code=KERNEL, options=(f"-I{root}", "--std=c++14"))
    return mod.get_function("bl_thomas")


def solve_native(cp, kernel, q, mass, cond, w, bottom=None, block=64):
    """q (K, n) or (r, K, n)."""
    qq = q if q.ndim == 3 else q[None]
    r, K, n = qq.shape
    out = cp.empty_like(qq)
    kernel(((n + block - 1) // block,), (block,),
           (np.int32(K), np.int32(n), np.int32(r), cp.ascontiguousarray(qq),
            cp.ascontiguousarray(mass), cp.ascontiguousarray(cond),
            np.int32(bottom is not None), cp.zeros(1) if bottom is None else
            cp.ascontiguousarray(bottom), np.float64(w), out))
    return out if q.ndim == 3 else out[0]


def solve_extended(q, mass, cond, w, bottom=None):
    """The same system in NumPy longdouble (one Thomas sweep, any order)."""
    L = np.longdouble
    q, m, g = q.astype(L), mass.astype(L), cond.astype(L)
    K, n = q.shape
    w = L(w)
    zero = np.zeros(n, dtype=L)
    ga = [zero] + [g[k - 1] for k in range(1, K)]
    gb = [g[k] for k in range(K - 1)] + [bottom.astype(L) if bottom is not None else zero]
    diag = [m[k] + w * (ga[k] + gb[k]) for k in range(K)]
    off = [-w * g[k] for k in range(K - 1)]
    rhs = [m[k] * q[k] for k in range(K)]
    c, d = [None] * K, [None] * K
    c[0] = off[0] / diag[0] if K > 1 else zero
    d[0] = rhs[0] / diag[0]
    for k in range(1, K):
        den = diag[k] - off[k - 1] * c[k - 1]
        c[k] = off[k] / den if k < K - 1 else zero
        d[k] = (rhs[k] - off[k - 1] * d[k - 1]) / den
    x = [None] * K
    x[K - 1] = d[K - 1]
    for k in range(K - 2, -1, -1):
        x[k] = d[k] - c[k] * x[k + 1]
    return np.stack(x)


def _ulps(a, b):
    """|a - b| in units in the last place of b (float64)."""
    return np.abs(a - b) / np.spacing(np.abs(b).astype(np.float64))


def main(argv=None) -> int:
    import cupy as cp
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default="thai-ben2-ck", choices=("thai-ben2-ck",),
                    help="a preset with a boundary layer (the Earth gray preset has none)")
    ap.add_argument("--warm-steps", type=int, default=144)
    ap.add_argument("--captures", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--tiny", action="store_true", help="T9 16x32 (smoke test only)")
    args = ap.parse_args(argv)
    import tropoi.temporal.tendencies.boundary_layer as tbl
    from tropoi.physics.boundary_layer import implicit_column_diffusion
    from tropoi.run.radiative.config import PRESETS
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = PRESETS["dev-" + args.case]()
    if args.tiny:
        cfg = cfg.with_(experiment_id=cfg.experiment_id + "-tiny", l_max=10,
                        retained_truncation=9, nlat=16, nlon=32)
    if cfg.boundary_layer is None:
        ap.error(f"case {args.case!r} has no boundary layer: no solve would be captured")
    run = RadiativeRun(cfg, pathlib.Path(tempfile.mkdtemp()) / "run", allow_dirty=True,
                       log=lambda s: None)
    for _ in range(args.warm_steps):
        run.stepper.step()
    captured = []

    def capture(q, layer_mass, conductance, window, *, bottom_conductance=None):
        out = implicit_column_diffusion(q, layer_mass, conductance, window,
                                        bottom_conductance=bottom_conductance)
        if len(captured) < args.captures:
            captured.append({"q": q.copy(), "mass": layer_mass.copy(),
                             "cond": conductance.copy(), "w": float(window),
                             "bottom": None if bottom_conductance is None
                             else bottom_conductance.copy(), "out": out.copy()})
        return out

    tbl.implicit_column_diffusion = capture
    try:
        while len(captured) < args.captures:
            run.stepper.step()
    finally:
        tbl.implicit_column_diffusion = implicit_column_diffusion
    kernel = _kernel(cp)
    rows = []
    for c in captured:
        prod = c["out"]
        nat = solve_native(cp, kernel, c["q"], c["mass"], c["cond"], c["w"], c["bottom"])
        h = {k: (v.get() if hasattr(v, "get") else v) for k, v in c.items()}
        ext = solve_extended(h["q"], h["mass"], h["cond"], h["w"], h["bottom"])
        P, N = prod.get(), nat.get()
        scale = np.max(np.abs(h["q"]), axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        L = np.longdouble
        e_prod = (np.abs(P.astype(L) - ext) / scale).astype(np.float64)
        e_nat = (np.abs(N.astype(L) - ext) / scale).astype(np.float64)
        d = np.abs(P - N) / scale
        exactly = float(np.mean(P == N))
        # rows coupled to a neighbour (above the boundary layer the system is
        # diagonal and every elimination returns D/B exactly)
        g = h["cond"]
        coupled = np.zeros(P.shape, dtype=bool)
        coupled[:-1] |= g != 0
        coupled[1:] |= g != 0
        if h["bottom"] is not None:
            coupled[-1] |= h["bottom"] != 0
        exactly_coupled = (float(np.mean(P[coupled] == N[coupled])) if coupled.any()
                           else None)
        row = {"window_s": h["w"], "levels": int(P.shape[0]), "columns": int(P.shape[1]),
               "bottom_stress": h["bottom"] is not None,
               "fraction_bitwise_equal": exactly,
               "fraction_coupled_rows": float(coupled.mean()),
               "fraction_bitwise_equal_coupled_rows": exactly_coupled,
               "max_rel_difference": float(d.max()), "p99_rel_difference": float(
                   np.percentile(d, 99)),
               "max_ulps_production_vs_canonical": float(np.max(_ulps(P, N))),
               "max_rel_error_production": float(e_prod.max()),
               "max_rel_error_canonical": float(e_nat.max()),
               "mean_rel_error_production": float(e_prod.mean()),
               "mean_rel_error_canonical": float(e_nat.mean())}
        if h["bottom"] is None:
            m = h["mass"].astype(np.longdouble)
            before = np.sum(m * h["q"], axis=0)
            for label, X in (("production", P), ("canonical", N)):
                after = np.sum(m * X.astype(np.longdouble), axis=0)
                row[f"max_conservation_rel_error_{label}"] = float(
                    np.max(np.abs(after - before) / np.abs(before)))
        rows.append(row)

    def timed(fn):
        ts = []
        for _ in range(args.repeats + 1):
            cp.cuda.Device().synchronize()
            t = time.perf_counter()
            fn()
            cp.cuda.Device().synchronize()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts[1:])

    mom = next(c for c in captured if c["bottom"] is not None)
    uv = cp.stack([mom["q"], mom["q"] * 0.5])
    timings = {
        "production_one_solve_s": timed(lambda: implicit_column_diffusion(
            mom["q"], mom["mass"], mom["cond"], mom["w"], bottom_conductance=mom["bottom"])),
        "canonical_one_solve_s": timed(lambda: solve_native(
            cp, kernel, mom["q"], mom["mass"], mom["cond"], mom["w"], mom["bottom"])),
        "production_uv_pair_s": timed(lambda: [implicit_column_diffusion(
            q, mom["mass"], mom["cond"], mom["w"], bottom_conductance=mom["bottom"])
            for q in uv]),
        "canonical_uv_pair_one_launch_s": timed(lambda: solve_native(
            cp, kernel, uv, mom["mass"], mom["cond"], mom["w"], mom["bottom"])),
    }
    props = cp.cuda.runtime.getDeviceProperties(0)
    keys = [k for k in rows[0] if k.startswith(("max_", "mean_", "p99_", "fraction"))]
    summary = {k: (min if k.startswith("fraction") else max)(r[k] for r in rows
                                                            if r.get(k) is not None)
               for k in keys + ["max_conservation_rel_error_production",
                                "max_conservation_rel_error_canonical"]
               if any(k in r for r in rows)}
    doc = {"case": args.case, "experiment_id": cfg.experiment_id, "warm_steps": args.warm_steps,
           "captures": len(rows), "gpu": props["name"].decode(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
           "summary_over_captures": summary, "timings": timings, "captures_detail": rows}
    pathlib.Path(args.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("case", "captures", "gpu", "summary_over_captures",
                                          "timings")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
