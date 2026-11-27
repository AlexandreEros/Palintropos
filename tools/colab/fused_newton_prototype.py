"""Feasibility prototype (NOT production code) for decision D7: one
backward-Euler substrate step as ONE kernel launch, with the whole safeguarded
Newton iteration resident on the device and one thread per column.

What is single-source and what is not:

* The material laws (heat capacity, enthalpy difference, conductivity, its
  derivative, the Kirchhoff difference) are **generated** from the unmodified
  ``tropoi.physics.materials`` methods by ``trace_prototype`` (no hand-written
  CUDA for them).
* The substrate's cell, face and skin arithmetic (storage, series face
  conductance, the skin equation, residual tolerances, the Jacobian rows) is
  **restated by hand below**, following ``ConductiveSubstrate._system`` term by
  term in its operation order. This is exactly the duplication that decisions
  D4/D9 must resolve before production. The prototype only measures what
  fusion achieves and whether bitwise agreement is reachable.
* The Newton driver (convergence test, Thomas solve, the positivity bound
  ``lambda <= 0.5 |T| / |dT|``, 30-halving backtracking) restates
  ``ConductiveSubstrate.step`` per column. The global loop of ``step`` is
  per-column separable: a converged column is frozen by ``lambda = 0``, and a
  column's line search only ever sees its own trial. So the per-column loop
  should reproduce each column's trajectory exactly.

Every product is ``__dmul_rn`` (never contracted into an FMA) and the
kernel is compiled with CuPy's default flags, so libdevice ``pow`` rounds as
it does in CuPy's own ``x ** 4`` (it does not under ``--fmad=false``). Per-cell arrays live in thread-local arrays,
which CUDA interleaves across the threads of a warp, so accesses coalesce.
The prototype supports what the production configuration uses: no source
term, scalar or per-column forcing. Its outputs are the new state, per-column
Newton iterations, the per-column final residual ratio and a failure flag.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_DRIVER = r"""
#define L {L}
#define NR {NR}
#define TREF 300.0
#define FLOOR {FLOOR}
// every product is __dmul_rn: never contracted into an FMA
#define MUL(a, b) __dmul_rn((a), (b))
#define ROUND {ROUND}

__device__ __forceinline__ double dmax_(double a, double b) {{
    return (isnan(a) | isnan(b)) ? (double)NAN : max(a, b);
}}
__device__ __forceinline__ double dmin_(double a, double b) {{
    return (isnan(a) | isnan(b)) ? (double)NAN : min(a, b);
}}

// Residuals (and optionally the Jacobian) at anomalies x = [ths, th_1..th_L]:
// a restatement of ConductiveSubstrate._system (see the module docstring).
__device__ double sub_system(const double* x, const double* tho, const int jac,
                             const double c, const double a, const double t_ref,
                             const double g_b, const double* rho, const double* kc,
                             const double* chi, const double* dz, const double* half,
                             const double dt, const double two_over_dz1,
                             const double neg_two_over_dz1, const double eps_sigma,
                             const double four_eps_sigma,
                             double* r, double* rs_out, double* lo, double* di, double* up)
{{
    double k[L], F[L + 1], cap[L];
    const double ths = x[0];
    const double Ts = TREF + ths;
    for (int i = 0; i < L; ++i) {{
        const double T = TREF + x[i + 1];
        k[i] = law_conductivity(kc[i], chi[i], T);
        cap[i] = MUL(rho[i], dz[i]);
    }}
    const double T0 = TREF + x[1];
    const double q_s = MUL(two_over_dz1, law_kirchhoff(kc[0], chi[0], Ts, T0, ths - x[1]));
    F[0] = q_s;
    for (int i = 0; i < L - 1; ++i) {{
        const double g = 1.0 / (half[i] / k[i] + half[i + 1] / k[i + 1]);
        F[i + 1] = MUL(g, x[i + 1] - x[i + 2]);
    }}
    F[L] = -g_b;
    double ratio = 0.0;
    for (int i = 0; i < L; ++i) {{
        const double T = TREF + x[i + 1];
        const double Told = TREF + tho[i];
        const double dh = {ENTHALPY}(i, T, Told, x[i + 1] - tho[i]);
        const double storage = MUL(cap[i], dh) / dt;
        r[i] = storage - (F[i] - F[i + 1]) - 0.0;
        const double tau = FLOOR + MUL(ROUND, fabs(storage) + fabs(F[i]) + fabs(F[i + 1]) + 0.0);
        ratio = (i == 0) ? fabs(r[i]) / tau : dmax_(ratio, fabs(r[i]) / tau);
    }}
    const double lin = MUL(a, ths - t_ref);
    const double emitted = MUL(eps_sigma, pow(Ts, 4.0));
    const double rs = c + lin - emitted - q_s;
    const double taus = FLOOR + MUL(ROUND, fabs(c) + fabs(lin) + emitted + fabs(q_s));
    *rs_out = rs;
    ratio = dmax_(ratio, fabs(rs) / taus);
    if (jac) {{
        const double dq_dts = MUL(two_over_dz1, law_conductivity(kc[0], chi[0], Ts));
        const double dq_dt1 = MUL(neg_two_over_dz1, k[0]);
        di[0] = a - MUL(four_eps_sigma, pow(Ts, 3.0)) - dq_dts;
        up[0] = -dq_dt1;
        lo[0] = -dq_dts;
        double dfa_prev = 0.0, dfb_prev = 0.0;
        for (int i = 0; i < L; ++i) {{
            const double T = TREF + x[i + 1];
            const double cpv = {SPECIFIC_HEAT}(i, T);
            double d = MUL(cap[i], cpv) / dt;
            d = d + MUL(0.0, T);
            if (i == 0) d = d - dq_dt1;
            if (i < L - 1) {{
                const double g = 1.0 / (half[i] / k[i] + half[i + 1] / k[i + 1]);
                const double drift = x[i + 1] - x[i + 2];
                const double gg = MUL(g, g);
                const double dki = law_conductivity_derivative(kc[i], chi[i], T);
                const double Tn = TREF + x[i + 2];
                const double dkn = law_conductivity_derivative(kc[i + 1], chi[i + 1], Tn);
                const double dfa = g + MUL(MUL(MUL(drift, gg), half[i]), dki) / MUL(k[i], k[i]);
                const double dfb = -g + MUL(MUL(MUL(drift, gg), half[i + 1]), dkn) / MUL(k[i + 1], k[i + 1]);
                d = d + dfa;
                lo[i + 1] = -dfa;
                up[i + 1] = dfb;
                dfa_prev = dfa;
                if (i > 0) d = d - dfb_prev;
                dfb_prev = dfb;
            }} else if (i > 0) {{
                d = d - dfb_prev;
            }}
            di[i + 1] = d;
        }}
    }}
    return ratio;
}}

extern "C" __global__
void fused_substrate_step(const double* __restrict__ th_old_g, const double* __restrict__ ths_old_g,
                          const double* __restrict__ cst, const double* __restrict__ aa,
                          const double* __restrict__ tref, const double* __restrict__ gb,
                          const double* __restrict__ rho, const double* __restrict__ kc,
                          const double* __restrict__ chi, const double* __restrict__ dz,
                          const double* __restrict__ half,
                          double* __restrict__ th_out, double* __restrict__ ths_out,
                          int* __restrict__ iters, int* __restrict__ fail,
                          double* __restrict__ ratio_out,
                          const double dt, const double two_over_dz1, const double neg_two_over_dz1,
                          const double eps_sigma, const double four_eps_sigma,
                          const int n, const int max_iter)
{{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double tho[L], x[NR], xt[NR], delta[NR], r[L], rt[L];
    double lo[L], di[NR], up[L], cp_[NR];
    for (int i = 0; i < L; ++i) {{ tho[i] = th_old_g[(long)i * n + j]; x[i + 1] = tho[i]; }}
    x[0] = ths_old_g[j];
    const double c = cst[j], a = aa[j], t_ref = tref[j], g_b = gb[j];
    int it = 0, failed = 0;
    double ratio, rs, rst;
    while (true) {{
        ratio = sub_system(x, tho, 1, c, a, t_ref, g_b, rho, kc, chi, dz, half, dt,
                           two_over_dz1, neg_two_over_dz1, eps_sigma, four_eps_sigma,
                           r, &rs, lo, di, up);
        if (ratio <= 1.0) break;
        if (it >= max_iter) {{ failed = 1; break; }}
        // rhs = -[r_s, r]; Thomas in the order of substrate._tridiagonal
        double cc = up[0] / di[0];
        double dd = (-rs) / di[0];
        cp_[0] = cc; delta[0] = dd;
        for (int i = 1; i < NR; ++i) {{
            const double l = lo[i - 1];
            const double den = di[i] - MUL(l, cc);
            if (i < NR - 1) cc = up[i] / den;
            dd = ((-r[i - 1]) - MUL(l, dd)) / den;
            cp_[i] = cc; delta[i] = dd;
        }}
        double xv = dd;
        for (int i = NR - 2; i >= 0; --i) {{ xv = delta[i] - MUL(cp_[i], xv); delta[i] = xv; }}
        // positivity: never more than halve an absolute temperature
        double m = INFINITY;
        for (int i = 0; i < NR; ++i) {{
            const double s = (delta[i] < 0.0) ? MUL(-0.5, TREF + x[i]) / delta[i] : INFINITY;
            m = (i == 0) ? s : min(m, s);
        }}
        double lam = dmin_(1.0, m);
        for (int t = 0; t < 30; ++t) {{
            for (int i = 0; i < NR; ++i) xt[i] = x[i] + MUL(lam, delta[i]);
            const double tr = sub_system(xt, tho, 0, c, a, t_ref, g_b, rho, kc, chi, dz, half, dt,
                                         two_over_dz1, neg_two_over_dz1, eps_sigma,
                                         four_eps_sigma, rt, &rst, lo, di, up);
            if ((tr < ratio) | (tr <= 1.0)) break;
            lam = MUL(0.5, lam);
        }}
        for (int i = 0; i < NR; ++i) x[i] = x[i] + MUL(lam, delta[i]);
        ++it;
    }}
    for (int i = 0; i < L; ++i) th_out[(long)i * n + j] = x[i + 1];
    ths_out[j] = x[0];
    iters[j] = it; fail[j] = failed; ratio_out[j] = ratio;
}}
"""


def _source(sub) -> str:
    import trace_prototype as tp
    from tropoi.physics.materials import CellMaterials
    from tropoi.physics.substrate import TOLERANCE_FLOOR, TOLERANCE_ROUNDOFF
    parts = [tp._PREAMBLE,
             tp.device_function(CellMaterials.conductivity, ["kc", "chi", "T"], "law_conductivity"),
             tp.device_function(CellMaterials.conductivity_derivative, ["kc", "chi", "T"],
                                "law_conductivity_derivative"),
             tp.device_function(CellMaterials.kirchhoff_difference, ["kc", "chi", "a", "b", "d"],
                                "law_kirchhoff")]
    # one generated enthalpy / specific-heat function per stratum segment
    dh_cases, cp_cases = [], []
    for s, (r0, r1, law) in enumerate(sub.materials.segments):
        parts.append(tp.device_function(law.enthalpy_difference, ["t1", "t0", "delta"], f"law_dh_{s}"))
        parts.append(tp.device_function(law.specific_heat, ["T"], f"law_cp_{s}"))
        dh_cases.append((r0, r1, f"law_dh_{s}"))
        cp_cases.append((r0, r1, f"law_cp_{s}"))

    def dispatch(name, cases, params, args):
        body = "".join(f"    if (i < {r1}) return {fn}({args});\n" for r0, r1, fn in cases)
        return (f"__device__ __forceinline__ double {name}(const int i, {params}) {{\n{body}"
                f"    return NAN;\n}}\n")
    parts.append(dispatch("seg_enthalpy", dh_cases,
                          "const double t1, const double t0, const double delta", "t1, t0, delta"))
    parts.append(dispatch("seg_specific_heat", cp_cases, "const double T", "T"))
    return "\n".join(parts) + _DRIVER.format(
        L=sub.layers, NR=sub.layers + 1, FLOOR=repr(TOLERANCE_FLOOR), ROUND=repr(TOLERANCE_ROUNDOFF),
        ENTHALPY="seg_enthalpy", SPECIFIC_HEAT="seg_specific_heat")


class FusedStep:
    """``step(state, dt, shortwave_down)`` for one ConductiveSubstrate (CuPy)."""

    def __init__(self, sub, block: int = 64):
        import cupy as cp
        from tropoi.physics.constants import SIGMA_SB
        self.sub, self.block = sub, block
        self.source = _source(sub)
        self.kernel = cp.RawKernel(self.source, "fused_substrate_step", options=())
        rho, kc, chi = sub.materials.on(cp)
        self.rho, self.kc, self.chi = (cp.ascontiguousarray(v.ravel()) for v in (rho, kc, chi))
        dz, half = sub._consts(cp)
        self.dz, self.half = cp.ascontiguousarray(dz.ravel()), cp.ascontiguousarray(half.ravel())
        dz1 = float(sub.thickness[0])
        self.two_over_dz1 = 2.0 / dz1
        self.neg_two_over_dz1 = -(2.0 / dz1)
        self.eps_sigma = sub.emissivity * SIGMA_SB
        self.four_eps_sigma = 4.0 * sub.emissivity * SIGMA_SB

    def step(self, state, dt, shortwave_down):
        import cupy as cp
        from tropoi.physics.substrate import SubstrateState
        sub = self.sub
        th, ths = state.anomaly, state.surface_anomaly
        n = th.shape[1]
        shape = (n,)
        sw = cp.broadcast_to(cp.asarray(shortwave_down, dtype=cp.float64), shape)
        zero = cp.zeros(shape)
        const = (1.0 - sub.albedo) * sw + sub.emissivity * zero + zero
        a = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(0.0), shape))
        gb = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(sub.bottom_heat_flux), shape))
        th_out, ths_out = cp.empty_like(th), cp.empty_like(ths)
        iters = cp.empty(n, dtype=cp.int32)
        fail = cp.empty(n, dtype=cp.int32)
        ratio = cp.empty(n)
        self.kernel(((n + self.block - 1) // self.block,), (self.block,),
                    (cp.ascontiguousarray(th), cp.ascontiguousarray(ths), cp.ascontiguousarray(const),
                     a, cp.ascontiguousarray(ths), gb, self.rho, self.kc, self.chi, self.dz,
                     self.half, th_out, ths_out, iters, fail, ratio, float(dt), self.two_over_dz1,
                     self.neg_two_over_dz1, self.eps_sigma, self.four_eps_sigma, cp.int32(n),
                     cp.int32(sub.max_iterations)))
        return SubstrateState(th_out, ths_out), iters, fail, ratio
