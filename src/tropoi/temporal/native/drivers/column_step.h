// One backward-Euler substrate step of one column (j of n), shared by the
// CUDA and the CPU drivers. Column-major-by-level arrays: element (i, j) of an
// (L, n) array is at i * n + j, so threads of consecutive columns read
// consecutive addresses.
//
// Besides the solve, the column step validates its forcing, forms the
// absorbed constant of the skin balance from it, and records the column's
// temperature extent per stratum, so that none of these needs a separate
// pass over the columns (audit P3).
#pragma once
#include "pt/numerics/newton.h"
#include "pt/physics/substrate.h"
#include "pt/temporal/backward_euler.h"

namespace pt {

//: scalars, host-computed: dt, 2/dz1, -(2/dz1), eps sigma, 4 eps sigma,
//: tolerance floor, tolerance round-off multiple, 1 - A, eps
struct StepScalars {
    double dt, two_over_dz1, neg_two_over_dz1, eps_sigma, four_eps_sigma, tol_floor, tol_roundoff,
        one_minus_albedo, emissivity;
};

// A per-column forcing input: p[j * stride] (stride 0 broadcasts one value),
// or `value` when p is null.
struct ColumnInput {
    const double* p;
    long long stride;
    double value;
    PT_HD double at(int j) const { return p ? p[(long long)j * stride] : value; }
};

struct ColumnArrays {
    int L, n, strata;
    const double *th_old, *ths_old;                    // (L, n), (n)
    // surface forcing (W m^-2): incident shortwave and longwave, the extra
    // flux F_0 and its derivative a, the linearization reference temperature
    // (K; when has_fref is 0 the reference is the skin's own anomaly), and G
    ColumnInput shortwave, longwave, flux, dflux, fref_temperature, bottom;
    int has_fref;
    const double* source;                              // element (i, j) at
    long long source_row, source_col;                  // i * row + j * col, or null
    const double *rho, *kc, *chi, *dz, *half;          // (L)
    const int* stratum;                                // (L)
    double *th_out, *ths_out;                          // (L, n), (n)
    int *iterations, *failed, *invalid;                // (n)
    double* ratio;                                     // (n)
    double *rep_storage, *rep_r, *rep_tau;             // (L, n)
    double *rep_rs, *rep_taus, *rep_qs, *rep_emitted, *rep_absorbed;   // (n)
    // per stratum s, the column's lowest (row s) and highest (row strata + s)
    // temperature after the step, skin in the top stratum (2 strata, n), or null
    double* extent;
};

template <class Physics>
PT_HD void configure(Physics& ph, const ColumnArrays& a, const StepScalars& s) {
    ph.L = a.L;
    ph.rho = a.rho; ph.kc = a.kc; ph.chi = a.chi; ph.dz = a.dz; ph.half = a.half;
    ph.stratum = a.stratum;
    ph.two_over_dz1 = s.two_over_dz1; ph.neg_two_over_dz1 = s.neg_two_over_dz1;
    ph.eps_sigma = s.eps_sigma; ph.four_eps_sigma = s.four_eps_sigma;
    ph.tol_floor = s.tol_floor; ph.tol_roundoff = s.tol_roundoff;
}

template <class Physics>
PT_HD void column_step(const Physics& ph, const ColumnArrays& a, const StepScalars& s,
                       int max_iterations, int j) {
    constexpr int LC = Physics::LAYER_CAP;
    const int L = a.L, n = a.n;
    real tho[LC], x[LC + 1];
    double src[LC];
    // the forcing, validated: every input finite and a <= 0
    const real sw(a.shortwave.at(j)), lw(a.longwave.at(j)), f0(a.flux.at(j));
    const real dflux(a.dflux.at(j)), gb(a.bottom.at(j));
    bool ok = is_finite(sw) && is_finite(lw) && is_finite(f0) && is_finite(dflux) &&
              dflux <= real(0.0) && is_finite(gb);
    real fref(a.ths_old[j]);
    if (a.has_fref) {
        fref = real(a.fref_temperature.at(j)) - real(SUBSTRATE_TREF);   // as an anomaly
        ok = ok && is_finite(fref);
    }
    for (int i = 0; i < L; ++i) {
        tho[i] = a.th_old[(long long)i * n + j];
        x[i + 1] = tho[i];
        if (a.source) {
            src[i] = a.source[(long long)i * a.source_row + (long long)j * a.source_col];
            ok = ok && is_finite(real(src[i]));
        }
    }
    a.invalid[j] = ok ? 0 : 1;
    x[0] = a.ths_old[j];
    // the absorbed constant (1 - A) SW + eps LW + F_0
    const real absorbed = real(s.one_minus_albedo) * sw + real(s.emissivity) * lw + f0;
    BackwardEulerColumn<Physics> be{ph, tho, a.source ? src : nullptr, real(s.dt), absorbed,
                                    dflux, fref, gb};
    const NewtonOutcome o = newton_solve(be, x, max_iterations);
    for (int i = 0; i < L; ++i) a.th_out[(long long)i * n + j] = x[i + 1].v;
    a.ths_out[j] = x[0].v;
    a.iterations[j] = o.iterations;
    a.failed[j] = o.failed;
    a.ratio[j] = o.ratio.v;
    if (a.rep_storage) {
        // the converged evaluation, re-evaluated (deterministic: the same
        // arithmetic as Newton's last residual), written for the step report
        real R[LC + 1];
        double storage[LC], r[LC], tau[LC];
        typename BackwardEulerColumn<Physics>::Report out{
            storage, r, tau, &a.rep_rs[j], &a.rep_taus[j], &a.rep_qs[j], &a.rep_emitted[j],
            &a.rep_absorbed[j]};
        // its largest |r| / tau: Newton's last ratio, recomputed with the report
        a.ratio[j] = be.residual(x, R, &out).v;
        for (int i = 0; i < L; ++i) {
            a.rep_storage[(long long)i * n + j] = storage[i];
            a.rep_r[(long long)i * n + j] = r[i];
            a.rep_tau[(long long)i * n + j] = tau[i];
        }
    }
    if (a.extent) {
        // T_ref + min(x) == min(T_ref + x) exactly, since rounding is monotonic
        const real T_ref(SUBSTRATE_TREF);
        const int S = a.strata;
        for (int st = 0; st < S; ++st) {
            a.extent[(long long)st * n + j] = PT_INF;            // min(+inf, T) == T
            a.extent[(long long)(S + st) * n + j] = -PT_INF;
        }
        for (int i = 0; i <= L; ++i) {
            const int st = (i == 0) ? 0 : ph.stratum[i - 1];      // the skin: top stratum
            const real T = T_ref + x[i];
            double& lo = a.extent[(long long)st * n + j];
            double& hi = a.extent[(long long)(S + st) * n + j];
            lo = minimum(real(lo), T).v;
            hi = maximum(real(hi), T).v;
        }
    }
}

}  // namespace pt
