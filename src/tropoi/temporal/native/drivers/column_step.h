// One backward-Euler substrate step of one column (j of n), shared by the
// CUDA and the CPU drivers. Column-major-by-level arrays: element (i, j) of an
// (L, n) array is at i * n + j, so threads of consecutive columns read
// consecutive addresses.
#pragma once
#include "pt/numerics/newton.h"
#include "pt/physics/substrate.h"
#include "pt/temporal/backward_euler.h"

namespace pt {

//: scalars, host-computed: dt, 2/dz1, -(2/dz1), eps sigma, 4 eps sigma,
//: tolerance floor, tolerance round-off multiple
struct StepScalars {
    double dt, two_over_dz1, neg_two_over_dz1, eps_sigma, four_eps_sigma, tol_floor, tol_roundoff;
};

struct ColumnArrays {
    int L, n;
    const double *th_old, *ths_old;                    // (L, n), (n)
    const double *absorbed, *dflux, *fref, *bottom;    // (n) each
    const double* source;                              // (L, n) or null
    const double *rho, *kc, *chi, *dz, *half;          // (L)
    const int* stratum;                                // (L)
    double *th_out, *ths_out;                          // (L, n), (n)
    int *iterations, *failed;                          // (n)
    double* ratio;                                     // (n)
    double *rep_storage, *rep_r, *rep_tau;             // (L, n)
    double *rep_rs, *rep_taus, *rep_qs, *rep_emitted, *rep_absorbed;   // (n)
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

// A per-column view of a (L, n) array: writes element i of column j.
struct Strided {
    double* base;
    long stride;
    PT_HD double& operator[](int i) const { return base[(long)i * stride]; }
};

template <class Physics>
PT_HD void column_step(const Physics& ph, const ColumnArrays& a, const StepScalars& s,
                       int max_iterations, int j) {
    constexpr int LC = Physics::LAYER_CAP;
    const int L = a.L, n = a.n;
    real tho[LC], x[LC + 1];
    double src[LC];
    for (int i = 0; i < L; ++i) {
        tho[i] = a.th_old[(long)i * n + j];
        x[i + 1] = tho[i];
        if (a.source) src[i] = a.source[(long)i * n + j];
    }
    x[0] = a.ths_old[j];
    BackwardEulerColumn<Physics> be{ph, tho, a.source ? src : nullptr, real(s.dt),
                                    real(a.absorbed[j]), real(a.dflux[j]), real(a.fref[j]),
                                    real(a.bottom[j])};
    const NewtonOutcome o = newton_solve(be, x, max_iterations);
    for (int i = 0; i < L; ++i) a.th_out[(long)i * n + j] = x[i + 1].v;
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
        be.residual(x, R, &out);
        for (int i = 0; i < L; ++i) {
            a.rep_storage[(long)i * n + j] = storage[i];
            a.rep_r[(long)i * n + j] = r[i];
            a.rep_tau[(long)i * n + j] = tau[i];
        }
    }
}

}  // namespace pt
