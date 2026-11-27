// One backward-Euler substrate step for one column: shared by both drivers.
// Feasibility prototype, NOT production code.
#pragma once
#include "pt/numerics/newton.h"
#include "pt/physics/substrate.h"
#include "pt/temporal/backward_euler.h"

namespace pt {

struct StepScalars {
    double dt, two_over_dz1, neg_two_over_dz1, eps_sigma, four_eps_sigma, tol_floor, tol_roundoff;
    int segment_end, max_iterations;
};

template <int L, class Law0, class Law1>
PT_HD void column_step(int j, int n, const double* th_old, const double* ths_old,
                       const double* absorbed, const double* dflux, const double* tref,
                       const double* bottom, const double* rho, const double* kc,
                       const double* chi, const double* dz, const double* half,
                       const double* law_params, const StepScalars& s,
                       double* th_out, double* ths_out, int* iters, int* fail, double* ratio_out) {
    Substrate<L, Law0, Law1> ph;
    ph.rho = rho; ph.kc = kc; ph.chi = chi; ph.dz = dz; ph.half = half;
    ph.segment_end = s.segment_end;
    ph.law0.load(law_params);
    ph.law1.load(law_params + Law0::nparams);
    ph.two_over_dz1 = s.two_over_dz1; ph.neg_two_over_dz1 = s.neg_two_over_dz1;
    ph.eps_sigma = s.eps_sigma; ph.four_eps_sigma = s.four_eps_sigma;
    ph.tol_floor = s.tol_floor; ph.tol_roundoff = s.tol_roundoff;
    real tho[L], x[L + 1];
    for (int i = 0; i < L; ++i) { tho[i] = th_old[(long)i * n + j]; x[i + 1] = tho[i]; }
    x[0] = ths_old[j];
    BackwardEulerColumn<Substrate<L, Law0, Law1>> be{ph, tho, real(s.dt), real(absorbed[j]),
                                                     real(dflux[j]), real(tref[j]), real(bottom[j])};
    const NewtonOutcome o = newton_solve(be, x, s.max_iterations);
    for (int i = 0; i < L; ++i) th_out[(long)i * n + j] = x[i + 1].v;
    ths_out[j] = x[0].v;
    iters[j] = o.iterations; fail[j] = o.failed; ratio_out[j] = o.ratio.v;
}

}  // namespace pt
