// Safeguarded Newton iteration for one column with a tridiagonal Jacobian.
// Generic: it knows nothing about the physics. The Problem supplies
//
//   static constexpr int N;                       // unknowns
//   real evaluate(const real* x, bool jacobian,   // returns max |r|/tau (<= 1: converged)
//                 real* r, real* lower, real* diag, real* upper);
//   real admissible_fraction(const real* x, const real* dx);   // caller's admissibility
//
// The iteration is the one in ConductiveSubstrate.step, per column: a step is
// the Newton update scaled by min(1, admissible fraction), halved up to
// `halvings` times until the residual ratio decreases or converges.
// Feasibility prototype, NOT production code.
#pragma once
#include "../portable.h"
#include "thomas.h"

namespace pt {

struct NewtonOutcome {
    int iterations;
    int failed;
    real ratio;
};

template <class Problem>
PT_HD NewtonOutcome newton_solve(Problem& p, real* x, int max_iterations, int halvings = 30) {
    constexpr int N = Problem::N;
    real r[N], rt[N], lo[N - 1], di[N], up[N - 1], rhs[N], cpr[N], dx[N], xt[N];
    int it = 0;
    real ratio;
    while (true) {
        ratio = p.evaluate(x, true, r, lo, di, up);
        if (ratio <= 1.0) return {it, 0, ratio};
        if (it >= max_iterations) return {it, 1, ratio};
        for (int i = 0; i < N; ++i) rhs[i] = -r[i];
        thomas<N>(lo, di, up, rhs, cpr, dx);
        real lam = minimum(real(1.0), p.admissible_fraction(x, dx));
        for (int t = 0; t < halvings; ++t) {
            for (int i = 0; i < N; ++i) xt[i] = x[i] + lam * dx[i];
            const real tr = p.evaluate(xt, false, rt, lo, di, up);
            if ((tr < ratio) | (tr <= 1.0)) break;
            lam = real(0.5) * lam;
        }
        for (int i = 0; i < N; ++i) x[i] = x[i] + lam * dx[i];
        ++it;
    }
}

}  // namespace pt
