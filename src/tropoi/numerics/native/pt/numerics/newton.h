// Safeguarded Newton iteration for one system with a tridiagonal Jacobian
// (tropoi.numerics). Generic: it knows nothing about any physical model.
//
// The Problem supplies
//
//   static constexpr int CAP;                 // capacity: unknowns <= CAP
//   int size() const;                         // unknowns n
//   real residual(const real* x, real* r) const;
//       // residuals r (n); returns the largest |r_i| / tolerance_i
//       // (<= 1 means converged)
//   void jacobian(const real* x, real* lower, real* diag, real* upper) const;
//       // the tridiagonal Jacobian at x
//   real admissible_fraction(const real* x, const real* dx) const;
//       // the largest admissible fraction of the full step dx (the problem's
//       // own admissibility constraint, e.g. positivity of a physical state)
//
// Iteration: x <- x + lambda dx, with dx the Newton step, lambda starting at
// min(1, admissible fraction) and halved up to `halvings` times until the
// residual ratio decreases or reaches convergence. A system not converged
// after max_iterations is reported as failed; x is then not a solution and
// must not be accepted by the caller.
//
// Each residual is evaluated once: an accepted trial point is exactly the
// next iterate (x + lambda dx, the same expression), so its residual is
// reused, and the Jacobian is assembled only when a step is taken. The
// iterates are those of evaluating residual and Jacobian at every pass.
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
PT_HD NewtonOutcome newton_solve(const Problem& p, real* x, const int max_iterations,
                                 const int halvings = 30) {
    constexpr int CAP = Problem::CAP;
    const int n = p.size();
    real r[CAP], rt[CAP], lo[CAP], di[CAP], up[CAP], rhs[CAP], cpr[CAP], dx[CAP], xt[CAP];
    int it = 0;
    real ratio = p.residual(x, r);
    while (true) {
        if (ratio <= 1.0) return {it, 0, ratio};
        if (it >= max_iterations) return {it, 1, ratio};
        p.jacobian(x, lo, di, up);
        for (int i = 0; i < n; ++i) rhs[i] = -r[i];
        thomas<CAP>(n, lo, di, up, rhs, cpr, dx);
        real lam = minimum(real(1.0), p.admissible_fraction(x, dx));
        bool accepted = false;
        real tr(0.0);
        for (int t = 0; t < halvings; ++t) {
            for (int i = 0; i < n; ++i) xt[i] = x[i] + lam * dx[i];
            tr = p.residual(xt, rt);
            if ((tr < ratio) | (tr <= 1.0)) { accepted = true; break; }
            lam = real(0.5) * lam;
        }
        for (int i = 0; i < n; ++i) x[i] = x[i] + lam * dx[i];
        if (accepted) {                     // x is the accepted trial point, bit for bit
            ratio = tr;
            for (int i = 0; i < n; ++i) r[i] = rt[i];
        } else {                            // lambda was halved once more after the last trial
            ratio = p.residual(x, r);
        }
        ++it;
    }
}

}  // namespace pt
