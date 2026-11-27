// Batched-column Thomas elimination, no pivoting (the caller guarantees a
// stable system). Same operation order as tropoi.physics.substrate._tridiagonal.
// lower[i-1] couples row i to row i-1, upper[i] couples row i to row i+1.
// Feasibility prototype, NOT production code.
#pragma once
#include "../portable.h"

namespace pt {

template <int N>
PT_HD void thomas(const real* lower, const real* diag, const real* upper, const real* rhs,
                  real* cprime, real* x) {
    real c = (N > 1) ? upper[0] / diag[0] : real(0.0);
    real d = rhs[0] / diag[0];
    cprime[0] = c;
    x[0] = d;
    for (int i = 1; i < N; ++i) {
        const real l = lower[i - 1];
        const real den = diag[i] - l * c;
        if (i < N - 1) c = upper[i] / den;
        d = (rhs[i] - l * d) / den;
        cprime[i] = c;
        x[i] = d;
    }
    real xv = d;
    for (int i = N - 2; i >= 0; --i) {
        xv = x[i] - cprime[i] * xv;
        x[i] = xv;
    }
}

}  // namespace pt
