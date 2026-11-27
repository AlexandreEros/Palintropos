// Batched-column Thomas elimination without pivoting (tropoi.numerics).
//
// Solves one tridiagonal system of n <= CAP rows: lower[i-1] couples row i to
// row i-1 and upper[i] couples row i to row i+1. No pivoting: the caller
// guarantees a system that is stable without it (diagonal dominance or an
// M-matrix). The operation order is the reference order of
// tropoi.numerics.tridiagonal.thomas.
#pragma once
#include "../portable.h"

namespace pt {

template <int CAP>
PT_HD void thomas(const int n, const real* lower, const real* diag, const real* upper,
                  const real* rhs, real* cprime, real* x) {
    real c = (n > 1) ? upper[0] / diag[0] : real(0.0);
    real d = rhs[0] / diag[0];
    cprime[0] = c;
    x[0] = d;
    for (int i = 1; i < n; ++i) {
        const real l = lower[i - 1];
        const real den = diag[i] - l * c;
        if (i < n - 1) c = upper[i] / den;
        d = (rhs[i] - l * d) / den;
        cprime[i] = c;
        x[i] = d;
    }
    real xv = d;
    for (int i = n - 2; i >= 0; --i) {
        xv = x[i] - cprime[i] * xv;
        x[i] = xv;
    }
}

}  // namespace pt
