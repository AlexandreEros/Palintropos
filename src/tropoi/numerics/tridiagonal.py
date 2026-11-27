"""Batched tridiagonal solution.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

__all__ = ["thomas"]


def thomas(lower, diag, upper, rhs, xp):
    """Solve a batch of tridiagonal systems along axis 0 (Thomas algorithm,
    no pivoting; the Newton Jacobians here are diagonally dominant)."""
    n = diag.shape[0]
    cp = [None] * n
    dp = [None] * n
    cp[0] = upper[0] / diag[0] if n > 1 else None
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / den
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / den
    out = xp.empty_like(rhs)
    out[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out
