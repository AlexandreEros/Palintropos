"""Feasibility prototype (NOT production code): a compiled batched Thomas
solve for the substrate's Newton systems.

The benchmark (``substrate_benchmark.py``) patches it in place of
``tropoi.physics.substrate._tridiagonal`` for the duration of a measurement.
Nothing in ``src/`` imports it. If the owner approves a CUDA solver, a
canonical one belongs behind a numerical-solver interface, not here.

Layout: the arrays are the substrate's own, rows (levels) along axis 0 and
columns contiguous along axis 1, C order. One thread owns one column, so at
each level a warp reads 32 consecutive doubles: the accesses coalesce
across columns with no transpose.

Arithmetic: the same operations in the same order as ``_tridiagonal``:

    c_0 = u_0 / b_0,  d_0 = r_0 / b_0
    den = b_i - l_{i-1} c_{i-1};  c_i = u_i / den;  d_i = (r_i - l_{i-1} d_{i-1}) / den
    x_{n-1} = d_{n-1};  x_i = d_i - c_i x_{i+1}

CuPy evaluates each of those as a separate elementwise kernel, so nothing is
fused into an FMA there. NVRTC contracts ``a - b * c`` into an FMA by
default, which rounds differently. ``fmad=False`` (NVRTC ``--fmad=false``)
forbids contraction, so the prototype agrees with the CuPy and NumPy solves
bit for bit (measured). ``fmad=True`` is measured separately; it changes
rounding and needs the owner's approval (D5). This kernel calls no libdevice
function, so ``--fmad=false`` suffices here. The rule for production kernels
is the one in ``fused_newton_prototype`` and audit finding F2: write every
product as ``__dmul_rn`` and keep the default flags, because ``--fmad=false``
also changes libdevice ``pow``.
"""
from __future__ import annotations

_SOURCE = r"""
extern "C" __global__
void thomas_columns(const double* __restrict__ lower,   // (n-1, m): lower[i-1] couples row i to i-1
                    const double* __restrict__ diag,    // (n, m)
                    const double* __restrict__ upper,   // (n-1, m): upper[i] couples row i to i+1
                    const double* __restrict__ rhs,     // (n, m)
                    double* __restrict__ cprime,        // (n, m) workspace
                    double* __restrict__ out,           // (n, m): d' then x, in place
                    const int n, const int m)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const double b0 = diag[j];
    double c = (n > 1) ? upper[j] / b0 : 0.0;
    double d = rhs[j] / b0;
    cprime[j] = c;
    out[j] = d;
    for (int i = 1; i < n; ++i) {
        const long k = (long)i * m + j;
        const double l = lower[k - m];
        const double den = diag[k] - l * c;
        if (i < n - 1) c = upper[k] / den;
        d = (rhs[k] - l * d) / den;
        cprime[k] = c;
        out[k] = d;
    }
    double x = d;
    for (int i = n - 2; i >= 0; --i) {
        const long k = (long)i * m + j;
        x = out[k] - cprime[k] * x;
        out[k] = x;
    }
}
"""

_KERNELS: dict = {}


def kernel(fmad: bool = False):
    import cupy as cp
    key = bool(fmad)
    if key not in _KERNELS:
        opts = ("--fmad=true",) if fmad else ("--fmad=false",)
        _KERNELS[key] = cp.RawKernel(_SOURCE, "thomas_columns", options=opts)
    return _KERNELS[key]


class Workspace:
    """Reusable device workspace keyed by shape (no per-call allocation)."""

    def __init__(self):
        self._buf: dict = {}

    def get(self, shape):
        import cupy as cp
        if shape not in self._buf:
            self._buf[shape] = cp.empty(shape, dtype=cp.float64)
        return self._buf[shape]


def make_solver(fmad: bool = False, block: int = 128, workspace: Workspace | None = None):
    """A drop-in for ``substrate._tridiagonal(lower, diag, upper, rhs, xp)``
    on CuPy arrays; NumPy inputs go to the reference implementation."""
    import cupy as cp
    from tropoi.physics import substrate as _sub
    reference = _sub._tridiagonal
    k = kernel(fmad)
    ws = workspace or Workspace()

    def solve(lower, diag, upper, rhs, xp):
        if xp is not cp:
            return reference(lower, diag, upper, rhs, xp)
        n, m = diag.shape
        lower = cp.ascontiguousarray(lower, dtype=cp.float64)
        upper = cp.ascontiguousarray(upper, dtype=cp.float64)
        diag = cp.ascontiguousarray(diag, dtype=cp.float64)
        rhs = cp.ascontiguousarray(rhs, dtype=cp.float64)
        out = cp.empty((n, m), dtype=cp.float64)
        k(((m + block - 1) // block,), (block,),
          (lower, diag, upper, rhs, ws.get((n, m)), out, cp.int32(n), cp.int32(m)))
        return out

    solve.reference = reference
    return solve
