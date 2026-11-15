"""A fused CUDA kernel for the Monin-Obukhov surface-layer solve.

Why a kernel at all
-------------------
The vectorized implementation in :mod:`tropoi.physics.monin_obukhov` runs
unchanged on CuPy, so the GPU is already served. What it cannot avoid is
the shape of the algorithm:

* the Newton loop is a *sequential* outer loop, each pass launching on
  the order of twenty-five elementwise kernels (four similarity functions,
  three profile functions, the derivative and the correction). Measured, it
  needs a median of two passes and at most six - nowhere near the 20-step
  limit - so a surface-layer solve is of the order of 150 kernel launches;
* the loop's exit test is a reduction over the columns
  (``active.any()``), which must be read on the host. That is one
  device-to-host synchronization per pass - half a dozen per solve - and a
  synchronization drains the pipeline: the cost is not the few bytes
  transferred but everything that cannot be queued behind it.

On the grids this model runs (T21 is 2048 product-grid points) the solve
is nowhere near arithmetic-bound: every column is a handful of flops and
there are only a couple of thousand of them. The time is launch overhead
and those twenty synchronizations. Fusing the whole solve into one kernel
with one thread per column replaces all of those launches and all of those
synchronizations with one launch and one.

Why it is correct to fuse
-------------------------
Because the columns do not interact. A column leaves the iteration the
moment its own correction passes the tolerance and its ``zeta`` is frozen
from then on, so Isca's *global* loop produces exactly the trajectory a
per-column loop produces; the global exit test only decides how many
times the loop spins, not what any column ends up with. That is what the
note in :func:`~tropoi.physics.monin_obukhov.solve_zeta` establishes, and
it is what this kernel relies on.

Numerical agreement
-------------------
Not bit-identical to the NumPy path, and it cannot be: ``log``, ``atan``
and ``pow`` are correctly rounded to within an ulp or two by both CUDA's
libdevice and the host libm, but not to the *same* ulp. The iteration is
self-correcting about this - it converges to the root of the same
equation - but its stopping rule is a relative tolerance of 1e-4, so in
principle a column could stop one iteration apart in the two
implementations. The measured agreement on the test battery is reported
in ``docs/radiation/MONIN_OBUKHOV.md``; the tests assert it.

Because of that, this path is **opt-in** and records itself in the run
signature. A run that used the kernel says so, so two runs are never
silently compared across the two implementations.

Requires CuPy and a CUDA device; importing this module does not.
"""
from __future__ import annotations

from typing import Optional

from tropoi.physics.monin_obukhov import (ISCA_ERROR, ISCA_MAX_ITER, ISCA_SMALL,
                                          ISCA_ZETA_MIN, VON_KARMAN,
                                          MoninObukhovOptions, SurfaceExchange)

__all__ = ["surface_drag_fused", "kernel_available", "clear_cache"]

_KERNEL_SOURCE = r"""
extern "C" {

__device__ __forceinline__ double phi_stable(double zeta, int stable_option,
                                             double b_stab, double zeta_trans,
                                             double lambda_stab)
{
    if (stable_option == 1) {
        return 1.0 + zeta * (5.0 + b_stab * zeta) / (1.0 + zeta);
    }
    return (zeta < zeta_trans) ? (1.0 + 5.0 * zeta) : (lambda_stab + b_stab * zeta);
}

__device__ __forceinline__ double phi_t_of(double zeta, int stable_option,
                                           double b_stab, double zeta_trans,
                                           double lambda_stab)
{
    if (zeta < 0.0) return pow(1.0 - 16.0 * zeta, -0.5);
    return phi_stable(zeta, stable_option, b_stab, zeta_trans, lambda_stab);
}

__device__ __forceinline__ double phi_m_of(double zeta, int stable_option,
                                           double b_stab, double zeta_trans,
                                           double lambda_stab)
{
    if (zeta < 0.0) return sqrt(pow(1.0 - 16.0 * zeta, -0.5));
    return phi_stable(zeta, stable_option, b_stab, zeta_trans, lambda_stab);
}

/* integral similarity function for momentum (Isca monin_obukhov_integral_m) */
__device__ __forceinline__ double psi_m_of(double zeta, double zeta_0, double ln_z_z0,
                                           int stable_option, double b_stab,
                                           double zeta_trans, double lambda_stab)
{
    if (zeta < 0.0) {
        double x   = sqrt(sqrt(1.0 - 16.0 * zeta));
        double x_0 = sqrt(sqrt(1.0 - 16.0 * zeta_0));
        double x1 = 1.0 + x, x1_0 = 1.0 + x_0;
        double num   = x1 * x1 * (1.0 + x * x);
        double denom = x1_0 * x1_0 * (1.0 + x_0 * x_0);
        double y = atan(x) - atan(x_0);
        return ln_z_z0 - log(num / denom) + 2 * y;
    }
    if (stable_option == 1) {
        return ln_z_z0 + (5.0 - b_stab) * log((1.0 + zeta) / (1.0 + zeta_0))
               + b_stab * (zeta - zeta_0);
    }
    if (zeta <= zeta_trans) return ln_z_z0 + 5.0 * (zeta - zeta_0);
    double bend = (lambda_stab - 1.0) * log(zeta / zeta_trans)
                  + b_stab * (zeta - zeta_trans);
    if (zeta_0 <= zeta_trans) return ln_z_z0 + bend + 5.0 * (zeta_trans - zeta_0);
    return lambda_stab * ln_z_z0 + b_stab * (zeta - zeta_0);
}

/* integral similarity function for heat and tracers (monin_obukhov_integral_tq) */
__device__ __forceinline__ double psi_t_of(double zeta, double zeta_t, double ln_z_zt,
                                           int stable_option, double b_stab,
                                           double zeta_trans, double lambda_stab)
{
    if (zeta < 0.0) {
        double x   = sqrt(1.0 - 16.0 * zeta);
        double x_t = sqrt(1.0 - 16.0 * zeta_t);
        return ln_z_zt - 2.0 * log((1.0 + x) / (1.0 + x_t));
    }
    if (stable_option == 1) {
        return ln_z_zt + (5.0 - b_stab) * log((1.0 + zeta) / (1.0 + zeta_t))
               + b_stab * (zeta - zeta_t);
    }
    if (zeta <= zeta_trans) return ln_z_zt + 5.0 * (zeta - zeta_t);
    double bend = (lambda_stab - 1.0) * log(zeta / zeta_trans)
                  + b_stab * (zeta - zeta_trans);
    if (zeta_t <= zeta_trans) return ln_z_zt + bend + 5.0 * (zeta_trans - zeta_t);
    return lambda_stab * ln_z_zt + b_stab * (zeta - zeta_t);
}

__global__ void monin_obukhov_drag(
    const double* __restrict__ pt,     const double* __restrict__ pt0,
    const double* __restrict__ zin,    const double* __restrict__ z0,
    const double* __restrict__ zt,     const double* __restrict__ zq,
    const double* __restrict__ speed,
    const double grav,     const double vonkarm,
    const double rich_crit, const double drag_min,
    const double zeta_trans, const int stable_option,
    const double tolerance, const double zeta_min, const int max_iter,
    const double small,    const int n,
    double* __restrict__ drag_m, double* __restrict__ drag_t,
    double* __restrict__ drag_q, double* __restrict__ u_star,
    double* __restrict__ b_star, double* __restrict__ richardson,
    double* __restrict__ zeta_out,
    signed char* __restrict__ converged, signed char* __restrict__ capped)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double b_stab = 1.0 / rich_crit;
    const double lambda_stab = 1.0 + (5.0 - b_stab) * zeta_trans;
    const double r_crit = 0.95 * rich_crit;
    const double sqrt_drag_min = (drag_min != 0.0) ? sqrt(drag_min) : 0.0;

    const double s = speed[i];
    const double delta_b = grav * (pt0[i] - pt[i]) / pt0[i];
    const double rich = -zin[i] * delta_b / (s * s + small);
    richardson[i] = rich;

    if (rich >= r_crit) {                      /* Isca backs off near Ri_c */
        drag_m[i] = drag_min; drag_t[i] = drag_min; drag_q[i] = drag_min;
        u_star[i] = sqrt_drag_min * s;
        b_star[i] = sqrt_drag_min * delta_b;
        /* nothing was iterated, so there is no failure to report; `capped`
           is what tells the caller this column took drag_min */
        converged[i] = 1; capped[i] = 1;
        zeta_out[i] = 0.0;
        return;
    }
    capped[i] = 0;

    /* the profile height is max(z, z0, zt, zq), as in the Fortran */
    double zz = zin[i];
    if (z0[i] > zz) zz = z0[i];
    if (zt[i] > zz) zz = zt[i];
    if (zq[i] > zz) zz = zq[i];

    const double z_z0 = zz / z0[i], z_zt = zz / zt[i], z_zq = zz / zq[i];
    const double ln_z_z0 = log(z_z0), ln_z_zt = log(z_zt), ln_z_zq = log(z_zq);

    double zeta = rich * ln_z_z0 * ln_z_z0 / ln_z_zt;
    if (rich >= 0.0) zeta = zeta / (1.0 - rich / rich_crit);

    double f_m = ln_z_z0, f_t = ln_z_zt, f_q = ln_z_zq;
    int done = 0;

    for (int it = 0; it < max_iter; ++it) {
        if (fabs(zeta) < zeta_min) {           /* taken as neutral */
            zeta = 0.0;
            f_m = ln_z_z0; f_t = ln_z_zt; f_q = ln_z_zq;
            done = 1;
            break;
        }
        const double rzeta = 1.0 / zeta;
        const double zeta_0 = zeta / z_z0;
        const double zeta_t = zeta / z_zt;
        const double zeta_q = zeta / z_zq;

        const double phi_m   = phi_m_of(zeta,   stable_option, b_stab, zeta_trans, lambda_stab);
        const double phi_m_0 = phi_m_of(zeta_0, stable_option, b_stab, zeta_trans, lambda_stab);
        const double phi_t   = phi_t_of(zeta,   stable_option, b_stab, zeta_trans, lambda_stab);
        const double phi_t_0 = phi_t_of(zeta_t, stable_option, b_stab, zeta_trans, lambda_stab);

        f_m = psi_m_of(zeta, zeta_0, ln_z_z0, stable_option, b_stab, zeta_trans, lambda_stab);
        f_t = psi_t_of(zeta, zeta_t, ln_z_zt, stable_option, b_stab, zeta_trans, lambda_stab);
        f_q = psi_t_of(zeta, zeta_q, ln_z_zq, stable_option, b_stab, zeta_trans, lambda_stab);

        const double df_m = (phi_m - phi_m_0) * rzeta;
        const double df_t = (phi_t - phi_t_0) * rzeta;
        const double rich_1 = zeta * f_t / (f_m * f_m);
        const double d_rich = rich_1 * (rzeta + df_t / f_t - 2.0 * df_m / f_m);
        const double correction = (rich - rich_1) / d_rich;
        const double a = fabs(correction);
        const double b = fabs(correction / zeta);
        const double measure = (a < b) ? a : b;
        if (measure <= tolerance) { done = 1; break; }
        zeta = zeta + correction;
    }

    converged[i] = (signed char) done;
    zeta_out[i] = zeta;

    double us = vonkarm / f_m; if (us < sqrt_drag_min) us = sqrt_drag_min;
    double bs = vonkarm / f_t; if (bs < sqrt_drag_min) bs = sqrt_drag_min;
    double qs = vonkarm / f_q; if (qs < sqrt_drag_min) qs = sqrt_drag_min;
    drag_m[i] = us * us;
    drag_t[i] = us * bs;
    drag_q[i] = us * qs;
    u_star[i] = us * s;
    b_star[i] = bs * delta_b;
}

}  /* extern "C" */
"""

_cache: dict = {}


def kernel_available() -> bool:
    """True when CuPy and a CUDA device are present and the kernel compiles."""
    try:
        _compiled()
        return True
    except Exception:
        return False


def clear_cache() -> None:
    _cache.clear()


def _compiled():
    kernel = _cache.get("kernel")
    if kernel is None:
        import cupy
        # No --use_fast_math: it is off by default and must stay off. It
        # would swap log/atan/pow and the divisions for fast approximations,
        # which is exactly the agreement this path has to keep.
        #
        # --fmad, by contrast, is left at its default (on). It is the one
        # measured departure from the NumPy path on Isca's reference case:
        # `s*s + small` contracts to fma(s, s, small), rounding the
        # Richardson denominator once instead of twice, worth 2 ulp in that
        # diagnostic and nothing in the coefficients. Disabling it would
        # spend accuracy in the iteration to buy agreement in a diagnostic.
        # tests/test_monin_obukhov.py asserts that contraction exactly.
        kernel = cupy.RawKernel(_KERNEL_SOURCE, "monin_obukhov_drag")
        _cache["kernel"] = kernel
    return kernel


def surface_drag_fused(potential_temperature, surface_temperature, height, speed, *,
                       gravity: float, roughness_momentum, roughness_heat,
                       roughness_moisture=None,
                       opts: Optional[MoninObukhovOptions] = None,
                       small: float = ISCA_SMALL,
                       max_iter: int = ISCA_MAX_ITER,
                       tolerance: float = ISCA_ERROR,
                       zeta_min: float = ISCA_ZETA_MIN,
                       block: int = 128) -> SurfaceExchange:
    """One fused CUDA launch for the whole Monin-Obukhov surface solve.

    Same arguments and same return type as
    :func:`tropoi.physics.monin_obukhov.surface_drag`, for CuPy arrays.
    ``opts.neutral`` is not handled here, because the neutral branch needs
    no iteration and the vectorized path costs nothing for it.
    """
    import cupy

    opts = MoninObukhovOptions() if opts is None else opts
    if opts.neutral:
        raise ValueError("the fused kernel is for the iterated solve; the neutral "
                         "branch has no iteration, so use the vectorized path")
    pt = cupy.ascontiguousarray(potential_temperature, dtype=cupy.float64).ravel()
    n = pt.size
    pt0 = cupy.ascontiguousarray(cupy.broadcast_to(
        cupy.asarray(surface_temperature, dtype=cupy.float64), pt.shape)).ravel()
    z = cupy.ascontiguousarray(cupy.broadcast_to(
        cupy.asarray(height, dtype=cupy.float64), pt.shape)).ravel()
    spd = cupy.ascontiguousarray(cupy.broadcast_to(
        cupy.asarray(speed, dtype=cupy.float64), pt.shape)).ravel()

    def _as_field(value):
        return cupy.ascontiguousarray(cupy.broadcast_to(
            cupy.asarray(value, dtype=cupy.float64), pt.shape)).ravel()

    z0 = _as_field(roughness_momentum)
    zt = _as_field(roughness_heat)
    zq = zt if roughness_moisture is None else _as_field(roughness_moisture)

    out = [cupy.empty(n, dtype=cupy.float64) for _ in range(7)]
    drag_m, drag_t, drag_q, u_star, b_star, richardson, zeta = out
    converged = cupy.zeros(n, dtype=cupy.int8)
    capped = cupy.zeros(n, dtype=cupy.int8)

    grid = (n + block - 1) // block
    _compiled()((grid,), (block,), (
        pt, pt0, z, z0, zt, zq, spd,
        cupy.float64(gravity), cupy.float64(VON_KARMAN),
        cupy.float64(opts.rich_crit), cupy.float64(opts.drag_min),
        cupy.float64(opts.zeta_trans), cupy.int32(opts.stable_option),
        cupy.float64(tolerance), cupy.float64(zeta_min), cupy.int32(max_iter),
        cupy.float64(small), cupy.int32(n),
        drag_m, drag_t, drag_q, u_star, b_star, richardson, zeta, converged, capped))

    shape = potential_temperature.shape
    return SurfaceExchange(
        drag_m=drag_m.reshape(shape), drag_t=drag_t.reshape(shape),
        drag_q=drag_q.reshape(shape), u_star=u_star.reshape(shape),
        b_star=b_star.reshape(shape), richardson=richardson.reshape(shape),
        zeta=zeta.reshape(shape),
        converged=converged.reshape(shape).astype(bool),
        capped=capped.reshape(shape).astype(bool))
