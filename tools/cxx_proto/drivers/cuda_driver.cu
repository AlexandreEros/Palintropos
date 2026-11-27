// CUDA execution driver: one thread per column. Compiled by NVRTC (CuPy
// RawModule) with name_expressions selecting the template specialization.
// Scalars arrive as a small device array (sd: dt, 2/dz1, -(2/dz1), eps sigma,
// 4 eps sigma, tolerance floor, tolerance round-off multiple).
#include "column_step.h"

template <int L, class Law0, class Law1>
__global__ void substrate_step(const double* th_old, const double* ths_old, const double* absorbed,
                               const double* dflux, const double* tref, const double* bottom,
                               const double* rho, const double* kc, const double* chi,
                               const double* dz, const double* half, const double* law_params,
                               const double* sd, const int segment_end, const int max_iterations,
                               const int n, double* th_out, double* ths_out, int* iters, int* fail,
                               double* ratio_out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const pt::StepScalars s{sd[0], sd[1], sd[2], sd[3], sd[4], sd[5], sd[6], segment_end,
                            max_iterations};
    pt::column_step<L, Law0, Law1>(j, n, th_old, ths_old, absorbed, dflux, tref, bottom, rho, kc,
                                   chi, dz, half, law_params, s, th_out, ths_out, iters, fail,
                                   ratio_out);
}
