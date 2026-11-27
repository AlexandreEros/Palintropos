// CUDA execution driver of the substrate step: one thread per column.
// Compiled at run time by NVRTC through CuPy (CuPy's default flags), with the
// template specialization chosen by name expression, for example
//   substrate_step<51, pt::StaticLaws<pt::Piecewise<pt::Poly<5>, pt::SatExp>, ...>>
#include "column_step.h"

template <int LAYER_CAP, class Laws>
__global__ void substrate_step(const int L, const int n, const double* th_old,
                               const double* ths_old, const double* absorbed, const double* dflux,
                               const double* fref, const double* bottom, const double* source,
                               const int has_source, const double* rho, const double* kc,
                               const double* chi, const double* dz, const double* half,
                               const int* stratum, const double* law_params, const double* sd,
                               const int max_iterations, double* th_out, double* ths_out,
                               int* iterations, int* failed, double* ratio, double* rep_storage,
                               double* rep_r, double* rep_tau, double* rep_rs, double* rep_taus,
                               double* rep_qs, double* rep_emitted, double* rep_absorbed) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const pt::StepScalars s{sd[0], sd[1], sd[2], sd[3], sd[4], sd[5], sd[6]};
    const pt::ColumnArrays a{L, n, th_old, ths_old, absorbed, dflux, fref, bottom,
                             has_source ? source : nullptr, rho, kc, chi, dz, half, stratum,
                             th_out, ths_out, iterations, failed, ratio, rep_storage, rep_r,
                             rep_tau, rep_rs, rep_taus, rep_qs, rep_emitted, rep_absorbed};
    pt::Substrate<LAYER_CAP, Laws> ph;
    pt::configure(ph, a, s);
    ph.laws.load(law_params);
    pt::column_step(ph, a, s, max_iterations, j);
}
