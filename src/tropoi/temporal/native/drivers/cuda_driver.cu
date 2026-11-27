// CUDA execution driver of the substrate step: one thread per column.
// Compiled at run time by NVRTC through CuPy (CuPy's default flags), with the
// template specialization chosen by name expression, for example
//   substrate_step<51, pt::StaticLaws<pt::Piecewise<pt::Poly<5>, pt::SatExp>, ...>>
//
// A forcing input is (pointer, stride, value): stride < 0 means the scalar
// value (the pointer is then a placeholder). summary (3 ints, zeroed by the
// caller) receives the largest iteration count, whether any column failed,
// and whether any column's forcing was invalid, so that the host reads one
// small array per step.
#include "column_step.h"

#define PT_INPUT(name) const double *name##_p, const long long name##_stride, const double name##_v
#define PT_COLUMN(name) pt::ColumnInput{name##_stride < 0 ? nullptr : name##_p, name##_stride, name##_v}

template <int LAYER_CAP, class Laws>
__global__ void substrate_step(const int L, const int n, const int strata, const double* th_old,
                               const double* ths_old, PT_INPUT(sw), PT_INPUT(lw), PT_INPUT(f0),
                               PT_INPUT(dflux), PT_INPUT(fref), const int has_fref,
                               PT_INPUT(bottom), const double* source, const int has_source,
                               const long long source_row, const long long source_col,
                               const double* rho, const double* kc, const double* chi,
                               const double* dz, const double* half, const int* stratum,
                               const double* law_params, const double* sd,
                               const int max_iterations, double* th_out, double* ths_out,
                               int* iterations, int* failed, int* invalid, double* ratio,
                               double* rep_storage, double* rep_r, double* rep_tau,
                               double* rep_rs, double* rep_taus, double* rep_qs,
                               double* rep_emitted, double* rep_absorbed, double* extent,
                               int* summary) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const pt::StepScalars s{sd[0], sd[1], sd[2], sd[3], sd[4], sd[5], sd[6], sd[7], sd[8]};
    const pt::ColumnArrays a{L, n, strata, th_old, ths_old, PT_COLUMN(sw), PT_COLUMN(lw),
                             PT_COLUMN(f0), PT_COLUMN(dflux), PT_COLUMN(fref), PT_COLUMN(bottom),
                             has_fref, has_source ? source : nullptr, source_row,
                             source_col, rho, kc, chi, dz, half, stratum, th_out, ths_out,
                             iterations, failed, invalid, ratio, rep_storage, rep_r, rep_tau,
                             rep_rs, rep_taus, rep_qs, rep_emitted, rep_absorbed, extent};
    pt::Substrate<LAYER_CAP, Laws> ph;
    pt::configure(ph, a, s);
    ph.laws.load(law_params);
    pt::column_step(ph, a, s, max_iterations, j);
    atomicMax(&summary[0], iterations[j]);
    if (failed[j]) atomicMax(&summary[1], 1);
    if (invalid[j]) atomicMax(&summary[2], 1);
}
