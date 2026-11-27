// CPU execution driver of the substrate step: the same column step in a loop
// over columns, with run-time law selection (pt::RuntimeLaws) and a fixed
// layer capacity. No CUDA headers or libraries. Built once at installation
// (setuptools Extension tropoi.temporal._substrate_native) and called through
// a plain C ABI with ctypes. Must be compiled with -ffp-contract=off.
#include <Python.h>
#include "column_step.h"

#ifndef PT_BUILD_DIGEST
#error "PT_BUILD_DIGEST must be defined by the build (setup.py)"
#endif
#ifndef PT_BUILD_FLAGS
#define PT_BUILD_FLAGS "unknown"
#endif

namespace {
//: layer capacity of the CPU build; larger columns are rejected before execution
constexpr int HOST_LAYER_CAP = 512;
using HostSubstrate = pt::Substrate<HOST_LAYER_CAP, pt::RuntimeLaws>;

void configure_host(HostSubstrate& ph, const pt::ColumnArrays& a, const pt::StepScalars& s,
                    const double* law_params, int strata) {
    pt::configure(ph, a, s);
    ph.laws.load(law_params, strata);
}
}  // namespace

extern "C" {

//: bumped whenever an exported signature changes
int pt_substrate_abi_version() { return 3; }
const char* pt_substrate_build_digest() { return PT_BUILD_DIGEST; }
const char* pt_substrate_build_flags() { return PT_BUILD_FLAGS; }
int pt_substrate_layer_capacity() { return HOST_LAYER_CAP; }
int pt_substrate_max_strata() { return pt::MAX_STRATA; }

// Advance n columns one step; summary (3 ints) receives the largest iteration
// count, whether any column failed and whether any column's forcing was
// invalid. Returns 0, or -1 if L or the strata count exceeds the build's
// capacity (nothing is computed then).
int pt_substrate_step(const pt::ColumnArrays* a, const pt::StepScalars* s,
                      const double* law_params, int max_iterations, int* summary) {
    if (a->L < 1 || a->L > HOST_LAYER_CAP || a->strata < 1 || a->strata > pt::MAX_STRATA)
        return -1;
    HostSubstrate ph;
    configure_host(ph, *a, *s, law_params, a->strata);
    summary[0] = summary[1] = summary[2] = 0;
    for (int j = 0; j < a->n; ++j) {
        pt::column_step(ph, *a, *s, max_iterations, j);
        if (a->iterations[j] > summary[0]) summary[0] = a->iterations[j];
        if (a->failed[j]) summary[1] = 1;
        if (a->invalid[j]) summary[2] = 1;
    }
    return 0;
}

// Residuals R (L + 1) and Jacobian (lower L, diag L + 1, upper L) of one
// column at the unknowns x (L + 1, anomalies), for verification against finite
// differences, with the skin balance's absorbed constant, a, the reference
// anomaly and G given directly. Column 0 of th_old and source in a is used.
// Returns 0, or -1 if too large.
int pt_substrate_evaluate(const pt::ColumnArrays* a, const pt::StepScalars* s,
                          const double* law_params, double absorbed, double dflux, double fref,
                          double bottom, const double* x_in, double* R, double* lower,
                          double* diag, double* upper) {
    if (a->L < 1 || a->L > HOST_LAYER_CAP || a->strata < 1 || a->strata > pt::MAX_STRATA)
        return -1;
    HostSubstrate ph;
    configure_host(ph, *a, *s, law_params, a->strata);
    const int L = a->L, n = a->n;
    pt::real tho[HOST_LAYER_CAP], x[HOST_LAYER_CAP + 1], r[HOST_LAYER_CAP + 1];
    pt::real lo[HOST_LAYER_CAP + 1], di[HOST_LAYER_CAP + 1], up[HOST_LAYER_CAP + 1];
    double src[HOST_LAYER_CAP];
    for (int i = 0; i < L; ++i) {
        tho[i] = a->th_old[(long long)i * n];
        if (a->source) src[i] = a->source[(long long)i * a->source_row];
    }
    for (int i = 0; i <= L; ++i) x[i] = x_in[i];
    pt::BackwardEulerColumn<HostSubstrate> be{ph, tho, a->source ? src : nullptr, pt::real(s->dt),
                                              pt::real(absorbed), pt::real(dflux),
                                              pt::real(fref), pt::real(bottom)};
    be.residual(x, r);
    be.jacobian(x, lo, di, up);
    for (int i = 0; i <= L; ++i) { R[i] = r[i].v; diag[i] = di[i].v; }
    for (int i = 0; i < L; ++i) { lower[i] = lo[i].v; upper[i] = up[i].v; }
    return 0;
}

// One tridiagonal system of n rows with pt::thomas, for verifying the
// numerics against an independent solver. Returns 0, or -1 if n is out of
// range.
int pt_numerics_thomas(int n, const double* lower, const double* diag, const double* upper,
                       const double* rhs, double* x) {
    constexpr int CAP = HOST_LAYER_CAP + 1;
    if (n < 1 || n > CAP) return -1;
    pt::real lo[CAP], di[CAP], up[CAP], r[CAP], c[CAP], out[CAP];
    for (int i = 0; i < n; ++i) {
        di[i] = diag[i];
        r[i] = rhs[i];
        if (i < n - 1) { lo[i] = lower[i]; up[i] = upper[i]; }
    }
    pt::thomas<CAP>(n, lo, di, up, r, c, out);
    for (int i = 0; i < n; ++i) x[i] = out[i].v;
    return 0;
}

}  // extern "C"

// A minimal module object, so that setuptools can build and install the
// library as an extension; the functions above are reached through ctypes.
static struct PyModuleDef pt_module = {PyModuleDef_HEAD_INIT, "_substrate_native",
                                       "Native CPU driver of the substrate step (C ABI).", -1,
                                       nullptr};
PyMODINIT_FUNC PyInit__substrate_native(void) { return PyModule_Create(&pt_module); }
