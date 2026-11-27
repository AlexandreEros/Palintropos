// Backward-Euler discretization of the substrate's finite-volume conservation
// law, with its diagnosed massless skin: the residual of
//
//   C_i [h_i(T_i) - h_i(T_i^n)] / dt - (F_{i-1/2} - F_{i+1/2}) - S_i = 0,
//   f(T_s) - q_s = 0,
//
// and its Jacobian, composed from the Physics' pointwise terms. It is the
// Problem handed to pt::newton_solve. As a template it is compiled into the
// same kernel as the physics and the solver: logically separate from both,
// computationally fused. Same operation order as
// tropoi.temporal.substrate.BackwardEulerSubstrate._system (the reference).
#pragma once
#include "pt/portable.h"

namespace pt {

template <class Physics>
struct BackwardEulerColumn {
    static constexpr int CAP = Physics::LAYER_CAP + 1;   // unknowns: skin and cells
    const Physics& ph;
    const real* th_old;                                 // (L) anomalies at t^n
    const double* source;                               // (L) per-cell source S_i, or null
    real dt;
    real absorbed_const, flux_derivative, flux_reference, bottom_flux;

    PT_HD int size() const { return ph.L + 1; }
    PT_HD real src(int i) const { return source ? real(source[i]) : real(0.0); }
    PT_HD static real storage_rate(real cap, real dh, real dt) { return cap * dh / dt; }

    // The converged-evaluation quantities a step reports (all optional).
    struct Report {
        double *storage, *r, *tau;                       // per cell
        double *r_s, *tau_s, *q_s, *emitted, *absorbed;  // skin
    };

    PT_HD real evaluate(const real* x, bool jacobian, real* R, real* lo, real* di, real* up,
                        const Report* out = nullptr) const {
        constexpr int LC = Physics::LAYER_CAP;
        const int L = ph.L;
        const real T_ref(SUBSTRATE_TREF);
        real k[LC], F[LC + 1], cap[LC];
        const real ths = x[0];
        const real Ts = T_ref + ths;
        for (int i = 0; i < L; ++i) {
            k[i] = ph.k(i, T_ref + x[i + 1]);
            cap[i] = ph.heat_capacity(i);
        }
        const real q_s = ph.skin_flux(Ts, T_ref + x[1], ths, x[1]);
        F[0] = q_s;
        for (int i = 0; i < L - 1; ++i)
            F[i + 1] = ph.face_flux(ph.face_conductance(i, k[i], k[i + 1]), x[i + 1], x[i + 2]);
        F[L] = -bottom_flux;
        real ratio(0.0);
        for (int i = 0; i < L; ++i) {
            const real dh = ph.enthalpy_difference(i, T_ref + x[i + 1], T_ref + th_old[i],
                                                   x[i + 1] - th_old[i]);
            const real storage = storage_rate(cap[i], dh, dt);
            const real s = src(i);
            const real r = storage - (F[i] - F[i + 1]) - s;
            const real tau = ph.cell_tolerance(storage, F[i], F[i + 1], s);
            R[i + 1] = r;
            const real q = abs(r) / tau;
            ratio = (i == 0) ? q : maximum(ratio, q);
            if (out) { out->storage[i] = storage.v; out->r[i] = r.v; out->tau[i] = tau.v; }
        }
        const real lin = flux_derivative * (ths - flux_reference);
        const real emit = ph.emitted(Ts);
        const real rs = absorbed_const + lin - emit - q_s;
        const real taus = ph.skin_tolerance(absorbed_const, lin, emit, q_s);
        R[0] = rs;
        ratio = maximum(ratio, abs(rs) / taus);
        if (out) {
            *out->r_s = rs.v; *out->tau_s = taus.v; *out->q_s = q_s.v;
            *out->emitted = emit.v; *out->absorbed = (absorbed_const + lin).v;
        }
        if (jacobian) {
            const real dqs = ph.dq_dts(Ts);
            const real dq1 = ph.dq_dt1(k[0]);
            di[0] = flux_derivative - ph.demitted_dts(Ts) - dqs;
            up[0] = -dq1;
            lo[0] = -dqs;
            real dfb_prev(0.0);
            for (int i = 0; i < L; ++i) {
                const real T = T_ref + x[i + 1];
                real d = cap[i] * ph.specific_heat(i, T) / dt;   // d(storage)/dT: the BE term
                d = d + real(0.0) * T;
                if (i == 0) d = d - dq1;
                if (i < L - 1) {
                    const real g = ph.face_conductance(i, k[i], k[i + 1]);
                    const real drift = x[i + 1] - x[i + 2];
                    const real dfa = ph.dface_upper(g, drift, i, ph.dk(i, T), k[i]);
                    const real dfb = ph.dface_lower(g, drift, i, ph.dk(i + 1, T_ref + x[i + 2]),
                                                    k[i + 1]);
                    d = d + dfa;
                    lo[i + 1] = -dfa;
                    up[i + 1] = dfb;
                    if (i > 0) d = d - dfb_prev;
                    dfb_prev = dfb;
                } else if (i > 0) {
                    d = d - dfb_prev;
                }
                di[i + 1] = d;
            }
        }
        return ratio;
    }

    PT_HD real admissible_fraction(const real* x, const real* dx) const {
        return ph.admissible_fraction(size(), x, dx);
    }
};

}  // namespace pt
