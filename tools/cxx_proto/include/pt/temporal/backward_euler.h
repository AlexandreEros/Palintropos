// Backward-Euler discretization of a finite-volume conservation law on one
// column with a diagnosed massless skin: the residual of
//   C_i [h_i(T_i) - h_i(T_i^n)] / dt - (F_{i-1/2} - F_{i+1/2}) - S_i = 0,
//   f(T_s) - q_s = 0,
// and its Jacobian, composed from the Physics' pointwise terms. It is the
// Problem handed to numerics::newton_solve. Being a template, it is compiled
// into the same kernel as the physics and the solver: logically separate,
// computationally fused. Same operation order as ConductiveSubstrate._system.
// Feasibility prototype, NOT production code.
#pragma once
#include "../portable.h"

namespace pt {

template <class Physics>
struct BackwardEulerColumn {
    static constexpr int L = Physics::L;
    static constexpr int N = L + 1;               // x = [skin, cell_1 .. cell_L] (anomalies)
    const Physics& ph;
    const real* th_old;                            // (L) anomalies at t^n
    real dt;
    real absorbed_const, flux_derivative, flux_reference, bottom_flux;

    PT_HD static real storage_rate(real cap, real dh, real dt) { return cap * dh / dt; }

    PT_HD real evaluate(const real* x, bool jacobian, real* R, real* lo, real* di, real* up) const {
        const real T_ref(SUBSTRATE_TREF);
        real k[L], F[L + 1], cap[L];
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
            const real r = storage - (F[i] - F[i + 1]) - 0.0;
            R[i + 1] = r;
            const real q = abs(r) / ph.cell_tolerance(storage, F[i], F[i + 1], real(0.0));
            ratio = (i == 0) ? q : maximum(ratio, q);
        }
        const real lin = flux_derivative * (ths - flux_reference);
        const real emit = ph.emitted(Ts);
        const real rs = absorbed_const + lin - emit - q_s;
        R[0] = rs;
        ratio = maximum(ratio, abs(rs) / ph.skin_tolerance(absorbed_const, lin, emit, q_s));
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
        return ph.template admissible_fraction<N>(x, dx);
    }
};

}  // namespace pt
