// Conductive substrate: pointwise physical terms of one column (the C++ form
// of ConductiveSubstrate._system's physics, same operation order). No time
// discretization and no solver here: see temporal/backward_euler.h and
// numerics/newton.h. Feasibility prototype, NOT production code.
#pragma once
#include "../portable.h"
#include "materials.h"

namespace pt {

//: internal enthalpy/anomaly reference (ENERGY_REFERENCE_TEMPERATURE)
constexpr double SUBSTRATE_TREF = 300.0;

template <int L_, class Law0, class Law1>
struct Substrate {
    static constexpr int L = L_;
    const double *rho, *kc, *chi, *dz, *half;      // per cell (L)
    int segment_end;                               // cells [0, segment_end) use law0
    Law0 law0;
    Law1 law1;
    double two_over_dz1, neg_two_over_dz1;         // 2/dz1 and -(2/dz1), host-computed
    double eps_sigma, four_eps_sigma;              // (eps sigma), (4 eps sigma), host-computed
    double tol_floor, tol_roundoff;                // frozen G-B tolerance (W m^-2; multiple of eps)

    // -- constitutive terms per cell ------------------------------------------------
    PT_HD real enthalpy_difference(int i, real t1, real t0, real d) const {
        return i < segment_end ? law0.enthalpy_difference(t1, t0, d)
                               : law1.enthalpy_difference(t1, t0, d);
    }
    PT_HD real specific_heat(int i, real T) const {
        return i < segment_end ? law0.specific_heat(T) : law1.specific_heat(T);
    }
    PT_HD real k(int i, real T) const { return conductivity(kc[i], chi[i], T); }
    PT_HD real dk(int i, real T) const { return conductivity_derivative(kc[i], chi[i], T); }
    PT_HD real heat_capacity(int i) const { return real(rho[i]) * dz[i]; }   // C_i = rho_i dz_i

    // -- fluxes (positive downward) ---------------------------------------------------
    PT_HD real face_conductance(int i, real k_a, real k_b) const {
        return real(1.0) / (real(half[i]) / k_a + real(half[i + 1]) / k_b);
    }
    PT_HD real face_flux(real g, real th_a, real th_b) const { return g * (th_a - th_b); }
    //: Kirchhoff-exact skin-to-cell-1 flux q_s
    PT_HD real skin_flux(real Ts, real T1, real ths, real th1) const {
        return real(two_over_dz1) * kirchhoff_difference(kc[0], chi[0], Ts, T1, ths - th1);
    }

    // -- skin energy balance f(T_s) - q_s, with its tolerance -------------------------
    PT_HD real emitted(real Ts) const { return real(eps_sigma) * pow(Ts, 4.0); }
    PT_HD real cell_tolerance(real storage, real f_in, real f_out, real source) const {
        return real(tol_floor) + real(tol_roundoff) * (abs(storage) + abs(f_in) + abs(f_out)
                                                       + abs(source));
    }
    PT_HD real skin_tolerance(real absorbed_const, real lin, real emit, real q_s) const {
        return real(tol_floor) + real(tol_roundoff) * (abs(absorbed_const) + abs(lin) + emit
                                                       + abs(q_s));
    }

    // -- partial derivatives ------------------------------------------------------------
    PT_HD real dq_dts(real Ts) const { return real(two_over_dz1) * k(0, Ts); }
    PT_HD real dq_dt1(real k1) const { return real(neg_two_over_dz1) * k1; }
    PT_HD real demitted_dts(real Ts) const { return real(four_eps_sigma) * pow(Ts, 3.0); }
    // d(face flux)/d(upper cell) and /d(lower cell)
    PT_HD real dface_upper(real g, real drift, int i, real dk_a, real k_a) const {
        return g + drift * (g * g) * half[i] * dk_a / (k_a * k_a);
    }
    PT_HD real dface_lower(real g, real drift, int i, real dk_b, real k_b) const {
        return -g + drift * (g * g) * half[i + 1] * dk_b / (k_b * k_b);
    }

    // -- admissibility (physics, not the solver): never more than halve an
    //    absolute temperature in one update; x and dx are anomalies -----------------
    template <int N>
    PT_HD real admissible_fraction(const real* x, const real* dx) const {
        real m(PT_INF);
        for (int i = 0; i < N; ++i) {
            const real s = (dx[i] < 0.0) ? (real(-0.5) * (real(SUBSTRATE_TREF) + x[i])) / dx[i]
                                         : real(PT_INF);
            m = (i == 0) ? s : (s.v < m.v ? s : m);
        }
        return m;
    }
};

}  // namespace pt
