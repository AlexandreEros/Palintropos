// Conductive substrate: the pointwise physical terms of one column, the
// canonical native form of the physics in tropoi.physics.substrate and
// tropoi.physics.materials (same operation order). No time discretization and
// no solver here: see pt/temporal/backward_euler.h and pt/numerics/newton.h.
//
// LAYER_CAP is a compile-time capacity; the column has L <= LAYER_CAP layers
// (run time). Laws is a law set: StaticLaws<...> (exact types, GPU) or
// RuntimeLaws (run-time selection, CPU); both evaluate the structs of
// materials.h.
#pragma once
#include "pt/portable.h"
#include "materials.h"

namespace pt {

//: ENERGY_REFERENCE_TEMPERATURE: anomalies are relative to it (internal reference)
constexpr double SUBSTRATE_TREF = 300.0;
//: largest number of strata of the run-time law set
constexpr int MAX_STRATA = 16;

// -- law sets: the heat-capacity law of each stratum ------------------------------

template <class... Ls> struct StaticLaws;
template <class L0> struct StaticLaws<L0> {
    L0 l0;
    PT_HD int load(const double* p) { return Typed<L0>::load(l0, p); }
    PT_HD real enthalpy_difference(int, real t1, real t0, real d) const {
        return l0.enthalpy_difference(t1, t0, d);
    }
    PT_HD real specific_heat(int, real T) const { return l0.specific_heat(T); }
};
template <class L0, class L1, class... Rest> struct StaticLaws<L0, L1, Rest...> {
    L0 l0;
    StaticLaws<L1, Rest...> rest;
    PT_HD int load(const double* p) {
        const int n = Typed<L0>::load(l0, p);
        return n + rest.load(p + n);
    }
    PT_HD real enthalpy_difference(int s, real t1, real t0, real d) const {
        return s == 0 ? l0.enthalpy_difference(t1, t0, d) : rest.enthalpy_difference(s - 1, t1, t0, d);
    }
    PT_HD real specific_heat(int s, real T) const {
        return s == 0 ? l0.specific_heat(T) : rest.specific_heat(s - 1, T);
    }
};

struct RuntimeLaws {
    int count;
    AnyLaw law[MAX_STRATA];
    PT_HD int load(const double* p, int strata) {
        count = strata;
        int k = 0;
        for (int s = 0; s < strata; ++s) k += law[s].load(p + k);
        return k;
    }
    PT_HD real enthalpy_difference(int s, real t1, real t0, real d) const {
        return law[s].enthalpy_difference(t1, t0, d);
    }
    PT_HD real specific_heat(int s, real T) const { return law[s].specific_heat(T); }
};

// -- the column ----------------------------------------------------------------------

template <int LAYER_CAP_, class Laws>
struct Substrate {
    static constexpr int LAYER_CAP = LAYER_CAP_;
    int L;                                         // layers (<= LAYER_CAP)
    const double *rho, *kc, *chi, *dz, *half;      // per cell (L)
    const int* stratum;                            // per cell (L): index into the law set
    Laws laws;
    double two_over_dz1, neg_two_over_dz1;         // 2/dz1 and -(2/dz1), host-computed
    double eps_sigma, four_eps_sigma;              // (eps sigma), (4 eps sigma), host-computed
    double tol_floor, tol_roundoff;                // frozen G-B tolerance (W m^-2; multiple of eps)

    // constitutive terms per cell
    PT_HD real enthalpy_difference(int i, real t1, real t0, real d) const {
        return laws.enthalpy_difference(stratum[i], t1, t0, d);
    }
    PT_HD real specific_heat(int i, real T) const { return laws.specific_heat(stratum[i], T); }
    PT_HD real k(int i, real T) const { return conductivity(kc[i], chi[i], T); }
    PT_HD real dk(int i, real T) const { return conductivity_derivative(kc[i], chi[i], T); }
    PT_HD real heat_capacity(int i) const { return real(rho[i]) * dz[i]; }   // C_i = rho_i dz_i

    // fluxes (positive downward)
    PT_HD real face_conductance(int i, real k_a, real k_b) const {
        return real(1.0) / (real(half[i]) / k_a + real(half[i + 1]) / k_b);
    }
    PT_HD real face_flux(real g, real th_a, real th_b) const { return g * (th_a - th_b); }
    // Kirchhoff-exact skin-to-cell-1 flux q_s
    PT_HD real skin_flux(real Ts, real T1, real ths, real th1) const {
        return real(two_over_dz1) * kirchhoff_difference(kc[0], chi[0], Ts, T1, ths - th1);
    }

    // skin energy balance and residual tolerances
    PT_HD real emitted(real Ts) const { return real(eps_sigma) * pow(Ts, 4.0); }
    PT_HD real cell_tolerance(real storage, real f_in, real f_out, real source) const {
        return real(tol_floor) + real(tol_roundoff) * (abs(storage) + abs(f_in) + abs(f_out)
                                                       + abs(source));
    }
    PT_HD real skin_tolerance(real absorbed_const, real lin, real emit, real q_s) const {
        return real(tol_floor) + real(tol_roundoff) * (abs(absorbed_const) + abs(lin) + emit
                                                       + abs(q_s));
    }

    // partial derivatives
    PT_HD real dq_dts(real Ts) const { return real(two_over_dz1) * k(0, Ts); }
    PT_HD real dq_dt1(real k1) const { return real(neg_two_over_dz1) * k1; }
    PT_HD real demitted_dts(real Ts) const { return real(four_eps_sigma) * pow(Ts, 3.0); }
    PT_HD real dface_upper(real g, real drift, int i, real dk_a, real k_a) const {
        return g + drift * (g * g) * half[i] * dk_a / (k_a * k_a);
    }
    PT_HD real dface_lower(real g, real drift, int i, real dk_b, real k_b) const {
        return -g + drift * (g * g) * half[i + 1] * dk_b / (k_b * k_b);
    }

    // Admissibility of an update (the substrate's, not the solver's): never
    // more than halve an absolute temperature in one update. x and dx hold
    // anomalies of the n unknowns [skin, cells].
    PT_HD real admissible_fraction(int n, const real* x, const real* dx) const {
        real m(PT_INF);
        for (int i = 0; i < n; ++i) {
            const real s = (dx[i] < 0.0) ? (real(-0.5) * (real(SUBSTRATE_TREF) + x[i])) / dx[i]
                                         : real(PT_INF);
            m = (i == 0) ? s : (s.v < m.v ? s : m);
        }
        return m;
    }
};

}  // namespace pt
