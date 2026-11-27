// Constitutive laws of the conductive substrate: the canonical native form of
// tropoi.physics.materials (heat-capacity laws, conductivity, Kirchhoff
// potential), each in the operation order of its Python counterpart.
//
// Coefficients are run-time data loaded from a flat array (layout in
// tropoi.temporal.substrate_native.law_parameters). Law *families* are types:
// the GPU instantiates exactly the configured families (compile-time
// specialization); the CPU uses AnyLaw, a run-time choice among the same
// structs, so both execute the same function bodies.
#pragma once
#include "pt/portable.h"

namespace pt {

//: largest polynomial degree + 1 of a heat-capacity law
constexpr int MAX_POLY = 8;

// c_p = factor * sum_j a_j x^j, x = (T - center) / scale   (PolynomialHeatCapacity)
template <int CAP = MAX_POLY>
struct Poly {
    int n;
    double c[CAP];
    double center, scale, factor;
    PT_HD int load(const double* p) {          // p = [n, a_0..a_{n-1}, center, scale, factor]
        n = int(p[0]);
        for (int j = 0; j < n; ++j) c[j] = p[1 + j];
        center = p[1 + n]; scale = p[2 + n]; factor = p[3 + n];
        return 4 + n;
    }
    PT_HD real specific_heat(real T) const {
        const real x = (T - center) / scale;
        real out = real(c[n - 1]) + real(0.0) * x;       // coefficients[-1] + _like(x)
        for (int j = n - 2; j >= 0; --j) out = out * x + c[j];
        return real(factor) * out;
    }
    PT_HD real enthalpy_difference(real t1, real t0, real delta) const {
        const real x1 = (t1 - center) / scale;
        const real x0 = (t0 - center) / scale;
        real s = real(1.0) + real(0.0) * x1;
        real p0 = real(1.0) + real(0.0) * x0;
        real total = real(c[0]) * s;
        for (int j = 1; j < n; ++j) {
            p0 = p0 * x0;
            s = x1 * s + p0;
            total = total + real(c[j]) * (s / double(j + 1));
        }
        return real(factor) * delta * total;               // factor * d * total
    }
};

// c_p = factor * (a + b [1 - exp(-(T - t_ref)/scale)])   (SaturatingExponentialHeatCapacity)
struct SatExp {
    double a, b, t_ref, scale, factor;
    PT_HD int load(const double* p) {          // p = [a, b, t_ref, scale, factor]
        a = p[0]; b = p[1]; t_ref = p[2]; scale = p[3]; factor = p[4];
        return 5;
    }
    PT_HD real specific_heat(real T) const {
        const real e = exp(-(T - t_ref) / scale);
        return real(factor) * (real(a) + real(b) * (real(1.0) - e));
    }
    PT_HD real enthalpy_difference(real t1, real t0, real delta) const {
        const real u0 = (t0 - t_ref) / scale;
        const real tail = exp(-u0) * expm1(-delta / scale);
        // (a + b) and b * scale are Python float operations on the host
        return real(factor) * (real(a + b) * delta + real(b * scale) * tail);
    }
};

// low for T <= breakpoint, high above; straddling differences are split (PiecewiseHeatCapacity)
template <class Low, class High>
struct Piecewise {
    double breakpoint;
    Low low;
    High high;
    PT_HD int load(const double* p) {          // p = [breakpoint, low..., high...]
        breakpoint = p[0];
        const int nl = low.load(p + 1);
        return 1 + nl + high.load(p + 1 + nl);
    }
    PT_HD real specific_heat(real T) const {
        // xp.where(T <= b, low(T), high(T)): both are evaluated in Python; the selected one is equal
        return where(T <= breakpoint, low.specific_heat(T), high.specific_heat(T));
    }
    PT_HD real enthalpy_difference(real t1, real t0, real delta) const {
        const real b(breakpoint);
        const real lo1 = minimum(t1, b), lo0 = minimum(t0, b);
        const real hi1 = maximum(t1, b), hi0 = maximum(t0, b);
        const bool below = (t1 <= b) & (t0 <= b);
        const bool above = (t1 >= b) & (t0 >= b);
        const real d_lo = where(below, delta, lo1 - lo0);
        const real d_hi = where(above, delta, hi1 - hi0);
        return low.enthalpy_difference(lo1, lo0, d_lo) + high.enthalpy_difference(hi1, hi0, d_hi);
    }
};

// -- run-time selection (CPU) over the same structs ---------------------------------

//: law kind codes of the flat parameter layout
enum LawKind { LAW_POLY = 0, LAW_SATEXP = 1, LAW_PIECEWISE = 2 };

struct SimpleLaw {
    int kind;
    Poly<> poly;
    SatExp sat;
    PT_HD int load(const double* p) {          // p = [kind, params...]
        kind = int(p[0]);
        return 1 + (kind == LAW_POLY ? poly.load(p + 1) : sat.load(p + 1));
    }
    PT_HD real specific_heat(real T) const {
        return kind == LAW_POLY ? poly.specific_heat(T) : sat.specific_heat(T);
    }
    PT_HD real enthalpy_difference(real t1, real t0, real d) const {
        return kind == LAW_POLY ? poly.enthalpy_difference(t1, t0, d)
                                : sat.enthalpy_difference(t1, t0, d);
    }
};

struct AnyLaw {
    bool piecewise;
    SimpleLaw simple;
    Piecewise<SimpleLaw, SimpleLaw> pw;
    PT_HD int load(const double* p) {          // p = [kind, ...]; piecewise: [2, b, low, high]
        piecewise = int(p[0]) == LAW_PIECEWISE;
        return piecewise ? 1 + pw.load(p + 1) : simple.load(p);
    }
    PT_HD real specific_heat(real T) const {
        return piecewise ? pw.specific_heat(T) : simple.specific_heat(T);
    }
    PT_HD real enthalpy_difference(real t1, real t0, real d) const {
        return piecewise ? pw.enthalpy_difference(t1, t0, d) : simple.enthalpy_difference(t1, t0, d);
    }
};

// Compile-time law types read the same flat layout, skipping the kind codes.
template <class L> struct Typed;
template <int CAP> struct Typed<Poly<CAP>> {
    static PT_HD int load(Poly<CAP>& l, const double* p) { return 1 + l.load(p + 1); }
};
template <> struct Typed<SatExp> {
    static PT_HD int load(SatExp& l, const double* p) { return 1 + l.load(p + 1); }
};
template <class Lo, class Hi> struct Typed<Piecewise<Lo, Hi>> {
    static PT_HD int load(Piecewise<Lo, Hi>& l, const double* p) {
        l.breakpoint = p[1];
        const int nl = Typed<Lo>::load(l.low, p + 2);
        return 2 + nl + Typed<Hi>::load(l.high, p + 2 + nl);
    }
};

// k = kc [1 + chi T^3] and its Kirchhoff potential   (CellMaterials static methods)
PT_HD real conductivity(real kc, real chi, real T) { return kc * (real(1.0) + chi * T * T * T); }
PT_HD real conductivity_derivative(real kc, real chi, real T) {
    return real(3.0) * kc * chi * T * T;
}
PT_HD real kirchhoff_difference(real kc, real chi, real a, real b, real d) {
    return kc * d * (real(1.0) + real(0.25) * chi * (a + b) * (a * a + b * b));
}

}  // namespace pt
