// Constitutive laws: the C++ form of tropoi.physics.materials, in the same
// operation order (each line cites the Python it reproduces). Coefficients
// are runtime data loaded from a flat array; the law *family* is a
// compile-time type, so a configuration specializes and fuses the kernel
// without generating source. Feasibility prototype, NOT production code.
#pragma once
#include "../portable.h"

namespace pt {

// c_p = factor * sum_j a_j x^j, x = (T - center) / scale   (PolynomialHeatCapacity)
template <int N>
struct Poly {
    static constexpr int nparams = N + 3;
    double c[N];
    double center, scale, factor;
    PT_HD void load(const double* p) {
        for (int j = 0; j < N; ++j) c[j] = p[j];
        center = p[N]; scale = p[N + 1]; factor = p[N + 2];
    }
    PT_HD real specific_heat(real T) const {
        const real x = (T - center) / scale;
        real out = real(c[N - 1]) + real(0.0) * x;       // coefficients[-1] + _like(x)
        for (int j = N - 2; j >= 0; --j) out = out * x + c[j];
        return real(factor) * out;
    }
    PT_HD real enthalpy_difference(real t1, real t0, real delta) const {
        const real x1 = (t1 - center) / scale;
        const real x0 = (t0 - center) / scale;
        real s = real(1.0) + real(0.0) * x1;
        real p0 = real(1.0) + real(0.0) * x0;
        real total = real(c[0]) * s;
        for (int j = 1; j < N; ++j) {
            p0 = p0 * x0;
            s = x1 * s + p0;
            total = total + real(c[j]) * (s / double(j + 1));
        }
        return real(factor) * delta * total;               // factor * d * total
    }
};

// c_p = factor * (a + b [1 - exp(-(T - t_ref)/scale)])   (SaturatingExponentialHeatCapacity)
struct SatExp {
    static constexpr int nparams = 5;
    double a, b, t_ref, scale, factor;
    PT_HD void load(const double* p) { a = p[0]; b = p[1]; t_ref = p[2]; scale = p[3]; factor = p[4]; }
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
    static constexpr int nparams = 1 + Low::nparams + High::nparams;
    double breakpoint;
    Low low;
    High high;
    PT_HD void load(const double* p) {
        breakpoint = p[0];
        low.load(p + 1);
        high.load(p + 1 + Low::nparams);
    }
    PT_HD real specific_heat(real T) const {
        // xp.where(T <= b, low(T), high(T)): both evaluated in Python; the selected one is equal
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

// k = kc [1 + chi T^3] and its Kirchhoff potential   (CellMaterials static methods)
PT_HD real conductivity(real kc, real chi, real T) { return kc * (real(1.0) + chi * T * T * T); }
PT_HD real conductivity_derivative(real kc, real chi, real T) {
    return real(3.0) * kc * chi * T * T;
}
PT_HD real kirchhoff_difference(real kc, real chi, real a, real b, real d) {
    return kc * d * (real(1.0) + real(0.25) * chi * (a + b) * (a * a + b * b));
}

}  // namespace pt
