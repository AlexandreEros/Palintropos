// Portability layer of the canonical native implementation (tropoi.numerics).
//
// The same headers compile for the host (any C++14 compiler, no CUDA) and for
// the device (NVRTC through CuPy). This file is the only place that knows the
// difference. Arithmetic contract (audit finding F2):
//
// * every product is one rounded IEEE multiplication: __dmul_rn on the device
//   (never contracted into an FMA, whatever the flags), a plain product on the
//   host, which must be compiled with -ffp-contract=off;
// * the device is compiled with CuPy's default flags, so libdevice pow, exp and
//   expm1 round as they do in CuPy's own ufuncs;
// * sums, differences and quotients are single IEEE operations either way.
#pragma once

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define PT_DEVICE 1
#define PT_HD __host__ __device__ __forceinline__
#define PT_NAN __longlong_as_double(0x7ff8000000000000LL)
#define PT_INF __longlong_as_double(0x7ff0000000000000LL)
#else
#include <math.h>
// No contraction of a*b+c into an FMA on the host either: honoured by Clang,
// and the build passes -ffp-contract=off for GCC, which ignores the pragma.
#pragma STDC FP_CONTRACT OFF
#define PT_HD inline
#define PT_NAN (NAN)
#define PT_INF (INFINITY)
#endif

namespace pt {

struct real {
    double v;
    PT_HD real() : v(0.0) {}
    PT_HD real(double x) : v(x) {}
};

PT_HD real operator*(real a, real b) {
#ifdef PT_DEVICE
    return real(__dmul_rn(a.v, b.v));
#else
    return real(a.v * b.v);
#endif
}
PT_HD real operator+(real a, real b) { return real(a.v + b.v); }
PT_HD real operator-(real a, real b) { return real(a.v - b.v); }
PT_HD real operator/(real a, real b) { return real(a.v / b.v); }
PT_HD real operator-(real a) { return real(-a.v); }
PT_HD bool operator<(real a, real b) { return a.v < b.v; }
PT_HD bool operator<=(real a, real b) { return a.v <= b.v; }
PT_HD bool operator>(real a, real b) { return a.v > b.v; }
PT_HD bool operator>=(real a, real b) { return a.v >= b.v; }

PT_HD bool is_nan(real a) { return isnan(a.v); }
PT_HD bool is_finite(real a) { return isfinite(a.v); }
PT_HD real exp(real a) { return real(::exp(a.v)); }
PT_HD real expm1(real a) { return real(::expm1(a.v)); }
PT_HD real pow(real a, double b) { return real(::pow(a.v, b)); }
PT_HD real abs(real a) { return real(::fabs(a.v)); }
// NumPy/CuPy minimum and maximum: NaN propagates.
PT_HD real minimum(real a, real b) {
    return (is_nan(a) || is_nan(b)) ? real(PT_NAN) : (a.v < b.v ? a : b);
}
PT_HD real maximum(real a, real b) {
    return (is_nan(a) || is_nan(b)) ? real(PT_NAN) : (a.v > b.v ? a : b);
}
PT_HD real where(bool c, real a, real b) { return c ? a : b; }

}  // namespace pt
