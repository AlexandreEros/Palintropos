#include <cupy/complex.cuh>

// Legendre stages of the separable (FFT + Legendre) Gauss-grid transform.
//
// The longitude stage is cuFFT (via cupy.fft.rfft / irfft); these two
// kernels are the latitude stage, which is what the separable factorization
// actually saves: the dense basis-matrix path costs n_points * n_basis per
// field, these cost nlat * n_basis.
//
//   analysis    a[k,l,m] = sum_j  Pw[j,l,m] * (F[k,j,bin(m)] conjugated?)
//   synthesis   H[k,j,m] = sum_l  Ps[j,l,m] * a[k,l,m]
//
// Both coefficient tables are RECTANGULAR (nlat, n, n) with n = l_max + 1
// and zeros wherever m > l. That wastes just under half the multiplies
// relative to the packed l(l+1)/2 + m layout, and buys: a uniform loop
// trip count (no warp divergence), no (l, m) index tables, no separate
// zero-fill of the padded output entries, and a fully coalesced fastest
// index (m) in every array. At the sizes in use (nlat ~ 65, n ~ 45, K ~ 25)
// the stage is launch- and cache-bound, not multiply-bound, so regularity
// is worth more than the factor of two.
//
// The tables are real and the fields complex, so each accumulation step is
// two FMAs rather than a complex multiply.
//
// Attribution: the separable FFT + Gauss-quadrature-Legendre factorization,
// and the orthonormal convention sqrt((2l+1)/(4pi) (l-m)!/(l+m)!) it is
// written in, follow NVIDIA torch-harmonics (BSD-3-Clause); that package's
// own CUDA extensions require compute capability >= 7.0 and do not cover
// this stage. The Legendre values themselves are produced by this
// repository's existing, already-validated sh_matrix.cu recurrence,
// evaluated at longitude zero -- see separable_gauss_sh.py.

extern "C" __global__
void legendre_analysis(const double* __restrict__ Pw,          // (nlat, n, n)
                       const complex<double>* __restrict__ F,  // (K, nlat, nrf)
                       const int* __restrict__ m_bin,          // (n,)
                       const double* __restrict__ m_sign,      // (n,)
                       complex<double>* __restrict__ out,      // (K, n, n)
                       int K, int nlat, int n, int nrf)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= K * n * n) return;

    const int m = idx % n;
    const int l = (idx / n) % n;
    const int k = idx / (n * n);

    // F holds only the rfft half-spectrum. bin(m) is the real bin that
    // carries zonal wavenumber m (m mod nlon, reflected into [0, nlon/2]),
    // and m_sign is -1 where that reflection means a conjugate. Because Pw
    // is real, conjugating every term is the same as negating the final
    // imaginary part, so the sign is applied once at the end.
    const int b = m_bin[m];
    const double s = m_sign[m];

    const double* p = Pw + (size_t)l * n + m;
    const complex<double>* f = F + (size_t)k * nlat * nrf + b;

    const size_t p_stride = (size_t)n * n;
    double ar = 0.0, ai = 0.0;
    for (int j = 0; j < nlat; ++j) {
        const double w = p[(size_t)j * p_stride];
        const complex<double> c = f[(size_t)j * nrf];
        ar += w * c.real();
        ai += w * c.imag();
    }
    out[idx] = complex<double>(ar, s * ai);
}

extern "C" __global__
void legendre_synthesis(const double* __restrict__ Ps,         // (nlat, n, n)
                        const complex<double>* __restrict__ a, // (K, n, n)
                        complex<double>* __restrict__ H,       // (K, nlat, n)
                        int K, int nlat, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= K * nlat * n) return;

    const int m = idx % n;
    const int j = (idx / n) % nlat;
    const int k = idx / (n * nlat);

    const double* p = Ps + (size_t)j * n * n + m;
    const complex<double>* ak = a + (size_t)k * n * n + m;

    const size_t stride = (size_t)n;
    double hr = 0.0, hi = 0.0;
    for (int l = 0; l < n; ++l) {
        const double w = p[(size_t)l * stride];
        const complex<double> c = ak[(size_t)l * stride];
        hr += w * c.real();
        hi += w * c.imag();
    }

    // A real field has real m = 0 coefficients; the dense reference
    // synthesis discards Im(a_l0) by taking 2 Re(Y a) - Re(Y a_m0). Zeroing
    // the DC imaginary part here reproduces that exactly and makes the
    // result independent of how irfft chooses to treat a non-real DC bin.
    H[idx] = complex<double>(hr, m == 0 ? 0.0 : hi);
}
