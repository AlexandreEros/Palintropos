"""Separable FFT + Legendre spherical-harmonic transform on a Gauss grid.

The dense evaluator (:class:`PointSetSphericalHarmonics`) works on *any*
point set, and pays for that generality: analysis and synthesis are each one
``n_points x n_basis`` complex matrix product. On a Gauss-Legendre x
uniform-longitude grid the basis factorizes,

    Y_lm(lat_j, lon_k) = N_lm P_l^m(x_j) * exp(i m lon_k),

and the longitude factor is a DFT because the longitudes are uniform and
their quadrature weights are equal. So the transform splits into a Fourier
stage along longitude (cuFFT) and a Legendre stage along latitude, and the
latitude stage no longer carries the ``nlon`` factor:

    dense      n_points * n_basis  =  nlat * nlon * n_basis
    separable  nlat * n_basis      +  O(nlat * nlon log nlon)

At T42 that removes a factor of ``nlon = 128`` from the dominant stage. This
is the standard spectral-transform decomposition used by spectral GCMs; see
the attribution note below.

Exactly reproduced conventions
------------------------------
This class is a drop-in alternative to the dense evaluator on this grid
family, not a new basis. It keeps:

* the orthonormal complex harmonics with the Condon-Shortley phase and
  ``N_lm = sqrt((2l+1)/(4pi) (l-m)!/(l+m)!)``, stored for ``m >= 0`` in a
  dense ``(l_max+1, l_max+1)`` array (axis 0 degree, axis 1 order), entries
  with ``m > l`` zero;
* the real-field synthesis ``f = 2 Re(Y a) - Re(Y a_m0)``, which discards
  ``Im(a_l0)``;
* point order: the C-order flattening of ``(nlat, nlon)``, latitude-major.
  Fields are flat ``(n_points,)`` arrays, batches ``(K, n_points)``;
* the Legendre values themselves. They are produced by this repository's
  existing ``cuda/sh_matrix.cu`` recurrence, evaluated at longitude zero --
  the same tabulation trick ``representation/visual/evaluate.py`` uses for
  its host-side view synthesis. No new numerical code evaluates the basis,
  so this backend inherits the dense path's measured accuracy and its
  documented ``m = 151`` overflow limit instead of acquiring its own.

Only the *order of summation* differs from the dense path (Fourier sum over
longitude first, then quadrature sum over latitude, instead of one sum over
all points), so results agree to round-off rather than bit-for-bit. The
dense evaluator remains the numerical reference and the fallback, and stays
the default for production runs.

Applicability
-------------
:func:`separable_is_applicable` is the single gate. It needs a
longitude-uniform grid whose quadrature weights do not depend on longitude,
and ``2 * l_max < nlon`` so that every retained zonal wavenumber has its own
real-FFT bin strictly below the Nyquist bin. Production grids satisfy this
with room to spare (the state grid is sized ``nlon >= 2 l_max + 1`` and the
3/2-rule product grid ``nlon >= 3 l_max + 1``). Grids that do not -- an
under-resolved longitude axis, or the unstructured geodesic grid -- keep the
dense path.

Attribution
-----------
The separable factorization and the normalization convention follow NVIDIA
``torch-harmonics`` (https://github.com/NVIDIA/torch-harmonics,
BSD-3-Clause), whose ``RealSHT``/``InverseRealSHT`` are an FFT in longitude
followed by a Gauss-quadrature Legendre contraction in latitude under the
same ``sqrt((2l+1)/(4pi) (l-m)!/(l+m)!)`` normalization. No code is
vendored: that package's custom CUDA extensions require compute capability
>= 7.0 and do not implement this stage (its transforms use ``torch.fft``
plus an ``einsum``), and PyTorch is deliberately not a dependency here. The
Legendre stage is a CuPy ``RawModule`` (``cuda/legendre_stages.cu``); the
Fourier stage is cuFFT through ``cupy.fft``.
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from tropoi.spatial.transforms.cuda.cuda_utils import raw_module_from_cuda

__all__ = [
    "SeparableGaussSphericalHarmonics",
    "SeparableStructureError",
    "separable_is_applicable",
]

_BLOCK = 256


class SeparableStructureError(ValueError):
    """The requested grid/truncation cannot use the separable transform."""


def separable_is_applicable(nlat: int, nlon: int, l_max: int,
                            weights=None) -> tuple[bool, str]:
    """Whether the separable path applies; ``(ok, reason)``.

    ``reason`` is empty when ``ok``, else states which condition failed --
    callers put it in the warning they emit when falling back to the dense
    evaluator, so a fallback is never silent.
    """
    nlat, nlon, l_max = int(nlat), int(nlon), int(l_max)
    if nlat < 1 or nlon < 1:
        raise SeparableStructureError(
            f"nlat and nlon must be positive, got ({nlat}, {nlon})")
    if 2 * l_max >= nlon:
        return False, (
            f"needs 2*l_max < nlon so every retained zonal wavenumber has a "
            f"real-FFT bin below Nyquist, got l_max={l_max}, nlon={nlon}")
    if weights is not None:
        w = cp.asarray(weights, dtype=cp.float64).reshape(-1)
        if w.size == nlat * nlon:
            w2 = w.reshape(nlat, nlon)
            spread = float(cp.max(cp.abs(w2 - w2[:, :1])))
            scale = float(cp.max(cp.abs(w2))) or 1.0
            if spread / scale > 1e-13:
                return False, (
                    "quadrature weights vary along longitude (relative "
                    f"spread {spread / scale:.2e}); the separable "
                    "factorization requires a longitude-independent weight")
        elif w.size != nlat:
            return False, (
                f"weights have size {w.size}, expected {nlat * nlon} "
                f"(per point) or {nlat} (per latitude row)")
    return True, ""


class SeparableGaussSphericalHarmonics:
    """FFT + Legendre transform on a uniform-longitude Gauss latitude grid.

    Parameters
    ----------
    latitudes : array_like, shape (nlat,)
        Latitudes of the grid rows, in the grid's own row order (radians).
        Row order is preserved exactly; nothing here assumes a direction.
    nlon : int
        Number of uniform longitudes, ``lon_k = 2 pi k / nlon``.
    l_max : int
        Truncation; the coefficient layout is ``(l_max+1, l_max+1)``.
    weights : array_like, optional
        Solid-angle quadrature weights: either per point ``(nlat*nlon,)`` in
        the flattened point order, or per latitude row ``(nlat,)`` giving the
        weight of *one* point in that row. Must not vary along longitude.
        Defaults to the uniform ``4 pi / n_points``, which is correct only
        for an equal-area sampling; it is here for parity with the dense
        evaluator's default, not because it suits this grid.

    Raises
    ------
    SeparableStructureError
        If :func:`separable_is_applicable` rejects the configuration. Use
        that function to choose a backend; this constructor does not fall
        back on its own.
    """

    def __init__(self, latitudes, nlon: int, l_max: int, *, weights=None):
        if not cp.is_available():
            raise ImportError("CuPy is required.")

        lat = cp.ascontiguousarray(
            cp.asarray(latitudes, dtype=cp.float64).ravel())
        self.nlat = int(lat.size)
        self.nlon = int(nlon)
        self.l_max = int(l_max)
        self.n_points = self.nlat * self.nlon
        self.n_lm = self.l_max + 1
        self.n_basis = (self.l_max + 1) * (self.l_max + 2) // 2
        self.n_rfft = self.nlon // 2 + 1

        ok, reason = separable_is_applicable(self.nlat, self.nlon, self.l_max,
                                             weights)
        if not ok:
            raise SeparableStructureError(
                f"separable transform not applicable: {reason}")

        # -- geometry, in the layouts the dense evaluator exposes ----------
        self.latitudes_1d = lat
        self.longitudes_1d = cp.asarray(
            np.linspace(0.0, 2.0 * np.pi, self.nlon, endpoint=False))
        # Per-point lat/lon and weights are kept so that code which rebuilds
        # a dense evaluator from `sh.latitudes` / `sh.longitudes` /
        # `sh.weights` keeps working against this object.
        self.latitudes = cp.repeat(lat, self.nlon)
        self.longitudes = cp.tile(self.longitudes_1d, self.nlat)
        self.grid_shape = (self.n_points,)
        self.phi_grid_gpu = cp.pi / 2.0 - self.latitudes
        self.lambda_grid_gpu = self.longitudes

        if weights is None:
            w_row = cp.full(self.nlat, 4.0 * cp.pi / self.n_points,
                            dtype=cp.float64)
        else:
            w = cp.asarray(weights, dtype=cp.float64).reshape(-1)
            w_row = (w.reshape(self.nlat, self.nlon)[:, 0]
                     if w.size == self.n_points else w)
        self._w_row = cp.ascontiguousarray(w_row)
        self.weights = cp.ascontiguousarray(
            cp.repeat(self._w_row, self.nlon))

        # -- (l, m) index arrays of the packed layout (API parity) ---------
        l_idx, m_idx = [], []
        for l in range(self.n_lm):
            for m in range(l + 1):
                l_idx.append(l)
                m_idx.append(m)
        self.l_indices = cp.asarray(l_idx, dtype=cp.int32)
        self.m_indices = cp.asarray(m_idx, dtype=cp.int32)

        self._build_legendre_tables()
        self._build_fourier_bins()

        self.module = raw_module_from_cuda("legendre_stages")
        self._k_analysis = self.module.get_function("legendre_analysis")
        self._k_synthesis = self.module.get_function("legendre_synthesis")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_legendre_tables(self) -> None:
        """Tabulate ``N_lm P_l^m(x_j)`` with the existing basis kernel.

        ``sh_matrix.cu`` computes ``N_lm P_l^m(x) exp(i m lon)``; evaluated at
        ``lon = 0`` the imaginary part vanishes and the real part is exactly
        the normalized Legendre function the latitude stage needs. Reusing
        that kernel -- rather than writing a second recurrence -- is what
        makes this backend inherit the dense path's numerics instead of
        acquiring its own.

        Two rectangular ``(nlat, n_lm, n_lm)`` tables come out of it, with
        the scale factors folded in so neither transform needs a scaling
        pass. ``_P_ana`` carries the row quadrature weight, because analysis
        is ``a_lm = sum_j w_j N_lm P_l^m(x_j) F_jm``. ``_P_syn`` carries
        ``nlon``, which cancels the ``1/n`` of :func:`cupy.fft.irfft` so the
        inverse FFT output *is* the field, with no post-pass.
        """
        basis = raw_module_from_cuda("sh_matrix").get_function(
            "generate_sph_harm_basis")
        y_row = cp.zeros((self.nlat, self.n_basis), dtype=cp.complex128)
        grid = (self.nlat + _BLOCK - 1) // _BLOCK
        basis((grid,), (_BLOCK,),
              (self.latitudes_1d, cp.zeros(self.nlat, dtype=cp.float64),
               y_row, self.nlat, self.l_max))

        table = cp.zeros((self.nlat, self.n_lm, self.n_lm), dtype=cp.float64)
        table[:, self.l_indices, self.m_indices] = y_row.real
        del y_row

        self._P_syn = cp.ascontiguousarray(table * float(self.nlon))
        self._P_ana = cp.ascontiguousarray(table * self._w_row[:, None, None])
        del table

    def _build_fourier_bins(self) -> None:
        """Map zonal wavenumber ``m`` to a real-FFT bin, with conjugate flag.

        ``rfft`` returns bins ``0 .. nlon//2``. Wavenumber ``m`` lives in bin
        ``m mod nlon``, and a bin above Nyquist is the conjugate of its
        reflection. Under the applicability gate ``2 l_max < nlon`` this is
        always the identity map with sign ``+1``; the general relation is
        written out because it is the exact one and costs nothing.
        """
        bins = np.empty(self.n_lm, dtype=np.int32)
        signs = np.empty(self.n_lm, dtype=np.float64)
        for m in range(self.n_lm):
            b = m % self.nlon
            if b <= self.nlon // 2:
                bins[m], signs[m] = b, 1.0
            else:
                bins[m], signs[m] = self.nlon - b, -1.0
        self._m_bin = cp.asarray(bins)
        self._m_sign = cp.asarray(signs)

    # ------------------------------------------------------------------
    # transforms
    # ------------------------------------------------------------------

    def transform_batch(self, values: cp.ndarray) -> cp.ndarray:
        """Analysis of ``K`` real fields: ``(K, n_points)`` -> ``(K, n, n)``.

        One ``rfft`` along longitude, then one Legendre kernel. The kernel
        writes every entry of the output, ``m > l`` padding included, so no
        zero-fill pass is needed.
        """
        values = cp.asarray(values)
        k = int(values.shape[0]) if values.ndim > 1 else 1
        rows = self._as_rows(values, k)
        spectrum = cp.ascontiguousarray(cp.fft.rfft(rows, axis=-1))
        out = cp.empty((k, self.n_lm, self.n_lm), dtype=cp.complex128)
        total = k * self.n_lm * self.n_lm
        self._k_analysis(((total + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                         (self._P_ana, spectrum, self._m_bin, self._m_sign,
                          out, k, self.nlat, self.n_lm, self.n_rfft))
        return out

    def inv_transform_batch(self, coeffs: cp.ndarray) -> cp.ndarray:
        """Synthesis of ``K`` coefficient grids -> ``(K, n_points)``.

        One Legendre kernel, then one ``irfft``. The kernel emits only the
        ``l_max+1`` retained zonal bins and ``irfft`` zero-pads the rest, so
        nothing wider than the retained spectrum is ever materialized.
        """
        coeffs = cp.ascontiguousarray(
            cp.asarray(coeffs, dtype=cp.complex128))
        if coeffs.ndim != 3 or coeffs.shape[1:] != (self.n_lm, self.n_lm):
            raise ValueError(
                f"coefficients must be (K, {self.n_lm}, {self.n_lm}), got "
                f"{tuple(coeffs.shape)}")
        k = int(coeffs.shape[0])
        half = cp.empty((k, self.nlat, self.n_lm), dtype=cp.complex128)
        total = k * self.nlat * self.n_lm
        self._k_synthesis(((total + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                          (self._P_syn, coeffs, half,
                           k, self.nlat, self.n_lm))
        field = cp.fft.irfft(half, n=self.nlon, axis=-1)
        return field.reshape(k, self.n_points)

    def transform(self, values: cp.ndarray) -> cp.ndarray:
        """Analysis of one field -> ``(l_max+1, l_max+1)`` coefficients."""
        return self.transform_batch(cp.asarray(values).reshape(1, -1))[0]

    def inv_transform(self, coeffs: cp.ndarray) -> cp.ndarray:
        """Synthesis of one coefficient grid -> flat ``(n_points,)`` field."""
        coeffs = cp.asarray(coeffs)
        return self.inv_transform_batch(
            coeffs.reshape(1, self.n_lm, self.n_lm))[0]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _as_rows(self, values: cp.ndarray, k: int) -> cp.ndarray:
        """``(K, ...)`` real field values -> contiguous ``(K, nlat, nlon)``."""
        if values.dtype.kind == "c":
            raise TypeError(
                "separable analysis takes real fields; got complex input. "
                "The dense evaluator accepts complex values, this one uses a "
                "real-to-complex FFT along longitude.")
        return cp.ascontiguousarray(
            values.reshape(k, self.nlat, self.nlon), dtype=cp.float64)

    def extended(self, l_max: int) -> "SeparableGaussSphericalHarmonics":
        """Same grid and weights at a different truncation.

        The vector weak form in ``SpectralOperators`` analyzes on the product
        points against a basis extended by one degree; this is how it gets
        that extended transform without building a second dense basis matrix.
        """
        return type(self)(self.latitudes_1d, self.nlon, l_max,
                          weights=self._w_row)

    def describe(self) -> dict:
        """Numerics provenance for run manifests."""
        return {
            "transform": type(self).__name__,
            "stages": "cufft-rfft-longitude + rawkernel-legendre-latitude",
            "nlat": self.nlat, "nlon": self.nlon, "l_max": self.l_max,
            "legendre_table_bytes": int(self._P_ana.nbytes
                                        + self._P_syn.nbytes),
        }

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(nlat={self.nlat}, nlon={self.nlon}, "
                f"l_max={self.l_max})")
