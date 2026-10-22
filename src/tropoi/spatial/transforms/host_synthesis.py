"""Host (NumPy) synthesis of spectral fields on a latitude-longitude mesh.

The same basis every backend synthesizes with (``PointSetSphericalHarmonics``,
``sh_matrix.cu``): complex orthonormal spherical harmonics with the
Condon-Shortley phase, ``Y_lm = N_lm P_lm(sin lat) e^{i m lon}``, stored for
``m >= 0`` (axis 0 = degree, axis 1 = order) with real fields implied, so

    f(lat, lon) = sum_m w_m Re( e^{i m lon} sum_l Pbar_lm(sin lat) a_lm ),
    w_0 = 1, w_m = 2.

This is an exact evaluation of the band-limited field, not an
interpolation: previews built from it show precisely the function a model
holds. It exists so that saved coefficients (a resolved terrain, a saved
initial state) can be drawn without CUDA; GPU parity with the backends'
basis is asserted by the test suite.

NumPy only; never imports CuPy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["synthesize_latlon"]


def _order_sums(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``G[i, m] = sum_l Pbar_lm(x_i) a_lm`` by the normalized recursion."""
    size = coeffs.shape[0]
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    out = np.zeros((x.size, size), dtype=np.complex128)
    pmm = np.full(x.size, 1.0 / math.sqrt(4.0 * math.pi))
    for m in range(size):
        if m > 0:
            pmm = -math.sqrt((2.0 * m + 1.0) / (2.0 * m)) * s * pmm
        acc = pmm * coeffs[m, m]
        if m + 1 < size:
            p_prev, p_curr = pmm, math.sqrt(2.0 * m + 3.0) * x * pmm
            acc = acc + p_curr * coeffs[m + 1, m]
            for l in range(m + 2, size):
                a = math.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
                b = math.sqrt(((l - 1.0) ** 2 - m * m)
                              / (4.0 * (l - 1.0) ** 2 - 1.0))
                p_prev, p_curr = p_curr, a * (x * p_curr - b * p_prev)
                acc = acc + p_curr * coeffs[l, m]
        out[:, m] = acc
    return out


def synthesize_latlon(coeffs, lat_deg, lon_deg) -> np.ndarray:
    """Evaluate ``coeffs`` on the tensor mesh ``lat_deg x lon_deg``.

    ``coeffs`` is a host ``(L+1, L+1)`` complex array in the repository
    layout. Returns float64 values of shape ``(len(lat_deg), len(lon_deg))``.
    """
    a = np.asarray(coeffs)
    if hasattr(coeffs, "get"):
        raise TypeError("synthesize_latlon takes host (NumPy) coefficients")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"coefficients must be a square (L+1, L+1) array, got {a.shape}")
    a = a.astype(np.complex128)
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64).ravel())
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64).ravel())
    sums = _order_sums(a, np.sin(lat))
    orders = np.arange(a.shape[0])
    weight = np.where(orders == 0, 1.0, 2.0)
    phase = np.exp(1j * np.outer(orders, lon))
    return ((sums * weight) @ phase).real
