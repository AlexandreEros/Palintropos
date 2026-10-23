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

__all__ = ["sin_theta_d_theta", "synthesize_latlon", "synthesize_wind_latlon"]

#: Distance (degrees) from a pole at which the wind's 1/cos(lat) metric
#: factor is evaluated for nodes ON a pole (see :func:`synthesize_wind_latlon`).
POLE_OFFSET_DEG = 1.0e-5


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


def sin_theta_d_theta(coeffs) -> np.ndarray:
    """Coefficients of ``sin(theta) d/dtheta`` (theta = colatitude), exactly.

    The same three-term coupling as the cores' spectral operator
    (``SpectralOperators.sin_theta_d_theta_coeffs``):
    ``sin(theta) dY_l^m/dtheta = C+_lm Y_{l+1}^m + C-_lm Y_{l-1}^m`` with
    ``C+_lm = l sqrt(((l+1)^2 - m^2) / ((2l+1)(2l+3)))`` and
    ``C-_lm = -(l+1) sqrt((l^2 - m^2) / ((2l-1)(2l+1)))``. The cores store
    degrees <= l_max only and drop the degree-(l_max+1) part; this host
    version returns an ``(L+2, L+2)`` array that keeps it, so the result is
    the exact derivative of the band-limited field (identical to the cores'
    on every state whose degree-l_max row is zero, e.g. a retained-truncation
    run). Host (NumPy) only.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"coefficients must be a square (L+1, L+1) array, got {a.shape}")
    size = a.shape[0]
    l = np.arange(size, dtype=np.float64)[:, None]
    m = np.arange(size, dtype=np.float64)[None, :]
    valid = m <= l
    c_plus = np.where(valid, l * np.sqrt(np.maximum((l + 1.0) ** 2 - m * m, 0.0)
                                         / ((2.0 * l + 1.0) * (2.0 * l + 3.0))),
                      0.0)
    c_minus = np.where(valid, -(l + 1.0) * np.sqrt(
        np.maximum(l * l - m * m, 0.0)
        / np.maximum((2.0 * l - 1.0) * (2.0 * l + 1.0), 1.0)), 0.0)
    c_minus[0, :] = 0.0
    out = np.zeros((size + 1, size + 1), dtype=np.complex128)
    out[1:, :size] += c_plus * a
    out[:size - 1, :size] += c_minus[1:] * a[1:]
    return out


def _pad(coeffs: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.complex128)
    n = coeffs.shape[0]
    out[:n, :n] = coeffs
    return out


def synthesize_wind_latlon(zeta_lm, delta_lm, radius_m: float, lat_deg,
                           lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal wind (u east, v north; m/s) from vorticity and divergence.

    The Helmholtz reconstruction of the cores
    (``PrimitiveEquationsModel.wind_on_state_grid``): ``psi = lap^-1 zeta``,
    ``chi = lap^-1 delta`` (``l = 0`` set to zero), and

        u cos(lat) = (1/R) [ sin(theta) d(psi)/d(theta) + d(chi)/d(lambda) ]
        v cos(lat) = (1/R) [ d(psi)/d(lambda) - sin(theta) d(chi)/d(theta) ]

    evaluated exactly in spectral space (:func:`sin_theta_d_theta`, ``i m``)
    and synthesized on the tensor mesh ``lat_deg x lon_deg``. The metric
    factor ``1/cos(lat)`` is applied on the mesh; on a node exactly at a pole,
    where the local east/north basis is defined only by the meridian, it is
    applied at :data:`POLE_OFFSET_DEG` from the pole along that node's
    meridian, which gives the limit of (u, v) along it (relative error of
    order 1e-7). Returns ``(u, v)``, each ``(len(lat_deg), len(lon_deg))``.
    Host (NumPy) only.
    """
    z = np.asarray(zeta_lm, dtype=np.complex128)
    d = np.asarray(delta_lm, dtype=np.complex128)
    if z.shape != d.shape or z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ValueError(
            f"zeta/delta must be equal square arrays, got {z.shape}, {d.shape}")
    radius = float(radius_m)
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError(f"radius_m must be finite and > 0, got {radius_m}")
    size = z.shape[0]
    l = np.arange(size, dtype=np.float64)[:, None]
    inv_lap = np.zeros((size, 1))
    inv_lap[1:] = -(radius ** 2) / (l[1:] * (l[1:] + 1.0))
    psi = z * inv_lap
    chi = d * inv_lap
    im = 1j * np.arange(size, dtype=np.float64)[None, :]
    big = size + 1
    u_cos = (sin_theta_d_theta(psi) + _pad(im * chi, big)) / radius
    v_cos = (_pad(im * psi, big) - sin_theta_d_theta(chi)) / radius
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    limit = 90.0 - POLE_OFFSET_DEG
    lat_eval = np.clip(lat, -limit, limit)
    cos_eval = np.cos(np.radians(lat_eval))[:, None]
    u = synthesize_latlon(u_cos, lat_eval, lon) / cos_eval
    v = synthesize_latlon(v_cos, lat_eval, lon) / cos_eval
    return u, v
