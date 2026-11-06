"""The spherical Laplacian's eigenvalues on the spherical-harmonic basis.

``lap Y_l^m = -l(l+1)/R**2 * Y_l^m``. Every core, diagnostic and preparation
step that applies, inverts or damps with this operator takes its eigenvalues
from :func:`laplacian_eigenvalues`, so a mistake in them cannot hide in one
copy while the others stay right. NumPy only: callers that need a device
array wrap the result in ``cp.asarray``.

The l = 0 eigenvalue is exactly zero, and the constant mode has no inverse.
:func:`invert_laplacian` is the one inversion: it returns zero at l = 0
(the gauge every core uses for psi and chi) and never substitutes a
stand-in eigenvalue there.
"""
from __future__ import annotations

import numpy as np

__all__ = ["invert_laplacian", "laplacian_eigenvalues"]


def laplacian_eigenvalues(l_max: int, radius: float) -> np.ndarray:
    """``-l(l+1)/radius**2`` for ``l = 0 .. l_max`` as a float64 array.

    Entry 0 is zero (it may carry a negative sign bit, as ``-0.0``, which
    compares equal to ``0.0``). The positive-definite operator ``-lap`` is
    ``-laplacian_eigenvalues(l_max, radius)``.
    """
    l = np.arange(int(l_max) + 1, dtype=np.float64)
    return -l * (l + 1.0) / radius ** 2


def invert_laplacian(coeffs, eigenvalues):
    """Solve ``lap(x) = coeffs`` in spectral space, with ``x`` zero at l = 0.

    ``coeffs`` has degree ``l`` on its second-to-last axis and order ``m`` on
    its last (any leading axes, e.g. vertical levels, are batched);
    ``eigenvalues`` is :func:`laplacian_eigenvalues` on the same device. Uses
    only array methods, so NumPy and CuPy inputs both work and the result
    stays on the input's device. The l = 0 input is ignored: the inverse is
    defined only up to a constant, and zero is the chosen constant.
    """
    out = coeffs.copy()
    out[..., 0, :] = 0
    out[..., 1:, :] /= eigenvalues[1:, None]
    return out
