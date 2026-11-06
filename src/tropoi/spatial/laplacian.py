"""The spherical Laplacian's eigenvalues on the spherical-harmonic basis.

``lap Y_l^m = -l(l+1)/R**2 * Y_l^m``. Every core, diagnostic and preparation
step that applies, inverts or damps with this operator takes its eigenvalues
from :func:`laplacian_eigenvalues`, so a mistake in them cannot hide in one
copy while the others stay right. NumPy only: callers that need a device
array wrap the result in ``cp.asarray``.

The l = 0 eigenvalue is exactly zero. A caller that inverts the operator
must special-case l = 0 itself (the constant mode has no inverse); this
module never substitutes a stand-in value for it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["laplacian_eigenvalues"]


def laplacian_eigenvalues(l_max: int, radius: float) -> np.ndarray:
    """``-l(l+1)/radius**2`` for ``l = 0 .. l_max`` as a float64 array.

    Entry 0 is zero (it may carry a negative sign bit, as ``-0.0``, which
    compares equal to ``0.0``). The positive-definite operator ``-lap`` is
    ``-laplacian_eigenvalues(l_max, radius)``.
    """
    l = np.arange(int(l_max) + 1, dtype=np.float64)
    return -l * (l + 1.0) / radius ** 2
