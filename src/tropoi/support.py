"""Spectral support contracts shared by every core (import-light).

This module deliberately imports nothing from the package: it is consumed
by ``numerics`` (operators), ``physics`` (shallow-water and primitive-
equation cores), ``run`` (diagnostics, configuration) and the CLI, so it
must stay free of CuPy, Matplotlib, and the heavy package initializers.
Keeping it neutral avoids a ``numerics -> physics`` dependency.

Vocabulary (docs/ARCHITECTURE.md, docs/validation/
preset_support_characterization.md):

* **stored capacity** ``l_max``: the triangular ``0 <= m <= l <= l_max``
  coefficient extent every prognostic array carries;
* **product truncation cut** :func:`product_truncation_cut`: the highest
  degree an analyzed nonlinear product retains (the 2/3 rule). Degrees
  ``cut < l <= l_max`` are stored and advanced by the exact *linear*
  spectral operators, but receive no nonlinear tendency.
"""
from __future__ import annotations

__all__ = ["product_truncation_cut"]


def product_truncation_cut(l_max: int) -> int:
    """The 2/3-rule truncation degree for analyzed nonlinear products.

    ``cut = floor(2 * l_max / 3)``: an analyzed product keeps degrees and
    orders ``<= cut`` and zeroes ``cut < l <= l_max`` (and ``m > cut``).
    Pinned values: ``1->0, 2->1, 3->2, 4->2, 5->3, 6->4, 10->6, 15->10,
    21->14, 42->28, 63->42``. This is the single production definition;
    every operator, core, and diagnostic band derives from it.
    """
    return (2 * int(l_max)) // 3
