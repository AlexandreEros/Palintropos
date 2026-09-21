"""Contract of the single product-truncation cut (CPU-only).

``tropoi.support.product_truncation_cut`` is the one production definition
of the 2/3-rule cut. These tests pin its values, its identity at the
historical primitive-equation import path, and its use by the BVE
diagnostic band; the operator/core sites are exercised by the CUDA suites
(the before/after extraction comparison is recorded in
docs/validation/preset_support_characterization.md).
"""
from __future__ import annotations

import pytest

from tropoi.support import product_truncation_cut

PINNED = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 10: 6, 15: 10, 21: 14,
          42: 28, 63: 42}


@pytest.mark.parametrize("l_max,cut", sorted(PINNED.items()))
def test_pinned_values(l_max, cut):
    assert product_truncation_cut(l_max) == cut


def test_matches_historical_formula_over_a_range():
    for l_max in range(0, 200):
        assert product_truncation_cut(l_max) == (2 * l_max) // 3


def test_accepts_integer_like_and_returns_int():
    import numpy as np
    assert product_truncation_cut(np.int64(21)) == 14
    assert type(product_truncation_cut(np.int64(21))) is int
    assert product_truncation_cut(21.0) == 14  # int() coercion, as before


def test_cut_never_exceeds_capacity_and_is_monotone():
    prev = -1
    for l_max in range(1, 100):
        cut = product_truncation_cut(l_max)
        assert 0 <= cut <= l_max
        assert cut >= prev
        prev = cut


def test_support_module_has_no_package_dependencies():
    import tropoi.support as support
    import tropoi.spatial.truncation as truncation
    assert support is truncation              # legacy path aliases the module
    # The cut plus the preset support guards relocated from the SWE/PE
    # configuration modules (which re-export the same objects).
    assert support.__all__ == ["product_truncation_cut", "SWE_SCENARIO_SUPPORT",
                               "require_scenario_support",
                               "THERMAL_WAVE_MIN_RETAINED_DEGREE",
                               "require_thermal_wave_support"]
    # Neutral module: no imports of tropoi subpackages, numpy, cupy, or
    # matplotlib (its import must stay CPU-safe; see tests/cli).
    import inspect
    src = inspect.getsource(support)
    for banned in ("import cupy", "import numpy", "import matplotlib",
                   "from tropoi.", "from .", "import tropoi."):
        assert banned not in src, banned


def test_primitive_equations_reexports_the_same_function():
    cp = pytest.importorskip("cupy")
    if not cp.is_available():
        pytest.skip("CUDA/CuPy not available")
    from tropoi.physics import primitive_equations as pe
    assert pe.product_truncation_cut is product_truncation_cut
