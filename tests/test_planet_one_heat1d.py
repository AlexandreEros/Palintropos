import json

import numpy as np
import pytest

from tools.planet_one import heat1d_compare as comparison


class _Profile:
    def __init__(self, surface_temperatures):
        self.samples = iter(surface_temperatures)
        self.T = np.zeros(1)
        self.config = None

    def update_T(self, dt, *, Qs, Qb):
        self.T[0] = next(self.samples)

    def update_properties(self):
        pass


@pytest.mark.parametrize("max_cycles, samples, change, converged", [
    (1, [100.0, 101.0], None, False),
    (2, [100.0, 101.0, 100.1, 101.2], 0.2, False),
    (2, [100.0, 101.0, 100.001, 101.005], 0.005, True),
    (3, [100.0, 101.0, 100.0, 101.0], 0.0, True),
])
def test_heat1d_cycle_limit_preserves_last_difference(max_cycles, samples, change, converged):
    result, count, measured_change = comparison._run_heat1d_cycles(
        _Profile(samples), 1.0, 2, max_cycles, lambda t: 0.0, 0.02, lambda p, cfg: 1.0)
    assert count == min(max_cycles, 2)
    np.testing.assert_array_equal(result, samples[-2:])
    assert measured_change == (None if change is None else pytest.approx(change))
    assert comparison._heat1d_is_periodic({
        "heat1d_cycles": count, "heat1d_last_cycle_change_K": measured_change,
        "heat1d_converged": converged,
    }) is converged


@pytest.mark.parametrize("verdict, levels", [(comparison.verdict, 3), (comparison.verdict4, 4)])
@pytest.mark.parametrize("change, converged", [(0.2, False), (0.0, None), (None, False)])
def test_verdict_rejects_unverified_reference_before_loading_arrays(
        tmp_path, verdict, levels, change, converged):
    valid = {"heat1d_cycles": 3, "heat1d_last_cycle_change_K": 0.005}
    invalid = {"heat1d_cycles": 3, "heat1d_last_cycle_change_K": change}
    if converged is not None:
        invalid["heat1d_converged"] = converged
    report = {"levels": [{"level": i, "columns": dict.fromkeys(comparison.COLUMNS, valid)}
                         for i in range(levels)]}
    report["levels"][-1]["columns"]["0N 90E"] = invalid
    path = tmp_path / "gi.json"
    original = json.dumps(report)
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match=f"level {levels - 1}.*0N 90E.*periodicity"):
        verdict(tmp_path)
    assert path.read_text(encoding="utf-8") == original
