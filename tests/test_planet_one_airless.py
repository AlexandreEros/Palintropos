import json

import numpy as np
import pytest

from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import build_columns, build_substrate, load_tier2, run_block


def _columns(cfg):
    """(lat, lon in radians, weights) of the configured columns, as run_block takes them."""
    c = build_columns(cfg)
    return c.lat, c.lon, c.weights


def test_retained_profiles_follow_requested_order(tmp_path):
    profiles = np.arange(24).reshape(2, 2, 6)
    columns = [4, 2, 5, 0, -2, 3]
    np.save(tmp_path / "profiles.npy", profiles)
    expected = load_tier2(tmp_path, "profiles", columns)
    (tmp_path / "blocks.json").write_text(json.dumps({"ranges": [[0, 3], [3, 6]]}))
    for i, block in enumerate(np.split(profiles, 2, axis=-1)):
        np.save(tmp_path / f"profiles-b{i:03d}.npy", block)
    np.testing.assert_array_equal(load_tier2(tmp_path, "profiles", columns), expected)
    assert load_tier2(tmp_path, "profiles", []).shape == (2, 2, 0)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_cycle_extent_includes_unsampled_temperatures(tmp_path, backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if not cp.is_available():
            pytest.skip("needs a CUDA GPU")
    cfg = mercury_vpw99(columns={"kind": "points", "lat_deg": [0.0], "lon_deg": [90.0]})
    cfg = cfg.with_(output={**cfg.output, "surface_samples": 1, "profile_samples": 1})
    sub = build_substrate(cfg)
    state = sub.initial_state(1)
    result = run_block(cfg.to_dict(), *_columns(cfg), state.anomaly,
                       state.surface_anomaly, 0, 8, True, str(tmp_path), None, 0, backend=backend)
    series = np.load(tmp_path / "every_step-b000.npy")
    first_stratum = np.count_nonzero(sub.materials.stratum_of_cell == 0)
    initial = np.concatenate([sub.surface_temperature(state)[None], sub.temperature(state)])
    history = np.concatenate([initial[None], series])
    top, bottom = np.split(history, [first_stratum + 1], axis=1)
    assert top.max() > series[-1, :first_stratum + 1].max()
    np.testing.assert_allclose(result["extent"], [(top.min(), top.max()),
                                                  (bottom.min(), bottom.max())])
