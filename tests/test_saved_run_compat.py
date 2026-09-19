"""Compatibility of the saved-run interface with REAL saved capsules.

These tests open the actual run capsules kept under ``runs/`` (ignored by
git, present on the development machine): a current BVE run, the historical
psx-bve capsule without a stored time axis, a Williamson-2 SWE run, the
authoritative Williamson-5 acceptance capsule, and a thermal-wave PE run.
They are skipped when a capsule is absent; synthetic capsules cover the
same contracts in test_saved_run_api.py. No CUDA, no Matplotlib.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

from tropoi.representation.archive import open_simulation

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"
CAPSULES = {
    "bve": RUNS / "20260718T224559Z_two-vortices_rot24h_r4_l21_dt1h_d9d06333_929a4b90",
    "bve_legacy": RUNS / "20260712T022921Z_two-vortices_rot24h_r4_l21_dt1h_501bce43",
    "swe": RUNS / "20260718T225147Z_williamson2_rot24h_r4_l21_dt1h_16f0c876_929a4b90",
    "w5": RUNS / "w5-acceptance" /
          "20260720T084339Z_williamson5_rot23p93h_r4_l21_dt120h_28083050_c583365f",
    "pe": RUNS / "20260719T224410Z_thermal-wave_rot24h_r3_l10_dt15m_fa2db863_96691b0f",
}


def _capsule(name: str) -> pathlib.Path:
    path = CAPSULES[name]
    if not path.is_dir():
        pytest.skip(f"real capsule {name} is not present at {path}")
    return path


def test_current_bve_capsule():
    sim = open_simulation(_capsule("bve"))
    assert sim.solver == "bve" and len(sim) == 25
    assert sim.times[1] == 3600.0 and sim.times[-1] == 86400.0
    prov = sim.metadata["provenance"]
    assert prov["schema"]["source"] == "inferred"      # predates state_schema
    assert prov["time_axis"] == {"source": "stored",
                                 "file": "bve_snapshot_times.npy"}
    zeta = sim[12].state["zeta"].validate()
    assert zeta.shape == (22, 22) and zeta.units == "s^-1"
    assert np.shares_memory(zeta.coeffs, sim.storage.coefficients)
    assert sim.metadata["run_config"]["snapshot_mode"] == "count"
    assert sim.metadata["numerics"]["backend"] == "GeodesicBackend"


def test_legacy_psx_bve_capsule_without_time_axis():
    sim = open_simulation(_capsule("bve_legacy"))
    assert sim.solver == "bve" and len(sim) == 25
    prov = sim.metadata["provenance"]
    assert "psx-bve" in prov["schema"]["solver"]
    assert prov["time_axis"]["source"] == "inferred"
    np.testing.assert_array_equal(sim.times, 3600.0 * np.arange(25))
    assert sim.metadata["status"] is None          # no status field then
    assert sim.storage.schema.geometry["grid"] == "geodesic"
    sim[-1].state["zeta"].validate()


def test_williamson2_swe_capsule():
    sim = open_simulation(_capsule("swe"))
    assert sim.solver == "swe" and sim.field_names == ("zeta", "delta", "phi")
    snap = sim[24]
    assert snap.time == 86400.0
    for name in snap.state:
        view = snap.state[name].validate()
        assert view.shape == (22, 22)
    phi = snap.state["phi"]
    assert phi.units == "m^2 s^-2"
    assert "9.80616 * 3000.0" in phi.description      # Phi0 = g*H from config
    assert snap.state["zeta"].spec.monopole == "zero-mean"
    assert sim.storage.schema.environment["topography"] == "flat"


def test_williamson5_authoritative_capsule_reads_without_change():
    path = _capsule("w5")
    before = {p.name: p.stat().st_mtime_ns for p in path.iterdir()}
    sim = open_simulation(path)
    assert len(sim) == 4
    np.testing.assert_array_equal(sim.times, [0.0, 432000.0, 864000.0,
                                              1296000.0])
    env = sim.storage.schema.environment
    assert env["topography"] == "williamson5_cone" and env["w5_canonical"]
    geometry = sim.storage.schema.geometry
    assert geometry["planet"] == "ideal_sphere"
    assert geometry["reference_radius_m"] == 6.37122e6
    for snap in sim:
        for name in snap.state:
            snap.state[name].validate()
    after = {p.name: p.stat().st_mtime_ns for p in path.iterdir()}
    assert after == before                             # never modified


def test_thermal_wave_pe_capsule_level_structure():
    sim = open_simulation(_capsule("pe"))
    assert sim.solver == "pe" and len(sim) == 3
    assert sim.field_names == ("zeta", "delta", "temperature", "ln_ps")
    snap = sim[2]
    K = 8
    assert snap.state["temperature"].shape == (K, 11, 11)
    assert snap.state["zeta"].shape == (K, 11, 11)
    assert snap.state["ln_ps"].shape == (11, 11)
    assert snap.state["temperature"].level_values == tuple(
        (k + 0.5) / K for k in range(K))
    for name in snap.state:
        snap.state[name].validate()
    # Full temperature: the monopole carries the horizontal mean.
    monopole = float(snap.state["temperature"].coeffs[0, 0, 0].real)
    assert abs(monopole / np.sqrt(4 * np.pi) - 260.0) < 1.0
    schema = sim.storage.schema
    assert schema.rows == 3 * K + 1
    assert "coefficient_ordering" in schema.provenance["convention_source"]
    assert sim.metadata["notes"]["coefficient_ordering"].startswith("pe_coeffs")


def test_metadata_open_does_not_scale_with_payload(tmp_path):
    """Opening reads headers and the time axis, never the coefficients."""
    import json
    from tropoi.representation.archive.schema import provenance_blocks
    from tropoi.run.swe.config import SWERunConfig

    run_config = SWERunConfig.resolve({"scenario": "williamson2", "lmax": 63,
                                       "n_snapshots": 2}).to_run_config_dict()
    run_config_small = dict(run_config, lmax=7)

    def make(name, lmax, frames):
        root = tmp_path / name
        root.mkdir()
        rc = dict(run_config, lmax=lmax)
        coeffs = np.zeros((frames, 3, lmax + 1, lmax + 1), dtype=np.complex128)
        np.save(root / "swe_coeffs.npy", coeffs)
        np.save(root / "swe_snapshot_times.npy",
                np.arange(frames, dtype=np.float64))
        (root / "manifest.json").write_text(json.dumps({
            "run_config": rc, **provenance_blocks("swe", rc)}),
            encoding="utf-8")
        return root, coeffs.nbytes

    small, small_bytes = make("small", 7, 2)
    large, large_bytes = make("large", 63, 400)      # ~ 78 MB
    assert large_bytes > 200 * small_bytes

    def timed(path):
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            open_simulation(path).metadata
            best = min(best, time.perf_counter() - t0)
        return best

    t_small, t_large = timed(small), timed(large)
    # Both are header-only opens; the large one must not cost anything like
    # reading its payload (a generous bound, well below a full read).
    assert t_large < max(0.5, 20.0 * t_small), (t_small, t_large)
