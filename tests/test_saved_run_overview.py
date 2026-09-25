"""Simulation.plot / Snapshot.plot(view=...) / tropoi plot on saved runs.

Host part (no CUDA): view validation, the quantity catalogue and its
per-run discovery, snapshot/time selection, the rest threshold from the
kinetic-energy spectrum and refusal to write inside a run,
all on the committed canonical Williamson-5 capsule.

CUDA part: the canonical T63 overview reproduces the published numbers from
one model build and twelve syntheses, agrees with the independently written
grid package, leaves the capsule byte-identical, and stays readable at README
width; default overviews of real BVE, SWE and PE capsules render, including
a PE state at rest, where no wind is drawn and no spectrum is claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tropoi.representation.archive import open_simulation
from tropoi.representation.visual.quantities import (
    QuantityUnavailableError, quantity)
from tropoi.representation.visual.views import (
    Complexity, Contours, Drift, Grid, Map, Overview, Sigma, Streamlines,
    describe, parse_time)

ROOT = pathlib.Path(__file__).resolve().parents[1]
W5 = ROOT / "docs" / "validation" / "williamson_5"
CANONICAL = (W5 / "capsules" / "t63" /
             "20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_"
             "668e6c9a")
RUNS = ROOT / "runs"
REAL = {
    "bve": RUNS / "20260718T224559Z_two-vortices_rot24h_r4_l21_dt1h_d9d06333_929a4b90",
    "w2": RUNS / "20260718T225147Z_williamson2_rot24h_r4_l21_dt1h_16f0c876_929a4b90",
    "pe": RUNS / "20260719T224410Z_thermal-wave_rot24h_r3_l10_dt15m_fa2db863_96691b0f",
    "pe_rest": RUNS / "20260720T035820Z_orographic-isothermal-rest_rot24h_r3_l15_dt15m_4efda8e3_882025ec",
}


def _has_cuda():
    try:
        import cupy
        return cupy.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


def _fields(path=CANONICAL):
    from tropoi.representation.visual.evaluate import RunFields
    return RunFields(open_simulation(path).storage)


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------

def test_views_validate_their_arguments():
    with pytest.raises(ValueError):
        Map(None)
    with pytest.raises(ValueError):
        Sigma(1.0)
    with pytest.raises(ValueError):
        Overview(map=Map("vorticity"), diagnostics="some")
    with pytest.raises(ValueError):
        Grid(rows=((),))
    assert parse_time("5d") == 432000.0
    assert parse_time("12h") == 43200.0
    assert parse_time("3600") == 3600.0
    with pytest.raises(ValueError):
        parse_time("five days")
    # A view description is plain JSON (it is embedded in every figure).
    json.dumps(describe(Overview(map=Map(
        "free_surface_height", contours=(Contours("terrain", (500.0,)),),
        vectors=Streamlines()), diagnostics=(Drift(("total_mass",)),
                                             Complexity()))))


def test_catalogue_refuses_what_a_core_cannot_have_with_the_reason():
    with pytest.raises(QuantityUnavailableError, match="non-divergent"):
        quantity("divergence").check_solver("bve")
    with pytest.raises(QuantityUnavailableError, match="no bottom"):
        quantity("terrain").check_solver("bve")
    with pytest.raises(QuantityUnavailableError, match="unknown quantity"):
        quantity("vorticity_squared")


def test_discovery_is_host_only_and_reports_this_runs_availability():
    code = (
        "import sys; from tropoi.representation.archive import "
        "open_simulation; rows = open_simulation(sys.argv[1]).quantities(); "
        "print(int('cupy' in sys.modules), int('matplotlib' in "
        "sys.modules)); print(sorted(r['id'] for r in rows if "
        "r['available']))")
    output = subprocess.run([sys.executable, "-c", code, str(CANONICAL)],
                            capture_output=True, text=True, check=True)
    loaded, available = output.stdout.splitlines()
    assert loaded == "0 0"
    for name in ("free_surface_height", "terrain", "potential_enstrophy",
                 "spectral_complexity", "total_energy", "total_mass"):
        assert f"'{name}'" in available
    assert "'temperature'" not in available


def test_snapshots_are_saved_states_never_interpolated_times():
    from tropoi.representation.visual.compose import resolve_snapshots
    fields = _fields()
    assert resolve_snapshots(fields, None, 4) == [0, 1, 2, 3]
    assert resolve_snapshots(fields, (0, "10d", -1), 4) == [0, 2, 3]
    with pytest.raises(QuantityUnavailableError, match="never interpolated"):
        resolve_snapshots(fields, ("7d",), 4)
    with pytest.raises(IndexError):
        resolve_snapshots(fields, (4,), 4)
    assert resolve_snapshots(fields, None, 2) == [0, 3]


def test_default_view_is_host_only_and_matches_the_run():
    from tropoi.representation.visual.compose import default_view
    view = default_view(_fields())
    assert view.map.background == "free_surface_height"
    assert view.map.contours == (Contours("terrain", (500.0, 1000.0, 1500.0)),)
    assert isinstance(view.map.vectors, Streamlines)


def test_rest_threshold_uses_the_host_kinetic_energy_spectrum():
    fields = _fields()
    # Day 0 is solid-body flow u = 20 cos(lat): rms speed 20 sqrt(2/3).
    assert fields.rms_speed(0) == pytest.approx(20.0 * np.sqrt(2.0 / 3.0),
                                                rel=1e-3)
    measures = fields.spectral_complexity(3)
    assert round(measures.effective_modes, 2) == 4.42


def test_nice_speed_key_values():
    from tropoi.representation.visual.compose import _nice_speeds
    assert _nice_speeds(42.35) == (10.0, 20.0, 40.0)
    assert _nice_speeds(0.043) == (0.01, 0.02, 0.04)
    assert _nice_speeds(1.0) == (0.25, 0.5, 1.0)


def test_writing_inside_the_run_is_refused_before_anything_is_built(tmp_path):
    sim = open_simulation(CANONICAL)
    with pytest.raises(ValueError, match="immutable"):
        sim.plot(CANONICAL / "overview.png")
    assert sim.storage.resource_builds == 0
    assert not (CANONICAL / "overview.png").exists()


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

def _tree_digest(path: pathlib.Path) -> dict:
    return {p.relative_to(path).as_posix(): hashlib.sha256(
        p.read_bytes()).hexdigest() for p in sorted(path.rglob("*"))
        if p.is_file()}


def _release_gpu() -> None:
    """The T63 model fills most of a 2 GB card: return pooled memory."""
    import gc
    gc.collect()
    if _has_cuda():
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()


@pytest.fixture(scope="class")
def sim():
    """One shared canonical Simulation: its model is built exactly once.

    The T63 model needs about 1.8 GB; blocks pooled by earlier test modules
    are returned first so it fits on a 2 GB card.
    """
    _release_gpu()
    simulation = open_simulation(CANONICAL)
    before = _tree_digest(CANONICAL)
    yield simulation
    assert simulation.storage.resource_builds == 1
    assert _tree_digest(CANONICAL) == before, "the capsule was modified"
    del simulation
    _release_gpu()


#: Set in the child process that runs the canonical T63 checks.
_FRESH_CONTEXT = "TROPOI_T63_FRESH_CUDA_CONTEXT"


@cuda
def test_canonical_t63_checks_run_in_a_fresh_cuda_context():
    """The T63 model needs ~1.8 GB of a 2 GB card: earlier test modules in
    the same process still hold device memory, so the canonical checks run
    in a child interpreter with its own CUDA context."""
    if os.environ.get(_FRESH_CONTEXT):
        pytest.skip("already inside the child process")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{__file__}::TestCanonicalRunOnGPU"],
        env={**os.environ, _FRESH_CONTEXT: "1"}, cwd=ROOT,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]
    assert "4 passed" in result.stdout, result.stdout[-2000:]


@cuda
@pytest.mark.skipif(not os.environ.get(_FRESH_CONTEXT),
                    reason="run through "
                           "test_canonical_t63_checks_run_in_a_fresh_cuda_context")
class TestCanonicalRunOnGPU:
    """Canonical T63 checks sharing one model build (see ``sim``)."""

    def test_overview_reproduces_the_published_numbers(
            self, sim, tmp_path, monkeypatch):
        from tropoi.representation.visual.matplotlib_renderer import (
            MatplotlibRenderer)
        captured = {}
        original = MatplotlibRenderer._save_atomic

        def capture(figure, output, *, dpi, metadata):
            texts = [t for t in figure.findobj(
                lambda a: hasattr(a, "get_fontsize") and
                hasattr(a, "get_text")) if t.get_text().strip()]
            captured["smallest"] = min(t.get_fontsize() for t in texts)
            captured["width"] = figure.get_figwidth()
            return original(figure, output, dpi=dpi, metadata=metadata)

        monkeypatch.setattr(MatplotlibRenderer, "_save_atomic",
                            staticmethod(capture))
        path = sim.plot(tmp_path / "w5.png", sidecar=True)
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["omitted"] == []

        drift = record["drift"]
        assert drift["total_mass"]["relative_drift_at_saved_states"] == (
            [0.0] * 4)
        assert drift["total_mass"]["saved_states_without_exact_row"] == []
        energy = drift["total_energy"]["relative_drift_at_saved_states"]
        enstrophy = drift["potential_enstrophy"]["relative_drift"]
        for measured, published in zip(energy[1:],
                                       (-4.85e-9, -1.81e-7, -7.92e-7)):
            assert float(f"{measured:.2e}") == published
        for measured, published in zip(enstrophy[1:],
                                       (4.91e-8, -1.46e-6, -8.52e-6)):
            assert float(f"{measured:.2e}") == published
        complexity = record["spectral_complexity"]
        assert [round(v, 3) for v in complexity["mean_degree"]] == [
            1.0, 1.563, 2.149, 2.663]
        assert [round(v, 2) for v in complexity["effective_modes"]] == [
            1.0, 1.89, 3.04, 4.42]
        maps = record["maps"]
        assert [round(m["wind_speed"]["area_mean"], 2) for m in maps] == [
            15.71, 15.61, 15.65, 15.99]
        assert [round(m["wind_speed"]["max"], 2) for m in maps] == [
            20.0, 38.92, 42.35, 40.35]
        assert [(round(m["free_surface_height"]["min"]),
                 round(m["free_surface_height"]["max"])) for m in maps] == [
            (4993, 5960), (4992, 5974), (5010, 5974), (5031, 5954)]
        assert record["run"]["coefficients_sha256"].startswith("dd28eff9")
        # A fresh evaluator per figure: 4 layer depths + 4 wind pairs.
        assert record["synthesis_count"] == 12
        # Readable at an 880 px README column: at least 11 CSS px.
        assert (captured["smallest"] * 880.0 /
                (72.0 * captured["width"])) >= 11.0

    def test_evaluator_agrees_with_the_runs_independent_grid_package(
            self, sim):
        from tropoi.representation.visual.evaluate import RunFields
        fields = RunFields(sim.storage)
        with np.load(W5 / "aeolus_w5_t63.npz") as package:
            for index in range(4):
                height = fields.state_values("free_surface_height", index)
                u, v = fields.state_wind(index)
                np.testing.assert_allclose(
                    height.reshape(96, 192),
                    package["free_surface_height"][index], rtol=0, atol=1e-9)
                np.testing.assert_allclose(
                    u.reshape(96, 192), package["u"][index], rtol=0,
                    atol=1e-11)
                np.testing.assert_allclose(
                    v.reshape(96, 192), package["v"][index], rtol=0,
                    atol=1e-11)
                np.testing.assert_allclose(
                    fields.potential_enstrophy(index),
                    package["potential_enstrophy"][index], rtol=1e-13)
        # The band-limited terrain the run itself recorded.
        assert fields.state_values("terrain", None).max() == pytest.approx(
            1885.842884527992, rel=1e-14)

    def test_streamlines_alone_and_a_custom_grid_at_one_saved_state(
            self, sim, tmp_path):
        alone = sim[-1].plot(tmp_path / "flow.png",
                             Map(None, vectors=Streamlines()), sidecar=True)
        record = json.loads(alone.with_suffix(".json").read_text())
        assert record["snapshot"] == {"index": 3, "time_s": 1296000.0}
        grid = Grid(((Map("vorticity", vectors=Streamlines()),
                      Map("divergence")),
                     (Drift(("total_energy",), ("potential_enstrophy",)),
                      Complexity())))
        path = sim[2].plot(tmp_path / "grid.png", grid, sidecar=True)
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["snapshot"]["index"] == 2
        assert [sorted(m) for m in record["maps"]] == [
            ["index", "time_s", "vorticity", "wind_speed"],
            ["divergence", "index", "time_s"]]

    def test_legacy_snapshot_plot_is_unchanged_without_a_view(
            self, sim, tmp_path):
        from PIL import Image
        path = sim[0].plot(tmp_path / "legacy.png")
        with Image.open(path) as image:
            assert image.info["Representation"] == "physical"
            assert image.info["Normalization"].startswith("timeline")


@cuda
@pytest.mark.parametrize("name", sorted(REAL))
def test_default_overviews_of_every_core_render(tmp_path, name):
    if not REAL[name].is_dir():
        pytest.skip(f"local capsule {REAL[name].name} not present")
    try:
        sim = open_simulation(REAL[name])
        path = sim.plot(tmp_path / f"{name}.png", sidecar=True)
        record = json.loads(path.with_suffix(".json").read_text())
        assert sim.storage.resource_builds == 1
        assert len(record["maps"]) == min(4, len(sim))
        if name == "pe_rest":
            # Roundoff winds (~1e-15 m/s) are rest: no speed key, no
            # spectrum claimed.
            assert any("speed key" in note for note in record["omitted"])
            complexity = record["spectral_complexity"]
            assert complexity["mean_degree"] == []
            assert len(complexity["undefined_at_s"]) == len(sim)
        if name in ("pe", "pe_rest"):
            assert record["level"]["sigma"] == pytest.approx(0.6875)
        if name == "bve":
            with pytest.raises(QuantityUnavailableError,
                               match="non-divergent"):
                sim[-1].plot(tmp_path / "div.png", Map("divergence"))
    finally:
        _release_gpu()
