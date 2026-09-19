"""Explicit, lazy rendering of saved snapshots through the existing builders.

Host part (Matplotlib, no CUDA): the coefficient-space representation of
BVE/SWE snapshots renders on the CPU through the real default renderer;
a CuPy-less interpreter gets an actionable ``PlotUnavailableError`` for
physical representations while metadata and host coefficients keep
working; representation/normalization selection is validated; nothing is
imported before the explicit call.

CUDA part: one representative saved BVE, SWE (Williamson 2 and the
authoritative Williamson 5), and PE frame is rendered through the OLD
per-core product path and the NEW ``Snapshot.plot`` path on copies of the
real capsules; the numerical panel inputs, resolved normalizations and
image pixels must agree. Resources are built once per capsule and shared
across snapshots/representations; ``normalization="frame"`` is disclosed
as different.
"""
from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tropoi.representation.archive import open_simulation
from tropoi.representation.archive.schema import (COEFFICIENT_FILES,
                                                  TIME_FILES,
                                                  provenance_blocks)

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"
REAL = {
    "bve": RUNS / "20260718T224559Z_two-vortices_rot24h_r4_l21_dt1h_d9d06333_929a4b90",
    "swe": RUNS / "20260718T225147Z_williamson2_rot24h_r4_l21_dt1h_16f0c876_929a4b90",
    "w5": RUNS / "w5-acceptance" /
          "20260720T084339Z_williamson5_rot23p93h_r4_l21_dt120h_28083050_c583365f",
    "pe": RUNS / "20260719T224410Z_thermal-wave_rot24h_r3_l10_dt15m_fa2db863_96691b0f",
}
L_MAX = 4
N = L_MAX + 1


def _has_cuda():
    try:
        import cupy
        return cupy.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


def _synthetic(root: pathlib.Path, solver: str, frames: int = 3) -> pathlib.Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = {"bve": None, "swe": 3}[solver]
    shape = (frames, N, N) if rows is None else (frames, rows, N, N)
    coeffs = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    l = np.arange(N)[:, None]
    m = np.arange(N)[None, :]
    coeffs[..., m > l] = 0.0
    coeffs[..., :, 0] = coeffs[..., :, 0].real
    coeffs[..., 0, 0] = 0.0
    coeffs[1] *= 4.0        # frame 1 dominates the run-wide magnitude
    config = {"lmax": L_MAX, "grid": "geodesic", "resolution": 2, "nlat": 8,
              "nlon": 16, "day_hours": 24.0, "radius_earth_units": 1.0,
              "duration_days": 1.0, "dt_snapshots": 3600.0,
              "scenario": "synthetic", "snapshot_mode": "count",
              "n_snapshots": frames, "product_quadrature": "fine",
              "snapshot_times": [3600.0 * k for k in range(frames)],
              "plots": []}
    if solver == "bve":
        config["viscosity"] = 0.0
    else:
        config.update(solver="swe", gravity=9.80616, mean_depth_m=3000.0)
    np.save(root / COEFFICIENT_FILES[solver], coeffs)
    np.save(root / TIME_FILES[solver],
            np.array([3600.0 * k for k in range(frames)]))
    (root / "manifest.json").write_text(json.dumps({
        "run_id": f"{solver}-synthetic", "run_config": config,
        **provenance_blocks(solver, config)}), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Host: coefficient-space rendering and the CPU/GPU boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["bve", "swe"])
def test_spectral_representation_renders_on_the_host(tmp_path, solver):
    import matplotlib.image as mpimg
    sim = open_simulation(_synthetic(tmp_path / solver, solver))
    out = sim[2].plot(tmp_path / "out" / "frame.png", representation="spectral")
    assert out == tmp_path / "out" / "frame.png" and out.stat().st_size > 0
    image = mpimg.imread(out)
    assert image.ndim == 3 and image.shape[0] > 100
    # Metadata discloses the normalization actually used.
    from PIL import Image
    info = Image.open(out).info
    assert info["Representation"] == "spectral"
    assert info["Normalization"].startswith("timeline")
    assert info["RunId"] == f"{solver}-synthetic" and info["Snapshot"] == "2"
    frame_only = sim[2].plot(tmp_path / "out" / "frame_only.png",
                             representation="spectral", normalization="frame")
    assert Image.open(frame_only).info["Normalization"].startswith("frame")
    # Run-wide limits (frame 1 dominates) differ from single-frame limits.
    assert np.any(mpimg.imread(frame_only) != image)


def test_timeline_normalization_matches_the_persisted_product_limits(tmp_path):
    from tropoi.representation.visual.snapshot import build_snapshot_timeline
    from tropoi.run.bve.visualization import (
        build_bve_spectral_snapshot_timeline)
    sim = open_simulation(_synthetic(tmp_path / "bve", "bve"))
    old = build_bve_spectral_snapshot_timeline(
        None, tmp_path / "bve", scenario="synthetic").resolve_normalizations()
    new = build_snapshot_timeline(sim.storage, "spectral").resolve_normalizations()
    for old_frame, new_frame in zip(old.frames, new.frames):
        o = old_frame.specification.panels[0].panel
        n = new_frame.specification.panels[0].panel
        assert (o.normalization.vmin, o.normalization.vmax) == \
            (n.normalization.vmin, n.normalization.vmax)
        np.testing.assert_array_equal(o.field.coefficients_at(o.time_index),
                                      n.field.coefficients_at(n.time_index))
    single = build_snapshot_timeline(
        sim.storage, "spectral", frame_indices=slice(0, 1)).resolve_normalizations()
    assert single.frames[0].specification.panels[0].panel.normalization.vmax \
        != new.frames[0].specification.panels[0].panel.normalization.vmax


def test_representation_and_normalization_selection_is_validated(tmp_path):
    sim = open_simulation(_synthetic(tmp_path / "swe", "swe"))
    with pytest.raises(ValueError, match="available: physical, spectral"):
        sim[0].plot(tmp_path / "x.png", representation="hovmoller")
    with pytest.raises(ValueError, match="unknown normalization"):
        sim[0].plot(tmp_path / "x.png", representation="spectral",
                    normalization="global")
    assert not (tmp_path / "x.png").exists()


def test_pe_has_no_spectral_representation(tmp_path):
    from tropoi.representation.visual.snapshot import REPRESENTATIONS
    assert REPRESENTATIONS["pe"] == ("physical",)
    pe = REAL["pe"]
    if not pe.is_dir():
        pytest.skip("real PE capsule not present")
    sim = open_simulation(pe)
    with pytest.raises(ValueError, match="no 'spectral'"):
        sim[0].plot(tmp_path / "x.png", representation="spectral")


def test_plot_is_lazy_and_cuda_less_host_gets_an_actionable_error(tmp_path):
    """Fresh interpreter with CuPy blocked: open, inspect, spectral OK; physical
    fails with PlotUnavailableError; the visual module loads only on plot."""
    root = _synthetic(tmp_path / "swe", "swe")
    code = (
        "import sys\n"
        "sys.modules['cupy'] = None\n"
        "from tropoi.representation.archive import open_simulation\n"
        f"sim = open_simulation({str(root)!r})\n"
        "snap = sim[1]\n"
        "assert snap.state['phi'].validate().shape == (5, 5)\n"
        "assert 'tropoi.representation.visual.snapshot' not in sys.modules\n"
        "assert 'matplotlib' not in sys.modules\n"
        f"out = snap.plot({str(tmp_path / 'spec.png')!r}, representation='spectral')\n"
        "assert out.stat().st_size > 0\n"
        "from tropoi.representation.visual.snapshot import PlotUnavailableError\n"
        "try:\n"
        f"    snap.plot({str(tmp_path / 'phys.png')!r})\n"
        "except PlotUnavailableError as err:\n"
        "    message = str(err)\n"
        "else:\n"
        "    raise AssertionError('physical rendering succeeded without CUDA')\n"
        "assert 'needs CUDA' in message and 'spectral' in message, message\n"
        "assert snap.state['zeta'].shape == (5, 5)\n"
        "assert sim.metadata['solver'] == 'swe'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "phys.png").exists()


# ---------------------------------------------------------------------------
# CUDA: old-vs-new equivalence on copies of the real capsules
# ---------------------------------------------------------------------------

def _copy_capsule(src: pathlib.Path, dst: pathlib.Path, frames: int) -> None:
    dst.mkdir(parents=True)
    for name in ("manifest.json", "config.json"):
        shutil.copy2(src / name, dst / name)
    for npy in src.glob("*.npy"):
        np.save(dst / npy.name, np.array(np.load(npy, mmap_mode="r")[:frames]))


def _panel_inputs(spec):
    from tropoi.viz.specs import (ScalarMapSpec, SpectralCoefficientMapSpec,
                                  StreamlineMapSpec, TextPanelSpec)
    out = []
    for placement in spec.panels:
        panel = placement.panel
        if isinstance(panel, TextPanelSpec):
            # The PE header text (run id, time, selected levels).
            out.append((panel.text, np.zeros(0), None))
        elif isinstance(panel, ScalarMapSpec):
            out.append((panel.title, panel.field.values_at(panel.time_index),
                        (panel.normalization.vmin, panel.normalization.vmax)))
        elif isinstance(panel, SpectralCoefficientMapSpec):
            out.append((panel.title, panel.field.coefficients_at(panel.time_index),
                        (panel.normalization.vmin, panel.normalization.vmax)))
        elif isinstance(panel, StreamlineMapSpec):
            out.append((panel.title, np.stack([panel.zonal_velocity,
                                               panel.meridional_velocity]),
                        (panel.normalization.vmin, panel.normalization.vmax)))
    return out


@cuda
@pytest.mark.parametrize("name,index,frames", [
    ("bve", 2, 3), ("swe", 1, 3), ("w5", 3, 4), ("pe", 2, 3)])
def test_old_and_new_paths_render_the_same_saved_frame(tmp_path, name, index,
                                                       frames):
    import matplotlib.image as mpimg
    from tropoi.representation.visual.snapshot import build_snapshot_timeline

    src = REAL[name]
    if not src.is_dir():
        pytest.skip(f"real capsule {name} not present")
    dst = tmp_path / name
    _copy_capsule(src, dst, frames)
    sim = open_simulation(dst)
    assert len(sim) == frames
    storage = sim.storage
    run_id = sim.metadata["run_id"]
    scenario = sim.metadata["run_config"]["scenario"]
    resources = storage.resources()

    if sim.solver == "bve":
        from tropoi.run.bve.visualization import (
            build_bve_snapshot_timelines, render_bve_snapshots)
        old = build_bve_snapshot_timelines(resources.planet, dst,
                                           scenario=scenario)
        render_bve_snapshots(resources.planet, dst, scenario=scenario,
                             metadata={"RunId": run_id})
    elif sim.solver == "swe":
        from tropoi.run.swe.visualization import (
            build_swe_snapshot_timelines, render_swe_snapshots)
        old = build_swe_snapshot_timelines(resources.model, dst,
                                           scenario=scenario)
        render_swe_snapshots(resources.model, dst, scenario=scenario,
                             metadata={"RunId": run_id})
    else:
        from tropoi.run.pe.snapshot_visualization import (
            build_pe_snapshot_timeline, render_pe_snapshots)
        old = {"physical": build_pe_snapshot_timeline(
            resources.model, dst, scenario=scenario, run_id=run_id)}
        render_pe_snapshots(resources.model, dst, scenario=scenario,
                            metadata={"RunId": run_id})

    for rep, old_timeline in old.items():
        old_spec = old_timeline.resolve_normalizations().frames[index].specification
        new_spec = build_snapshot_timeline(
            storage, rep).resolve_normalizations().frames[index].specification
        old_inputs, new_inputs = _panel_inputs(old_spec), _panel_inputs(new_spec)
        assert len(old_inputs) == len(new_inputs) == len(old_spec.panels)
        for (t1, a1, n1), (t2, a2, n2) in zip(old_inputs, new_inputs):
            assert t1 == t2 and n1 == n2
            np.testing.assert_array_equal(a1, a2)
        assert old_spec.panel_groups == new_spec.panel_groups
        assert old_spec.size_inches == new_spec.size_inches

        new_png = sim[index].plot(tmp_path / f"{name}_{rep}.png",
                                  representation=rep)
        # The product names frames without the scenario prefix
        # (render_snapshot_product replaces filename_prefix with None).
        old_png = (dst / "snapshots" / rep /
                   replace(old_timeline, filename_prefix=None).filename_for(index))
        assert old_png.is_file()
        old_img, new_img = mpimg.imread(old_png), mpimg.imread(new_png)
        assert old_img.shape == new_img.shape
        np.testing.assert_array_equal(old_img, new_img)

    # Resources were built exactly once and geometry is shared everywhere.
    assert storage.resource_builds == 1
    assert storage.resources() is resources
    assert storage.resources().grid is resources.planet.grid


@cuda
def test_resources_are_shared_across_snapshots_and_fields(tmp_path):
    src = REAL["swe"]
    if not src.is_dir():
        pytest.skip("real SWE capsule not present")
    dst = tmp_path / "swe"
    _copy_capsule(src, dst, 2)
    sim = open_simulation(dst)
    storage = sim.storage
    assert storage.resource_builds == 0          # lazy until an explicit plot
    sim[0].state["zeta"]; sim[1].state["phi"]
    assert storage.resource_builds == 0
    sim[0].plot(tmp_path / "a.png")
    sim[1].plot(tmp_path / "b.png", normalization="frame")
    sim[1].plot(tmp_path / "c.png", representation="spectral")
    assert storage.resource_builds == 1
    resources = storage.resources()
    assert resources.model.planet is resources.planet
    assert resources.model.grid is resources.planet.grid
    assert resources.model.sh is resources.planet.sh
    # Coefficient storage sharing is a separate contract from resource sharing.
    assert np.shares_memory(sim[1].state["phi"].coeffs, storage.coefficients)
