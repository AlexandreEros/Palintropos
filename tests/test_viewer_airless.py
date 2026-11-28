"""The viewer's columnar (airless) archives: archived values, native levels, cadences.

Archives are written by today's airless runner on small configurations
(latitude-longitude, Gaussian and point collections, with several blocks and
a short final transient segment), then read through the viewer's selection
endpoint and compared with the tier-2 arrays read directly. A Mercury
archive from an earlier runner version is checked when present.
"""
import json
import pathlib
import shutil
import tracemalloc

import numpy as np
import pytest

from tropoi.physics.insolation import insolation_from_spec
from tropoi.representation.archive.airless import AirlessArchive
from tropoi.representation.visual.selection import SelectionError
from tropoi.representation.visual.viewer_server import ViewerSession
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import AirlessRun, load_tier2

ROOT = pathlib.Path(__file__).resolve().parents[1]
MERCURY = ROOT / "runs" / "mercury-global4-vpw99-two-G0.02"


def _run(tmp, columns, steps, *, every_step=False, workers=2):
    cfg = mercury_vpw99(columns=columns)
    cfg = cfg.with_(cycle_steps=24, spinup_cycle_steps=None,
                    output={**cfg.output, "surface_samples": 12, "profile_samples": 4,
                            "every_step_profiles": every_step})
    run = AirlessRun(cfg, tmp, backend="numpy", workers=workers, allow_dirty=True, log=lambda s: None)
    run.run_transient(steps)
    return tmp


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    base = tmp_path_factory.mktemp("airless")
    return dict(
        latlon=_run(base / "latlon", {"kind": "latlon", "nlat": 4, "nlon": 8}, 30),
        gauss=_run(base / "gauss", {"kind": "gauss", "nlat": 4, "nlon": 8}, 24),
        points=_run(base / "points", {"kind": "points", "lat_deg": [10.0, -35.0], "lon_deg": [0.0, 200.0]},
                    30, every_step=True, workers=1))


def _select(session, run, **selection):
    data, mime = session.select(run, json.dumps(selection))
    return data if mime == "application/octet-stream" else json.loads(data)


def _frame(data):
    import struct
    size, = struct.unpack_from("<I", data)
    directory = json.loads(data[4:4 + size])
    return {k: np.frombuffer(data, dtype="u1" if k in ("mask", "cells") else "<f4", count=n, offset=4 + size + o)
            for k, (o, n) in directory.items()}


def _directories(root):
    return sorted((root / "cycles").iterdir())


def test_sample_times_are_the_forcing_times(archives):
    for name, root in archives.items():
        a = AirlessArchive(root)
        cfg = a.config
        ins = insolation_from_spec(cfg["illumination"])
        lat, lon = np.radians(a.lat), np.radians(a.lon)
        normals = (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
        times = a.times("surface")
        dt = cfg["cycle_s"] / cfg["cycle_steps"]
        # Every 2 steps (24 / 12), in both the full and the short final segment.
        steps = a.steps("surface")
        assert list(np.diff(steps)) == [2] * (len(steps) - 1) and steps[0] == 2
        np.testing.assert_array_equal(times, steps * dt)
        for i in range(len(times)):
            expected, _ = ins.incident(*normals, float(times[i]))
            np.testing.assert_array_equal(a.surface(i, 1), expected)
        assert list(np.diff(a.steps("profiles"))) == [6] * (len(a.steps("profiles")) - 1)


def test_short_final_segment_keeps_the_cadence(archives):
    a = AirlessArchive(archives["latlon"])
    counts = [s.count for s in a.segments["surface"]]
    assert counts == [12, 3] and [s.count for s in a.segments["profiles"]] == [4, 1]
    assert a.steps("surface")[-1] == 30 and a.steps("profiles")[-1] == 30


@pytest.mark.parametrize("kind", ["latlon", "gauss"])
def test_maps_are_archived_values_at_native_levels(archives, kind):
    root = archives[kind]
    session = ViewerSession([root])
    manifest = session.manifests[0]
    quantity = manifest["explorer"]["quantities"]["temperature"]
    a = AirlessArchive(root)
    lats, lons, index = a.layout
    assert np.all(np.diff(lats) > 0) and np.all(np.diff(lons) > 0)
    last = _directories(root)[-1]
    surface = load_tier2(last, "surface")
    profiles = load_tier2(last, "profiles")
    n_surface = len(manifest["time_axes"]["surface_samples"]["values"])
    base = dict(quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"))
    skin = _frame(_select(session, 0, **base, vertical=dict(role="fixed", representation="native", value=0),
                          time=dict(role="fixed", index=n_surface - 1)))
    np.testing.assert_array_equal(skin["values"], surface[-1, 0][index].ravel().astype("<f4"))
    assert skin["cells"][0] == 1 and not skin["mask"].any()
    n_profile = len(manifest["time_axes"]["profile_samples"]["values"])
    for k in (0, 7, a.layers - 1):
        level = quantity["vertical"]["levels"][k + 1]
        for rep, value in (("native", k + 1), ("depth", level["depth"]),
                           ("height_above_surface", level["height_above_surface"])):
            got = _frame(_select(session, 0, **base, vertical=dict(role="fixed", representation=rep, value=value),
                                 time=dict(role="fixed", index=n_profile - 1)))
            np.testing.assert_array_equal(got["values"], profiles[-1, k][index].ravel().astype("<f4"))
    # Between native depths nothing is reconstructed.
    middle = float(np.mean(a.centres[:2]))
    with pytest.raises(SelectionError, match="not reconstructed"):
        _select(session, 0, **base, vertical=dict(role="fixed", representation="depth", value=middle),
                time=dict(role="fixed", index=0))
    # The surface time axis has more samples than the cell axis: a cell map at a surface-only index is refused.
    with pytest.raises(SelectionError, match="outside"):
        _select(session, 0, **base, vertical=dict(role="fixed", representation="native", value=1),
                time=dict(role="fixed", index=n_surface - 1))
    flux = _frame(_select(session, 0, quantity="conducted_flux", longitude=dict(role="x"), latitude=dict(role="y"),
                          vertical=dict(role="none"), time=dict(role="fixed", index=3)))
    np.testing.assert_array_equal(flux["values"], load_tier2(_directories(root)[0], "surface")[3, 4][index]
                                  .ravel().astype("<f4"))


def test_column_section_and_histories(archives):
    root = archives["latlon"]
    session = ViewerSession([root])
    a = AirlessArchive(root)
    lats, lons, index = a.layout
    fixed = lambda v: dict(role="fixed", value=v)
    j = int(index[2, 5])
    lat, lon = float(a.lat[j]), float(a.lon[j])
    column = _select(session, 0, quantity="temperature", longitude=fixed(lon + 1), latitude=fixed(lat - 1),
                     vertical=dict(role="y", representation="height_above_surface"), time=dict(role="fixed", index=2))
    assert column["location"]["column"] == j
    first = _directories(root)[0]
    np.testing.assert_array_equal(column["values"], load_tier2(first, "profiles")[2, :, j])
    np.testing.assert_array_equal(column["y"]["values"], -a.centres)
    np.testing.assert_array_equal(column["y"]["edges"], -a.interfaces)
    # The skin is the surface sample of the same step (profile 2 is step 18 = surface sample 8).
    skin = [m for m in column["markers"] if m["kind"] == "surface"][0]
    assert skin["value"] == load_tier2(first, "surface")[8, 0, j] and skin["position"] == 0.0
    section = _select(session, 0, quantity="temperature", latitude=dict(role="x"), longitude=fixed(lon),
                      vertical=dict(role="y", representation="depth"), time=dict(role="fixed", index=2))
    np.testing.assert_array_equal(section["x"]["values"], lats)
    np.testing.assert_allclose(section["x"]["edges"], np.linspace(-90, 90, lats.size + 1))
    np.testing.assert_array_equal(np.array(section["values"]),
                                  load_tier2(first, "profiles")[2][:, index[:, 5]])
    np.testing.assert_array_equal(section["surface"], load_tier2(first, "surface")[8, 0, index[:, 5]])
    profile_times = session.manifests[0]["time_axes"]["profile_samples"]["values"]
    history = _select(session, 0, quantity="temperature", longitude=fixed(lon), latitude=fixed(lat),
                      vertical=dict(role="y", representation="depth"),
                      time=dict(role="x", first=0, last=len(profile_times) - 1))
    assert history["x"]["values"] == profile_times
    full = np.concatenate([load_tier2(d, "profiles")[:, :, j] for d in _directories(root)])
    np.testing.assert_array_equal(np.array(history["values"]).T, full)
    surface_times = session.manifests[0]["time_axes"]["surface_samples"]["values"]
    series = _select(session, 0, quantity="temperature", longitude=fixed(lon), latitude=fixed(lat),
                     vertical=dict(role="fixed", representation="native", value=0),
                     time=dict(role="x", first=0, last=len(surface_times) - 1))
    assert series["x"]["values"] == surface_times
    np.testing.assert_array_equal(series["values"],
                                  np.concatenate([load_tier2(d, "surface")[:, 0, j] for d in _directories(root)]))


def test_point_collection_is_not_a_global_field(archives):
    root = archives["points"]
    session = ViewerSession([root])
    manifest = session.manifests[0]
    quantity = manifest["explorer"]["quantities"]["temperature"]
    assert manifest["explorer"]["horizontal"]["kind"] == "points"
    assert manifest["explorer"]["horizontal"]["area_weights"] is False
    assert set(quantity["products"]) == {"column", "time_series", "time_section"}
    assert "global field" in quantity["unavailable"]["map"]
    assert quantity["time_axis"] == "every_step"
    with pytest.raises(SelectionError, match="not available"):
        _select(session, 0, quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"),
                vertical=dict(role="fixed", representation="native", value=0), time=dict(role="fixed", index=0))
    every = np.concatenate([load_tier2(d, "every_step")[:, :, 1] for d in _directories(root)])
    history = _select(session, 0, quantity="temperature", longitude=dict(role="fixed", value=200),
                      latitude=dict(role="fixed", value=-35), vertical=dict(role="y", representation="depth"),
                      time=dict(role="x", first=0, last=29))
    assert history["location"]["column"] == 1 and len(history["x"]["values"]) == 30
    np.testing.assert_array_equal(np.array(history["values"]).T, every[:, 1:])
    np.testing.assert_array_equal(history["surface"], every[:, 0])


def test_linked_resource_inside_an_archive_is_refused(archives, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(archives["points"], copy)
    target = copy / "cycles" / "cycle-0000" / "surface-b000.npy"
    target.unlink()
    target.symlink_to(archives["points"] / "cycles" / "cycle-0000" / "surface-b000.npy")
    with pytest.raises(ValueError, match="Linked archive resource"):
        AirlessArchive(copy).surface(0, 0)


def test_viewer_code_names_no_planet():
    """Archives are read by their contract; no viewer code recognizes a world by name."""
    src = ROOT / "src" / "tropoi" / "representation"
    files = [src / "archive" / "airless.py", src / "archive" / "viewer_manifest.py",
             *(src / "visual").glob("viewer_*.py"), src / "visual" / "selection.py",
             *(src / "visual" / "viewer_assets").iterdir()]
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for name in ("mercury", "ceres", "vpw99", "mola", "thai", "ben2", "ben 2"):
            assert name not in text, f"{path.name} mentions {name}"


@pytest.mark.skipif(not MERCURY.is_dir(), reason="Needs the 4° Mercury archive in runs/")
def test_large_archive_reads_stay_bounded():
    """Maps, columns and histories of a 1.6 GB archive read only what they show."""
    session = ViewerSession([MERCURY])
    manifest = session.manifests[0]
    # Written before column_kind and state.npz existed: the kind and the timing say where they came from.
    assert manifest["explorer"]["horizontal"]["source"] == "config.json:columns.kind"
    assert manifest["time_axes"]["surface_samples"]["segments"][0]["timing"] == "cycles.json"
    assert manifest["source"]["status"] == "PERIODIC"
    tracemalloc.start()
    _select(session, 0, quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"),
            vertical=dict(role="fixed", representation="native", value=20), time=dict(role="fixed", index=100))
    _select(session, 0, quantity="temperature", longitude=dict(role="fixed", value=0),
            latitude=dict(role="fixed", value=0), vertical=dict(role="y", representation="depth"),
            time=dict(role="x", first=0, last=539))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 16 << 20, peak


def test_tropoi_run_airless_is_the_module_entry_point(tmp_path):
    """``tropoi run airless`` runs exactly what ``python -m tropoi.run.airless run`` runs."""
    from tropoi.cli.main import main as tropoi
    from tropoi.run.airless.__main__ import main as module
    config = str(ROOT / "tests" / "fixtures" / "airless" / "ceres-equatorial-column.airless.json")
    common = ["--config", config, "--steps", "5", "--device", "cpu", "--allow-dirty"]
    assert module(["run", *common, "--out", str(tmp_path / "module")]) == 0
    assert tropoi(["run", "airless", *common, "--out", str(tmp_path / "cli")]) == 0
    for name in ("surface", "profiles", "every_step"):
        np.testing.assert_array_equal(load_tier2(tmp_path / "module" / "retained", name),
                                      load_tier2(tmp_path / "cli" / "retained", name))
    documents = [json.loads((tmp_path / d / "config.json").read_text()) for d in ("module", "cli")]
    assert documents[0]["config_sha256"] == documents[1]["config_sha256"]
    session = ViewerSession([tmp_path / "cli"])
    assert "temperature" in session.manifests[0]["explorer"]["quantities"]
