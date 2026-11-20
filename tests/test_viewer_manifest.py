"""Stage 1 contracts. Missing local evidence is reported by its exact path."""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.cli.main import main
from tropoi.representation.archive import open_simulation
from tropoi.representation.archive.local_resource import resource
from tropoi.representation.archive.viewer_manifest import viewer_manifest

ROOT = Path(__file__).resolve().parents[1]
BVE = "docs/runs/20260712T023305Z_two-vortices_rot24h_r4_l21_dt10h_501bce43"
SWE = "docs/runs/20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a"
MARS = "docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a"
CORE = "docs/runs/hs-T42L20-prod-002"
FULL = "runs/hs-T42L20-prod-002"
GRAY = "docs/runs/MOST_H"
BEN = "runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-30d"


def fixture(name):
    path = ROOT / name
    if not path.is_dir():
        pytest.skip(f"Missing archive: {path}")
    return path


@pytest.mark.parametrize("name,count,vertical", [
    (BVE, 25, False), (SWE, 4, False), (MARS, 9, True),
    (CORE, 0, False), (FULL, 120, True), (GRAY, 60, True), (BEN, 30, True)])
def test_fixture_discovery_without_frame_values(monkeypatch, name, count, vertical):
    path = fixture(name)
    load = np.load
    from numpy.lib.npyio import NpzFile
    member = NpzFile.__getitem__

    class HeaderOnly(np.ndarray):
        def __getitem__(self, index):
            pytest.fail("Discovery indexed scientific frame data")

    def audited_load(file, *args, **kwargs):
        frame = Path(file).name.startswith(("state-d", "surface-d")) or str(file).endswith("_coeffs.npy")
        if frame:
            assert kwargs.get("mmap_mode") == "r", "Discovery requested frame payload"
        result = load(file, *args, **kwargs)
        return result.view(HeaderOnly) if frame else result

    def audited_member(archive, key):
        assert key in {"count", "counts", "sigma", "sigma_interfaces", "lat"}
        return member(archive, key)

    monkeypatch.setattr(np, "load", audited_load)
    monkeypatch.setattr(NpzFile, "__getitem__", audited_member)
    manifest = viewer_manifest(path)
    json.dumps(manifest, allow_nan=False)
    assert len(manifest["time_axes"]["snapshots"]["values"]) == count
    operations = manifest["operations"]
    assert ("column" in operations) == vertical
    assert ("globe" in operations) == bool(count)
    assert "series" in operations
    if name == BVE:
        assert manifest["time_axes"]["snapshots"]["provenance"]["source"] == "inferred"
        assert "terrain" not in manifest["fields"]
    if name == SWE:
        assert {"free_surface_height", "layer_depth", "terrain"} <= manifest["fields"].keys()
    if name == MARS:
        assert {"pressure", "geopotential_height"} <= operations.keys()
        assert "terrain" in manifest["resources"]
    if name == CORE:
        assert "state" not in manifest["resources"] and operations["zonal_section"]
    if name in (GRAY, BEN):
        assert manifest["axes"]["support"]["product_truncation_cut"] == 21
        assert manifest["time_axes"]["slab"]["values"] == list(range(86400, (count+1)*86400, 86400))
        assert manifest["diagnostics"]["zonal.T"]["count"] == count // 2
        assert manifest["fields"]["surface_temperature"]["resource"] == "slab"


@pytest.mark.parametrize("name,levels", [(GRAY, 25), (BEN, 40)])
def test_actual_radiative_final_frame(name, levels):
    path = fixture(name)
    sim = open_simulation(path)
    day = int(sim.times[-1] / 86400)
    final = np.load(path / "states" / f"state-d{day:05d}.npy")
    np.testing.assert_array_equal(sim[-1].state["temperature"].coeffs, final[2*levels:3*levels])
    slab = sim.storage.surface(-1)
    assert slab.shape == (32, 64) and not slab.flags.writeable
    np.testing.assert_array_equal(slab.ravel(), np.load(path / "states" / f"surface-d{day:05d}.npy"))


def test_cli_and_missing_archive(tmp_path, capsys):
    assert main(["inspect", str(fixture(BVE)), "--viewer-manifest"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "palintropos.viewer/1.0"
    assert main(["inspect", str(tmp_path / "missing"), "--viewer-manifest"]) != 0
    (tmp_path / "latest_run.txt").write_text(str(fixture(BVE)))
    with pytest.raises(ValueError, match="No capsule metadata"):
        viewer_manifest(tmp_path)


@pytest.mark.parametrize("name", ["../external", "C:/external", "/external", "name:other"])
def test_resource_boundary(tmp_path, name):
    with pytest.raises(ValueError):
        resource(tmp_path, name)


def test_unknown_version_is_inspectable(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"config": {"schema": "palintropos.radiative.config/99"}}))
    m = viewer_manifest(tmp_path)
    assert not m["operations"] and "Unsupported" in m["unavailable"]["snapshots"]


def test_partial_radiative_archive(tmp_path):
    config = fixture(BEN) / "config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    folder = tmp_path / "states"
    folder.mkdir()
    np.save(folder / "state-d00030.npy", np.zeros((121, 23, 23), dtype=complex))
    m = viewer_manifest(tmp_path)
    assert m["time_axes"]["snapshots"]["values"] == [30 * 86400]
    assert "surface_temperature" not in m["fields"]
    np.savez(tmp_path / "means.npz", count=1, sigma=np.zeros(40))
    with pytest.raises(ValueError, match="disagrees"):
        viewer_manifest(tmp_path)

def test_canonical_unknown_schema_without_frames(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "future", "state_schema": {"version": 999}}))
    manifest = viewer_manifest(tmp_path)
    assert manifest["source"]["run_id"] == "future"
    assert "999" in manifest["unavailable"]["snapshots"]


def test_directory_link_is_refused(tmp_path):
    folder = tmp_path / "outside"
    folder.mkdir()
    try:
        (tmp_path / "states").symlink_to(folder, target_is_directory=True)
    except OSError:
        # Windows junctions normally do not require the symbolic-link privilege.
        import sys
        if sys.platform != "win32":
            raise
        import _winapi
        try:
            _winapi.CreateJunction(str(folder), str(tmp_path / "states"))
        except PermissionError:
            pytest.skip("Windows sandbox denies creation of both symlinks and junctions")
    with pytest.raises(ValueError, match="Linked"):
        resource(tmp_path, "states/state-d00001.npy")


def test_orphan_slab_is_reported(tmp_path):
    (tmp_path / "config.json").write_bytes((fixture(GRAY) / "config.json").read_bytes())
    states = tmp_path / "states"
    states.mkdir()
    np.save(states / "surface-d00001.npy", np.zeros(2048))
    with pytest.raises(ValueError, match="matching atmospheric day"):
        viewer_manifest(tmp_path)
