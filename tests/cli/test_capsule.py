"""Offline integrity and interruption behavior of published capsule retrieval."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from tropoi.cli import capsule
from tropoi.cli.main import main
from .conftest import assert_probe_passes


@pytest.fixture
def catalogue(tmp_path):
    root = tmp_path / "catalog" / "demo"
    root.mkdir(parents=True)
    files = {"config.json": b"{}\n", "states/state.npy": b"saved-state"}
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items()))
    (root / "config.json").write_bytes(files["config.json"])
    (root / "archive.json").write_text(json.dumps({
        "schema_version": 1, "provider": "google-drive", "included_files": ["config.json"],
        "files": {name: {"drive_id": f"id{i}", "size": len(data)}
                  for i, (name, data) in enumerate(files.items())}}))
    return root, files


def test_core_is_explicit_and_missing_full_data_fails(catalogue):
    root, _ = catalogue
    assert capsule.verify_capsule(root, root, core=True) == {"checked": 1, "missing": 1, "total": 2}
    with pytest.raises(ValueError, match="missing: states/state.npy"):
        capsule.verify_capsule(root, root)
    (root / "config.json").write_bytes(b"fabricated")
    with pytest.raises(ValueError, match="checksum mismatch"):
        capsule.verify_capsule(root, root, core=True)


def test_fetch_roundtrip_and_resume(catalogue, tmp_path, monkeypatch):
    root, files = catalogue
    destination = tmp_path / "download"
    calls = []
    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(files["config.json"] if "id=id0&" in url else files["states/state.npy"])
    monkeypatch.setattr(capsule.urllib.request, "urlopen", download)
    assert capsule.fetch_capsule(root, destination)["checked"] == 2
    assert len(calls) == 2
    assert capsule.fetch_capsule(root, destination)["checked"] == 2
    assert len(calls) == 2
    assert capsule.verify_capsule(destination, root)["missing"] == 0


def test_conflict_refused_before_network(catalogue, tmp_path, monkeypatch):
    root, _ = catalogue
    destination = tmp_path / "download"
    destination.mkdir()
    (destination / "config.json").write_bytes(b"keep this")
    def forbidden(*args, **kwargs):
        pytest.fail("Must not contact network on conflict")
    monkeypatch.setattr(capsule.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        capsule.fetch_capsule(root, destination)
    assert (destination / "config.json").read_bytes() == b"keep this"


@pytest.mark.parametrize("body", [b"sign in", b"x" * 100, b""])
def test_invalid_download_never_published(catalogue, tmp_path, monkeypatch, body):
    root, _ = catalogue
    destination = tmp_path / "download"
    monkeypatch.setattr(capsule.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(ValueError):
        capsule.fetch_capsule(root, destination)
    assert not list(destination.rglob("*"))


def test_interruption_keeps_only_verified_files(catalogue, tmp_path, monkeypatch):
    root, files = catalogue
    destination = tmp_path / "download"
    def download(url, timeout):
        if "id=id0&" in url:
            return io.BytesIO(files["config.json"])
        raise OSError("connection lost")
    monkeypatch.setattr(capsule.urllib.request, "urlopen", download)
    with pytest.raises(OSError, match="connection lost"):
        capsule.fetch_capsule(root, destination)
    assert (destination / "config.json").read_bytes() == files["config.json"]
    assert not list((destination / "states").iterdir())
    monkeypatch.setattr(capsule.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(files["states/state.npy"]))
    assert capsule.fetch_capsule(root, destination)["checked"] == 2


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/file", "a\\b", "a/../b",
                                  "a//b", "CON.txt", "a.", "a ", "a\x00b"])
def test_unsafe_manifest_paths_rejected(tmp_path, name):
    receipt = tmp_path / "SHA256SUMS"
    receipt.write_text("0" * 64 + "  " + name + "\n")
    with pytest.raises(ValueError, match="Unsafe"):
        capsule.read_receipt(receipt)


@pytest.mark.parametrize("names", [["x", "X"], ["x", "x/y"]])
def test_colliding_paths_rejected(tmp_path, names):
    receipt = tmp_path / "SHA256SUMS"
    receipt.write_text("".join("0" * 64 + "  " + n + "\n" for n in names))
    with pytest.raises(ValueError):
        capsule.read_receipt(receipt)


def test_unlisted_file_and_bad_index_fail(catalogue):
    root, _ = catalogue
    (root / "extra").write_text("unrecorded")
    with pytest.raises(ValueError, match="unlisted"):
        capsule.verify_capsule(root, root, core=True)
    data = json.loads((root / "archive.json").read_text())
    del data["files"]["states/state.npy"]
    (root / "archive.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="match SHA256SUMS"):
        capsule.verify_capsule(root, root, core=True)


def test_cli_core_full_and_errors(catalogue, tmp_path, capsys):
    root, _ = catalogue
    assert main(["capsule", "verify", "demo", "--catalog", str(root.parent), "--core"]) == 0
    assert "CORE VERIFIED: 1/2" in capsys.readouterr().out
    assert main(["capsule", "verify", str(root)]) == 1
    assert "missing:" in capsys.readouterr().err
    assert main(["capsule", "fetch", "absent", "--catalog", str(root.parent)]) == 1


def test_descriptor_size_is_checked(catalogue):
    root, _ = catalogue
    data = json.loads((root / "archive.json").read_text())
    data["files"]["config.json"]["size"] += 1
    (root / "archive.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch"):
        capsule.verify_capsule(root, root, core=True)


def test_links_refused_before_network(catalogue, tmp_path, monkeypatch):
    root, _ = catalogue
    destination = tmp_path / "linked"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == destination or original(p))
    with pytest.raises(ValueError, match="linked capsule path"):
        capsule.fetch_capsule(root, destination)


def test_concurrent_file_is_not_overwritten(catalogue, tmp_path, monkeypatch):
    root, files = catalogue
    destination = tmp_path / "download"
    def download(*args, **kwargs):
        (destination / "config.json").write_bytes(b"concurrent content")
        return io.BytesIO(files["config.json"])
    monkeypatch.setattr(capsule.urllib.request, "urlopen", download)
    with pytest.raises(FileExistsError):
        capsule.fetch_capsule(root, destination)
    assert (destination / "config.json").read_bytes() == b"concurrent content"
    assert not list(destination.glob(".capsule-*"))


@pytest.mark.parametrize("args", [["capsule", "--help"], ["capsule", "fetch", "--help"],
                                  ["capsule", "verify", "--help"]])
def test_capsule_help_is_import_light(args):
    assert_probe_passes("from tropoi.cli.main import main", f"main({args!r})")
