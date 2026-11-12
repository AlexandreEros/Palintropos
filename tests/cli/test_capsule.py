"""Offline integrity and interruption behavior of published capsule retrieval."""
import errno
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


# --- Publication on filesystems without hard links ------------------------------
# These tests simulate POSIX filesystem behaviour with mocks on any host; they do
# not exercise a real Linux kernel, FUSE mount or Colab Drive.

def _unsupported_link(*args, **kwargs):
    raise OSError(errno.EPERM, "Operation not permitted")


def _noreplace_rename(source, target):
    # Simulates renameat2(RENAME_NOREPLACE) honouring its flag.
    if Path(target).exists():
        raise FileExistsError(errno.EEXIST, "File exists", str(target))
    Path(source).rename(target)


def _hidden_capsule_files(root):
    return list(root.rglob(".capsule-*"))


def _serve(files):
    return lambda url, timeout: io.BytesIO(
        files["config.json"] if "id=id0&" in url else files["states/state.npy"])


def _quiet(*args):
    pass


@pytest.fixture
def posix(monkeypatch):
    monkeypatch.setattr(capsule, "_NT", False)
    return monkeypatch


def test_unsupported_destination_refused_before_network(catalogue, tmp_path, posix):
    root, _ = catalogue
    parent = tmp_path / "drive-mount"
    parent.mkdir()
    destination = parent / "demo"
    posix.setattr(capsule.os, "link", _unsupported_link)
    def no_renameat2(source, target):
        raise OSError(errno.EINVAL, "Invalid argument")
    posix.setattr(capsule, "_renameat2_noreplace", no_renameat2)
    posix.setattr(capsule.urllib.request, "urlopen",
                  lambda *a, **k: pytest.fail("Must not download to an unsafe destination"))
    with pytest.raises(ValueError, match="/content/runs") as raised:
        capsule.fetch_capsule(root, destination, progress=_quiet)
    assert str(parent) in str(raised.value)
    assert not destination.exists()
    assert not list(parent.iterdir())  # Probe files cleaned up.


def test_cli_reports_unsupported_destination(catalogue, tmp_path, posix, capsys):
    root, _ = catalogue
    posix.setattr(capsule.os, "link", _unsupported_link)
    posix.setattr(capsule, "_renameat2_noreplace", _unsupported_link)
    assert main(["capsule", "fetch", "demo", "--catalog", str(root.parent),
                 "--path", str(tmp_path / "out")]) == 1
    error = capsys.readouterr().err
    assert "neither hard links nor atomic no-overwrite renames" in error
    assert "/content/runs" in error
    assert not _hidden_capsule_files(tmp_path)


def test_renameat2_fallback_publishes_verified_files(catalogue, tmp_path, posix):
    root, files = catalogue
    destination = tmp_path / "download"
    published = []
    def renameat2(source, target):
        published.append(Path(target).name)
        _noreplace_rename(source, target)
    posix.setattr(capsule.os, "link", _unsupported_link)
    posix.setattr(capsule, "_renameat2_noreplace", renameat2)
    posix.setattr(capsule.urllib.request, "urlopen", _serve(files))
    assert capsule.fetch_capsule(root, destination, progress=_quiet)["checked"] == 2
    assert {"config.json", "state.npy"} <= set(published)
    assert not _hidden_capsule_files(tmp_path)


def test_renameat2_fallback_never_overwrites(catalogue, tmp_path, posix):
    root, files = catalogue
    destination = tmp_path / "download"
    posix.setattr(capsule.os, "link", _unsupported_link)
    posix.setattr(capsule, "_renameat2_noreplace", _noreplace_rename)
    def download(*args, **kwargs):
        (destination / "config.json").write_bytes(b"concurrent content")
        return io.BytesIO(files["config.json"])
    posix.setattr(capsule.urllib.request, "urlopen", download)
    with pytest.raises(FileExistsError):
        capsule.fetch_capsule(root, destination, progress=_quiet)
    assert (destination / "config.json").read_bytes() == b"concurrent content"
    assert not _hidden_capsule_files(tmp_path)


def test_rename_that_ignores_noreplace_is_rejected(catalogue, tmp_path, posix):
    root, _ = catalogue
    posix.setattr(capsule.os, "link", _unsupported_link)
    posix.setattr(capsule, "_renameat2_noreplace", lambda s, t: Path(s).replace(t))
    posix.setattr(capsule.urllib.request, "urlopen",
                  lambda *a, **k: pytest.fail("Must not download"))
    with pytest.raises(ValueError, match="replaced an existing file"):
        capsule.fetch_capsule(root, tmp_path / "out", progress=_quiet)
    assert not _hidden_capsule_files(tmp_path)


def test_partial_download_never_visible_under_final_name(catalogue, tmp_path, posix):
    root, files = catalogue
    destination = tmp_path / "download"
    posix.setattr(capsule.os, "link", _unsupported_link)
    posix.setattr(capsule, "_renameat2_noreplace", _noreplace_rename)
    seen = []
    class Trickle(io.BytesIO):
        def read(self, n=-1):
            seen.append((destination / "config.json").exists())
            return super().read(1)
    posix.setattr(capsule.urllib.request, "urlopen", lambda url, timeout: Trickle(
        files["config.json"] if "id=id0&" in url else files["states/state.npy"]))
    capsule.fetch_capsule(root, destination, progress=_quiet)
    # Every read of config.json (its bytes plus the EOF read) precedes publication.
    assert len(seen) > len(files["config.json"])
    assert not any(seen[:len(files["config.json"]) + 1])


def test_hard_link_strategy_used_when_supported(catalogue, tmp_path, posix):
    # os.link is real here (NTFS and POSIX test hosts both support it).
    root, files = catalogue
    posix.setattr(capsule, "_renameat2_noreplace",
                  lambda *a: pytest.fail("Fallback must not be used when links work"))
    posix.setattr(capsule.urllib.request, "urlopen", _serve(files))
    assert capsule.fetch_capsule(root, tmp_path / "download", progress=_quiet)["checked"] == 2
    assert not _hidden_capsule_files(tmp_path)


def test_unwritable_destination_reported_clearly(catalogue, tmp_path, posix):
    root, _ = catalogue
    def read_only(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied")
    posix.setattr(capsule.tempfile, "NamedTemporaryFile", read_only)
    posix.setattr(capsule.urllib.request, "urlopen",
                  lambda *a, **k: pytest.fail("Must not download"))
    with pytest.raises(ValueError, match="Cannot write to"):
        capsule.fetch_capsule(root, tmp_path / "out", progress=_quiet)


def test_complete_capsule_reuse_needs_no_write_access(catalogue, tmp_path, monkeypatch):
    root, files = catalogue
    destination = tmp_path / "complete"
    for name, data in files.items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(data)
    def forbidden(*args, **kwargs):
        pytest.fail("Complete capsules must not probe, write or download")
    monkeypatch.setattr(capsule, "_probe_directory", forbidden)
    monkeypatch.setattr(capsule.tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(capsule.urllib.request, "urlopen", forbidden)
    assert capsule.fetch_capsule(root, destination, progress=_quiet)["checked"] == 2
    assert capsule.verify_capsule(destination, root)["checked"] == 2


# --- Verifying a capsule directory that does not exist ---------------------------

def test_missing_capsule_directory_is_one_clear_error(catalogue, tmp_path):
    root, _ = catalogue
    absent = tmp_path / "runs" / "demo"
    for core in (False, True):
        with pytest.raises(ValueError) as raised:
            capsule.verify_capsule(absent, root, core=core)
        message = str(raised.value)
        assert str(absent) in message and "does not exist" in message
        assert "tropoi capsule fetch demo" in message and "--path" in message
        assert "missing:" not in message


def test_cli_missing_directory_single_line(catalogue, tmp_path, capsys, monkeypatch):
    root, _ = catalogue
    monkeypatch.chdir(tmp_path)  # The default destination runs/demo does not exist.
    assert main(["capsule", "verify", "demo", "--catalog", str(root.parent)]) == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and "runs" in error and "fetch demo" in error
    assert main(["capsule", "verify", "demo", "--catalog", str(root.parent), "--core",
                 "--path", str(tmp_path / "elsewhere")]) == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and "elsewhere" in error
    # Directories that exist still get per-file detail.
    (tmp_path / "runs" / "demo").mkdir(parents=True)
    assert main(["capsule", "verify", "demo", "--catalog", str(root.parent)]) == 1
    error = capsys.readouterr().err
    assert "missing: config.json" in error and "missing: states/state.npy" in error


def test_capsule_path_that_is_a_file_is_rejected(catalogue, tmp_path):
    root, _ = catalogue
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="is not a directory"):
        capsule.verify_capsule(tmp_path / "file", root)
