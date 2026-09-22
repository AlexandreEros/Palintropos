"""End-to-end rendering: determinism, --check, and frozen-snapshot protection."""
from __future__ import annotations

import pytest

from tools.architecture import generate
from tools.architecture.generate import Options, check, render, write

FIXTURE_ROOTS = {"entry": ["archfixture.a.make_child"], "uses": ["archfixture.a.uses"]}


@pytest.fixture(scope="module")
def repo(fixture_src):
    # render() expects <root>/<src>/<package>; fixture_src is the source root.
    return fixture_src


def _opts(**kw):
    base = dict(src=".", package="archfixture", call_roots=FIXTURE_ROOTS, title="fixture")
    base.update(kw)
    return Options(**base)


def test_render_is_deterministic(repo):
    first = render(_opts(), root=repo)
    second = render(_opts(), root=repo)
    assert first == second
    assert {"README.md", "metrics.json", "metrics.md", "modules.mmd", "classes.mmd",
            "calls/entry.mmd", "calls/uses.mmd", "graph/modules.json", "graph/calls.json"} <= set(first)
    assert "modules-cycles.mmd" in first  # a <-> b via a function-scope import


def test_outputs_have_no_absolute_paths_or_crlf(repo):
    for rel, text in render(_opts(), root=repo).items():
        assert str(repo) not in text, rel
        assert str(repo).replace("\\", "/") not in text, rel
        assert "\r" not in text, rel


def test_write_then_check_then_rewrite_is_stable(repo, tmp_path):
    out = tmp_path / "out"
    files = render(_opts(), root=repo)
    write(files, out, overwrite_frozen=False)
    assert check(files, out) == []
    snapshot = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    write(render(_opts(), root=repo), out, overwrite_frozen=False)
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == snapshot


def test_check_reports_drift(repo, tmp_path):
    out = tmp_path / "out"
    files = render(_opts(), root=repo)
    write(files, out, overwrite_frozen=False)
    (out / "modules.mmd").write_text("edited\n", encoding="utf-8")
    (out / "stray.txt").write_text("x\n", encoding="utf-8")
    (out / "metrics.md").unlink()
    assert check(files, out) == ["missing: metrics.md", "differs: modules.mmd",
                                 "unexpected: stray.txt"]


def test_write_removes_stale_generated_files(repo, tmp_path):
    out = tmp_path / "out"
    write(render(_opts(), root=repo), out, overwrite_frozen=False)
    assert (out / "calls/uses.mmd").exists()
    fewer = render(_opts(call_roots={"entry": FIXTURE_ROOTS["entry"]}), root=repo)
    write(fewer, out, overwrite_frozen=False)
    assert not (out / "calls/uses.mmd").exists()
    assert check(fewer, out) == []


def test_frozen_snapshot_is_protected(repo, tmp_path):
    out = tmp_path / "frozen"
    files = render(_opts(freeze=True, source_commit="0" * 40), root=repo)
    assert generate.FROZEN_MARKER in files["README.md"]
    write(files, out, overwrite_frozen=False)
    with pytest.raises(SystemExit, match="frozen"):
        write(files, out, overwrite_frozen=False)
    assert check(files, out) == []  # --check never writes and is allowed
    write(files, out, overwrite_frozen=True)


def test_refuses_foreign_directories(repo, tmp_path):
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "notes.md").write_text("hand written\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing"):
        write(render(_opts(), root=repo), out, overwrite_frozen=False)


def test_readme_documents_provenance_rules_and_definitions(repo):
    files = render(_opts(freeze=True, source_commit="abc123", generator_commit="def456"),
                   root=repo)
    readme = files["README.md"]
    for needle in ("`abc123`", "commit `def456`", "## Inclusion and exclusion rules",
                   "## Extraction rules", "Cannot be resolved statically",
                   "## Metric definitions", "`density`", "`condensation`"):
        assert needle in readme, needle


def test_real_package_cli_write_then_check(tmp_path):
    """The documented command on the repository's own package, in fresh
    interpreters (the analysis refuses to run where the package is imported)."""
    import subprocess
    import sys

    out = tmp_path / "current"
    cmd = [sys.executable, "-m", "tools.architecture.generate", "--out", str(out)]
    subprocess.run(cmd, cwd=generate.REPO_ROOT, check=True, capture_output=True)
    first = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    result = subprocess.run([*cmd, "--check"], cwd=generate.REPO_ROOT, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run(cmd, cwd=generate.REPO_ROOT, check=True, capture_output=True)
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == first
