"""Documentation coverage: packages, CLI handlers and public entry points.

Every test builds its own small package under ``tmp_path``, so a rule is
exercised on code it has never seen: a new standalone class in a new
subpackage, a new CLI handler, a new exported function. The checks are the
generic ones in :mod:`tools.architecture.coverage`; nothing is named after a
real subsystem.
"""
from __future__ import annotations

import json

import pytest

from tools.architecture import generate
from tools.architecture.coverage import CoverageError
from tools.architecture.generate import Options, render

BASE = {
    "covpkg/__init__.py": """
        from covpkg.api import publish

        __all__ = ["publish"]
    """,
    "covpkg/api.py": """
        __all__ = ["publish", "helper"]


        def publish():
            return _inner()


        def _inner():
            return helper()


        def helper():
            return 1
    """,
    "covpkg/cli.py": """
        import argparse


        def _cmd_go(args):
            return 0


        def main(argv=None):
            parser = argparse.ArgumentParser()
            sub = parser.add_subparsers()
            go = sub.add_parser("go")
            go.set_defaults(_handler=_cmd_go)
            return parser.parse_args(argv)._handler(None)
    """,
    "covpkg/shapes/__init__.py": "",
    "covpkg/shapes/base.py": """
        class Shape:
            def area(self):
                return 0


        class Square(Shape):
            def area(self):
                return 1
    """,
}

SCOPE = """
[[entry_path]]
name = "publishing"
title = "Publishing"
roots = ["covpkg.api.publish"]
"""


def _repo(make_tree, tmp_path, extra=None, scope=SCOPE, drop=()):
    files = {f"src/{k}": v for k, v in {**BASE, **(extra or {})}.items() if k not in drop}
    if scope is not None:
        files["docs/entry-points.toml"] = scope
    return make_tree(tmp_path, files)


def _render(root, scope="docs/entry-points.toml", **kw):
    return render(Options(src="src", package="covpkg", title="cov", scope=scope, **kw),
                  root=root)


def _problems(root, **kw):
    with pytest.raises(CoverageError) as err:
        _render(root, **kw)
    return err.value.problems


# -- packages and classes ---------------------------------------------------
def test_new_subpackage_with_standalone_class_gets_a_page(make_tree, tmp_path):
    root = _repo(make_tree, tmp_path, {
        "covpkg/optics/__init__.py": "",
        "covpkg/optics/lens.py": """
            class Lens:
                def focus(self):
                    return 2
        """,
        "covpkg/optics/table.csv": "a,b\n",
    })
    files = _render(root)
    page = files["packages/covpkg.optics.md"]
    assert "Lens" in page and "focus()" in page              # standalone class drawn
    assert "`table.csv`" in page and "not analysed" in page  # data file listed
    assert "covpkg.optics" in files["index.md"]
    # the inheritance overview still shows only the hierarchy
    assert "Lens" not in files["classes.mmd"] and "Square" in files["classes.mmd"]
    cov = json.loads(files["coverage.json"])
    assert cov["discovered"]["classes"] == cov["rendered"]["classes"] == 3
    assert cov["discovered"]["modules"] == cov["rendered"]["modules"]
    assert cov["discovered"]["packages"] == cov["rendered"]["package_pages"] == 3
    assert cov["limits"]["not_analysed_files"] == {"covpkg.optics": ["table.csv"]}


def test_base_class_in_another_package_is_a_stub(make_tree, tmp_path):
    root = _repo(make_tree, tmp_path, {
        "covpkg/extra.py": """
            from covpkg.shapes.base import Shape


            class Circle(Shape):
                pass
        """,
    })
    page = _render(root)["packages/covpkg.md"]
    assert "Circle" in page and "Shape (shapes.base), defined elsewhere" in page


# -- CLI handlers -----------------------------------------------------------
def test_cli_handlers_are_discovered_without_configuration(make_tree, tmp_path):
    files = _render(_repo(make_tree, tmp_path))
    assert "calls/cli-cli-go.mmd" in files
    root2 = _repo(make_tree, tmp_path / "two", {
        "covpkg/tool.py": """
            def run_tool(args):
                return 0


            def register(sub):
                sub.add_parser("tool").set_defaults(_handler=run_tool)
        """,
    })
    files = _render(root2)
    assert {"calls/cli-cli-go.mmd", "calls/cli-tool-run-tool.mmd"} <= set(files)
    cov = json.loads(files["coverage.json"])
    assert cov["discovered"]["cli_handlers"]["cli-tool-run-tool"] == "covpkg.tool.run_tool"
    assert "cli-tool-run-tool" in files["index.md"]


# -- public Python entry points -----------------------------------------------
def test_entry_point_candidates_follow_reexports_and_internal_calls(make_tree, tmp_path):
    cov = json.loads(_render(_repo(make_tree, tmp_path))["coverage.json"])
    # publish is exported twice (module and package __all__) and called by
    # nobody; helper is exported but called internally, so it is not an entry
    assert cov["discovered"]["entry_point_candidates"] == {
        "covpkg.api.publish": ["covpkg.api.publish", "covpkg.publish"]}
    assert cov["declared"]["candidates_as_roots"] == ["covpkg.api.publish"]


def test_new_exported_function_must_be_declared(make_tree, tmp_path):
    root = _repo(make_tree, tmp_path, {
        "covpkg/report.py": """
            __all__ = ["summarize"]


            def summarize():
                return 3
        """,
    })
    problems = _problems(root)
    assert len(problems) == 1
    assert "undeclared public entry point covpkg.report.summarize" in problems[0]
    # declared as a root: it gets its own call graph
    declared = SCOPE + '\n[[entry_path]]\nname = "reports"\ntitle = "Reports"\n' \
                       'roots = ["covpkg.report.summarize"]\n'
    (root / "docs/entry-points.toml").write_text(declared, encoding="utf-8")
    assert "calls/reports.mmd" in _render(root)
    # or excluded with a reason: accepted and listed
    excluded = SCOPE + '\n[[excluded]]\nfunction = "covpkg.report.summarize"\n' \
                       'reason = "used by notebooks only"\n'
    (root / "docs/entry-points.toml").write_text(excluded, encoding="utf-8")
    files = _render(root)
    assert "used by notebooks only" in files["index.md"]
    # an exclusion without a reason is not accepted
    (root / "docs/entry-points.toml").write_text(
        SCOPE + '\n[[excluded]]\nfunction = "covpkg.report.summarize"\nreason = ""\n',
        encoding="utf-8")
    assert any("non-empty 'reason'" in p for p in _problems(root))


def test_no_scope_file_reports_every_candidate(make_tree, tmp_path):
    root = _repo(make_tree, tmp_path, scope=None)
    problems = _problems(root, scope=None)
    assert problems == [p for p in problems if "undeclared" in p] and len(problems) == 1
    assert _problems(root)[0] == "scope file docs/entry-points.toml not found"


@pytest.mark.parametrize("scope, expected", [
    (SCOPE + '\n[[entry_path]]\nname = "gone"\nroots = ["covpkg.api.vanished"]\n',
     "root covpkg.api.vanished is not a function"),
    (SCOPE + '\n[[excluded]]\nfunction = "covpkg.api.vanished"\nreason = "old"\n',
     "stale exclusion"),
    (SCOPE + '\n[[excluded]]\nfunction = "covpkg.api.helper"\nreason = "internal"\n',
     "no longer an entry-point candidate"),
    (SCOPE + '\n[[excluded]]\nfunction = "covpkg.api.publish"\nreason = "dup"\n',
     "both excluded and a root"),
    (SCOPE.replace('"publishing"', '"cli-publishing"'), "not start with 'cli-'"),
    (SCOPE + SCOPE, "declared more than once"),
    ("[unknown]\nx = 1\n" + SCOPE, "unknown sections"),
])
def test_stale_or_malformed_declarations_are_reported(make_tree, tmp_path, scope, expected):
    problems = _problems(_repo(make_tree, tmp_path, scope=scope))
    assert any(expected in p for p in problems), problems


# -- required narrative pages -------------------------------------------------
def test_declared_narrative_page_must_exist_and_link_its_diagram(make_tree, tmp_path):
    scope = SCOPE + 'doc = "docs/publishing.md"\n'
    root = _repo(make_tree, tmp_path, scope=scope)
    assert any("docs/publishing.md is missing" in p for p in _problems(root))
    page = root / "docs/publishing.md"
    page.write_text("# Publishing\nNo link yet.\n", encoding="utf-8")
    assert any("does not link calls/publishing.mmd" in p for p in _problems(root))
    page.write_text("# Publishing\nSee [calls/publishing.mmd](x).\n", encoding="utf-8")
    files = _render(root)
    assert "docs/publishing.md" in files["index.md"]


# -- stale outputs and the CLI ------------------------------------------------
def test_check_detects_a_new_class_and_main_fails_on_undeclared(make_tree, tmp_path,
                                                                monkeypatch):
    import subprocess

    root = _repo(make_tree, tmp_path)
    for cmd in (["init"], ["add", "."], ["commit", "-m", "fixture"]):
        subprocess.run(["git", "-c", "user.name=Fixture", "-c",
                        "user.email=fixture@example.invalid", *cmd],
                       cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(generate, "REPO_ROOT", root)
    args =["--package", "covpkg", "--out", "snapshot", "--scope", "docs/entry-points.toml"]
    assert generate.main(args) == 0
    assert generate.main([*args, "--check"]) == 0
    # a new standalone class: the committed views are now stale
    (root / "src/covpkg/shapes/extra.py").write_text(
        "class Triangle:\n    def area(self):\n        return 3\n", encoding="utf-8")
    page = root / "snapshot/packages/covpkg.shapes.md"
    assert "Triangle" not in page.read_text(encoding="utf-8")
    assert generate.main([*args, "--check"]) == 1
    assert generate.main(args) == 0
    assert "Triangle" in page.read_text(encoding="utf-8")
    assert generate.main([*args, "--check"]) == 0
    # a new exported, uncalled function: generation itself refuses
    (root / "src/covpkg/shapes/extra.py").write_text(
        "__all__ = ['make']\n\n\ndef make():\n    return 1\n", encoding="utf-8")
    assert generate.main([*args, "--check"]) == 1
    assert generate.main(args) == 1


def test_real_scope_file_covers_the_real_package():
    """The committed declarations match the committed source exactly.

    Run in a fresh interpreter: the generator refuses to analyse a package
    that is already imported, and other tests in the session import it.
    """
    import subprocess
    import sys

    from tools.architecture.generate import REPO_ROOT

    code = ("import sys; from tools.architecture.generate import DEFAULT_SCOPE, Options, "
            "render; sys.stdout.write(render(Options(scope=DEFAULT_SCOPE))['coverage.json'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    cov = json.loads(out)
    declared = set(cov["declared"]["candidates_as_roots"]) | set(cov["declared"]["excluded"])
    assert declared == set(cov["discovered"]["entry_point_candidates"])
    assert cov["rendered"]["classes"] == cov["discovered"]["classes"]
    assert {"sky-hypothetical", "sky-saved-column", "sky-numpy-reference"} <= set(
        cov["rendered"]["call_graphs"])
    assert "sky_view.html" in cov["limits"]["not_analysed_files"][
        "tropoi.representation.visual"]
