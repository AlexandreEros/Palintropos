"""Exhaustive containment coverage, including defects in either renderer."""
from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from tools.architecture import class_tree, generate
from tools.architecture.coverage import CoverageError


@pytest.fixture
def atlas_repo(make_tree, tmp_path):
    return make_tree(tmp_path, {
        "atlaspkg/__init__.py": "class FromInit: pass",
        "atlaspkg/empty.py": "",
        "atlaspkg/sub/__init__.py": "",
        "atlaspkg/sub/a.py": """
            class Standalone:
                if True:
                    class Inner: pass

                def method(self):
                    class Local: pass

            class Standalone: pass
        """,
        "atlaspkg/ns/b.py": "class Standalone: pass",
        "atlaspkg/.hidden/ignored.py": "class Hidden: pass",
    })


def _render(root):
    return generate.render(generate.Options(src=".", package="atlaspkg"), root)


def _nodes(html):
    return json.loads(re.search(r'<script id="tree-data" type="application/json">\s*(.*?)\s*</script>',
                                html, re.DOTALL)[1])


def test_lexical_containment_in_both_outputs(atlas_repo):
    files = _render(atlas_repo)
    assert files == _render(atlas_repo)
    nodes = _nodes(files["class-atlas.html"])
    by_id = {node["id"]: node for node in nodes}
    assert len(by_id) == len(nodes) == 14
    classes = [node for node in nodes if node["kind"] == "class"]
    assert len(classes) == 6
    assert sum(node["name"] == "Standalone" for node in classes) == 3
    for name, parent, path in [
        ("FromInit", "atlaspkg.__init__", "atlaspkg.FromInit"),
        ("Inner", "atlaspkg.sub.a.Standalone", "atlaspkg.sub.a.Standalone.Inner"),
        ("Local", "atlaspkg.sub.a.Standalone", "atlaspkg.sub.a.Standalone.method.<locals>.Local"),
    ]:
        (node,) = [node for node in classes if node["name"] == name]
        assert node["path"] == path
        assert by_id[node["parent"]]["path"] == parent
        assert class_tree.mermaid_id(node["id"]) in files["structure.mmd"]
    assert any(node["source"] == "atlaspkg/empty.py" for node in nodes)
    assert not any(node["source"] == "atlaspkg/ns/__init__.py" for node in nodes)
    assert "Standalone" not in files["class-inheritance.mmd"]
    assert "<title>atlaspkg — class atlas</title>" in files["class-atlas.html"]
    assert "Palintropos" not in files["class-atlas.html"]
    assert "classes.mmd" not in files  # the containment tree is structure.mmd


def test_atlas_title_is_escaped(atlas_repo):
    page = class_tree.render_html(class_tree.build_tree(generate.Project(atlas_repo, "atlaspkg")),
                                  "<b>&")
    assert "<title>&lt;b&gt;&amp; — class atlas</title>" in page and "<b>&" not in page


def test_new_file_and_class_make_both_artifacts_stale(atlas_repo):
    out = atlas_repo / "out"
    generate.write(_render(atlas_repo), out, overwrite_frozen=False)
    (atlas_repo / "atlaspkg/new.py").write_text("class NewlyDefined: pass\n", encoding="utf-8")
    files = _render(atlas_repo)
    assert {"differs: structure.mmd", "differs: class-atlas.html"} <= set(generate.check(files, out))
    assert "atlaspkg.new.NewlyDefined" in files["structure.mmd"]
    assert any(node["path"] == "atlaspkg.new.NewlyDefined"
               for node in _nodes(files["class-atlas.html"]))


@pytest.mark.parametrize("renderer,artifact", [
    ("render_mermaid", "structure.mmd"),
    ("render_html", "class-atlas.html"),
])
@pytest.mark.parametrize("kind", ["package", "module", "class"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong parent"])
def test_rendered_coverage_rejects_defects(atlas_repo, monkeypatch, renderer, artifact, kind, defect):
    original = getattr(class_tree, renderer)

    def broken(nodes, *args):
        edited = list(nodes)
        index = next(i for i, node in enumerate(edited) if node.kind == kind and node.parent)
        if defect == "missing":
            edited.pop(index)
        elif defect == "duplicate":
            edited.append(edited[index])
        else:
            edited[index] = replace(edited[index], parent=edited[index].id)
        return original(edited, *args)

    monkeypatch.setattr(class_tree, renderer, broken)
    with pytest.raises(CoverageError, match=rf"{re.escape(artifact)}:.*{defect}"):
        _render(atlas_repo)


def test_coverage_is_independent_of_the_tree_builder(atlas_repo, monkeypatch):
    original = class_tree.build_tree
    monkeypatch.setattr(class_tree, "build_tree",
                        lambda project: [n for n in original(project) if n.name != "Inner"])
    with pytest.raises(CoverageError) as error:
        _render(atlas_repo)
    assert "structure.mmd: missing" in str(error.value)
    assert "class-atlas.html: missing" in str(error.value)


def test_unrecognized_mermaid_edges_cannot_bypass_coverage(atlas_repo, monkeypatch):
    original = class_tree.render_mermaid
    monkeypatch.setattr(class_tree, "render_mermaid",
                        lambda nodes: original(nodes) + '  unexpected["Extra node"]\n')
    with pytest.raises(CoverageError, match=r"structure.mmd: unexpected statement"):
        _render(atlas_repo)


@pytest.mark.parametrize("limit,message", [
    ("MERMAID_MAX_TEXT_SIZE", "exceed Mermaid's default maxTextSize"),
    ("MERMAID_MAX_EDGES", "exceed Mermaid's default maxEdges"),
])
def test_mermaid_tree_must_fit_default_host_limits(atlas_repo, monkeypatch, limit, message):
    files = _render(atlas_repo)
    edges = files["structure.mmd"].count(" --> ")
    actual = len(files["structure.mmd"]) if limit == "MERMAID_MAX_TEXT_SIZE" else edges
    monkeypatch.setattr(class_tree, limit, actual)
    _render(atlas_repo)  # exactly at the limit still renders
    monkeypatch.setattr(class_tree, limit, actual - 1)
    with pytest.raises(CoverageError, match=rf"structure.mmd: {actual} .*{re.escape(message)}"):
        _render(atlas_repo)


def test_symlinked_subpackage_is_covered(atlas_repo, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("", encoding="utf-8")
    (shared / "linked.py").write_text("class ThroughLink: pass\n", encoding="utf-8")
    try:
        (atlas_repo / "atlaspkg/shared").symlink_to(shared, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    files = _render(atlas_repo)
    (node,) = [n for n in _nodes(files["class-atlas.html"]) if n["name"] == "ThroughLink"]
    assert node["path"] == "atlaspkg.shared.linked.ThroughLink"
    assert node["source"] == "atlaspkg/shared/linked.py"
