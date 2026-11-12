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
        assert class_tree.mermaid_id(node["id"]) in files["classes.mmd"]
    assert any(node["source"] == "atlaspkg/empty.py" for node in nodes)
    assert not any(node["source"] == "atlaspkg/ns/__init__.py" for node in nodes)
    assert "Standalone" not in files["class-inheritance.mmd"]


def test_new_file_and_class_make_both_artifacts_stale(atlas_repo):
    out = atlas_repo / "out"
    generate.write(_render(atlas_repo), out, overwrite_frozen=False)
    (atlas_repo / "atlaspkg/new.py").write_text("class NewlyDefined: pass\n", encoding="utf-8")
    files = _render(atlas_repo)
    assert {"differs: classes.mmd", "differs: class-atlas.html"} <= set(generate.check(files, out))
    assert "atlaspkg.new.NewlyDefined" in files["classes.mmd"]
    assert any(node["path"] == "atlaspkg.new.NewlyDefined"
               for node in _nodes(files["class-atlas.html"]))


@pytest.mark.parametrize("renderer,artifact", [
    ("render_mermaid", "classes.mmd"),
    ("render_html", "class-atlas.html"),
])
@pytest.mark.parametrize("kind", ["package", "module", "class"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong parent"])
def test_rendered_coverage_rejects_defects(atlas_repo, monkeypatch, renderer, artifact, kind, defect):
    original = getattr(class_tree, renderer)

    def broken(nodes):
        edited = list(nodes)
        index = next(i for i, node in enumerate(edited) if node.kind == kind and node.parent)
        if defect == "missing":
            edited.pop(index)
        elif defect == "duplicate":
            edited.append(edited[index])
        else:
            edited[index] = replace(edited[index], parent=edited[index].id)
        return original(edited)

    monkeypatch.setattr(class_tree, renderer, broken)
    with pytest.raises(CoverageError, match=rf"{re.escape(artifact)}:.*{defect}"):
        _render(atlas_repo)


def test_coverage_is_independent_of_the_tree_builder(atlas_repo, monkeypatch):
    original = class_tree.build_tree
    monkeypatch.setattr(class_tree, "build_tree",
                        lambda project: [n for n in original(project) if n.name != "Inner"])
    with pytest.raises(CoverageError) as error:
        _render(atlas_repo)
    assert "classes.mmd: missing" in str(error.value)
    assert "class-atlas.html: missing" in str(error.value)


def test_unrecognized_mermaid_edges_cannot_bypass_coverage(atlas_repo, monkeypatch):
    original = class_tree.render_mermaid
    monkeypatch.setattr(class_tree, "render_mermaid",
                        lambda nodes: original(nodes) + '  unexpected["Extra node"]\n')
    with pytest.raises(CoverageError, match=r"classes.mmd: unexpected statement"):
        _render(atlas_repo)
