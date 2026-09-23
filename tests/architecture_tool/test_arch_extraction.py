"""Import, class, and call extraction on the synthetic ``archfixture`` package."""
from __future__ import annotations

import pytest

from tools.architecture.calls import build_call_graph, scoped
from tools.architecture.classes import build_class_graph
from tools.architecture.imports import (aggregate_by_package, build_import_graph,
                                        excluded_modules, import_time_subgraph)
from tools.architecture.sources import Project, classify

P = "archfixture"


def q(name: str) -> str:
    return f"{P}.{name}"


# -- inclusion / exclusion rules --------------------------------------------
def test_discovery_follows_generic_path_rules(project):
    assert set(project.modules) == {
        P, q("a"), q("b"), q("c"), q("d"), q("base"), q("child"), q("empty"),
        q("shim"), q("ns"), q("ns.tool"),
    }
    # *.bak, __pycache__, dot-directories, and directories without *.py are not modules.
    assert not any(m.endswith(("notes", "stale", "readme")) or ".hidden" in m or ".data" in m
                   for m in project.modules)


def test_categories(project):
    cats = {name: info.category for name, info in project.modules.items()}
    assert cats[P] == "empty"
    assert cats[q("empty")] == "empty"
    assert cats[q("shim")] == "reexport"
    assert cats[q("ns")] == "namespace"
    assert cats[q("a")] == "code"


@pytest.mark.parametrize("source, expected", [
    ("", "empty"),
    ('"""doc"""', "empty"),
    ("from x import y\n__all__ = ['y']\nZ = y", "reexport"),
    ("from x import y\nZ = y()", "code"),
    ("def f():\n    pass", "code"),
])
def test_classify(source, expected):
    import ast
    assert classify(ast.parse(source)) == expected


def test_root_must_be_a_regular_package(tmp_path):
    (tmp_path / "nopkg").mkdir()
    with pytest.raises(FileNotFoundError):
        Project(tmp_path, "nopkg")


# -- imports -----------------------------------------------------------------
@pytest.fixture(scope="module")
def modules(project):
    return build_import_graph(project)


def test_import_edges_and_scopes(modules):
    edges = {(s, t): a["scopes"] for (s, t, _k), a in modules.edges.items()}
    assert edges == {
        (q("a"), q("b")): ["module"],
        (q("a"), q("c")): ["module"],
        (q("a"), q("ns.tool")): ["module"],
        (q("a"), q("d")): ["function"],
        (q("a"), q("child")): ["function", "type_checking"],
        (q("b"), q("a")): ["function"],
        (q("child"), q("shim")): ["module"],
        (q("child"), q("base")): ["module"],
        (q("shim"), q("base")): ["module"],
    }


def test_external_imports_are_attributes_not_edges(modules):
    node = modules.nodes[q("a")]
    assert node["stdlib_imports"] == ["json", "typing"]
    assert node["third_party_imports"] == ["thirdpartylib"]
    assert all(t.startswith(P) for (_s, t, _k) in modules.edges)


def test_codeless_unimported_modules_are_excluded(project, modules):
    # Empty root __init__, an empty module, and the namespace directory itself.
    assert excluded_modules(project, modules) == [P, q("empty"), q("ns")]


def test_import_time_subgraph_drops_deferred_edges(modules):
    sub = import_time_subgraph(modules)
    assert (q("b"), q("a"), "imports") not in sub.edges
    assert (q("a"), q("child"), "imports") not in sub.edges
    assert (q("a"), q("b"), "imports") in sub.edges
    assert set(sub.nodes) == set(modules.nodes)


def test_package_aggregation(project, modules):
    agg = aggregate_by_package(modules, project)
    assert set(agg.nodes) == {P, q("ns")}
    assert agg.edges == {(P, q("ns"), "imports"): {"module_edges": 1}}
    assert agg.nodes[P]["internal_edges"] == len(modules.edges) - 1


def test_grimp_resolution_is_cross_checked(tmp_path, make_tree):
    # grimp attributes an import of a module it cannot see to the root
    # package; the cross-check against the ast target must reject that.
    make_tree(tmp_path, {"xchk/__init__.py": "", "xchk/a.py": "import xchk.b\n", "xchk/b.py": ""})
    project = Project(tmp_path, "xchk")
    project.modules["xchk.a"].import_candidates = {1: {"somewhere.else"}}
    with pytest.raises(RuntimeError, match="does not name"):
        build_import_graph(project)


def test_analysis_never_executes_the_package(tmp_path, make_tree):
    # A namespace subpackage below regular packages forces grimp to locate a
    # dotted root; the parents' __init__ must still not run.
    boom = "raise RuntimeError('analysed code was executed')\n"
    make_tree(tmp_path, {
        "noexec/__init__.py": boom,
        "noexec/sub/__init__.py": boom,
        "noexec/sub/ns/x.py": "import os\n",
        "noexec/y.py": "from noexec.sub.ns import x\n",
    })
    graph = build_import_graph(Project(tmp_path, "noexec"))
    assert ("noexec.y", "noexec.sub.ns.x", "imports") in graph.edges
    import sys
    assert not any(m == "noexec" or m.startswith("noexec.") for m in sys.modules)


# -- classes -----------------------------------------------------------------
@pytest.fixture(scope="module")
def classes(project):
    return build_class_graph(project)


def test_inheritance_resolves_through_aliases_and_reexports(classes):
    inherits = {(s, t) for (s, t, k) in classes.edges if k == "inherits"}
    assert inherits == {
        (q("child.Child"), q("base.Base")),   # via shim.Alias = Base
        (q("child.Grand"), q("base.Base")),   # via module alias base_mod.Base
    }


def test_external_bases_and_nesting(classes):
    assert classes.nodes[q("base.Base")]["external_bases"] == ["abc.ABC"]
    assert classes.nodes[q("child.Error")]["external_bases"] == ["ValueError"]
    assert (q("base.Base"), q("base.Base.Options"), "nests") in classes.edges


def test_members_are_sorted_with_kinds(classes):
    members = classes.nodes[q("base.Base")]["members"]
    assert members == [
        {"name": "__init__", "kind": "method"},
        {"name": "build", "kind": "classmethod"},
        {"name": "make", "kind": "staticmethod"},
        {"name": "run", "kind": "method"},
        {"name": "size", "kind": "property"},
        {"name": "step", "kind": "method"},
    ]


# -- calls -------------------------------------------------------------------
@pytest.fixture(scope="module")
def calls(project):
    return build_call_graph(project)


def _edges(calls, source):
    graph, _totals = calls
    return {(t, k) for (s, t, k) in graph.edges if s == source}


def test_direct_calls_through_imports(calls):
    assert _edges(calls, q("a.uses")) == {
        (q("b.g"), "call"),         # import archfixture.b; archfixture.b.g()
        (q("c.cfun"), "call"),      # from . import c; c.cfun()
        (q("ns.tool.helper"), "call"),  # from ... import helper as h; h()
    }


def test_function_local_import_is_resolved(calls):
    assert _edges(calls, q("a.local_import")) == {(q("d.deep"), "call")}


def test_parameter_shadows_module_binding(calls):
    assert _edges(calls, q("a.shadowed")) == set()


def test_instantiation_and_constructor(calls):
    graph, _ = calls
    assert (q("child.Child"), "instantiate") in _edges(calls, q("a.make_child"))
    # Child has no __init__; the inherited Base.__init__ is found.
    assert (q("child.Child"), q("base.Base.__init__"), "constructor") in graph.edges


def test_self_super_and_override_edges(calls):
    assert _edges(calls, q("base.Base.run")) == {
        (q("base.Base.step"), "self"),
        (q("child.Child.step"), "override"),
    }
    assert _edges(calls, q("child.Child.step")) == {(q("base.Base.step"), "super")}
    assert _edges(calls, q("base.Base.build")) == {(q("base.Base.make"), "self")}
    assert (q("base.Base"), "instantiate") in _edges(calls, q("base.Base.make"))


def test_call_site_outcomes(calls):
    _graph, totals = calls
    assert set(totals) == {"internal", "external", "builtin", "unresolved"}
    # uses(): 3 internal, len/object builtin, json.dumps external,
    # obj.anything() and archfixture.child_missing() unresolved.
    assert totals["builtin"] >= 2 and totals["external"] >= 1 and totals["unresolved"] >= 2


def test_scoped_bfs_depth_and_truncation(calls):
    graph, _ = calls
    sub = scoped(graph, [q("a.make_child")], max_depth=1)
    assert set(sub.nodes) == {q("a.make_child"), q("child.Child")}
    assert sub.nodes[q("child.Child")]["truncated"] is True
    deeper = scoped(graph, [q("a.make_child")], max_depth=3)
    assert q("base.Base.__init__") in deeper.nodes
    with pytest.raises(KeyError):
        scoped(graph, [q("nope")], max_depth=1)
