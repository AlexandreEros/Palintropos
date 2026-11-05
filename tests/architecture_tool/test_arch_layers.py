"""Runtime edge selection, the declared layer view, and its Mermaid rendering."""
from __future__ import annotations

import json

import networkx as nx

from tools.architecture.generate import Options, render
from tools.architecture.imports import runtime_subgraph
from tools.architecture.layers import DEFAULT_LAYERS, assign, boxes, quotient, upward
from tools.architecture.mermaid import UPWARD_STYLE, layer_flowchart
from tools.architecture.model import Graph

#: Two layers over three packages: ``low`` imports ``high`` once (an upward
#: edge), ``high`` imports ``low.side``, and ``low`` imports ``low.side``
#: (inside a layer, so not a violation).
LAYERS = [("high", ("pkg.high",)), ("low", ("pkg.low",))]


def _packages() -> Graph:
    g = Graph("packages")
    g.add_node("pkg.high", modules=2, internal_edges=0)
    g.add_node("pkg.low", modules=3, internal_edges=0)
    g.add_node("pkg.low.side", modules=1, internal_edges=0)
    g.add_node("pkg.other", modules=1, internal_edges=0)
    g.add_edge("pkg.high", "pkg.low.side", "imports", module_edges=4)
    g.add_edge("pkg.low", "pkg.high", "imports", module_edges=1)
    g.add_edge("pkg.low", "pkg.low.side", "imports", module_edges=2)
    return g


# -- runtime edge selection --------------------------------------------------
def test_runtime_subgraph_keeps_executed_edges_only():
    g = Graph("modules")
    for name in ("a", "b", "c", "d", "e"):
        g.add_node(name, path=f"{name}.py")
    g.add_edge("a", "b", "imports", scopes=["module"])
    g.add_edge("a", "c", "imports", scopes=["function"])
    g.add_edge("a", "d", "imports", scopes=["function", "type_checking"])
    g.add_edge("a", "e", "imports", scopes=["type_checking"])

    sub = runtime_subgraph(g)
    assert set(sub.nodes) == set(g.nodes)
    # A function-scope import runs when the function is called; an import that
    # exists only under TYPE_CHECKING never runs.
    assert {(s, t) for (s, t, _k) in sub.edges} == {("a", "b"), ("a", "c"), ("a", "d")}
    assert sub.edges[("a", "b", "imports")]["scopes"] == ["module"]


# -- assignment and boxes ----------------------------------------------------
def test_first_declared_prefix_wins_and_unmatched_is_unassigned():
    placed = assign(_packages().nodes, LAYERS)
    assert placed == {"pkg.high": "high", "pkg.low": "low",
                      "pkg.low.side": "low", "pkg.other": None}
    # A prefix matches the package itself and its subpackages, not a sibling
    # whose name merely starts with the same characters.
    assert assign(["pkg.lowest"], LAYERS)["pkg.lowest"] is None


def test_boxes_follow_declaration_order_with_unassigned_last():
    assert boxes(_packages(), LAYERS) == [
        ("high", ["pkg.high"]),
        ("low", ["pkg.low", "pkg.low.side"]),
        ("unassigned", ["pkg.other"]),
    ]


# -- quotient graph and violations -------------------------------------------
def test_upward_edges_are_the_ones_against_the_declaration():
    assert upward(_packages(), LAYERS) == [
        {"source": "pkg.low", "target": "pkg.high", "source_layer": "low",
         "target_layer": "high", "module_edges": 1},
    ]


def test_quotient_counts_intra_layer_edges_instead_of_self_loops():
    q = quotient(_packages(), LAYERS)
    assert q.nodes["low"] == {"depth": 1, "packages": 2, "modules": 4,
                              "internal_package_edges": 1}
    assert q.nodes["high"]["depth"] == 0
    # No self-loop for the intra-layer edge: a layer must not look like a cycle.
    assert set(q.edges) == {("high", "low", "imports"), ("low", "high", "imports")}
    assert q.edges[("low", "high", "imports")] == {"direction": "up", "package_edges": 1,
                                                   "module_edges": 1}
    assert q.edges[("high", "low", "imports")]["direction"] == "down"
    assert nx.number_of_selfloops(q.to_networkx()) == 0


def test_quotient_of_a_levelized_declaration_is_acyclic():
    g = _packages()
    del g.edges[("pkg.low", "pkg.high", "imports")]  # the only upward edge
    assert upward(g, LAYERS) == []
    assert nx.is_directed_acyclic_graph(quotient(g, LAYERS).to_networkx())


# -- Mermaid rendering -------------------------------------------------------
def test_layer_flowchart_boxes_packages_and_styles_upward_edges():
    agg = _packages()
    text = layer_flowchart(agg, boxes(agg, LAYERS), "pkg", "title",
                           upward={("pkg.low", "pkg.high")})
    lines = [line.strip() for line in text.splitlines()]
    assert lines[2] == "flowchart TD"
    # One box per non-empty layer, in declaration order, unassigned last.
    assert [line for line in lines if line.startswith("subgraph")] == [
        'subgraph g_high["high"]', 'subgraph g_low["low"]',
        'subgraph g_unassigned["unassigned"]']
    # Nodes carry the short name and module count, as in the package graph.
    assert 'n_pkg_ohigh["high (2 modules)"]' in lines
    assert 'n_pkg_olow_oside["low.side (1 module)"]' in lines
    # Edges are emitted after every box, sorted; the upward one is labelled and
    # styled by its index in that order (Mermaid numbers links as defined).
    edges = [line for line in lines if "-->" in line]
    assert edges == ['n_pkg_ohigh -->|4| n_pkg_olow_oside',
                     'n_pkg_olow -->|"1 up"| n_pkg_ohigh',
                     'n_pkg_olow -->|2| n_pkg_olow_oside']
    assert lines[-1] == f"linkStyle 1 {UPWARD_STYLE}"


def test_layer_flowchart_without_violations_has_no_link_styling():
    agg = _packages()
    text = layer_flowchart(agg, boxes(agg, LAYERS), "pkg", "title", upward=set())
    assert "linkStyle" not in text and '"1 up"' not in text


# -- end to end --------------------------------------------------------------
def _opts(**kw):
    base = dict(src=".", package="archfixture", call_roots={}, title="fixture")
    base.update(kw)
    return Options(**base)


def test_render_emits_the_layer_view_for_a_matching_declaration(fixture_src):
    # archfixture.a imports archfixture.ns.tool, so declaring ns above the
    # root package makes that import an upward edge.
    files = render(_opts(layers=[("tools", ("archfixture.ns",)),
                                 ("base", ("archfixture",))]), root=fixture_src)
    assert '"1 up"' in files["layers.mmd"] and "linkStyle" in files["layers.mmd"]
    layer_graph = json.loads(files["metrics.json"])["layer_graph"]
    assert layer_graph["upward_package_edges"] == [
        {"source": "archfixture", "target": "archfixture.ns", "source_layer": "base",
         "target_layer": "tools", "module_edges": 1}]
    assert layer_graph["unassigned_packages"] == []
    assert "Edges against the declared order" in files["metrics.md"]
    assert "layers.mmd" in files["README.md"]


def test_render_omits_the_layer_view_when_no_declared_layer_matches(fixture_src):
    files = render(_opts(layers=list(DEFAULT_LAYERS)), root=fixture_src)
    assert "layers.mmd" not in files and "graph/layers.json" not in files
    assert "layer_graph" not in json.loads(files["metrics.json"])
    assert "Declared layer graph" not in files["metrics.md"]
