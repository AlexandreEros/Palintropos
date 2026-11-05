"""The declared layers, and ``packages.mmd`` with and without them."""
from __future__ import annotations

import json

import networkx as nx

from tools.architecture.generate import Options, render
from tools.architecture.layers import (COMMAND_LINE_SOURCE, DEFAULT_LAYERS, DEFAULT_SOURCE,
                                       assign, box_title, boxes, legend_lines, quotient,
                                       source_of, upward)
from tools.architecture.mermaid import LAYERED_WRAPPING, UPWARD_STYLE, package_flowchart
from tools.architecture.model import Graph

#: Two layers over three packages, plus one package no prefix matches:
#: ``high`` imports ``low.side`` (down), ``low`` imports ``low.side`` (inside a
#: layer), ``low`` imports ``high`` (up), and ``low.side`` imports ``high`` only
#: under ``if TYPE_CHECKING:`` (up, never executed).
LAYERS = [("high", ("pkg.high",)), ("low", ("pkg.low",))]


def _packages() -> Graph:
    g = Graph("packages")
    g.add_node("pkg.high", modules=2, internal_edges=0)
    g.add_node("pkg.low", modules=3, internal_edges=0)
    g.add_node("pkg.low.side", modules=1, internal_edges=0)
    g.add_node("pkg.other", modules=1, internal_edges=0)
    g.add_edge("pkg.high", "pkg.low.side", "imports", module_edges=4, runtime_module_edges=4)
    g.add_edge("pkg.low", "pkg.high", "imports", module_edges=1, runtime_module_edges=1)
    g.add_edge("pkg.low", "pkg.low.side", "imports", module_edges=2, runtime_module_edges=2)
    g.add_edge("pkg.low.side", "pkg.high", "imports", module_edges=1, runtime_module_edges=0)
    return g


def _layered(agg: Graph, legend=None, layers=LAYERS) -> str:
    return package_flowchart(
        agg, "pkg", "title",
        boxes=[(name, box_title(name, layers), members) for name, members in boxes(agg, layers)],
        upward={(v["source"], v["target"]) for v in upward(agg, layers)}, legend=legend)


def _edge_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "->" in line]


# -- declaration -------------------------------------------------------------
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


def test_box_titles_are_short_and_the_legend_states_the_declaration():
    # Captions stay one short line (Mermaid wraps and can hide caption text);
    # the legend carries the source and every layer's prefixes.
    assert box_title("low", LAYERS) == "layer low"
    assert box_title("unassigned", LAYERS) == "unassigned"
    lines = legend_lines(DEFAULT_LAYERS, DEFAULT_SOURCE, unassigned=False)
    assert lines[0] == f"Layers declared in {DEFAULT_SOURCE},"
    assert "assigned by package-name prefix, not inferred from imports:" in lines
    assert "temporal = tropoi.temporal" in lines and "spatial = tropoi.spatial" in lines
    assert not any(line.startswith("unassigned") for line in lines)
    # A layer with several prefixes lists them all on its line.
    assert "both = pkg.a, pkg.b" in legend_lines([("both", ("pkg.a", "pkg.b"))], "x",
                                                 unassigned=False)
    assert "unassigned = matched by no declared prefix" in legend_lines(
        LAYERS, "x", unassigned=True)


def test_source_is_stated_or_derived_never_guessed():
    assert source_of(DEFAULT_LAYERS) == DEFAULT_SOURCE
    assert source_of(LAYERS) == "caller-supplied Options.layers"
    assert source_of(DEFAULT_LAYERS, COMMAND_LINE_SOURCE) == COMMAND_LINE_SOURCE


# -- violations and the quotient ---------------------------------------------
def test_upward_edges_include_type_checking_only_ones_and_flag_them():
    assert upward(_packages(), LAYERS) == [
        {"source": "pkg.low", "target": "pkg.high", "source_layer": "low",
         "target_layer": "high", "module_edges": 1, "type_checking_only": False},
        {"source": "pkg.low.side", "target": "pkg.high", "source_layer": "low",
         "target_layer": "high", "module_edges": 1, "type_checking_only": True},
    ]


def test_quotient_counts_intra_layer_edges_instead_of_self_loops():
    q = quotient(_packages(), LAYERS)
    assert q.nodes["low"] == {"depth": 1, "packages": 2, "modules": 4,
                              "internal_package_edges": 1}
    assert q.nodes["high"]["depth"] == 0
    # No self-loop for the intra-layer edge: a layer must not look like a cycle.
    assert set(q.edges) == {("high", "low", "imports"), ("low", "high", "imports")}
    assert q.edges[("low", "high", "imports")] == {
        "direction": "up", "package_edges": 2, "type_checking_only_package_edges": 1,
        "module_edges": 2}
    assert q.edges[("high", "low", "imports")]["direction"] == "down"
    assert nx.number_of_selfloops(q.to_networkx()) == 0


def test_quotient_of_a_levelized_declaration_is_acyclic():
    g = _packages()
    del g.edges[("pkg.low", "pkg.high", "imports")]
    del g.edges[("pkg.low.side", "pkg.high", "imports")]
    assert upward(g, LAYERS) == []
    assert nx.is_directed_acyclic_graph(quotient(g, LAYERS).to_networkx())


# -- packages.mmd, layered ---------------------------------------------------
def test_layered_drawing_boxes_packages_and_marks_every_upward_edge():
    lines = [line.strip() for line in _layered(_packages()).splitlines()]
    assert lines[2:4] == [LAYERED_WRAPPING, "flowchart TD"]
    # One box per non-empty layer, in declaration order, unassigned last.
    assert [line for line in lines if line.startswith("subgraph")] == [
        'subgraph g_high["layer high"]', 'subgraph g_low["layer low"]',
        'subgraph g_unassigned["unassigned"]']
    assert 'n_pkg_ohigh["high (2 modules)"]' in lines
    assert 'n_pkg_olow_oside["low.side (1 module)"]' in lines
    # Edges after every box, sorted, each between the two packages themselves.
    # Both upward edges are drawn and labelled; the TYPE_CHECKING-only one is
    # dashed as well. linkStyle indexes them in that order (Mermaid numbers
    # links as defined).
    assert _edge_lines("\n".join(lines)) == [
        'n_pkg_ohigh -->|4| n_pkg_olow_oside',
        'n_pkg_olow -->|"1 up"| n_pkg_ohigh',
        'n_pkg_olow -->|2| n_pkg_olow_oside',
        'n_pkg_olow_oside -.->|"1 up, type-only"| n_pkg_ohigh']
    assert lines[-1] == f"linkStyle 1,3 {UPWARD_STYLE}"


def test_layered_drawing_states_its_provenance_in_the_rendered_legend():
    text = _layered(_packages(), legend=["Layers declared in somewhere", "second line"])
    assert '    legend["Layers declared in somewhere<br>second line"]' in text.splitlines()
    assert "style legend" in text
    # The legend is a standalone node: no edge touches it.
    assert not any("legend" in line for line in _edge_lines(text))


def test_layered_drawing_without_violations_has_no_link_styling():
    g = _packages()
    del g.edges[("pkg.low", "pkg.high", "imports")]
    del g.edges[("pkg.low.side", "pkg.high", "imports")]
    text = _layered(g)
    assert "linkStyle" not in text and " up" not in text


# -- packages.mmd, plain -----------------------------------------------------
def test_plain_drawing_is_an_ordinary_dependency_graph():
    text = package_flowchart(_packages(), "pkg", "title")
    lines = [line.strip() for line in text.splitlines()]
    assert lines[2] == "flowchart LR"
    assert not any(line.startswith(("subgraph", "linkStyle", "legend", "style", "%%{"))
                   for line in lines)
    # Same edges, same counts, no layer marking; TYPE_CHECKING-only still dashed.
    assert _edge_lines(text) == [
        'n_pkg_ohigh -->|4| n_pkg_olow_oside',
        'n_pkg_olow -->|1| n_pkg_ohigh',
        'n_pkg_olow -->|2| n_pkg_olow_oside',
        'n_pkg_olow_oside -.->|"1, type-only"| n_pkg_ohigh']


# -- end to end --------------------------------------------------------------
def _opts(**kw):
    base = dict(src=".", package="archfixture", call_roots={}, title="fixture")
    base.update(kw)
    return Options(**base)


def _drawn_edges(files) -> int:
    return len(_edge_lines(files["packages.mmd"]))


def test_render_boxes_packages_for_a_matching_declaration(fixture_src):
    # archfixture imports archfixture.ns.tool, so declaring ns above the root
    # package makes that import an upward edge.
    files = render(_opts(layers=[("tools", ("archfixture.ns",)),
                                 ("base", ("archfixture",))]), root=fixture_src)
    drawing = files["packages.mmd"]
    assert "subgraph" in drawing and '"1 up"' in drawing and "linkStyle" in drawing
    assert "Layers declared in caller-supplied Options.layers," in drawing
    assert "tools = archfixture.ns<br>base = archfixture<br>" in drawing
    packages = json.loads(files["graph/packages.json"])
    assert _drawn_edges(files) == len(packages["edges"])
    layer_graph = json.loads(files["metrics.json"])["layer_graph"]
    assert layer_graph["declared_by"] == "caller-supplied Options.layers"
    assert layer_graph["upward_package_edges"] == [
        {"source": "archfixture", "target": "archfixture.ns", "source_layer": "base",
         "target_layer": "tools", "module_edges": 1, "type_checking_only": False}]
    assert layer_graph["unassigned_packages"] == []
    assert "Edges against the declared order" in files["metrics.md"]
    assert "Declared layers in `packages.mmd`" in files["README.md"]


def test_render_draws_a_plain_graph_when_no_declared_layer_matches(fixture_src):
    files = render(_opts(layers=list(DEFAULT_LAYERS)), root=fixture_src)
    drawing = files["packages.mmd"]
    assert "subgraph" not in drawing and "legend" not in drawing
    assert "No layer view: no declared layer matches a package." in drawing
    assert _drawn_edges(files) == len(json.loads(files["graph/packages.json"])["edges"])
    assert "graph/layers.json" not in files
    assert "layer_graph" not in json.loads(files["metrics.json"])
    assert "Declared layer graph" not in files["metrics.md"]
    assert "No layer declaration applied" in files["README.md"]


def test_render_draws_a_plain_graph_when_no_layers_are_declared(fixture_src):
    files = render(_opts(layers=[]), root=fixture_src)
    assert "No layer view: no layers declared." in files["packages.mmd"]
    assert "subgraph" not in files["packages.mmd"]


def test_no_layers_flag_is_recorded_and_excludes_layer():
    import pytest

    from tools.architecture.generate import main

    with pytest.raises(SystemExit):
        main(["--no-layers", "--layer", "a=b", "--check"])
