"""Graph metrics on hand-checkable graphs, and Mermaid identifier/label escaping."""
from __future__ import annotations

import networkx as nx
import pytest

from tools.architecture.mermaid import (call_flowchart, class_diagram, label, member_text,
                                        node_id)
from tools.architecture.metrics import DEFINITIONS, describe
from tools.architecture.model import Graph, dumps


# -- metrics -----------------------------------------------------------------
@pytest.fixture
def cyclic():
    # 3-cycle a->b->c->a, reciprocal pair c<->d, tail d->e, isolated f.
    g = nx.DiGraph()
    g.add_nodes_from("abcdef")
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "c"), ("d", "e")])
    return g


def test_counts_density_components(cyclic):
    m = describe(cyclic)
    assert (m["nodes"], m["edges"], m["self_loops"]) == (6, 6, 0)
    assert m["density"] == round(6 / (6 * 5), 6)
    assert m["weakly_connected_components"] == {"count": 2, "sizes": [5, 1]}
    # SCCs: {a,b,c,d}, {e}, {f}
    assert m["strongly_connected_components"]["count"] == 3
    assert m["nontrivial_sccs"] == {"count": 1, "sizes": [4], "members": [["a", "b", "c", "d"]]}
    assert m["nodes_in_cycles"] == {"count": 4, "fraction": round(4 / 6, 6)}


def test_reciprocity_and_condensation(cyclic):
    m = describe(cyclic)
    assert m["reciprocal_pairs"] == {"count": 1, "pairs": [["c", "d"]]}
    assert m["reciprocity"] == round(2 / 6, 6)
    # Condensation: {abcd} -> {e}; {f} alone.
    assert m["condensation"] == {"nodes": 3, "edges": 1, "sources": 2, "sinks": 2,
                                 "longest_path_edges": 1}


def test_degree_summaries_and_top(cyclic):
    m = describe(cyclic)
    # in-degrees: a1 b1 c2 d1 e1 f0 ; out-degrees: a1 b1 c2 d2 e0 f0
    assert m["in_degree"]["histogram"] == {"0": 1, "1": 4, "2": 1}
    assert m["out_degree"]["histogram"] == {"0": 2, "1": 2, "2": 2}
    assert m["in_degree"]["mean"] == m["out_degree"]["mean"] == 1.0
    assert m["out_degree"]["median"] == 1.0
    assert (m["sources"], m["sinks"]) == (1, 2)
    assert m["top_fan_out"]["nodes"][:2] == [{"node": "c", "degree": 2}, {"node": "d", "degree": 2}]
    assert m["top_fan_in"]["nodes"][0] == {"node": "c", "degree": 2}
    # Zero-degree nodes are not listed as "top".
    assert all(e["degree"] > 0 for e in m["top_fan_out"]["nodes"])


def test_dag_has_no_cycles_and_longest_path():
    chain = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "c")])
    m = describe(chain)
    assert m["nontrivial_sccs"]["count"] == 0
    assert m["nodes_in_cycles"]["count"] == 0
    assert m["reciprocity"] == 0.0
    assert m["condensation"]["longest_path_edges"] == 2
    assert m["density"] == 0.5


def test_empty_and_singleton_graphs():
    assert describe(nx.DiGraph())["nodes"] == 0
    single = describe(nx.DiGraph([("x", "x")]))
    assert single["self_loops"] == 1
    assert single["nontrivial_sccs"]["members"] == [["x"]]


def test_every_reported_metric_is_defined(cyclic):
    defined = {part.strip() for key in DEFINITIONS for part in key.split("/")}
    for key in describe(cyclic):
        assert key in defined, key


# -- model -------------------------------------------------------------------
def test_graph_serialisation_is_order_independent():
    g1, g2 = Graph("x"), Graph("x")
    for g, order in ((g1, "abc"), (g2, "cba")):
        for n in order:
            g.add_node(n, tags={"z", "y"})
        for s, t in (("a", "b"), ("b", "c")) if order == "abc" else (("b", "c"), ("a", "b")):
            g.add_edge(s, t, "k")
    assert dumps(g1.to_dict()) == dumps(g2.to_dict())
    assert '"tags": [\n        "y",\n        "z"\n      ]' in dumps(g1.to_dict())
    with pytest.raises(KeyError):
        g1.add_edge("a", "missing", "k")


# -- mermaid -----------------------------------------------------------------
def test_node_id_is_injective_and_safe():
    names = ["a.b", "a_b", "a__b", "a._b", "a_.b", "end", "a-b", "a b", "é"]
    ids = [node_id(n) for n in names]
    assert len(set(ids)) == len(ids)
    for ident in ids:
        assert ident.replace("_", "").isalnum() and ident.isascii()
    assert node_id("end") != "end"  # Mermaid keyword cannot leak through


def test_label_escapes_mermaid_syntax():
    text = label('f(x) -> "y" [a|b] {c} #d <e>')
    assert text.startswith('"') and text.endswith('"')
    inner = text[1:-1]
    for ch in '"[]|{}<>':
        assert ch not in inner
    assert "#quot;" in inner and "#35;" in inner


def test_member_text_escapes_underscores():
    assert member_text("__init__()") == "#95;#95;init#95;#95;()"


def _call_graph():
    g = Graph("calls")
    g.add_node("m.f", type="function", kind="function", module="m", qualname="f")
    g.add_node("m.C", type="class", kind="class", module="m", qualname="C")
    g.add_node("m.C.run", type="function", kind="method", module="m", qualname="C.run")
    g.add_edge("m.f", "m.C", "instantiate")
    g.add_edge("m.C.run", "m.C.run", "self")
    for n in g.nodes:
        g.nodes[n].update(depth=0, truncated=n == "m.C")
    return g


def test_call_flowchart_styles_dispatch_edges():
    text = call_flowchart(_call_graph(), "m", "t", ["m.f"])
    assert "-->|instantiate|" in text
    assert "-.->|self|" in text
    assert f"class {node_id('m.f')} root" in text
    assert f"class {node_id('m.C')} truncated" in text
    assert text.endswith("\n") and "\r" not in text


def test_class_diagram_draws_parent_arrow():
    g = Graph("classes")
    for cid in ("m.Base", "m.Child"):
        g.add_node(cid, module="m", qualname=cid[2:], decorators=[], external_bases=[],
                   members=[{"name": "__init__", "kind": "method"}])
    g.add_edge("m.Child", "m.Base", "inherits", position=0)
    text = class_diagram(g, ["m.Base", "m.Child"], "m", "t")
    assert f"{node_id('m.Base', 'c')} <|-- {node_id('m.Child', 'c')}" in text
    assert "+#95;#95;init#95;#95;()" in text
