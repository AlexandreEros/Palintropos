"""Descriptive metrics of a directed graph (networkx standard definitions).

Every quantity is a plain measurement. None is weighted, combined into a
score, or ranked as better or worse. ``DEFINITIONS`` is rendered verbatim
into ``metrics.md`` so the numbers and their definitions cannot drift apart.
"""
from __future__ import annotations

import statistics

import networkx as nx

TOP_K = 10

#: metric key -> definition (exact networkx function where one is used).
DEFINITIONS: dict[str, str] = {
    "nodes": "|V|, the number of nodes (`G.number_of_nodes()`).",
    "edges": "|E|, the number of directed edges; parallel import statements "
             "between the same ordered pair count once (`G.number_of_edges()`).",
    "self_loops": "Edges u->u (`nx.number_of_selfloops`). Always 0 for the import "
                  "graph, which drops a module importing itself.",
    "density": "|E| / (|V|(|V|-1)), the fraction of possible ordered pairs that "
               "are edges (`nx.density` for a DiGraph).",
    "weakly_connected_components": "Number of maximal node sets connected when edge "
                                   "direction is ignored (`nx.weakly_connected_components`); "
                                   "`sizes` lists their sizes, descending.",
    "strongly_connected_components": "Number of maximal node sets in which every node "
                                     "reaches every other along directed edges "
                                     "(`nx.strongly_connected_components`), singletons included.",
    "nontrivial_sccs": "SCCs with at least 2 nodes, or one node with a self-loop. "
                       "Each is a set of mutually "
                       "reachable nodes, i.e. a dependency cycle or union of cycles. "
                       "Listed with sizes and members.",
    "nodes_in_cycles": "Nodes that belong to a nontrivial SCC (equivalently, lie on at "
                       "least one directed cycle); `fraction` = count / |V|.",
    "in_degree / out_degree": "Per-node number of incoming / outgoing edges (fan-in / "
                              "fan-out). Summary: min, max, mean (= |E|/|V| for both), "
                              "median (`statistics.median`), quartiles "
                              "(`statistics.quantiles(n=4, method='inclusive')`), count "
                              "of zeros, and the full histogram {degree: node count}.",
    "top_fan_in / top_fan_out": f"The {TOP_K} nodes with the highest in- / out-degree, "
                                "ties broken by name; `ties_at_cutoff` gives how many "
                                "nodes share the last listed value.",
    "sources / sinks": "Nodes with in-degree 0 / out-degree 0.",
    "reciprocal_pairs": "Unordered pairs {u, v} with both u->v and v->u.",
    "reciprocity": "Fraction of edges whose reverse edge also exists "
                   "(`nx.overall_reciprocity`) = 2 x reciprocal_pairs / |E|.",
    "condensation": "The DAG obtained by contracting each SCC to one node "
                    "(`nx.condensation`): its node and edge counts, source and sink "
                    "counts, and `longest_path_edges`, the number of edges on a "
                    "longest directed path in that DAG (`nx.dag_longest_path_length`).",
    "intra_package_module_edges": "Package graph only: module-level edges whose two "
                                  "modules map to the same package. They are counted "
                                  "here instead of becoming package self-loops.",
}


def _degree_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    quartiles = (statistics.quantiles(ordered, n=4, method="inclusive")
                 if len(ordered) >= 2 else [float(ordered[0])] * 3 if ordered else [])
    histogram: dict[str, int] = {}
    for v in ordered:
        histogram[str(v)] = histogram.get(str(v), 0) + 1
    return {
        "min": ordered[0] if ordered else 0,
        "max": ordered[-1] if ordered else 0,
        "mean": round(statistics.fmean(ordered), 6) if ordered else 0.0,
        "median": float(statistics.median(ordered)) if ordered else 0.0,
        "quartiles": [round(q, 6) for q in quartiles],
        "zeros": sum(1 for v in ordered if v == 0),
        "histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
    }


def _top(degree: dict[str, int]) -> dict:
    ranked = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [{"node": n, "degree": d} for n, d in ranked[:TOP_K] if d > 0]
    ties = sum(1 for _n, d in ranked if top and d == top[-1]["degree"])
    return {"nodes": top, "ties_at_cutoff": ties}


def describe(graph: nx.DiGraph) -> dict:
    """All metrics in ``DEFINITIONS`` for one directed graph."""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())

    wccs = sorted((sorted(c) for c in nx.weakly_connected_components(graph)),
                  key=lambda c: (-len(c), c))
    sccs = sorted((sorted(c) for c in nx.strongly_connected_components(graph)),
                  key=lambda c: (-len(c), c))
    nontrivial = [c for c in sccs if len(c) > 1 or graph.has_edge(c[0], c[0])]
    in_cycles = sorted(v for c in nontrivial for v in c)

    reciprocal = sorted((u, v) for u, v in graph.edges if u < v and graph.has_edge(v, u))
    cond = nx.condensation(graph) if n else nx.DiGraph()

    return {
        "nodes": n,
        "edges": m,
        "self_loops": nx.number_of_selfloops(graph),
        "density": round(nx.density(graph), 6) if n > 1 else 0.0,
        "weakly_connected_components": {"count": len(wccs), "sizes": [len(c) for c in wccs]},
        "strongly_connected_components": {"count": len(sccs)},
        "nontrivial_sccs": {
            "count": len(nontrivial),
            "sizes": [len(c) for c in nontrivial],
            "members": nontrivial,
        },
        "nodes_in_cycles": {"count": len(in_cycles),
                            "fraction": round(len(in_cycles) / n, 6) if n else 0.0},
        "in_degree": _degree_summary(list(in_deg.values())),
        "out_degree": _degree_summary(list(out_deg.values())),
        "top_fan_in": _top(in_deg),
        "top_fan_out": _top(out_deg),
        "sources": sum(1 for v in in_deg.values() if v == 0),
        "sinks": sum(1 for v in out_deg.values() if v == 0),
        "reciprocal_pairs": {"count": len(reciprocal), "pairs": [list(p) for p in reciprocal]},
        "reciprocity": round(nx.overall_reciprocity(graph), 6) if m else 0.0,
        "condensation": {
            "nodes": cond.number_of_nodes(),
            "edges": cond.number_of_edges(),
            "sources": sum(1 for _v, d in cond.in_degree() if d == 0),
            "sinks": sum(1 for _v, d in cond.out_degree() if d == 0),
            "longest_path_edges": nx.dag_longest_path_length(cond) if n else 0,
        },
    }
