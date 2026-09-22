"""Neutral in-memory graph model with deterministic serialisation.

A :class:`Graph` is a set of string-identified nodes and directed, typed edges,
each carrying a flat dictionary of JSON-serialisable attributes. It knows
nothing about Mermaid, grimp, or ``ast``; extractors fill it and renderers read
it. Serialisation sorts nodes, edges, and every list/set attribute so that the
same source tree always yields byte-identical output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable


def canonical(value: Any) -> Any:
    """Return a JSON-ready copy of ``value`` with a deterministic order.

    Sets and frozensets become sorted lists; dict keys are stringified and
    sorted at dump time; lists and tuples keep their order (callers that need
    them sorted must sort them, because order can be meaningful, e.g. bases).
    """
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN is not valid JSON
        raise ValueError("NaN cannot be serialised deterministically")
    return value


def dumps(value: Any) -> str:
    """Deterministic JSON text (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(canonical(value), indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


@dataclass
class Graph:
    """Directed multigraph keyed by ``(source, target, kind)``."""

    name: str
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, source: str, target: str, kind: str, **attrs: Any) -> dict:
        """Add (or return the existing) edge; attributes are merged."""
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"edge {source!r} -> {target!r} references an unknown node")
        edge = self.edges.setdefault((source, target, kind), {})
        edge.update(attrs)
        return edge

    def edges_of_kind(self, *kinds: str) -> list[tuple[str, str, str]]:
        wanted = set(kinds)
        return sorted(k for k in self.edges if not wanted or k[2] in wanted)

    def successors(self, node_id: str, kinds: Iterable[str] = ()) -> list[str]:
        wanted = set(kinds)
        return sorted({t for (s, t, k) in self.edges
                       if s == node_id and (not wanted or k in wanted)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [{"id": n, **canonical(self.nodes[n])} for n in sorted(self.nodes)],
            "edges": [{"source": s, "target": t, "kind": k, **canonical(self.edges[(s, t, k)])}
                      for (s, t, k) in sorted(self.edges)],
        }

    def to_networkx(self, kinds: Iterable[str] = ()):
        """Simple ``networkx.DiGraph`` over all nodes and the selected edge kinds.

        Parallel edges of different kinds collapse into one directed edge.
        Nodes and edges are inserted in sorted order so that any
        order-sensitive networkx routine is also deterministic.
        """
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(self.nodes))
        graph.add_edges_from((s, t) for (s, t, _k) in self.edges_of_kind(*kinds))
        return graph
