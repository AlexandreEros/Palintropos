"""Group packages into declared layers and find the edges that point up.

Everything else in this tool extracts facts and takes no position on how the
analysed package should be arranged. This module is the one place where a
*declaration* enters: an ordered list of layers, top (most dependent) first,
each naming package prefixes. The declaration is an option of the generator
(``--layer``), not an extraction rule, and it changes no measurement: the
layer view renders the same neutral import graph grouped by the declaration,
and marks every edge that contradicts it.

With the declaration, two questions become answerable:

* is the quotient graph over the declared layers acyclic (can the layers be
  levelized, i.e. does every dependency point downward)?
* if not, which package edges point up? Those are the layer violations, and
  removing them (a feedback arc set) is what makes the quotient a DAG.

An edge inside one layer is not a violation; it is counted on the layer node
rather than drawn as a self-loop, so a layer never looks like a cycle of its
own.
"""
from __future__ import annotations

from .model import Graph

#: Declared layers of ``tropoi``, top (most dependent) first. Every package
#: whose dotted name starts with one of a layer's prefixes belongs to it; a
#: package matched by no prefix is reported as unassigned. ``spatial`` and
#: ``temporal`` are declared as one ``core`` layer here: their own direction
#: (``temporal`` imports ``spatial``, never the reverse) stays visible as an
#: ordinary edge inside the box.
DEFAULT_LAYERS: list[tuple[str, tuple[str, ...]]] = [
    ("cli", ("tropoi.cli",)),
    ("run", ("tropoi.run",)),
    ("representation", ("tropoi.representation",)),
    ("core", ("tropoi.spatial", "tropoi.temporal")),
]

#: Name of the box collecting packages that no declared prefix matches.
UNASSIGNED = "unassigned"


def _matches(package: str, prefix: str) -> bool:
    return package == prefix or package.startswith(prefix + ".")


def assign(packages, layers) -> dict[str, str | None]:
    """Map each package to its layer name, or ``None`` when no prefix matches.

    The first matching layer in declaration order wins, so a more specific
    layer declared earlier can carve a subpackage out of a later one.
    """
    out: dict[str, str | None] = {}
    for package in sorted(packages):
        out[package] = next((name for name, prefixes in layers
                             if any(_matches(package, p) for p in prefixes)), None)
    return out


def order_of(layers) -> dict[str, int]:
    """Layer name -> depth index; unassigned packages sort after every layer."""
    depth = {name: i for i, (name, _prefixes) in enumerate(layers)}
    depth[UNASSIGNED] = len(depth)
    return depth


def boxes(agg: Graph, layers) -> list[tuple[str, list[str]]]:
    """Non-empty layer boxes in declaration order, each with its packages."""
    placed = assign(agg.nodes, layers)
    grouped: dict[str, list[str]] = {}
    for package, name in placed.items():
        grouped.setdefault(name or UNASSIGNED, []).append(package)
    names = [name for name, _prefixes in layers] + [UNASSIGNED]
    return [(name, sorted(grouped[name])) for name in names if grouped.get(name)]


def upward(agg: Graph, layers) -> list[dict]:
    """Package edges of ``agg`` that run from a lower layer to a higher one.

    These are the layer violations: each is one upward dependency, recorded
    with the layers it crosses and how many module-level import edges were
    merged into it.
    """
    placed = assign(agg.nodes, layers)
    depth = order_of(layers)
    found = []
    for (s, t, _k), attrs in sorted(agg.edges.items()):
        source, target = placed[s] or UNASSIGNED, placed[t] or UNASSIGNED
        if depth[target] < depth[source]:
            found.append({"source": s, "target": t, "source_layer": source,
                          "target_layer": target,
                          "module_edges": attrs.get("module_edges", 0)})
    return found


def quotient(agg: Graph, layers) -> Graph:
    """Collapse the package graph onto the declared layers.

    Node attributes: ``depth`` (0 = top), ``packages``, ``modules``, and
    ``internal_package_edges`` (edges between two packages of the same
    layer, counted instead of drawn as a self-loop). Edge attributes:
    ``package_edges``, ``module_edges``, and ``direction`` (``down`` or
    ``up``).
    """
    placed = assign(agg.nodes, layers)
    depth = order_of(layers)
    out = Graph("layers")
    for name, packages in boxes(agg, layers):
        out.add_node(name, depth=depth[name], packages=len(packages),
                     modules=sum(agg.nodes[p]["modules"] for p in packages),
                     internal_package_edges=0)
    for (s, t, _k), attrs in sorted(agg.edges.items()):
        source, target = placed[s] or UNASSIGNED, placed[t] or UNASSIGNED
        if source == target:
            out.nodes[source]["internal_package_edges"] += 1
            continue
        edge = out.add_edge(source, target, "imports",
                            direction="up" if depth[target] < depth[source] else "down")
        edge["package_edges"] = edge.get("package_edges", 0) + 1
        edge["module_edges"] = edge.get("module_edges", 0) + attrs.get("module_edges", 0)
    return out
