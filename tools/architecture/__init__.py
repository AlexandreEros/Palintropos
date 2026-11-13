"""Deterministic, architecture-neutral structural analysis of a Python package.

Pipeline::

    Python source
        +-- stdlib ``ast``  -> classes, inheritance, methods, static call sites
        +-- ``grimp``       -> module import graph
                    |
            neutral graph model (``model.Graph``)
                    |
            Mermaid diagrams + descriptive graph metrics (``networkx``)

The extractors record generic structural facts only (A imports B, class A
inherits from B, function f has a statically resolvable call to g). They carry
no knowledge of, or expectations about, any particular layering of the analysed
package. One input can state such an expectation: a layer declaration
(``--layer``, default ``DEFAULT_LAYERS`` in ``tools.architecture.layers``). With it,
``packages.mmd`` boxes the packages by declared layer, says where the
declaration came from, and marks every edge that runs against its order;
without one it is a plain dependency graph. Either way every edge is drawn and
no measurement changes. See ``docs/architecture/current/README.md``
for the extraction rules and metric definitions as rendered for a snapshot.

``grimp`` and ``networkx`` are development-only dependencies
(``requirements-dev.txt``); nothing in ``src/`` imports this package.
"""

#: Bumped whenever a change to the tool can change generated output.
TOOL_VERSION = "2.2.0"
