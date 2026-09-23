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
package. See ``docs/architecture/history/pre-sprint3/README.md`` for the
extraction rules and metric definitions as rendered for a snapshot.

``grimp`` and ``networkx`` are development-only dependencies
(``requirements-dev.txt``); nothing in ``src/`` imports this package.
"""

#: Bumped whenever a change to the tool can change generated output.
TOOL_VERSION = "1.0.0"
