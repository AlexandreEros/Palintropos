"""Class graph from the stdlib ``ast``: classes, inheritance, methods.

Facts recorded
--------------
* One node per class statement found at module scope or nested in another
  class body (``module.Outer.Inner``). Classes defined inside functions are
  not recorded.
* Node attributes: module, qualified name, decorators (source text), the
  functions defined directly in the class body (name, kind) sorted by name,
  where kind is method/classmethod/staticmethod/property; and any base
  that is not an analysed class (``external_bases``, e.g. ``abc.ABC``) as the
  absolute dotted name when an import makes it known, else the source text.
* ``inherits`` edge ``A -> B`` when a base expression of ``A`` statically
  resolves, through imports and re-exports, to analysed class ``B``. The
  ``position`` attribute is the base's index in the ``class`` statement.
* ``nests`` edge ``Outer -> Inner`` for a class statement in a class body.

No other class-to-class relation (composition, usage, instantiation) is
inferred here: those need type information that plain syntax does not give.
"""
from __future__ import annotations

from .model import Graph
from .sources import Project


def build_class_graph(project: Project) -> Graph:
    graph = Graph("classes")
    for cid in sorted(project.classes):
        cls = project.classes[cid]
        members = [{"name": name, "kind": fn.kind}
                   for name, fn in sorted(cls.methods.items())]
        graph.add_node(cid, module=cls.module, qualname=cls.qualname,
                       decorators=list(cls.decorators), members=members,
                       external_bases=[name for origin, name in cls.bases
                                       if origin == "external"])
    for cid in sorted(project.classes):
        cls = project.classes[cid]
        for position, (origin, base) in enumerate(cls.bases):
            if origin == "internal":
                graph.add_edge(cid, base, "inherits", position=position)
        for inner in sorted(cls.nested.values()):
            graph.add_edge(cid, inner, "nests")
    return graph
