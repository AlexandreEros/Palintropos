"""Conservative static call graph from the stdlib ``ast``.

Nodes are module-level functions, methods (``module.Class.method``), and
classes (as instantiation targets). Every call expression in a function body,
including calls inside nested functions, lambdas, and comprehensions (which
are attributed to the enclosing module-level function or method), is
classified as exactly one of:

``internal``   resolved to an analysed definition; produces edge(s) below;
``external``   resolved to a name outside the analysed package (e.g. numpy);
``builtin``    an unshadowed builtin name (``len``, ``isinstance``, ...);
``unresolved`` anything else.

Edge kinds (``caller -> callee``) and what they claim
----------------------------------------------------
``call``         ``f(...)``, ``mod.f(...)``, ``Class.method(...)`` where the
                 callee name is bound by a def or an import/re-export chain.
                 Statically certain up to rebinding at run time.
``instantiate``  ``C(...)`` where ``C`` resolves to an analysed class.
``constructor``  ``C -> C.__init__`` (found in ``C`` or its analysed bases);
                 what instantiation runs unless a metaclass/``__new__``
                 intervenes.
``self``         ``self.m(...)`` / ``cls.m(...)`` in a method of class ``C``:
                 the definition found by looking ``m`` up from ``C`` through
                 its analysed bases. Runtime dispatch depends on the receiver.
``super``        ``super().m(...)``: lookup starting after ``C``.
``override``     for a ``self``-call resolved to ``C.m``: every analysed
                 subclass of ``C`` that redefines ``m``. These are *possible*
                 targets (class-hierarchy analysis), not observed ones.

Lookups walk bases depth-first, left to right, and give up (``unresolved``)
at the first base that is not an analysed class, since it could define the
member. Not resolved: calls on arbitrary objects (``obj.run()``,
``self.backend.solve()``), calls through parameters, containers, callbacks,
``getattr``, decorator-returned wrappers, ``functools.partial``, and
anything bound dynamically. A name bound in a function (parameter,
assignment, loop target, ...) shadows module scope and makes calls through it
unresolved unless the binding is a function-local import.
"""
from __future__ import annotations

import ast
import builtins
from collections import Counter, defaultdict, deque

from .model import Graph
from .sources import FunctionInfo, Project, dotted, import_bindings

TRAVERSED_KINDS = ("call", "instantiate", "constructor", "self", "super", "override")
_BUILTINS = frozenset(dir(builtins))


def _local_bindings(project: Project, fn: FunctionInfo) -> dict[str, str | None]:
    """Names bound anywhere inside ``fn`` -> import target, or None if opaque."""
    info = project.modules[fn.module]
    bound: dict[str, set[str | None]] = defaultdict(set)
    globals_: set[str] = set()
    args = fn.node.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None:
            bound[arg.arg].add(None)
    for stmt in fn.node.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound[node.id].add(None)
            elif isinstance(node, ast.arg):
                bound[node.arg].add(None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound[node.name].add(None)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound[node.name].add(None)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                bound[node.name].add(None)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bound[node.rest].add(None)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                globals_.update(node.names)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                pairs, _stars = import_bindings(node, info.name, info.is_package)
                for name, target in pairs:
                    bound[name].add(target)
    local = {}
    for name, targets in bound.items():
        if name in globals_:
            continue
        local[name] = next(iter(targets)) if len(targets) == 1 else None
    return local


def _receiver_name(fn: FunctionInfo) -> str | None:
    """``self``/``cls`` parameter name of a method, if it has one."""
    if fn.class_id is None or fn.kind == "staticmethod":
        return None
    params = [*fn.node.args.posonlyargs, *fn.node.args.args]
    return params[0].arg if params else None


def build_call_graph(project: Project) -> tuple[Graph, dict[str, int]]:
    """Whole-package call graph plus call-site totals per outcome category."""
    graph = Graph("calls")
    for fid in sorted(project.functions):
        fn = project.functions[fid]
        graph.add_node(fid, type="function", kind=fn.kind, module=fn.module,
                       qualname=fn.qualname)
    for cid in sorted(project.classes):
        cls = project.classes[cid]
        graph.add_node(cid, type="class", kind="class", module=cls.module,
                       qualname=cls.qualname)

    edges: set[tuple[str, str, str]] = set()
    totals = Counter({"internal": 0, "external": 0, "builtin": 0, "unresolved": 0})
    for fid in sorted(project.functions):
        fn = project.functions[fid]
        local = _local_bindings(project, fn)
        receiver = _receiver_name(fn)
        calls = [n for stmt in fn.node.body for n in ast.walk(stmt) if isinstance(n, ast.Call)]
        for call in calls:
            outcome = _resolve_call(project, fn, call, local, receiver)
            if isinstance(outcome, str):
                totals[outcome] += 1
                continue
            totals["internal"] += 1
            for kind, target in outcome:
                edges.add((fid, target, kind))

    for source, target, kind in sorted(edges):
        graph.add_edge(source, target, kind)
    for cid in sorted(project.classes):
        found = project.lookup_member(cid, "__init__")
        if found is not None and found[0] == "function":
            graph.add_edge(cid, found[1], "constructor")
    return graph, dict(totals)


def _resolve_call(project: Project, fn: FunctionInfo, call: ast.Call,
                  local: dict[str, str | None], receiver: str | None):
    """Return a list of ``(kind, target)`` edges, or an outcome category string."""
    func = call.func
    # super().m(...)
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name) and func.value.func.id == "super"
            and "super" not in local):
        if fn.class_id is None:
            return "unresolved"
        found = project.lookup_member(fn.class_id, func.attr, start_after_self=True)
        if found is None or found[0] != "function":
            return "unresolved"
        return [("super", found[1])]

    name = dotted(func)
    if name is None:
        return "unresolved"
    head, _, rest = name.partition(".")

    # self.m(...) / cls.m(...)
    if receiver is not None and head == receiver:
        if not rest or "." in rest:
            return "unresolved"
        found = project.lookup_member(fn.class_id, rest)
        if found is None or found[0] != "function":
            return "unresolved"
        owner = project.functions[found[1]].class_id
        edges = [("self", found[1])]
        for sub in project.subclasses(owner):
            if rest in project.classes[sub].methods:
                edges.append(("override", project.classes[sub].methods[rest].id))
        return edges

    target = project.expand_in_scope(fn.module, name, local)
    if target is None:
        if (head not in local and head in _BUILTINS
                and head not in project.modules[fn.module].bindings):
            return "builtin" if not rest else "unresolved"
        return "unresolved"
    if not project.is_internal(target):
        return "external"
    found = project.resolve(target)
    if found is None:
        return "unresolved"
    kind, ident = found
    if kind == "function":
        return [("call", ident)]
    if kind == "class":
        return [("instantiate", ident)]
    return "unresolved"


def scoped(graph: Graph, roots: list[str], max_depth: int,
           kinds: tuple[str, ...] = TRAVERSED_KINDS) -> Graph:
    """Subgraph reachable from ``roots`` within ``max_depth`` edges (BFS).

    Node attribute ``depth`` is the BFS distance from the nearest root.
    Nodes at ``max_depth`` whose further internal edges were cut get
    ``truncated=True``.
    """
    missing = [r for r in roots if r not in graph.nodes]
    if missing:
        raise KeyError(f"call-graph roots not found: {missing}")
    depth = {r: 0 for r in roots}
    queue = deque(sorted(roots))
    while queue:
        node = queue.popleft()
        if depth[node] >= max_depth:
            continue
        for succ in graph.successors(node, kinds):
            if succ not in depth:
                depth[succ] = depth[node] + 1
                queue.append(succ)
    sub = Graph(graph.name)
    for node in sorted(depth):
        truncated = depth[node] >= max_depth and any(
            s not in depth for s in graph.successors(node, kinds))
        sub.add_node(node, **graph.nodes[node], depth=depth[node], truncated=truncated)
    for (s, t, k), attrs in sorted(graph.edges.items()):
        if k in kinds and s in depth and t in depth:
            sub.add_edge(s, t, k, **attrs)
    return sub
