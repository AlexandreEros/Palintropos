"""Module import graph, extracted with grimp.

Facts recorded
--------------
* One node per analysed module (see :mod:`tools.architecture.sources` for the
  inclusion rules), with its path, syntactic category, and the external
  top-level packages it imports (split into standard library / third party).
* One ``imports`` edge ``A -> B`` whenever module ``A`` contains at least one
  ``import``/``from ... import`` statement that grimp resolves to module ``B``
  of the same package. Each edge keeps the scopes in which those statements
  occur:

  - ``module``        -- executed when ``A`` is imported;
  - ``function``      -- inside a function/method body (from the ``ast``
    line spans of import statements nested in a ``def``);
  - ``type_checking`` -- under ``if TYPE_CHECKING:`` (never executed at run
    time; detected by rebuilding with grimp's
    ``exclude_type_checking_imports=True`` and taking the difference).

Not recorded: dynamic imports (``importlib.import_module``, ``__import__``,
PEP 562 lazy ``__getattr__`` tables), because their targets are strings. The
number of such call sites per module is recorded as ``dynamic_import_calls``
so that the gap is visible. Importing ``a.b.c`` also executes ``a`` and
``a.b``; grimp does not add those implicit parent-package edges, and neither
does this tool.
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

from .model import Graph
from .sources import Project


def _owned(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


@contextlib.contextmanager
def _source_on_path(src_root: Path, package: str, stub_packages: list[str]):
    """Make ``package`` resolve to ``src_root`` for grimp without executing it.

    grimp locates a dotted root (a namespace subpackage) with
    ``importlib.util.find_spec``, which imports every parent package. Inert
    module objects carrying only ``__path__`` stand in for those parents while
    grimp runs, so no ``__init__`` of the analysed package is executed. On exit
    the stubs are removed and the absence of any real import is asserted.
    """
    if any(_owned(m, package) for m in sys.modules):
        raise RuntimeError(f"{package!r} is already imported; the analysis must not import it")
    src = Path(src_root).resolve()
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    stubs = {}
    for name in stub_packages:
        stub = types.ModuleType(name)
        stub.__path__ = [str(src.joinpath(*name.split(".")))]
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        stub.__spec__.submodule_search_locations = stub.__path__
        stubs[name] = stub
    try:
        spec = importlib.util.find_spec(package)
        locations = list(spec.submodule_search_locations or []) if spec else []
        expected = str((src / package).resolve())
        if [str(Path(p).resolve()) for p in locations] != [expected]:
            raise RuntimeError(f"{package!r} resolves to {locations}, expected [{expected}]")
        sys.modules.update(stubs)
        yield
    finally:
        for name, stub in stubs.items():
            if sys.modules.get(name) is stub:
                del sys.modules[name]
        sys.path.remove(str(src))
        importlib.invalidate_caches()
    executed = sorted(m for m in sys.modules if _owned(m, package))
    if executed:
        raise RuntimeError(f"analysed modules were imported during analysis: {executed}")


def _build(packages: list[str], exclude_type_checking: bool):
    import grimp

    return grimp.build_graph(*packages, include_external_packages=True,
                             exclude_type_checking_imports=exclude_type_checking,
                             cache_dir=None)


def build_import_graph(project: Project) -> Graph:
    # grimp does not descend into namespace subpackages from the root package
    # alone (it would attribute imports of their modules to the root package),
    # so the top-most ones are passed as additional packages.
    roots = [project.package] + [ns for ns in project.namespace_packages
                                 if ns.rpartition(".")[0] not in project.namespace_packages]
    parents = sorted({".".join(ns.split(".")[:i]) for ns in roots[1:]
                      for i in range(1, ns.count(".") + 1)})
    with _source_on_path(project.src_root, project.package, parents):
        full = _build(roots, exclude_type_checking=False)
        runtime = _build(roots, exclude_type_checking=True)

    internal = {m for m in full.modules if project.is_internal(m)}
    if internal != set(project.modules):
        raise RuntimeError(
            "grimp and the ast discovery disagree on the module set: "
            f"only grimp={sorted(internal - set(project.modules))}, "
            f"only ast={sorted(set(project.modules) - internal)}")

    stdlib = set(sys.stdlib_module_names)
    graph = Graph("modules")
    for name in sorted(project.modules):
        info = project.modules[name]
        external = sorted(m for m in full.find_modules_directly_imported_by(name)
                          if not project.is_internal(m))
        graph.add_node(
            name, path=info.path, is_package=info.is_package, category=info.category,
            dynamic_import_calls=info.dynamic_import_calls,
            stdlib_imports=[m for m in external if m in stdlib or m == "__future__"],
            third_party_imports=[m for m in external if m not in stdlib and m != "__future__"],
        )

    for importer in sorted(internal):
        for imported in sorted(full.find_modules_directly_imported_by(importer)):
            if imported not in internal or imported == importer:
                continue
            runtime_lines = {d["line_number"] for d in runtime.get_import_details(
                importer=importer, imported=imported)}
            scopes = set()
            for detail in full.get_import_details(importer=importer, imported=imported):
                candidates = project.modules[importer].import_candidates.get(
                    detail["line_number"], set())
                if imported not in candidates:
                    raise RuntimeError(
                        f"grimp resolved {importer}:{detail['line_number']} "
                        f"({detail['line_contents']!r}) to {imported}, which that "
                        f"statement does not name (candidates: {sorted(candidates)})")
                if detail["line_number"] not in runtime_lines:
                    scopes.add("type_checking")
                elif detail["line_number"] in project.modules[importer].function_import_lines:
                    scopes.add("function")
                else:
                    scopes.add("module")
            graph.add_edge(importer, imported, "imports", scopes=sorted(scopes))

    # The only exclusion derived from a category: a module without code that
    # nothing imports has no edges at all (e.g. an empty package marker).
    targets = {t for (_s, t, _k) in graph.edges}
    for name in sorted(graph.nodes):
        if graph.nodes[name]["category"] in ("empty", "namespace") and name not in targets:
            del graph.nodes[name]
    return graph


def excluded_modules(project: Project, graph: Graph) -> list[str]:
    return sorted(set(project.modules) - set(graph.nodes))


def import_time_subgraph(graph: Graph) -> Graph:
    """Subgraph keeping only edges with at least one ``module``-scope statement."""
    sub = Graph(f"{graph.name}-import-time")
    for node, attrs in graph.nodes.items():
        sub.add_node(node, **attrs)
    for (s, t, k), attrs in graph.edges.items():
        if "module" in attrs["scopes"]:
            sub.add_edge(s, t, k, **attrs)
    return sub


def containing_package(module: str, graph: Graph) -> str:
    """The package a module belongs to (a package ``__init__`` is its own)."""
    if graph.nodes.get(module, {}).get("is_package"):
        return module
    return module.rpartition(".")[0]


def aggregate_by_package(graph: Graph, project: Project) -> Graph:
    """Collapse modules onto their containing package.

    Edge attribute ``module_edges`` counts the module-level edges merged into
    each package-level edge. Intra-package edges become the node attribute
    ``internal_edges`` rather than self-loops.
    """
    packages = sorted({m for m, info in project.modules.items() if info.is_package})
    agg = Graph("packages")
    for pkg in packages:
        members = sorted(m for m in graph.nodes if containing_package(m, graph) == pkg)
        if members:
            agg.add_node(pkg, modules=len(members), internal_edges=0)
    for (s, t, _k) in sorted(graph.edges):
        ps, pt = containing_package(s, graph), containing_package(t, graph)
        if ps == pt:
            agg.nodes[ps]["internal_edges"] += 1
            continue
        edge = agg.add_edge(ps, pt, "imports")
        edge["module_edges"] = edge.get("module_edges", 0) + 1
    return agg
