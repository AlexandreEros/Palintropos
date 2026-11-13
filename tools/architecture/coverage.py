"""Documentation scope: what the generated views must cover, and the checks.

Three kinds of entry are discovered from the source, by generic rules that
name no particular subpackage:

* **Packages.** Every analysed package (a module's containing package; a
  package's ``__init__`` belongs to the package itself) gets its own view,
  ``packages/<package>.md``, holding all of its modules, classes and public
  functions. Coverage of packages, modules and classes is therefore total by
  construction, and :func:`rendered_problems` asserts it.
* **CLI handlers.** A keyword argument of a ``<parser>.set_defaults(...)``
  call whose value is a plain name resolving to an analysed function: the
  argparse idiom that binds a subcommand to the function it runs. Each one
  gets a scoped call graph ``calls/cli-<name>.mmd``; none can be left out.
* **Public Python entry points** (candidates). A module-level function that
  a name in some module's ``__all__`` resolves to (directly or through a
  re-export chain) and that no analysed function calls. Such a function is
  reachable only from outside the package: a script, a notebook, a test, a
  user. Static analysis cannot tell which of these matter, so each candidate
  must be *declared* in the scope file, either as a root of a named entry
  path (which then gets a call graph) or as an exclusion with a reason. An
  undeclared candidate fails generation, so a new public entry point cannot
  fall outside the documented scope silently.

The scope file (TOML; default ``docs/architecture/entry-points.toml``)
states expectations, like the layer declaration. Its rules:

* ``[[entry_path]]`` with ``name`` (``[a-z0-9-]+``, not starting ``cli-``),
  ``title``, ``roots`` (functions *or methods*: a declared path may start
  where the candidate rule cannot see, e.g. a method reached through an
  object) and an optional ``doc``: a repository-relative page that must
  exist and must mention ``calls/<name>.mmd``.
* ``[[excluded]]`` with ``function`` and ``reason``.
* Every candidate is declared exactly once; every declared root and
  exclusion exists in the source; an exclusion names a current candidate
  (otherwise it is stale). Violations are reported together.

Method calls through arbitrary objects, ``getattr`` and callbacks are not
followed (see :mod:`.calls`): a candidate is "uncalled" statically, not
necessarily unused.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .calls import TRAVERSED_KINDS
from . import class_tree
from .class_tree import Node, mermaid_id, mermaid_label
from .model import Graph
from .sources import Project, dotted

NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


class CoverageError(Exception):
    """The documentation scope is incomplete or stale; ``problems`` says how."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass
class EntryPath:
    name: str
    title: str
    roots: list[str]
    doc: str | None = None


@dataclass
class Scope:
    path: str | None = None          # repository-relative, as given
    sha256: str | None = None
    entry_paths: list[EntryPath] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)   # function -> reason
    problems: list[str] = field(default_factory=list)        # malformed declarations


def package_of(project: Project, module: str) -> str:
    """The package a module belongs to (a package's ``__init__``: itself)."""
    info = project.modules[module]
    return module if info.is_package else module.rpartition(".")[0]


def packages(project: Project) -> dict[str, list[str]]:
    """Every analysed package -> its own modules (not its subpackages')."""
    out: dict[str, list[str]] = {}
    for module in project.modules:
        out.setdefault(package_of(project, module), []).append(module)
    return {p: sorted(m) for p, m in sorted(out.items())}


def non_python_files(project: Project) -> dict[str, list[str]]:
    """Package -> files in its directory that are not ``*.py`` (package data,
    templates, ...): shipped with the package but outside static analysis."""
    out: dict[str, list[str]] = {}
    for pkg in packages(project):
        info = project.modules[pkg]
        rel = info.path[:-len("__init__.py")] if info.path.endswith("__init__.py") \
            else info.path
        directory = project.src_root / rel
        if not directory.is_dir():
            continue
        names = sorted(p.name for p in directory.iterdir()
                       if p.is_file() and p.suffix != ".py" and not p.name.startswith(".")
                       and p.suffix != ".pyc")
        if names:
            out[pkg] = names
    return out


# -- CLI handlers ----------------------------------------------------------
def _handler_name(function_id: str, project: Project) -> str:
    fn = project.functions[function_id]
    stem = fn.qualname.lstrip("_")
    stem = stem[4:] if stem.startswith("cmd_") else stem
    tail = fn.module.rsplit(".", 1)[-1]
    parts = ([] if tail == "main" else [tail]) + [stem]
    return "cli-" + "-".join(parts).replace("_", "-").lower()


def cli_handlers(project: Project) -> dict[str, str]:
    """``cli-<name>`` -> function id, for every ``set_defaults`` handler."""
    found: dict[str, str] = {}
    for module, info in project.modules.items():
        for node in ast.walk(info.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_defaults"):
                continue
            for keyword in node.keywords:
                name = dotted(keyword.value) if keyword.value is not None else None
                target = project._expand(module, name) if name else None
                resolved = project.resolve(target) if target else None
                if resolved and resolved[0] == "function":
                    fid = resolved[1]
                    found.setdefault(_handler_name(fid, project), fid)
    return dict(sorted(found.items()))


# -- public Python entry points --------------------------------------------
def _all_names(tree: ast.Module) -> list[str]:
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                try:
                    value = ast.literal_eval(stmt.value)
                except (ValueError, TypeError, SyntaxError):
                    return []
                return [v for v in value if isinstance(v, str)]
    return []


def exported_functions(project: Project) -> dict[str, list[str]]:
    """Module-level function id -> the ``module.name`` exports naming it."""
    out: dict[str, list[str]] = {}
    for module, info in project.modules.items():
        for name in _all_names(info.tree):
            found = project.resolve(f"{module}.{name}")
            if found and found[0] == "function" and \
                    project.functions[found[1]].class_id is None:
                out.setdefault(found[1], []).append(f"{module}.{name}")
    return {k: sorted(v) for k, v in sorted(out.items())}


def entry_candidates(project: Project, calls: Graph,
                     handlers: dict[str, str]) -> dict[str, list[str]]:
    """Exported module-level functions that no analysed function calls."""
    called = {t for (s, t, k) in calls.edges if k in TRAVERSED_KINDS and s != t}
    skip = set(handlers.values())
    return {f: names for f, names in exported_functions(project).items()
            if f not in called and f not in skip}


# -- the scope file --------------------------------------------------------
def load_scope(root: Path, rel: str | None) -> Scope:
    if rel is None:
        return Scope()
    path = root / rel
    if not path.is_file():
        return Scope(path=rel, problems=[f"scope file {rel} not found"])
    # Hash with LF line endings so a Windows checkout (CRLF) and CI (LF)
    # record the same identity for the same declarations.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    scope = Scope(path=rel, sha256=hashlib.sha256(data).hexdigest())
    try:
        doc = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        scope.problems.append(f"{rel}: not valid TOML ({exc})")
        return scope
    for i, item in enumerate(doc.get("entry_path", [])):
        name, roots = item.get("name"), item.get("roots")
        where = f"{rel}: entry_path[{i}]"
        if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name.startswith("cli-"):
            scope.problems.append(f"{where}: name must match [a-z0-9-]+ and not start "
                                  f"with 'cli-' (got {name!r})")
            continue
        if not isinstance(roots, list) or not roots or not all(isinstance(r, str) for r in roots):
            scope.problems.append(f"{where} ({name}): roots must be a non-empty list of names")
            continue
        scope.entry_paths.append(EntryPath(name, str(item.get("title", name)), list(roots),
                                           item.get("doc")))
    for i, item in enumerate(doc.get("excluded", [])):
        fn, reason = item.get("function"), item.get("reason")
        if not isinstance(fn, str) or not isinstance(reason, str) or not reason.strip():
            scope.problems.append(f"{rel}: excluded[{i}] needs 'function' and a non-empty "
                                  "'reason'")
            continue
        if fn in scope.excluded:
            scope.problems.append(f"{rel}: {fn} is excluded twice")
        scope.excluded[fn] = reason.strip()
    unknown = sorted(set(doc) - {"entry_path", "excluded"})
    if unknown:
        scope.problems.append(f"{rel}: unknown sections {unknown}")
    return scope


def scope_problems(scope: Scope, project: Project, calls: Graph,
                   candidates: dict[str, list[str]], root: Path) -> list[str]:
    """Everything that makes the declared scope incomplete or stale."""
    problems = list(scope.problems)
    names = [p.name for p in scope.entry_paths]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"entry path {dup!r} is declared more than once")
    rooted: dict[str, str] = {}
    for path in scope.entry_paths:
        for r in path.roots:
            if r not in calls.nodes:
                problems.append(f"entry path {path.name!r}: root {r} is not a function, "
                                "method or class in the source (stale declaration?)")
            rooted.setdefault(r, path.name)
        if path.doc is not None:
            page = root / path.doc
            if not page.is_file():
                problems.append(f"entry path {path.name!r}: its page {path.doc} is missing")
            elif f"calls/{path.name}.mmd" not in page.read_text(encoding="utf-8"):
                problems.append(f"entry path {path.name!r}: {path.doc} does not link "
                                f"calls/{path.name}.mmd")
    for fn in sorted(scope.excluded):
        if fn not in calls.nodes:
            problems.append(f"excluded {fn} is not in the source (stale exclusion)")
        elif fn not in candidates:
            problems.append(f"excluded {fn} is no longer an entry-point candidate "
                            "(now called inside the package, or no longer exported): "
                            "remove the exclusion")
        if fn in rooted:
            problems.append(f"{fn} is both excluded and a root of {rooted[fn]!r}")
    for fn in sorted(candidates):
        if fn not in rooted and fn not in scope.excluded:
            problems.append(
                f"undeclared public entry point {fn} (exported as "
                f"{', '.join(candidates[fn])}, called by no analysed function): add it to "
                f"the roots of an [[entry_path]] or to [[excluded]] with a reason in "
                f"{scope.path or 'the scope file'}")
    return problems


def rendered_problems(project: Project, rendered_classes: set[str],
                      rendered_modules: set[str], package_pages: set[str]) -> list[str]:
    """Discovered but not rendered (should never happen; checked anyway)."""
    problems = []
    for pkg in packages(project):
        if pkg not in package_pages:
            problems.append(f"package {pkg} has no view")
    problems += [f"class {c} is in no package view"
                 for c in sorted(set(project.classes) - rendered_classes)]
    problems += [f"module {m} is in no package view"
                 for m in sorted(set(project.modules) - rendered_modules)]
    return problems


def _expected_containment(project: Project) -> list[Node]:
    """Audit the files and ASTs without using the discovery or rendering indexes."""
    root = project.src_root / project.package
    expected = {}
    # Follow directory links, as discovery does through Path.is_dir() and iterdir().
    for directory, subdirs, filenames in os.walk(root, followlinks=True):
        subdirs[:] = [name for name in subdirs
                      if not name.startswith(".") and name != "__pycache__"]
        for name in filenames:
            file = Path(directory) / name
            if file.suffix != ".py" or not file.is_file():
                continue
            relative = file.relative_to(project.src_root)
            source = relative.as_posix()
            for folder in relative.parents:
                if not folder.parts:
                    break
                identity = f"d:{folder.as_posix()}"
                expected[identity] = Node(
                    identity, folder.name, "package",
                    None if len(folder.parts) == 1 else f"d:{folder.parent.as_posix()}",
                    ".".join(folder.parts), folder.as_posix())

            module_id = f"m:{source}"
            qualified = ".".join(relative.with_suffix("").parts)
            expected[module_id] = Node(module_id, name, "module",
                                       f"d:{relative.parent.as_posix()}", qualified, source)
            module = ".".join(relative.parent.parts if name == "__init__.py"
                              else relative.with_suffix("").parts)
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            parents = {child: parent for parent in ast.walk(tree)
                       for child in ast.iter_child_nodes(parent)}
            for cls in (node for node in parents if isinstance(node, ast.ClassDef)):
                parts = [cls.name]
                owner = module_id
                ancestor = parents[cls]
                while ancestor is not tree:
                    if isinstance(ancestor, ast.ClassDef):
                        parts.append(ancestor.name)
                        if owner == module_id:
                            owner = f"c:{source}:{ancestor.lineno}"
                    elif isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        parts.extend(("<locals>", ancestor.name))
                    ancestor = parents[ancestor]
                path = ".".join([module, *reversed(parts)])
                identity = f"c:{source}:{cls.lineno}"
                expected[identity] = Node(identity, cls.name, "class", owner, path, source,
                                          cls.lineno)
    # Only classes, their ancestors and the package root belong in the views.
    wanted = {f"d:{project.package}"}
    for node in expected.values():
        if node.kind == "class":
            ancestor = node
            while ancestor is not None:
                wanted.add(ancestor.id)
                ancestor = expected.get(ancestor.parent)
    return [node for node in expected.values() if node.id in wanted]


def _count_problems(expected: list, actual: list, artifact: str, item: str) -> list[str]:
    counts = Counter(actual)
    wanted = set(expected)
    return (
        [f"{artifact}: missing {item} {value}" for value in sorted(wanted - counts.keys())]
        + [f"{artifact}: duplicate {item} {value}" for value, count in sorted(counts.items())
           if count > 1]
        + [f"{artifact}: unexpected {item} {value}" for value in sorted(counts.keys() - wanted)]
    )


def containment_problems(project: Project, mermaid: str, html: str) -> list[str]:
    """Read back the serialized nodes and edges and compare with source facts."""
    expected = _expected_containment(project)
    diagram = "structure.mmd"
    declarations = []
    edges = []
    problems = []
    lines = mermaid.splitlines()
    if [line.strip() for line in lines if line.startswith("flowchart")] != ["flowchart LR"]:
        problems.append(f"{diagram}: expected one left-to-right flowchart")
    for line in lines:
        line = line.strip()
        if not line or line == "flowchart LR" or (line.startswith("%% ") and "%%{" not in line):
            continue
        if re.fullmatch(r"classDef (?:package|module|class)Node [\w:#,-]+", line):
            continue
        declaration = re.fullmatch(r'(\w+)\[("[^"]*")\]:::(\w+)Node', line)
        edge = re.fullmatch(r"(\w+) --> (\w+)", line)
        if declaration:
            declarations.append(declaration.groups())
        elif edge:
            edges.append(edge.groups())
        else:
            problems.append(f"{diagram}: unexpected statement {line}")
    expected_ids = [mermaid_id(n.id) for n in expected]
    if len(set(expected_ids)) != len(expected_ids):
        problems.append(f"{diagram}: source identities produce a Mermaid identifier collision")
    problems += _count_problems(expected_ids, [identity for identity, _, _ in declarations],
                                diagram, "node")
    by_id = {identity: (text, kind) for identity, text, kind in declarations}
    for node in expected:
        identity = mermaid_id(node.id)
        if identity in by_id and by_id[identity] != (mermaid_label(node), node.kind):
            problems.append(f"{diagram}: incorrect label or kind for {node.id}")

    parents: dict[str, list[str]] = {}
    for parent, child in edges:
        parents.setdefault(child, []).append(parent)
    for node in expected:
        actual_parents = parents.pop(mermaid_id(node.id), [])
        expected_parents = [mermaid_id(node.parent)] if node.parent else []
        if actual_parents != expected_parents:
            problems.append(f"{diagram}: wrong parent for {node.id}: {actual_parents}")
    problems.extend(f"{diagram}: unexpected edge to {child}" for child in sorted(parents))
    # Mermaid measures JavaScript string length, in UTF-16 code units.
    size = len(mermaid.encode("utf-16-le")) // 2
    if size > class_tree.MERMAID_MAX_TEXT_SIZE:
        problems.append(f"{diagram}: {size} characters exceed Mermaid's default maxTextSize "
                        f"of {class_tree.MERMAID_MAX_TEXT_SIZE}")
    if len(edges) > class_tree.MERMAID_MAX_EDGES:
        problems.append(f"{diagram}: {len(edges)} edges exceed Mermaid's default maxEdges "
                        f"of {class_tree.MERMAID_MAX_EDGES}")

    atlas = "class-atlas.html"
    payloads = re.findall(r'<script id="tree-data" type="application/json">\s*(.*?)\s*</script>',
                          html, re.DOTALL)
    if len(payloads) != 1:
        return problems + [f"{atlas}: expected exactly one tree-data payload"]
    try:
        data = json.loads(payloads[0])
        if not isinstance(data, list):
            raise ValueError("tree-data is not a list")
        nodes = [Node(**record) for record in data]
    except (ValueError, TypeError) as error:
        return problems + [f"{atlas}: invalid tree-data: {error}"]
    problems += _count_problems([n.id for n in expected], [n.id for n in nodes], atlas, "node")
    html_by_id = {node.id: node for node in nodes}
    for node in expected:
        actual = html_by_id.get(node.id)
        if actual is not None and actual != node:
            detail = "wrong parent" if actual.parent != node.parent else "incorrect node data"
            problems.append(f"{atlas}: {detail} for {node.id}")
    return problems
