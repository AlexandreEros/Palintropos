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
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .calls import TRAVERSED_KINDS
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
