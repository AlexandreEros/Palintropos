"""Source discovery, inclusion rules, and static name binding (stdlib ``ast``).

Inclusion rules (generic; nothing here names a particular subpackage):

1. Only the root package passed on the command line, found under the source
   root, is analysed. Tests, scripts, notebooks, and docs live outside it and
   are therefore never read.
2. A directory below the root is a subpackage if it contains ``__init__.py``
   (regular package) or, lacking one, contains ``*.py`` files at any depth
   (implicit namespace package, PEP 420, which Python imports all the same).
   ``__pycache__`` and dot-directories are skipped.
3. Only ``*.py`` files are modules. Other files (``*.bak``, ``*.cu``, data)
   are ignored.
4. Every module is classified by its syntax alone:

   * ``namespace`` -- an implicit namespace package (a directory, no file);
   * ``empty``     -- no statements other than an optional docstring;
   * ``reexport``  -- only imports, ``__all__`` assignments, and plain alias
     assignments (``A = B`` / ``A = m.B``); typical of compatibility shims and
     facade ``__init__`` modules;
   * ``code``      -- anything else.

   Categories are descriptive attributes. The only exclusion derived from them
   is applied by the import graph: a ``namespace`` or ``empty`` module that no
   analysed module imports is dropped, because it has no code and no edges.

The AST is used purely in memory; nothing in it is serialised.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

#: Marker for a name bound to something that is not a static alias.
OPAQUE = "<opaque>"


@dataclass
class FunctionInfo:
    id: str
    module: str
    qualname: str
    lineno: int
    kind: str  # function | method | classmethod | staticmethod | property
    class_id: str | None
    decorators: list[str]
    node: ast.FunctionDef | ast.AsyncFunctionDef = field(repr=False)


@dataclass
class ClassInfo:
    id: str
    module: str
    qualname: str
    lineno: int
    decorators: list[str]
    base_exprs: list[str]  # source text of each base, in order
    node: ast.ClassDef = field(repr=False)
    methods: dict[str, FunctionInfo] = field(default_factory=dict)
    nested: dict[str, str] = field(default_factory=dict)  # name -> class id
    bases: list[tuple[str, str]] = field(default_factory=list)  # (internal|external, name)


@dataclass
class ModuleInfo:
    name: str
    path: str  # posix path relative to the source root
    is_package: bool
    tree: ast.Module = field(repr=False)
    category: str = "code"
    #: module-scope name -> set of dotted targets (OPAQUE if not a static alias)
    bindings: dict[str, set[str]] = field(default_factory=dict)
    star_imports: list[str] = field(default_factory=list)
    dynamic_import_calls: int = 0
    #: source lines covered by import statements inside a function body
    function_import_lines: set[int] = field(default_factory=set)
    #: first line of each import statement -> absolute modules it may name
    import_candidates: dict[int, set[str]] = field(default_factory=dict)


def dotted(expr: ast.expr) -> str | None:
    """``a.b.c`` for a pure Name/Attribute chain, else ``None``."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def import_base(module: str, is_package: bool, level: int, target: str | None) -> str:
    """Absolute dotted name for ``from <level dots><target> import ...``."""
    if level == 0:
        return target or ""
    package = module if is_package else module.rpartition(".")[0]
    for _ in range(level - 1):
        package = package.rpartition(".")[0]
    return f"{package}.{target}" if target else package


def import_bindings(stmt: ast.Import | ast.ImportFrom, module: str,
                    is_package: bool) -> tuple[list[tuple[str, str]], list[str]]:
    """``(name, dotted target)`` pairs bound by an import, plus star-import bases."""
    pairs, stars = [], []
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            if alias.asname:
                pairs.append((alias.asname, alias.name))
            else:
                head = alias.name.split(".")[0]
                pairs.append((head, head))
    else:
        base = import_base(module, is_package, stmt.level, stmt.module)
        for alias in stmt.names:
            if alias.name == "*":
                stars.append(base)
            else:
                pairs.append((alias.asname or alias.name, f"{base}.{alias.name}"))
    return pairs, stars


def _decorator_names(node) -> list[str]:
    return [ast.unparse(d) for d in node.decorator_list]


def _method_kind(decorators: list[str]) -> str:
    names = {d.split("(")[0].rsplit(".", 1)[-1] for d in decorators}
    for kind in ("staticmethod", "classmethod", "property"):
        if kind in names:
            return kind
    if any(d.endswith((".setter", ".getter", ".deleter")) for d in decorators):
        return "property"
    return "method"


def _is_docstring(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def classify(tree: ast.Module) -> str:
    body = list(tree.body)
    if body and _is_docstring(body[0]):
        body = body[1:]
    if not body:
        return "empty"
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if len(names) == len(targets) and names == ["__all__"] * len(names):
                continue
            if (isinstance(stmt, ast.Assign) and len(names) == len(targets)
                    and stmt.value is not None and dotted(stmt.value) is not None):
                continue
        return "code"
    return "reexport"


def _module_scope_statements(body: list[ast.stmt]):
    """Statements executed at module scope, descending into if/try/with blocks."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.If, ast.Try, ast.With)) or type(stmt).__name__ == "TryStar":
            for block in ("body", "orelse", "finalbody"):
                yield from _module_scope_statements(getattr(stmt, block, []) or [])
            for handler in getattr(stmt, "handlers", []) or []:
                yield from _module_scope_statements(handler.body)


def _assigned_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _assigned_names(elt)]
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    return []


class Project:
    """All modules of one package plus a static symbol resolver."""

    def __init__(self, src_root: Path, package: str):
        self.src_root = Path(src_root)
        self.package = package
        self.modules: dict[str, ModuleInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.functions: dict[str, FunctionInfo] = {}
        self._discover()
        for info in self.modules.values():
            self._index_module(info)
        for cls in self.classes.values():
            self._resolve_bases(cls)

    # -- discovery -----------------------------------------------------
    def _discover(self) -> None:
        root = self.src_root / self.package
        if not (root / "__init__.py").is_file():
            raise FileNotFoundError(f"{root} is not a regular package (no __init__.py)")
        self.namespace_packages: list[str] = []
        stack = [(root, self.package)]
        while stack:
            directory, dotted_name = stack.pop()
            for entry in sorted(directory.iterdir(), key=lambda p: p.name):
                if entry.is_dir():
                    if entry.name.startswith(".") or entry.name == "__pycache__":
                        continue
                    child = f"{dotted_name}.{entry.name}"
                    if (entry / "__init__.py").is_file():
                        stack.append((entry, child))
                    elif any("__pycache__" not in p.parts for p in entry.rglob("*.py")):
                        self.namespace_packages.append(child)
                        rel = entry.relative_to(self.src_root).as_posix()
                        self.modules[child] = ModuleInfo(child, rel + "/", True,
                                                         ast.Module([], []), category="namespace")
                        stack.append((entry, child))
                elif entry.suffix == ".py" and entry.is_file():
                    is_package = entry.name == "__init__.py"
                    name = dotted_name if is_package else f"{dotted_name}.{entry.stem}"
                    text = entry.read_text(encoding="utf-8")
                    tree = ast.parse(text, filename=str(entry))
                    rel = entry.relative_to(self.src_root).as_posix()
                    self.modules[name] = ModuleInfo(name, rel, is_package, tree,
                                                    category=classify(tree))
        self.namespace_packages.sort()
        self.modules = dict(sorted(self.modules.items()))

    # -- indexing ------------------------------------------------------
    def _bind(self, info: ModuleInfo, name: str, target: str) -> None:
        info.bindings.setdefault(name, set()).add(target)

    def _index_module(self, info: ModuleInfo) -> None:
        for stmt in _module_scope_statements(info.tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                pairs, stars = import_bindings(stmt, info.name, info.is_package)
                for name, target in pairs:
                    self._bind(info, name, target)
                info.star_imports.extend(stars)
            elif isinstance(stmt, ast.ClassDef):
                cls = self._index_class(info, stmt, prefix="")
                self._bind(info, stmt.name, cls.id)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = FunctionInfo(f"{info.name}.{stmt.name}", info.name, stmt.name,
                                  stmt.lineno, "function", None, _decorator_names(stmt), stmt)
                self.functions.setdefault(fn.id, fn)
                self._bind(info, stmt.name, fn.id)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                value = getattr(stmt, "value", None)
                alias = (dotted(value) if isinstance(stmt, ast.Assign) and value is not None
                         else None)
                for target in targets:
                    for name in _assigned_names(target):
                        if alias is not None and isinstance(target, ast.Name):
                            self._bind(info, name, self._expand(info.name, alias) or OPAQUE)
                        else:
                            self._bind(info, name, OPAQUE)
        info.star_imports.sort()
        for node in ast.walk(info.tree):
            if isinstance(node, ast.Import):
                info.import_candidates.setdefault(node.lineno, set()).update(
                    alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = import_base(info.name, info.is_package, node.level, node.module)
                info.import_candidates.setdefault(node.lineno, set()).update(
                    {base} | {f"{base}.{alias.name}" for alias in node.names})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        info.function_import_lines.update(
                            range(inner.lineno, (inner.end_lineno or inner.lineno) + 1))
            if isinstance(node, ast.Call):
                name = dotted(node.func)
                if name == "__import__" or (
                        name is not None and self._expand(info.name, name) == "importlib.import_module"):
                    info.dynamic_import_calls += 1

    def _index_class(self, info: ModuleInfo, node: ast.ClassDef, prefix: str) -> ClassInfo:
        qualname = f"{prefix}{node.name}"
        cls = ClassInfo(f"{info.name}.{qualname}", info.name, qualname, node.lineno,
                        _decorator_names(node), [ast.unparse(b) for b in node.bases], node)
        self.classes.setdefault(cls.id, cls)
        cls = self.classes[cls.id]
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = _decorator_names(stmt)
                if stmt.name in cls.methods:  # e.g. property getter + setter
                    cls.methods[stmt.name].decorators.extend(decorators)
                    continue
                fn = FunctionInfo(f"{cls.id}.{stmt.name}", info.name, f"{qualname}.{stmt.name}",
                                  stmt.lineno, _method_kind(decorators), cls.id, decorators, stmt)
                cls.methods[stmt.name] = fn
                self.functions.setdefault(fn.id, fn)
            elif isinstance(stmt, ast.ClassDef):
                inner = self._index_class(info, stmt, prefix=f"{qualname}.")
                cls.nested[stmt.name] = inner.id
        return cls

    # -- resolution ----------------------------------------------------
    def is_internal(self, name: str) -> bool:
        return name == self.package or name.startswith(self.package + ".")

    def _expand(self, module: str, name: str) -> str | None:
        """Dotted ``name`` as written in ``module`` -> absolute dotted target.

        Uses the module-scope binding of the head segment. Returns ``None`` when
        the head is unbound, bound more than once, or bound opaquely.
        """
        head, _, rest = name.partition(".")
        targets = self.modules[module].bindings.get(head)
        if not targets or len(targets) != 1:
            return None
        (target,) = targets
        if target == OPAQUE:
            return None
        return f"{target}.{rest}" if rest else target

    def expand_in_scope(self, module: str, name: str, local: dict[str, str | None]) -> str | None:
        """Like :meth:`_expand` but a function-local binding shadows module scope."""
        head, _, rest = name.partition(".")
        if head in local:
            target = local[head]
            if target is None:
                return None
            return f"{target}.{rest}" if rest else target
        return self._expand(module, name)

    def resolve(self, name: str, _seen: frozenset = frozenset()) -> tuple[str, str] | None:
        """Absolute dotted name -> ``(kind, id)`` of an internal definition.

        ``kind`` is ``module``, ``class``, or ``function``. Follows static
        re-export chains (``from .x import y`` in a package ``__init__``) and
        member lookup through statically known internal base classes. Returns
        ``None`` for external names and anything not statically determinable.
        """
        if name in _seen or not self.is_internal(name):
            return None
        _seen = _seen | {name}
        parts = name.split(".")
        for cut in range(len(parts), 0, -1):
            prefix = ".".join(parts[:cut])
            if prefix in self.modules:
                break
        else:
            return None
        current: tuple[str, str] = ("module", prefix)
        for attr in parts[cut:]:
            current = self._member(current, attr, _seen)
            if current is None:
                return None
        return current

    def _member(self, obj: tuple[str, str], attr: str, seen: frozenset) -> tuple[str, str] | None:
        kind, ident = obj
        if kind == "module":
            info = self.modules[ident]
            targets = info.bindings.get(attr)
            if targets:
                if len(targets) != 1:
                    return None
                (target,) = targets
                if target == OPAQUE:
                    return None
                if target in self.classes:
                    return ("class", target)
                if target in self.functions:
                    return ("function", target)
                return self.resolve(target, seen)
            if f"{ident}.{attr}" in self.modules:
                return ("module", f"{ident}.{attr}")
            for star in info.star_imports:
                found = self.resolve(f"{star}.{attr}", seen)
                if found is not None:
                    return found
            return None
        if kind == "class":
            return self.lookup_member(ident, attr)
        return None

    def mro(self, class_id: str) -> tuple[list[str], bool]:
        """Depth-first, left-to-right internal ancestry of a class.

        Returns ``(classes, complete)`` where ``complete`` is False if an
        external or unresolved base was met (its members are unknown). This is
        not Python's C3 linearisation; for single inheritance the two agree.
        """
        order: list[str] = []
        complete = True

        def visit(cid: str) -> None:
            nonlocal complete
            if cid in order:
                return
            order.append(cid)
            for origin, base in self.classes[cid].bases:
                if origin == "internal":
                    visit(base)
                elif base not in ("object",):
                    complete = False

        visit(class_id)
        return order, complete

    def lookup_member(self, class_id: str, attr: str, start_after_self: bool = False
                      ) -> tuple[str, str] | None:
        """Find ``attr`` in a class or its internal bases, conservatively.

        Stops (returns ``None``) at the first external/unresolved base met
        before a definition is found, because that base might define ``attr``.
        """
        seen: set[str] = set()

        def visit(cid: str, skip: bool):
            if cid in seen:
                return None, True
            seen.add(cid)
            cls = self.classes[cid]
            if not skip:
                if attr in cls.methods:
                    return ("function", cls.methods[attr].id), True
                if attr in cls.nested:
                    return ("class", cls.nested[attr]), True
            for origin, base in cls.bases:
                if origin == "internal":
                    found, ok = visit(base, False)
                    if found is not None or not ok:
                        return found, ok
                elif base != "object":
                    return None, False
            return None, True

        found, _ok = visit(class_id, start_after_self)
        return found

    def _resolve_bases(self, cls: ClassInfo) -> None:
        for expr in cls.node.bases:
            name = dotted(expr)
            target = self._expand(cls.module, name) if name else None
            found = self.resolve(target) if target else None
            if found is not None and found[0] == "class":
                cls.bases.append(("internal", found[1]))
            else:
                cls.bases.append(("external", target or ast.unparse(expr)))

    def subclasses(self, class_id: str) -> list[str]:
        """All analysed classes that have ``class_id`` among their internal ancestors."""
        return sorted(c for c in self.classes
                      if c != class_id and class_id in self.mro(c)[0])
