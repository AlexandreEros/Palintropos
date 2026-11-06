"""Fail when a function or method under ``src/tropoi`` is never referenced.

Dead code has been removed in bulk before (two large clean-up branches) and
then crept back, because nothing failed when a function lost its last caller.
This test is that check.

A function counts as referenced when its name appears anywhere under ``src/``,
``tests/``, ``tools/`` or ``examples/`` as an attribute (``obj.name``), a
bare name, an imported name, a keyword argument, or an identifier inside a
string literal (code generated into notebooks lives in strings). Docstrings do
not count: a function must not keep itself alive by being described.

Deliberately out of scope, so a failure is never noise:

* dunder methods, and properties / setters / abstract methods / overloads,
  which are public accessors rather than behaviour (an unused accessor is an
  API decision, not dead code);
* names exported in a module's ``__all__``;
* whether a *reference* is itself live. A helper called only from another
  dead helper, or only from a test, still passes. Matching is by name, so a
  method sharing its name with a used one elsewhere also passes. The check
  finds the common case (the last caller disappeared) with no false alarms; it
  is not a call-graph analysis.

If this fails: delete the function, or use it. If it is genuinely reached
only dynamically (``getattr`` with a computed name, a plugin entry point),
add it to ``ALLOWED`` with the reason.
"""
from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "tropoi"
SCAN_ROOTS = [REPO / "src", REPO / "tests", REPO / "tools", REPO / "examples"]

#: name -> why it is reached without a textual reference.
ALLOWED: dict[str, str] = {}

#: Sources carry non-ASCII (the operator comments), so every read names UTF-8:
#: ``read_text()`` would otherwise use the platform encoding and fail on Windows.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ACCESSOR_DECORATORS = {"property", "cached_property", "abstractmethod", "overload", "setter",
                        "getter", "deleter"}


def _python_files(root: pathlib.Path):
    if root.exists():
        yield from (p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _references() -> set[str]:
    names: set[str] = set()
    for root in SCAN_ROOTS:
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            docstrings = _docstring_nodes(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.alias):
                    names.add(node.name.split(".")[-1])
                    if node.asname:
                        names.add(node.asname)
                elif isinstance(node, ast.keyword) and node.arg:
                    names.add(node.arg)
                elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                      and id(node) not in docstrings):
                    names.update(_TOKEN.findall(node.value))
    return names


def _exported(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                return {str(x) for x in ast.literal_eval(node.value)}
            except (ValueError, SyntaxError):
                return set()
    return set()


def _decorator_names(fn) -> set[str]:
    out = set()
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        out.add(target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", ""))
    return out


def unreferenced_functions() -> list[tuple[str, int, str]]:
    referenced = _references()
    found = []
    for path in _python_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exported = _exported(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if (name.startswith("__") and name.endswith("__")) or name in exported:
                continue
            if _decorator_names(node) & _ACCESSOR_DECORATORS:
                continue
            if name not in referenced and name not in ALLOWED:
                found.append((str(path.relative_to(REPO)), node.lineno, name))
    return sorted(found)


def test_every_function_in_src_is_referenced_somewhere():
    dead = unreferenced_functions()
    assert not dead, (
        "no references found for these functions; delete them, use them, or add them to "
        "ALLOWED with a reason:\n" + "\n".join(f"  {p}:{ln}  {name}" for p, ln, name in dead))


def test_allowed_names_are_all_still_unreferenced():
    """An ALLOWED entry that is now referenced (or gone) is stale: remove it."""
    stale = [n for n in ALLOWED if n in _references()]
    assert not stale, f"ALLOWED entries that no longer need an exemption: {stale}"
