"""Find every ``${...}`` interpolation in the atlas template's JavaScript.

A small scanner, not a JavaScript parser: it skips quoted strings and nested
template literals while matching braces, which is all the template needs
(regular-expression literals there contain no braces or quotes inside an
interpolation). Used by ``test_arch_atlas_escaping.py`` (issue #62).
"""
from __future__ import annotations

import re


def _skip_quoted(text: str, i: int) -> int:
    """``text[i]`` is a quote; return the index after its closing quote."""
    q = text[i]
    i += 1
    while text[i] != q:
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _skip_template(text: str, i: int) -> int:
    """``text[i]`` is a backtick; return the index after the closing one."""
    i += 1
    while text[i] != "`":
        if text[i] == "\\":
            i += 2
        elif text.startswith("${", i):
            i = match_interpolation(text, i)
        else:
            i += 1
    return i + 1


def match_interpolation(text: str, start: int) -> int:
    """``text[start:start+2] == '${'``; return the index after its ``}``."""
    i, depth = start + 2, 1
    while depth:
        c = text[i]
        if c in "'\"":
            i = _skip_quoted(text, i)
            continue
        if c == "`":
            i = _skip_template(text, i)
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        i += 1
    return i


def _match_paren(text: str, i: int) -> int:
    """``text[i] == '('``; return the index of its ``)``."""
    depth = 0
    while True:
        c = text[i]
        if c in "'\"":
            i = _skip_quoted(text, i)
            continue
        if c == "`":
            i = _skip_template(text, i)
            continue
        depth += {"(": 1, ")": -1}.get(c, 0)
        if depth == 0:
            return i
        i += 1


def interpolations(script: str) -> list[str]:
    """Every interpolation expression, nested ones included, in source order."""
    return [script[m.start() + 2:match_interpolation(script, m.start()) - 1]
            for m in re.finditer(r"\$\{", script)]


def is_single_call(expr: str, names: set[str]) -> bool:
    """True when ``expr`` is exactly one call ``name(...)`` with ``name`` in ``names``."""
    m = re.match(r"([A-Za-z_$][\w$]*)\(", expr)
    return bool(m) and m.group(1) in names and _match_paren(expr, m.end() - 1) == len(expr) - 1


#: Local names that hold embedded data records in the template (``D`` is the
#: payload; the rest are the aliases its renderers bind records to).
DATA_ROOTS = ("D", "P", "A", "R", "e", "t", "b", "v", "st", "x", "n", "r", "c", "s", "u",
              "f", "p", "om", "ln", "sites", "env", "cap", "sy", "tm",
              "ci")  # an observation's static-counterpart call index (e.static)
_PATH = (r"(?<![\w$.'\"])(?:%s)(?:\.[A-Za-z_$][\w$]*|\[[^\]]*\])*(?![\w$(\[.])"
         % "|".join(DATA_ROOTS))


def _strip_calls(expr: str, names: set[str]) -> str:
    """Replace every call to a name in ``names`` (with its arguments) by ``_``."""
    out, i = [], 0
    for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\(", expr):
        if m.start() < i or m.group(1) not in names:
            continue
        out.append(expr[i:m.start()] + "_")
        i = _match_paren(expr, m.end() - 1) + 1
    return "".join(out) + expr[i:]


def raw_data_reads(expr: str, safe: set[str]) -> list[str]:
    """Data paths ``expr`` prints unescaped: the whole expression, or a ``+`` operand."""
    rest = _strip_calls(expr, safe)
    reads = []
    for m in re.finditer(_PATH, rest):
        path = m.group(0)
        if path.endswith((".length", ".size")):
            continue  # computed by JavaScript, always a number
        whole = rest.strip() == path
        concatenated = re.search(r"\+\s*$", rest[:m.start()]) or re.match(r"\s*\+", rest[m.end():])
        if whole or concatenated:
            reads.append(path)
    return reads
