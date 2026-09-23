"""Markdown renderers for ``README.md`` and ``metrics.md`` of a snapshot.

Only descriptive text: what was extracted, how, what was left out, and what
each number means. No interpretation of the measured values.
"""
from __future__ import annotations

from .calls import TRAVERSED_KINDS
from .metrics import DEFINITIONS, TOP_K

FILE_DESCRIPTIONS = {
    "README.md": "This file: provenance, extraction rules, exclusions, limitations, metric definitions.",
    "metrics.json": "All measurements, machine-readable.",
    "metrics.md": "The same measurements as tables, with definitions.",
    "modules.mmd": "Package-level import graph (modules collapsed onto their containing package).",
    "modules-cycles.mmd": "Module-level import graph restricted to modules in import cycles (nontrivial SCCs).",
    "classes.mmd": "Classes linked by inheritance between analysed classes, with their methods.",
    "graph/modules.json": "Full module import graph: every analysed module and import edge.",
    "graph/packages.json": "Package-aggregated import graph.",
    "graph/classes.json": "Every class with members, bases, and inheritance/nesting edges.",
    "graph/calls.json": "Whole-package static call graph (all resolvable edges).",
}


def _describe_file(name: str) -> str:
    if name in FILE_DESCRIPTIONS:
        return FILE_DESCRIPTIONS[name]
    if name.startswith("classes/"):
        return f"All classes of package `{name[len('classes/'):-len('.mmd')]}`, with methods."
    if name.startswith("calls/"):
        return "Scoped static call graph (roots and depth: see *Call graphs* below)."
    return ""


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value else "0"
    return str(value)


def _describe_rows(columns: dict[str, dict]) -> list[str]:
    """Table rows for one or more ``metrics.describe`` results side by side."""
    names = list(columns)
    rows = [f"| Metric | {' | '.join(names)} |", "|---|" + "---|" * len(names)]

    def row(label, getter):
        rows.append(f"| {label} | " + " | ".join(_fmt(getter(columns[n])) for n in names) + " |")

    row("Nodes \\|V\\|", lambda d: d["nodes"])
    row("Directed edges \\|E\\|", lambda d: d["edges"])
    row("Density", lambda d: d["density"])
    row("Weakly connected components", lambda d: d["weakly_connected_components"]["count"])
    row("WCC sizes", lambda d: ", ".join(map(str, d["weakly_connected_components"]["sizes"])))
    row("Strongly connected components", lambda d: d["strongly_connected_components"]["count"])
    row("Nontrivial SCCs", lambda d: d["nontrivial_sccs"]["count"])
    row("Nontrivial SCC sizes", lambda d: ", ".join(map(str, d["nontrivial_sccs"]["sizes"])) or "-")
    row("Nodes in cycles", lambda d: d["nodes_in_cycles"]["count"])
    row("Fraction of nodes in cycles", lambda d: d["nodes_in_cycles"]["fraction"])
    row("Sources (in-degree 0)", lambda d: d["sources"])
    row("Sinks (out-degree 0)", lambda d: d["sinks"])
    row("Reciprocal pairs", lambda d: d["reciprocal_pairs"]["count"])
    row("Reciprocity", lambda d: d["reciprocity"])
    for side in ("in_degree", "out_degree"):
        tag = "In" if side == "in_degree" else "Out"
        row(f"{tag}-degree min / median / max", lambda d, s=side: f"{d[s]['min']} / {_fmt(d[s]['median'])} / {d[s]['max']}")
        row(f"{tag}-degree mean", lambda d, s=side: d[s]["mean"])
        row(f"{tag}-degree quartiles", lambda d, s=side: ", ".join(_fmt(q) for q in d[s]["quartiles"]))
        row(f"{tag}-degree zeros", lambda d, s=side: d[s]["zeros"])
    row("Condensation nodes", lambda d: d["condensation"]["nodes"])
    row("Condensation edges", lambda d: d["condensation"]["edges"])
    row("Condensation sources / sinks", lambda d: f"{d['condensation']['sources']} / {d['condensation']['sinks']}")
    row("Condensation longest path (edges)", lambda d: d["condensation"]["longest_path_edges"])
    return rows


def _top_rows(title: str, top: dict) -> list[str]:
    lines = [f"**{title}** (ties at cutoff: {top['ties_at_cutoff']})", "",
             "| Node | Degree |", "|---|---|"]
    lines += [f"| `{e['node']}` | {e['degree']} |" for e in top["nodes"]]
    return lines + [""]


def _histogram(d: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d["histogram"].items())


def definitions_table() -> list[str]:
    lines = ["| Metric | Definition |", "|---|---|"]
    lines += [f"| `{k}` | {v.replace('|', chr(92) + '|')} |" for k, v in DEFINITIONS.items()]
    return lines


def render_metrics_md(metrics: dict, opts) -> str:
    mg = metrics["module_graph"]
    allm, imp = mg["all_imports"], mg["import_time"]
    pg = metrics["package_graph"]
    out = [
        f"# Architecture metrics: {opts.title}",
        "",
        "Descriptive graph measurements. No value is ranked as better or worse. "
        "Definitions are at the end; machine-readable values are in `metrics.json`.",
        "",
        "## Module import graph",
        "",
        "Nodes are analysed modules; a directed edge A -> B means A has at least one "
        "static import statement that resolves to module B. Two edge sets are measured:",
        "",
        "* **all imports**: every static import statement, wherever it appears;",
        "* **import-time**: only edges with at least one import statement at module "
        "scope (executed when the importing module is imported). Edges made only of "
        "function-scope or `if TYPE_CHECKING:` imports are dropped.",
        "",
        "Edge counts by scope set: " + ", ".join(f"`{k}`: {v}" for k, v in mg["edge_scopes"].items()) + ".",
        "",
        *_describe_rows({"all imports": allm, "import-time": imp}),
        "",
        f"In-degree histogram (all imports) {{degree: modules}}: {_histogram(allm['in_degree'])}",
        "",
        f"Out-degree histogram (all imports) {{degree: modules}}: {_histogram(allm['out_degree'])}",
        "",
        *_top_rows(f"Top {TOP_K} fan-in (all imports)", allm["top_fan_in"]),
        *_top_rows(f"Top {TOP_K} fan-out (all imports)", allm["top_fan_out"]),
        "**Nontrivial strongly connected components (all imports)**",
        "",
    ]
    for i, members in enumerate(allm["nontrivial_sccs"]["members"], 1):
        out.append(f"{i}. ({len(members)} modules) " + ", ".join(f"`{m}`" for m in members))
    if not allm["nontrivial_sccs"]["members"]:
        out.append("None.")
    out += ["", "**Reciprocal pairs (all imports)**", ""]
    out += [f"* `{a}` <-> `{b}`" for a, b in allm["reciprocal_pairs"]["pairs"]] or ["None."]
    out += [
        "",
        "## Package-aggregated import graph",
        "",
        "Every module is mapped to its containing package (a package's `__init__` "
        "to the package itself) and parallel edges merged; edges inside one package "
        f"are not self-loops but are counted separately: {pg['intra_package_module_edges']} "
        "module edges stay inside their package.",
        "",
        *_describe_rows({"packages": pg}),
        "",
        *_top_rows(f"Top {TOP_K} fan-in (packages)", pg["top_fan_in"]),
        *_top_rows(f"Top {TOP_K} fan-out (packages)", pg["top_fan_out"]),
    ]
    if pg["nontrivial_sccs"]["members"]:
        out += ["**Nontrivial SCCs (packages)**", ""]
        out += [f"{i}. " + ", ".join(f"`{m}`" for m in members)
                for i, members in enumerate(pg["nontrivial_sccs"]["members"], 1)]
        out.append("")
    cg, cl = metrics["call_graph"], metrics["class_graph"]
    out += [
        "## Class graph",
        "",
        f"* Classes: {cl['classes']}",
        f"* `inherits` edges between analysed classes: {cl['inherits_edges']}",
        f"* `nests` edges (class defined in a class body): {cl['nests_edges']}",
        "* Bases outside the analysed package: "
        + (", ".join(f"`{k}` x{v}" for k, v in cl["external_bases"].items()) or "none"),
        "",
        "## Call graph",
        "",
        f"* Nodes (functions, methods, classes): {cg['nodes']}",
        "* Edges by kind: " + ", ".join(f"`{k}` {v}" for k, v in cg["edges_by_kind"].items()),
        "* Call sites by resolution outcome: "
        + ", ".join(f"{k} {v}" for k, v in cg["call_sites"].items()),
        "",
        "| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in cg["scoped"].items():
        out.append(f"| `calls/{name}.mmd` | " + ", ".join(f"`{r}`" for r in s["roots"])
                   + f" | {s['max_depth']} | {s['nodes']} | {s['edges']} | {s['truncated_nodes']}"
                   f" | {s['reachable_nodes_unbounded']} |")
    out += ["", "## Definitions", "", *definitions_table(), ""]
    return "\n".join(out)


def render_readme(metrics: dict, opts, files: list[str], generated_marker: str,
                  frozen_marker: str) -> str:
    tool, source, inc = metrics["tool"], metrics["source"], metrics["inclusion"]
    out = [generated_marker]
    if opts.freeze:
        out.append(frozen_marker)
    out += [f"# Architecture snapshot: {opts.title}", ""]
    if opts.freeze:
        out += [
            "> **Frozen historical evidence.** These files record the structure of the "
            "source commit below. They are not regenerated from later commits and must "
            "not be edited. The generator refuses to overwrite this directory; "
            "`--check` verifies it (see *Reproducing*).",
            "",
        ]
    out += [
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        f"| Source commit | `{source['commit']}` |" if source["commit"] else
        "| Source commit | not recorded (working tree at generation time) |",
        f"| Analysed package | `{source['package']}` under `{source['src']}/` |",
        f"| Generator | `tools.architecture` {tool['version']}"
        + (f", commit `{tool['generator_commit']}`" if tool["generator_commit"] else "") + " |",
        f"| Toolchain | Python {tool['python']}, grimp {tool['grimp']}, networkx {tool['networkx']} |",
        f"| Command | `{opts.command}` |",
        "",
        "## Reproducing",
        "",
        "From the repository root, with the development requirements installed "
        "(`pip install -r requirements-dev.txt`):",
        "",
        "```bash",
        f"{opts.command}{' --check' if opts.freeze else ''}",
        "```",
        "",
    ]
    if opts.freeze:
        out += [
            "`--check` renders everything in memory and compares it byte for byte with "
            "this directory; it writes nothing. It must be run on a checkout whose "
            f"`{source['src']}/{source['package']}` equals the source commit (the "
            "generator verifies this) and with the generator commit above, since a "
            "later generator version may render differently.",
            "",
        ]
    out += ["## Files", "", "| File | Content |", "|---|---|"]
    out += [f"| `{f}` | {_describe_file(f)} |" for f in files]
    out += [
        "",
        "Generated files contain no timestamps, absolute paths, or source line numbers, "
        "so they change only when the extracted structure changes (the call-site totals "
        "in `metrics.json` are the exception: they count call expressions).",
        "",
        "## Inclusion and exclusion rules",
        "",
        "The rules are generic path and syntax rules; none names a particular subpackage.",
        "",
        f"1. Only the package `{source['package']}` under `{source['src']}/` is read. Tests, "
        "scripts, notebooks, and documentation are outside it and never analysed.",
        "2. A directory below the root is a subpackage if it contains `__init__.py`, or, "
        "lacking one, contains `*.py` files at any depth (an implicit namespace package, "
        "PEP 420, which Python imports all the same). `__pycache__` and dot-directories "
        "are skipped.",
        "3. Only `*.py` files are modules; other files (`*.bak`, `*.cu`, data) are ignored.",
        "4. Each module gets a syntactic category: `namespace` (a namespace package "
        "directory), `empty` (no statement besides a docstring), `reexport` (only imports, "
        "`__all__` assignments, and alias assignments `A = B`), or `code`. Categories are "
        "attributes, not filters, with one exception: a `namespace` or `empty` module that "
        "no analysed module imports is dropped from the module graph, because it has no "
        "code and no edges.",
        "5. Imports of modules outside the package are not graph edges; they are listed per "
        "module (`stdlib_imports`, `third_party_imports` in `graph/modules.json`).",
        "",
        f"Modules discovered: {inc['modules_discovered']}; in the module graph: "
        f"{inc['modules_in_graph']}. Categories: "
        + ", ".join(f"{k} {v}" for k, v in inc["module_categories"].items()) + ".",
        "",
        "Excluded by rule 4: " + (", ".join(f"`{m}`" for m in inc["excluded_modules"]) or "none") + ".",
        "",
        "`reexport` modules (kept): " + (", ".join(f"`{m}`" for m in inc["reexport_modules"]) or "none") + ".",
        "",
        "## Extraction rules",
        "",
        "### Import graph (grimp)",
        "",
        "* One edge A -> B when module A contains at least one `import` or "
        "`from ... import` statement that grimp resolves to module B of the package. "
        "`from pkg import name` resolves to the submodule `pkg.name` if one exists, "
        "otherwise to `pkg` itself.",
        "* Each edge records the scopes of its statements: `module` (runs when A is "
        "imported), `function` (inside a `def`, from the AST line spans of nested import "
        "statements), `type_checking` (under `if TYPE_CHECKING:`, found by rebuilding with "
        "grimp's `exclude_type_checking_imports=True` and taking the difference).",
        "* Not edges: dynamic imports (`importlib.import_module`, `__import__`, PEP 562 "
        "`__getattr__` tables), whose targets are strings. Their call sites are counted "
        "instead: " + (", ".join(f"`{m}` {n}" for m, n in inc["dynamic_import_sites"].items())
                       or "none") + ".",
        "* Safeguards: namespace subpackages are passed to grimp as additional roots "
        "(from the root alone grimp attributes imports of their modules to the root "
        "package); every edge grimp reports is cross-checked against the module names "
        "the import statement itself spells out (AST), and generation fails on a "
        "mismatch; the analysed package is never imported (asserted after extraction).",
        "* Not edges: the implicit execution of parent packages (`import a.b.c` runs "
        "`a/__init__` and `a/b/__init__`).",
        "* `modules.mmd` collapses modules onto their containing package; each edge label "
        "is the number of module edges merged into it. `modules-cycles.mmd` shows the "
        "modules of every nontrivial SCC; dashed edges there consist only of "
        "function-scope or `TYPE_CHECKING` imports.",
        "",
        "### Class graph (ast)",
        "",
        "* Every `class` statement at module scope or inside another class body; classes "
        "defined inside functions are not recorded.",
        "* Members are the functions defined directly in the class body, sorted by name, "
        "with kind `method`, `classmethod`, `staticmethod`, or `property` (from decorators). "
        "In the diagrams `$` marks class/static methods; properties have no parentheses.",
        "* `inherits` edge when a base expression resolves statically (through imports and "
        "re-exports) to an analysed class. Other bases are listed as external bases, by "
        "absolute dotted name when an import makes it known.",
        "* `nests` edge for a class statement inside a class body.",
        "* No composition, association, or usage relation is inferred: plain syntax does "
        "not determine them without type information.",
        "",
        "### Call graphs (ast)",
        "",
        "Each call expression inside a function or method body (nested functions, "
        "lambdas, and comprehensions are attributed to the enclosing function) is "
        "classified as `internal` (resolved to an analysed definition), `external` "
        "(resolved to a name outside the package), `builtin` (an unshadowed builtin), or "
        "`unresolved`. Totals: "
        + ", ".join(f"{k} {v}" for k, v in metrics["call_graph"]["call_sites"].items()) + ".",
        "",
        "| Edge kind | Syntax | What it claims |",
        "|---|---|---|",
        "| `call` | `f()`, `mod.f()`, `Class.method()` | The name is bound to this definition by a `def` or an import/re-export chain. |",
        "| `instantiate` | `C()` | `C` resolves to this analysed class. |",
        "| `constructor` | class -> `__init__` | The `__init__` found in the class or its analysed bases, which instantiation runs unless a metaclass or `__new__` intervenes. |",
        "| `super` | `super().m()` | The first `m` after the enclosing class in its analysed bases. |",
        "| `self` | `self.m()`, `cls.m()` | The `m` found from the enclosing class through its analysed bases. **Receiver-dependent**: a subclass instance may dispatch elsewhere. Drawn dashed. |",
        "| `override` | (derived) | Each analysed subclass that redefines the `m` of a `self` edge: a *possible* runtime target (class-hierarchy analysis), not an observed one. Drawn dashed. |",
        "",
        "Base lookups go depth-first, left to right (not the exact C3 order, which "
        "coincides for single inheritance) and give up at the first base that is not "
        "an analysed class, since that base might define the member.",
        "",
        "**Cannot be resolved statically, therefore absent:** calls on arbitrary objects "
        "(`obj.run()`, `self.backend.solve()`), calls through parameters, attributes, "
        "containers, and callbacks (including argparse `set_defaults(handler=...)` "
        "dispatch), `getattr`, decorator-returned wrappers, `functools.partial`, and any "
        "name rebound at run time. A name bound inside the function (parameter, "
        "assignment, loop target, ...) shadows the module-scope binding and makes calls "
        "through it unresolved, unless the binding is a function-local import. A missing "
        "edge therefore does not mean \"never called\".",
        "",
        f"Scoped graphs follow edges of kinds {', '.join(f'`{k}`' for k in TRAVERSED_KINDS)} "
        "breadth-first from their roots up to the stated depth. Nodes with further "
        "resolvable calls beyond the cut have a dashed border; roots have a thick border.",
        "",
        "| Graph | Roots | Depth |",
        "|---|---|---|",
    ]
    for name, s in metrics["call_graph"]["scoped"].items():
        out.append(f"| `calls/{name}.mmd` | " + ", ".join(f"`{r}`" for r in s["roots"])
                   + f" | {s['max_depth']} |")
    out += [
        "",
        "## Metric definitions",
        "",
        "Computed with networkx on the module import graph (all imports, and the "
        "import-time subset) and on the package-aggregated graph. They are descriptive: "
        "no value is ranked as better or worse and no combined score is formed. Values "
        "are in `metrics.md` / `metrics.json`; floats are rounded to 6 decimals.",
        "",
        *definitions_table(),
        "",
    ]
    return "\n".join(out)
