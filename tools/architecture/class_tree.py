"""Physical source containment and its two exhaustive views."""
from __future__ import annotations

import ast
import json
from base64 import b32encode
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path, PurePosixPath

from .mermaid import HEADER, label
from .sources import Project


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    kind: str
    parent: str | None
    path: str
    source: str
    line: int | None = None


class _ClassDefinitions(ast.NodeVisitor):
    """Index each definition site, with Python lexical names and tree parents."""

    def __init__(self, module: str, source: str, nodes: list[Node]):
        self.scope = module
        self.source = source
        self.nodes = nodes
        self.parent = f"m:{source}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        identity = f"c:{self.source}:{node.lineno}"
        path = f"{self.scope}.{node.name}"
        self.nodes.append(Node(identity, node.name, "class", self.parent, path,
                               self.source, node.lineno))
        previous = self.scope, self.parent
        self.scope, self.parent = path, identity
        self.generic_visit(node)
        self.scope, self.parent = previous

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self.scope
        self.scope = f"{previous}.{node.name}.<locals>"
        self.generic_visit(node)
        self.scope = previous

    visit_AsyncFunctionDef = visit_FunctionDef


def build_tree(project: Project) -> list[Node]:
    root = PurePosixPath(project.package)
    directories = {root}
    nodes: list[Node] = []
    for source, module in sorted(project.files.items()):
        file = PurePosixPath(source)
        directories.update(parent for parent in file.parents if parent.is_relative_to(root))
        qualified = f"{module.name}.__init__" if module.is_package else module.name
        nodes.append(Node(f"m:{source}", file.name, "module", f"d:{file.parent}", qualified,
                          source))
        _ClassDefinitions(module.name, source, nodes).visit(module.tree)

    folders = [
        Node(f"d:{directory}", directory.name, "package",
             None if directory == root else f"d:{directory.parent}",
             ".".join(directory.parts), directory.as_posix())
        for directory in sorted(directories)
    ]
    return folders + nodes


def mermaid_id(identity: str) -> str:
    # Short, stable IDs keep the complete chart within Mermaid's text limit.
    return "n" + b32encode(blake2b(identity.encode(), digest_size=8).digest()).decode().rstrip("=")


def mermaid_label(node: Node) -> str:
    if node.kind == "module":
        return label(node.source)
    return label(node.path if node.line is None else f"{node.path} (L{node.line})")


def render_mermaid(nodes: list[Node]) -> str:
    lines = [HEADER, "%% Source containment. Open class-atlas.html to explore subtrees.",
             "%% Large trees may need higher maxTextSize and maxEdges in the Mermaid host.",
             "flowchart LR",
             "    classDef packageNode fill:#eee9fc,color:#423781,stroke:#b9abe5",
             "    classDef moduleNode fill:#e5f2ff,color:#28567c,stroke:#a5c9e8",
             "    classDef classNode fill:#e3f5ea,color:#1f6441,stroke:#a3d3b5"]
    for node in nodes:
        lines.append(f"    {mermaid_id(node.id)}[{mermaid_label(node)}]:::{node.kind}Node")
    for node in nodes:
        if node.parent:
            lines.append(f"    {mermaid_id(node.parent)} --> {mermaid_id(node.id)}")
    return "\n".join(lines) + "\n"


def render_html(nodes: list[Node]) -> str:
    template = Path(__file__).with_name("class_atlas.html").read_text(encoding="utf-8")
    data = json.dumps([asdict(node) for node in nodes], ensure_ascii=False, indent=None)
    # Escape the HTML parser's script delimiters, including in paths from the filesystem.
    return template.replace("__TREE_DATA__", data.replace("<", r"\u003c"), 1)
