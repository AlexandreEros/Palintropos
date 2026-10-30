"""Canonical layout, layer direction, CPU imports and packaged resources."""
from __future__ import annotations

import ast
import importlib.resources
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "tropoi"


def _imports(path: pathlib.Path):
    """Every absolute import in a source file, TYPE_CHECKING blocks excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            skip.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


#: Layer -> the tropoi layers it may import (eagerly or lazily).
ALLOWED = {
    "tropoi.spatial": ("tropoi.spatial",),
    "tropoi.temporal": ("tropoi.spatial", "tropoi.temporal"),
    "tropoi.representation": ("tropoi.spatial", "tropoi.temporal",
                              "tropoi.representation"),
}
#: Deliberate, documented exceptions (docs/ARCHITECTURE.md).
EXCEPTIONS = {
    # the run-id hash consumes the stdlib-only configuration layer's
    # scientific subset
    ("tropoi.representation.archive.writer", "tropoi.run.bve.config"),
    # Snapshot.plot rebuilds model resources lazily through the solver
    # modules' own builders (orchestration owns model construction)
    ("tropoi.representation.visual.snapshot", "tropoi.run.bve.model"),
    ("tropoi.representation.visual.snapshot", "tropoi.run.swe.model"),
    ("tropoi.representation.visual.snapshot", "tropoi.run.pe.model"),
}


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_dependency_direction(layer):
    root = SRC / layer.split(".", 1)[1]
    violations = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(SRC.parent).with_suffix("")
        module = ".".join(p for p in rel.parts if p != "__init__")
        for target in _imports(path):
            if not target.startswith("tropoi."):
                continue
            if target.startswith(ALLOWED[layer]):
                continue
            if (module, target) not in EXCEPTIONS:
                violations.append((module, target))
    assert not violations, violations


def test_documented_cpu_only_modules_import_without_cupy_or_matplotlib():
    # docs/ARCHITECTURE.md "CPU and GPU requirements": these canonical
    # modules must load with CuPy blocked.
    modules = ["tropoi.spatial.truncation", "tropoi.spatial.environment",
               "tropoi.spatial.williamson5", "tropoi.spatial.modes",
               "tropoi.spatial.sigma_coordinate",
               "tropoi.temporal.integration", "tropoi.temporal.simulation",
               "tropoi.representation.archive",
               "tropoi.representation.archive.writer",
               "tropoi.run.bve.config", "tropoi.run.swe.config",
               "tropoi.run.pe.config",
               # custom planets and saved initial states (load/inspect/
               # validate/configure without CUDA)
               "tropoi.spatial.specification",
               "tropoi.representation.archive.initial_state",
               "tropoi.run.custom"]
    probe = (
        "import importlib, sys\n"
        "sys.modules['cupy'] = None\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "banned = [m for m in ('cupy', 'cupyx', 'matplotlib') "
        "if sys.modules.get(m) is not None]\n"
        "assert not banned, banned\n"
        "from tropoi.spatial.environment import PlanetaryParameters\n"
        "assert PlanetaryParameters.from_earth_like().radius > 0\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_kernel_sources_are_package_resources():
    sources = importlib.resources.files("tropoi.spatial.transforms.cuda")
    names = {entry.name for entry in sources.iterdir()}
    assert {"legendre.cu", "sh_matrix.cu"} <= names


def test_pyproject_packages_the_relocated_kernels():
    text = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"tropoi.spatial.transforms.cuda" = ["*.cu"]' in text


def test_library_does_not_import_command_interface():
    violations = []
    for folder in ("spatial", "temporal", "representation", "run"):
        for path in (SRC / folder).rglob("*.py"):
            for target in _imports(path):
                if target == "tropoi.cli" or target.startswith("tropoi.cli."):
                    violations.append((str(path.relative_to(SRC)), target))
    assert not violations, violations
