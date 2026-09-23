"""Fixtures for the ``tools.architecture`` tests.

``tools/`` is repository tooling, not part of the installed package, so the
repository root is put on ``sys.path`` here. The tests build small synthetic
packages under ``tmp_path``; they never import the analysed code.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("grimp")
pytest.importorskip("networkx")


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Create ``files`` ({relative path: source}) under ``root``."""
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return root


#: A synthetic package exercising every extraction rule. Package name is
#: unique so that it can never collide with an importable module.
FIXTURE_PACKAGE = "archfixture"
FIXTURE_FILES = {
    "archfixture/__init__.py": "",
    "archfixture/empty.py": '"""Only a docstring."""\n',
    "archfixture/notes.bak": "import archfixture.b\n",
    "archfixture/data/readme.txt": "not a package\n",
    "archfixture/__pycache__/stale.py": "import archfixture.b\n",
    "archfixture/.hidden/mod.py": "import archfixture.b\n",
    "archfixture/ns/tool.py": """
        def helper():
            return 1
    """,
    "archfixture/shim.py": """
        \"\"\"Compatibility alias.\"\"\"
        from archfixture.base import Base  # noqa: F401
        Alias = Base
        __all__ = ["Base", "Alias"]
    """,
    "archfixture/base.py": """
        import abc


        class Base(abc.ABC):
            def __init__(self, n):
                self.n = n

            def run(self):
                return self.step()

            def step(self):
                return 0

            @property
            def size(self):
                return self.n

            @staticmethod
            def make():
                return Base(1)

            @classmethod
            def build(cls):
                return cls.make()

            class Options:
                pass
    """,
    "archfixture/child.py": """
        from archfixture.shim import Alias
        from archfixture import base as base_mod


        class Child(Alias):
            def step(self):
                return super().step() + 1


        class Error(ValueError):
            pass


        class Grand(base_mod.Base):
            pass
    """,
    "archfixture/a.py": """
        import json
        import thirdpartylib
        from typing import TYPE_CHECKING

        import archfixture.b
        from . import c
        from archfixture.ns.tool import helper as h

        if TYPE_CHECKING:
            from archfixture.child import Child


        def local_import():
            from archfixture.d import deep
            return deep()


        def uses():
            archfixture.b.g()
            c.cfun()
            h()
            len([])
            json.dumps({})
            obj = object()
            obj.anything()
            return archfixture.child_missing()


        def shadowed(c):
            return c.cfun()


        def make_child():
            from archfixture.child import Child
            return Child(3).run()
    """,
    "archfixture/b.py": """
        def g():
            from archfixture.a import uses
            return uses
    """,
    "archfixture/c.py": """
        def cfun():
            return 2
    """,
    "archfixture/d.py": """
        def deep():
            return 3
    """,
}


@pytest.fixture
def make_tree():
    return write_tree


@pytest.fixture(scope="module")
def fixture_src(tmp_path_factory) -> Path:
    return write_tree(tmp_path_factory.mktemp("src"), FIXTURE_FILES)


@pytest.fixture(scope="module")
def project(fixture_src):
    from tools.architecture.sources import Project

    return Project(fixture_src, FIXTURE_PACKAGE)
