"""Compatibility import paths for the 0.1 series.

The 2026-09 reorganization moved every implementation module under
``tropoi.spatial``, ``tropoi.temporal`` or ``tropoi.representation``
(docs/ARCHITECTURE.md). Each former module path is kept as a short module
that calls :func:`alias_module`, which binds the old name to the canonical
module object in ``sys.modules``. The old name therefore IS the new module:
classes and functions keep a single identity, module-level state is shared,
and monkeypatching either name patches both. Canonical modules never import
these paths. Removing them requires an explicit release decision.
"""
from __future__ import annotations

import importlib
import runpy
import sys


def alias_module(old_name: str, new_name: str) -> None:
    """Make ``old_name`` resolve to the already-canonical ``new_name``."""
    if old_name == "__main__":
        # ``python -m <old path>`` keeps running the module as a script.
        runpy.run_module(new_name, run_name="__main__", alter_sys=True)
        return
    sys.modules[old_name] = importlib.import_module(new_name)
