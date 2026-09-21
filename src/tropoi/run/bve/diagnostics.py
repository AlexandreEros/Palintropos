"""Compatibility import path; the implementation lives in :mod:`tropoi.representation.diagnostics.bve`.

This module object IS ``tropoi.representation.diagnostics.bve`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.representation.diagnostics.bve")
