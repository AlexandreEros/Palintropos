"""Compatibility import path; the implementation lives in :mod:`tropoi.representation.visual.renderers`.

This module object IS ``tropoi.representation.visual.renderers`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.representation.visual.renderers")
