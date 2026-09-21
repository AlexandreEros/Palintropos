"""Compatibility import path; the implementation lives in :mod:`tropoi.temporal.tendencies.barotropic`.

This module object IS ``tropoi.temporal.tendencies.barotropic`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.temporal.tendencies.barotropic")
