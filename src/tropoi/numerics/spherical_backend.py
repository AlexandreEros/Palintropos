"""Compatibility import path; the implementation lives in :mod:`tropoi.spatial.spherical_backend`.

This module object IS ``tropoi.spatial.spherical_backend`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.spatial.spherical_backend")
