"""Compatibility import path; the implementation lives in :mod:`tropoi.spatial.terrain.terrain_spectral`.

This module object IS ``tropoi.spatial.terrain.terrain_spectral`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.spatial.terrain.terrain_spectral")
