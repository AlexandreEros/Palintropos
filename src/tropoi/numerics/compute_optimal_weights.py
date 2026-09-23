"""Compatibility import path; the implementation lives in :mod:`tropoi.spatial.transforms.compute_optimal_weights`.

This module object IS ``tropoi.spatial.transforms.compute_optimal_weights`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.spatial.transforms.compute_optimal_weights")
