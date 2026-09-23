"""Compatibility import path; the implementation lives in :mod:`tropoi.spatial.transforms.cuda.cuda_utils`.

This module object IS ``tropoi.spatial.transforms.cuda.cuda_utils`` (aliased in ``sys.modules``), so
classes, functions and module state are identical under both names.
"""
from tropoi._compat import alias_module

alias_module(__name__, "tropoi.spatial.transforms.cuda.cuda_utils")
