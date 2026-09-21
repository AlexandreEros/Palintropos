"""Visualization: figure data/specifications, renderers and per-core compositions.

Backend-neutral fields, normalization, declarative specs and timelines
(``fields``, ``normalization``, ``specs``, ``timeline``, ``renderers``,
``matplotlib_renderer``, ``complex_encoding``, ``grid_adapter``); the
per-core scientific compositions (``bve``, ``swe``, ``pe``,
``pe_snapshots``) and viewers; and ``snapshot``, the lazy adapter behind
``Snapshot.plot``.

This initializer imports nothing, and nothing in this package is imported
by archive inspection or host coefficient access: ``Snapshot.plot`` and the
runners import the modules they need on use. (Legacy paths:
``tropoi.viz`` and ``tropoi.run.{bve,swe,pe}.visualization``.)
"""
