"""Compatibility package: the planet facade moved to ``tropoi.spatial``.

``Planet`` lives in :mod:`tropoi.spatial.planet`, ``PlanetaryParameters`` (the
prescribed environment, CPU-only) in :mod:`tropoi.spatial.environment`, and
the terrain helpers in :mod:`tropoi.spatial.terrain`. These re-exports are the
same objects; each ``tropoi.planet.<module>`` path aliases its canonical module.
"""
from tropoi.spatial.planet import Planet
from tropoi.spatial.environment import PlanetaryParameters
from tropoi.spatial.terrain.elevation_data import ElevationData