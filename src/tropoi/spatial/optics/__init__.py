"""Gas optics, explicit atmospheric composition and 1-D columns (CPU only).

A prescribed-world description, like ``environment`` or ``terrain``: it
holds sourced optical data and the explicit gas mixture a user states. No
solver reads it yet. It lives in the spatial layer so that a future
radiative-transfer tendency (``temporal``) can import the same tables and
composition the sky view (``representation.visual.sky``) uses.

* :mod:`.constants` -- exact SI constants and IUPAC atomic weights;
* :mod:`.rayleigh` -- per-species refractive index and King factor,
  Rayleigh cross sections (each species carries its citation);
* :mod:`.composition` -- :class:`Composition` (explicit mole fractions)
  and the report comparing its gas constant with a run's ``r_dry``;
* :mod:`.column` -- :class:`AtmosphericColumn`, a horizontally uniform
  column (altitude <-> pressure, number density), from saved native levels
  or from a labelled hypothetical profile;
* :mod:`.star` -- a blackbody illuminating star;
* :mod:`.colour` -- the CIE 1931 2-degree colour-matching functions
  (packaged verbatim from the CIE dataset).

NumPy only. Never CuPy. Composition never changes solver behaviour: a
mixture is never inferred from ``r_dry`` or a molar mass, and ``r_dry`` is
never replaced by a value derived from a mixture.
"""
from tropoi.spatial.optics.column import AtmosphericColumn, ColumnError
from tropoi.spatial.optics.composition import (COMPOSITION_PRESETS,
                                               Composition,
                                               CompositionError,
                                               gas_constant_report)
from tropoi.spatial.optics.rayleigh import SPECIES, rayleigh_cross_section
from tropoi.spatial.optics.star import Star

__all__ = ["AtmosphericColumn", "COMPOSITION_PRESETS", "ColumnError",
           "Composition", "CompositionError", "SPECIES", "Star",
           "gas_constant_report", "rayleigh_cross_section"]
