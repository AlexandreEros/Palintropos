"""Species identity, and the packaged-data loader of ``tropoi.physics``.

A species here is an identity: a name, its atomic makeup and the molar mass
that follows from one table of atomic weights
(``tropoi/physics/data/species.toml``). Whether a species has thermodynamic
data (:mod:`tropoi.physics.thermodynamics`), optical records
(:mod:`tropoi.physics.rayleigh`) or absorption data (none yet) are separate
questions; each property consumer checks the coverage it actually needs and
refuses, by name, species it has no data for.

Every scientific dataset of this package is a TOML or CSV file under
``tropoi/physics/data/``, read through ``importlib.resources`` so installed
wheels work without a source checkout. :func:`load_dataset` parses a file
once, records its SHA-256 and schema string, and caches the result;
:func:`dataset_identities` reports every dataset loaded so far (and loads
the known ones on request) so run serialization can state exactly which
data produced a result.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import tomllib
from dataclasses import dataclass

__all__ = ["ATOMIC_WEIGHTS", "ATOMIC_WEIGHT_SOURCE", "DATASETS", "SPECIES",
           "Species", "dataset_identities", "load_dataset"]

#: The packaged datasets this module knows how to enumerate.
DATASETS = ("species.toml", "thermodynamics.toml", "rayleigh.toml",
            "mixtures.toml", "stars.toml")

_CACHE: dict[str, dict] = {}
_IDENTITY: dict[str, dict] = {}


def load_dataset(filename: str) -> dict:
    """Parse ``tropoi/physics/data/<filename>`` once (TOML), caching the
    result and recording the file's SHA-256 and schema string."""
    if filename not in _CACHE:
        raw = importlib.resources.files("tropoi.physics").joinpath(
            f"data/{filename}").read_bytes()
        doc = tomllib.loads(raw.decode("utf-8"))
        _IDENTITY[filename] = {"sha256": hashlib.sha256(raw).hexdigest(),
                               "schema": doc.get("schema")}
        _CACHE[filename] = doc
    return _CACHE[filename]


def dataset_identities(*, all_known: bool = True) -> dict[str, dict]:
    """``{filename: {"sha256": ..., "schema": ...}}`` of the packaged
    datasets (``all_known=True`` loads every known one first). Meant for run
    manifests, so a recorded result names the data it was computed from."""
    if all_known:
        for name in DATASETS:
            load_dataset(name)
    return {k: dict(v) for k, v in sorted(_IDENTITY.items())}


@dataclass(frozen=True)
class Species:
    """One gas: its name, atomic makeup and molar mass (kg/mol), computed
    from the packaged atomic weights, never transcribed per molecule."""

    name: str
    atoms: tuple[tuple[str, int], ...]
    molar_mass: float                     # kg/mol

    @property
    def molar_mass_g_per_mol(self) -> float:
        return self.molar_mass * 1000.0

    def formula_dict(self) -> dict[str, int]:
        return dict(self.atoms)


def _build_registry() -> tuple[dict[str, Species], dict[str, float], str]:
    doc = load_dataset("species.toml")
    weights = {k: float(v) for k, v in doc["atomic_weights"]["values"].items()}
    source = doc["atomic_weights"]["source"]
    registry: dict[str, Species] = {}
    for name, record in doc["species"].items():
        atoms = tuple((el, int(n)) for el, n in record["atoms"].items())
        unknown = [el for el, _ in atoms if el not in weights]
        if unknown:
            raise ValueError(f"species {name!r} uses elements {unknown} that "
                             "the atomic-weight table does not list")
        mass = sum(weights[el] * n for el, n in atoms) / 1000.0
        registry[name] = Species(name, atoms, mass)
    return registry, weights, source


SPECIES, ATOMIC_WEIGHTS, ATOMIC_WEIGHT_SOURCE = _build_registry()
