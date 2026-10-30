"""Saved prepared terrain: resolved elevation coefficients and their record.

A prepared terrain is what ``tropoi.run.custom.prepare_planet`` resolved and
what ``PreparedPlanet.plot()`` / ``view()`` showed: surface-elevation
coefficients on one discretization, with the specification that says where
they came from. Saving it lets a later run use exactly the inspected terrain
(``tropoi run pe --terrain DIR``); a PE run also copies it into its capsule
under ``terrain/``, the way a custom SWE run keeps ``initial_state/``.

A saved directory holds

* ``terrain.json`` -- format, version, label, the planet and discretization
  specifications the terrain was resolved for, the topography
  specification (a preset, or an ``imported`` provenance record), the band
  limit (``l_cut``: the PE dealiased product cut, or ``null`` for the full
  truncation), the coefficient file's shape and SHA-256, and an identity
  digest over all of these;
* ``topography_elevation_lm.npy`` -- little-endian complex128
  ``(L+1, L+1)`` elevation coefficients (metres), the same file name and
  conventions as a saved SWE initial state's terrain.

Writes are staged and installed atomically exactly like initial states;
loading verifies every recorded property, including the coefficient
SHA-256 and that nothing lies above ``l_cut``. Host only; never imports
CuPy.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from tropoi.representation.archive.initial_state import (
    TOPOGRAPHY_FIELD, TOPOGRAPHY_FILE, InitialStateError, _canonical_json,
    _check_block, _install, _load_array, _reject_constant, _require_mapping,
    _software, _write_npy, array_digest)
from tropoi.spatial.modes import SpectralModes
from tropoi.spatial.specification import (DiscretizationSpec, PlanetSpec,
                                          SpecificationError, TopographySpec)

__all__ = [
    "PREPARED_TERRAIN_FORMAT",
    "PREPARED_TERRAIN_FORMAT_VERSION",
    "PreparedTerrainError",
    "SavedTerrain",
    "TERRAIN_METADATA_FILE",
    "is_prepared_terrain",
    "load_prepared_terrain",
    "save_prepared_terrain",
]

PREPARED_TERRAIN_FORMAT = "tropoi.prepared_terrain"
PREPARED_TERRAIN_FORMAT_VERSION = 1
TERRAIN_METADATA_FILE = "terrain.json"
_IDENTITY_FORMAT = "tropoi.prepared_terrain.identity"
_DOCUMENT_KEYS = {"format", "version", "label", "digest", "planet",
                  "discretization", "topography", "band_limit",
                  "coefficients", "created_utc", "software"}


class PreparedTerrainError(InitialStateError):
    """A prepared terrain is invalid, or a saved one cannot be read."""


@dataclass(frozen=True, eq=False)
class SavedTerrain:
    """A prepared terrain as stored on disk (host arrays, verified)."""

    planet: PlanetSpec
    discretization: DiscretizationSpec
    topography: TopographySpec
    topography_lm: np.ndarray
    l_cut: int | None = None
    label: str = ""
    path: pathlib.Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.planet, PlanetSpec) or not isinstance(
                self.discretization, DiscretizationSpec) or not isinstance(
                self.topography, TopographySpec):
            raise PreparedTerrainError(
                "planet, discretization and topography must be "
                "PlanetSpec, DiscretizationSpec and TopographySpec")
        if self.topography.is_flat:
            raise PreparedTerrainError(
                "a flat bottom has no terrain to save")
        n = self.discretization.lmax + 1
        lm = np.asarray(self.topography_lm)
        if lm.dtype != np.complex128 or lm.shape != (n, n):
            raise PreparedTerrainError(
                f"terrain coefficients must be complex128 of shape {(n, n)} "
                f"for lmax={self.discretization.lmax}, got {lm.dtype} "
                f"{lm.shape}")
        lm = np.array(lm, copy=True)
        lm.flags.writeable = False
        SpectralModes(lm, TOPOGRAPHY_FIELD).validate()
        if not np.any(lm):
            raise PreparedTerrainError("terrain coefficients are all zero")
        if self.l_cut is not None:
            cut = int(self.l_cut)
            if not 1 <= cut <= self.discretization.lmax:
                raise PreparedTerrainError(
                    f"l_cut must be in [1, {self.discretization.lmax}], got "
                    f"{self.l_cut}")
            if np.any(lm[cut + 1:, :]) or np.any(lm[:, cut + 1:]):
                raise PreparedTerrainError(
                    f"terrain has content above its band limit l_cut={cut}")
            object.__setattr__(self, "l_cut", cut)
        object.__setattr__(self, "topography_lm", lm)
        object.__setattr__(self, "label", str(self.label or ""))

    @property
    def digest(self) -> str:
        """Identity: specifications, band limit and coefficient bytes."""
        identity = {
            "format": _IDENTITY_FORMAT,
            "planet": self.planet.to_dict(),
            "discretization": self.discretization.to_dict(),
            "topography": self.topography.to_dict(),
            "l_cut": self.l_cut,
            "elevation_lm": array_digest(self.topography_lm),
        }
        return "sha256:" + hashlib.sha256(
            _canonical_json(identity).encode("utf-8")).hexdigest()

    def to_document(self) -> dict:
        n = self.discretization.lmax + 1
        return {
            "format": PREPARED_TERRAIN_FORMAT,
            "version": PREPARED_TERRAIN_FORMAT_VERSION,
            "label": self.label,
            "digest": self.digest,
            "planet": self.planet.to_dict(),
            "discretization": self.discretization.to_dict(),
            "topography": self.topography.to_dict(),
            "band_limit": {
                "l_cut": self.l_cut,
                "meaning": ("coefficients above l_cut are exactly zero (the "
                            "primitive-equation dealiased product cut)"
                            if self.l_cut is not None else
                            "full truncation (shallow-water use)")},
            "coefficients": {"coefficient_file": TOPOGRAPHY_FILE,
                             "shape": [n, n],
                             "sha256": array_digest(self.topography_lm),
                             "units": TOPOGRAPHY_FIELD.units,
                             "description": TOPOGRAPHY_FIELD.description},
        }


def is_prepared_terrain(path) -> bool:
    return (pathlib.Path(path) / TERRAIN_METADATA_FILE).is_file()


def save_prepared_terrain(terrain: SavedTerrain, path, *,
                          overwrite: bool = False) -> pathlib.Path:
    """Write ``terrain`` to the directory ``path`` (staged, atomic)."""
    if not isinstance(terrain, SavedTerrain):
        raise PreparedTerrainError(
            f"expected a SavedTerrain, got {type(terrain).__name__}")
    target = pathlib.Path(path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True to replace it")
        if not is_prepared_terrain(target):
            raise PreparedTerrainError(
                f"refusing to overwrite {target}: it is not a saved prepared "
                "terrain directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    document = terrain.to_document()
    document["created_utc"] = datetime.now(timezone.utc).isoformat()
    document["software"] = _software()
    text = json.dumps(document, indent=2, allow_nan=False)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_npy(staging / TOPOGRAPHY_FILE, terrain.topography_lm)
        with open(staging / TERRAIN_METADATA_FILE, "w", encoding="utf-8",
                  newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            shutil.rmtree(staging)
        raise
    _install(staging, target)
    return target


def load_prepared_terrain(path) -> SavedTerrain:
    """Read and verify a saved prepared terrain (host only)."""
    directory = pathlib.Path(path)
    meta = directory / TERRAIN_METADATA_FILE
    if not meta.is_file():
        raise PreparedTerrainError(
            f"{directory} is not a saved prepared terrain (no "
            f"{TERRAIN_METADATA_FILE})")
    try:
        document = json.loads(meta.read_text(encoding="utf-8"),
                              parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise PreparedTerrainError(f"malformed {meta}: {err}") from err
    document = _require_mapping(document, TERRAIN_METADATA_FILE)
    if document.get("format") != PREPARED_TERRAIN_FORMAT:
        raise PreparedTerrainError(
            f"{meta} declares format {document.get('format')!r}, expected "
            f"{PREPARED_TERRAIN_FORMAT!r}")
    if document.get("version") != PREPARED_TERRAIN_FORMAT_VERSION:
        raise PreparedTerrainError(
            f"{meta} has format version {document.get('version')!r}; this "
            f"reader implements {PREPARED_TERRAIN_FORMAT_VERSION}")
    unknown = set(document) - _DOCUMENT_KEYS
    if unknown:
        raise PreparedTerrainError(
            f"{meta} has unknown key(s) {sorted(unknown)}")
    band = _require_mapping(document.get("band_limit"), "band_limit block")
    _check_block(band, "band_limit block", {"l_cut": None, "meaning": None})
    block = _require_mapping(document.get("coefficients"),
                             "coefficients block")
    _check_block(block, "coefficients block", {
        "coefficient_file": None, "shape": None, "sha256": None,
        "units": TOPOGRAPHY_FIELD.units, "description": None})
    try:
        terrain = SavedTerrain(
            PlanetSpec.from_dict(document.get("planet")),
            DiscretizationSpec.from_dict(document.get("discretization")),
            TopographySpec.from_dict(document.get("topography")),
            _load_array(directory, block, TOPOGRAPHY_FILE, "terrain"),
            l_cut=band.get("l_cut"), label=document.get("label") or "",
            path=directory)
    except PreparedTerrainError:
        raise
    except (SpecificationError, InitialStateError) as err:
        raise PreparedTerrainError(f"{meta}: {err}") from err
    if terrain.digest != document.get("digest"):
        raise PreparedTerrainError(
            f"{meta}: the recorded digest {document.get('digest')!r} does "
            f"not match the content ({terrain.digest})")
    return terrain


def summary(terrain: SavedTerrain) -> Mapping[str, Any]:
    """Small JSON-able summary (inspect output)."""
    params = terrain.topography.parameters
    return {"label": terrain.label, "preset": terrain.topography.preset,
            "source_name": params.get("source_name"),
            "lmax": terrain.discretization.lmax, "l_cut": terrain.l_cut,
            "digest": terrain.digest}
