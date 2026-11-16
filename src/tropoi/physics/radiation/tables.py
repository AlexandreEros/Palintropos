"""Finding and verifying the k-tables a radiative configuration names.

A configuration refers to a table as ``{"file": "<name>.npz", "sha256":
"<content hash>"}``. The file is a :class:`~tropoi.physics.radiation.kdist.
KTable` archive; the hash is :meth:`KTable.content_sha256`, the identity of
its numbers (not of the ``.npz`` container). The hash is part of the
configuration, hence of ``config_sha256``: two runs that read different
opacities are different experiments, and a table that was silently
replaced refuses to load.

Tables are data, kept out of the source tree. A bare file name is looked
up, in order, in

1. each directory of ``TROPOI_OPACITY_PATH`` (``os.pathsep``-separated);
2. ``opacity_cache/tables`` under the current directory (where
   ``tools/radiation/opacity`` writes).

A name with a directory part is used as given. Any table that follows the
``KTable`` format takes this path, whatever produced it.

Import-light: NumPy only.
"""
from __future__ import annotations

import os
import pathlib
from typing import Mapping, Sequence

from tropoi.physics.radiation.kdist import KTable

__all__ = ["OPACITY_PATH_ENV", "DEFAULT_TABLE_DIR", "KTableError", "search_path",
           "resolve_table", "load_table", "table_spec", "check_premixed"]

OPACITY_PATH_ENV = "TROPOI_OPACITY_PATH"
DEFAULT_TABLE_DIR = pathlib.Path("opacity_cache") / "tables"
#: Relative tolerance on the gas constant of a premixed table (see check_premixed).
GAS_CONSTANT_TOLERANCE = 1.0e-6


class KTableError(ValueError):
    """A named table is missing, is not the one the configuration pinned,
    or does not belong to the configured gas."""


def search_path() -> list[pathlib.Path]:
    dirs = [pathlib.Path(d) for d in os.environ.get(OPACITY_PATH_ENV, "").split(os.pathsep)
            if d]
    return dirs + [pathlib.Path.cwd() / DEFAULT_TABLE_DIR]


def resolve_table(file: str) -> pathlib.Path:
    path = pathlib.Path(file)
    if path.parent != pathlib.Path("."):
        if not path.is_file():
            raise KTableError(f"k-table {file!r} does not exist")
        return path
    tried = search_path()
    for d in tried:
        if (d / path).is_file():
            return d / path
    raise KTableError(
        f"k-table {file!r} not found in {[str(d) for d in tried]}. Build or fetch it "
        f"(docs/radiation/OPACITY.md) or point {OPACITY_PATH_ENV} at its directory.")


def load_table(spec: Mapping) -> KTable:
    """Load the table of ``{"file": ..., "sha256": ...}`` and check its
    content hash. The hash is required: an unpinned table would let the
    same configuration hash mean different physics."""
    unknown = sorted(set(spec) - {"file", "sha256"})
    if unknown or "file" not in spec or "sha256" not in spec:
        raise KTableError(f"a table is {{'file': ..., 'sha256': ...}}; got {dict(spec)}")
    path = resolve_table(str(spec["file"]))
    table = KTable.load(path)
    found = table.content_sha256()
    if found != spec["sha256"]:
        raise KTableError(f"{path}: content hash {found} is not the configured "
                          f"{spec['sha256']}; the table was rebuilt from different inputs")
    return table


def table_spec(path) -> dict:
    """The configuration entry that pins the table stored at ``path``."""
    path = pathlib.Path(path)
    return {"file": path.name, "sha256": KTable.load(path).content_sha256()}


def check_premixed(tables: Sequence[KTable], mass_fractions: Mapping, *,
                   gas_constant: float) -> None:
    """Hold a premixed table to the gas it was mixed for.

    A premixed table is per kg of one fixed mixture, and any density-
    dependent part of it (collision-induced absorption) was evaluated with
    that mixture's gas constant. Used with another bulk gas it would be
    silently wrong, so the model's gas constant must equal the table's and
    the mass fraction must be exactly 1.
    """
    for table in tables:
        basis = table.metadata.get("mass_basis") or {}
        if basis.get("kind") != "premixed":
            continue
        q = mass_fractions.get(table.species)
        if q != 1.0:
            raise KTableError(f"{table.species!r} is a premixed table (per kg of mixture); "
                              f"its mass fraction must be 1.0, not {q!r}")
        r_table = float(basis["gas_constant"])
        if abs(gas_constant / r_table - 1.0) > GAS_CONSTANT_TOLERANCE:
            raise KTableError(
                f"{table.species!r} was premixed for a gas with R = {r_table:.6f} "
                f"J kg^-1 K^-1 ({basis.get('composition')}); the model's gas constant is "
                f"{gas_constant:.6f}. The radiative and the bulk composition must agree.")
