"""Identity of the native substrate implementation: which sources, which
compile flags.

Loaded by ``setup.py`` by file path, without importing ``tropoi``, so it
uses only the standard library. The same function computes the digest at
build time (it is compiled into the CPU library) and at run time (from the
installed headers and drivers), and a mismatch refuses the library, so a
stale or foreign build is never used.
"""
from __future__ import annotations

import hashlib
import pathlib

#: Host compile flags. -ffp-contract=off is part of the arithmetic contract
#: (pt/portable.h): no product is fused into an FMA.
HOST_FLAGS = ("-O2", "-std=c++14", "-ffp-contract=off", "-fno-fast-math")
#: NVRTC options for the device driver: CuPy's defaults plus C++14 (the
#: include paths are added at run time and are not part of the identity).
DEVICE_OPTIONS = ("--std=c++14",)

TROPOI = pathlib.Path(__file__).resolve().parents[2]
#: The include roots of the native substrate, in layer order.
INCLUDE_DIRS = (TROPOI / "numerics" / "native", TROPOI / "physics" / "native",
                TROPOI / "temporal" / "native")
DRIVERS = TROPOI / "temporal" / "native" / "drivers"


def source_files() -> list:
    """Every header and driver the native substrate is built from, sorted by
    their path relative to the tropoi package."""
    files = [p for root in INCLUDE_DIRS for p in root.rglob("*.h")]
    files += [p for p in DRIVERS.iterdir() if p.suffix in (".h", ".cu", ".cpp")]
    return sorted(set(files), key=lambda p: p.relative_to(TROPOI).as_posix())


def digest(flags=HOST_FLAGS) -> str:
    """SHA-256 over the sources (path and content) and ``flags``."""
    h = hashlib.sha256()
    for p in source_files():
        h.update(p.relative_to(TROPOI).as_posix().encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    h.update(("\0".join(flags)).encode())
    return h.hexdigest()
