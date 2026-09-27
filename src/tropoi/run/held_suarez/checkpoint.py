"""Atomic, hash-verified Held–Suarez checkpoints (CPU-only).

One checkpoint is one uncompressed ``.npz`` (``numpy.load(allow_pickle=
False)`` — no pickle anywhere) holding named arrays plus ``__meta__``, a
UTF-8 JSON document stored as a uint8 array. The metadata carries
(plan §3 "Checkpoint schema"):

* ``schema`` — :data:`CHECKPOINT_SCHEMA`; any other value is refused;
* ``stepper`` — the stepper state dict without its arrays (scheme, dt, t,
  step, RAW coefficients, T_ref, the full operator signature, startup
  substeps, hook signatures); the two leapfrog levels are the arrays
  ``x_prev`` (X^{n-1}_f) and ``x_curr`` (X^n);
* ``rng_state`` — the perturbation generator's bit-generator state;
* ``statistics`` — accumulator layout (block/count state is in the arrays);
* ``day``, ``config`` and ``config_sha256``, ``protocol_sha256``;
* ``code`` — ``git_commit``, ``git_dirty``;
* ``environment`` — Python/NumPy/CuPy versions, CUDA runtime and driver,
  device name (recorded, compared with a warning only: resuming on other
  hardware is scientifically valid but not bit-identical);
* ``arrays`` — ``{name: {"sha256", "dtype", "shape"}}`` for EVERY stored
  array, verified on read before anything is used;
* ``meta_sha256`` — SHA-256 of the canonical JSON of all the other metadata
  (so the stepper counters, RNG state and hash manifest are verified too).

Writes go to a unique ``<name>.<pid>.tmp`` in the same directory, are
flushed and fsynced, then renamed over the final name with
:func:`os.replace` (atomic on POSIX and Windows; on POSIX the directory is
fsynced as well), so a reader only ever sees complete checkpoints.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import re
import shutil
from typing import Any

import numpy as np

__all__ = ["CHECKPOINT_SCHEMA", "CheckpointError", "array_sha256", "write_checkpoint",
           "read_checkpoint", "list_checkpoints", "prune_checkpoints", "atomic_write_bytes",
           "atomic_copy", "backup_files"]

CHECKPOINT_SCHEMA = "palintropos.held_suarez.checkpoint/1"
_NAME = re.compile(r"^checkpoint-s(\d{9})\.npz$")


class CheckpointError(RuntimeError):
    """A checkpoint is missing, corrupt, or incompatible with this run."""


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _fsync_dir(directory: pathlib.Path) -> None:
    if os.name != "posix":                     # Windows: os.replace is durable enough
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tmp_name(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    path = pathlib.Path(path)
    tmp = _tmp_name(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _meta_digest(meta: dict[str, Any]) -> str:
    body = {k: v for k, v in meta.items() if k != "meta_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def checkpoint_name(step: int) -> str:
    return f"checkpoint-s{int(step):09d}.npz"


def write_checkpoint(directory: pathlib.Path, step: int, arrays: dict[str, np.ndarray],
                     meta: dict[str, Any]) -> pathlib.Path:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    if "__meta__" in arrays:
        raise ValueError("'__meta__' is reserved")
    meta = dict(meta)
    meta["schema"] = CHECKPOINT_SCHEMA
    meta["arrays"] = {k: {"sha256": array_sha256(v), "dtype": str(v.dtype),
                          "shape": list(v.shape)} for k, v in sorted(arrays.items())}
    meta["meta_sha256"] = _meta_digest(meta)
    blob = json.dumps(meta, sort_keys=True).encode("utf-8")
    buf = io.BytesIO()
    np.savez(buf, __meta__=np.frombuffer(blob, dtype=np.uint8), **arrays)
    path = directory / checkpoint_name(step)
    atomic_write_bytes(path, buf.getvalue())
    return path


def read_checkpoint(path: pathlib.Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and verify: schema, the array set, and the SHA-256 of every array."""
    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            files = {k: np.array(z[k]) for k in z.files}
    except Exception as exc:                          # truncated / not a zip
        raise CheckpointError(f"{path}: unreadable checkpoint ({exc})") from exc
    if "__meta__" not in files:
        raise CheckpointError(f"{path}: no metadata")
    try:
        meta = json.loads(files.pop("__meta__").tobytes().decode("utf-8"))
    except Exception as exc:
        raise CheckpointError(f"{path}: corrupt metadata ({exc})") from exc
    if meta.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            f"{path}: checkpoint schema {meta.get('schema')!r} is not {CHECKPOINT_SCHEMA!r}")
    if meta.get("meta_sha256") != _meta_digest(meta):
        raise CheckpointError(f"{path}: SHA-256 mismatch for the checkpoint metadata")
    listed = meta.get("arrays", {})
    if set(listed) != set(files):
        raise CheckpointError(
            f"{path}: stored arrays {sorted(files)} do not match the manifest {sorted(listed)}")
    for name, a in files.items():
        info = listed[name]
        if str(a.dtype) != info["dtype"] or list(a.shape) != info["shape"]:
            raise CheckpointError(f"{path}: array {name!r} dtype/shape differs from manifest")
        if array_sha256(a) != info["sha256"]:
            raise CheckpointError(f"{path}: SHA-256 mismatch for array {name!r}")
    return files, meta


def list_checkpoints(directory: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    """(step, path) of every complete checkpoint, ascending (temp files ignored)."""
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for p in directory.iterdir():
        m = _NAME.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def prune_checkpoints(directory: pathlib.Path, keep: int) -> list[pathlib.Path]:
    """Delete all but the newest ``keep`` checkpoints; return the deleted paths."""
    found = list_checkpoints(directory)
    doomed = [p for _, p in found[:-keep]] if keep > 0 else [p for _, p in found]
    for p in doomed:
        p.unlink()
    return doomed


def atomic_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(dst)
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)


def backup_files(checkpoint: pathlib.Path, extra: list[pathlib.Path],
                 backup_dir: pathlib.Path, keep: int) -> pathlib.Path:
    """Copy a checkpoint (and the small run files in ``extra``) into
    ``backup_dir`` atomically, verify the copied checkpoint, and keep only the
    newest ``keep`` checkpoints there. (On a network or FUSE-mounted drive the
    read-back verifies what the local client holds, not the remote copy.)"""
    backup_dir = pathlib.Path(backup_dir)
    dst = backup_dir / "checkpoints" / pathlib.Path(checkpoint).name
    atomic_copy(checkpoint, dst)
    read_checkpoint(dst)                                   # verify the copy
    for f in extra:
        f = pathlib.Path(f)
        if f.exists():
            atomic_copy(f, backup_dir / f.name)
    prune_checkpoints(backup_dir / "checkpoints", keep)
    return dst
