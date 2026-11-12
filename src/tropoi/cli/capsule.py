"""Explicit, CPU-only retrieval and SHA-256 verification of published capsules.

The trusted catalogue is local: downloads never supply their own expected hash.
Drive IDs identify bytes to retrieve; SHA256SUMS determines whether to accept them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import urllib.error
import urllib.request


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe capsule path: {name!r}")
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if (PurePosixPath(name).is_absolute() or any(
            p in {"", ".", ".."} or p.endswith((".", " "))
            or p.split(".")[0].upper() in reserved
            or any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in parts)):
        raise ValueError(f"Unsafe capsule path: {name!r}")
    return name


def read_receipt(path: Path) -> dict[str, str]:
    """Read conventional sha256sum text, rejecting ambiguous or escaping paths."""
    result = {}
    folded = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise ValueError(f"Malformed SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        _safe_name(name)
        if name.casefold() in folded:
            raise ValueError(f"Duplicate capsule path: {name}")
        result[name] = digest.lower()
        folded.add(name.casefold())
    if not result:
        raise ValueError("Empty SHA256SUMS")
    for name in result:
        if any(p.as_posix().casefold() in folded for p in PurePosixPath(name).parents
               if p.as_posix() != "."):
            raise ValueError(f"File/directory collision: {name}")
    return result


def _file(root: Path, name: str) -> Path:
    target = root / _safe_name(name)
    # Reject links/junctions even if they currently point inside the destination.
    for part in (target, *target.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError(f"Refusing linked capsule path: {part}")
    return target


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_archive(catalogue: Path, sums: dict[str, str]) -> dict:
    """Load a complete per-file Drive index and the explicitly published core."""
    data = json.loads((catalogue / "archive.json").read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("provider") != "google-drive"):
        raise ValueError("Unsupported archive schema or provider")
    files = data.get("files")
    included = data.get("included_files")
    if not isinstance(files, dict) or set(files) != set(sums):
        raise ValueError("Archive file index must match SHA256SUMS exactly")
    if (not isinstance(included, list) or not included
            or not all(isinstance(n, str) for n in included)
            or len(set(included)) != len(included) or not set(included) <= set(sums)):
        raise ValueError("Invalid included_files in archive.json")
    if {"SHA256SUMS", "archive.json"} & set(sums):
        raise ValueError("Archive metadata cannot be capsule payload")
    for name, record in files.items():
        if (not isinstance(record, dict)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", str(record.get("drive_id", "")))
                or type(record.get("size")) is not int or record["size"] < 0):
            raise ValueError(f"Invalid archive record: {name}")
    return data


def verify_capsule(root: Path, catalogue: Path, *, core: bool = False) -> dict:
    """Verify all expected bytes; a core check explicitly reports omitted files."""
    sums = read_receipt(catalogue / "SHA256SUMS")
    archive = read_archive(catalogue, sums) if (catalogue / "archive.json").exists() else None
    if core and archive is None:
        raise ValueError("--core requires archive.json with an explicit included_files list")
    required = set(archive["included_files"]) if core else set(sums)
    problems = []
    checked = 0
    missing = 0
    for name, expected in sums.items():
        path = _file(root, name)
        if not path.exists():
            missing += 1
            if name in required:
                problems.append(f"missing: {name}")
        elif (not path.is_file() or _digest(path) != expected
              or (archive is not None and path.stat().st_size != archive["files"][name]["size"])):
            problems.append(f"checksum mismatch: {name}")
        else:
            checked += 1
    if archive and root.exists():
        for path in root.rglob("*"):
            name = path.relative_to(root).as_posix()
            _file(root, name)
            if path.is_file() and name not in sums and name not in {"SHA256SUMS", "archive.json"}:
                problems.append(f"unlisted file: {name}")
    if problems:
        raise ValueError("Capsule verification failed:\n" + "\n".join(problems))
    return {"checked": checked, "missing": missing, "total": len(sums)}


def _download(drive_id: str, destination: Path, size: int, digest: str) -> None:
    url = f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"
    # Temporary bytes are never published under their final name before verification.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".capsule-",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                count = 0
                actual = hashlib.sha256()
                while chunk := response.read(min(1024 * 1024, size - count + 1)):
                    count += len(chunk)
                    if count > size:
                        raise ValueError(f"Download exceeds recorded size: {destination.name}")
                    actual.update(chunk)
                    stream.write(chunk)
            if count != size or actual.hexdigest() != digest:
                raise ValueError(f"Download checksum/size mismatch: {destination.name}; "
                                 "Drive may have returned a sign-in/quota page or changed bytes")
            stream.close()
            # Atomic no-clobber publication (unlike replace, a concurrent file wins).
            if os.name == "nt":
                # Windows rename is atomic and refuses an existing destination.
                os.rename(temporary, destination)
            else:
                os.link(temporary, destination)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def fetch_capsule(catalogue: Path, destination: Path, *, progress=print) -> dict:
    """Fetch missing files, reuse verified files, and refuse conflicting local data."""
    sums = read_receipt(catalogue / "SHA256SUMS")
    archive = read_archive(catalogue, sums)
    pending = []
    for name, expected in sums.items():
        path = _file(destination, name)
        if path.exists():
            if not path.is_file() or _digest(path) != expected:
                raise ValueError(f"Refusing to overwrite differing local file: {path}")
        else:
            pending.append(name)
    # Detect unrelated files before spending bandwidth.
    if destination.exists():
        for path in destination.rglob("*"):
            name = path.relative_to(destination).as_posix()
            _file(destination, name)
            if path.is_file() and name not in sums and name not in {"archive.json", "SHA256SUMS"}:
                raise ValueError(f"Unlisted local file: {path}")
    progress(f"Fetching {len(pending)} files; reusing {len(sums) - len(pending)} verified files")
    for i, name in enumerate(pending, 1):
        path = _file(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = archive["files"][name]
        progress(f"[{i}/{len(pending)}] {name}")
        _download(record["drive_id"], path, record["size"], sums[name])
    return verify_capsule(destination, catalogue)


def _catalogue(target: str, catalog: str) -> Path:
    path = Path(target)
    if path.is_dir() and (path / "SHA256SUMS").is_file():
        return path
    _safe_name(target)
    if "/" in target:
        raise ValueError("Use a capsule ID or a directory containing SHA256SUMS")
    result = Path(catalog) / target
    if not (result / "SHA256SUMS").is_file():
        raise ValueError(f"No receipt at {result}; run from the repository root "
                         "or specify --catalog PATH")
    return result


def _command(args) -> int:
    try:
        catalogue = _catalogue(args.capsule, args.catalog)
        destination = Path(args.path) if args.path else Path("runs") / catalogue.name
        if args.action == "fetch":
            result = fetch_capsule(catalogue, destination)
        else:
            if args.path is None and (Path(args.capsule).is_dir() or args.core):
                destination = catalogue
            result = verify_capsule(destination, catalogue, core=args.core)
        label = "CORE VERIFIED" if getattr(args, "core", False) else "VERIFIED"
        print(f"{label}: {result['checked']}/{result['total']} files at {destination}; "
              f"{result['missing']} archived files absent")
        return 0
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"capsule: {error}", file=sys.stderr)
        return 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("fetch", "verify"):
        command = actions.add_parser(action, help=("Download missing files and verify" if action == "fetch"
                                                   else "Verify files without network access"))
        command.add_argument("capsule", help="Published capsule ID or receipt directory")
        command.add_argument("--catalog", default="docs/runs", help="Local catalogue (default: docs/runs)")
        command.add_argument("--path", help="Full capsule directory (default: runs/ID)")
        if action == "verify":
            command.add_argument("--core", action="store_true", help="Check the published core; report omitted files")
        command.set_defaults(_handler=_command)
