"""Fetch the ROCKE-3D SOCRATES spectral files the k-table importer reads.

    python -I tools/radiation/opacity/fetch_socrates.py [--set core|reference|shortwave|all]
                                                        [--cache opacity_cache/socrates/rocke3d]

Source: the public NASA NCCS data share of ROCKE-3D,
``https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/spectral_files/``
(documented at https://simplex.giss.nasa.gov/gcm/ROCKE-3D/Spectralfiles.html).
Each spectral file is a text file plus its ``_k`` look-up companion.

* ``core`` (10 MB): the long-wave files of the two THAI dry cases,
  ``sp_lw_17_dsa_mars`` (Ben 2) and ``sp_lw_15_dsa_ar10bar`` (Ben 1);
* ``reference`` (450 MB): their documented 350-band high-resolution
  references, used only by the column comparison;
* ``shortwave`` (22 MB): the TRAPPIST-1 short-wave files (fetched for a
  follow-up; no importer reads them yet).

Files are written once and never transformed. A file whose SHA-256 is
pinned below must match it, on download and on every later run; the sizes
and hashes found are written to ``manifest.json`` beside the files. The
cache holds data only: no script lives in it, and the importer reads it
with ``python -I``.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import urllib.request

BASE_URL = "https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/spectral_files"

#: (set, path under the base URL, SHA-256 as fetched on 2026-11-16).
FILES = (
    ("core", "sp_lw_dsa_mars/sp_lw_17_dsa_mars",
     "e1197e49351953c3c71f5d01a674f8a617778557c4559750adb5b742a6bc75f5"),
    ("core", "sp_lw_dsa_mars/sp_lw_17_dsa_mars_k",
     "57e679a713d2ffb1921540d6a23ae4fcbda2abe7ec1a6d451192141191dd2785"),
    ("core", "sp_lw_dsa_ar10bar/sp_lw_15_dsa_ar10bar",
     "7bfb63eb1c0ca2fd1bfb446a358034592f3decabbb6db3450217a14095540bb7"),
    ("core", "sp_lw_dsa_ar10bar/sp_lw_15_dsa_ar10bar_k",
     "da6393d5646fe5ba13f4e6d21352858232095def311042ee14412d372e13af88"),
    ("reference", "sp_lw_dsa_mars/sp_lw_350_dsa_mars",
     "99dd5038379d84120f849fcd66e3e297391f2d1b040e2b70d074b60aac4bc2bb"),
    ("reference", "sp_lw_dsa_mars/sp_lw_350_dsa_mars_k",
     "88398a8afdc0b0ecd50a0e29af47baa9925758bc69f07df2dcbec93e4f655e78"),
    ("reference", "sp_lw_dsa_ar10bar/sp_lw_350_dsa_ar10bar",
     "ac711eaa35fa47e7ffdda4a8376895348daca6d6a9d23b32db0bde9cc5691839"),
    ("reference", "sp_lw_dsa_ar10bar/sp_lw_350_dsa_ar10bar_k", None),
    ("shortwave", "sp_sw_dsa_mars/sp_sw_42_dsa_mars_trappist1",
     "f2730774ae08ad722524cf18c65e5ab4eb1f1bdd35b3c08b6af76c22567099fe"),
    ("shortwave", "sp_sw_dsa_mars/sp_sw_42_dsa_mars_trappist1_k",
     "222801f3faf244ed561a00c51f6ba29f7d7b5b09ca248c6fa8e4293af6bef4ed"),
    ("shortwave", "sp_sw_dsa_ar10bar/sp_sw_43_dsa_ar10bar_trappist1",
     "704f9776a1fd16f32b7046207c51c5393172e8b591c542aed15a2267aa3c54b4"),
    ("shortwave", "sp_sw_dsa_ar10bar/sp_sw_43_dsa_ar10bar_trappist1_k",
     "977f2cdd90f73412f77ac723e4a8ad27408f2de372da80ecea2e72f8344a65c1"),
)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="core",
                    choices=("core", "reference", "shortwave", "all"))
    ap.add_argument("--cache", default="opacity_cache/socrates/rocke3d")
    args = ap.parse_args(argv)
    cache = pathlib.Path(args.cache)
    manifest_path = cache / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"schema": "palintropos.opacity.manifest/1",
                                                "files": {}})
    for group, rel, pinned in FILES:
        if args.set not in ("all", group):
            continue
        path = cache / rel
        url = f"{BASE_URL}/{rel}"
        fetched = None
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(part, "wb") as fh:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    fh.write(block)
            part.replace(path)
            fetched = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        sha = _sha256(path)
        if pinned is not None and sha != pinned:
            raise SystemExit(f"{rel}: SHA-256 {sha} is not the pinned {pinned}; the "
                             "source file changed or the download is damaged")
        entry = manifest["files"].get(rel, {})
        entry.update({"url": url, "bytes": path.stat().st_size, "sha256": sha, "set": group,
                      "pinned": pinned is not None})
        if fetched or "fetched_utc" not in entry:
            entry["fetched_utc"] = fetched or "unknown (file pre-existed)"
        manifest["files"][rel] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        print(f"{rel:58s} {path.stat().st_size:>11d}  {sha[:16]}  "
              f"{'fetched' if fetched else 'kept'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
