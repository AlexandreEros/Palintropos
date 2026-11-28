"""Equivalence of the viewer's selection endpoint with the endpoints it replaced.

    PYTHONPATH=src python tools/viewer/selection_equivalence.py \
        --old-src /path/to/checkout-before/src --out result.json ARCHIVE [ARCHIVE ...]

For each atmospheric archive the old implementation (``ViewerSession.frame``
and ``ViewerSession.structure`` of the checkout given by ``--old-src``, run in
a subprocess) and the new one (``ViewerSession.select``) evaluate the same
matched selections:

* maps: every map quantity on native levels (top, middle, bottom) and, where
  offered, on sigma, pressure and geopotential-height surfaces (not pressure
  on an isobar or Z on a constant-Z surface, which neither page offers), at the first
  and last saved times. The binary frames must be **byte-identical** (values,
  terrain, heights, winds and masks).
* columns (snapped and unsnapped), meridional and zonal sections, and zonal
  means of one state and of a range: every number the old JSON carried for
  the quantity must equal the new payload's **exactly**.

Nothing is tolerance-based. The result lists every comparison.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import subprocess
import sys

OLD = r'''
import base64, json, sys
from tropoi.representation.visual.viewer_server import ViewerSession
roots, cases = json.loads(sys.stdin.read())
session = ViewerSession(roots)
out = []
for case in cases:
    run, kind, query = case
    try:
        if kind == "frame":
            out.append(base64.b64encode(session.frame(run, query)).decode())
        else:
            out.append(json.loads(session.structure(run, query)))
    except Exception as err:          # recorded and compared as an error
        out.append({"error": f"{type(err).__name__}: {err}"})
print(json.dumps(out))
'''


def cases_for(manifest, run):
    """Old queries and the matching new selections."""
    from tropoi.representation.visual.viewer_atmosphere import _excluded
    fields, ops = manifest["fields"], manifest["operations"]
    times = manifest["time_axes"]["snapshots"]["values"]
    last = len(times) - 1
    full = (manifest["axes"].get("vertical") or {}).get("full_levels") or []
    reference = manifest["physics"].get("reference_pressure") or manifest["physics"].get("surface_pressure") or 1e5
    out = []
    for key in ops.get("map", []):
        if key == "wind":
            continue
        surfaces = [("native", 0)]
        if fields[key].get("levels"):
            surfaces = [("native", k) for k in sorted({0, len(full) // 2, len(full) - 1})]
            for rep, value in (("sigma", 0.5), ("pressure", 0.5 * reference), ("geopotential_height", 5000.0)):
                # p on an isobar and Z on a constant-Z surface are constants: Viewer 2's
                # page never offered them, and the selection description does not either.
                if key in ops.get(rep, []) and not _excluded(rep, key):
                    surfaces.append((rep, value))
        for t in sorted({0, last}):
            for rep, value in surfaces:
                old = dict(field=key, time=str(t), coordinate=rep, value=str(value))
                vertical = (dict(role="fixed", representation=rep, value=value) if fields[key].get("levels")
                            else dict(role="none"))
                new = dict(quantity=key, longitude=dict(role="x"), latitude=dict(role="y"),
                           vertical=vertical, time=dict(role="fixed", index=t))
                out.append(("frame", old, new, None))
    structure = ops.get("structure", [])
    fixed = lambda v: dict(role="fixed", value=v)
    for key in ("temperature", "u", "v", "pressure", "geopotential_height"):
        if "column" not in structure or key not in fields:
            continue
        for t in sorted({0, last}):
            for snap in (1, 0):
                old = dict(kind="column", time=str(t), lat="23.5", lon="301.25", snap=str(snap))
                for rep in ("geopotential_height", "pressure", "sigma"):
                    new = dict(quantity=key, longitude=fixed(301.25), latitude=fixed(23.5),
                               vertical=dict(role="y", representation=rep), time=dict(role="fixed", index=t),
                               options=dict(snap=bool(snap)))
                    out.append(("column", old, new, (key, rep)))
            for kind, at in (("meridional", 301.25), ("zonal", 23.5)):
                old = dict(kind=kind, time=str(t), at=str(at))
                coord = dict(latitude=dict(role="x"), longitude=fixed(at)) if kind == "meridional" else \
                    dict(longitude=dict(role="x"), latitude=fixed(at))
                new = dict(quantity=key, vertical=dict(role="y", representation="sigma"),
                           time=dict(role="fixed", index=t), **coord)
                out.append(("section", old, new, key))
    if "zonal_mean" in structure:
        for first, final in ((0, 0), (max(0, last - 3), last)):
            old = dict(kind="zonal_mean", first=str(first), last=str(final))
            for key in ("temperature", "u", "v", "mass_streamfunction"):
                time = (dict(role="fixed", index=first) if first == final else
                        dict(role="reduce", op="mean", first=first, last=final))
                new = dict(quantity=key, latitude=dict(role="x"), longitude=dict(role="reduce", op="mean"),
                           vertical=dict(role="y", representation="sigma"), time=time)
                out.append(("zonal_mean", old, new, key))
    return [(run, *c) for c in out]


def compare(kind, old, new_bytes, extra):
    """``(equal, detail)``."""
    if isinstance(old, dict) and "error" in old:
        return False, "old implementation raised " + old["error"]
    if kind == "frame":
        same = base64.b64decode(old) == new_bytes
        return same, "byte-identical frame" if same else "frames differ"
    new = json.loads(new_bytes)
    if kind == "column":
        key, rep = extra
        checks = {"values": (old[key], new["values"]), "table.T": (old["temperature"], [r[5] for r in new["table"]["rows"]]),
                  "table.u": (old["u"], [r[6] for r in new["table"]["rows"]]),
                  "table.v": (old["v"], [r[7] for r in new["table"]["rows"]]),
                  "table.p": (old["pressure"], [r[2] for r in new["table"]["rows"]]),
                  "table.Z": (old["geopotential_height"], [r[3] for r in new["table"]["rows"]]),
                  "location": ([old["latitude"], old["longitude"]],
                               [new["location"]["latitude"], new["location"]["longitude"]]),
                  "time": (old["time"], new["time"]["seconds"])}
        axis = {"geopotential_height": old["geopotential_height"], "pressure": old["pressure"],
                "sigma": old["sigma"]}[rep]
        checks["y"] = (axis, new["y"]["values"])
        if key == "temperature" and old.get("surface_temperature"):
            checks["surface marker"] = (old["surface_temperature"]["value"],
                                        [m["value"] for m in new["markers"] if m["kind"] == "surface"][0])
    elif kind == "section":
        checks = {"values": (old[extra], new["values"]), "x": (old["points"], new["x"]["values"]),
                  "sigma": (old["sigma"], new["y"]["values"]), "at": (old["at"], list(new["location"].values())[0]),
                  "p": (old["pressure"], new["companions"]["pressure"]["values"]),
                  "Z": (old["geopotential_height"], new["companions"]["geopotential_height"]["values"])}
    else:
        values = old["streamfunction"] if extra == "mass_streamfunction" else old[extra]
        sigma = old["sigma_interfaces"] if extra == "mass_streamfunction" else old["sigma"]
        checks = {"values": (values, new["values"]), "lat": (old["lat"], new["x"]["values"]),
                  "sigma": (sigma, new["y"]["values"]), "sample": (old["sample"], new["sample"])}
    bad = [name for name, (a, b) in checks.items() if a != b]
    return not bad, ("exactly equal: " + ", ".join(checks)) if not bad else "differ: " + ", ".join(bad)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("archives", nargs="+")
    args = ap.parse_args(argv)
    from tropoi.representation.visual.viewer_server import ViewerSession
    roots = [str(pathlib.Path(a).resolve()) for a in args.archives]
    session = ViewerSession(roots)
    cases = [c for run in range(len(roots)) for c in cases_for(session.manifests[run], run)]
    env = dict(PYTHONPATH=args.old_src, PATH="/usr/bin:/bin")
    old = subprocess.run([sys.executable, "-c", OLD], input=json.dumps(
        [roots, [(run, "frame" if kind == "frame" else "structure", query) for run, kind, query, _, _ in cases]]),
        capture_output=True, text=True, env=env, check=True)
    old = json.loads(old.stdout)
    rows, failures = [], 0
    for (run, kind, query, selection, extra), before in zip(cases, old):
        try:
            data, _ = session.select(run, json.dumps(selection))
            same, detail = compare(kind, before, data, extra)
        except Exception as err:
            same, detail = False, f"new implementation raised {type(err).__name__}: {err}"
        failures += not same
        rows.append(dict(archive=pathlib.Path(roots[run]).name, kind=kind, old_query=query,
                         selection=selection, equal=same, detail=detail))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    old_git = subprocess.run(["git", "-C", args.old_src, "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    result = dict(new_commit=git, old_commit=old_git, archives=[pathlib.Path(r).name for r in roots],
                  comparisons=len(rows), failures=failures, rows=rows)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("old_commit", "new_commit", "archives", "comparisons", "failures")}))
    for row in rows:
        if not row["equal"]:
            print("FAIL", row["archive"], row["kind"], row["old_query"], row["detail"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
