"""Equivalence of the four-coordinate evaluation engine with the product table it replaced.

    PYTHONPATH=src python tools/viewer/engine_equivalence.py \
        --old-src /path/to/checkout-at-b47180c/src --out result.json ARCHIVE [ARCHIVE ...]

The old implementation (``ViewerSession.select`` of the checkout given by
``--old-src``, which matched role assignments against a fixed table of eight
plot products, run in a subprocess) and the new one (one evaluation path for
every role assignment) receive the **same selection text** for every
combination the old table supported:

* longitude × latitude: the binary frame of the globe (``options.format =
  "frame"``) must be **byte-identical**;
* atmospheric columns (snapped and not, every offered profile
  representation), meridional and zonal sections, zonal means (one state and
  a range, ψ included), time series and time sections; columnar profiles,
  sections, time series (skin, cell and flux) and time sections: every
  number the old payload carried for the selection (values, axis positions,
  table, markers, companions, locations, times, sample descriptions) must
  equal the new payload's **exactly**.

Nothing is tolerance-based. Band edges of σ axes are not compared: the old
axis drew native levels between log-midpoints, the new one between the
model's own interfaces (a drawing choice, recorded in the documentation).
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
for run, text in cases:
    try:
        data, mime = session.select(run, text)
        out.append(base64.b64encode(data).decode() if mime == "application/octet-stream" else json.loads(data))
    except Exception as err:
        out.append({"error": f"{type(err).__name__}: {err}"})
print(json.dumps(out))
'''

fixed = lambda v: dict(role="fixed", value=v)        # noqa: E731


def _atmosphere(manifest):
    quantities = manifest["explorer"]["quantities"]
    times = manifest["time_axes"]["snapshots"]["values"]
    last = len(times) - 1
    out = []
    for key, q in quantities.items():
        caps = q["coordinates"]
        if "x" not in caps["longitude"]["roles"]:
            continue
        surfaces = [dict(role="none")]
        if q["vertical"]:
            n = len(q["vertical"]["levels"])
            surfaces = [dict(role="fixed", representation="native", value=k) for k in sorted({0, n // 2, n - 1})]
            values = dict(sigma=0.5, pressure=0.5 * (manifest["physics"].get("reference_pressure") or 1e5),
                          geopotential_height=5000.0, height_above_surface=1500.0)
            surfaces += [dict(role="fixed", representation=r, value=values[r])
                         for r in caps["vertical"]["fixed"] if r != "native"]
        for t in sorted({0, last}):
            for vertical in surfaces:
                out.append(("frame", dict(quantity=key, longitude=dict(role="x"), latitude=dict(role="y"),
                                          vertical=vertical, time=dict(role="fixed", index=t),
                                          options=dict(format="frame"))))
    for key in ("temperature", "u", "v", "pressure", "geopotential_height"):
        caps = (quantities.get(key) or {}).get("coordinates", {}).get("vertical", {})
        if "y" not in caps.get("roles", []):
            continue
        for t in sorted({0, last}):
            for snap in (True, False):
                for rep in [*caps["per_sample"], *caps["axis"]]:
                    out.append(("column", dict(quantity=key, longitude=fixed(301.25), latitude=fixed(23.5),
                                               vertical=dict(role="y", representation=rep),
                                               time=dict(role="fixed", index=t), options=dict(snap=snap))))
            for coords in (dict(latitude=dict(role="x"), longitude=fixed(301.25)),
                           dict(longitude=dict(role="x"), latitude=fixed(23.5))):
                out.append(("section", dict(quantity=key, vertical=dict(role="y", representation="sigma"),
                                            time=dict(role="fixed", index=t), options=dict(snap=False), **coords)))
        first = max(0, last - 3)
        out.append(("history", dict(quantity=key, longitude=fixed(301.25), latitude=fixed(23.5),
                                    vertical=dict(role="fixed", representation="native", value=7),
                                    time=dict(role="x", first=first, last=last))))
        out.append(("history", dict(quantity=key, longitude=fixed(301.25), latitude=fixed(23.5),
                                    vertical=dict(role="y", representation="sigma"),
                                    time=dict(role="x", first=first, last=last))))
    for key in ("temperature", "u", "v", "mass_streamfunction"):
        if key not in quantities or "reduce" not in quantities[key]["coordinates"]["longitude"]["roles"]:
            continue
        for first, final in ((0, 0), (max(0, last - 3), last)):
            time = (dict(role="fixed", index=first) if first == final else
                    dict(role="reduce", op="mean", first=first, last=final))
            out.append(("zonal_mean", dict(quantity=key, latitude=dict(role="x"),
                                           longitude=dict(role="reduce", op="mean"),
                                           vertical=dict(role="y", representation="sigma"), time=time)))
    return out


def _columnar(manifest):
    quantities = manifest["explorer"]["quantities"]
    q = quantities["temperature"]
    levels = q["vertical"]["levels"]
    axes = manifest["time_axes"]
    out = []
    global_ = "x" in q["coordinates"]["longitude"]["roles"]
    point = dict(longitude=fixed(0.0), latitude=fixed(0.0))
    if not global_:
        p = manifest["explorer"]["horizontal"]["points"][0]
        point = dict(longitude=fixed(p["longitude"]), latitude=fixed(p["latitude"]))
    cell_axis = q["time_axis"]
    n = len(axes[cell_axis]["values"])
    for rep in q["coordinates"]["vertical"]["axis"]:
        out.append(("column", dict(quantity="temperature", vertical=dict(role="y", representation=rep),
                                   time=dict(role="fixed", index=n - 1), **point)))
        out.append(("history", dict(quantity="temperature", vertical=dict(role="y", representation=rep),
                                    time=dict(role="x", first=0, last=n - 1), **point)))
        if global_:
            for coords in (dict(latitude=dict(role="x"), longitude=fixed(0.0)),
                           dict(longitude=dict(role="x"), latitude=fixed(0.0))):
                out.append(("section", dict(quantity="temperature", vertical=dict(role="y", representation=rep),
                                            time=dict(role="fixed", index=n - 1), **coords)))
    for level in (levels[0], levels[1], levels[-1]):
        count = len(axes[level["time_axis"]]["values"])
        vertical = dict(role="fixed", representation="native", value=level["index"])
        out.append(("history", dict(quantity="temperature", vertical=vertical,
                                    time=dict(role="x", first=0, last=count - 1), **point)))
        if global_:
            out.append(("frame", dict(quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"),
                                      vertical=vertical, time=dict(role="fixed", index=count - 1),
                                      options=dict(format="frame"))))
    for key, flux in quantities.items():
        if flux.get("vertical") is None:
            count = len(axes[flux["time_axis"]]["values"])
            out.append(("history", dict(quantity=key, vertical=dict(role="none"),
                                        time=dict(role="x", first=0, last=count - 1), **point)))
            if global_:
                out.append(("frame", dict(quantity=key, longitude=dict(role="x"), latitude=dict(role="y"),
                                          vertical=dict(role="none"), time=dict(role="fixed", index=count - 1),
                                          options=dict(format="frame"))))
    return out


def cases_for(manifest):
    columnar = manifest["source"].get("format") == "airless-columns"
    return _columnar(manifest) if columnar else _atmosphere(manifest)


def _location(new):
    c = new["coordinates"]
    return [c[k]["value"] for k in ("latitude", "longitude") if c[k]["role"] == "fixed"]


def compare(kind, old, new_bytes):
    """``(equal, detail)``."""
    if isinstance(old, dict) and "error" in old:
        return False, "old implementation raised " + old["error"]
    if kind == "frame":
        same = base64.b64decode(old) == new_bytes
        return same, "byte-identical frame" if same else "frames differ"
    new = json.loads(new_bytes)
    checks = {"values": (old["values"], new["values"])}
    for a in ("x", "y"):
        if isinstance(old.get(a), dict) and "values" in old[a]:
            checks[a] = (old[a]["values"], new[a]["values"])
    if "table" in old:
        checks["table"] = (old["table"]["rows"], new["table"]["rows"])
    if "markers" in old:
        checks["markers"] = ([(m["value"], m["position"]) for m in old["markers"]],
                             [(m["value"], m["position"]) for m in new["markers"]])
    if isinstance(old.get("surface"), dict):              # an atmospheric column's pₛ and Zₛ
        checks["surface"] = ([old["surface"]["pressure"], old["surface"]["height"]],
                             [new["companions"]["surface_pressure"]["values"],
                              new["companions"]["surface_height"]["values"]])
    elif old.get("surface") is not None or new.get("surface") is not None:
        checks["skin"] = (old.get("surface"), new.get("surface"))
    for name, companion in (old.get("companions") or {}).items():
        checks["companion " + name] = (companion["values"], new["companions"][name]["values"])
    location = old.get("location") or {}
    if "latitude" in location and "longitude" in location:
        checks["location"] = ([location["latitude"], location["longitude"]], _location(new))
    elif location:
        name = "longitude" if "longitude" in location else "latitude"
        checks["location"] = (location[name], new["coordinates"][name]["value"])
    t = old.get("time") or {}
    if "seconds" in t:
        checks["time"] = (t["seconds"], new["coordinates"]["time"]["seconds"])
    if "sample" in old:
        s, n = old["sample"], new["coordinates"]["time"]
        new_sample = ([1, {}, "single sample"] if n["role"] == "fixed" else
                      [n["count"], n["intervals_seconds"], n["cadence"]])
        checks["sample"] = ([s["count"], s["intervals_seconds"], s["cadence"].replace("state", "sample")], new_sample)
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
    cases = [(run, kind, json.dumps(sel)) for run in range(len(roots))
             for kind, sel in cases_for(session.manifests[run])]
    env = dict(PYTHONPATH=args.old_src, PATH="/usr/bin:/bin", TROPOI_EXECUTION="cpu-verification")
    old = subprocess.run([sys.executable, "-c", OLD], input=json.dumps(
        [roots, [(run, text) for run, _, text in cases]]), capture_output=True, text=True, env=env, check=True)
    old = json.loads(old.stdout)
    rows, failures = [], 0
    for (run, kind, text), before in zip(cases, old):
        try:
            data, _ = session.select(run, text)
            same, detail = compare(kind, before, data)
        except Exception as err:
            refused = isinstance(before, dict) and "error" in before
            same = refused                       # both refuse: no value exists to differ
            detail = (("both refuse; old " + before["error"] + "; new ") if refused else
                      "new implementation raised ") + f"{type(err).__name__}: {err}"
        failures += not same
        rows.append(dict(archive=pathlib.Path(roots[run]).name, kind=kind, selection=json.loads(text),
                         equal=same, detail=detail))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    old_git = subprocess.run(["git", "-C", args.old_src, "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    result = dict(new_commit=git, old_commit=old_git, archives=[pathlib.Path(r).name for r in roots],
                  comparisons=len(rows), failures=failures, rows=rows)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("old_commit", "new_commit", "archives", "comparisons", "failures")}))
    for row in rows:
        if not row["equal"]:
            print("FAIL", row["archive"], row["kind"], json.dumps(row["selection"]), row["detail"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
