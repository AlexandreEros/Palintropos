"""Viewer selections of columnar substrate archives (airless runs).

Values are read, never reconstructed. The selection's roles choose a set of
archived columns per horizontal coordinate, a set of native levels and a set
of saved samples; this provider reads exactly those values and
``selection.evaluate`` reduces and arranges them.

Horizontal sample sets of a global archive (a latitude–longitude product of
columns), by role: ``x``/``y`` every native row or meridian; ``fixed`` the
nearest native row or meridian (the nearest column on the sphere when both
are fixed); ``reduce`` every row or meridian, weighted by the recorded column
area weights (the mean over longitude is the plain mean of a row, whose cells
have equal areas). A diagnostic point collection is not a global field: its
only horizontal role is ``fixed`` (the nearest point), and it has no area
weights.

A vertical value must be a native level: the diagnosed skin at depth 0, or a
cell centre. Values between native depths are refused, not interpolated. The
skin is its own native level with its own time axis: under a vertical axis it
is shown as a separate row only when it was sampled at the step of every
cell sample used, and never joined to the cells across a time it was not.
"""
from __future__ import annotations

import math

import numpy as np

from tropoi.representation.archive.airless import SURFACE_PLANES, AirlessArchive
from tropoi.representation.visual.selection import (
    MAX_SAMPLES, MAX_STATES, SelectionError, axis, latitude_axis, level_time_axis, longitude_axis)

SKIN = ("The surface temperature is the diagnosed, massless skin of each column (depth 0). "
        "It is its own native level, not the top cell, and it is drawn apart from the cells.")
POINTS = ("A diagnostic point collection is not a global field: it has no horizontal axes, "
          "reductions or area weights")
NO_WEIGHTS = "this archive records no column area weights"
NO_CELL_MEAN = ("a mean over substrate cells needs a thickness weighting rule, which this release "
                "does not define")
MEAN = dict(longitude="plain mean over the archived meridians of a row (cells of equal area)",
            latitude="mean over the archived rows weighted by the recorded column area weights",
            time="equal-weight mean over the saved samples first..last inclusive")


class ColumnArchiveProvider:
    """The viewer's evaluator for an :class:`AirlessArchive`."""

    backend = "cpu"
    backend_reason = "Columnar archives are read, not synthesized."

    def __init__(self, root, manifest, *, backend="cpu"):
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be cpu or cuda")
        self.archive = AirlessArchive(root)
        self.manifest = manifest
        self.quantities = manifest["explorer"]["quantities"]

    # ------------------------------------------------------------------
    # Levels and time
    # ------------------------------------------------------------------

    def _level(self, quantity, vertical):
        rep, value = vertical["representation"], vertical["value"]
        for level in quantity["vertical"]["levels"]:
            if rep == "native" and level["index"] == value:
                return level
            if rep != "native" and math.isclose(level[rep], value, rel_tol=1e-9, abs_tol=1e-12):
                return level
        raise SelectionError(
            "Not a native level of this archive. Columnar archives are shown at their native "
            "levels only (the skin at depth 0 and the cell centres); values between native "
            "depths are not reconstructed.")

    def _family(self, axis_id):
        return self.manifest["time_axes"][axis_id]["family"]

    def _cells_axis(self, quantity):
        levels = quantity["vertical"]["levels"]
        return self.manifest["time_axes"][levels[-1]["time_axis"]]

    def _skin_index(self, cell_axis, index):
        """``(family, sample)`` of the skin taken at the step of cell sample
        ``index``, or None."""
        if cell_axis["family"] == "every_step":
            return ("every_step", index)
        steps = self.archive.steps("surface")
        step = self.archive.steps(cell_axis["family"])[index]
        k = int(np.searchsorted(steps, step))
        return ("surface", k) if k < steps.size and steps[k] == step else None

    def _vertical_axis(self, quantity, rep):
        a = self.archive
        sign = -1.0 if rep == "height_above_surface" else 1.0
        label = quantity["vertical"]["representations"][rep]["label"]
        rows = [dict(kind="cell", label=f"cell {k + 1}", depth=float(d)) for k, d in enumerate(a.centres)]
        out = axis("vertical", rep, label, "m", sign * a.centres, edges=sign * a.interfaces,
                   scale="linear", downward=rep == "depth", rows=rows)
        out["surface_row"] = dict(label="Surface · diagnosed skin (depth 0)", value=0.0)
        return out

    # ------------------------------------------------------------------
    # The selection protocol (``selection.evaluate``)
    # ------------------------------------------------------------------

    def plan(self, sel):
        quantity = self.quantities[sel["quantity"]]
        a = self.archive
        coordinates, axes = {}, {}
        out = dict(presentation=quantity["presentation"], coordinates=coordinates, axes=axes, companions={},
                   notes=[], time_label="Saved sample time (forcing time of the step)")
        lat_role, lon_role = sel["latitude"]["role"], sel["longitude"]["role"]
        if lat_role == "fixed" and lon_role == "fixed":
            j = a.nearest_column(sel["latitude"]["value"], sel["longitude"]["value"])
            out["columns"] = np.array([[j]])
            for c, value in (("latitude", float(a.lat[j])), ("longitude", float(a.lon[j] % 360))):
                coordinates[c] = dict(role="fixed", value=value, requested=sel[c]["value"], column=j,
                                      sampling="the archived column whose centre is nearest the requested point")
            out["latitude"], out["longitude"] = dict(role="fixed"), dict(role="fixed")
        else:
            lats, lons, index = a.layout
            picks = {}
            for c, values in (("latitude", lats), ("longitude", lons)):
                part = sel[c]
                if part["role"] == "fixed":
                    at = part["value"] % 360 if c == "longitude" else part["value"]
                    distance = np.abs((values - at + 180) % 360 - 180) if c == "longitude" else np.abs(values - at)
                    i = int(np.argmin(distance))
                    picks[c] = np.array([i])
                    coordinates[c] = dict(role="fixed", value=float(values[i]), requested=part["value"],
                                          sampling=f"the native {'meridian' if c == 'longitude' else 'row'} "
                                                   "of archived columns nearest the request")
                else:
                    picks[c] = np.arange(values.size)
                    coordinates[c] = (dict(role="reduce", op=part["op"], rule=MEAN[c], nodes=f"{values.size} native "
                                           f"{'rows' if c == 'latitude' else 'meridians'}") if part["role"] == "reduce"
                                      else dict(role=part["role"], sampling="every archived native "
                                                + ("row" if c == "latitude" else "meridian")))
                    if part["role"] != "reduce":
                        edges = a.cell_edges
                        e = None if edges is None else list(map(float, edges[0 if c == "latitude" else 1]))
                        axes[c] = latitude_axis(values, edges=e) if c == "latitude" else longitude_axis(values, edges=e)
                out[c] = dict(role=part["role"])
            grid = index[np.ix_(picks["latitude"], picks["longitude"])]
            out["columns"] = grid
            if lat_role == "reduce":
                w = a.weights[grid]
                if not np.allclose(w, w[:, :1], rtol=1e-12, atol=0):
                    raise SelectionError("The recorded area weights vary along a row; a latitude mean "
                                         "with separable weights is not defined for this archive")
                out["latitude"]["weights"] = w[:, 0]
        out["max_samples"] = MAX_SAMPLES if out["columns"].size == 1 else MAX_STATES
        self._vertical_plan(sel, quantity, out)
        out["sampling"] = ("archived values at native levels and cells; no vertical or horizontal "
                           "interpolation")
        t = sel["time"]
        if t["role"] == "fixed":
            coordinates["time"] = dict(role="fixed", axis=t["axis"], index=t["index"],
                                       seconds=self.manifest["time_axes"][t["axis"]]["values"][t["index"]])
        return out

    def _vertical_plan(self, sel, quantity, out):
        v = sel["vertical"]
        coordinates = out["coordinates"]
        if v["role"] == "none":
            out["read"] = ("surface", [SURFACE_PLANES.index(quantity["plane"])])
            coordinates["vertical"] = dict(role="none")
            return
        if v["role"] == "fixed":
            level = self._level(quantity, v)
            family = self._family(level["time_axis"])
            rows = [level["row"]] if family != "surface" else [0]
            out["read"] = (family, rows)
            coordinates["vertical"] = dict(role="fixed", representation=v["representation"], value=v["value"],
                                           level=level["label"], sampling="a native level; no interpolation")
            return
        cell_axis = self._cells_axis(quantity)
        family = cell_axis["family"]
        out["read"] = (family, list(range(1, self.archive.layers + 1)) if family == "every_step" else None)
        out["axes"]["vertical"] = self._vertical_axis(quantity, v["representation"])
        out["cell_axis"] = cell_axis
        out["companions"]["skin"] = dict(label="Diagnosed skin Tₛ (depth 0)", units="K", surface=True)
        coordinates["vertical"] = dict(role=v["role"], representation=v["representation"],
                                       sampling="native cell centres; the skin is shown as its own level")
        out["notes"].append(SKIN)

    def _read(self, family, index, rows, columns):
        """One saved sample of the given rows at the given (sorted) columns: ``(rows, ncol)``."""
        if family == "every_step":
            return self.archive.every_step(index, rows, columns)
        if family == "surface":
            return np.stack([self.archive.surface(index, r, columns) for r in rows])
        return self.archive.profile(index, rows, columns)

    def samples(self, sel, plan, indices):
        """Yield ``(seconds, values (nv, nlat, nlon), companions)`` per saved sample."""
        grid = plan["columns"]
        flat = grid.ravel()
        order = np.argsort(flat)
        sorted_columns = flat[order]
        family, rows = plan["read"]
        # The parser resolved the axis: a fixed level's own, else the quantity's (the cells').
        times = self.manifest["time_axes"][sel["time"]["axis"]]["values"]
        skin = None
        if "cell_axis" in plan:
            matches = [self._skin_index(plan["cell_axis"], i) for i in indices]
            if all(m is not None for m in matches):
                skin = matches
            else:
                plan["companions"].pop("skin")
                plan["notes"].append("The skin is not shown: it was not sampled at the step of every cell "
                                     "sample selected.")

        def shaped(block):                         # (rows, ncol in sorted order) -> (rows, nlat, nlon)
            out = np.empty_like(block)
            out[:, order] = block
            return out.reshape(block.shape[0], *grid.shape)

        if flat.size == 1 and len(indices) > 1:
            # One column: read the record of the range at once.
            j, first, last = int(flat[0]), indices[0], indices[-1]
            block = self.archive.history(family, j, first, last, rows)
            skin_block = None
            if skin is not None:
                if family == "every_step":
                    skin_block = self.archive.history("every_step", j, first, last, [0])[:, 0]
                else:
                    k = np.array([m[1] for m in skin])
                    lo, hi = int(k.min()), int(k.max())
                    skin_block = self.archive.history("surface", j, lo, hi, [0])[:, 0][k - lo]
            for n, i in enumerate(indices):
                companions = {} if skin_block is None else dict(skin=np.full((1, 1, 1), skin_block[n]))
                yield times[i], block[n].reshape(-1, 1, 1), companions
            return
        for n, i in enumerate(indices):
            values = shaped(self._read(family, i, rows, sorted_columns))
            companions = {}
            if skin is not None:
                skin_family, k = skin[n]
                companions["skin"] = shaped(self._read(skin_family, k, [0], sorted_columns))
            yield times[i], values, companions

    def finish(self, sel, plan, payload):
        comp = payload.get("companions") or {}
        if "skin" in comp:
            payload["surface"] = comp["skin"]["values"]
        v = sel["vertical"]
        profile = v["role"] in ("x", "y") and all(sel[c]["role"] in ("fixed", "reduce")
                                                   for c in ("longitude", "latitude", "time"))
        if profile:
            mean = " (mean)" if any(sel[c]["role"] == "reduce" for c in ("longitude", "latitude", "time")) else ""
            skin = comp.get("skin", {}).get("values")
            depth = self.archive.centres
            payload["markers"] = [] if skin is None else [dict(
                label=f"Surface skin Tₛ{mean} = {skin:.6g} K (depth 0)", value=skin, position=0.0, kind="surface")]
            payload["table"] = dict(columns=["Level", "Depth (m)", "Height above surface (m)", f"T (K){mean}"],
                                    rows=([["skin", 0.0, 0.0, skin]] if skin is not None else []) +
                                         [[k + 1, float(d), float(-d), value]
                                          for k, (d, value) in enumerate(zip(depth, payload["values"]))])
            payload["notes"].append(f"{self.archive.layers} cells; the faces run from 0 to "
                                    f"{self.archive.interfaces[-1]:.6g} m and are not equally spaced.")
        if sel["time"]["role"] in ("x", "y", "reduce"):
            payload["notes"].append("Every archived sample of the range at its own cadence; gaps between "
                                    "kept cycles are real.")

    def frame(self, sel):
        """One saved sample of one native level (or a surface plane) on the archived mesh."""
        layout = self.archive.layout
        if layout is None:
            raise SelectionError(POINTS)
        lats, lons, index = layout
        quantity = self.quantities[sel["quantity"]]
        t = sel["time"]["index"]
        if quantity.get("vertical"):
            level = self._level(quantity, sel["vertical"])
            family = self._family(level["time_axis"])
            flat = self._read(family, t, [level["row"]] if family != "surface" else [0], None)[0]
        else:
            flat = self.archive.surface(t, SURFACE_PLANES.index(quantity["plane"]))
        values = flat[index]
        zero = np.zeros(values.shape)
        return dict(lat=lats, lon=lons, values=values, terrain=zero, height=zero,
                    mask=np.zeros(values.shape, dtype=np.uint8), cells=np.ones(1))


def describe(archive: AirlessArchive) -> tuple[dict, dict]:
    """``(time_axes, explorer)`` of a columnar archive for its viewer manifest."""
    axes = {}
    labels = dict(surface="surface sample", profiles="cell-profile sample", every_step="every-step")
    ids = dict(surface="surface_samples", profiles="profile_samples", every_step="every_step")
    for family, segments in archive.segments.items():
        if not segments:
            continue
        axes[ids[family]] = dict(
            id=ids[family], family=family, label=labels[family], units="s",
            values=archive.times(family).tolist(),
            meaning="forcing time t^(n+1) of the backward-Euler step after which the sample was taken",
            segments=[dict(directory=s.directory, cycle=s.cycle, stage=s.stage, count=s.count,
                           every_steps=s.every, first_step=s.first, steps=s.steps, dt_s=s.dt,
                           cadence=s.cadence_source, timing=s.timing_source) for s in segments],
            provenance="reconstructed as (first + (k + 1) * every) * dt from the timing named in "
                       "each segment")
    every = "every_step" in axes
    surface_axis = "every_step" if every else "surface_samples"
    cell_axis = "every_step" if every else "profile_samples"
    global_ = archive.kind != "points"
    reps = dict(
        native=dict(label="Native level", units=""),
        depth=dict(label="Depth below the local surface d (m, positive downward)", units="m"),
        height_above_surface=dict(label="Height relative to the local surface, −d (m)", units="m"))
    levels = [dict(index=0, kind="surface", row=0, label="Surface · diagnosed skin (depth 0)",
                   depth=0.0, height_above_surface=0.0, time_axis=surface_axis)]
    for k, d in enumerate(archive.centres):
        levels.append(dict(index=k + 1, kind="cell", row=k + 1 if every else k,
                           label=f"Cell {k + 1} centre · depth {float(d):.6g} m",
                           depth=float(d), height_above_surface=float(-d), time_axis=cell_axis))
    if global_:
        weights = archive.weights is not None
        horizontal = dict(roles=["x", "y", "fixed"] + (["reduce"] if weights else []),
                          unavailable={} if weights else dict(reduce=NO_WEIGHTS))
        horizontal_caps = {c: dict(horizontal, **({"reduce": {"mean": MEAN[c]}} if weights else {}))
                           for c in ("longitude", "latitude")}
    else:
        horizontal_caps = {c: dict(roles=["fixed"], unavailable=dict(x=POINTS, y=POINTS, reduce=POINTS))
                           for c in ("longitude", "latitude")}
    time = dict(roles=["x", "y", "fixed", "reduce"], reduce={"mean": MEAN["time"]})
    vertical_reps = ["depth", "height_above_surface"]
    quantities = dict(temperature=dict(
        presentation=dict(name="Temperature", symbol="T", units="K", role="prognostic", origin="stored",
                          definition="Substrate temperature at the native cell centres, and the "
                                     "diagnosed skin temperature at depth 0."),
        sources=["profiles (cell centres)" if not every else "every_step", "surface plane 0 (skin)"],
        evaluation="archived values at native levels; no vertical or horizontal interpolation",
        horizontal="global" if global_ else "points",
        vertical=dict(domain="substrate", representations=reps, levels=levels),
        time_axis=cell_axis,
        coordinates=dict(horizontal_caps, time=time, vertical=dict(
            roles=["x", "y", "fixed"], fixed=["native", *vertical_reps], axis=vertical_reps, per_sample=[],
            unavailable=dict(reduce=NO_CELL_MEAN)))))
    flux = {"incident_shortwave": ("Incident shortwave flux", "SW↓",
                                   "Shortwave flux reaching the surface before reflection."),
            "absorbed_forcing": ("Absorbed surface forcing", "F_abs",
                                 "Every term of the surface balance except emission: (1 − A) SW + ε LW + F₀ "
                                 "+ a (Tₛ − T_ref)."),
            "emitted_longwave": ("Emitted longwave flux", "εσTₛ⁴", "Thermal emission of the skin, ε σ Tₛ⁴."),
            "conducted_flux": ("Conducted flux into the first cell", "q_s",
                               "Conductive flux from the skin into the first cell, positive downward.")}
    for key, (name, symbol, definition) in flux.items():
        if "surface_samples" not in axes:
            break
        quantities[key] = dict(
            presentation=dict(name=name, symbol=symbol, units="W m⁻²", role="diagnostic",
                              origin="recorded", definition=definition),
            sources=[f"surface plane {SURFACE_PLANES.index(key)}"], plane=key,
            evaluation="archived values; no interpolation", horizontal="global" if global_ else "points",
            vertical=None, time_axis="surface_samples",
            coordinates=dict(horizontal_caps, time=time, vertical=dict(roles=["none"])))
    horizontal = dict(kind=archive.kind, source=archive.kind_source, columns=archive.columns,
                      area_weights=global_ and archive.weights is not None)
    if global_:
        lats, lons, _ = archive.layout
        horizontal.update(nlat=int(lats.size), nlon=int(lons.size),
                          cells="regular cells" if archive.cell_edges is not None else "nodes")
    else:
        horizontal["points"] = [dict(column=j, latitude=float(a), longitude=float(b) % 360)
                                for j, (a, b) in enumerate(zip(archive.lat, archive.lon))]
    return axes, dict(quantities=quantities, horizontal=horizontal)


__all__ = ["ColumnArchiveProvider", "describe", "level_time_axis"]
