"""Viewer selections of columnar substrate archives (airless runs).

Values are read, never reconstructed: a map is one archived sample of one
native level of every column; a section, a column profile or a time history
is the archived samples of the native cells of the selected columns. A
location selects the column whose centre is nearest on the sphere (the
column it reports), and a vertical value must be a native level: the
diagnosed skin at depth 0, or a cell centre. Values between native depths
are refused, not interpolated, and the skin is never joined to a cell
profile across a time at which it was not sampled.
"""
from __future__ import annotations

import math

import numpy as np

from tropoi.representation.archive.airless import SURFACE_PLANES, AirlessArchive
from tropoi.representation.visual.selection import (
    MAX_SAMPLES, SelectionError, axis, latitude_axis, level_time_axis, longitude_axis, time_axis)

SKIN = ("The surface temperature is the diagnosed, massless skin of each column (depth 0). "
        "It is its own native level, not the top cell, and it is drawn apart from the cells.")


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
    # Level and time resolution
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

    def _read_level(self, level, index, columns=None):
        family = self.manifest["time_axes"][level["time_axis"]]["family"]
        if family == "every_step":
            return self.archive.every_step(index, [level["row"]], columns)[0]
        if level["kind"] == "surface":
            return self.archive.surface(index, 0, columns)
        return self.archive.profile(index, [level["row"]], columns)[0]

    def _flux(self, quantity, index, columns=None):
        return self.archive.surface(index, SURFACE_PLANES.index(quantity["plane"]), columns)

    def _cells_axis(self, quantity):
        levels = quantity["vertical"]["levels"]
        return self.manifest["time_axes"][levels[-1]["time_axis"]]

    def _skin_at(self, cell_axis, index):
        """The surface-sample index taken at the same step as cell sample
        ``index``, or None (the skin is then not shown for that time)."""
        if cell_axis["family"] == "every_step":
            return ("every_step", index)
        steps = self.archive.steps("surface")
        step = self.archive.steps(cell_axis["family"])[index]
        k = int(np.searchsorted(steps, step))
        return ("surface", k) if k < steps.size and steps[k] == step else None

    def _vertical_axis(self, quantity, rep, skin=True):
        a = self.archive
        sign = -1.0 if rep == "height_above_surface" else 1.0
        label = quantity["vertical"]["representations"][rep]["label"]
        rows = [dict(kind="cell", label=f"cell {k + 1}", depth=float(d))
                for k, d in enumerate(a.centres)]
        out = axis("vertical", rep, label, "m", sign * a.centres, edges=sign * a.interfaces,
                   scale="linear", downward=rep == "depth", rows=rows)
        if skin:
            out["surface_row"] = dict(label="Surface · diagnosed skin (depth 0)", value=0.0)
        return out

    def _payload(self, sel, quantity, **parts):
        out = dict(product=sel["product"], quantity=sel["quantity"],
                   presentation=quantity["presentation"], notes=[])
        out.update(parts)
        return out

    def _time(self, axis_id, index):
        values = self.manifest["time_axes"][axis_id]["values"]
        return dict(axis=axis_id, index=index, seconds=values[index])

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------

    def select(self, sel):
        quantity = self.quantities[sel["quantity"]]
        handler = getattr(self, "_" + sel["product"], None)
        if handler is None:
            raise SelectionError("Product not offered by columnar archives")
        return handler(sel, quantity)

    def _map(self, sel, quantity):
        layout = self.archive.layout
        if layout is None:
            raise SelectionError("A diagnostic point collection has no map")
        lats, lons, index = layout
        t = sel["time"]["index"]
        if quantity.get("vertical"):
            flat = self._read_level(self._level(quantity, sel["vertical"]), t)
        else:
            flat = self._flux(quantity, t)
        values = flat[index]
        zero = np.zeros(values.shape)
        return "frame", dict(lat=lats, lon=lons, values=values, terrain=zero, height=zero,
                             mask=np.zeros(values.shape, dtype=np.uint8), cells=np.ones(1))

    def _section(self, sel, quantity, along):
        layout = self.archive.layout
        if layout is None:
            raise SelectionError("A diagnostic point collection has no sections")
        lats, lons, index = layout
        rep = sel["vertical"]["representation"]
        cell_axis = self._cells_axis(quantity)
        t = sel["time"]["index"]
        if along == "latitude":
            at = sel["longitude"]["value"] % 360
            j = int(np.argmin(np.abs((lons - at + 180) % 360 - 180)))
            columns, at_actual, x = index[:, j], float(lons[j]), latitude_axis(lats)
        else:
            at = sel["latitude"]["value"]
            i = int(np.argmin(np.abs(lats - at)))
            columns, at_actual, x = index[i, :], float(lats[i]), longitude_axis(lons)
        order = np.argsort(columns)
        cells = np.empty((self.archive.layers, columns.size))
        cells[:, order] = self._cells(cell_axis, t, columns[order])
        skin = self._skin(cell_axis, t, columns, order)
        edges = self.archive.cell_edges
        if edges is not None:
            x["edges"] = list(map(float, edges[0] if along == "latitude" else edges[1]))
        y = self._vertical_axis(quantity, rep)
        name = "longitude" if along == "latitude" else "latitude"
        return "json", self._payload(
            sel, quantity, x=x, y=y, values=cells.tolist(),
            surface=None if skin is None else skin.tolist(),
            time=self._time(cell_axis["id"], t),
            location={name: at_actual, "requested": at,
                      "sampling": f"the native {name} row of archived columns nearest the request"},
            sampling="archived samples at native cell centres; cells drawn between their archived faces",
            notes=[SKIN] + ([] if skin is not None else
                            ["The skin was not sampled at this cell-sample step, so it is not shown."]))

    def _meridional_section(self, sel, quantity):
        return self._section(sel, quantity, "latitude")

    def _zonal_section(self, sel, quantity):
        return self._section(sel, quantity, "longitude")

    def _cells(self, cell_axis, t, columns):
        if cell_axis["family"] == "every_step":
            return self.archive.every_step(t, None, columns)[1:]
        return self.archive.profile(t, None, columns)

    def _skin(self, cell_axis, t, columns, order=None):
        match = self._skin_at(cell_axis, t)
        if match is None:
            return None
        family, k = match
        cols = columns if order is None else columns[order]
        values = (self.archive.every_step(k, [0], cols)[0] if family == "every_step"
                  else self.archive.surface(k, 0, cols))
        if order is None:
            return values
        out = np.empty(columns.size)
        out[order] = values
        return out

    def _column(self, sel, quantity):
        j, location = self._column_index(sel)
        rep = sel["vertical"]["representation"]
        cell_axis = self._cells_axis(quantity)
        t = sel["time"]["index"]
        cells = self._cells(cell_axis, t, np.array([j]))[:, 0]
        skin = self._skin(cell_axis, t, np.array([j]))
        y = self._vertical_axis(quantity, rep)
        markers = []
        if skin is not None:
            markers.append(dict(label=f"Surface skin Tₛ = {float(skin[0]):.6g} K (depth 0)",
                                value=float(skin[0]), position=0.0, kind="surface"))
        depth = self.archive.centres
        table = dict(columns=["Level", "Depth (m)", "Height above surface (m)", "T (K)"],
                     rows=([["skin", 0.0, 0.0, float(skin[0])]] if skin is not None else []) +
                          [[k + 1, float(d), float(-d), float(v)]
                           for k, (d, v) in enumerate(zip(depth, cells))])
        return "json", self._payload(
            sel, quantity, x=dict(label=quantity["presentation"]["name"],
                                  units=quantity["presentation"]["units"]),
            y=y, values=cells.tolist(), markers=markers, table=table,
            time=self._time(cell_axis["id"], t), location=location,
            sampling="archived cell-centre temperatures on the nonuniform native mesh",
            notes=[SKIN, f"{self.archive.layers} cells; the faces run from 0 to "
                         f"{self.archive.interfaces[-1]:.6g} m and are not equally spaced."])

    def _column_index(self, sel):
        lat, lon = sel["latitude"]["value"], sel["longitude"]["value"]
        j = self.archive.nearest_column(lat, lon)
        return j, dict(latitude=float(self.archive.lat[j]), longitude=float(self.archive.lon[j] % 360),
                       column=j, requested=dict(latitude=lat, longitude=lon % 360),
                       sampling="the archived column whose centre is nearest the requested point")

    def _range(self, sel, axis_id):
        first, last = sel["time"]["first"], sel["time"]["last"]
        if last - first + 1 > MAX_SAMPLES:
            raise SelectionError(f"Select at most {MAX_SAMPLES} saved samples for one history")
        return first, last, self.manifest["time_axes"][axis_id]["values"][first:last + 1]

    def _time_series(self, sel, quantity):
        j, location = self._column_index(sel)
        if quantity.get("vertical"):
            level = self._level(quantity, sel["vertical"])
            axis_id = level["time_axis"]
            family = self.manifest["time_axes"][axis_id]["family"]
            first, last, times = self._range(sel, axis_id)
            rows = [level["row"]] if family != "surface" else [0]
            label = level["label"]
        else:
            axis_id = quantity["time_axis"]
            family = "surface"
            first, last, times = self._range(sel, axis_id)
            rows = [SURFACE_PLANES.index(quantity["plane"])]
            label = "surface"
        values = self.archive.history(family, j, first, last, rows)[:, 0]
        return "json", self._payload(
            sel, quantity, x=time_axis(times, label="Saved sample time (forcing time of the step)"),
            y=dict(label=quantity["presentation"]["name"], units=quantity["presentation"]["units"]),
            values=values.tolist(), level=label, location=location,
            time=dict(axis=axis_id, first=first, last=last, count=len(times)),
            sampling=f"every archived sample of the {self.manifest['time_axes'][axis_id]['label']} "
                     "record, at its own cadence; gaps between kept cycles are real")

    def _time_section(self, sel, quantity):
        j, location = self._column_index(sel)
        rep = sel["vertical"]["representation"]
        cell_axis = self._cells_axis(quantity)
        first, last, times = self._range(sel, cell_axis["id"])
        if cell_axis["family"] == "every_step":
            block = self.archive.history("every_step", j, first, last)
            cells, skin = block[:, 1:].T, block[:, 0]
            note = "The skin and the cells come from the same every-step record."
        else:
            cells = self.archive.history("profiles", j, first, last).T
            steps = self.archive.steps("profiles")[first:last + 1]
            surface_steps = self.archive.steps("surface")
            k = np.searchsorted(surface_steps, steps)
            ok = (k < surface_steps.size) & (surface_steps[np.minimum(k, surface_steps.size - 1)] == steps)
            skin = None
            note = "The skin is not shown: it was not sampled at the cell-sample steps."
            if ok.all():
                lo, hi = int(k.min()), int(k.max())
                history = self.archive.history("surface", j, lo, hi, [0])[:, 0]
                skin = history[k - lo]
                note = "The skin row is the surface sample taken at the same step as each cell sample."
        return "json", self._payload(
            sel, quantity, x=time_axis(times, label="Saved sample time (forcing time of the step)"),
            y=self._vertical_axis(quantity, rep), values=cells.tolist(),
            surface=None if skin is None else skin.tolist(), location=location,
            time=dict(axis=cell_axis["id"], first=first, last=last, count=len(times)),
            sampling="archived cell-centre samples at their own cadence; no time interpolation",
            notes=[SKIN, note])


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
    vertical_reps = ["depth", "height_above_surface"]
    products = dict(column=dict(vertical=vertical_reps), time_series=dict(vertical=["native", *vertical_reps]),
                    time_section=dict(vertical=vertical_reps))
    if global_:
        products.update(map=dict(vertical=["native", *vertical_reps]),
                        meridional_section=dict(vertical=vertical_reps),
                        zonal_section=dict(vertical=vertical_reps))
    unavailable = dict(zonal_mean="Zonal reductions of columnar archives are not offered in this release")
    if not global_:
        reason = ("A diagnostic point collection is not a global field: it has no map, sections "
                  "or area weights")
        unavailable.update(map=reason, meridional_section=reason, zonal_section=reason)
    quantities = dict(temperature=dict(
        presentation=dict(name="Temperature", symbol="T", units="K", role="prognostic", origin="stored",
                          definition="Substrate temperature at the native cell centres, and the "
                                     "diagnosed skin temperature at depth 0."),
        sources=["profiles (cell centres)" if not every else "every_step", "surface plane 0 (skin)"],
        evaluation="archived values at native levels; no vertical or horizontal interpolation",
        horizontal="global" if global_ else "points",
        vertical=dict(domain="substrate", representations=reps, levels=levels),
        time_axis=cell_axis, products=products, unavailable=unavailable))
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
        p = dict(time_series={})
        if global_:
            p["map"] = {}
        quantities[key] = dict(
            presentation=dict(name=name, symbol=symbol, units="W m⁻²", role="diagnostic",
                              origin="recorded", definition=definition),
            sources=[f"surface plane {SURFACE_PLANES.index(key)}"], plane=key,
            evaluation="archived values; no interpolation", horizontal="global" if global_ else "points",
            vertical=None, time_axis="surface_samples", products=p, unavailable=unavailable)
    horizontal = dict(kind=archive.kind, source=archive.kind_source, columns=archive.columns,
                      area_weights=global_ and archive.weights is not None)
    if global_:
        lats, lons, _ = archive.layout
        horizontal.update(nlat=int(lats.size), nlon=int(lons.size),
                          cells="regular cells" if archive.cell_edges is not None else "nodes")
    else:
        horizontal["points"] = [dict(column=j, latitude=float(a), longitude=float(b) % 360)
                                for j, (a, b) in enumerate(zip(archive.lat, archive.lon))]
    return axes, dict(quantities=quantities, horizontal=horizontal, unavailable=unavailable)


__all__ = ["ColumnArchiveProvider", "describe", "level_time_axis"]
