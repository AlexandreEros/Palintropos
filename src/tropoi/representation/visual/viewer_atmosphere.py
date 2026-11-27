"""Selections of atmospheric archives, on the evaluators of FrameProvider.

Each product calls one existing evaluator unchanged and only rearranges
its result into the viewer's common payload (axes, values, markers, notes):

* ``map``: :meth:`FrameProvider.slice` (exact band-limited synthesis
  sampled on the 2° display mesh, then the declared vertical interpolation);
* ``column``: :meth:`FrameProvider.column` (exact point synthesis, or the
  native Gaussian node when snapped);
* sections: :meth:`FrameProvider.section` (exact synthesis on native full
  levels, 2° horizontal spacing);
* ``zonal_mean``: :meth:`FrameProvider.zonal_mean` (native Gaussian grid,
  plain longitude mean, equal-weight time mean);
* ``time_series`` and ``time_section``: :meth:`FrameProvider.column` at each
  saved state of the range, at most :data:`MAX_STATES` of them.
"""
from __future__ import annotations

from tropoi.representation.visual.selection import (
    MAX_STATES, SelectionError, axis, latitude_axis, level_edges, longitude_axis, time_axis)

#: Quantities the column and section evaluators return per level.
STRUCTURE_FIELDS = ("temperature", "u", "v", "pressure", "geopotential_height")
ZONAL_FIELDS = ("temperature", "u", "v")

REPRESENTATIONS = {
    "native": dict(label="Native full level", units=""),
    "sigma": dict(label="Nominal σ = p/pₛ", units=""),
    "pressure": dict(label="Pressure p", units="Pa"),
    "geopotential_height": dict(label="Geopotential height Z = Φ/g above the model datum", units="m"),
    "height_above_surface": dict(
        label="Geopotential height above the local surface, Z − Zₛ", units="m",
        definition="Z − Zₛ with Zₛ the terrain height: a geopotential height relative to the "
                   "lower boundary, not a geometric altitude"),
}
PSI = dict(name="Meridional mass streamfunction", symbol="ψ", units="kg s⁻¹", role="diagnostic",
           origin="derived", definition=(
               "ψ(φ, σ_{k+½}) = 2πa cos φ / g · Σ_{j≤k} [pₛ v_j] Δσ_j, integrated downward from ψ = 0 at "
               "σ = 0; positive for northward mass transport above the interface."))


def _excluded(representation, field):
    """A surface of constant p shows p; of constant Z shows Z: not offered."""
    return ((representation == "pressure" and field == "pressure") or
            (representation in ("geopotential_height", "height_above_surface")
             and field in ("geopotential_height", "geopotential")))


def describe(manifest) -> dict:
    """The explorer description of an atmospheric viewer manifest."""
    fields, ops = manifest["fields"], manifest["operations"]
    full = (manifest["axes"].get("vertical") or {}).get("full_levels") or []
    interfaces = (manifest["axes"].get("vertical") or {}).get("interfaces") or []
    structure = ops.get("structure", [])
    unavailable = {k: v for k, v in manifest["unavailable"].items() if isinstance(v, str)}
    if "structure" not in ops:
        unavailable.setdefault("column", "This archive has no atmospheric column evaluator")
    unavailable.setdefault("zonal_mean", "Zonal means need the full atmospheric column evaluator")
    levels = [dict(index=k, label=f"{k + 1} · σ = {float(s):.6g}", sigma=float(s))
              for k, s in enumerate(full)]
    spectral = manifest["source"].get("solver") in ("bve", "swe", "pe")
    quantities = {}
    for key in ops.get("map", []):
        if key == "wind":
            continue
        field = fields[key]
        q = dict(presentation=field["presentation"], sources=field.get("inputs") or [key],
                 evaluation=("exact band-limited synthesis sampled on a 2° display mesh" if spectral
                             and key != "surface_temperature" else
                             "bilinear between native slab nodes on the 2° display mesh"),
                 horizontal="global", time_axis="snapshots", products={}, unavailable={})
        if field.get("levels"):
            reps = ["native"] + [r for r in ("sigma", "pressure", "geopotential_height",
                                             "height_above_surface")
                                 if key in ops.get(r, []) and not _excluded(r, key)]
            q["vertical"] = dict(domain="atmosphere", levels=levels,
                                 representations={r: REPRESENTATIONS[r] for r in REPRESENTATIONS})
            q["products"]["map"] = dict(vertical=reps)
            if key in STRUCTURE_FIELDS and "column" in structure:
                profile = [r for r in ("geopotential_height", "height_above_surface", "pressure", "sigma")
                           if r != "height_above_surface" or "height_above_surface" in ops]
                q["products"].update(
                    column=dict(vertical=profile, evaluation="exact point synthesis on native full levels"),
                    meridional_section=dict(vertical=["sigma"]), zonal_section=dict(vertical=["sigma"]),
                    time_series=dict(vertical=["native"]), time_section=dict(vertical=["sigma"]))
            if key in ZONAL_FIELDS and "zonal_mean" in structure:
                q["products"]["zonal_mean"] = dict(
                    vertical=["sigma"], reductions=dict(longitude=["mean"], time=["mean"]),
                    evaluation="plain mean over the native longitudes of the Gaussian state grid")
        else:
            q["vertical"] = None
            q["products"]["map"] = {}
        quantities[key] = q
    if "mass_streamfunction" in structure:
        quantities["mass_streamfunction"] = dict(
            presentation=PSI, sources=["v", "ln_ps"], horizontal="global", time_axis="snapshots",
            evaluation="native Gaussian grid; [pₛ v] formed per state before the longitude mean",
            vertical=dict(domain="atmosphere", representations={"sigma": REPRESENTATIONS["sigma"]},
                          levels=[dict(index=k, label=f"interface {k} · σ = {float(s):.6g}", sigma=float(s))
                                  for k, s in enumerate(interfaces)]),
            products=dict(zonal_mean=dict(vertical=["sigma"],
                                          reductions=dict(longitude=["mean"], time=["mean"]))),
            unavailable={})
    for key, product in manifest["diagnostics"].items():
        if product["kind"] == "series":
            quantities["series:" + key] = dict(
                presentation=product["presentation"], sources=[product["path"]], horizontal="none",
                vertical=None, time_axis=product["time_axis"], recorded=key,
                evaluation="recorded by the run at its own cadence",
                products=dict(recorded_series={}), unavailable={})
    return dict(quantities=quantities, unavailable=unavailable,
                horizontal=dict(kind="spectral" if spectral else manifest["geometry"].get("grid"),
                                grid=manifest["geometry"].get("grid")))


def _time(provider, index):
    return dict(axis="snapshots", index=index,
                seconds=provider.manifest["time_axes"]["snapshots"]["values"][index])


def _vertical(representation, d, surface):
    """Profile coordinate per level and at the surface, from a column result."""
    if representation == "geopotential_height":
        return d["geopotential_height"], d["surface_height"], "linear", False
    if representation == "height_above_surface":
        return [z - d["surface_height"] for z in d["geopotential_height"]], 0.0, "linear", False
    if representation == "pressure":
        return d["pressure"], d["surface_pressure"], "log", True
    return d["sigma"], 1.0, "log", True


def _sigma_axis(sigma, label):
    return axis("vertical", "sigma", label, "", sigma, edges=level_edges(sigma), scale="log", downward=True,
                rows=[dict(kind="level", label=f"level {k + 1}") for k in range(len(sigma))])


def select(provider, sel):
    product, key = sel["product"], sel["quantity"]
    quantity = provider.manifest["explorer"]["quantities"][key]
    payload = dict(product=product, quantity=key, presentation=quantity["presentation"], notes=[])
    physics = provider.manifest["physics"]
    reference = physics.get("reference_pressure") or physics.get("surface_pressure")
    if product == "map":
        vertical = sel["vertical"]
        rep = vertical.get("representation", "native")
        value = vertical.get("value", 0) if quantity["vertical"] else 0
        return "frame", provider.slice(sel["time"]["index"], key, rep, value)
    if product == "column":
        t = sel["time"]["index"]
        d = provider.column(t, sel["latitude"]["value"], sel["longitude"]["value"],
                            bool(sel["options"].get("snap", True)))
        rep = sel["vertical"]["representation"]
        y, surface, scale, downward = _vertical(rep, d, None)
        markers = []
        if key == "temperature":
            low = d["lowest_air"]
            markers.append(dict(label=f"Lowest full level air T = {low['value']:.6g} K (σ {low['sigma']:.6g})",
                                value=low["value"], position=y[-1], kind="level"))
            if d["surface_temperature"]:
                markers.append(dict(label=f"Slab surface Tₛ = {d['surface_temperature']['value']:.6g} K",
                                    value=d["surface_temperature"]["value"], position=surface, kind="surface"))
        heights = [z - d["surface_height"] for z in d["geopotential_height"]]
        payload.update(
            x=dict(label=quantity["presentation"]["name"], units=quantity["presentation"]["units"]),
            y=axis("vertical", rep, REPRESENTATIONS[rep]["label"], REPRESENTATIONS[rep]["units"], y,
                   scale=scale, downward=downward),
            values=d[key], markers=markers, time=_time(provider, t),
            location=dict(latitude=d["latitude"], longitude=d["longitude"], sampling=d["sampling"]),
            sampling=d["sampling"],
            table=dict(columns=["Level", "σ", "p (Pa)", "Z (m)", "Z − Zₛ (m)", "T (K)", "u (m s⁻¹)", "v (m s⁻¹)"],
                       rows=[[k + 1, d["sigma"][k], d["pressure"][k], d["geopotential_height"][k], heights[k],
                              d["temperature"][k], d["u"][k], d["v"][k]] for k in range(len(d["sigma"]))]),
            surface=dict(pressure=d["surface_pressure"], height=d["surface_height"]),
            notes=[d["lowest_air"]["definition"] + ".",
                   (d["surface_temperature"]["definition"] + "; " + d["surface_temperature"]["sampling"] + "."
                    if d["surface_temperature"] else "This archive has no slab surface temperature at this time."),
                   *d["conventions"].values()])
        return "json", payload
    if product in ("meridional_section", "zonal_section"):
        t = sel["time"]["index"]
        meridional = product == "meridional_section"
        at = sel["longitude"]["value"] if meridional else sel["latitude"]["value"]
        d = provider.section(t, "meridional" if meridional else "zonal", at)
        x = latitude_axis(d["points"]) if meridional else longitude_axis(d["points"])
        y = _sigma_axis(d["sigma"], "Nominal σ, native full levels (log)")
        if reference:
            y["secondary"] = dict(label="nominal p ≈ σ × p_ref", factor=reference, units="Pa")
        payload.update(
            x=x, y=y, values=d[key], time=_time(provider, t), sampling=d["sampling"],
            location={("longitude" if meridional else "latitude"): d["at"]},
            companions=dict(pressure=dict(label="p", units="Pa", values=d["pressure"]),
                            geopotential_height=dict(label="Z", units="m", values=d["geopotential_height"])))
        return "json", payload
    if product == "zonal_mean":
        time = sel["time"]
        first, last = (time["index"], time["index"]) if time["role"] == "fixed" else (time["first"], time["last"])
        d = provider.zonal_mean(first, last)
        psi = key == "mass_streamfunction"
        y = _sigma_axis(d["sigma_interfaces"] if psi else d["sigma"],
                        "Nominal σ at layer interfaces (log)" if psi else "Nominal σ, native full levels (log)")
        if reference:
            y["secondary"] = dict(label="nominal p ≈ σ × p_ref", factor=reference, units="Pa")
        notes = ["Native grid: " + d["grid"]["latitudes"] + "; " + d["grid"]["longitudes"] + "."]
        if psi:
            r = d["column_flux_residual"]
            notes.append(f"Bottom (σ = 1) whole-column flux: max |ψ| {r['max_abs']:.4g} kg s⁻¹, against max |ψ| "
                         f"{r['max_abs_interior']:.4g} kg s⁻¹ in the section. {r['note']}.")
        payload.update(
            x=latitude_axis(d["lat"], label=f"Latitude ({len(d['lat'])} Gauss–Legendre nodes)"), y=y,
            values=d["streamfunction"] if psi else d[key], sample=d["sample"],
            time=dict(axis="snapshots", first=first, last=last, count=d["sample"]["count"]),
            sampling=d["definitions"]["streamfunction" if psi else "zonal_mean"],
            companions=None if psi else dict(surface_pressure=dict(label="[pₛ]", units="Pa",
                                                                   values=d["surface_pressure"])),
            notes=notes)
        return "json", payload
    if product in ("time_series", "time_section"):
        time = sel["time"]
        first, last = time["first"], time["last"]
        if last - first + 1 > MAX_STATES:
            raise SelectionError(f"Select at most {MAX_STATES} saved states for one history")
        lat, lon = sel["latitude"]["value"], sel["longitude"]["value"]
        snap = bool(sel["options"].get("snap", True))
        columns = [provider.column(t, lat, lon, snap) for t in range(first, last + 1)]
        times = [c["time"] for c in columns]
        c0 = columns[0]
        location = dict(latitude=c0["latitude"], longitude=c0["longitude"], sampling=c0["sampling"])
        if product == "time_series":
            level = int(sel["vertical"]["value"])
            if not 0 <= level < len(c0["sigma"]):
                raise SelectionError("Native level outside the saved axis")
            payload.update(x=time_axis(times), y=dict(label=quantity["presentation"]["name"],
                                                      units=quantity["presentation"]["units"]),
                           values=[c[key][level] for c in columns], level=f"native level {level + 1}",
                           location=location, sampling=c0["sampling"] + ", at every saved state of the range")
        else:
            payload.update(x=time_axis(times), y=_sigma_axis(c0["sigma"], "Nominal σ, native full levels (log)"),
                           values=[[c[key][k] for c in columns] for k in range(len(c0["sigma"]))],
                           location=location, sampling=c0["sampling"] + ", at every saved state of the range")
        payload["time"] = dict(axis="snapshots", first=first, last=last, count=len(times))
        payload["notes"].append("Saved states only: variations between saved states are not represented.")
        return "json", payload
    raise SelectionError("Product not offered by atmospheric archives")
