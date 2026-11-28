"""Selections of atmospheric archives: sample sets per role, existing kernels.

The selection's roles choose a set of sample points per coordinate; each
saved state is then evaluated on the tensor product of the horizontal sets
with the kernels that already defined the viewer's numbers, unchanged:

* native full levels: :func:`~tropoi.representation.diagnostics.pe_levels.
  reconstruct_levels` (exact band-limited synthesis at any points);
* a fixed σ, p, Z or Z − Zₛ surface: :func:`~tropoi.representation.
  diagnostics.pe_surfaces.pressure_surface` (linear in ln σ between the
  bracketing native levels; masked, never extrapolated);
* surface fields: exact synthesis of ln pₛ, the terrain or a BVE/SWE state;
  the slab temperature bilinear between native slab nodes.

Horizontal sample sets, by role:

* ``x``/``y``: the 2° display mesh (latitudes −90…90; longitudes 0…360 with
  the closing meridian), as the sections always used;
* ``fixed``: the requested point, or the nearest native Gaussian node when
  snapping is requested on a Gaussian grid;
* ``reduce``: the native Gaussian state grid, where the means are exact
  quadratures (longitude: plain mean over the equally spaced longitudes;
  latitude: Gauss–Legendre weights, ∝ cos φ dφ). When either horizontal
  coordinate is reduced, an axis along the other is also taken at the native
  nodes, so the whole result lives on the grid its quadrature is defined on
  (the zonal mean's long-standing rule).

The globe of the main page draws one saved state on one surface; it asks for
the binary frame (``options.format = "frame"``), which is
:meth:`FrameProvider.slice`, the same evaluator as before.
"""
from __future__ import annotations

import numpy as np

from tropoi.representation.diagnostics.held_suarez_sections import state_grid
from tropoi.representation.diagnostics.pe_levels import LEVEL_FIELDS, reconstruct_levels
from tropoi.representation.diagnostics.pe_structure import mass_streamfunction
from tropoi.representation.diagnostics.pe_surfaces import pressure_surface
from tropoi.representation.visual.selection import (
    MAX_STATES, SelectionError, axis, latitude_axis, longitude_axis)
from tropoi.spatial.grids.quadrature import gauss_latitude_nodes
from tropoi.spatial.transforms.host_synthesis import synthesize_latlon, synthesize_wind_latlon

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
#: The space in which values on a fixed surface between native levels are obtained.
INTERPOLATION = {
    "native": None,
    "sigma": "linear in ln σ between the bracketing native full levels",
    "pressure": "linear in ln σ (= ln p in a column) between the bracketing native full levels "
                "at their hydrostatic positions",
    "geopotential_height": "inversion of Φ, piecewise linear in ln σ between native levels",
    "height_above_surface": "inversion of Φ, piecewise linear in ln σ between native levels, "
                            "target Φ₀ = g (h + Zₛ) per column",
}
PSI = dict(name="Meridional mass streamfunction", symbol="ψ", units="kg s⁻¹", role="diagnostic",
           origin="derived", definition=(
               "ψ(φ, σ_{k+½}) = 2πa cos φ / g · Σ_{j≤k} [pₛ v_j] Δσ_j, integrated downward from ψ = 0 at "
               "σ = 0; positive for northward mass transport above the interface."))
CONVENTIONS = dict(
    pressure="p_k = σ_k pₛ at the nominal full level (Pa)",
    geopotential_height=("Z_k = Φ_k/g, Simmons–Burridge full-level geopotential, absolute, at the "
                         "hydrostatic level position σʰ_k (m above the model datum)"),
    wind="u eastward, v northward (m s⁻¹)")
LOWEST_AIR = "Air temperature at the lowest full level, not a surface temperature"
SLAB = "Slab surface temperature Tₛ, a stored grid-point prognostic"
#: Quantities a native-level reconstruction provides along a vertical axis.
AXIS_FIELDS = (*LEVEL_FIELDS, "wind_speed")
NO_VERTICAL_MEAN = ("a vertical mean needs a mass (Δp) weighting rule, which this release does not "
                    "define")
NO_QUADRATURE = ("this archive's native grid is not a Gaussian latitude–longitude grid, so no "
                 "quadrature rule for a horizontal mean is defined")
MEAN = dict(
    longitude="plain mean over the equally spaced longitudes of the native Gaussian grid",
    latitude="area-weighted mean over the native Gauss–Legendre latitudes (Gauss weights, ∝ cos φ dφ)",
    time="equal-weight mean over the saved states first..last inclusive")


def _excluded(representation, field):
    """A surface of constant p shows p; of constant Z shows Z: not offered."""
    return ((representation == "pressure" and field == "pressure") or
            (representation in ("geopotential_height", "height_above_surface")
             and field in ("geopotential_height", "geopotential")))


def _quadrature(manifest):
    g = manifest["geometry"]
    return g.get("grid") == "latlon" and bool(g.get("nlat")) and bool(g.get("nlon"))


def _horizontal_caps(quadrature):
    roles = ["x", "y", "fixed"] + (["reduce"] if quadrature else [])
    out = dict(roles=roles)
    if quadrature:
        return {c: dict(out, reduce={"mean": MEAN[c]}) for c in ("longitude", "latitude")}
    return {c: dict(out, unavailable=dict(reduce=NO_QUADRATURE)) for c in ("longitude", "latitude")}


def describe(manifest) -> dict:
    """The explorer description of an atmospheric viewer manifest: per
    quantity, what each coordinate accepts."""
    fields, ops = manifest["fields"], manifest["operations"]
    full = (manifest["axes"].get("vertical") or {}).get("full_levels") or []
    interfaces = (manifest["axes"].get("vertical") or {}).get("interfaces") or []
    structure = ops.get("structure", [])
    quadrature = _quadrature(manifest)
    levels = [dict(index=k, label=f"{k + 1} · σ = {float(s):.6g}", sigma=float(s))
              for k, s in enumerate(full)]
    spectral = manifest["source"].get("solver") in ("bve", "swe", "pe")
    time = dict(roles=["x", "y", "fixed", "reduce"], reduce={"mean": MEAN["time"]})
    no_axis = "this archive has no native-level column evaluator" if "structure" not in ops else None
    quantities = {}
    for key in ops.get("map", []):
        if key == "wind":
            continue
        field = fields[key]
        q = dict(presentation=field["presentation"], sources=field.get("inputs") or [key],
                 evaluation=("exact band-limited synthesis" if spectral and key != "surface_temperature"
                             else "bilinear between native slab nodes"),
                 horizontal="global", time_axis="snapshots")
        coordinates = dict(_horizontal_caps(quadrature), time=time)
        if field.get("levels"):
            fixed = ["native"] + [r for r in ("sigma", "pressure", "geopotential_height", "height_above_surface")
                                  if key in ops.get(r, []) and not _excluded(r, key)]
            on_axis = "structure" in ops and key in AXIS_FIELDS
            per_sample = [r for r in ("geopotential_height", "height_above_surface", "pressure")
                          if (r != "height_above_surface" or "height_above_surface" in ops)] if on_axis else []
            vertical = dict(roles=["fixed"] + (["x", "y"] if on_axis else []), fixed=fixed,
                            axis=["sigma"] if on_axis else [], per_sample=per_sample,
                            unavailable=dict(reduce=NO_VERTICAL_MEAN))
            if not on_axis:
                vertical["unavailable"].update(x=no_axis or "not reconstructed on native levels",
                                               y=no_axis or "not reconstructed on native levels")
            coordinates["vertical"] = vertical
            q["vertical"] = dict(domain="atmosphere", levels=levels,
                                 representations={r: REPRESENTATIONS[r] for r in REPRESENTATIONS})
        else:
            coordinates["vertical"] = dict(roles=["none"])
            q["vertical"] = None
        q["coordinates"] = coordinates
        quantities[key] = q
    if "zonal_mean" in structure and quadrature:
        only_mean = "ψ is defined from the longitude mean [pₛ v]"
        quantities["mass_streamfunction"] = dict(
            presentation=PSI, sources=["v", "ln_ps"], horizontal="global", time_axis="snapshots",
            evaluation="native Gaussian grid; [pₛ v] formed per state before the longitude mean",
            vertical=dict(domain="atmosphere", representations={"sigma": REPRESENTATIONS["sigma"]},
                          levels=[dict(index=k, label=f"interface {k} · σ = {float(s):.6g}", sigma=float(s))
                                  for k, s in enumerate(interfaces)]),
            coordinates=dict(
                longitude=dict(roles=["reduce"], reduce={"mean": MEAN["longitude"]},
                               unavailable=dict(x=only_mean, y=only_mean, fixed=only_mean)),
                latitude=dict(roles=["x", "y"], unavailable=dict(
                    fixed="ψ is offered as a latitude × σ field", reduce="ψ is offered as a latitude × σ field")),
                vertical=dict(roles=["x", "y"], fixed=[], axis=["sigma"], unavailable=dict(
                    fixed="ψ is offered as a latitude × σ field", reduce=NO_VERTICAL_MEAN)),
                time=dict(roles=["fixed", "reduce"], reduce={"mean": MEAN["time"] + "; ψ is formed from the "
                                                                             "time mean of [pₛ v]"},
                          unavailable=dict(x="ψ histories are not offered in this release",
                                           y="ψ histories are not offered in this release"))))
    for key, product in manifest["diagnostics"].items():
        if product["kind"] == "series":
            none = dict(roles=["none"])
            quantities["series:" + key] = dict(
                presentation=product["presentation"], sources=[product["path"]], horizontal="none",
                vertical=None, time_axis=product["time_axis"], recorded=key,
                evaluation="recorded by the run at its own cadence",
                coordinates=dict(longitude=none, latitude=none, vertical=none,
                                 time=dict(roles=["x"], unavailable=dict(
                                     reduce="recorded series are shown as recorded"))))
    return dict(quantities=quantities,
                horizontal=dict(kind="spectral" if spectral else manifest["geometry"].get("grid"),
                                grid=manifest["geometry"].get("grid")))


# ----------------------------------------------------------------------
# Plan: one sample set per coordinate
# ----------------------------------------------------------------------

def _native(provider):
    """Native Gaussian state grid: latitudes north to south, longitudes, Gauss weights."""
    g = provider.geometry
    lat, lon, _ = state_grid(g["nlat"], g["nlon"])
    return lat, lon, gauss_latitude_nodes(g["nlat"])[1]


def _snap(provider, coordinate, value, sel):
    if not (sel["options"]["snap"] and provider.geometry.get("grid") == "latlon"):
        return value, False
    lat, lon, _ = _native(provider)
    if coordinate == "latitude":
        return float(lat[int(np.abs(lat - value).argmin())]), True
    return float(lon[int(np.abs((lon - value % 360 + 180) % 360 - 180).argmin())]), True


def plan(provider, sel):
    key = sel["quantity"]
    quantity = provider.manifest["explorer"]["quantities"][key]
    native = any(sel[c]["role"] == "reduce" for c in ("longitude", "latitude"))
    coordinates, axes = {}, {}
    lat_n, lon_n, weights = _native(provider) if native else (None, None, None)
    out = dict(presentation=quantity["presentation"], coordinates=coordinates, axes=axes,
               max_samples=MAX_STATES, companions={}, notes=[], reverse_latitude=False)
    for c in ("latitude", "longitude"):
        part = sel[c]
        if part["role"] == "fixed":
            value = part["value"] % 360.0 if c == "longitude" else part["value"]
            value, snapped = _snap(provider, c, value, sel)
            out[c] = dict(role="fixed", points=np.array([value]))
            coordinates[c] = dict(role="fixed", value=value, requested=part["value"],
                                  sampling="native Gaussian grid node" if snapped else
                                  "exact band-limited synthesis at the requested point")
        elif native:
            points = lat_n if c == "latitude" else lon_n
            out[c] = dict(role=part["role"], points=points,
                          weights=weights if (c == "latitude" and part["role"] == "reduce") else None)
            label = (f"{points.size} Gauss–Legendre nodes" if c == "latitude" else
                     f"{points.size} equally spaced native longitudes")
            if part["role"] == "reduce":
                coordinates[c] = dict(role="reduce", op=part["op"], rule=MEAN[c], nodes=label,
                                      domain="the whole sphere" if c == "latitude" else "the full circle")
            else:
                coordinates[c] = dict(role=part["role"], sampling="native Gaussian grid nodes: " + label)
            if c == "latitude":
                out["reverse_latitude"] = True        # evaluated north to south, shown south to north
                if part["role"] != "reduce":
                    axes[c] = latitude_axis(points[::-1], label=f"Latitude ({label})")
            elif part["role"] != "reduce":
                width = 360.0 / points.size
                axes[c] = longitude_axis(points, edges=list(points - width / 2) + [points[-1] + width / 2])
                axes[c]["limits"] = [-width / 2, 360 - width / 2]
        else:
            points = provider.lat if c == "latitude" else np.append(provider.lon, 360.0)
            # The closing meridian is shown at 360° and evaluated at 0°, as the sections always did.
            out[c] = dict(role=part["role"], points=points if c == "latitude" else points % 360.0)
            coordinates[c] = dict(role=part["role"], sampling="exact band-limited synthesis on the 2° display mesh")
            axes[c] = latitude_axis(points) if c == "latitude" else longitude_axis(points)
    if key == "mass_streamfunction":
        _psi_plan(provider, sel, out)
    else:
        _vertical_plan(provider, sel, quantity, out)
    t = sel["time"]
    if t["role"] == "fixed":
        coordinates["time"] = dict(role="fixed", axis="snapshots", index=t["index"],
                                   seconds=provider.manifest["time_axes"]["snapshots"]["values"][t["index"]])
    out["sampling"] = "; ".join(f"{c} {d.get('sampling') or d.get('rule')}" for c, d in coordinates.items()
                                if c in ("longitude", "latitude"))
    return out


def _reference(provider):
    physics = provider.manifest["physics"]
    return physics.get("reference_pressure") or physics.get("surface_pressure")


def _sigma_axis(provider, centres, edges, label):
    y = axis("vertical", "sigma", label, "", centres, edges=edges, scale="log", downward=True,
             rows=[dict(kind="level", label=f"level {k + 1}") for k in range(len(centres))])
    if _reference(provider):
        y["secondary"] = dict(label="nominal p ≈ σ × p_ref", factor=_reference(provider), units="Pa")
    return y


def _vertical_plan(provider, sel, quantity, out):
    v, coordinates = sel["vertical"], out["coordinates"]
    if v["role"] == "none":
        coordinates["vertical"] = dict(role="none")
        return
    rep = v["representation"]
    if v["role"] == "fixed":
        coordinates["vertical"] = dict(role="fixed", representation=rep, value=v["value"],
                                       label=REPRESENTATIONS[rep]["label"], interpolation=INTERPOLATION[rep])
        return
    context = provider._structure_context()
    coordinates["vertical"] = dict(role=v["role"], representation=rep, sampling="native full levels")
    heights = context.gravity is not None
    if rep == "sigma":
        out["axes"]["vertical"] = _sigma_axis(provider, list(context.full_levels), list(context.sigma_interfaces),
                                              "Nominal σ, native full levels")
    else:
        # Positions move with the column: filled from the evaluated sample in finish().
        out["axes"]["vertical"] = None
        coordinates["vertical"]["positions"] = ("this column's p_k = σ_k pₛ" if rep == "pressure" else
                                                "this column's Z_k" if rep == "geopotential_height" else
                                                "this column's Z_k − Zₛ")
    out["companions"]["pressure"] = dict(label="p", units="Pa")
    if heights:
        out["companions"]["geopotential_height"] = dict(label="Z", units="m")
        out["companions"]["surface_height"] = dict(label="Zₛ", units="m", surface=True)
    out["companions"]["surface_pressure"] = dict(label="pₛ", units="Pa", surface=True)
    single_column = sel["longitude"]["role"] in ("fixed", "reduce") and sel["latitude"]["role"] in ("fixed", "reduce")
    if single_column and sel["time"]["role"] in ("fixed", "reduce"):
        for name, (label, units) in dict(temperature=("T", "K"), u=("u", "m s⁻¹"), v=("v", "m s⁻¹")).items():
            out["companions"].setdefault(name, dict(label=label, units=units))
        if sel["quantity"] == "temperature" and "surface_temperature" in provider.manifest["fields"]:
            out["companions"]["surface_temperature"] = dict(label="Tₛ (slab)", units="K", surface=True)
    out["vertical"] = dict(heights=heights)


def _psi_plan(provider, sel, out):
    context = provider._structure_context()
    interfaces = list(context.sigma_interfaces)
    out["coordinates"]["vertical"] = dict(role=sel["vertical"]["role"], representation="sigma",
                                          sampling="layer interfaces")
    out["axes"]["vertical"] = _sigma_axis(provider, interfaces, [interfaces[0], *context.full_levels, interfaces[-1]],
                                          "Nominal σ at layer interfaces")
    out["companions"] = {}
    out["notes"].append("ψ is formed from the time mean of [pₛ v] over the selected states; the bottom "
                        "(σ = 1) row is the whole-column northward mass flux and is not forced to zero.")


# ----------------------------------------------------------------------
# Per-state evaluation
# ----------------------------------------------------------------------

def _wind(values):
    return np.hypot(values["u"], values["v"])


def _field(values, key):
    return _wind(values) if key in ("wind", "wind_speed") else values[key]


def _surface(provider, index, key, lat, lon):
    """A field without vertical coordinate at saved state ``index``: ``(nlat, nlon)``."""
    times = provider.manifest["time_axes"]["snapshots"]["values"]
    if key == "surface_temperature":
        from tropoi.representation.visual.viewer_fields import _slab_map
        slab_times = provider.manifest["time_axes"]["slab"]["values"]
        if times[index] not in slab_times:
            raise SelectionError("No slab snapshot at the saved state "
                                 f"{index} (t = {times[index]} s); the slab is saved at its own times")
        return _slab_map(provider._read("slab", slab_times.index(times[index])), provider.geometry, lat, lon)
    if provider.solver == "pe" and key == "terrain":
        lm = provider._terrain_lm
        return np.zeros((lat.size, lon.size)) if lm is None else synthesize_latlon(lm, lat, lon)
    frame = provider._read("state", index)
    if provider.solver == "pe":
        if key == "ln_ps":
            return np.exp(synthesize_latlon(frame[-1], lat, lon))
        raise SelectionError(f"{key} has native levels: give the vertical coordinate a role")
    if provider.solver not in ("bve", "swe"):
        raise SelectionError("Run has no evaluator for this selection")
    swe = provider.solver == "swe"
    zeta = frame[0] if swe else frame
    delta = frame[1] if swe else np.zeros_like(zeta)
    if key in ("u", "v", "wind", "wind_speed"):
        u, v = synthesize_wind_latlon(zeta, delta, provider.radius, lat, lon)
        return np.hypot(u, v) if key in ("wind", "wind_speed") else (u if key == "u" else v)
    if key in ("zeta", "delta"):
        return synthesize_latlon(zeta if key == "zeta" else delta, lat, lon)
    terrain = (np.zeros((lat.size, lon.size)) if provider._terrain_lm is None
               else synthesize_latlon(provider._terrain_lm, lat, lon))
    if not swe:
        raise SelectionError(f"Unavailable field {key!r}")
    gravity = provider.physics["gravity"]
    phi = synthesize_latlon(frame[2], lat, lon)
    thickness = provider.physics["mean_depth_m"] + phi / gravity
    height = terrain + thickness
    values = dict(phi=phi, terrain=terrain, layer_depth=thickness, free_surface_height=height,
                  free_surface_geopotential=gravity * height)
    return values[key]


def _fixed_surface(provider, frame, key, lat, lon, rep, value):
    """The quantity on one fixed vertical surface: ``(nlat, nlon)``, NaN where masked."""
    context = provider._context
    if rep == "native":
        if int(value) != value or not 0 <= value < context.nlev:
            raise SelectionError("Native level must be a zero-based full-level index")
        heights = context.gravity is not None
        levels = reconstruct_levels(frame, context, lat, lon, [int(value)],
                                    vorticity_divergence=key in ("zeta", "delta"), heights=heights)
        values = {name: arr[0] for name, arr in levels.values.items()}
        return _field(values, key)
    if rep != "sigma" and key in ("zeta", "delta"):
        raise SelectionError("Vorticity and divergence require a native or sigma surface")
    surface = pressure_surface(frame, context, lat, lon, (rep, value))
    values = dict(surface.values)
    if rep in ("geopotential_height", "height_above_surface") and key == "geopotential":
        values["geopotential"] = values["geopotential_height"] * context.gravity
    out = np.array(_field(values, key), dtype=float)
    out[surface.mask != 0] = np.nan
    return out


def samples(provider, sel, plan_, indices):
    """Yield ``(seconds, values (nv, nlat, nlon), companions)`` per saved state."""
    key = sel["quantity"]
    lat, lon = plan_["latitude"]["points"], plan_["longitude"]["points"]
    times = provider.manifest["time_axes"]["snapshots"]["values"]
    v = sel["vertical"]
    flip = (lambda a: a[..., ::-1, :]) if plan_["reverse_latitude"] else (lambda a: a)
    for index in indices:
        if key == "mass_streamfunction":
            yield times[index], *_psi_sample(provider, index, lat, lon, flip)
            continue
        if v["role"] == "none":
            yield times[index], flip(_surface(provider, index, key, lat, lon)[None]), {}
            continue
        frame = provider._read("state", index)
        if v["role"] == "fixed":
            yield times[index], flip(_fixed_surface(provider, frame, key, lat, lon,
                                                    v["representation"], v["value"])[None]), {}
            continue
        heights = plan_["vertical"]["heights"]
        levels = reconstruct_levels(frame, provider._context, lat, lon, None,
                                    vorticity_divergence=key in ("zeta", "delta"), heights=heights)
        companions = {}
        for name in plan_["companions"]:
            if name == "surface_pressure":
                companions[name] = flip(levels.surface_pressure[None])
            elif name == "surface_height":
                companions[name] = flip((np.zeros((lat.size, lon.size)) if levels.surface_height is None
                                         else levels.surface_height)[None])
            elif name == "surface_temperature":
                companions[name] = flip(_surface(provider, index, name, lat, lon)[None])
            else:
                companions[name] = flip(levels[name])
        yield times[index], flip(_field(levels.values, key)), companions


def _psi_sample(provider, index, lat, lon, flip):
    """[pₛ v] of one state, longitude mean formed on the grid: ``(nlev, nlat, 1)``."""
    levels = reconstruct_levels(provider._read("state", index), provider._context, lat, lon, None,
                                vorticity_divergence=False)
    ps_v = (levels.surface_pressure[None] * levels["v"]).mean(axis=2)
    return flip(ps_v[..., None]), {}


def post(provider, sel, plan_, cube):
    """ψ from the time-reduced [pₛ v]: ``(1, nlev + 1, nlat, 1)``."""
    if sel["quantity"] != "mass_streamfunction":
        return cube
    context = provider._context
    lat = plan_["latitude"]["points"][::-1]
    psi = mass_streamfunction(cube[0, :, :, 0], context.sigma_interfaces, lat, context.radius_m, context.gravity)
    r = psi[-1]
    plan_["notes"].append(f"Bottom (σ = 1) whole-column flux: max |ψ| {float(np.abs(r).max()):.4g} kg s⁻¹, "
                          f"against max |ψ| {float(np.abs(psi).max()):.4g} kg s⁻¹ in the section.")
    return psi[None, :, :, None]


def finish(provider, sel, plan_, payload):
    """Axes whose positions come from the sample, single-column tables and markers."""
    comp = payload.get("companions") or {}
    v = sel["vertical"]
    if v["role"] in ("x", "y") and plan_["axes"].get("vertical") is None:
        rep = v["representation"]
        positions = (comp["pressure"]["values"] if rep == "pressure" else comp["geopotential_height"]["values"])
        if rep == "height_above_surface":
            positions = [z - comp["surface_height"]["values"] for z in positions]
        payload[v["role"]] = axis("vertical", rep, REPRESENTATIONS[rep]["label"], REPRESENTATIONS[rep]["units"],
                                  positions, scale="log" if rep == "pressure" else "linear",
                                  downward=rep == "pressure")
    profile = v["role"] in ("x", "y") and all(sel[c]["role"] in ("fixed", "reduce")
                                               for c in ("longitude", "latitude", "time"))
    if profile and "temperature" in comp:
        reduced = [c for c in ("longitude", "latitude", "time") if sel[c]["role"] == "reduce"]
        mean = " (mean)" if reduced else ""
        sigma = list(provider._context.full_levels)
        z = comp.get("geopotential_height", {}).get("values")
        zs = comp.get("surface_height", {}).get("values")
        rows = []
        for k, s in enumerate(sigma):
            rows.append([k + 1, s, comp["pressure"]["values"][k], None if z is None else z[k],
                         None if z is None else z[k] - zs, comp["temperature"]["values"][k],
                         comp["u"]["values"][k], comp["v"]["values"][k]])
        payload["table"] = dict(columns=["Level", "σ", f"p (Pa){mean}", f"Z (m){mean}", f"Z − Zₛ (m){mean}",
                                         f"T (K){mean}", f"u (m s⁻¹){mean}", f"v (m s⁻¹){mean}"], rows=rows)
        payload["notes"] += [LOWEST_AIR + ".", *CONVENTIONS.values()]
        if reduced:
            payload["notes"].append("Means are taken on native σ levels: each row is the mean of the level's "
                                    "values over " + " and ".join(reduced) + ", and its p and Z are the means "
                                    "of that level's p_k and Z_k, not a mean on a surface of constant p or Z.")
        if sel["quantity"] == "temperature":
            y = payload[v["role"]]["values"]
            markers = [dict(label=f"Lowest full level air T{mean} = {payload['values'][-1]:.6g} K "
                                  f"(σ {sigma[-1]:.6g})", value=payload["values"][-1], position=y[-1], kind="level")]
            slab = comp.get("surface_temperature")
            if slab is not None:
                rep = v["representation"]
                surface = (comp["surface_pressure"]["values"] if rep == "pressure" else 1.0 if rep == "sigma"
                           else comp["surface_height"]["values"] if rep == "geopotential_height" else 0.0)
                markers.append(dict(label=f"Slab surface Tₛ{mean} = {slab['values']:.6g} K", value=slab["values"],
                                    position=surface, kind="surface"))
                payload["notes"].append(SLAB + "; bilinear between native slab nodes (periodic in longitude), "
                                               "exactly the node value at a native node.")
            payload["markers"] = markers
    if sel["time"]["role"] in ("x", "y", "reduce"):
        payload["notes"].append("Saved states only: variations between saved states are not represented.")
