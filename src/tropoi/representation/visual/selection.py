"""One selection model for every archive the viewer opens.

A scalar quantity is a function of four coordinates: longitude, latitude, a
vertical coordinate and time. A selection gives each coordinate a **role**:

* ``x`` or ``y``: a plotting axis;
* ``fixed``: held at one value (``value``; for time, a saved ``index``);
* ``reduce``: reduced by a statistic (``op``; only ``mean`` today);
* ``none``: the quantity has no such coordinate (a surface field has no
  vertical coordinate; a recorded global series has no horizontal ones).

The browser's *animate* role is ``fixed`` here: playback is a sequence of
fixed selections at the actual saved times.

The remaining plotting coordinates decide what is drawn: two give a map or
section, one a line, none a value. Only the role combinations listed in
:data:`PRODUCTS` are evaluated, and each archive's manifest
(``explorer.quantities``) says which of them each quantity supports, with
which vertical representations. Anything else is refused with the reason.

Evaluation stays where it was: each product calls the provider's existing
evaluator, so the numerical definition of a map, a column, a section or a
zonal mean does not depend on how it was requested.
"""
from __future__ import annotations

import json
import math

__all__ = ["COORDINATES", "PRODUCTS", "ROLES", "SelectionError", "parse_selection",
           "product_for", "axis", "level_edges"]

COORDINATES = ("longitude", "latitude", "vertical", "time")
ROLES = ("x", "y", "fixed", "reduce", "none")

#: Supported role combinations. ``time`` "fixed" also accepts "reduce" where
#: noted; a coordinate the quantity lacks has the role "none".
PRODUCTS = {
    "map": dict(label="Map · longitude × latitude",
                roles=dict(longitude="x", latitude="y", vertical=("fixed", "none"), time="fixed")),
    "meridional_section": dict(label="Section · latitude × vertical at a fixed longitude",
                               roles=dict(latitude="x", vertical="y", longitude="fixed", time="fixed")),
    "zonal_section": dict(label="Section · longitude × vertical at a fixed latitude",
                          roles=dict(longitude="x", vertical="y", latitude="fixed", time="fixed")),
    "zonal_mean": dict(label="Zonal mean · latitude × vertical",
                       roles=dict(latitude="x", vertical="y", longitude="reduce",
                                  time=("fixed", "reduce"))),
    "column": dict(label="Column profile at a fixed location",
                   roles=dict(vertical="y", longitude="fixed", latitude="fixed", time="fixed")),
    "time_series": dict(label="Time series at a fixed location",
                        roles=dict(time="x", longitude="fixed", latitude="fixed",
                                   vertical=("fixed", "none"))),
    "time_section": dict(label="Time × vertical at a fixed location",
                         roles=dict(time="x", vertical="y", longitude="fixed", latitude="fixed")),
    "recorded_series": dict(label="Recorded series",
                            roles=dict(time="x", longitude="none", latitude="none", vertical="none")),
}

#: Cap on saved states one request may read, so a request stays bounded.
MAX_STATES = 400
#: Cap on samples of a time series or time section read from columnar archives.
MAX_SAMPLES = 20000


class SelectionError(ValueError):
    """The selection is malformed or not supported for this quantity."""


def product_for(roles: dict) -> str | None:
    """The product whose role pattern matches ``roles``, or None."""
    for name, spec in PRODUCTS.items():
        if all(roles.get(c, "none") in ((want,) if isinstance(want, str) else want)
               for c, want in spec["roles"].items()):
            return name
    return None


def _number(raw, name):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SelectionError(f"{name} must be a number") from None
    if not math.isfinite(value):
        raise SelectionError(f"{name} must be finite")
    return value


def _index(raw, name, count):
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SelectionError(f"{name} must be a saved-sample index")
    if not 0 <= raw < count:
        raise SelectionError(f"{name} is outside the {count} saved samples")
    return raw


def parse_selection(manifest: dict, text: str) -> dict:
    """Validate a JSON selection against the archive's explorer description.

    Returns ``dict(product, quantity, longitude, latitude, vertical, time,
    options)`` with numbers parsed and indices checked."""
    if len(text) > 4096:
        raise SelectionError("Selection too long")
    try:
        raw = json.loads(text)
    except ValueError:
        raise SelectionError("Selection is not valid JSON") from None
    if not isinstance(raw, dict) or set(raw) - {"quantity", *COORDINATES, "options"}:
        raise SelectionError("Unknown selection member")
    explorer = manifest.get("explorer") or {}
    name = raw.get("quantity")
    quantity = explorer.get("quantities", {}).get(name) if isinstance(name, str) else None
    if quantity is None:
        raise SelectionError("Select an available scalar quantity")
    coords = {}
    for c in COORDINATES:
        part = raw.get(c, {"role": "none"})
        if not isinstance(part, dict) or part.get("role") not in (*ROLES, "animate"):
            raise SelectionError(f"Give the {c} coordinate one of the roles {', '.join(ROLES)}")
        part = dict(part)
        if part["role"] == "animate":
            part["role"] = "fixed"
        coords[c] = part
    roles = {c: coords[c]["role"] for c in COORDINATES}
    product = product_for(roles)
    if product is None:
        raise SelectionError("This combination of coordinate roles has no plot: " +
                             ", ".join(f"{c} {r}" for c, r in roles.items()))
    support = quantity["products"].get(product)
    if support is None:
        reason = quantity.get("unavailable", {}).get(product) or explorer.get("unavailable", {}).get(product)
        raise SelectionError(f"{PRODUCTS[product]['label']} is not available for "
                             f"{quantity['presentation']['name']}" + (f": {reason}" if reason else ""))
    has_vertical = quantity.get("vertical") is not None
    if has_vertical == (roles["vertical"] == "none"):
        raise SelectionError("The vertical coordinate role does not match the quantity" if has_vertical
                             else "This quantity has no vertical coordinate")
    out = dict(product=product, quantity=name, options=raw.get("options") or {})
    if not isinstance(out["options"], dict):
        raise SelectionError("Options must be an object")
    for c in ("longitude", "latitude"):
        part = coords[c]
        entry = dict(role=part["role"])
        if part["role"] == "fixed":
            entry["value"] = _number(part.get("value"), c.capitalize())
            if c == "latitude" and not -90 <= entry["value"] <= 90:
                raise SelectionError("Latitude must lie in [-90, 90]")
        if part["role"] == "reduce":
            if part.get("op", "mean") not in support.get("reductions", {}).get(c, ()):
                raise SelectionError(f"Reduction over {c} is not available")
            entry["op"] = part.get("op", "mean")
        out[c] = entry
    vertical = coords["vertical"]
    entry = dict(role=vertical["role"])
    if has_vertical:
        rep = vertical.get("representation", "native")
        allowed = support.get("vertical", ())
        if rep not in allowed:
            raise SelectionError(f"The vertical representation {rep!r} is not available here; "
                                 f"choose {', '.join(allowed)}")
        entry["representation"] = rep
        if vertical["role"] == "fixed":
            entry["value"] = _number(vertical.get("value"), "Vertical value")
    out["vertical"] = entry
    axis_id = quantity.get("time_axis")
    if has_vertical and vertical["role"] == "fixed":
        axis_id = level_time_axis(quantity, entry) or axis_id
    count = len((manifest["time_axes"].get(axis_id) or {}).get("values", []))
    time = coords["time"]
    entry = dict(role=time["role"], axis=axis_id)
    if product == "recorded_series":         # the whole record, at its own cadence
        out["time"] = entry
        return out
    if time["role"] == "fixed":
        entry["index"] = _index(time.get("index"), "Saved time", count)
    elif time["role"] in ("reduce", "x"):
        entry["first"] = _index(time.get("first", 0), "First saved time", count)
        entry["last"] = _index(time.get("last", count - 1), "Last saved time", count)
        if entry["last"] < entry["first"]:
            raise SelectionError("The last saved time precedes the first")
        if time["role"] == "reduce":
            entry["op"] = time.get("op", "mean")
            if entry["op"] not in support.get("reductions", {}).get("time", ()):
                raise SelectionError("Reduction over time is not available")
    out["time"] = entry
    return out


def level_time_axis(quantity, vertical):
    """The time axis of a fixed native level, where levels have their own."""
    levels = quantity["vertical"].get("levels") or []
    rep, value = vertical["representation"], vertical.get("value")
    for level in levels:
        if level.get("time_axis") and _matches(level, rep, value):
            return level["time_axis"]
    return None


def _matches(level, rep, value):
    if rep == "native":
        return level["index"] == value
    if rep in level:
        return math.isclose(level[rep], value, rel_tol=1e-9, abs_tol=1e-12)
    return False


def level_edges(centres, scale="log"):
    """Band edges between level centres: midpoints in ``log10`` (σ, p) or in
    the value itself. A top centre of 0 (σ at the top interface) is replaced
    by the geometric continuation ``c1^2 / c2``; the last edge never passes
    σ = 1. The browser draws native levels as these bands, so it never
    invents values between them."""
    c = [float(v) for v in centres]
    if scale == "log":
        if c[0] <= 0:
            c[0] = c[1] * c[1] / c[2]
        logs = [math.log10(v) for v in c]
        edges = [logs[0] - (logs[1] - logs[0]) / 2]
        edges += [(a + b) / 2 for a, b in zip(logs, logs[1:])]
        edges.append(min(0.0, logs[-1] + (logs[-1] - logs[-2]) / 2))
        return [10 ** e for e in edges]
    edges = [c[0] - (c[1] - c[0]) / 2] + [(a + b) / 2 for a, b in zip(c, c[1:])]
    return edges + [c[-1] + (c[-1] - c[-2]) / 2]


def axis(coordinate, representation, label, units, values, *, edges=None, scale="linear",
         downward=False, ticks=None, rows=None, limits=None):
    """A plotting axis: centres, optional cell edges and how to draw it.

    ``downward`` draws increasing values lower (pressure, σ, depth).
    ``rows`` describes discrete rows (a level's label and kind, for example
    the diagnosed skin of a substrate column); ``limits`` the plotted range."""
    out = dict(coordinate=coordinate, representation=representation, label=label, units=units,
               values=[float(v) for v in values], scale=scale, downward=bool(downward))
    if edges is not None:
        out["edges"] = [float(v) for v in edges]
    if ticks is not None:
        out["ticks"] = ticks
    if rows is not None:
        out["rows"] = rows
    if limits is not None:
        out["limits"] = [float(v) for v in limits]
    return out


def latitude_axis(values, edges=None, label="Latitude"):
    return axis("latitude", "geocentric", label, "°N", values, edges=edges, ticks="latitude",
                limits=[-90, 90])


def longitude_axis(values, edges=None, label="Longitude (east)"):
    return axis("longitude", "geocentric", label, "°E", values, edges=edges, ticks="longitude",
                limits=[0, 360])


def time_axis(values, label="Saved time"):
    return axis("time", "elapsed", label, "s", values, ticks="time")
