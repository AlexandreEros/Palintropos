"""One selection model for every archive the viewer opens.

A scalar quantity is a function of up to four coordinates: longitude,
latitude, a vertical coordinate and time. A selection gives each coordinate a
**role**:

* ``x`` or ``y``: a plotting axis;
* ``fixed``: held at one value (``value``; for time, a saved ``index``);
* ``reduce``: reduced by a statistic over its whole sampled domain or, for
  time, over the saved samples ``first..last`` (``op``; only ``mean``);
* ``none``: the quantity has no such coordinate (a surface field has no
  vertical coordinate; a recorded global series has no horizontal ones).

The browser's *animate* role is ``fixed`` here: playback is a sequence of
fixed selections at the actual saved times.

**Capability is per coordinate, never per plot type.** Each quantity of an
archive's manifest (``explorer.quantities[...]["coordinates"]``) lists, for
every coordinate, the roles it accepts, the reductions it defines and, for
the vertical coordinate, the representations it can be fixed at or plotted
along. :func:`parse_selection` checks each coordinate against that list and
nothing else; the result's dimensions follow from the roles: two axes give a
two-dimensional field, one a line, none a single value. A longitude ×
latitude field is one such assignment among all others.

Where a combination is refused for a scientific reason the refusal is local
to the coordinate that causes it and carries the reason (for example a
vertical representation whose level positions differ from sample to sample,
see :data:`PER_SAMPLE`).

:data:`PRESETS` are named role assignments offered for convenience. They
define nothing: a preset is valid exactly when its roles are, and every
assignment a preset can produce can also be made by hand.

:func:`evaluate` is the one evaluation path. The archive's provider maps each
role to a set of sample points (and weights, for a reduction) and evaluates
the quantity on their tensor product, one saved sample at a time; this module
reduces and arranges the result. The numerical definition of a value
therefore depends on the sample set of each coordinate, never on a plot type.

**Reduction semantics** (``mean``; the payload states each in
``coordinates[name]``):

* time: the equal-weight mean of every saved sample from ``first`` to
  ``last`` inclusive, ``1/N`` each, not weighted by the intervals between
  samples (the cadence and its regularity are reported);
* longitude: the plain mean over the equally spaced longitudes of the
  archive's native grid (exact quadrature of a band-limited field);
* latitude: the area-weighted mean over the native latitudes: Gauss–Legendre
  weights of a Gaussian grid (``∝ cos φ dφ``), or the recorded cell area
  weights of a columnar archive;
* longitude and latitude together: the area mean (the product of the two);
* vertical: not defined in this release (a vertical mean needs a mass or
  thickness weighting rule that has not been specified).

A sample that is masked (a surface below the terrain, outside the sampled
levels) makes every mean that includes it masked: no mean is formed over a
subset of the requested samples.
"""
from __future__ import annotations

import json
import math

import numpy as np

__all__ = ["COORDINATES", "PRESETS", "ROLES", "SelectionError", "parse_selection", "evaluate",
           "describe_roles", "axis", "time_samples"]

COORDINATES = ("longitude", "latitude", "vertical", "time")
ROLES = ("x", "y", "fixed", "reduce", "none")
AXES = ("x", "y")

#: Named role assignments: convenience only. A quantity lacking a vertical
#: coordinate takes "none" where a preset says "fixed"; nothing else adapts.
PRESETS = {
    "map": dict(label="Longitude × latitude",
                roles=dict(longitude="x", latitude="y", vertical="fixed", time="fixed")),
    "time_mean_map": dict(label="Longitude × latitude, time mean",
                          roles=dict(longitude="x", latitude="y", vertical="fixed", time="reduce")),
    "meridional_section": dict(label="Latitude × vertical at a fixed longitude",
                               roles=dict(longitude="fixed", latitude="x", vertical="y", time="fixed")),
    "zonal_section": dict(label="Longitude × vertical at a fixed latitude",
                          roles=dict(longitude="x", latitude="fixed", vertical="y", time="fixed")),
    "zonal_mean": dict(label="Latitude × vertical, longitude mean",
                       roles=dict(longitude="reduce", latitude="x", vertical="y", time="fixed")),
    "profile": dict(label="Vertical profile at a point",
                    roles=dict(longitude="fixed", latitude="fixed", vertical="y", time="fixed")),
    "time_mean_profile": dict(label="Vertical profile at a point, time mean",
                              roles=dict(longitude="fixed", latitude="fixed", vertical="y", time="reduce")),
    "time_series": dict(label="Time series at a point",
                        roles=dict(longitude="fixed", latitude="fixed", vertical="fixed", time="x")),
    "time_section": dict(label="Time × vertical at a point",
                         roles=dict(longitude="fixed", latitude="fixed", vertical="y", time="x")),
    "hovmoller": dict(label="Longitude × time at a fixed latitude",
                      roles=dict(longitude="x", latitude="fixed", vertical="fixed", time="y")),
    "recorded_series": dict(label="Recorded series",
                            roles=dict(longitude="none", latitude="none", vertical="none", time="x")),
}

#: Cap on saved states one request synthesizes or reads in full, so a request stays bounded.
MAX_STATES = 400
#: Cap on saved samples of a single archived column (read as one record).
MAX_SAMPLES = 20000

#: The vertical representations whose native-level positions are the same for
#: every sample are listed per quantity under ``axis``; those under
#: ``per_sample`` (pressure p_k = σ_k pₛ, geopotential heights Z_k) move with
#: the column and the state, so they can be an axis only of a result made of
#: one sample of every other coordinate. Otherwise there is no common axis to
#: draw the levels on, and averaging levels whose positions differ is not an
#: average on surfaces of constant p or Z.
PER_SAMPLE = ("Its native-level positions differ from column to column and from state to state, "
              "so it can be the plotting axis only of a single column at a single saved time. "
              "Use σ, whose levels are common to every sample, or fix the vertical coordinate on a "
              "surface of constant p or Z")


class SelectionError(ValueError):
    """The selection is malformed or not supported for this quantity."""


def _number(raw, name):
    if isinstance(raw, bool):
        raise SelectionError(f"{name} must be a number")
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


ROLE_TEXT = dict(x="the horizontal axis", y="the vertical axis", fixed="fixed", reduce="reduced",
                 none="absent")


def _refuse_role(quantity, coordinate, role):
    caps = quantity["coordinates"][coordinate]
    reason = (caps.get("unavailable") or {}).get(role)
    name = quantity["presentation"]["name"]
    if role == "none":
        raise SelectionError(f"{name} has a {coordinate} coordinate: give it a role")
    if caps["roles"] == ["none"]:
        raise SelectionError(f"{name} has no {coordinate} coordinate")
    raise SelectionError(f"The {coordinate} of {name} cannot be {ROLE_TEXT[role]}" +
                         (f": {reason}" if reason else ""))


def parse_selection(manifest: dict, text: str) -> dict:
    """Validate a JSON selection against the quantity's coordinate capabilities.

    Returns ``dict(quantity, longitude, latitude, vertical, time, axes, options)``
    with numbers parsed and indices checked; ``axes`` maps ``x``/``y`` to the
    coordinate drawn along it (or None)."""
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
    caps = quantity["coordinates"]
    coords = {}
    for c in COORDINATES:
        part = raw.get(c, {"role": "none"})
        if not isinstance(part, dict) or part.get("role") not in (*ROLES, "animate"):
            raise SelectionError(f"Give the {c} coordinate one of the roles {', '.join(ROLES)}")
        part = dict(part)
        if part["role"] == "animate":
            part["role"] = "fixed"
        if part["role"] not in caps[c]["roles"]:
            _refuse_role(quantity, c, part["role"])
        coords[c] = part
    roles = {c: coords[c]["role"] for c in COORDINATES}
    axes = {}
    for a in AXES:
        along = [c for c in COORDINATES if roles[c] == a]
        if len(along) > 1:
            raise SelectionError(f"Only one coordinate can be the {ROLE_TEXT[a]}: " + " and ".join(along))
        axes[a] = along[0] if along else None
    options = raw.get("options") or {}
    if not isinstance(options, dict) or set(options) - {"snap", "format"}:
        raise SelectionError("Options must be an object with snap and format only")
    out = dict(quantity=name, axes=axes, options=dict(snap=bool(options.get("snap", True)),
                                                      format=options.get("format", "json")))
    if out["options"]["format"] not in ("json", "frame"):
        raise SelectionError("The format must be json or frame")
    for c in ("longitude", "latitude"):
        part = coords[c]
        entry = dict(role=part["role"])
        if part["role"] == "fixed":
            entry["value"] = _number(part.get("value"), c.capitalize())
            if c == "latitude" and not -90 <= entry["value"] <= 90:
                raise SelectionError("Latitude must lie in [-90, 90]")
        if part["role"] == "reduce":
            entry["op"] = _reduction(quantity, c, part)
        out[c] = entry
    vertical = coords["vertical"]
    entry = dict(role=vertical["role"])
    if vertical["role"] != "none":
        rep = vertical.get("representation", "native")
        entry["representation"] = rep
        v = caps["vertical"]
        if vertical["role"] == "fixed":
            if rep not in v["fixed"]:
                raise SelectionError(f"The vertical coordinate cannot be fixed in the representation "
                                     f"{rep!r} here; choose {', '.join(v['fixed'])}")
            entry["value"] = _number(vertical.get("value"), "Vertical value")
        elif vertical["role"] in AXES:
            single = all(roles[c] in ("fixed", "none") for c in COORDINATES if c != "vertical")
            if rep in v.get("per_sample", ()) and not single:
                raise SelectionError(f"The representation {rep!r} cannot be the vertical axis here. " + PER_SAMPLE)
            allowed = [*v["axis"], *(v.get("per_sample", ()) if single else ())]
            if rep not in allowed:
                raise SelectionError(f"The vertical axis cannot use the representation {rep!r} here; "
                                     f"choose {', '.join(allowed)}")
        else:
            entry["op"] = _reduction(quantity, "vertical", vertical)
    out["vertical"] = entry
    axis_id = quantity.get("time_axis")
    if entry["role"] == "fixed":
        axis_id = level_time_axis(quantity, entry) or axis_id
    time = coords["time"]
    entry = dict(role=time["role"], axis=axis_id)
    if quantity.get("recorded"):          # the whole record, at its own cadence
        out["time"] = entry
        return out
    count = len((manifest["time_axes"].get(axis_id) or {}).get("values", []))
    if time["role"] == "fixed":
        entry["index"] = _index(time.get("index"), "Saved time", count)
    elif time["role"] in ("reduce", *AXES):
        entry["first"] = _index(time.get("first", 0), "First saved time", count)
        entry["last"] = _index(time.get("last", count - 1), "Last saved time", count)
        if entry["last"] < entry["first"]:
            raise SelectionError("The last saved time precedes the first")
        if time["role"] == "reduce":
            entry["op"] = _reduction(quantity, "time", time)
    out["time"] = entry
    return out


def _reduction(quantity, coordinate, part):
    op = part.get("op", "mean")
    if op not in quantity["coordinates"][coordinate].get("reduce", {}):
        raise SelectionError(f"The reduction {op!r} over {coordinate} is not available")
    return op


def level_time_axis(quantity, vertical):
    """The time axis of a fixed native level, where levels have their own."""
    levels = (quantity.get("vertical") or {}).get("levels") or []
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


def describe_roles(sel, vertical_label=None):
    """A plain description generated from the roles, e.g. ``Latitude × vertical ·
    longitude mean · time fixed``. Labels only; it selects nothing."""
    axes = [sel["axes"][a] for a in AXES if sel["axes"][a]]
    head = " × ".join(c.capitalize() if i == 0 else c for i, c in enumerate(axes)) or "Single value"
    rest = []
    for c in COORDINATES:
        part = sel.get(c) or {}
        if part.get("role") == "reduce":
            rest.append(f"{c} {part['op']}")
        elif part.get("role") == "fixed":
            rest.append(f"{c} fixed")
    return " · ".join([head, *rest])


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def time_samples(sel):
    """Saved-sample indices the selection reads, in order."""
    time = sel["time"]
    if time["role"] == "fixed":
        return [time["index"]]
    if time["role"] == "none":
        return [None]
    return list(range(time["first"], time["last"] + 1))


def _reduce(array, weights, axis):
    """Mean along ``axis``: the plain mean for equal weights (the arithmetic of
    the zonal-mean accumulator), else the normalized weighted sum."""
    if weights is None:
        return array.mean(axis=axis, keepdims=True)
    w = np.asarray(weights, dtype=float)
    shape = [1] * array.ndim
    shape[axis] = w.size
    return (array * w.reshape(shape)).sum(axis=axis, keepdims=True) / w.sum()


def _horizontal(array, plan):
    """Apply the longitude then the latitude reduction to ``(..., nlat, nlon)``."""
    if plan["longitude"]["role"] == "reduce":
        array = _reduce(array, plan["longitude"].get("weights"), -1)
    if plan["latitude"]["role"] == "reduce":
        array = _reduce(array, plan["latitude"].get("weights"), -2)
    return array


def evaluate(provider, sel):
    """Evaluate a parsed selection on ``provider``; the JSON payload.

    The provider supplies ``plan(sel)`` (sample sets, axes and descriptions
    per coordinate) and ``samples(sel, plan, indices)``, which yields, for
    each saved sample, an array ``(nvertical, nlat, nlon)`` of the quantity
    (NaN where masked) and a dict of companion arrays of the same layout. This
    function applies the reductions and arranges the axes."""
    plan = provider.plan(sel)
    indices = time_samples(sel)
    limit = plan.get("max_samples", MAX_STATES)
    if len(indices) > limit:
        raise SelectionError(f"Select at most {limit} saved samples for this selection "
                             f"({len(indices)} requested)")
    time_role = sel["time"]["role"]
    total, companions_total, stacked, stacked_companions = None, None, [], []
    seconds = []
    for t, values, companions in provider.samples(sel, plan, indices):
        values = _horizontal(np.asarray(values, dtype=float), plan)
        companions = {k: _horizontal(np.asarray(a, dtype=float), plan) for k, a in companions.items()}
        seconds.append(t)
        if time_role == "reduce":
            if total is None:
                total = np.zeros_like(values)
                companions_total = {k: np.zeros_like(a) for k, a in companions.items()}
            total += values
            for k, a in companions.items():
                companions_total[k] += a
        else:
            stacked.append(values)
            stacked_companions.append(companions)
    if time_role == "reduce":
        n = len(indices)
        cube = (total / n)[None]
        companion_cubes = {k: (a / n)[None] for k, a in companions_total.items()}
    else:
        cube = np.stack(stacked)
        companion_cubes = {k: np.stack([c[k] for c in stacked_companions]) for k in stacked_companions[0]}
    # cube: (time, vertical, latitude, longitude); singleton where not an axis.
    if hasattr(provider, "post"):            # quantities defined from a reduced state (ψ from [pₛ v])
        cube = provider.post(sel, plan, cube)
    order = dict(time=0, vertical=1, latitude=2, longitude=3)
    x, y = sel["axes"]["x"], sel["axes"]["y"]
    payload = dict(quantity=sel["quantity"], presentation=plan["presentation"],
                   roles={c: sel[c]["role"] for c in COORDINATES}, axes=dict(x=x, y=y),
                   description=describe_roles(sel), coordinates=plan["coordinates"],
                   sampling=plan.get("sampling", ""), notes=list(plan.get("notes", [])))
    if time_role in ("reduce", *AXES):
        payload["coordinates"]["time"] = _time_description(provider.manifest, sel, seconds)
    for a, c in (("x", x), ("y", y)):
        if c is not None:
            payload[a] = plan["axes"][c] if c != "time" else time_axis(seconds, label=plan.get("time_label",
                                                                                                "Saved time"))
    payload["values"] = _arrange(cube, order, x, y)
    payload["companions"] = {
        k: dict(plan["companions"][k], values=_arrange(a, order, x, y, not plan["companions"][k].get("surface")))
        for k, a in companion_cubes.items()} or None
    if hasattr(provider, "finish"):
        provider.finish(sel, plan, payload)
    return payload


def _arrange(cube, order, x, y, vertical=True):
    """The plotted array: ``[y][x]`` for two axes, a list for one, a number
    for none. ``vertical=False`` arranges a companion that has no vertical
    dimension (a surface value under a vertical axis) along its other axes."""
    keep = [c for c in (y, x) if c is not None and (vertical or c != "vertical")]
    drop = tuple(order[c] for c in order if c not in keep)
    if any(cube.shape[d] != 1 for d in drop):
        raise ValueError("A reduced or fixed coordinate kept more than one sample")
    array = np.squeeze(cube, axis=drop) if drop else cube
    if len(keep) == 2 and order[keep[0]] > order[keep[1]]:
        array = array.T
    return _plain(array)


def _plain(array):
    """JSON-safe values: NaN (masked) becomes null."""
    if np.ndim(array) == 0:
        v = float(array)
        return v if math.isfinite(v) else None
    return [_plain(a) for a in array] if np.ndim(array) > 1 else \
        [float(v) if math.isfinite(v) else None for v in np.asarray(array, dtype=float)]


def _time_description(manifest, sel, seconds):
    time = sel["time"]
    values = np.asarray(seconds, dtype=float)
    steps = np.diff(values)
    intervals = {str(float(s)): int(n) for s, n in zip(*np.unique(steps, return_counts=True))}
    out = dict(role=time["role"], axis=time["axis"], first=time["first"], last=time["last"],
               count=len(seconds), intervals_seconds=intervals,
               cadence="single sample" if len(seconds) == 1 else
                       "regular" if len(intervals) == 1 else "irregular")
    if time["role"] == "reduce":
        out.update(op=time["op"], rule=("equal-weight mean of every saved sample from the first to the last "
                                        "selection inclusive (1/N each), not weighted by the intervals "
                                        "between samples"))
    return out


# ----------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------

def axis(coordinate, representation, label, units, values, *, edges=None, scale="linear",
         downward=False, ticks=None, rows=None, limits=None, interpolation=None):
    """A plotting axis: centres, optional cell edges and how to draw it by default.

    ``values`` and ``edges`` are physical coordinates. ``scale`` (linear or
    log) and ``downward`` (increasing values drawn lower) are the default
    display only: the viewer may draw the same axis linearly or
    logarithmically, upward or downward, without changing a value.
    ``interpolation`` names the space in which values between native levels
    were obtained, where they were. ``rows`` describes discrete rows (a
    level's label and kind); ``limits`` the plotted range."""
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
    if interpolation is not None:
        out["interpolation"] = interpolation
    return out


def latitude_axis(values, edges=None, label="Latitude"):
    return axis("latitude", "geocentric", label, "°N", values, edges=edges, ticks="latitude",
                limits=[-90, 90])


def longitude_axis(values, edges=None, label="Longitude (east)"):
    return axis("longitude", "geocentric", label, "°E", values, edges=edges, ticks="longitude",
                limits=[0, 360])


def time_axis(values, label="Saved time"):
    return axis("time", "elapsed", label, "s", values, ticks="time")
