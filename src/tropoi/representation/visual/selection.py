"""One selection model for every archive the viewer opens.

A scalar quantity is a function of up to four coordinates: longitude,
latitude, a vertical coordinate and time. A selection gives each coordinate a
**role**:

* ``x`` or ``y``: a plotting axis;
* ``fixed``: held at one value (``value``; for time, a saved ``index``);
* ``reduce``: reduced by a statistic (``op``: ``mean``, ``min`` or ``max``)
  over its whole sampled domain or, for time, over the saved samples
  ``first..last``;
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

**Reduction semantics** (the payload states each in ``coordinates[name]``).
Every reduction acts on the *native samples* the provider evaluates for it
(saved states; native grid nodes; native levels), never on display-mesh or
interpolated values. Operators:

* ``mean``
  - time: the time-weighted mean of the saved samples ``first..last``
    inclusive with *cell weights*: each sample stands for the interval around
    it, ``wᵢ = (tᵢ₊₁ − tᵢ₋₁)/2`` with the ends mirrored (``w₀ = t₁ − t₀``,
    ``w_N = t_N − t_{N−1}``). For a regular cadence every weight is equal and
    the mean is the plain ``1/N`` mean (computed as such, bit for bit);
  - longitude: the plain mean over the equally spaced longitudes of the
    archive's native grid (exact quadrature of a band-limited field);
  - latitude: the area-weighted mean over the native latitudes:
    Gauss–Legendre weights of a Gaussian grid (``∝ cos φ dφ``), or the
    recorded cell area weights of a columnar archive;
  - longitude and latitude together: the area mean (the product of the two);
  - vertical: not defined in this release (a vertical mean needs a mass or
    thickness weighting rule that has not been approved).
* ``min`` / ``max``: the extremum over the same native samples, unweighted.
  It is the extremum of the *sampled* values, not of the continuous field
  between samples.

Masks. A masked sample (a surface below the terrain, outside the sampled
levels) makes every *mean* that includes it masked: no mean is formed over a
subset of the requested samples. ``min``/``max`` take the extremum over the
valid samples only and are masked only where every sample is masked; the
payload counts the masked samples they excluded.

Order. Reductions over different coordinates need not commute when their
operators differ (the time maximum of a zonal mean is not the zonal mean of a
time maximum). They are applied in one fixed order: longitude, latitude,
vertical (within each saved sample), then time. The payload states it.

Companions (level positions p_k, Z_k and the like that accompany a vertical
axis) are always reduced with the mean over the same samples: they locate
the rows, whatever statistic the quantity takes.
"""
from __future__ import annotations

import json
import math

import numpy as np

__all__ = ["COORDINATES", "PRESETS", "ROLES", "OPERATORS", "SelectionError", "parse_selection", "evaluate",
           "describe_roles", "axis", "time_samples", "time_weights", "reductions"]

COORDINATES = ("longitude", "latitude", "vertical", "time")
ROLES = ("x", "y", "fixed", "reduce", "none")
AXES = ("x", "y")
#: Reduction operators. Their per-coordinate rules come from the provider.
OPERATORS = ("mean", "min", "max")
#: Order in which reductions are applied (see the module documentation).
REDUCTION_ORDER = ("longitude", "latitude", "vertical", "time")
EXTREMUM = ("{op} over the same native samples, unweighted; masked samples are excluded and the "
            "result is masked only where every sample is masked")
TIME_MEAN = ("time-weighted mean over the saved samples first..last inclusive, each weighted by the "
             "interval it stands for, (t_{i+1} - t_{i-1})/2 with the ends mirrored; equal weights "
             "(the plain 1/N mean) for a regular cadence")


def reductions(mean_rule=None, *, extrema=True):
    """A coordinate's ``reduce`` capability: ``{op: rule}`` for the operators it defines."""
    out = {} if mean_rule is None else {"mean": mean_rule}
    if extrema:
        out.update(min=EXTREMUM.format(op="minimum"), max=EXTREMUM.format(op="maximum"))
    return out

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


#: The wind layer of a longitude × latitude result: streamlines of the
#: horizontal wind on the result's own surface, at its own saved time or
#: averaged over its own saved states. A layer is evaluated with the quantity
#: (one pass over the states, as companions), so it shares every role.
WIND_LAYER = ("Instantaneous streamlines of the horizontal wind (u, v) on the same surface as the field: "
              "curves everywhere tangent to the wind of one saved state, or of the time-mean wind over "
              "the same saved states as the field. They are not trajectories, and their spacing does "
              "not encode speed")


class SelectionError(ValueError):
    """The selection is malformed or not supported for this quantity."""


def _wind_layer(quantity, out):
    """Refuse a wind layer the result cannot carry, with the reason."""
    layer = (quantity.get("layers") or {}).get("wind")
    if not layer or not layer.get("available"):
        reason = (layer or {}).get("reason") or "this archive has no horizontal wind"
        raise SelectionError("No wind layer for " + quantity["presentation"]["name"] + ": " + reason)
    if {out["axes"]["x"], out["axes"]["y"]} != {"longitude", "latitude"}:
        raise SelectionError("A wind layer is drawn on a longitude × latitude result")
    if out["vertical"]["role"] not in ("fixed", "none"):
        raise SelectionError("A wind layer needs the field's own vertical surface: fix the vertical "
                             "coordinate")
    time = out["time"]
    if time["role"] == "reduce" and time.get("op") != "mean":
        raise SelectionError("Streamlines are drawn of one saved wind or of the time-mean wind; a time "
                             f"{time['op']}imum has no wind")


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
    if not isinstance(options, dict) or set(options) - {"snap", "format", "wind"}:
        raise SelectionError("Options must be an object with snap, format and wind only")
    out = dict(quantity=name, axes=axes, options=dict(snap=bool(options.get("snap", True)),
                                                      format=options.get("format", "json"),
                                                      wind=options.get("wind") is True))
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
            # A vertical reduction acts on the native levels, whatever representation was shown.
            entry["representation"] = "native"
    out["vertical"] = entry
    axis_id = quantity.get("time_axis")
    if entry["role"] == "fixed":
        axis_id = level_time_axis(quantity, entry) or axis_id
    time = coords["time"]
    entry = dict(role=time["role"], axis=axis_id)
    if quantity.get("recorded"):          # the whole record, at its own cadence
        out["time"] = entry
        if out["options"]["wind"]:
            _wind_layer(quantity, out)
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
    if out["options"]["wind"]:
        _wind_layer(quantity, out)
    return out


def _reduction(quantity, coordinate, part):
    op = part.get("op", "mean")
    caps = quantity["coordinates"][coordinate]
    if op not in caps.get("reduce", {}):
        reason = (caps.get("reduce_unavailable") or {}).get(op)
        raise SelectionError(f"The reduction {op!r} over {coordinate} is not available" +
                             (f": {reason}" if reason else ""))
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


def _reduce(array, weights, axis, op="mean", masked=None):
    """``op`` along ``axis`` (kept, length 1).

    ``mean``: the plain mean for equal weights (the arithmetic of the
    zonal-mean accumulator), else the normalized weighted sum; NaN (masked)
    propagates. ``min``/``max``: the extremum of the valid samples, NaN only
    where all are masked; ``masked`` (a one-element list) counts the masked
    samples excluded."""
    if op in ("min", "max"):
        invalid = np.isnan(array)
        if masked is not None:
            masked[0] += int(invalid.sum())
        fill = np.inf if op == "min" else -np.inf
        out = (np.where(invalid, fill, array).min if op == "min" else np.where(invalid, fill, array).max)(
            axis=axis, keepdims=True)
        out[np.isinf(out) & invalid.all(axis=axis, keepdims=True)] = np.nan
        return out
    if weights is None:
        return array.mean(axis=axis, keepdims=True)
    w = np.asarray(weights, dtype=float)
    shape = [1] * array.ndim
    shape[axis] = w.size
    return (array * w.reshape(shape)).sum(axis=axis, keepdims=True) / w.sum()


def _spatial(array, plan, sel, counts=None, same=False):
    """Reduce one saved sample ``(nvertical, nlat, nlon)``: longitude, latitude,
    then vertical (:data:`REDUCTION_ORDER`). ``counts`` is None for companions,
    which take the mean whatever the quantity's operator unless ``same`` (a
    companion that is a value of the quantity, such as a skin temperature)."""
    for c, axis_ in (("longitude", -1), ("latitude", -2), ("vertical", -3)):
        part = sel[c]
        if part["role"] != "reduce":
            continue
        op = part["op"] if counts is not None or same else "mean"
        array = _reduce(array, (plan.get(c) or {}).get("weights"), axis_, op,
                        None if counts is None else counts[c])
    return array


def _same(plan, name):
    """A companion that is a value of the quantity itself takes the quantity's statistic."""
    return (plan.get("companions") or {}).get(name, {}).get("statistic") == "quantity"


def time_weights(seconds):
    """Cell weights of saved times: ``wᵢ = (tᵢ₊₁ − tᵢ₋₁)/2``, ends mirrored.

    ``None`` when every weight is equal (a regular cadence, one or two
    samples): the mean is then the plain ``1/N`` mean."""
    t = np.asarray(seconds, dtype=float)
    if t.size < 3:
        return None
    w = np.empty_like(t)
    w[1:-1] = (t[2:] - t[:-2]) / 2
    w[0], w[-1] = t[1] - t[0], t[-1] - t[-2]
    return None if np.all(w == w[0]) else w


class _TimeReduction:
    """Accumulates one reduction over saved samples in bounded memory."""

    def __init__(self, op, weights, same=()):
        self.op, self.weights, self.masked, self.n = op, weights, [0], 0
        self.value, self.companions, self.same = None, {}, set(same)

    def _accumulate(self, total, values, op, w):
        if total is None:
            total = np.zeros_like(values) if op == "mean" else np.full_like(values, np.nan)
        if op == "mean":
            return total + (values if self.weights is None else w * values)
        return (np.fmin if op == "min" else np.fmax)(total, values)

    def add(self, values, companions):
        w = 1.0 if self.weights is None else self.weights[self.n]
        self.n += 1
        if self.op != "mean":
            self.masked[0] += int(np.isnan(values).sum())
        if self.value is None and self.op == "mean":
            self.value = np.zeros_like(values)
            self.value += values if self.weights is None else w * values      # the original arithmetic
        else:
            self.value = self._accumulate(self.value, values, self.op, w)
        for k, a in companions.items():
            op = self.op if k in self.same else "mean"
            self.companions[k] = self._accumulate(self.companions.get(k), a, op, w)

    def result(self):
        total = self.n if self.weights is None else float(np.sum(self.weights))
        value = self.value / total if self.op == "mean" else self.value
        return value[None], {k: (a if k in self.same and self.op != "mean" else a / total)[None]
                             for k, a in self.companions.items()}


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
    counts = {c: [0] for c in COORDINATES}
    seconds, stacked, stacked_companions = [], [], []
    reduction = None
    if time_role == "reduce":
        values_ = (provider.manifest["time_axes"].get(sel["time"]["axis"]) or {}).get("values") or []
        # Cell weights whatever the quantity's operator: mean-reduced companions (p, Z) use them
        # under Min/Max too, so an interval's mean positions do not depend on the operator.
        weights = (time_weights([values_[i] for i in indices])
                   if values_ and len(values_) > indices[-1] else None)
        reduction = _TimeReduction(sel["time"]["op"], weights,
                                   [k for k in plan.get("companions") or {} if _same(plan, k)])
    for t, values, companions in provider.samples(sel, plan, indices):
        values = _spatial(np.asarray(values, dtype=float), plan, sel, counts)
        companions = {k: _spatial(np.asarray(a, dtype=float), plan, sel, same=_same(plan, k))
                      for k, a in companions.items()}
        seconds.append(t)
        if reduction is not None:
            reduction.add(values, companions)
        else:
            stacked.append(values)
            stacked_companions.append(companions)
    if reduction is not None:
        cube, companion_cubes = reduction.result()
        counts["time"] = reduction.masked
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
        payload["coordinates"]["time"] = _time_description(provider.manifest, sel, seconds,
                                                           reduction.weights if reduction else None)
    reduced = [c for c in REDUCTION_ORDER if sel[c]["role"] == "reduce"]
    for c in reduced:
        entry = payload["coordinates"].setdefault(c, dict(role="reduce"))
        entry.setdefault("op", sel[c]["op"])
        if sel[c]["op"] != "mean":
            entry["rule"] = EXTREMUM.format(op="minimum" if sel[c]["op"] == "min" else "maximum")
            entry["masked_excluded"] = counts[c][0]
    if len(reduced) > 1:
        payload["reduction_order"] = reduced
        if len({sel[c]["op"] for c in reduced}) > 1:
            payload["notes"].append("Reductions applied in the order " + " → ".join(
                f"{c} {sel[c]['op']}" for c in reduced) + "; with different operators the order matters.")
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


def _time_description(manifest, sel, seconds, weights=None):
    time = sel["time"]
    values = np.asarray(seconds, dtype=float)
    steps = np.diff(values)
    intervals = {str(float(s)): int(n) for s, n in zip(*np.unique(steps, return_counts=True))}
    out = dict(role=time["role"], axis=time["axis"], first=time["first"], last=time["last"],
               count=len(seconds), intervals_seconds=intervals,
               first_seconds=float(values[0]), last_seconds=float(values[-1]),
               cadence="single sample" if len(seconds) == 1 else
                       "regular" if len(intervals) == 1 else "irregular")
    if time["role"] == "reduce":
        out["op"] = time["op"]
        if time["op"] == "mean":
            out["rule"] = TIME_MEAN
            out["weighting"] = "equal (regular cadence)" if weights is None else "cell weights (irregular cadence)"
            if weights is not None:
                out["weights_seconds"] = [float(w) for w in weights]
        elif weights is not None:
            out["companion_weighting"] = "cell weights (irregular cadence) for mean-reduced companions"
            out["weights_seconds"] = [float(w) for w in weights]
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
