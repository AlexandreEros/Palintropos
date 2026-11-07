"""3-D atmosphere view of a saved primitive-equation run (CPU export).

``Simulation.view()`` ends here. It writes ONE self-contained HTML file: the
terrain globe of :mod:`~tropoi.representation.visual.terrain` (same
template, camera, mesh conventions, interaction and offline packaging) plus
one translucent atmospheric surface at a selected NATIVE sigma full level.

Geometry. The surface of level k sits at the radius

    r = R + E * Z_k,      Z_k = Phi_k / g,

where ``Phi_k`` is the run's absolute hydrostatic full-level geopotential
(Simmons–Burridge, including the surface geopotential ``Phi_s = g h_s``
exactly once; :mod:`~tropoi.representation.diagnostics.pe_levels`), ``g``
the model's gravity, ``R`` the run's dynamical radius and ``E`` the display
exaggeration. Terrain is drawn at ``R + E * h_s`` with the SAME ``E`` and
the same reference sphere, so the two keep their physical relation; ``E``
changes display geometry only. Nothing is normalized per level and no mean
is removed. ``Z`` is geopotential height (constant ``g``, spherical
geopotential: the model's own convention); using it as a radial distance
treats geopotential height as geometric height, which is what the model's
shallow-atmosphere, constant-gravity formulation does.

Fields per exported (time, level): temperature, pressure ``sigma_k p_s``,
relative vorticity and divergence; geometry ``Z_k``; optional horizontal
streamlines of that state's wind on that level
(:mod:`~tropoi.representation.visual.sphere_streamlines`), draped on the
displayed surface by the page. Colour limits are fixed per field across all
exported times and levels by default (a per-level option is in the page):
sequential for temperature and pressure, symmetric about zero for vorticity
and divergence.

Size. Values are stored as 16-bit integers per (field, time, level) slice
with that slice's own min/max (display quantization, at most
(max - min)/131070 per value). The payload is estimated BEFORE anything is
reconstructed. With ``times=None`` as many evenly spaced saved states as fit
``max_mb`` are exported (first and last always included) and a warning
names the subsampling; if not even the endpoints fit (or, for a
single-snapshot run, its one state), the export raises. An explicit
selection that does not fit raises too.

Display resolution (``resolution_deg``) is a property of the page only, not
of the run: fields are evaluated exactly from the saved coefficients on that
mesh (host synthesis), whatever grid the run used.

CPU only: NumPy, the archive readers and Matplotlib (colour tables). Never
CuPy, never a model, never writes into the run directory.
"""
from __future__ import annotations

import base64
import hashlib
import math
import pathlib
import re
import tempfile
import warnings

import numpy as np

from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
from tropoi.representation.visual.sphere_streamlines import (
    integrate_streamlines)
from tropoi.representation.visual.terrain import (_color_map,
                                                  build_terrain_preview,
                                                  globe_payload,
                                                  write_globe_page)

__all__ = [
    "AtmosphereViewWarning",
    "DEFAULT_MAX_MB",
    "FIELDS",
    "PayloadTooLargeError",
    "atmosphere_payload",
    "default_resolution_deg",
    "estimate_payload_bytes",
    "view_atmosphere",
]

#: Default bound on the embedded data (MB, after base64 encoding).
DEFAULT_MAX_MB = 32.0

#: Field id -> (label, display units, SI-to-display divisor, scale, colours).
FIELDS = {
    "temperature": ("Temperature", "K", 1.0, "sequential",
                    "inferno:0.25:1.0"),
    "pressure": ("Pressure", "hPa", 100.0, "sequential", "viridis"),
    "zeta": ("Relative vorticity", "10⁻⁵ s⁻¹", 1.0e-5, "diverging",
             "RdBu_r"),
    "delta": ("Divergence", "10⁻⁵ s⁻¹", 1.0e-5, "diverging", "RdBu_r"),
}

#: Streamline settings (bounded): seeds, step, steps each way.
STREAMLINE_SEEDS = 160
STREAMLINE_STEP_DEG = 1.5
STREAMLINE_MAX_STEPS = 24
STREAMLINE_MIN_SPEED = 1.0e-3      # m/s: calmer air draws no line
_STREAMLINE_POINT_BYTES = 4        # int16 lat, lon in centidegrees

_RESOLUTIONS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


_PACKAGE_DIR = str(pathlib.Path(__file__).resolve().parents[2])


class AtmosphereViewWarning(UserWarning):
    """The export differs from "everything saved" (e.g. subsampled times)."""


class PayloadTooLargeError(ValueError):
    """The requested export exceeds the payload bound."""


def default_resolution_deg(l_max: int) -> float:
    """About four mesh nodes per shortest resolved wavelength (360/l_max
    degrees), rounded down to a round spacing in [1, 3] degrees."""
    target = 90.0 / max(int(l_max), 1)
    candidates = [r for r in _RESOLUTIONS if r <= target]
    return candidates[-1] if candidates else _RESOLUTIONS[0]


def _mesh(resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    d = float(resolution_deg)
    if not (math.isfinite(d) and 0.25 <= d <= 10.0):
        raise ValueError(f"resolution_deg must be in [0.25, 10], got {d}")
    nlat = int(round(180.0 / d))
    nlon = int(round(360.0 / d))
    if not (math.isclose(nlat * d, 180.0) and math.isclose(nlon * d, 360.0)):
        raise ValueError(f"resolution_deg {d} must divide 180 degrees")
    return (np.linspace(-90.0, 90.0, nlat + 1),
            np.linspace(-180.0, 180.0, nlon + 1))


def estimate_payload_bytes(n_times: int, n_levels: int, nlat: int, nlon: int,
                           *, streamlines: bool = True) -> int:
    """Upper bound of the encoded atmosphere data (bytes)."""
    nodes = nlat * nlon
    per_slice = 2 * nodes * (len(FIELDS) + 1)            # 4 fields + height
    if streamlines:
        per_slice += STREAMLINE_SEEDS * (
            (2 * STREAMLINE_MAX_STEPS + 1) * _STREAMLINE_POINT_BYTES + 4)
    raw = n_times * n_levels * per_slice
    return int(math.ceil(raw * 4 / 3)) + 200_000            # + metadata


def _terrain_bytes(l_max: int) -> int:
    from tropoi.representation.visual.terrain import display_resolution_deg
    d = display_resolution_deg(l_max)
    return int((180 / d + 1) * (360 / d + 1) * 4 * 4 / 3)


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(
        np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _quantize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = float(values.min()), float(values.max())
    if hi > lo:
        q = np.rint((values - lo) / (hi - lo) * 65535.0)
    else:
        q = np.zeros(values.shape)
    return q.astype("<u2"), lo, hi


def _lut(policy: str) -> list:
    cmap = _color_map(policy)
    rgb = np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3] * 255.0
    return np.rint(rgb).astype(int).tolist()


def _time_label(t: float) -> str:
    days = t / 86400.0
    if math.isclose(days, round(days), abs_tol=1e-9):
        day = f"day {int(round(days))}"
    else:
        day = f"day {days:.6g}"
    return f"{day} (t = {t:.10g} s)"


def _select_times(times, count: int, requested, budget, per_time: int,
                  fixed: int) -> tuple[list[int], str | None]:
    if count == 0:
        raise ValueError("no saved snapshots")
    note = None
    if isinstance(requested, str):
        if requested != "all":
            raise ValueError(f"times must be None, 'all' or saved indices, "
                             f"got {requested!r}")
        chosen = list(range(count))
    elif requested is None:
        if budget is None:
            return list(range(count)), None
        # The automatic contract: every state when they all fit, otherwise
        # evenly spaced states that always include the first and the last.
        # The minimum is therefore one state for a single-snapshot run and
        # two (both endpoints) otherwise; below that the export is refused
        # here, before any field is reconstructed.
        minimum = 1 if count == 1 else 2
        fit = int((budget - fixed) // per_time)
        if fit < minimum:
            if count == 1:
                what, single = "the single saved state", ""
            else:
                what = f"the first and last of {count} saved states"
                single = (", or pass an explicit times=[i] if one state "
                          "is what you want")
            raise PayloadTooLargeError(
                f"{what} would embed about "
                f"{(fixed + per_time * minimum) / 1e6:.3g} MB, above the "
                f"{budget / 1e6:.3g} MB bound (max_mb). Choose fewer levels "
                "or a coarser resolution_deg, or raise max_mb (max_mb=None "
                f"removes the bound){single}")
        if fit >= count:
            return list(range(count)), None
        # 2 <= fit < count: distinct indices, 0 and count - 1 included
        chosen = sorted({int(round(x)) for x in
                         np.linspace(0, count - 1, fit)})
        days = ", ".join(f"{times[i] / 86400:g}" for i in chosen)
        note = (
            f"exported {len(chosen)} of {count} saved states, evenly "
            f"subsampled to fit the {budget / 1e6:.3g} MB payload bound "
            f"(saved indices {chosen}; days {days}). Pass times='all', a "
            "list of saved indices, fewer levels, a coarser resolution_deg "
            "or a larger max_mb to choose otherwise")
    else:
        chosen = []
        for i in requested:
            if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
                raise TypeError(f"times are saved snapshot indices (int), "
                                f"got {i!r}")
            j = int(i) + (count if int(i) < 0 else 0)
            if not 0 <= j < count:
                raise IndexError(f"saved index {i} is out of range for "
                                 f"{count} saved snapshots")
            chosen.append(j)
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"repeated saved indices {list(requested)}")
        chosen.sort()
    # Every path (automatic, 'all', explicit) passes this guard.
    if budget is not None and fixed + per_time * len(chosen) > budget:
        raise PayloadTooLargeError(
            f"{len(chosen)} saved states would embed about "
            f"{(fixed + per_time * len(chosen)) / 1e6:.3g} MB, above the "
            f"{budget / 1e6:.3g} MB bound (max_mb). Choose fewer times or "
            "levels, a coarser resolution_deg, or raise max_mb (max_mb=None "
            "removes the bound)")
    return chosen, note


def _levels(levels, nlev: int) -> list[int]:
    if levels is None:
        return list(range(nlev))
    out = []
    for k in levels:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError(f"levels are native level indices (int, 0 = "
                            f"top), got {k!r}")
        j = int(k) + (nlev if int(k) < 0 else 0)
        if not 0 <= j < nlev:
            raise IndexError(f"level {k} is out of range for {nlev} native "
                             "levels")
        out.append(j)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated levels {list(levels)}")
    return sorted(out)


def _nice(raw: float) -> float:
    if raw <= 1.0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    return float(max(1.0, min((1, 2, 2.5, 5, 10),
                              key=lambda f: abs(f * mag - raw)) * mag))


def _streamline_block(lines_per_slice) -> dict:
    points, line_start, slice_start = [], [0], [0]
    for lines in lines_per_slice:
        for lat, lon in lines:
            pts = np.stack([np.rint(lat * 100.0), np.rint(lon * 100.0)],
                           axis=1)
            points.append(pts)
            line_start.append(line_start[-1] + pts.shape[0])
        slice_start.append(len(line_start) - 1)
    flat = (np.concatenate(points, axis=0) if points
            else np.zeros((0, 2)))
    return {"points": _b64(flat.ravel(), "<i2"),
            "line_start": _b64(np.asarray(line_start), "<u4"),
            "slice_line_start": slice_start}


def atmosphere_payload(storage, *, times=None, levels=None,
                       resolution_deg: float | None = None,
                       streamlines: bool = True,
                       max_mb: float | None = DEFAULT_MAX_MB,
                       gravity: float | None = None, allow_cuda: bool = False,
                       field: str = "temperature", level: int | None = None,
                       time: int | None = None, opacity: float = 0.6,
                       exaggeration: float | None = None,
                       centre: tuple[float, float] | None = None
                       ) -> tuple[dict, dict]:
    """The page document and an export report (exposed for tests).

    Returns ``(payload, report)``; ``report`` has the selected saved
    indices and levels, the mesh spacing, the payload estimate (bytes,
    computed before reconstruction), the selection note, the column
    context and the initial exaggeration.
    """
    if field not in FIELDS:
        raise ValueError(f"field must be one of {list(FIELDS)}, got "
                         f"{field!r}")
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}")
    if storage.solver != "pe":
        raise ValueError(
            f"the 3-D atmosphere view is for primitive-equation runs; this "
            f"is a {storage.solver!r} run")
    count = int(len(storage.times))
    if count == 0:
        from tropoi.representation.archive.availability import (
            NoSnapshotsError, statistics_only_description)
        reason = statistics_only_description(storage.run_dir) or (
            f"{storage.run_dir} stores no snapshots")
        raise NoSnapshotsError(reason)
    context = storage.pe_context(allow_cuda=allow_cuda).with_gravity(gravity)
    if context.gravity is None:
        raise ValueError(
            "this run records no gravity (a flat tropoi run pe preset never "
            "uses one), so geopotential height Phi/g is undefined. Pass the "
            "gravity to read it with, e.g. view(gravity=9.80665) for "
            "geopotential metres; it scales the displayed height only")
    chosen_levels = _levels(levels, context.nlev)
    res = float(resolution_deg or default_resolution_deg(context.l_max))
    lat, lon = _mesh(res)
    nlat, nlon = lat.size, lon.size
    budget = None if max_mb is None else float(max_mb) * 1e6
    per_time = estimate_payload_bytes(1, len(chosen_levels), nlat, nlon,
                                      streamlines=streamlines) - 200_000
    fixed = 200_000 + _terrain_bytes(context.l_max)
    times_s = np.asarray(storage.times, dtype=np.float64)
    chosen, note = _select_times(times_s, count, times, budget, per_time,
                                 fixed)
    if note is not None:
        # attribute the warning to the caller's line, outside this package
        warnings.warn(note, AtmosphereViewWarning,
                      skip_file_prefixes=(_PACKAGE_DIR,))
    estimate = fixed + per_time * len(chosen)

    # -- reconstruction (CPU), one saved state at a time -------------------
    nl = len(chosen_levels)
    data = {name: [] for name in (*FIELDS, "height")}
    bounds = {name: [] for name in (*FIELDS, "height")}
    stats = {name: np.zeros((nl, 2)) for name in FIELDS}
    stats_init = False
    lines = []
    max_speed = 0.0
    for index in chosen:
        f = reconstruct_levels(storage.frame(index), context, lat, lon,
                               chosen_levels, winds=streamlines)
        source = {"temperature": f["temperature"], "pressure": f["pressure"],
                  "zeta": f["zeta"], "delta": f["delta"],
                  "height": f["geopotential_height"]}
        for name, values in source.items():
            for i in range(nl):
                q, lo, hi = _quantize(values[i])
                data[name].append(q)
                bounds[name].append((lo, hi))
        for name in FIELDS:
            lo = source[name].min(axis=(1, 2))
            hi = source[name].max(axis=(1, 2))
            if not stats_init:
                stats[name][:, 0], stats[name][:, 1] = lo, hi
            else:
                stats[name][:, 0] = np.minimum(stats[name][:, 0], lo)
                stats[name][:, 1] = np.maximum(stats[name][:, 1], hi)
        stats_init = True
        if streamlines:
            for i in range(nl):
                speed = np.hypot(f["u"][i], f["v"][i])
                max_speed = max(max_speed, float(speed.max()))
                lines.append(integrate_streamlines(
                    f["u"][i], f["v"][i], lat, lon,
                    seed_count=STREAMLINE_SEEDS,
                    step_deg=STREAMLINE_STEP_DEG,
                    max_steps=STREAMLINE_MAX_STEPS,
                    min_speed=STREAMLINE_MIN_SPEED))

    # -- colour limits: fixed per field over all exported times and levels -
    fields = []
    for name, (label, units, factor, scale, policy) in FIELDS.items():
        lo_l, hi_l = stats[name][:, 0], stats[name][:, 1]
        if scale == "diverging":
            per_level = [[-b, b] for b in np.maximum(np.abs(lo_l),
                                                     np.abs(hi_l))]
            bound = max(b for _, b in per_level)
            whole = [-bound, bound]
        else:
            per_level = [[float(a), float(b)] for a, b in zip(lo_l, hi_l)]
            whole = [float(lo_l.min()), float(hi_l.max())]
        fields.append({
            "id": name, "label": label, "units": units, "factor": factor,
            "scale": scale, "lut": _lut(policy), "colours": policy,
            "range": [float(v) for v in whole],
            "level_ranges": [[float(a), float(b)] for a, b in per_level],
            "data": _b64(np.concatenate([q.ravel() for q in data[name]]),
                         "<u2"),
            "bounds": [[lo, hi] for lo, hi in bounds[name]],
        })
    height = {"data": _b64(np.concatenate([q.ravel() for q in
                                           data["height"]]), "<u2"),
              "bounds": [[lo, hi] for lo, hi in bounds["height"]]}
    z_max = max(hi for _, hi in bounds["height"])

    # -- terrain (same radius, same datum) ---------------------------------
    elevation = context.surface_elevation_lm
    n = context.l_max + 1
    title = f"Atmosphere: {context.label}"
    preview = build_terrain_preview(
        np.zeros((n, n), complex) if elevation is None else elevation,
        radius_m=context.radius_m, title=title,
        subtitle="", details=())
    if exaggeration is None:
        exaggeration = _nice(0.08 * context.radius_m / max(z_max, 1.0))
    if centre is None:
        centre = preview.highest_point() if elevation is not None else (
            35.0, 0.0)
    page = globe_payload(preview, units="m" if elevation is None else None,
                         centre=centre, exaggeration=exaggeration)
    full = context.full_levels
    k_default = chosen_levels[len(chosen_levels) // 2] if level is None \
        else int(level)
    if k_default not in chosen_levels:
        raise ValueError(f"initial level {level} is not among the exported "
                         f"levels {chosen_levels}")
    t_default = len(chosen) - 1 if time is None else int(time)
    if not 0 <= t_default < len(chosen):
        raise ValueError(f"initial time position {time} is outside the "
                         f"{len(chosen)} exported times")
    g = float(context.gravity)
    interpretation = [
        f"Surface radius: R + E·Z, with Z = Φ/g the geopotential height of "
        f"the native full level (Simmons–Burridge hydrostatic geopotential "
        f"Φ of the saved temperature, absolute: it includes the surface "
        f"geopotential Φs = g·hs once), g = {g:g} m s⁻², R = "
        f"{context.radius_m / 1000:.6g} km, E the display exaggeration.",
        "Terrain: R + E·hs with the same E and the same reference sphere, "
        "so atmosphere and terrain keep their physical relation; E changes "
        "display geometry only. No level is normalized and no mean height "
        "is removed.",
        "Geopotential height is used as radial distance: the model's "
        "constant-gravity, spherical-geopotential convention (geometric and "
        "geopotential height coincide in it).",
        f"Pressure on the level: p = σk·ps (pure sigma, model top p = 0); "
        f"it varies along the surface with the saved surface pressure.",
        "Streamlines: instantaneous horizontal streamlines of the saved "
        "wind on this level (tangent to it at one saved time), draped on "
        "the displayed surface. Not parcel trajectories, not 3-D "
        "streamlines; no vertical motion is computed or implied.",
        "Values: exact synthesis of the saved coefficients on a "
        f"{res:g}° display mesh, stored with 16-bit display quantization per "
        "slice.",
    ]
    details = [
        ["Run", context.label],
        ["Archive", context.kind],
        ["Saved snapshots", f"{count} (exported {len(chosen)})"],
        ["Native levels", f"{context.nlev} sigma full levels, 1 = top "
                          f"(exported {nl})"],
        ["Vertical coordinate", context.pressure_convention],
        ["Sigma interfaces", ", ".join(f"{s:.6g}"
                                       for s in context.sigma_interfaces)],
        ["R_d, c_p", f"{context.r_dry:.6g}, {context.cp_dry:.6g} J kg⁻¹ K⁻¹"],
        ["Gravity", f"{g:g} m s⁻² ({context.provenance.get('gravity')})"],
        ["Radius", f"{context.radius_m:.10g} m "
                   f"({context.provenance.get('radius_m')})"],
        ["Terrain", context.terrain_label],
        ["Spectral truncation", f"l_max = {context.l_max} (saved "
                                "coefficients)"],
        ["Display mesh", f"{res:g}° atmosphere, {preview.resolution_deg:g}° "
                         "terrain (display only)"],
    ]
    if note:
        details.append(["Time selection", note])
    page["title"] = title
    page["subtitle"] = (f"Primitive-equation run on native sigma levels · "
                        f"{len(chosen)} saved time(s) × {nl} level(s)")
    page["details"] = details
    page["atmosphere"] = {
        "version": 1,
        "mesh": {"nlat": nlat, "nlon": nlon, "lat0": float(lat[0]),
                 "lon0": float(lon[0]), "dlat": res, "dlon": res},
        "times": [{"index": int(i), "time_s": float(times_s[i]),
                   "label": _time_label(float(times_s[i]))} for i in chosen],
        "saved_count": count,
        "selection_note": note,
        "levels": [{"index": k, "number": k + 1, "sigma": float(full[k])}
                   for k in chosen_levels],
        "nlev": context.nlev,
        "gravity": g, "radius_m": context.radius_m,
        "terrain_flat": elevation is None,
        "fields": fields,
        "height": height,
        "streamlines": (dict(_streamline_block(lines), enabled=True,
                             max_speed=max_speed,
                             min_speed=STREAMLINE_MIN_SPEED)
                        if streamlines else {"enabled": False}),
        "defaults": {"time": t_default,
                     "level": chosen_levels.index(k_default),
                     "field": field, "opacity": float(opacity),
                     "streamlines": bool(streamlines)},
        "interpretation": interpretation,
    }
    report = {"times": chosen, "levels": chosen_levels,
              "resolution_deg": res, "estimate_bytes": estimate,
              "selection_note": note, "context": context,
              "exaggeration": float(exaggeration)}
    return page, report


def _default_path(storage, report) -> pathlib.Path:
    label = report["context"].label
    base = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()[:48]
    key = repr((str(pathlib.Path(storage.run_dir).resolve()), report["times"],
                report["levels"], report["resolution_deg"])).encode()
    tag = hashlib.sha256(key).hexdigest()[:8]
    return pathlib.Path(tempfile.gettempdir()) / "tropoi" / (
        f"{base or 'run'}-atmosphere-{tag}.html")


def _check_outside_run(path: pathlib.Path, run_dir) -> None:
    run = pathlib.Path(run_dir).resolve()
    target = path.resolve()
    if target == run or run in target.parents:
        assets = run / "assets"
        if assets not in target.parents:
            raise ValueError(
                f"{path} is inside the run directory {run}; a view never "
                "adds files to a saved run (only its assets/ directory may "
                "hold derived products)")


def view_atmosphere(storage, path=None, *, open_browser: bool = True,
                    **options) -> pathlib.Path:
    """Write the atmosphere page for ``storage``; open it; return its path.

    ``options`` are those of :func:`atmosphere_payload`. ``path`` defaults
    to ``<tempdir>/tropoi/<run>-atmosphere-<tag>.html`` (outside the run).
    """
    payload, report = atmosphere_payload(storage, **options)
    if path is None:
        path = _default_path(storage, report)
    path = pathlib.Path(path)
    _check_outside_run(path, storage.run_dir)
    path = write_globe_page(payload, payload["title"], path)
    if open_browser:
        import webbrowser
        webbrowser.open(path.resolve().as_uri())
    return path
