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

Derived surface (``surface=("pressure", Pa)``, issue #13, first slice).
Instead of native levels the page carries ONE fixed-pressure surface per
exported saved time, from :mod:`~tropoi.representation.diagnostics.
pe_surfaces` (log-sigma interpolation of the reconstructed column; masks
for below-ground, above-top-level and below-bottom-level nodes). Fields:
temperature, geopotential height and wind speed; pressure is constant on
the surface, and vorticity/divergence are not exported (a sigma-level
derivative is not a pressure-level one). Masked nodes carry no sample: the
page drops the triangles touching them, greys them on the flat map, names
the reason on hover and clips streamlines at them; colour ranges and
quantization use valid nodes only. Everything else (budget, quantization,
playback, offline packaging) is the native-level path. ``levels=`` and
``surface=`` are exclusive; without ``surface`` nothing here changes.

Interactive pressure (``interactive_pressure=True``, issue #13, second
slice). The page carries the NATIVE COLUMNS of each exported saved state
(temperature, geopotential, u, v on every full level, 16-bit per level
slice; the surface pressure as float32) and the shared level positions, and
performs the vertical interpolation of :mod:`pe_surfaces` itself for any
pressure typed or slid in the browser: the same ``ln sigma`` rule, end
tolerance, mask codes and refusal to extrapolate. ``surface=`` gives the
initial isobar. Geometry, fields and masks of a frame always come from one
(pressure, saved time) pair computed in one pass; only the selected saved
state is decoded (two held at most). Colour ranges are conservative bounds
from the exported columns, fixed while the pressure moves. Streamlines are
traced in the page from the interpolated ``u, v`` of the surface shown, by a
transcription of :func:`~tropoi.representation.visual.sphere_streamlines.
integrate_streamlines` (same seeds, steps, stopping rules and mask
clipping; the tests compare the two on identical winds): nothing extra is
exported for them, and while a slider is dragged they are traced once the
selection settles, the previous surface's lines being cleared at once. The
payload is estimated before reconstruction and the decoded memory per saved
state is reported separately; ``times="all"`` fails above the budget as
elsewhere.

Geopotential-height surfaces (``surface=("geopotential_height", m)``).
The same two modes for the other derived coordinate: one height surface
per exported saved time (temperature, pressure, wind speed; masks below
the terrain, above the highest and between the terrain and the lowest full
level), or, in a column page, any height chosen in the browser. Every
column page (``interactive_surface=True``; ``interactive_pressure=True`` is
the same switch under its older name) offers both coordinates, switchable
without regenerating the file; ``surface`` picks the initial one. The
surface sits at ``R + E * Z0`` for every valid node, the datum of the
terrain and of the native reconstruction (``Phi = g Z``, the model's ``g``);
validity uses the physical surface geopotential ``g h_s`` (exported once,
float32), never the exaggerated display. The inversion is
:func:`~tropoi.representation.diagnostics.pe_surfaces.
interpolate_geopotential`, mirrored by the page.

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
from tropoi.representation.diagnostics.pe_surfaces import (
    END_TOLERANCE, HEIGHT_MASK_LABELS, MASK_LABELS, MASK_VALID,
    SurfaceUnavailableError, level_sigma_positions, parse_surface,
    pressure_columns, pressure_surface)
from tropoi.representation.visual.sphere_streamlines import (
    integrate_streamlines)
from tropoi.representation.visual.terrain import (_color_map,
                                                  build_terrain_preview,
                                                  globe_payload,
                                                  write_globe_page)

__all__ = [
    "AtmosphereViewWarning",
    "COLUMN_FIELDS",
    "DEFAULT_MAX_MB",
    "FIELDS",
    "HEIGHT_SURFACE_FIELDS",
    "MAX_DECODED_TIMES",
    "PayloadTooLargeError",
    "SURFACE_FIELDS",
    "atmosphere_payload",
    "decoded_column_bytes",
    "default_resolution_deg",
    "dequantize_columns",
    "estimate_column_payload_bytes",
    "estimate_payload_bytes",
    "streamline_cache_bytes",
    "surface_height_bytes",
    "quantize_columns",
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

#: Fields of a derived pressure surface (same tuple layout as ``FIELDS``).
#: Pressure is constant on the surface; vorticity and divergence are not
#: exported (no target-coordinate derivative in this slice of issue #13).
SURFACE_FIELDS = {
    "temperature": FIELDS["temperature"],
    "geopotential_height": ("Geopotential height", "m", 1.0, "sequential",
                            "viridis"),
    "wind_speed": ("Wind speed", "m s⁻¹", 1.0, "sequential", "magma"),
}

#: Fields of a derived geopotential-height surface: the height is constant
#: on it, pressure is ``p_s exp(ln sigma*)``.
HEIGHT_SURFACE_FIELDS = {
    "temperature": FIELDS["temperature"],
    "pressure": FIELDS["pressure"],
    "wind_speed": SURFACE_FIELDS["wind_speed"],
}

#: Fields of a column page and the coordinates each is offered on (the
#: first three in the order of the earlier pressure-only page).
COLUMN_PAGE_FIELDS = (
    ("temperature", ("pressure", "geopotential_height")),
    ("geopotential_height", ("pressure",)),
    ("wind_speed", ("pressure", "geopotential_height")),
    ("pressure", ("geopotential_height",)),
)

#: The page's own mask code for a stored column it cannot invert (the
#: export refuses such columns, so only an altered file shows it).
MASK_INVALID_STORED = 4
_INVALID_STORED_LABEL = ("invalid stored column (geopotential not "
                         "decreasing downward or not above the terrain)")

#: Native stacks the column page carries per saved state, in this order;
#: wind speed, geopotential height and pressure are derived in the page.
COLUMN_FIELDS = ("temperature", "geopotential", "u", "v")

#: Decoded saved states the interactive page keeps at once (LRU).
MAX_DECODED_TIMES = 2

#: Streamline settings (bounded): seeds, step, steps each way.
STREAMLINE_SEEDS = 160
STREAMLINE_STEP_DEG = 1.5
STREAMLINE_MAX_STEPS = 24
STREAMLINE_MIN_SPEED = 1.0e-3      # m/s: calmer air draws no line
STREAMLINE_RELATIVE_MIN_SPEED = 0.02   # ... or air below 2 % of the maximum
#: Interactive pages trace streamlines in the browser: while a slider is
#: dragged, this long after the last movement (or on release); the newest
#: line sets are kept, at most this many.
STREAMLINE_SETTLE_MS = 150
STREAMLINE_CACHE = 8
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
                           *, streamlines: bool = True,
                           n_fields: int = len(FIELDS),
                           mask: bool = False) -> int:
    """Upper bound of the encoded atmosphere data (bytes).

    ``n_fields`` 16-bit fields plus the 16-bit height per node and slice,
    plus one mask byte per node and slice when ``mask`` (derived surfaces).
    """
    nodes = nlat * nlon
    per_slice = 2 * nodes * (n_fields + 1)                # fields + height
    if mask:
        per_slice += nodes
    if streamlines:
        per_slice += STREAMLINE_SEEDS * (
            (2 * STREAMLINE_MAX_STEPS + 1) * _STREAMLINE_POINT_BYTES + 4)
    raw = n_times * n_levels * per_slice
    return int(math.ceil(raw * 4 / 3)) + 200_000            # + metadata


def estimate_column_payload_bytes(n_times: int, n_levels: int, nlat: int,
                                  nlon: int) -> int:
    """Upper bound of the encoded interactive-pressure data (bytes): the
    four 16-bit stacks of ``n_levels`` per node plus a float32 surface
    pressure per node, per saved state, base64-encoded, plus metadata."""
    nodes = nlat * nlon
    per_time = 2 * nodes * n_levels * len(COLUMN_FIELDS) + 4 * nodes
    return int(math.ceil(n_times * per_time * 4 / 3)) + 200_000


#: Browser bytes per node of the arrays derived for the surface shown, held
#: once whatever the number of decoded states: six float32 fields
#: (temperature, geopotential height, pressure, wind speed, u, v), the mask
#: byte, and the float32 terrain height the height masks read.
DERIVED_BYTES_PER_NODE = 6 * 4 + 1 + 4
#: The streamline integrator's working arrays per node: the float64
#: Cartesian wind, a float64 validity indicator and a validity byte.
STREAMLINE_WORK_BYTES_PER_NODE = 3 * 8 + 8 + 1


def streamline_cache_bytes() -> int:
    """Upper bound of the page's cached line sets: :data:`STREAMLINE_CACHE`
    sets of at most ``STREAMLINE_SEEDS`` lines of ``2 * STREAMLINE_MAX_STEPS
    + 1`` float64 (lat, lon) points."""
    return (STREAMLINE_CACHE * STREAMLINE_SEEDS
            * (2 * STREAMLINE_MAX_STEPS + 1) * 16)


def decoded_column_bytes(n_levels: int, nlat: int, nlon: int) -> int:
    """Browser memory of ONE decoded saved state (float32 columns, four
    stacks, and the float32 surface pressure) plus the arrays derived for
    the surface shown and the streamline working arrays
    (:data:`DERIVED_BYTES_PER_NODE`, :data:`STREAMLINE_WORK_BYTES_PER_NODE`).
    The page holds at most :data:`MAX_DECODED_TIMES` decoded states; the
    line cache is bounded apart (:func:`streamline_cache_bytes`)."""
    nodes = nlat * nlon
    return (4 * nodes * n_levels * len(COLUMN_FIELDS) + 4 * nodes
            + nodes * (DERIVED_BYTES_PER_NODE
                       + STREAMLINE_WORK_BYTES_PER_NODE))


def surface_height_bytes(nlat: int, nlon: int) -> int:
    """Encoded size of the terrain height a column page carries ONCE (not
    per saved state) for its height masks: float32 per node, base64."""
    return int(math.ceil(4 * nlat * nlon * 4 / 3))


def _terrain_bytes(l_max: int) -> int:
    from tropoi.representation.visual.terrain import display_resolution_deg
    d = display_resolution_deg(l_max)
    return int((180 / d + 1) * (360 / d + 1) * 4 * 4 / 3)


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(
        np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _quantize(values: np.ndarray, valid=None
              ) -> tuple[np.ndarray, float, float]:
    """16-bit display quantization over ``valid`` nodes (all when None);
    masked nodes are stored as the slice minimum and never read back."""
    inside = values if valid is None else values[valid]
    lo, hi = float(inside.min()), float(inside.max())
    if valid is not None:
        values = np.where(valid, values, lo)
    if hi > lo:
        q = np.rint((values - lo) / (hi - lo) * 65535.0)
    else:
        q = np.zeros(values.shape)
    return q.astype("<u2"), lo, hi


def quantize_columns(columns) -> tuple[dict, dict, np.ndarray]:
    """The stored form of :class:`~tropoi.representation.diagnostics.
    pe_surfaces.PressureColumns`: for each of :data:`COLUMN_FIELDS` a
    ``(K, nodes)`` ``uint16`` array and ``K`` ``[lo, hi]`` bounds (16-bit
    display quantization per level slice, at most ``(hi - lo)/131070`` per
    value), and the float32 surface pressure (``nodes``). The seam column
    (last longitude) is stored equal to the first: the mesh repeats it and
    the synthesis differs there only by round-off.
    """
    quantized, bounds = {}, {}
    for name in COLUMN_FIELDS:
        stack = np.array(getattr(columns, name), dtype=np.float64)
        stack[..., -1] = stack[..., 0]
        q = np.empty((stack.shape[0], stack[0].size), dtype="<u2")
        b = []
        for k in range(stack.shape[0]):
            qk, lo, hi = _quantize(stack[k])
            q[k] = qk.ravel()
            b.append([lo, hi])
        quantized[name], bounds[name] = q, b
    p_s = np.array(columns.surface_pressure, dtype=np.float64)
    p_s[:, -1] = p_s[:, 0]
    return quantized, bounds, p_s.ravel().astype("<f4")


def dequantize_columns(quantized: dict, bounds: dict, p_s: np.ndarray,
                       shape: tuple[int, int], template):
    """The columns the browser works with (float32 arithmetic, as the
    page), as a :class:`~tropoi.representation.diagnostics.pe_surfaces.
    PressureColumns` sharing ``template``'s positions, mesh and gravity:
    the reference for the quantization error of the stored form."""
    from tropoi.representation.diagnostics.pe_surfaces import PressureColumns
    stacks = {}
    for name in COLUMN_FIELDS:
        # exactly the page's arithmetic: float32 lo and step, the product
        # and sum in double, the result rounded to float32 on store
        q = np.asarray(quantized[name], dtype=np.float64)
        out = np.empty(q.shape, dtype=np.float32)
        for k, (lo, hi) in enumerate(bounds[name]):
            step = float(np.float32((hi - lo) / 65535.0))
            out[k] = float(np.float32(lo)) + q[k] * step
        stacks[name] = out.reshape(q.shape[0], *shape).astype(np.float64)
    return PressureColumns(
        level_sigma=template.level_sigma,
        level_positions=template.level_positions, lat_deg=template.lat_deg,
        lon_deg=template.lon_deg, temperature=stacks["temperature"],
        geopotential=stacks["geopotential"], u=stacks["u"], v=stacks["v"],
        surface_pressure=np.asarray(p_s, dtype=np.float32).astype(
            np.float64).reshape(shape),
        surface_height=_stored_height(template.surface_height),
        gravity=template.gravity)


def _stored_height(h):
    """The terrain height as the column page stores it: float32 (the seam
    column equal to the first), or None for a flat lower boundary."""
    if h is None:
        return None
    out = np.array(h, dtype=np.float64)
    out[:, -1] = out[:, 0]
    return out.astype(np.float32).astype(np.float64)


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


_LEVEL_ADVICE = ("Choose fewer times or levels, a coarser resolution_deg, "
                 "or raise max_mb (max_mb=None removes the bound)")
_COLUMN_ADVICE = ("Choose fewer times (an explicit list of saved indices), "
                  "a coarser resolution_deg, or raise max_mb (max_mb=None "
                  "removes the bound); the interactive-pressure page embeds "
                  "every native level of each exported saved state")


def _select_times(times, count: int, requested, budget, per_time: int,
                  fixed: int, advice: str = _LEVEL_ADVICE
                  ) -> tuple[list[int], str | None]:
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
                f"{budget / 1e6:.3g} MB bound (max_mb). {advice}{single}")
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
            f"{budget / 1e6:.3g} MB bound (max_mb). {advice}")
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


def atmosphere_payload(storage, *, times=None, levels=None, surface=None,
                       level_sigma: str = "hydrostatic",
                       interactive_pressure: bool = False,
                       interactive_surface: bool = False,
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
    context and the initial exaggeration. With ``surface=("pressure", Pa)``
    or ``("geopotential_height", m)`` the report also has ``surface`` and
    the per-time ``mask_counts``; with ``interactive_surface=True`` (or its
    older name ``interactive_pressure=True``; ``surface`` is then the
    initial surface) it has ``decoded_bytes_per_time``, ``pressure_range``,
    ``height_range`` and the column ``bounds`` instead of ``mask_counts``.
    """
    derived = surface is not None
    interactive = bool(interactive_pressure or interactive_surface)
    if interactive and not derived:
        raise ValueError(
            "interactive_surface=True (interactive_pressure=True) needs the "
            "initial isobar or height surface: surface=('pressure', Pa) or "
            "surface=('geopotential_height', m)")
    target = parse_surface(surface) if derived else None
    height = derived and target.kind == "geopotential_height"
    table = (dict(COLUMN_PAGE_FIELDS) if interactive else
             HEIGHT_SURFACE_FIELDS if height else
             SURFACE_FIELDS if derived else FIELDS)
    if field not in table or (interactive and target.kind not in
                              dict(COLUMN_PAGE_FIELDS)[field]):
        offered = [f for f, kinds in COLUMN_PAGE_FIELDS
                   if target.kind in kinds] if interactive else list(table)
        raise ValueError(f"field must be one of {offered}, got {field!r}")
    if derived and levels is not None:
        raise ValueError("levels= and surface= are exclusive: a derived "
                         "surface replaces the native levels")
    if derived and level is not None:
        raise ValueError("level= selects a native level; a derived surface "
                         "has none")
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
    chosen_levels = [0] if derived else _levels(levels, context.nlev)
    if derived and context.nlev < 2:
        raise ValueError("a derived surface needs at least two full levels "
                         f"to interpolate between; this run has {context.nlev}")
    res = float(resolution_deg or default_resolution_deg(context.l_max))
    lat, lon = _mesh(res)
    nlat, nlon = lat.size, lon.size
    budget = None if max_mb is None else float(max_mb) * 1e6
    if interactive:
        per_time = estimate_column_payload_bytes(1, context.nlev, nlat,
                                                 nlon) - 200_000
    else:
        per_time = estimate_payload_bytes(1, len(chosen_levels), nlat, nlon,
                                          streamlines=streamlines,
                                          n_fields=len(table),
                                          mask=derived) - 200_000
    fixed = 200_000 + _terrain_bytes(context.l_max)
    if interactive:
        fixed += surface_height_bytes(nlat, nlon)       # once, not per time
    times_s = np.asarray(storage.times, dtype=np.float64)
    chosen, note = _select_times(times_s, count, times, budget, per_time,
                                 fixed, _COLUMN_ADVICE if interactive else
                                 _LEVEL_ADVICE)
    if note is not None:
        # attribute the warning to the caller's line, outside this package
        warnings.warn(note, AtmosphereViewWarning,
                      skip_file_prefixes=(_PACKAGE_DIR,))
    estimate = fixed + per_time * len(chosen)
    t_default = len(chosen) - 1 if time is None else int(time)
    if not 0 <= t_default < len(chosen):
        raise ValueError(f"initial time position {time} is outside the "
                         f"{len(chosen)} exported times")
    if interactive:
        return _interactive_page(
            storage=storage, context=context, target=target,
            level_sigma=level_sigma, res=res, lat=lat, lon=lon, count=count,
            chosen=chosen, times_s=times_s, note=note, estimate=estimate,
            t_default=t_default, field=field, opacity=float(opacity),
            exaggeration=exaggeration, centre=centre,
            streamlines=bool(streamlines))

    # -- reconstruction (CPU), one saved state at a time -------------------
    nl = len(chosen_levels)
    data = {name: [] for name in (*table, "height")}
    bounds = {name: [] for name in (*table, "height")}
    stats = {name: np.zeros((nl, 2)) for name in table}
    stats_init = False
    lines = []
    max_speed = 0.0
    masks, mask_counts = [], []
    for index in chosen:
        if derived:
            # Every full level is needed to bracket the target; winds are
            # needed for the wind-speed field whether or not streamlines are
            # drawn. The full column is dropped after this iteration.
            s = pressure_surface(storage.frame(index), context, lat, lon,
                                 target, winds=True, level_sigma=level_sigma,
                                 time_label=_time_label(float(times_s[index])))
            valid = s.valid
            source = {"temperature": s["temperature"][None],
                      "wind_speed": np.hypot(s["u"], s["v"])[None],
                      "height": s["geopotential_height"][None]}
            if height:
                source["pressure"] = s["pressure"][None]
            else:
                source["geopotential_height"] = \
                    s["geopotential_height"][None]
            valid_levels = valid[None]
            winds_uv = (s["u"][None], s["v"][None])
            masks.append(s.mask.astype(np.uint8))
            mask_counts.append(s.counts())
        else:
            f = reconstruct_levels(storage.frame(index), context, lat, lon,
                                   chosen_levels, winds=streamlines)
            source = {"temperature": f["temperature"],
                      "pressure": f["pressure"],
                      "zeta": f["zeta"], "delta": f["delta"],
                      "height": f["geopotential_height"]}
            valid_levels = None
            winds_uv = (f["u"], f["v"]) if streamlines else None
        for name, values in source.items():
            for i in range(nl):
                q, lo, hi = _quantize(
                    values[i], None if valid_levels is None
                    else valid_levels[i])
                data[name].append(q)
                bounds[name].append((lo, hi))
        for name in table:
            if valid_levels is None:
                lo = source[name].min(axis=(1, 2))
                hi = source[name].max(axis=(1, 2))
            else:
                lo = np.array([source[name][i][valid_levels[i]].min()
                               for i in range(nl)])
                hi = np.array([source[name][i][valid_levels[i]].max()
                               for i in range(nl)])
            if not stats_init:
                stats[name][:, 0], stats[name][:, 1] = lo, hi
            else:
                stats[name][:, 0] = np.minimum(stats[name][:, 0], lo)
                stats[name][:, 1] = np.maximum(stats[name][:, 1], hi)
        stats_init = True
        if streamlines:
            u_all, v_all = winds_uv
            for i in range(nl):
                ok = None if valid_levels is None else valid_levels[i]
                speed = np.hypot(u_all[i], v_all[i])
                max_speed = max(max_speed, float(
                    speed.max() if ok is None else speed[ok].max()))
                lines.append(integrate_streamlines(
                    u_all[i], v_all[i], lat, lon,
                    seed_count=STREAMLINE_SEEDS,
                    step_deg=STREAMLINE_STEP_DEG,
                    max_steps=STREAMLINE_MAX_STEPS,
                    min_speed=STREAMLINE_MIN_SPEED,
                    relative_min_speed=STREAMLINE_RELATIVE_MIN_SPEED,
                    valid=ok))

    # -- colour limits: fixed per field over all exported times and levels -
    fields = []
    for name, (label, units, factor, scale, policy) in table.items():
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
    title = f"Atmosphere: {context.label}"
    page, preview, elevation, exaggeration = _terrain_page(
        context, title, z_max, exaggeration, centre)
    full = context.full_levels
    k_default = chosen_levels[len(chosen_levels) // 2] if level is None \
        else int(level)
    if k_default not in chosen_levels:
        raise ValueError(f"initial level {level} is not among the exported "
                         f"levels {chosen_levels}")
    g = float(context.gravity)
    if derived:
        return _surface_page(
            page=page, title=title, context=context, target=target,
            level_sigma=level_sigma, res=res, preview=preview, count=count,
            chosen=chosen, times_s=times_s, note=note, lat=lat, lon=lon,
            fields=fields, height=height, masks=masks,
            mask_counts=mask_counts, lines=lines, max_speed=max_speed,
            streamlines=streamlines, elevation=elevation, g=g,
            t_default=t_default, field=field, opacity=opacity,
            estimate=estimate, exaggeration=exaggeration)
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


def _terrain_page(context, title, z_max, exaggeration, centre):
    """The terrain globe every atmosphere page is drawn over: same radius,
    same datum; the default exaggeration from the highest exported Z."""
    elevation = context.surface_elevation_lm
    n = context.l_max + 1
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
    return page, preview, elevation, exaggeration


def _run_details(context, preview, res, count, chosen, g) -> list:
    return [
        ["Run", context.label],
        ["Archive", context.kind],
        ["Saved snapshots", f"{count} (exported {len(chosen)})"],
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


def _interactive_page(*, storage, context, target, level_sigma, res, lat,
                      lon, count, chosen, times_s, note, estimate, t_default,
                      field, opacity, exaggeration, centre, streamlines
                      ) -> tuple[dict, dict]:
    """The interactive-pressure page: native columns per exported saved
    state, interpolated in the browser (module doc)."""
    nlat, nlon = lat.size, lon.size
    nlev = context.nlev
    g = float(context.gravity)
    positions = level_sigma_positions(context, level_sigma)
    decoded = decoded_column_bytes(nlev, nlat, nlon)
    # -- reconstruction (CPU), one saved state at a time; the columns of a
    #    state are quantized and dropped before the next is reconstructed --
    column_times, bounds_all = [], []
    t_lo, t_hi, z_lo, z_hi, speed_hi = math.inf, -math.inf, math.inf, \
        -math.inf, 0.0
    ps_lo, ps_hi, sampled_lo, sampled_hi = math.inf, -math.inf, math.inf, \
        -math.inf
    zs_lo, zs_hi = math.inf, -math.inf         # heights the levels sample
    h_store = None
    for index in chosen:
        cols = pressure_columns(storage.frame(index), context, lat, lon,
                                winds=True, level_sigma=level_sigma)
        quantized, bounds, p_s = quantize_columns(cols)
        if h_store is None:
            h_store = _stored_height(cols.surface_height)
        _check_stored_columns(quantized, bounds, h_store, cols,
                              _time_label(float(times_s[index])))
        entry = {name: {"data": _b64(quantized[name], "<u2"),
                        "bounds": bounds[name]} for name in COLUMN_FIELDS}
        entry["surface_pressure"] = {"data": _b64(p_s, "<f4")}
        column_times.append(entry)
        bounds_all.append(bounds)
        t_lo = min(t_lo, float(cols.temperature.min()))
        t_hi = max(t_hi, float(cols.temperature.max()))
        z_lo = min(z_lo, float(cols.geopotential.min()) / g)
        z_hi = max(z_hi, float(cols.geopotential.max()) / g)
        speed_hi = max(speed_hi, float(np.hypot(cols.u, cols.v).max()))
        ps_lo = min(ps_lo, float(cols.surface_pressure.min()))
        ps_hi = max(ps_hi, float(cols.surface_pressure.max()))
        lo, hi = cols.sampled_pressure_range()
        sampled_lo, sampled_hi = min(sampled_lo, lo), max(sampled_hi, hi)
        lo, hi = cols.sampled_height_range()
        zs_lo, zs_hi = min(zs_lo, lo), max(zs_hi, hi)
        del cols, quantized, p_s
    if target.kind == "pressure" and not (
            sampled_lo <= target.pressure_pa <= sampled_hi):
        raise SurfaceUnavailableError(
            f"the initial {target.label} surface has no valid node at any "
            f"exported saved state: the saved surface pressure spans "
            f"{ps_lo:.6g} to {ps_hi:.6g} Pa and the full levels sample "
            f"{sampled_lo:.6g} to {sampled_hi:.6g} Pa; choose an initial "
            "pressure inside that range (any pressure can be chosen in the "
            "page afterwards)")
    if target.kind == "geopotential_height" and not (
            zs_lo <= target.height_m <= zs_hi):
        raise SurfaceUnavailableError(
            f"the initial {target.label} height surface has no valid node "
            f"at any exported saved state: the full levels sample {zs_lo:.6g} "
            f"to {zs_hi:.6g} m; choose an initial height inside that range "
            "(any height can be chosen in the page afterwards)")
    # each coordinate keeps its own value; the one not chosen starts in the
    # middle of its range (never converted from the other)
    p_initial = (target.pressure_pa if target.kind == "pressure"
                 else math.sqrt(sampled_lo * sampled_hi))
    z_initial = (target.height_m if target.kind == "geopotential_height"
                 else 0.5 * (zs_lo + zs_hi))

    # -- colour limits: conservative bounds from the columns. Interpolation
    #    is linear, so T and Z stay inside their column ranges; the speed of
    #    an interpolated vector is at most the larger bracket speed. --------
    #    On a height surface p = p_s exp(ln sigma*) lies between the first
    #    level position times the lowest p_s and the last times the highest.
    ranges = {"temperature": [t_lo, t_hi],
              "geopotential_height": [z_lo, z_hi],
              "pressure": [positions[0] * ps_lo, positions[-1] * ps_hi],
              "wind_speed": [0.0, speed_hi]}
    tables = {**SURFACE_FIELDS, **HEIGHT_SURFACE_FIELDS}
    fields = []
    for name, kinds in COLUMN_PAGE_FIELDS:
        label, units, factor, scale, policy = tables[name]
        fields.append({
            "id": name, "label": label, "units": units, "factor": factor,
            "scale": scale, "lut": _lut(policy), "colours": policy,
            "range": [float(v) for v in ranges[name]],
            "level_ranges": [[float(v) for v in ranges[name]]],
            "coordinates": list(kinds),
        })

    title = f"Atmosphere: {context.label}"
    page, preview, elevation, exaggeration = _terrain_page(
        context, title, z_hi, exaggeration, centre)
    where = ("the Simmons–Burridge position σʰₖ = σₖ₊½·e^(−αₖ) of each full "
             "level (the pressure at which Φₖ is the hydrostatic geopotential "
             "of an isothermal layer)" if level_sigma == "hydrostatic" else
             "the nominal position σₖ of each full level (the pressure the "
             "native-level page labels it with)")
    interpretation = [
        "Surface: the isobar p₀ or the geopotential-height surface Z₀ chosen "
        "in the page (a coordinate switch, a slider or a typed value; "
        f"starting at the {target.label} "
        f"{'isobar' if target.kind == 'pressure' else 'height surface'}) of "
        "the selected saved state, interpolated in the browser from the "
        "exported native columns. The solver never produced this surface; "
        "its fields are display quantities, not solver state. Nothing is "
        "interpolated in time: a new saved time recomputes the surface from "
        "that state's columns. Each coordinate keeps its own last value; "
        "switching never converts one into the other (neither is one global "
        "value of the other).",
        f"Interpolation: linear in ln σ at the target σ* = p₀/pₛ, with pₛ = "
        f"exp(ln pₛ) of the saved state, between {where}. Temperature, "
        "geopotential and the wind components are interpolated; Z = Φ/g "
        "and the wind speed follow. No extrapolation. The rule is that of "
        "representation.diagnostics.pe_surfaces, the Python reference.",
        "Height surfaces: Φ₀ = g·Z₀ with the model's g and the datum of the "
        "terrain and the native levels (no other gravity law, no "
        "rotational potential). Each column's Φₖ is piecewise linear in "
        "ln σ between the same level positions; it is inverted at Φ₀ "
        "(w = (Φ₀ − Φₖ)/(Φₖ₊₁ − Φₖ), ln σ* between ln σₖ and ln σₖ₊₁), "
        "temperature and both wind components take the same bracket and "
        "weight, p = pₛ·exp(ln σ*). The rule is that of "
        "pe_surfaces.interpolate_geopotential; no extrapolation.",
        "Masks: a node is not a sample where an isobar is under ground "
        "(p₀ > pₛ) or a height surface below the terrain (Z₀ < hₛ, the "
        "physical terrain, not the exaggerated display), above the top "
        "full level, or between the bottom full level and the ground. Such "
        "nodes draw no triangle, are grey on the flat map and say why on "
        "hover; a surface no node samples at the selected time is empty and "
        "says so.",
        f"Surface radius: R + E·Z, Z = Φ(p₀)/g the interpolated geopotential "
        f"height on an isobar (absolute: Φ includes Φs = g·hs once) and the "
        f"constant Z₀ on a height surface, so at a fixed Z₀ the valid nodes "
        f"sit at the same radius at every saved time; g = {g:g} m s⁻², R = "
        f"{context.radius_m / 1000:.6g} km, E the display exaggeration; the "
        "terrain is drawn at R + E·hs with the same E and datum.",
        "Colour ranges: fixed bounds from the exported columns over all "
        "times and levels (temperature, geopotential height and pressure "
        "stay inside their column ranges under linear interpolation; the "
        "interpolated wind speed never exceeds the larger bracket speed), "
        "so colours are comparable while the surface moves.",
        "Streamlines: instantaneous horizontal streamlines of the wind "
        "interpolated to the surface shown, at one saved time, traced in "
        "the page (the integrator of representation.visual."
        "sphere_streamlines, transcribed: deterministic seeds, RK4 on the "
        "sphere, the same stopping rules) and draped on the surface, "
        "clipped to cells whose four nodes are valid, every step checked "
        "along its length. Not parcel trajectories, not 3-D flow; no "
        "vertical motion is computed or implied. While a slider is dragged "
        "the lines are traced once it settles; the previous surface's lines "
        "are removed at once.",
        "Not available in this mode: pressure on an isobar and geopotential "
        "height on a height surface (each constant there), relative "
        "vorticity and divergence on either (a horizontal derivative on a "
        "sigma level is not the derivative on a pressure or height "
        "surface).",
        "Values: exact synthesis of the saved coefficients on a "
        f"{res:g}° display mesh, every full level stored with 16-bit display "
        "quantization per level and saved state (at most (max − min)/131070 "
        "per value), the surface pressure and the terrain height as "
        "float32; the browser interpolates in double precision and stores "
        "the results in single precision.",
    ]
    details = _run_details(context, preview, res, count, chosen, g)
    details[3:3] = [
        ["Surface", f"interactive isobar or geopotential-height surface, "
                    f"initially the {target.label} "
                    + ("isobar" if target.kind == "pressure" else
                       "height surface")
                    + f" ({target.value:g} {target.units})"],
        ["Level positions", f"{level_sigma}: σ = "
                            + ", ".join(f"{s:.6g}" for s in positions)],
        ["Native levels", f"{nlev} sigma full levels, all exported as "
                          "columns"],
    ]
    details += [
        ["Sampled pressures", f"{sampled_lo:.6g} to {sampled_hi:.6g} Pa over "
                              "the exported states (surface pressure "
                              f"{ps_lo:.6g} to {ps_hi:.6g} Pa)"],
        ["Sampled heights", f"{zs_lo:.6g} to {zs_hi:.6g} m (lowest bottom "
                            "to highest top full level)"],
        ["Embedded data", f"about {estimate / 1e6:.3g} MB (estimated before "
                          "reconstruction)"],
        ["Decoded memory", f"{decoded / 1e6:.3g} MB per saved state "
                           f"(at most {MAX_DECODED_TIMES} held) and at most "
                           f"{streamline_cache_bytes() / 1e6:.2g} MB of "
                           "cached streamlines"],
    ]
    if note:
        details.append(["Time selection", note])
    page["title"] = title
    page["subtitle"] = (f"Primitive-equation run · any isobar or "
                        f"geopotential-height surface (interpolated in the "
                        f"browser) · {len(chosen)} saved time(s)")
    page["details"] = details
    page["atmosphere"] = {
        "version": 4,
        "mesh": {"nlat": nlat, "nlon": nlon, "lat0": float(lat[0]),
                 "lon0": float(lon[0]), "dlat": res, "dlon": res},
        "times": [{"index": int(i), "time_s": float(times_s[i]),
                   "label": _time_label(float(times_s[i]))} for i in chosen],
        "saved_count": count,
        "selection_note": note,
        "levels": [{"index": None, "number": None, "sigma": None,
                    "label": "derived surface (interactive)"}],
        "nlev": nlev,
        "surface": {
            "kind": target.kind, "value": target.value,
            "units": target.units, "label": target.label,
            "level_sigma": level_sigma,
            "level_positions": [float(s) for s in positions],
            "mask_labels": {str(k): v for k, v in MASK_LABELS.items()},
            "interactive": True,
            "pressure_range": [sampled_lo, sampled_hi],
            "surface_pressure_range": [ps_lo, ps_hi],
            "end_tolerance": END_TOLERANCE,
            "coordinates": {
                "pressure": {
                    "units": "Pa", "range": [sampled_lo, sampled_hi],
                    "initial": p_initial,
                    "mask_labels": {str(k): v
                                    for k, v in MASK_LABELS.items()}},
                "geopotential_height": {
                    "units": "m", "range": [zs_lo, zs_hi],
                    "initial": z_initial,
                    "mask_labels": {
                        **{str(k): v for k, v in HEIGHT_MASK_LABELS.items()},
                        str(MASK_INVALID_STORED): _INVALID_STORED_LABEL}},
            },
            "surface_height_range": (
                None if h_store is None else
                [float(h_store.min()), float(h_store.max())]),
        },
        "columns": {
            "fields": list(COLUMN_FIELDS), "dtype": "uint16",
            "surface_pressure_dtype": "float32", "nlev": nlev,
            "decoded_bytes": decoded, "max_decoded_times": MAX_DECODED_TIMES,
            "times": column_times,
            # the terrain height, once, float32, for the height masks (null
            # for a flat lower boundary: h_s = 0)
            "surface_height": (None if h_store is None else
                               {"data": _b64(h_store.ravel(), "<f4"),
                                "dtype": "float32"}),
        },
        "gravity": g, "radius_m": context.radius_m,
        "terrain_flat": elevation is None,
        "fields": fields,
        "streamlines": {
            "enabled": streamlines, "mode": "browser",
            "seed_count": STREAMLINE_SEEDS, "step_deg": STREAMLINE_STEP_DEG,
            "max_steps": STREAMLINE_MAX_STEPS,
            "min_speed": STREAMLINE_MIN_SPEED,
            "relative_min_speed": STREAMLINE_RELATIVE_MIN_SPEED,
            "settle_ms": STREAMLINE_SETTLE_MS,
            "cache_size": STREAMLINE_CACHE, "max_speed": speed_hi},
        "defaults": {"time": t_default, "level": 0, "field": field,
                     "opacity": float(opacity), "streamlines": streamlines,
                     "coordinate": target.kind, "pressure_pa": p_initial,
                     "height_m": z_initial},
        "interpretation": interpretation,
    }
    report = {"times": chosen, "levels": [], "surface": target,
              "interactive_pressure": True, "interactive_surface": True,
              "level_sigma": level_sigma,
              "resolution_deg": res, "estimate_bytes": estimate,
              "decoded_bytes_per_time": decoded,
              "pressure_range": (sampled_lo, sampled_hi),
              "height_range": (zs_lo, zs_hi),
              "bounds": bounds_all, "selection_note": note,
              "context": context, "exaggeration": float(exaggeration)}
    return page, report


def _check_stored_columns(quantized, bounds, h_store, cols, label) -> None:
    """Refuse to export stored columns the page could not invert: the 16-bit
    geopotential must still decrease strictly downward and its lowest level
    lie above the float32 terrain, as the exact columns do. Quantization
    errors are far below layer thicknesses, so this never fires on a real
    run; if it did, the file would otherwise divide by an inverted
    interval."""
    shape = cols.surface_pressure.shape
    stored = np.empty((cols.nlev, *shape), dtype=np.float64)
    for k, (lo, hi) in enumerate(bounds["geopotential"]):
        # the page's arithmetic (dequantize_columns)
        step = float(np.float32((hi - lo) / 65535.0))
        stored[k] = (float(np.float32(lo)) + np.asarray(
            quantized["geopotential"][k], dtype=np.float64) * step).astype(
                np.float32).reshape(shape)
    phi_s = (np.zeros(shape) if h_store is None
             else cols.gravity * h_store)
    bad = ~(np.all(stored[:-1] > stored[1:], axis=0) & (stored[-1] > phi_s))
    if bad.any():
        raise ValueError(
            f"the 16-bit stored geopotential at {label} would not decrease "
            f"strictly downward above the terrain at {int(bad.sum())} display "
            "nodes, so the page could not invert it; nothing is exported "
            "(choose another resolution_deg or fewer levels)")


def _surface_page(*, page, title, context, target, level_sigma, res, preview,
                  count, chosen, times_s, note, lat, lon, fields, height,
                  masks, mask_counts, lines, max_speed, streamlines,
                  elevation, g, t_default, field, opacity, estimate,
                  exaggeration) -> tuple[dict, dict]:
    """Finish the page for a derived surface (one slice per time)."""
    nlat, nlon = lat.size, lon.size
    isobar = target.kind == "pressure"
    noun = "isobar" if isobar else "geopotential-height surface"
    labels = MASK_LABELS if isobar else HEIGHT_MASK_LABELS
    positions = level_sigma_positions(context, level_sigma)
    total = sum(c[MASK_VALID] for c in mask_counts)
    nodes = nlat * nlon * len(chosen)
    where = ("the Simmons–Burridge position σʰₖ = σₖ₊½·e^(−αₖ) of each full "
             "level (the pressure at which Φₖ is the hydrostatic geopotential "
             "of an isothermal layer)" if level_sigma == "hydrostatic" else
             "the nominal position σₖ of each full level (the pressure the "
             "native-level page labels it with)")
    what = (f"p₀ = {target.pressure_pa:g} Pa" if isobar else
            f"Z₀ = {target.height_m:g} m, Φ₀ = g·Z₀")
    interpretation = [
        f"Surface: the {target.label} {noun} ({what}) of each saved state, "
        "interpolated from the reconstructed native column. The solver "
        "never produced this surface; its fields are display quantities, "
        "not solver state.",
        (f"Interpolation: linear in ln σ at the target σ* = p₀/pₛ, with pₛ = "
         f"exp(ln pₛ) of the saved state, between {where}. Temperature, "
         "geopotential and the wind components are interpolated; Z = Φ/g "
         "and the wind speed follow. No extrapolation." if isobar else
         f"Interpolation: each column's Φ, piecewise linear in ln σ between "
         f"{where}, is inverted at Φ₀ (the model's g; the datum of the "
         "terrain and the native levels); temperature and the wind "
         "components take the same bracket and weight, p = pₛ·exp(ln σ*) "
         "and the wind speed follow. No extrapolation."),
        ("Masks: a node is not a sample where the isobar is under ground "
         "(p₀ > pₛ), above the top full level or between the bottom full "
         "level and the ground." if isobar else
         "Masks: a node is not a sample where the surface is below the "
         "terrain (Z₀ < hₛ, the physical terrain, not the exaggerated "
         "display), above the highest sampled full level or between the "
         "terrain and the lowest full level.")
        + " Such nodes draw no triangle, are grey on the flat map and say "
        "why on hover; colour ranges and streamlines use valid nodes only. "
        f"{total} of {nodes} exported nodes are valid.",
        (f"Surface radius: R + E·Z, Z = Φ(p₀)/g the interpolated geopotential "
         f"height (absolute: Φ includes Φs = g·hs once)" if isobar else
         "Surface radius: R + E·Z₀ at every valid node and saved time")
        + f", g = {g:g} m s⁻², R = "
        f"{context.radius_m / 1000:.6g} km, E the display exaggeration; the "
        "terrain is drawn at R + E·hs with the same E and datum.",
        ("Not exported: pressure (constant on this surface), relative "
         if isobar else "Not exported: geopotential height (constant on "
         "this surface), relative ")
        + "vorticity and divergence (a horizontal derivative on a sigma "
        "level is not the derivative on a derived surface; a "
        "target-coordinate derivative is a later slice of issue #13).",
        "Streamlines: instantaneous horizontal streamlines of the "
        "interpolated wind on this surface at one saved time, clipped to "
        "cells whose four nodes are valid. Not parcel trajectories; no "
        "vertical motion is computed or implied.",
        "Values: exact synthesis of the saved coefficients on a "
        f"{res:g}° display mesh, then vertical interpolation, stored with "
        "16-bit display quantization per time over the valid nodes.",
    ]
    details = [
        ["Run", context.label],
        ["Archive", context.kind],
        ["Saved snapshots", f"{count} (exported {len(chosen)})"],
        ["Surface", f"{target.label} {noun} ({target.value:g} "
                    f"{target.units}), interpolated"],
        ["Level positions", f"{level_sigma}: σ = "
                            + ", ".join(f"{s:.6g}" for s in positions)],
        ["Native levels", f"{context.nlev} sigma full levels (source of "
                          "the interpolation, not exported)"],
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
        ["Valid nodes", f"{total} of {nodes} over the exported times"],
    ]
    if note:
        details.append(["Time selection", note])
    page["title"] = title
    page["subtitle"] = (f"Primitive-equation run · {target.label} "
                        + ("isobaric surface" if isobar else
                           "geopotential-height surface")
                        + f" (interpolated) · {len(chosen)} saved time(s)")
    page["details"] = details
    page["atmosphere"] = {
        "version": 2,
        "mesh": {"nlat": nlat, "nlon": nlon, "lat0": float(lat[0]),
                 "lon0": float(lon[0]), "dlat": res, "dlon": res},
        "times": [{"index": int(i), "time_s": float(times_s[i]),
                   "label": _time_label(float(times_s[i]))} for i in chosen],
        "saved_count": count,
        "selection_note": note,
        "levels": [{"index": None, "number": None, "sigma": None,
                    "label": target.label}],
        "nlev": context.nlev,
        "surface": {
            "kind": target.kind, "value": target.value,
            "units": target.units,
            "label": target.label, "level_sigma": level_sigma,
            "level_positions": [float(s) for s in positions],
            "mask_labels": {str(k): v for k, v in labels.items()},
            "valid_counts": [{str(k): v for k, v in c.items()}
                             for c in mask_counts],
        },
        "mask": {"data": _b64(np.concatenate([m.ravel() for m in masks]),
                              "u1")},
        "gravity": g, "radius_m": context.radius_m,
        "terrain_flat": elevation is None,
        "fields": fields,
        "height": height,
        "streamlines": (dict(_streamline_block(lines), enabled=True,
                             max_speed=max_speed,
                             min_speed=STREAMLINE_MIN_SPEED)
                        if streamlines else {"enabled": False}),
        "defaults": {"time": t_default, "level": 0,
                     "field": field, "opacity": float(opacity),
                     "streamlines": bool(streamlines)},
        "interpretation": interpretation,
    }
    report = {"times": chosen, "levels": [], "surface": target,
              "level_sigma": level_sigma, "mask_counts": mask_counts,
              "resolution_deg": res, "estimate_bytes": estimate,
              "selection_note": note, "context": context,
              "exaggeration": float(exaggeration)}
    return page, report


def _default_path(storage, report) -> pathlib.Path:
    label = report["context"].label
    base = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()[:48]
    surface = report.get("surface")
    interactive = bool(report.get("interactive_pressure"))
    # pressure keeps its earlier key, so existing default names are stable
    value = (None if surface is None else surface.pressure_pa
             if surface.kind == "pressure" else (surface.kind, surface.value))
    key = repr((str(pathlib.Path(storage.run_dir).resolve()), report["times"],
                report["levels"], report["resolution_deg"], value,
                interactive)).encode()
    tag = hashlib.sha256(key).hexdigest()[:8]
    isobar = surface is not None and surface.kind == "pressure"
    kind = ("atmosphere" if surface is None else
            ("isobars" if isobar else "heights") if interactive else
            "isobar" if isobar else "height")
    return pathlib.Path(tempfile.gettempdir()) / "tropoi" / (
        f"{base or 'run'}-{kind}-{tag}.html")


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
