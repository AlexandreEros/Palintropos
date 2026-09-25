"""Compose view objects (``views``) into declarative figures for saved runs.

``render_view`` is the implementation behind ``Simulation.plot``,
``Snapshot.plot(view=...)`` and ``tropoi plot``. It evaluates every field
through one :class:`~tropoi.representation.visual.evaluate.RunFields` (one
model build, one synthesis per field), builds a ``FigureSpec`` from the
layered-map, key and line panels, freezes shared scales with
``resolve_figure_normalizations`` and hands the result to the renderer.

Rules kept here:

* maps of one quantity across times share one colour scale, and their
  vectors share one speed scale; the keys show exactly those scales;
* statistics come from the state grid with its quadrature weights;
* per-step diagnostics are lines, values known only at saved states are
  unconnected markers, and each label says which;
* ``"auto"`` parts of a view are omitted (and the omission recorded) when a
  run cannot provide them; explicitly requested parts raise instead;
* nothing is ever written inside the run directory.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import pathlib

import numpy as np

from tropoi.representation.visual.evaluate import REST_SPEED_MS, RunFields
from tropoi.representation.visual.normalization import NormalizationPolicy
from tropoi.representation.visual.quantities import (
    QuantityUnavailableError, quantity)
from tropoi.representation.visual.specs import (
    ColorKeySpec, ContourLayer, FigureSpec, LayeredMapSpec, LinePanelSpec,
    LineSeriesSpec, LineWidthKeySpec, PanelPlacement, ScalarLayer,
    TextPanelSpec, VectorLayer)
from tropoi.representation.visual.timeline import (
    resolve_figure_normalizations, select_representative_frame_indices)
from tropoi.representation.visual.views import (
    Arrows, Complexity, Contours, Drift, Grid, Map, Overview, Sigma,
    Streamlines, Style, describe, parse_time)

__all__ = ["compose_view", "default_view", "render_view"]

#: How each quantity is drawn when a Map does not say otherwise:
#: (colour policy, symmetric about zero).
DISPLAY_DEFAULTS = {
    "vorticity": ("RdBu_r", True),
    "divergence": ("RdBu_r", True),
    "streamfunction": ("PuOr_r", True),
    "velocity_potential": ("PuOr_r", True),
    "temperature_anomaly": ("RdBu_r", True),
    "surface_pressure_anomaly": ("RdBu_r", True),
    "free_surface_height": ("cividis:0.35:1.0", False),
    "layer_depth": ("cividis:0.35:1.0", False),
    "wind_speed": ("YlGnBu:0.0:0.75", False),
    "temperature": ("inferno:0.25:1.0", False),
    "terrain": ("YlOrBr:0.0:0.85", False),
}

_SCENARIO_TITLES = {
    "williamson5": "Williamson test case 5: flow over an isolated mountain",
    "williamson2": "Williamson test case 2: steady zonal flow",
    "rh4": "Rossby–Haurwitz wave 4",
    "two_vortices": "Two opposite-signed vortices",
    "gravity_wave": "Linear gravity wave",
    "thermal_wave": "Thermal wave",
    "orographic_isothermal_rest": "Isothermal atmosphere at rest over terrain",
}
_SOLVER_NAMES = {"bve": "barotropic vorticity core",
                 "swe": "shallow-water core",
                 "pe": "dry primitive-equation core"}
_MEASURE_LABELS = {
    "mean_degree": "mean degree ⟨ℓ⟩",
    "mean_order": "mean zonal wavenumber ⟨|m|⟩",
    "effective_modes": "effective modes N = exp(S)",
    "entropy": "Shannon entropy S",
}
_RECORDED_LABELS = {
    "total_mass": "layer mass", "total_energy": "total energy",
    "energy": "kinetic energy", "enstrophy_abs": "absolute enstrophy",
    "potential_enstrophy": "potential enstrophy",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def pretty_units(units: str) -> str:
    table = {"^-1": "⁻¹", "^-2": "⁻²", "^2": "²", "^3": "³", "^4": "⁴"}
    for plain, fancy in table.items():
        units = units.replace(plain, fancy)
    return units


def _time_axis(times: np.ndarray) -> tuple[float, str]:
    """(divisor, unit name) for labelling a run's time axis."""
    span = float(times[-1]) if times.size else 0.0
    if span >= 2.0 * 86400.0:
        return 86400.0, "day"
    if span >= 2.0 * 3600.0:
        return 3600.0, "h"
    return 1.0, "s"


def _time_label(seconds: float, divisor: float, unit: str) -> str:
    value = seconds / divisor
    text = f"{value:.4g}"
    return f"day {text}" if unit == "day" else f"t = {text} {unit}"


def _level_label(fields: RunFields, level: int | None) -> str:
    if level is None:
        return ""
    sigma = fields.sigma_levels[level]
    return f"σ = {sigma:.3f} (level {level + 1} of {fields.nlev})"


def resolve_level(fields: RunFields, level) -> int | None:
    if level is None:
        return None
    if fields.solver != "pe":
        raise QuantityUnavailableError(
            f"{fields.solver.upper()} runs have no vertical levels")
    if isinstance(level, Sigma):
        sigma = np.asarray(fields.sigma_levels, dtype=np.float64)
        return int(np.argmin(np.abs(sigma - float(level.value))))
    index = int(level)
    if not 0 <= index < fields.nlev:
        raise QuantityUnavailableError(
            f"level {index} is outside 0..{fields.nlev - 1} (top to bottom)")
    return index


def _level_for(fields: RunFields, identifier: str, level: int | None):
    """The map's level if the quantity is per-level in this run."""
    entry = quantity(identifier)
    if fields.solver == "pe" and entry.per_level_in_pe:
        if level is None:
            raise QuantityUnavailableError(
                f"{identifier!r} needs a level in PE runs (level=... or "
                "Sigma(...))")
        return level
    return None


def resolve_snapshots(fields: RunFields, selection, max_maps: int
                      ) -> list[int]:
    times = fields.times
    count = times.size
    if count == 0:
        return []
    if selection is None:
        if count <= max_maps:
            return list(range(count))
        if max_maps == 1:
            return [count - 1]
        return list(select_representative_frame_indices(
            times, max_frames=max_maps))
    indices = []
    for item in selection:
        if isinstance(item, str):
            target = parse_time(item)
            matches = np.flatnonzero(np.isclose(times, target, rtol=0.0,
                                                atol=1e-6))
            if not matches.size:
                divisor, unit = _time_axis(times)
                stored = ", ".join(f"{t / divisor:g}" for t in times)
                raise QuantityUnavailableError(
                    f"no saved state at {item!r}; saved times ({unit}): "
                    f"{stored}. Saved runs are never interpolated in time.")
            indices.append(int(matches[0]))
        else:
            index = int(item)
            if not -count <= index < count:
                raise IndexError(f"snapshot {index} is outside the "
                                 f"{count} saved state(s)")
            indices.append(index % count)
    return indices


def _normalization(entry_id: str, map_view: Map) -> NormalizationPolicy:
    if map_view.limits is not None:
        return NormalizationPolicy.fixed(*map_view.limits)
    symmetric = map_view.symmetric
    if symmetric is None:
        symmetric = DISPLAY_DEFAULTS.get(entry_id, ("viridis", False))[1]
    return (NormalizationPolicy.symmetric() if symmetric
            else NormalizationPolicy.automatic())


def _color_policy(entry_id: str, map_view: Map) -> str:
    if map_view.color_policy is not None:
        return map_view.color_policy
    return DISPLAY_DEFAULTS.get(entry_id, ("viridis", False))[0]


def _nice_speeds(vmax: float) -> tuple[float, ...]:
    """Three round speeds up to ``vmax`` for a line-width key."""
    if not vmax > 0.0:
        return (1.0,)
    exponent = math.floor(math.log10(vmax))
    step = 10.0 ** exponent
    # The largest of 1, 2, 4, 5 x 10^k not above vmax.
    top = max(factor * step for factor in (1.0, 2.0, 4.0, 5.0)
              if factor * step <= vmax)
    values = sorted({top / 4.0, top / 2.0, top})
    return tuple(float(f"{value:.3g}") for value in values)


# ---------------------------------------------------------------------------
# panel builders
# ---------------------------------------------------------------------------

def _map_panel(fields: RunFields, view: Map, index: int | None,
               level: int | None, *, title: str, groups: dict | None,
               colorbar: bool, label_longitudes: bool = True,
               label_latitudes: bool = True) -> LayeredMapSpec:
    background, corner = None, None
    if view.background is not None:
        entry = quantity(view.background)
        bg_level = _level_for(fields, view.background, level)
        field_index = None if entry.cadence == "static" else index
        grid_field = fields.scalar_field(view.background, field_index,
                                         bg_level)
        label = f"{entry.long_name} ({pretty_units(entry.units)})"
        background = ScalarLayer(
            grid_field, normalization=_normalization(entry.id, view),
            normalization_group=None if groups is None else groups[
                "background"],
            color_policy=_color_policy(entry.id, view), label=label)
    contours = []
    for contour in view.contours:
        entry = quantity(contour.quantity)
        c_level = _level_for(fields, contour.quantity, level)
        c_index = None if entry.cadence == "static" else index
        contours.append(ContourLayer(
            fields.scalar_field(contour.quantity, c_index, c_level),
            tuple(contour.levels), color=contour.color,
            line_width=contour.line_width, line_style=contour.line_style,
            label=f"{entry.long_name} at "
                  f"{', '.join(f'{v:g}' for v in contour.levels)} "
                  f"{pretty_units(entry.units)}"))
    vectors = None
    if view.vectors is not None:
        if view.vectors.vector != "wind":
            raise QuantityUnavailableError(
                f"unknown vector quantity {view.vectors.vector!r}; "
                "available: wind")
        if index is None:
            raise QuantityUnavailableError("a static map cannot carry winds")
        wind_level = _level_for(fields, "wind", level)
        state_u, state_v = fields.state_wind(index, wind_level)
        at_rest = float(np.hypot(state_u, state_v).max()) < REST_SPEED_MS
        wind = fields.display_wind(index, wind_level)
        common = dict(
            latitudes=wind.latitudes, longitudes=wind.longitudes,
            zonal=wind.zonal, meridional=wind.meridional, radius=wind.radius,
            color_by=view.vectors.color_by, color=view.vectors.color,
            normalization_group=None if groups is None else groups["speed"],
            color_policy="YlGnBu:0.25:1.0")
        if isinstance(view.vectors, Streamlines):
            vectors = VectorLayer(
                style="streamlines", width_by=view.vectors.width_by,
                line_width_range=view.vectors.line_width_range,
                density=view.vectors.density,
                arrow_size=view.vectors.arrow_size,
                max_length=view.vectors.max_length,
                seed_count=view.vectors.seed_count, **common)
        else:
            vectors = VectorLayer(style="arrows", width_by=None,
                                  arrow_stride=view.vectors.stride, **common)
        if at_rest:
            # Roundoff-level winds have no direction worth drawing.
            vectors = None
            corner = f"wind below {REST_SPEED_MS:g} m/s: at rest, not drawn"
    if background is None and not contours and vectors is None:
        # Keep an empty, labelled map rather than failing on a rest state.
        background = ScalarLayer(
            fields.scalar_field("wind_speed", index,
                                _level_for(fields, "wind_speed", level)),
            color_policy="Greys:0.0:0.05", label="wind speed (m s⁻¹)")
    return LayeredMapSpec(
        title=title, background=background, contours=tuple(contours),
        vectors=vectors, colorbar=colorbar,
        label_longitudes=label_longitudes, label_latitudes=label_latitudes,
        corner_text=corner)


def _drift_panel(fields: RunFields, drift: Drift, divisor: float,
                 unit: str, sidecar: dict) -> LinePanelSpec:
    series, notes, record = [], [], {}
    colors = iter(("#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad", "#d35400"))
    largest = 0.0
    for column in drift.recorded:
        data = fields.recorded(column)
        relative = data.values / data.values[0] - 1.0
        largest = max(largest, float(np.max(np.abs(relative))))
        name = _RECORDED_LABELS.get(column, column)
        exact = "" if np.any(relative) else ", exactly 0"
        series.append(LineSeriesSpec(
            data.times_seconds / divisor, relative,
            f"{name} (every step{exact})", color=next(colors),
            line_width=1.0))
        at_saved, missing = fields.recorded_at_snapshots(column)
        record[column] = {
            "cadence": "every step",
            "relative_drift_at_saved_states": [
                None if not np.isfinite(v) else float(v / at_saved[0] - 1.0)
                for v in at_saved],
            "saved_states_without_exact_row": missing}
    for name_id in drift.at_snapshots:
        if name_id != "potential_enstrophy":
            raise QuantityUnavailableError(
                f"{name_id!r} is not a saved-state diagnostic; available: "
                "potential_enstrophy")
        values = np.array([fields.potential_enstrophy(i)
                           for i in range(fields.times.size)])
        relative = values / values[0] - 1.0
        largest = max(largest, float(np.max(np.abs(relative))))
        name = _RECORDED_LABELS.get(name_id, name_id)
        series.append(LineSeriesSpec(
            fields.times / divisor, relative,
            f"{name} ({values.size} saved states)", color=next(colors),
            marker="o", marker_size=4.0, draw_line=False))
        record[name_id] = {"cadence": "saved states",
                           "values": values.tolist(),
                           "relative_drift": relative.tolist()}
    if not series:
        raise ValueError("a drift panel needs at least one quantity")
    threshold = drift.linear_threshold
    if threshold is None and largest > 0.0:
        threshold = 10.0 ** (math.floor(math.log10(largest)) - 3)
    sidecar["drift"] = record
    return LinePanelSpec(
        tuple(series), "Conservation", f"time ({unit})",
        "relative drift from t = 0",
        y_scale="symlog" if threshold else "linear",
        y_linear_threshold=threshold, notes=tuple(notes),
        legend_location="below")


def _complexity_panel(fields: RunFields, complexity: Complexity,
                      level: int | None, divisor: float, unit: str,
                      sidecar: dict) -> LinePanelSpec | TextPanelSpec:
    from tropoi.representation.diagnostics.spectral import (
        ZeroKineticEnergyError)
    times, measured, undefined = [], [], []
    for index, seconds in enumerate(fields.times):
        try:
            measured.append(fields.spectral_complexity(index, level))
            times.append(seconds / divisor)
        except ZeroKineticEnergyError:
            undefined.append(float(seconds))
    colors = iter(("#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad"))
    markers = iter(("o", "s", "^", "D"))
    series = tuple(
        LineSeriesSpec(np.array(times),
                       np.array([getattr(m, name) for m in measured]),
                       _MEASURE_LABELS[name], color=next(colors),
                       marker=next(markers), marker_size=4.0,
                       draw_line=False)
        for name in complexity.measures)
    notes = []
    if undefined:
        notes.append("undefined (at rest) at t = " + ", ".join(
            f"{t / divisor:.4g}" for t in undefined) + f" {unit}")
    sidecar["spectral_complexity"] = {
        "cadence": "saved states", "level": level,
        "times_s": [float(t) * divisor for t in times],
        **{name: [getattr(m, name) for m in measured]
           for name in ("mean_degree", "mean_order", "entropy",
                        "effective_modes")},
        "undefined_at_s": undefined}
    if not measured:
        return TextPanelSpec(
            "Kinetic-energy spectral complexity:\nundefined at every saved "
            f"state (at rest:\nrms wind below {REST_SPEED_MS:g} m/s)",
            font_family="sans-serif", font_size=8.0, color="#444444")
    title = "Kinetic-energy spectral complexity"
    if level is not None:
        title = f"KE spectral complexity, σ = {fields.sigma_levels[level]:.3f}"
    return LinePanelSpec(
        series, title, f"time ({unit})",
        f"dimensionless ({len(measured)} saved states)",
        notes=tuple(notes), legend_location="below")


# ---------------------------------------------------------------------------
# default views
# ---------------------------------------------------------------------------

def default_view(fields: RunFields) -> Overview:
    """The solver's default overview for this run."""
    solver = fields.solver
    if solver == "bve":
        return Overview(map=Map("vorticity", vectors=Streamlines()))
    if solver == "swe":
        contours = ((Contours("terrain", (500.0, 1000.0, 1500.0)),)
                    if fields.has_terrain else ())
        return Overview(map=Map("free_surface_height", contours=contours,
                                vectors=Streamlines()))
    from tropoi.representation.visual.pe_snapshots import (
        select_snapshot_levels)
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    interfaces = fields.run_config.get("sigma_interfaces")
    sigma = (SigmaGrid.uniform(fields.nlev) if interfaces is None
             else SigmaGrid(tuple(float(s) for s in interfaces)))
    level = select_snapshot_levels(sigma).lower_index
    return Overview(map=Map("temperature_anomaly", level=level,
                            vectors=Streamlines()))


def _auto_diagnostics(fields: RunFields, level, omitted: list):
    solver = fields.solver
    panels = []
    recorded = {"bve": ("energy", "enstrophy_abs"),
                "swe": ("total_mass", "total_energy"),
                "pe": ("total_mass",)}[solver]
    if fields.diagnostics_path.is_file():
        panels.append(Drift(recorded=recorded,
                            at_snapshots=("potential_enstrophy",)
                            if solver == "swe" and fields.times.size else ()))
    else:
        omitted.append("conservation panel: diagnostics/timeseries.csv "
                       "not found")
    if fields.times.size:
        panels.append(Complexity(level=level))
    return tuple(panels)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def _header(fields: RunFields, title: str | None) -> tuple[str, str]:
    config = fields.run_config
    meta = fields.storage.metadata
    scenario = str(config.get("scenario") or fields.solver)
    heading = title or (
        f"{_SCENARIO_TITLES.get(scenario, scenario)} · "
        f"{_SOLVER_NAMES[fields.solver]}")
    if config.get("grid") == "latlon":
        grid = f"Gauss–Legendre {config.get('nlat')} × {config.get('nlon')}"
    else:
        grid = f"geodesic grid, resolution {config.get('resolution')}"
    git = meta.get("git") or {}
    commit = str(git.get("commit") or "unknown")[:8]
    if git.get("dirty"):
        commit += " (uncommitted changes)"
    parts = [f"{grid}, ℓ ≤ {config.get('lmax')}"]
    if fields.solver == "pe":
        parts.append(f"{fields.nlev} sigma levels")
    parts.append(f"commit {commit}")
    return heading, " · ".join(parts) + chr(10) + (
        f"run {meta.get('run_id') or '?'}")


def _sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_view(fields: RunFields, view, *, snapshot: int | None = None
                 ) -> tuple[FigureSpec, dict]:
    """Build the figure for ``view`` and the record of every number shown."""
    if isinstance(view, Map):
        if snapshot is None:
            view = Overview(map=view)
        else:
            view = Grid(((view,),), snapshot=snapshot, title=None)
    elif isinstance(view, Grid) and snapshot is not None:
        view = replace(view, snapshot=snapshot)
    elif isinstance(view, Overview) and snapshot is not None:
        view = replace(view, snapshots=(snapshot,))
    if isinstance(view, Grid):
        return _compose_grid(fields, view)
    if not isinstance(view, Overview):
        raise TypeError(f"cannot plot a {type(view).__name__}; use Map, "
                        "Overview or Grid")
    return _compose_overview(fields, view)


def _record_base(fields: RunFields, view) -> dict:
    storage = fields.storage
    return {
        "view": describe(view),
        "run": {"run_id": storage.metadata.get("run_id"),
                "solver": fields.solver,
                "git": storage.metadata.get("git"),
                "coefficients_sha256": _sha256(
                    pathlib.Path(storage.coefficient_path)),
                "diagnostics_sha256": _sha256(fields.diagnostics_path)},
        "omitted": [],
    }


def _snapshot_statistics(fields: RunFields, view_map: Map, index: int,
                         level: int | None) -> dict:
    stats = {"index": int(index), "time_s": float(fields.times[index])}
    if view_map.background is not None:
        entry = quantity(view_map.background)
        bg_level = _level_for(fields, entry.id, level)
        values = (fields._terrain_state() if entry.cadence == "static" else
                  fields.state_values(entry.id, index, bg_level))
        stats[entry.id] = {"min": float(values.min()),
                           "max": float(values.max()),
                           "area_mean": float(np.sum(
                               fields.state_weights() * values)),
                           "units": entry.units}
    if view_map.vectors is not None:
        u, v = fields.state_wind(index, _level_for(fields, "wind", level))
        speed = np.hypot(u, v)
        stats["wind_speed"] = {"max": float(speed.max()),
                               "area_mean": float(np.sum(
                                   fields.state_weights() * speed)),
                               "units": "m s^-1"}
    return stats


def _compose_overview(fields: RunFields, view: Overview
                      ) -> tuple[FigureSpec, dict]:
    style: Style = view.style
    record = _record_base(fields, view)
    level = resolve_level(fields, view.map.level)
    indices = resolve_snapshots(fields, view.snapshots, view.max_maps)
    divisor, unit = _time_axis(fields.times)

    static = view.static
    if static == "auto":
        static = Map("terrain") if fields.has_terrain else None
    diagnostics = view.diagnostics
    if diagnostics == "auto":
        diagnostics = _auto_diagnostics(fields, level, record["omitted"])

    columns = max(1, style.map_columns)
    map_rows = math.ceil(len(indices) / columns) if indices else 0
    panels: list[PanelPlacement] = []
    heights: list[float] = []

    heading, provenance = _header(fields, view.title)
    panels.append(PanelPlacement(TextPanelSpec(
        heading, font_family="sans-serif", font_size=style.base_font_size + 2,
        horizontal_alignment="left", font_weight="semibold"), 0, 0,
        column_span=columns))
    panels.append(PanelPlacement(TextPanelSpec(
        provenance, font_family="sans-serif",
        font_size=style.base_font_size - 1, horizontal_alignment="left",
        color="#555555"), 1, 0, column_span=columns))
    heights += [0.26, 0.34]
    row = 2

    groups = {"background": "overview-background", "speed": "overview-speed"}
    keys = []
    if indices and view.map.background is not None:
        entry = quantity(view.map.background)
        label = f"{entry.long_name} ({pretty_units(entry.units)})"
        bg_level = _level_for(fields, entry.id, level)
        if bg_level is not None:
            label += f", {_level_label(fields, bg_level)}"
        magnitudes = [float(np.max(np.abs(fields.state_values(
            entry.id, i, bg_level)))) for i in indices]
        if max(magnitudes) == 0.0:
            label += "\n(identically zero in every map)"
        elif max(magnitudes) <= 1e-12 and _normalization(
                entry.id, view.map).kind.value == "symmetric":
            label += (f"\n(every |value| ≤ {max(magnitudes):.1e}; "
                      "scale shown ±1)")
        keys.append(ColorKeySpec(
            groups["background"], label,
            color_policy=_color_policy(entry.id, view.map),
            normalization=_normalization(entry.id, view.map)))
    vectors = view.map.vectors
    speed_scale = None
    if indices and vectors is not None:
        speeds = [np.hypot(*fields.state_wind(
            i, _level_for(fields, "wind", level))).max() for i in indices]
        speed_scale = float(max(speeds))
        if speed_scale < REST_SPEED_MS:
            record["omitted"].append(
                f"speed key: every wind is below {REST_SPEED_MS:g} m/s")
        elif vectors.color_by == "speed":
            keys.append(ColorKeySpec(groups["speed"],
                                     "wind speed (m s⁻¹)",
                                     color_policy="YlGnBu:0.25:1.0"))
        elif isinstance(vectors, Streamlines) and vectors.width_by == "speed":
            keys.append(LineWidthKeySpec(
                groups["speed"],
                f"streamline width: wind speed (m s⁻¹), "
                f"max {speed_scale:.3g}",
                _nice_speeds(speed_scale),
                line_width_range=vectors.line_width_range,
                color=vectors.color))

    if static is not None:
        static_level = (_level_for(fields, static.background, level)
                        if static.background else None)
        title = "Terrain (static)" if static.background == "terrain" else (
            quantity(static.background).long_name if static.background
            else "")
        panels.append(PanelPlacement(_map_panel(
            fields, static, None, static_level, title=title, groups=None,
            colorbar=True), row, 0, row_span=len(keys) or 1))
        for offset, key in enumerate(keys):
            panels.append(PanelPlacement(key, row + offset, 1 % columns))
        block = max(len(keys), 1)
        heights += [style.static_height_inches / block] * block
        row += block
    elif keys:
        for column, key in enumerate(keys[:columns]):
            panels.append(PanelPlacement(key, row, column))
        heights.append(0.62)
        row += 1

    map_titles = []
    for position, index in enumerate(indices):
        title = _time_label(fields.times[index], divisor, unit)
        map_titles.append(title)
        r, c = divmod(position, columns)
        panels.append(PanelPlacement(_map_panel(
            fields, view.map, index, level, title=title, groups=groups,
            colorbar=False, label_longitudes=r == map_rows - 1,
            label_latitudes=c == 0), row + r, c))
    heights += [style.map_row_height_inches] * map_rows
    row += map_rows
    if not indices:
        record["omitted"].append("maps: this run stores no snapshots")

    if diagnostics:
        for column, panel in enumerate(diagnostics[:columns]):
            if isinstance(panel, Drift):
                spec = _drift_panel(fields, panel, divisor, unit, record)
            elif isinstance(panel, Complexity):
                c_level = resolve_level(fields, panel.level)
                if fields.solver == "pe" and c_level is None:
                    c_level = level
                spec = _complexity_panel(fields, panel, c_level, divisor,
                                         unit, record)
            else:
                raise TypeError(f"unsupported diagnostics panel {panel!r}")
            panels.append(PanelPlacement(spec, row, column))
        heights.append(style.diagnostics_height_inches)
        row += 1

    record["maps"] = [_snapshot_statistics(fields, view.map, i, level)
                      for i in indices]
    record["map_titles"] = map_titles
    record["level"] = None if level is None else {
        "index": level, "sigma": fields.sigma_levels[level]}
    figure = FigureSpec(
        panels=tuple(panels), rows=row, columns=columns,
        size_inches=(style.width_inches, float(sum(heights))),
        dpi=style.dpi, height_ratios=tuple(heights),
        base_font_size=style.base_font_size, constrained_layout=True)
    return resolve_figure_normalizations(figure), record


def _compose_grid(fields: RunFields, view: Grid) -> tuple[FigureSpec, dict]:
    style = view.style
    record = _record_base(fields, view)
    (index,) = resolve_snapshots(fields, (view.snapshot,), 1)
    divisor, unit = _time_axis(fields.times)
    columns = max(len(row) for row in view.rows)
    panels: list[PanelPlacement] = []
    heights: list[float] = []
    heading, provenance = _header(fields, view.title)
    heading += f" · {_time_label(fields.times[index], divisor, unit)}"
    panels.append(PanelPlacement(TextPanelSpec(
        heading, font_family="sans-serif", font_size=style.base_font_size + 1,
        horizontal_alignment="left", font_weight="semibold"), 0, 0,
        column_span=columns))
    panels.append(PanelPlacement(TextPanelSpec(
        provenance, font_family="sans-serif",
        font_size=style.base_font_size - 1.5, horizontal_alignment="left",
        color="#555555"), 1, 0, column_span=columns))
    heights += [0.26, 0.2]
    record["maps"] = []
    for r, row_panels in enumerate(view.rows, start=2):
        is_map_row = any(isinstance(p, Map) for p in row_panels)
        for c, panel in enumerate(row_panels):
            if isinstance(panel, Map):
                level = resolve_level(fields, panel.level)
                entry = (quantity(panel.background)
                         if panel.background else None)
                title = (f"{entry.long_name}" if entry else "wind")
                if panel.vectors is not None and entry is not None:
                    title += " + wind"
                if level is not None:
                    title += f", {_level_label(fields, level)}"
                groups = {"background": f"grid-{r}-{c}-background",
                          "speed": f"grid-{r}-{c}-speed"}
                spec = _map_panel(fields, panel, index, level, title=title,
                                  groups=groups, colorbar=True)
                record["maps"].append(
                    _snapshot_statistics(fields, panel, index, level))
            elif isinstance(panel, Drift):
                spec = _drift_panel(fields, panel, divisor, unit, record)
            elif isinstance(panel, Complexity):
                spec = _complexity_panel(
                    fields, panel, resolve_level(fields, panel.level),
                    divisor, unit, record)
            else:
                raise TypeError(f"unsupported grid panel {panel!r}")
            panels.append(PanelPlacement(spec, r, c))
        width = style.width_inches / columns
        heights.append(max(width * 0.62, 1.4) if is_map_row
                       else style.diagnostics_height_inches)
    figure = FigureSpec(
        panels=tuple(panels), rows=2 + len(view.rows), columns=columns,
        size_inches=(style.width_inches, float(sum(heights))),
        dpi=style.dpi, height_ratios=tuple(heights),
        base_font_size=style.base_font_size, constrained_layout=True)
    record["snapshot"] = {"index": index, "time_s": float(fields.times[index])}
    return resolve_figure_normalizations(figure), record


def _png_metadata(fields: RunFields, record: dict) -> dict:
    run = record["run"]
    return {
        "Software": "palintropos (tropoi.representation.visual.compose)",
        "RunId": str(run.get("run_id") or ""),
        "Solver": fields.solver,
        "Commit": str((run.get("git") or {}).get("commit") or ""),
        "CoefficientsSHA256": str(run.get("coefficients_sha256") or ""),
        "DiagnosticsSHA256": str(run.get("diagnostics_sha256") or ""),
        "View": json.dumps(record["view"], sort_keys=True),
    }


def render_view(storage, view, output_path, *, snapshot: int | None = None,
                sidecar: bool = False, renderer=None) -> pathlib.Path:
    """Render ``view`` for a saved run to ``output_path`` (a new file).

    ``view=None`` draws the solver's default overview. ``snapshot`` (an
    index) turns a single :class:`Map` into a one-panel figure at that
    saved state. With ``sidecar=True`` every number the figure shows is
    also written to ``<output>.json``.
    """
    output = pathlib.Path(output_path).resolve()
    run_dir = pathlib.Path(storage.run_dir).resolve()
    if output == run_dir or run_dir in output.parents:
        raise ValueError(
            f"refusing to write {output} inside the saved run {run_dir}; "
            "saved runs are immutable")
    fields = RunFields(storage)
    if view is None:
        view = default_view(fields)
    figure, record = compose_view(fields, view, snapshot=snapshot)
    from tropoi.representation.visual.renderers import get_default_renderer
    backend = renderer or get_default_renderer()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = pathlib.Path(backend.render_figure(
        figure, output, metadata=_png_metadata(fields, record)))
    record["synthesis_count"] = fields.synthesis_count
    if sidecar:
        side = written.with_suffix(".json")
        side.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return written
