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
* figures belong to the run: by default they go to ``RUN/assets/``, and
  nothing is ever written anywhere else inside the run directory, so the
  primary run files stay untouched.
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
    StreamfunctionLayer, TextPanelSpec, VectorLayer)
from tropoi.representation.visual.timeline import (
    resolve_figure_normalizations, select_representative_frame_indices)
from tropoi.representation.visual.views import (
    Arrows, AutoVectors, Complexity, Contours, Drift, Grid, Map, Overview,
    Sigma, StreamfunctionContours, Streamlines, Style, describe, parse_time)

__all__ = ["ASSETS_DIRNAME", "compose_view", "default_output_path",
           "default_view", "elapsed_label", "render_view"]

#: Sub-directory of a run that holds its derived, reproducible products
#: (figures, sidecars). Nothing in it is run evidence: it never enters the
#: run id, the completion status or a published run's SHA256SUMS, and
#: ``--overwrite`` sweeps it with the other generated results.
ASSETS_DIRNAME = "assets"

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
    # White at the resting level, blue above it, red below: reads like
    # water, and fades to white where the surface is level.
    "free_surface_perturbation": ("RdBu", True),
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


def elapsed_label(seconds: float, span: float | None = None) -> str:
    """Elapsed physical time in plain words: ``Day 5, 3 h 17 min``.

    Elapsed simulation time, not a date. Zero components are omitted;
    seconds appear only when the stored time is not a whole minute, and
    fractions of a second only when present (to the millisecond, the
    precision the label claims). Whole days read ``Day N``; shorter times
    read ``18 h``, ``42 min``, ``7.5 s``. Zero takes the largest unit the
    run reaches (``span``): ``Day 0`` for a multi-day run, ``0 h`` for an
    hours-long one.
    """
    millis = int(round(float(seconds) * 1000.0))
    if millis < 0:
        raise ValueError("elapsed time must be nonnegative")
    days, millis = divmod(millis, 86_400_000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    parts = []
    if hours:
        parts.append(f"{hours} h")
    if minutes:
        parts.append(f"{minutes} min")
    if millis:
        whole, frac = divmod(millis, 1000)
        parts.append(f"{whole} s" if not frac else
                     f"{whole}.{frac:03d}".rstrip("0") + " s")
    rest = " ".join(parts)
    if days:
        return f"Day {days}" + (f", {rest}" if rest else "")
    if rest:
        return rest
    span = 0.0 if span is None else float(span)
    if span >= 86400.0:
        return "Day 0"
    if span >= 3600.0:
        return "0 h"
    if span >= 60.0:
        return "0 min"
    return "0 s"


def _time_label(seconds: float, times: np.ndarray) -> str:
    return elapsed_label(seconds, float(times[-1]) if times.size else 0.0)


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


def _nice_step(raw: float) -> float:
    """The smallest of 1, 2, 2.5, 5 x 10^k that is at least ``raw``."""
    base = 10.0 ** math.floor(math.log10(raw))
    step = next(factor * base for factor in (1.0, 2.0, 2.5, 5.0, 10.0)
                if factor * base >= raw * (1.0 - 1e-12))
    return float(f"{step:.3g}")


def _scientific(value: float) -> str:
    """``2.5e6`` -> ``2.5 × 10⁶``; moderate values stay plain."""
    mantissa, exponent = f"{value:.2e}".split("e")
    exponent = int(exponent)
    if -2 <= exponent <= 3:
        return f"{value:g}"
    mantissa = mantissa.rstrip("0").rstrip(".")
    power = "10" + str(exponent).translate(str.maketrans(
        "-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return power if mantissa == "1" else f"{mantissa} × {power}"


def _rotational_indices(fields: RunFields, indices, level) -> list[int]:
    """Saved states whose rotational wind is not at rest."""
    return [i for i in indices
            if fields.helmholtz_rms_speeds(i, level)[0] >= REST_SPEED_MS]


def _psi_interval(fields: RunFields, view: StreamfunctionContours, indices,
                  level) -> float:
    """One psi step for every map shown: the view's, or a round step giving
    about ``level_count`` steps across the range of psi in those maps."""
    if view.interval is not None:
        return float(view.interval)
    low = min(float(fields.state_values("streamfunction", i, level).min())
              for i in indices)
    high = max(float(fields.state_values("streamfunction", i, level).max())
               for i in indices)
    return _nice_step((high - low) / view.level_count)


def _percent(fraction: float) -> str:
    """A kinetic-energy share as a percentage: ``0.08``, ``27``, ``10⁻⁶``."""
    value = 100.0 * float(fraction)
    if value == 0.0:
        return "0"
    if value >= 10.0:
        return f"{value:.0f}"
    if value >= 0.01:
        return f"{value:.2g}"
    return _scientific(float(f"{value:.2g}"))


def _largest_share(fields: RunFields, indices, level) -> float | None:
    """The largest divergent kinetic-energy share over the saved states
    ``indices``; states at rest are skipped (``None`` if all are)."""
    shares = [fields.divergent_fraction(i, level) for i in indices]
    shares = [share for share in shares if share is not None]
    return max(shares) if shares else None


def _resolve_vectors(fields: RunFields, map_view: Map, indices, level
                     ) -> tuple[Map, dict | None]:
    """Replace an :class:`AutoVectors` overlay by the style it selects for
    the maps at ``indices``; return the map and the record of the choice
    (``None`` when the overlay was given explicitly)."""
    policy = map_view.vectors
    if not isinstance(policy, AutoVectors):
        return map_view, None
    wind_level = _level_for(fields, "wind", level)
    shares = [fields.divergent_fraction(i, wind_level) for i in indices]
    largest = _largest_share(fields, indices, wind_level)
    contours = (largest is not None
                and largest <= policy.divergent_threshold)
    chosen = policy.streamfunction if contours else policy.streamlines
    choice = {
        "policy": "automatic",
        "selected": type(chosen).__name__,
        "divergent_threshold": float(policy.divergent_threshold),
        "max_divergent_kinetic_energy_fraction": largest,
        "divergent_kinetic_energy_fraction": shares,
        "rule": "StreamfunctionContours when the largest E_div / (E_rot + "
                "E_div) over the maps shown is at most the threshold, else "
                "Streamlines of the full wind; states at rest (rms wind "
                f"below {REST_SPEED_MS:g} m/s, null share) are skipped",
    }
    return replace(map_view, vectors=chosen), choice


def _divergent_share(fields: RunFields, index: int, level) -> float | None:
    """Fraction of the kinetic energy carried by the divergent wind."""
    return fields.divergent_fraction(index, level)


def _streamfunction_description(fields: RunFields, interval: float,
                                peak_speed: float, level, indices,
                                threshold: float | None = None
                                ) -> list[str]:
    """Key or caption lines saying how psi contours encode the wind.

    ``threshold``: the automatic limit that selected psi contours, if any.
    """
    rotational_only = fields.solver != "bve"
    head = "rotational-wind streamlines" if rotational_only else "streamlines"
    lines = [f"{head}: ψ contours every {_scientific(interval)} m² s⁻¹"]
    speed = _nice_speeds(peak_speed)[-1]
    arc = math.degrees(interval / speed / fields.radius)
    lines.append(f"closer = faster: {arc:.2g}° apart at {speed:g} m s⁻¹ "
                 f"(max {peak_speed:.3g})")
    if rotational_only:
        share = _largest_share(fields, indices, level) or 0.0
        line = (f"divergent wind not drawn: ≤ {_percent(share)} % of "
                "kinetic energy")
        if threshold is not None:
            line += f" (auto limit {_percent(threshold)} %)"
        lines.append(line)
    return lines


def _full_wind_note(choice: dict | None) -> str:
    """Why an automatic choice kept streamlines of the full wind (two
    short lines; empty when the style was not chosen automatically)."""
    if choice is None or choice["selected"] != "Streamlines":
        return ""
    share = choice["max_divergent_kinetic_energy_fraction"]
    limit = _percent(choice["divergent_threshold"])
    if share is None:
        return "full wind: every state at rest"
    return (f"full wind drawn: divergent part up to {_percent(share)} %"
            + chr(10) + f"of kinetic energy (ψ contours need ≤ {limit} %)")


def _level_surface_scale(fields: RunFields, map_view: Map, indices,
                         largest: float) -> tuple[Map, str]:
    """Key note for free-surface perturbation maps, and the map to draw.

    The note names the reference level. A perturbation within roundoff of
    the level (every ``|eta'| <= 1e-12 H_ref``, the relative tolerance at
    which a colour scale of H itself is flat) is drawn on a fixed +-1 m
    scale, so a lake at rest stays white instead of stretching noise.
    """
    references = [fields.free_surface_reference(i) for i in indices]
    low, high = min(references), max(references)
    level = (f"{low:.0f} m" if f"{low:.0f}" == f"{high:.0f}"
             else f"{low:.0f}–{high:.0f} m")
    note = f"\nη′ = H − H̄;  H̄ = {level}, the level at rest"
    if largest <= 1e-12 * max(abs(high), 1.0) and map_view.limits is None:
        note += f"\n(level to roundoff: |η′| ≤ {largest:.1e} m; ±1 m shown)"
        map_view = replace(map_view, limits=(-1.0, 1.0))
    return map_view, note


# ---------------------------------------------------------------------------
# panel builders
# ---------------------------------------------------------------------------

def _map_panel(fields: RunFields, view: Map, index: int | None,
               level: int | None, *, title: str, groups: dict | None,
               colorbar: bool, label_longitudes: bool = True,
               label_latitudes: bool = True,
               psi_interval: float | None = None,
               background_note: str = "") -> LayeredMapSpec:
    background, corner = None, None
    if view.background is not None:
        entry = quantity(view.background)
        bg_level = _level_for(fields, view.background, level)
        field_index = None if entry.cadence == "static" else index
        grid_field = fields.scalar_field(view.background, field_index,
                                         bg_level)
        label = (f"{entry.long_name} ({pretty_units(entry.units)})"
                 + background_note)
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
        vector = getattr(view.vectors, "vector", "wind")
        if vector != "wind":
            raise QuantityUnavailableError(
                f"unknown vector quantity {vector!r}; available: wind")
        if index is None:
            raise QuantityUnavailableError("a static map cannot carry winds")
        wind_level = _level_for(fields, "wind", level)
        state_u, state_v = fields.state_wind(index, wind_level)
        if float(np.hypot(state_u, state_v).max()) < REST_SPEED_MS:
            # Roundoff-level winds have no direction worth drawing.
            corner = f"wind below {REST_SPEED_MS:g} m/s: at rest, not drawn"
        elif isinstance(view.vectors, StreamfunctionContours):
            if not _rotational_indices(fields, (index,), wind_level):
                corner = (f"rotational wind below {REST_SPEED_MS:g} m/s: "
                          "no streamfunction contours")
            else:
                psi_view = view.vectors
                vectors = StreamfunctionLayer(
                    fields.display_streamfunction(index, wind_level),
                    psi_interval or _psi_interval(fields, psi_view, (index,),
                                                  wind_level),
                    color=psi_view.color, line_width=psi_view.line_width,
                    alpha=psi_view.alpha, arrow_size=psi_view.arrow_size,
                    arrow_spacing=psi_view.arrow_spacing)
        else:
            wind = fields.display_wind(index, wind_level)
            common = dict(
                latitudes=wind.latitudes, longitudes=wind.longitudes,
                zonal=wind.zonal, meridional=wind.meridional,
                radius=wind.radius, color_by=view.vectors.color_by,
                color=view.vectors.color,
                normalization_group=None if groups is None else groups[
                    "speed"],
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
                                      arrow_stride=view.vectors.stride,
                                      **common)
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
        # Non-divergent flow: psi contours are its exact streamlines.
        return Overview(map=Map("vorticity",
                                vectors=StreamfunctionContours()))
    # SWE and PE: psi contours when the divergent wind is negligible in the
    # maps shown, streamlines of the full wind otherwise (AutoVectors).
    if solver == "swe":
        contours = ((Contours("terrain", (500.0, 1000.0, 1500.0)),)
                    if fields.has_terrain else ())
        return Overview(map=Map("free_surface_perturbation",
                                contours=contours, vectors=AutoVectors()))
    from tropoi.representation.visual.pe_snapshots import (
        select_snapshot_levels)
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    interfaces = fields.run_config.get("sigma_interfaces")
    sigma = (SigmaGrid.uniform(fields.nlev) if interfaces is None
             else SigmaGrid(tuple(float(s) for s in interfaces)))
    level = select_snapshot_levels(sigma).lower_index
    return Overview(map=Map("temperature_anomaly", level=level,
                            vectors=AutoVectors()))


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
        if entry.id == "free_surface_perturbation":
            stats[entry.id]["reference_m"] = fields.free_surface_reference(
                index)
    if view_map.vectors is not None:
        u, v = fields.state_wind(index, _level_for(fields, "wind", level))
        speed = np.hypot(u, v)
        stats["wind_speed"] = {"max": float(speed.max()),
                               "area_mean": float(np.sum(
                                   fields.state_weights() * speed)),
                               "units": "m s^-1"}
        if isinstance(view_map.vectors, StreamfunctionContours):
            wind_level = _level_for(fields, "wind", level)
            psi = fields.state_values("streamfunction", index, wind_level)
            rotational, divergent = fields.helmholtz_rms_speeds(
                index, wind_level)
            stats["streamfunction"] = {"min": float(psi.min()),
                                       "max": float(psi.max()),
                                       "units": "m^2 s^-1"}
            stats["rms_speed"] = {"rotational": rotational,
                                  "divergent": divergent,
                                  "units": "m s^-1"}
    return stats


def _compose_overview(fields: RunFields, view: Overview
                      ) -> tuple[FigureSpec, dict]:
    style: Style = view.style
    record = _record_base(fields, view)
    level = resolve_level(fields, view.map.level)
    indices = resolve_snapshots(fields, view.snapshots, view.max_maps)
    divisor, unit = _time_axis(fields.times)
    map_view, choice = _resolve_vectors(fields, view.map, indices, level)
    if choice is not None:
        record["vector_choice"] = choice

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
    if indices and map_view.background is not None:
        entry = quantity(map_view.background)
        label = f"{entry.long_name} ({pretty_units(entry.units)})"
        bg_level = _level_for(fields, entry.id, level)
        if bg_level is not None:
            label += f", {_level_label(fields, bg_level)}"
        magnitudes = [float(np.max(np.abs(fields.state_values(
            entry.id, i, bg_level)))) for i in indices]
        if entry.id == "free_surface_perturbation":
            map_view, note = _level_surface_scale(fields, map_view, indices,
                                                  max(magnitudes))
            label += note
        elif max(magnitudes) == 0.0:
            label += "\n(identically zero in every map)"
        elif max(magnitudes) <= 1e-12 and _normalization(
                entry.id, map_view).kind.value == "symmetric":
            label += (f"\n(every |value| ≤ {max(magnitudes):.1e}; "
                      "scale shown ±1)")
        keys.append(ColorKeySpec(
            groups["background"], label,
            color_policy=_color_policy(entry.id, map_view),
            normalization=_normalization(entry.id, map_view)))
    vectors = map_view.vectors
    speed_scale, psi_interval = None, None
    if indices and vectors is not None:
        wind_level = _level_for(fields, "wind", level)
        speeds = [np.hypot(*fields.state_wind(i, wind_level)).max()
                  for i in indices]
        speed_scale = float(max(speeds))
        rotational = (_rotational_indices(fields, indices, wind_level)
                      if isinstance(vectors, StreamfunctionContours) else [])
        if speed_scale < REST_SPEED_MS:
            record["omitted"].append(
                f"speed key: every wind is below {REST_SPEED_MS:g} m/s")
        elif isinstance(vectors, StreamfunctionContours) and not rotational:
            record["omitted"].append(
                "streamfunction key: every rotational wind is below "
                f"{REST_SPEED_MS:g} m/s")
        elif isinstance(vectors, StreamfunctionContours):
            # One psi step for every map, so spacing means one speed.
            psi_interval = _psi_interval(fields, vectors, rotational,
                                         wind_level)
            keys.append(TextPanelSpec(
                chr(10).join(_streamfunction_description(
                    fields, psi_interval, speed_scale, wind_level,
                    rotational, threshold=None if choice is None else
                    choice["divergent_threshold"])),
                font_family="sans-serif",
                font_size=style.base_font_size - 0.5, color="#222222"))
            record["streamfunction"] = {
                "interval_m2_s": psi_interval,
                "definition": "psi_lm = -R^2 zeta_lm / (l(l+1)), l = 0 set "
                              "to 0; lines at every multiple of the interval",
                "rotational_part_only": fields.solver != "bve",
                "divergent_kinetic_energy_fraction": [
                    _divergent_share(fields, i, wind_level)
                    for i in indices]}
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
                color=vectors.color, note=_full_wind_note(choice)))

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
        title = _time_label(fields.times[index], fields.times)
        map_titles.append(title)
        r, c = divmod(position, columns)
        panels.append(PanelPlacement(_map_panel(
            fields, map_view, index, level, title=title, groups=groups,
            colorbar=False, label_longitudes=r == map_rows - 1,
            label_latitudes=c == 0, psi_interval=psi_interval), row + r, c))
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

    record["maps"] = [_snapshot_statistics(fields, map_view, i, level)
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


def _vector_caption(fields: RunFields, panel: Map, index: int,
                    level: int | None, choice: dict | None = None) -> str:
    """A second title line saying how a single map encodes the wind.

    ``choice``: the record of an automatic choice that produced ``panel``.
    """
    vectors = panel.vectors
    if vectors is None:
        return ""
    u, v = fields.state_wind(index, _level_for(fields, "wind", level))
    peak = float(np.hypot(u, v).max())
    if peak < REST_SPEED_MS:
        return ""
    if isinstance(vectors, StreamfunctionContours):
        wind_level = _level_for(fields, "wind", level)
        if not _rotational_indices(fields, (index,), wind_level):
            return ""
        interval = _psi_interval(fields, vectors, (index,), wind_level)
        return chr(10) + chr(10).join(_streamfunction_description(
            fields, interval, peak, wind_level, (index,),
            threshold=None if choice is None else
            choice["divergent_threshold"]))
    style = ("streamlines" if isinstance(vectors, Streamlines)
             else "arrows")
    if isinstance(vectors, Streamlines) and vectors.width_by == "speed":
        note = _full_wind_note(choice)
        return (chr(10) + f"{style}: width ∝ wind speed, "
                f"max {peak:.3g} m s⁻¹" + (chr(10) + note if note else ""))
    if isinstance(vectors, Arrows):
        return (chr(10) + f"{style}: length ∝ wind speed, "
                f"max {peak:.3g} m s⁻¹")
    return chr(10) + f"{style} (max wind {peak:.3g} m s⁻¹)"


def _compose_grid(fields: RunFields, view: Grid) -> tuple[FigureSpec, dict]:
    style = view.style
    record = _record_base(fields, view)
    (index,) = resolve_snapshots(fields, (view.snapshot,), 1)
    divisor, unit = _time_axis(fields.times)
    columns = max(len(row) for row in view.rows)
    panels: list[PanelPlacement] = []
    heights: list[float] = []
    heading, provenance = _header(fields, view.title)
    heading += f" · {_time_label(fields.times[index], fields.times)}"
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
                panel, choice = _resolve_vectors(fields, panel, (index,),
                                                 level)
                entry = (quantity(panel.background)
                         if panel.background else None)
                title = entry.long_name if entry else "wind"
                if level is not None:
                    title += f", {_level_label(fields, level)}"
                title += _vector_caption(fields, panel, index, level, choice)
                note = ""
                if entry is not None and entry.id == (
                        "free_surface_perturbation"):
                    largest = float(np.max(np.abs(fields.state_values(
                        entry.id, index))))
                    panel, note = _level_surface_scale(fields, panel,
                                                       (index,), largest)
                groups = {"background": f"grid-{r}-{c}-background",
                          "speed": f"grid-{r}-{c}-speed"}
                spec = _map_panel(fields, panel, index, level, title=title,
                                  groups=groups, colorbar=True,
                                  background_note=note)
                stats = _snapshot_statistics(fields, panel, index, level)
                if choice is not None:
                    stats["vector_choice"] = choice
                record["maps"].append(stats)
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
        heights.append(max(width * 0.58, 1.4) if is_map_row
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


def default_output_path(storage, name: str = "overview.png"
                        ) -> pathlib.Path:
    """``RUN/assets/<name>``: where a run's derived figures live."""
    return pathlib.Path(storage.run_dir) / ASSETS_DIRNAME / name


def render_view(storage, view, output_path=None, *,
                snapshot: int | None = None, sidecar: bool = False,
                renderer=None) -> pathlib.Path:
    """Render ``view`` for a saved run and return the written image path.

    ``view=None`` draws the solver's default overview. ``snapshot`` (an
    index) turns a single :class:`Map` into a one-panel figure at that
    saved state. ``output_path=None`` writes the default overview to
    ``RUN/assets/overview.png``; any other view needs an explicit path.
    Inside the run directory only ``assets/`` may be written. With
    ``sidecar=True`` every number the figure shows is also written to
    ``<output>.json``.
    """
    if output_path is None:
        if view is not None or snapshot is not None:
            raise ValueError(
                "give an output path for a custom view; only the default "
                "overview has a default location (RUN/assets/overview.png)")
        output_path = default_output_path(storage)
    output = pathlib.Path(output_path).resolve()
    run_dir = pathlib.Path(storage.run_dir).resolve()
    assets = run_dir / ASSETS_DIRNAME
    if (output == run_dir or run_dir in output.parents) and (
            assets not in output.parents):
        raise ValueError(
            f"refusing to write {output}: inside the saved run {run_dir} "
            f"only {ASSETS_DIRNAME}/ holds derived figures, and the primary "
            "run files are never modified")
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
        # LF on every platform, so the sidecar's bytes do not depend on the OS.
        side.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    return written
