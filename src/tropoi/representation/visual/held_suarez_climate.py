"""The six-panel Held–Suarez latitude–sigma climate figure (PNG + PDF).

``Simulation.plot()`` on a Held–Suarez experiment with saved states draws,
from ONE set of saved states reduced by
:mod:`tropoi.representation.diagnostics.held_suarez_sections`:

1. mean zonal wind [u]           2. mean temperature [T]
3. eddy momentum flux [u*v*]     4. eddy heat flux [v*T*]
5. eddy kinetic energy           6. eddy temperature variance [T*²]
   ½[u*² + v*²]

on a white background with a LINEAR sigma axis increasing downward (the
protocol's levels are uniform in sigma; no log axis, no assumed tropopause,
no season labels). Every panel uses a fixed, documented scale
(:data:`HS_SECTION_SCALES`) so two runs drawn with the defaults can be
compared panel by panel; ``scales=`` overrides any of them. Values outside
a scale are drawn in the colour bar's extension colours AND hatched, and
the panel prints the data range and the share of points outside the scale.
The header and footer state units, the anomaly and averaging definitions,
the requested window, the actual first/last samples, count, cadence and
coverage, the run identity, resolution and the config/code provenance.

The PNG and the PDF are written from the same Matplotlib figure (one
reduction). ``plot()`` returns the PNG path; the PDF path is recorded in the
PNG metadata (``PDF``) and, with ``sidecar=True``, in ``<png>.json`` next to
every number drawn.

Reduction is separate from layout: :func:`reduce_experiment` returns a
:class:`~tropoi.representation.diagnostics.held_suarez_sections.
SectionResult`; :func:`draw_sections` only draws one.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tropoi.representation.diagnostics.held_suarez_sections import (
    SECTION_FIELDS, SECTION_KEYS, SectionError, SectionResult, load_state,
    reduce_sections, select_samples, state_grid)

__all__ = ["HS_SECTION_SCALES", "SectionScale", "draw_sections",
           "reduce_experiment", "render_held_suarez_climate"]

ASSETS_DIRNAME = "assets"           # = compose.ASSETS_DIRNAME (a test checks)
DEFAULT_NAME = "overview.png"


@dataclass(frozen=True)
class SectionScale:
    """Fixed filled-contour levels of one panel.

    ``levels`` are the colour boundaries (increasing); values below the
    first or above the last are out of range (extension colours + hatching).
    ``lines`` are the labelled contour lines; ``zero_line`` draws 0 bold.
    """

    levels: tuple[float, ...]
    cmap: str
    lines: tuple[float, ...] | None = None
    zero_line: bool = False

    def __post_init__(self):
        levels = tuple(float(x) for x in self.levels)
        if len(levels) < 2 or any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError(f"levels must increase strictly, got {self.levels}")
        object.__setattr__(self, "levels", levels)
        if self.lines is not None:
            object.__setattr__(self, "lines",
                               tuple(float(x) for x in self.lines))

    @property
    def step(self) -> float:
        return self.levels[1] - self.levels[0]

    def to_dict(self) -> dict:
        return {"levels": list(self.levels), "cmap": self.cmap,
                "lines": None if self.lines is None else list(self.lines),
                "zero_line": self.zero_line}


def _arange(lo: float, hi: float, step: float) -> tuple[float, ...]:
    n = int(round((hi - lo) / step))
    return tuple(round(lo + i * step, 10) for i in range(n + 1))


#: The fixed comparison scales (a repository recipe, documented with its
#: rationale in docs/held_suarez/CLIMATE_FIGURE.md): each range covers the
#: measured T42L20 climate of hs-T42L20-prod-002 (days 210-1200) with
#: headroom, so runs drawn with the defaults compare panel by panel. Colour
#: intervals 5 m/s, 5 K, 10 m²/s², 2 K m/s, 25 m²/s², 2.5 K²; labelled lines
#: every second interval (10 K for T).
HS_SECTION_SCALES: dict[str, SectionScale] = {
    "u": SectionScale(_arange(-40.0, 40.0, 5.0), "RdBu_r",
                      lines=_arange(-40.0, 40.0, 10.0), zero_line=True),
    "T": SectionScale(_arange(185.0, 315.0, 5.0), "YlOrRd",
                      lines=_arange(190.0, 310.0, 10.0)),
    "usvs": SectionScale(_arange(-80.0, 80.0, 10.0), "PuOr_r",
                         lines=_arange(-80.0, 80.0, 20.0), zero_line=True),
    "vsTs": SectionScale(_arange(-24.0, 24.0, 2.0), "RdBu_r",
                         lines=_arange(-24.0, 24.0, 4.0), zero_line=True),
    "eke": SectionScale(_arange(0.0, 450.0, 25.0), "YlGnBu",
                        lines=_arange(50.0, 450.0, 50.0)),
    "TsTs": SectionScale(_arange(0.0, 50.0, 2.5), "PuRd",
                         lines=_arange(5.0, 50.0, 5.0)),
}

_PANEL_SYMBOLS = {"u": "[u]", "T": "[T]", "usvs": "[u*v*]",
                  "vsTs": "[v*T*]", "eke": "½[u*² + v*²]", "TsTs": "[T*²]"}


def _resolve_scales(scales) -> dict[str, SectionScale]:
    out = dict(HS_SECTION_SCALES)
    if scales is None:
        return out
    if not isinstance(scales, Mapping):
        raise TypeError("scales must map panel keys to SectionScale or to a "
                        "sequence of levels")
    unknown = sorted(set(scales) - set(SECTION_KEYS))
    if unknown:
        raise ValueError(f"unknown panel keys {unknown}; expected some of "
                         f"{list(SECTION_KEYS)}")
    for key, value in scales.items():
        if isinstance(value, SectionScale):
            out[key] = value
        else:
            base = HS_SECTION_SCALES[key]
            out[key] = SectionScale(tuple(value), base.cmap, lines=None,
                                    zero_line=base.zero_line)
    return out


# ---------------------------------------------------------------------------
# reduction from an experiment directory
# ---------------------------------------------------------------------------

def _stored_latitudes(run_dir: pathlib.Path):
    path = run_dir / "statistics.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as npz:
        return np.array(npz["lat"]) if "lat" in npz.files else None


def _nominal_truncation(config: Mapping[str, Any]) -> int:
    """The evolved truncation the config declares (cf. HeldSuarezConfig
    .retained_cut): ``retained_truncation`` or the 2/3 product cut."""
    retained = config.get("retained_truncation")
    if retained is not None:
        return int(retained)
    from tropoi.spatial.truncation import product_truncation_cut
    return int(product_truncation_cut(int(config["l_max"])))


def reduce_experiment(storage, *, interval=None, min_states: int = 2,
                      progress=None) -> SectionResult:
    """Select, reconstruct and reduce a Held–Suarez experiment's states."""
    config = dict(storage.experiment_config)
    if storage.frame_count == 0:
        from tropoi.representation.archive.availability import (
            statistics_only_description)
        reason = statistics_only_description(storage.run_dir) or ""
        raise SectionError(
            f"{storage.run_dir} has no saved states (states/state-d*.npy), so "
            "the six climate sections cannot be reconstructed. " + reason
            + ". Plotting from statistics.npz alone is not implemented yet.")
    selection = select_samples(storage.days, config, interval=interval,
                               min_states=min_states)
    lat, lon, lat_source = state_grid(int(config["nlat"]),
                                      int(config["nlon"]),
                                      _stored_latitudes(storage.run_dir))
    ctx = storage.pe_context()
    nlev = int(config["nlev"])

    def frames():
        for index in selection.indices:
            path = storage.state_path(index)
            array, digest = load_state(path)
            yield array, path.name, digest

    return reduce_sections(
        frames(), selection, nlev=nlev, l_max=int(config["l_max"]),
        nominal_truncation=_nominal_truncation(config), lat_deg=lat,
        lon_deg=lon, sigma=np.asarray(ctx.full_levels, dtype=np.float64),
        sigma_interfaces=np.asarray(ctx.sigma_interfaces, dtype=np.float64),
        radius_m=float(ctx.radius_m), latitude_source=lat_source,
        progress=progress)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def _short(value, n=12) -> str:
    return str(value)[:n] if value else "unknown"


def describe_run(storage) -> dict:
    """Identity, resolution and provenance strings of an experiment."""
    cfg = dict(storage.experiment_config)
    doc = storage.document
    code = doc.get("code") or {}
    commit = code.get("git_commit") or code.get("commit")
    dirty = code.get("git_dirty")
    l_max = int(cfg["l_max"])
    trunc = _nominal_truncation(cfg)
    if cfg.get("retained_truncation") is not None:
        spectral = f"T{trunc} retained (stored l_max {l_max})"
    else:
        spectral = (f"l_max {l_max} with the 2/3 product cut "
                    f"(evolves l ≤ {trunc})")
    return {
        "experiment_id": cfg.get("experiment_id"),
        "spectral": spectral,
        "levels": f"L{cfg['nlev']} uniform σ",
        "grid": f"{cfg['nlat']} × {cfg['nlon']} Gaussian state grid",
        "dt_s": cfg.get("dt"),
        "damping_scheme": cfg.get("damping_scheme", "lagged"),
        "config_sha256": doc.get("config_sha256"),
        "protocol_sha256": doc.get("protocol_sha256"),
        "run_commit": commit,
        "run_dirty": dirty,
        "created": doc.get("created"),
    }


def _plotter_commit() -> str | None:
    """Short commit of the plotting code, marked when the tree is modified."""
    try:
        from tropoi.representation.archive.writer import _git
        commit = _git(["rev-parse", "--short=8", "HEAD"])
        if commit is None:
            return None
        dirty = _git(["status", "--porcelain", "--untracked-files=no"])
        return commit + (" + local changes" if dirty else "")
    except Exception:          # pragma: no cover - no git, no repository
        return None


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if x == 0.0 or not np.isfinite(x):
        return f"{x:g}"
    mag = abs(x)
    if mag >= 100:
        return f"{x:.0f}"
    if mag >= 10:
        return f"{x:.1f}"
    if mag >= 0.1:
        return f"{x:.2f}"
    return f"{x:.1e}"


_LAT_TICKS = (-60, -30, 0, 30, 60)
_LAT_LABELS = ("60°S", "30°S", "0°", "30°N", "60°N")


def panel_facts(values, scale: SectionScale) -> dict:
    """Data range and share of points outside the panel's fixed scale."""
    z = np.asarray(values, dtype=np.float64)
    lo, hi = float(z.min()), float(z.max())
    below = float(np.mean(z < scale.levels[0]))
    above = float(np.mean(z > scale.levels[-1]))
    return {"min": lo, "max": hi, "fraction_below_scale": below,
            "fraction_above_scale": above,
            "fraction_outside_scale": below + above}


def _panel(axes, cax, key, values, result, scale, base_font, *,
           xlabel: bool) -> dict:
    import matplotlib.colors as mcolors
    from matplotlib import colormaps

    title, units, _ = SECTION_FIELDS[key]
    facts = panel_facts(values, scale)
    # Drawn with latitude ascending (south left, north right); the data
    # contract (north -> south rows) is unchanged.
    order = np.argsort(result.lat_deg)
    lat = result.lat_deg[order]
    z = np.asarray(values, dtype=np.float64)[order, :].T      # (nlev, nlat)
    sigma = result.sigma
    levels = np.asarray(scale.levels)
    cmap = colormaps[scale.cmap]
    n_colors = len(levels) - 1
    colors = cmap(np.linspace(0.06, 0.94, n_colors)) if n_colors > 1 \
        else cmap([0.5])
    listed = mcolors.ListedColormap(colors).with_extremes(
        over=cmap(1.0), under=cmap(0.0))
    norm = mcolors.BoundaryNorm(levels, n_colors)
    filled = axes.contourf(lat, sigma, z, levels=levels, cmap=listed,
                           norm=norm, extend="both", zorder=-1.0)
    # The filled bands (zorder < 0) are rasterized at the save dpi so PDF
    # viewers that antialias adjacent polygons show no hairline seams
    # between bands; hatching, contour lines, labels and text stay vector.
    axes.set_rasterization_zorder(0.0)
    lo, hi = facts["min"], facts["max"]
    # Out-of-range values: extension colours AND hatching (visible in print).
    if facts["fraction_below_scale"] > 0.0:
        axes.contourf(lat, sigma, z, levels=[-np.inf, levels[0]],
                      colors="none", hatches=["xxxx"])
    if facts["fraction_above_scale"] > 0.0:
        axes.contourf(lat, sigma, z, levels=[levels[-1], np.inf],
                      colors="none", hatches=["xxxx"])
    if hi > lo:
        lines = [x for x in (scale.lines or ()) if lo < x < hi
                 and not (scale.zero_line and x == 0.0)]
        if lines:
            cs = axes.contour(lat, sigma, z, levels=lines, colors="#1f1f1f",
                              linewidths=0.6)
            labelled = [x for x in cs.levels[::2]]
            axes.clabel(cs, levels=labelled, fmt=lambda x: f"{x:g}",
                        fontsize=base_font - 1.5, inline=True,
                        inline_spacing=3)
        if scale.zero_line and lo < 0.0 < hi:
            axes.contour(lat, sigma, z, levels=[0.0], colors="black",
                         linewidths=1.2)
    axes.set_xlim(-90.0, 90.0)
    # Linear sigma, increasing downward, spanning the native full levels
    # (the data); every native level is a minor tick.
    axes.set_ylim(float(sigma.max()), float(sigma.min()))
    axes.set_xticks(_LAT_TICKS)
    axes.set_xticklabels(_LAT_LABELS if xlabel else [])
    axes.set_yticks([t for t in (0.2, 0.4, 0.6, 0.8)
                     if sigma.min() <= t <= sigma.max()])
    axes.set_yticks(sigma, minor=True)
    axes.tick_params(axis="y", which="minor", length=2.5, width=0.4)
    axes.tick_params(axis="x", which="both", direction="out")
    axes.set_ylabel("σ")
    if xlabel:
        axes.set_xlabel("latitude")
    axes.set_title(f"{title}  {_PANEL_SYMBOLS[key]}", loc="left",
                   fontsize=base_font + 1.0, fontweight="semibold", pad=13.0)
    if hi == lo == 0.0:
        note = "identically zero"
    else:
        note = f"data {_fmt(lo)} to {_fmt(hi)} {units}"
        share = facts["fraction_outside_scale"]
        if share > 0.0:
            note += f"; {100 * share:.1f}% of points beyond scale (hatched)"
    axes.text(0.0, 1.012, note, transform=axes.transAxes, ha="left",
              va="bottom", fontsize=base_font - 1.5, color="#333333")
    bar = axes.figure.colorbar(filled, cax=cax, extend="both",
                               spacing="proportional")
    bar.set_label(units, fontsize=base_font)
    bar.ax.tick_params(labelsize=base_font - 1.5, length=2.5)
    ticks = scale.lines if scale.lines else scale.levels
    bar.set_ticks([t for t in ticks if levels[0] <= t <= levels[-1]])
    return facts


def _header_lines(result: SectionResult, run: dict) -> tuple[str, list[str]]:
    sel = result.selection
    title = f"Held–Suarez climate: {run['experiment_id']}"
    dt = run.get("dt_s")
    start, end = sel.requested
    lines = [
        f"{run['spectral']} · {run['levels']} · {run['grid']}"
        + (f" · dt {dt:g} s" if dt else "")
        + f" · {run['damping_scheme']} damping",
        f"Time mean: equal-weight mean of {sel.count} saved states with "
        f"{start:g} < day ≤ {end:g} ({sel.window_source}).",
        f"Samples: day {sel.days[0]} to day {sel.days[-1]}, "
        f"{sel.cadence_label()}. Coverage: {sel.coverage_label()}.",
    ]
    return title, lines


def _footer_lines(result: SectionResult, run: dict, plotter: str | None,
                  overridden: tuple[str, ...] = ()) -> list[str]:
    nlon = result.lon_deg.size
    band = result.bandlimit
    commit = _short(run.get("run_commit"))
    if run.get("run_dirty"):
        commit += " (dirty)"
    sigma = result.sigma
    return [
        f"[ ] = unweighted mean over the {nlon} equally spaced longitudes "
        "at one latitude on one native σ level; * = departure from that "
        "state's instantaneous [ ]. Eddy products are formed for each "
        "state, then averaged.",
        "All six panels come from the same states. No trapezoidal "
        "weighting, interpolation or gap filling. σ-level means are not "
        f"pressure-level means. The σ axis spans the full levels "
        f"{sigma.min():g} to {sigma.max():g} (minor ticks).",
        f"u, v, T synthesized on the host from the saved coefficients on the "
        f"run's Gaussian latitudes; measured zonal bandlimit m_max = "
        f"{band['max_nonzero_order']} < nlon/2, so longitude means of "
        "products are exact. "
        + ("Fixed scales: HS_SECTION_SCALES" if not overridden else
           "Fixed scales: HS_SECTION_SCALES except "
           + ", ".join(_PANEL_SYMBOLS[k] for k in overridden)
           + " (overridden)")
        + "; values beyond the scale are hatched.",
        f"config sha256 {_short(run.get('config_sha256'))}…  ·  run commit "
        f"{commit}  ·  protocol sha256 {_short(run.get('protocol_sha256'))}…"
        f"  ·  states sha256 {result.states_sha256()[:12]}…  ·  plotted "
        f"with palintropos {plotter or '(commit unknown)'}",
    ]


#: Figure geometry (inches; A4 portrait) and fractions of the figure.
FIGURE_SIZE = (8.27, 11.69)
_LEFT, _RIGHT = 0.075, 0.925
_PANELS_TOP, _PANELS_BOTTOM = 0.83, 0.145


def draw_sections(result: SectionResult, png_path, pdf_path, *, run: dict,
                  scales: Mapping[str, SectionScale] | None = None,
                  png_metadata: dict | None = None,
                  pdf_metadata: dict | None = None, dpi: int = 200,
                  plotter: str | None = None) -> dict:
    """Draw the six panels of ``result``; write the PNG and the PDF.

    Both files are saved from the same figure. Returns per-panel facts
    (data range, share of points beyond the fixed scale).
    """
    import textwrap

    import matplotlib
    import matplotlib.pyplot as plt

    from tropoi.representation.visual.matplotlib_renderer import (
        MatplotlibRenderer)

    scales = _resolve_scales(scales)
    overridden = tuple(k for k in SECTION_KEYS
                       if scales[k] != HS_SECTION_SCALES[k])
    base_font = 8.5
    rc = {"font.size": base_font, "axes.titlesize": base_font + 1.0,
          "axes.labelsize": base_font, "xtick.labelsize": base_font - 0.5,
          "ytick.labelsize": base_font - 0.5, "axes.linewidth": 0.6,
          "figure.facecolor": "white", "axes.facecolor": "white",
          "savefig.facecolor": "white", "pdf.fonttype": 42,
          "hatch.linewidth": 0.6, "hatch.color": "#111111"}
    facts = {}
    with matplotlib.rc_context(rc):
        figure = plt.figure(figsize=FIGURE_SIZE)
        try:
            title, header = _header_lines(result, run)
            figure.text(_LEFT, 0.972, title, ha="left", va="top",
                        fontsize=base_font + 4.5, fontweight="bold")
            figure.text(_LEFT, 0.945, "\n".join(
                textwrap.fill(line, 112) for line in header), ha="left",
                va="top", fontsize=base_font, linespacing=1.5)
            grid = figure.add_gridspec(
                3, 5, left=_LEFT, right=_RIGHT, top=_PANELS_TOP,
                bottom=_PANELS_BOTTOM, hspace=0.36, wspace=0.08,
                width_ratios=(1.0, 0.035, 0.30, 1.0, 0.035))
            for n, key in enumerate(SECTION_KEYS):
                row, col = divmod(n, 2)
                axes = figure.add_subplot(grid[row, 3 * col])
                cax = figure.add_subplot(grid[row, 3 * col + 1])
                facts[key] = _panel(axes, cax, key, result[key], result,
                                    scales[key], base_font,
                                    xlabel=row == 2)
            footer = "\n".join(textwrap.fill(line, 138)
                               for line in _footer_lines(
                                   result, run, plotter, overridden))
            figure.text(_LEFT, 0.098, footer, ha="left", va="top",
                        fontsize=base_font - 1.5, linespacing=1.45,
                        color="#333333")
            png = pathlib.Path(png_path)
            pdf = pathlib.Path(pdf_path)
            png.parent.mkdir(parents=True, exist_ok=True)
            MatplotlibRenderer._save_atomic(figure, png, dpi=dpi,
                                            metadata=png_metadata)
            MatplotlibRenderer._save_atomic(figure, pdf, dpi=dpi,
                                            metadata=pdf_metadata)
        finally:
            plt.close(figure)
    return facts


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

def _checked_output(storage, output_path) -> pathlib.Path:
    run_dir = pathlib.Path(storage.run_dir).resolve()
    if output_path is None:
        output = run_dir / ASSETS_DIRNAME / DEFAULT_NAME
    else:
        output = pathlib.Path(output_path).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError(
            f"the Held–Suarez climate figure is written as a PNG plus a PDF "
            f"beside it; give a .png path, got {output.name!r}")
    assets = run_dir / ASSETS_DIRNAME
    if (output == run_dir or run_dir in output.parents) and (
            assets not in output.parents):
        raise ValueError(
            f"refusing to write {output}: inside the saved run {run_dir} "
            f"only {ASSETS_DIRNAME}/ holds derived figures, and the primary "
            "run files are never modified")
    return output


def render_held_suarez_climate(storage, output_path=None, *, interval=None,
                               scales=None, min_states: int = 2,
                               sidecar: bool = False, dpi: int = 200,
                               progress=None) -> pathlib.Path:
    """Reduce and draw the six-panel climate figure; return the PNG path.

    ``output_path=None`` writes ``RUN/assets/overview.png``; the PDF goes
    beside the PNG with the same stem. ``interval=(start, end)`` overrides
    the averaging window (``start < day <= end``); ``scales`` overrides
    panel scales; ``min_states`` is the fewest states accepted.
    """
    png = _checked_output(storage, output_path)
    nlev = int(storage.experiment_config["nlev"])
    if nlev < 2:
        raise SectionError(
            f"{storage.run_dir} has nlev = {nlev}: a latitude–σ section "
            "needs at least two σ levels to contour, so this figure is not "
            "drawn. reduce_experiment(storage) still gives the (nlat, 1) "
            "zonal and eddy statistics of the single level")
    pdf = png.with_suffix(".pdf")
    scales = _resolve_scales(scales)
    started = time.perf_counter()
    result = reduce_experiment(storage, interval=interval,
                               min_states=min_states, progress=progress)
    reduce_s = time.perf_counter() - started
    run = describe_run(storage)
    plotter = _plotter_commit()
    sel = result.selection
    view = {"view": "held_suarez_climate", "panels": list(SECTION_KEYS),
            "interval_days": list(sel.requested),
            "window_source": sel.window_source, "min_states": min_states,
            "scales": {k: s.to_dict() for k, s in scales.items()}}
    png_meta = {
        "Software": "palintropos (tropoi.representation.visual."
                    "held_suarez_climate)",
        "RunId": str(run["experiment_id"] or ""),
        "Solver": "pe",
        "Experiment": "held_suarez",
        "Commit": str(run.get("run_commit") or ""),
        "ConfigSHA256": str(run.get("config_sha256") or ""),
        "StatesSHA256": result.states_sha256(),
        "Samples": json.dumps(sel.to_dict(), sort_keys=True),
        "PDF": pdf.name,
        "View": json.dumps(view, sort_keys=True),
    }
    pdf_meta = {
        "Title": f"Held–Suarez climate: {run['experiment_id']}",
        "Subject": sel.summary(),
        "Keywords": f"config_sha256={run.get('config_sha256')}; "
                    f"states_sha256={result.states_sha256()}",
        "Creator": png_meta["Software"],
        "CreationDate": None,
    }
    facts = draw_sections(result, png, pdf, run=run, scales=scales,
                          png_metadata=png_meta, pdf_metadata=pdf_meta,
                          dpi=dpi, plotter=plotter)
    if sidecar:
        record = {
            "view": view, "run": run, "plotted_by": plotter,
            "outputs": {"png": png.name, "pdf": pdf.name},
            "samples": sel.to_dict(),
            "state_files": [{"name": n, "sha256": d}
                            for n, d in result.state_files],
            "states_sha256": result.states_sha256(),
            "grid": {"latitude_deg": result.lat_deg.tolist(),
                     "latitude_order": "north to south (as stored)",
                     "latitude_source": result.latitude_source,
                     "nlon": int(result.lon_deg.size),
                     "longitude_rule": "lon_j = 360 j / nlon degrees",
                     "sigma_full": result.sigma.tolist(),
                     "sigma_interfaces": result.sigma_interfaces.tolist(),
                     "level_order": "top (smallest sigma) to bottom"},
            "bandlimit": result.bandlimit,
            "panels": {key: {"title": SECTION_FIELDS[key][0],
                             "units": SECTION_FIELDS[key][1],
                             "definition": SECTION_FIELDS[key][2],
                             "values": np.asarray(result[key]).tolist(),
                             **facts[key]} for key in SECTION_KEYS},
            "timing_s": {"reduce": reduce_s},
        }
        png.with_suffix(".json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
    return png
