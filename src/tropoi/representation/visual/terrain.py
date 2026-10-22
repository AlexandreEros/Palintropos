"""Terrain previews: a static figure and an interactive HTML globe (CPU).

Both render one :class:`TerrainPreview`. Its model layer is the resolved
elevation coefficients evaluated exactly on a regular latitude-longitude
node mesh (:func:`~tropoi.spatial.transforms.host_synthesis.synthesize_latlon`)
-- the band-limited terrain the solver uses, not a re-projection. An
optional source layer shows the input raster before truncation
(area-averaged to the display spacing), on the same colour scale.

* :func:`plot_terrain` -- Matplotlib figure (static output).
* :func:`write_globe_html` -- one self-contained HTML file: a rotatable,
  zoomable WebGL globe with an elevation colour bar, graticules, labelled
  display-only relief exaggeration and a matching flat map. It needs no
  network, server, Python or CUDA to open.

NumPy at import; Matplotlib only inside :func:`plot_terrain`. Never CuPy.
"""
from __future__ import annotations

import base64
import importlib.resources
import json
import math
import pathlib
import textwrap
from dataclasses import dataclass, field

import numpy as np

from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

__all__ = [
    "TERRAIN_COLOR_POLICY",
    "TerrainPreview",
    "build_terrain_preview",
    "display_resolution_deg",
    "plot_terrain",
    "write_globe_html",
]

#: Colour policy of terrain everywhere in the repository's figures
#: (``compose.DISPLAY_DEFAULTS["terrain"]``): sequential, light low to dark
#: high.
TERRAIN_COLOR_POLICY = "YlOrBr:0.0:0.85"


def display_resolution_deg(l_max: int) -> float:
    """Node spacing of the preview mesh: 1 degree, or 0.5 above l_max 44.

    At least about four nodes per shortest resolved wavelength, so the
    exact synthesis on the mesh shows every resolved feature.
    """
    return 1.0 if 180.0 / (4.0 * (int(l_max) + 1)) >= 1.0 else 0.5


@dataclass(frozen=True, eq=False)
class TerrainPreview:
    """Everything a terrain figure or globe draws (host arrays, metres).

    ``model_m`` / ``source_m`` are ``(nlat, nlon)`` on the node mesh
    ``lat_deg`` (``-90 .. 90``) x ``lon_deg`` (``-180 .. 180``, seam
    repeated). ``details`` are (label, value) lines describing the planet,
    the terrain's provenance and its preprocessing.
    """

    title: str
    radius_m: float
    l_max: int
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    model_m: np.ndarray
    model_label: str
    source_m: np.ndarray | None = None
    source_label: str | None = None
    subtitle: str = ""
    details: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def resolution_deg(self) -> float:
        return float(self.lat_deg[1] - self.lat_deg[0])

    def value_range(self) -> tuple[float, float]:
        """Shared colour range (m) over every layer."""
        arrays = [self.model_m] + ([] if self.source_m is None
                                   else [self.source_m])
        return (float(min(a.min() for a in arrays)),
                float(max(a.max() for a in arrays)))

    def highest_point(self) -> tuple[float, float]:
        """(lat, lon) in degrees of the model terrain's maximum."""
        j, i = np.unravel_index(int(np.argmax(self.model_m)),
                                self.model_m.shape)
        return float(self.lat_deg[j]), float(self.lon_deg[i])


def build_terrain_preview(elevation_lm, *, radius_m: float, title: str,
                          subtitle: str = "", details=(), raster=None,
                          resolution_deg: float | None = None
                          ) -> TerrainPreview:
    """Evaluate resolved coefficients (and an optional source raster).

    ``elevation_lm`` are host ``(L+1, L+1)`` surface-elevation coefficients
    (metres) in the repository layout; ``raster`` an
    :class:`~tropoi.spatial.terrain.raster.ElevationRaster` shown as the
    "before truncation" layer.
    """
    coeffs = np.asarray(elevation_lm)
    l_max = int(coeffs.shape[0]) - 1
    d = float(resolution_deg or display_resolution_deg(l_max))
    lat = np.linspace(-90.0, 90.0, int(round(180.0 / d)) + 1)
    lon = np.linspace(-180.0, 180.0, int(round(360.0 / d)) + 1)
    model = synthesize_latlon(coeffs, lat, lon)
    source = source_label = None
    if raster is not None:
        lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
        source = raster.coarsened(d).sample(lat2, lon2)
        spacing = max(d, raster.resolution_deg)
        source_label = (f"Source raster before truncation "
                        f"({raster.provenance.get('source_name', 'array')}, "
                        f"area-averaged to {spacing:g}°)")
    return TerrainPreview(
        title=str(title), radius_m=float(radius_m), l_max=l_max,
        lat_deg=lat, lon_deg=lon, model_m=model,
        model_label=(f"Model terrain as the solver uses it: spectral "
                     f"truncation l_max = {l_max}, evaluated exactly"),
        source_m=source, source_label=source_label, subtitle=str(subtitle),
        details=tuple((str(k), str(v)) for k, v in details))


def _display_units(lo: float, hi: float) -> tuple[str, float]:
    return ("km", 1000.0) if max(abs(lo), abs(hi)) >= 2000.0 else ("m", 1.0)


def _color_map(policy: str = TERRAIN_COLOR_POLICY):
    from tropoi.representation.visual.matplotlib_renderer import (
        MatplotlibRenderer)
    return MatplotlibRenderer._color_map(policy)


def default_exaggeration(preview: TerrainPreview) -> float:
    """Relief factor that lifts the full range to ~8% of the radius."""
    lo, hi = preview.value_range()
    span = hi - lo
    if not span > 0:
        return 1.0
    raw = 0.08 * preview.radius_m / span
    if raw <= 1.0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    return float(max(1.0, min((1, 2, 2.5, 5, 10),
                              key=lambda f: abs(f * mag - raw)) * mag))


# ---------------------------------------------------------------------------
# Static figure
# ---------------------------------------------------------------------------

def _lon_label(v: float) -> str:
    v = (v + 180.0) % 360.0 - 180.0 if abs(v) != 180.0 else v
    if v == 0:
        return "0°"
    return f"{abs(v):g}°{'E' if v > 0 else 'W'}"


def _lat_label(v: float) -> str:
    return "0°" if v == 0 else f"{abs(v):g}°{'N' if v > 0 else 'S'}"


def _map_axes(ax, preview: TerrainPreview, values, cmap, norm, title: str):
    d = preview.resolution_deg
    image = ax.imshow(values, origin="lower", cmap=cmap, norm=norm,
                      interpolation="nearest",
                      extent=(-180 - d / 2, 180 + d / 2, -90 - d / 2,
                              90 + d / 2))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_xticklabels([_lon_label(v) for v in np.arange(-180, 181, 60)])
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_yticklabels([_lat_label(v) for v in np.arange(-90, 91, 30)])
    for v in range(-150, 180, 30):
        ax.axvline(v, color="0.15", lw=0.4, alpha=0.35)
    for v in (-60, -30, 0, 30, 60):
        ax.axhline(v, color="0.15", lw=0.4, alpha=0.35)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    return image


def plot_terrain(preview: TerrainPreview, *, figsize=None):
    """Static terrain figure: model terrain, and source + difference if any.

    Returns the Matplotlib figure (created with ``pyplot``, so
    ``plt.show()`` displays it; save it with ``fig.savefig``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    lo, hi = preview.value_range()
    units, factor = _display_units(lo, hi)
    cmap = _color_map()
    norm = Normalize(vmin=lo / factor, vmax=(hi if hi > lo else lo + 1.0)
                     / factor)
    has_source = preview.source_m is not None
    model_title = (f"Model terrain (l_max = {preview.l_max}, as the solver "
                   "uses it)")
    if has_source:
        fig, axes = plt.subplots(2, 2, figsize=figsize or (13.0, 8.2),
                                 layout="constrained")
        ax_src, ax_mod, ax_dif, ax_txt = axes.ravel()
        _map_axes(ax_src, preview, preview.source_m / factor, cmap, norm,
                  "Source raster (before truncation)")
        image = _map_axes(ax_mod, preview, preview.model_m / factor, cmap,
                          norm, model_title)
        fig.colorbar(image, ax=[ax_src, ax_mod], location="bottom",
                     shrink=0.5, aspect=40,
                     label=f"elevation ({units}), one scale for both maps")
        diff = preview.model_m - preview.source_m
        dunits, dfactor = _display_units(float(diff.min()),
                                         float(diff.max()))
        bound = max(float(np.abs(diff).max()), 1e-9) / dfactor
        dimage = _map_axes(
            ax_dif, preview, diff / dfactor, "RdBu_r",
            TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound),
            "Model − source (resampling, smoothing, truncation)")
        fig.colorbar(dimage, ax=ax_dif, location="bottom", shrink=0.9,
                     aspect=30, label=f"difference ({dunits})")
    else:
        fig, (ax_mod, ax_txt) = plt.subplots(
            2, 1, figsize=figsize or (9.5, 8.0), layout="constrained",
            height_ratios=(1.0, 0.55))
        image = _map_axes(ax_mod, preview, preview.model_m / factor, cmap,
                          norm, model_title)
        fig.colorbar(image, ax=ax_mod, location="bottom", shrink=0.6,
                     aspect=40, label=f"elevation ({units})")
    ax_txt.axis("off")
    lines = textwrap.wrap(preview.subtitle, width=92)
    for key, value in preview.details:
        lines.extend(textwrap.wrap(f"{key}: {value}", width=92,
                                   subsequent_indent="    ") or [key])
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                fontsize=7.5, family="monospace", transform=ax_txt.transAxes)
    fig.suptitle(preview.title, fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Interactive globe
# ---------------------------------------------------------------------------

def _b64_f32(values: np.ndarray) -> str:
    return base64.b64encode(
        np.ascontiguousarray(values, dtype="<f4").tobytes()).decode("ascii")


def globe_payload(preview: TerrainPreview) -> dict:
    """The JSON document the globe page reads (exposed for tests)."""
    lo, hi = preview.value_range()
    units, factor = _display_units(lo, hi)
    cmap = _color_map()
    lut = (np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3] * 255.0)
    layers = [{"id": "model", "short": "Model",
               "label": preview.model_label,
               "data": _b64_f32(preview.model_m),
               "min": float(preview.model_m.min()),
               "max": float(preview.model_m.max())}]
    if preview.source_m is not None:
        layers.append({"id": "source", "short": "Source",
                       "label": preview.source_label,
                       "data": _b64_f32(preview.source_m),
                       "min": float(preview.source_m.min()),
                       "max": float(preview.source_m.max())})
    view_lat, view_lon = preview.highest_point()
    return {
        "title": preview.title, "subtitle": preview.subtitle,
        "radius_m": preview.radius_m, "l_max": preview.l_max,
        "nlat": int(preview.lat_deg.size), "nlon": int(preview.lon_deg.size),
        "lat0": float(preview.lat_deg[0]), "lon0": float(preview.lon_deg[0]),
        "dlat": preview.resolution_deg,
        "dlon": float(preview.lon_deg[1] - preview.lon_deg[0]),
        "range": [lo, hi], "units": units, "unit_factor": factor,
        "lut": np.rint(lut).astype(int).tolist(),
        "layers": layers,
        "default_exaggeration": default_exaggeration(preview),
        "view_lat": max(-60.0, min(60.0, view_lat)), "view_lon": view_lon,
        "details": [list(item) for item in preview.details],
    }


def write_globe_html(preview: TerrainPreview, path) -> pathlib.Path:
    """Write the self-contained globe page to ``path``; return the path."""
    template = importlib.resources.files(
        "tropoi.representation.visual").joinpath(
        "terrain_globe.html").read_text(encoding="utf-8")
    payload = json.dumps(globe_payload(preview), allow_nan=False,
                         separators=(",", ":"))
    # Keep the JSON inert inside its <script> element.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    title = (preview.title.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))
    page = (template.replace("__PAGE_TITLE__", title)
            .replace("__TERRAIN_DATA__", payload))
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
