"""Layered-map specs, shared-scale resolution and the Matplotlib renderer.

CPU only: synthetic fields on a Gauss latitude grid exercise cell
registration, seam closure, vector geometry, key resolution and a full
render (Agg), including the minimum on-screen text size at README width.
"""
from __future__ import annotations

import numpy as np
import pytest

from tropoi.representation.visual import matplotlib_renderer as mpl
from tropoi.representation.visual.fields import ScalarGridField
from tropoi.representation.visual.matplotlib_renderer import MatplotlibRenderer
from tropoi.representation.visual.normalization import (NormalizationKind,
                                                        NormalizationPolicy)
from tropoi.representation.visual.specs import (
    ColorKeySpec, ContourLayer, FigureSpec, LayeredMapSpec, LinePanelSpec,
    LineSeriesSpec, LineWidthKeySpec, PanelPlacement, ScalarLayer,
    TextPanelSpec, VectorLayer)
from tropoi.representation.visual.timeline import (
    resolve_figure_normalizations)

RADIUS = 6.0e6


def _gauss_grid(nlat=24, nlon=48):
    x, _ = np.polynomial.legendre.leggauss(nlat)
    lat = np.sort(np.arcsin(x))[::-1]
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)
    return lat, lon


def _layered(lat, lon, values, *, group="h", speed_group="speed",
             u=None, v=None, style="streamlines"):
    lat_grid, _ = np.meshgrid(lat, lon, indexing="ij")
    u = 10.0 * np.cos(lat_grid) if u is None else u
    v = np.zeros_like(u) if v is None else v
    field = ScalarGridField(values, lat, lon, "height", "m")
    return LayeredMapSpec(
        "map", background=ScalarLayer(field, normalization_group=group),
        contours=(ContourLayer(field, (float(values.mean()),)),),
        vectors=VectorLayer(lat, lon, u, v, RADIUS, style=style,
                            normalization_group=speed_group),
        colorbar=False)


def test_latitude_cells_are_centred_on_samples_and_close_at_the_poles():
    lat, _ = _gauss_grid()
    south_to_north = np.rad2deg(lat)[::-1]
    edges = mpl._latitude_edges(south_to_north)
    assert edges[0] == -90.0 and edges[-1] == 90.0
    inner = edges[1:-1]
    np.testing.assert_allclose(
        inner, 0.5 * (south_to_north[1:] + south_to_north[:-1]))
    assert np.all((edges[:-1] < south_to_north) &
                  (south_to_north < edges[1:]))


def test_longitude_cells_are_centred_and_the_seam_is_closed():
    lat, lon = _gauss_grid(nlat=4, nlon=8)
    values = np.arange(32.0).reshape(4, 8)
    lat_s2n, lon_closed, closed = mpl._closed_south_to_north(lat, lon, values)
    assert lon_closed[-1] == 360.0
    np.testing.assert_array_equal(closed[:, -1], closed[:, 0])
    np.testing.assert_array_equal(closed[0], np.append(values[-1],
                                                       values[-1, 0]))
    edges = mpl._longitude_edges(lon_closed)
    step = 360.0 / 8
    # Sample i sits at the centre of its cell (no half-cell shift).
    np.testing.assert_allclose(0.5 * (edges[1:] + edges[:-1]), lon_closed)
    assert edges[0] == pytest.approx(-step / 2.0)


def test_map_direction_follows_the_equirectangular_metric():
    lat_deg = np.array([-60.0, 0.0, 60.0])
    u = np.full((3, 2), 10.0)
    v = np.full((3, 2), 5.0)
    u_map, v_map = mpl._map_direction(lat_deg, u, v, RADIUS)
    np.testing.assert_allclose(u_map[:, 0],
                               10.0 / (RADIUS * np.cos(np.deg2rad(lat_deg))))
    np.testing.assert_allclose(v_map, 5.0 / RADIUS)


def test_gauss_vectors_are_resampled_exactly_for_linear_fields():
    lat, lon = _gauss_grid()
    lat_grid, _ = np.meshgrid(lat, lon, indexing="ij")
    u = 3.0 * np.rad2deg(lat_grid) + 1.0      # linear in latitude
    layer = VectorLayer(lat, lon, u, np.zeros_like(u), RADIUS)
    even_lat, closed_lon, u_even, _ = mpl._uniform_south_to_north_vectors(
        layer)
    np.testing.assert_allclose(np.diff(even_lat), np.diff(even_lat)[0])
    np.testing.assert_allclose(u_even[:, 0], 3.0 * even_lat + 1.0)
    assert closed_lon[-1] == pytest.approx(360.0)


def test_seed_points_are_deterministic_and_inside_the_polar_limit():
    first = mpl._r2_seed_points(200, 85.0)
    np.testing.assert_array_equal(first, mpl._r2_seed_points(200, 85.0))
    assert first.shape == (200, 2)
    assert first[:, 0].min() >= 0.0 and first[:, 0].max() < 360.0
    assert np.abs(first[:, 1]).max() <= 85.0


def test_shared_groups_resolve_to_one_scale_and_keys_show_it():
    lat, lon = _gauss_grid()
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    low = 5000.0 + 100.0 * np.sin(lat_grid)
    high = 5500.0 + 400.0 * np.cos(lon_grid)
    fast = 40.0 * np.cos(lat_grid)
    spec = FigureSpec((
        PanelPlacement(_layered(lat, lon, low), 0, 0),
        PanelPlacement(_layered(lat, lon, high, u=fast), 0, 1),
        PanelPlacement(ColorKeySpec("h", "height (m)"), 1, 0),
        PanelPlacement(LineWidthKeySpec("speed", "speed", (10.0, 20.0)), 1, 1),
    ), rows=2, columns=2, size_inches=(6.0, 4.0))
    resolved = resolve_figure_normalizations(spec)
    first, second, key, width_key = (p.panel for p in resolved.panels)
    for policy in (first.background.normalization,
                   second.background.normalization, key.normalization):
        assert (policy.vmin, policy.vmax) == (float(low.min()),
                                              float(high.max()))
    for policy in (first.vectors.normalization, second.vectors.normalization,
                   width_key.normalization):
        assert policy.vmax == float(fast.max())


def test_a_key_without_panel_data_is_an_error():
    spec = FigureSpec((PanelPlacement(ColorKeySpec("nothing", "x"), 0, 0),),
                      rows=1, columns=1, size_inches=(3.0, 1.0))
    with pytest.raises(ValueError, match="no panel data"):
        resolve_figure_normalizations(spec)


def test_layer_validation():
    lat, lon = _gauss_grid()
    field = ScalarGridField(np.zeros((lat.size, lon.size)), lat, lon, "x", "m")
    with pytest.raises(ValueError):
        LayeredMapSpec("empty")
    with pytest.raises(ValueError):
        ContourLayer(field, (2.0, 1.0))
    u = np.zeros((lat.size, lon.size))
    with pytest.raises(ValueError):
        VectorLayer(lat, lon, u, u, RADIUS, style="barbs")
    with pytest.raises(ValueError):
        LineSeriesSpec(np.arange(3.0), np.arange(3.0), "x", draw_line=False)
    with pytest.raises(ValueError):
        LinePanelSpec((), "t", "x", "y", y_scale="symlog")


def _full_figure():
    lat, lon = _gauss_grid()
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    height = 5500.0 + 300.0 * np.sin(lat_grid) + 40.0 * np.cos(3 * lon_grid)
    u = 15.0 * np.cos(lat_grid) + 5.0 * np.sin(2 * lon_grid)
    v = 4.0 * np.cos(2 * lon_grid) * np.cos(lat_grid)
    zero = ScalarGridField(np.zeros_like(height), lat, lon, "divergence",
                           "s^-1")
    t = np.linspace(0.0, 10.0, 50)
    return FigureSpec((
        PanelPlacement(TextPanelSpec("Title", font_family="sans-serif",
                                     horizontal_alignment="left"), 0, 0,
                       column_span=2),
        PanelPlacement(_layered(lat, lon, height, u=u, v=v), 1, 0),
        PanelPlacement(_layered(lat, lon, height, u=u, v=v, style="arrows"),
                       1, 1),
        PanelPlacement(LayeredMapSpec(
            "zero", background=ScalarLayer(
                zero, normalization=NormalizationPolicy.symmetric())), 2, 0),
        PanelPlacement(ColorKeySpec("h", "height (m)"), 2, 1),
        PanelPlacement(LinePanelSpec((
            LineSeriesSpec(t, -1e-7 * t, "energy (every step)"),
            LineSeriesSpec(t[::10], 1e-6 * t[::10], "Z (saved states)",
                           marker="o", draw_line=False)),
            "Drift", "day", "relative drift", y_scale="symlog",
            y_linear_threshold=1e-10, legend_location="below"), 3, 0,
            column_span=2),
    ), rows=4, columns=2, size_inches=(7.2, 7.0), dpi=80,
        height_ratios=(0.3, 1.8, 1.2, 2.0), base_font_size=8.0,
        constrained_layout=True)


def test_every_layer_type_renders_and_text_is_readable_at_readme_width(
        tmp_path, monkeypatch):
    captured = {}
    original = MatplotlibRenderer._save_atomic

    def capture(figure, output, *, dpi, metadata):
        captured["figure"] = figure
        return original(figure, output, dpi=dpi, metadata=metadata)

    monkeypatch.setattr(MatplotlibRenderer, "_save_atomic",
                        staticmethod(capture))
    spec = resolve_figure_normalizations(_full_figure())
    path = MatplotlibRenderer().render_figure(spec, tmp_path / "f.png")
    assert path.stat().st_size > 0
    figure = captured["figure"]
    texts = [text for text in figure.findobj(
        lambda artist: hasattr(artist, "get_fontsize") and
        hasattr(artist, "get_text")) if text.get_text().strip()]
    assert any("identically zero" in text.get_text() for text in texts)
    # At an 880 px README column every text must be at least 11 CSS px.
    width_in = figure.get_figwidth()
    smallest = min(text.get_fontsize() for text in texts)
    assert smallest * 880.0 / (72.0 * width_in) >= 11.0, smallest


def test_legacy_panels_keep_their_normalization_grouping():
    """Unnamed historical panels still group by placement across frames."""
    from tropoi.representation.visual.specs import ScalarMapSpec
    from tropoi.representation.visual.timeline import FigureTimeline
    lat, lon = _gauss_grid(8, 16)
    frames = []
    for scale in (1.0, 3.0):
        field = ScalarGridField(scale * np.ones((8, 16)), lat, lon, "x", "m")
        frames.append(FigureSpec((PanelPlacement(ScalarMapSpec(
            field, "x", normalization=NormalizationPolicy.symmetric()), 0,
            0),), rows=1, columns=1, size_inches=(4.0, 2.0)))
    timeline = FigureTimeline.from_figures((0.0, 1.0), frames,
                                           filename_prefix=None)
    resolved = timeline.resolve_normalizations()
    for frame in resolved.frames:
        policy = frame.specification.panels[0].panel.normalization
        assert policy.kind is NormalizationKind.SYMMETRIC
        assert (policy.vmin, policy.vmax) == (-3.0, 3.0)
