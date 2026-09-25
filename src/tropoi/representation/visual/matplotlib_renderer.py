"""Matplotlib implementation of the small visualization backend protocol."""
from __future__ import annotations

import contextlib
import os
import pathlib
import uuid

import numpy as np

from tropoi.representation.visual.complex_encoding import phase_magnitude_hsv
from tropoi.representation.visual.normalization import NormalizationKind
from tropoi.representation.visual.specs import (
    ColorKeySpec, ContourLayer, FigureSpec, LayeredMapSpec, LinePanelSpec,
    LineWidthKeySpec, PanelPlacement, ScalarLayer, ScalarMapSpec,
    SpectralCoefficientMapSpec, SpectralEncoding, StreamlineMapSpec,
    TextPanelSpec, VectorLayer)


_SEMANTIC_COLORS = {
    "signed": "RdBu_r",
    "magnitude": "viridis",
    "sequential": "viridis",
}


class MatplotlibRenderer:
    """Render visualization specifications with Matplotlib's non-GUI backend."""

    def __init__(self) -> None:
        import matplotlib
        matplotlib.use("Agg")

    def render_scalar_map(self, specification: ScalarMapSpec,
                          output_path: pathlib.Path | str, *,
                          metadata: dict | None = None,
                          dpi: int = 200) -> pathlib.Path:
        figure = FigureSpec(
            panels=(PanelPlacement(specification, 0, 0),),
            rows=1, columns=1, size_inches=(12.0, 6.0), dpi=dpi)
        return self.render_figure(figure, output_path, metadata=metadata)

    def render_spectral_coefficient_map(
            self, specification: SpectralCoefficientMapSpec,
            output_path: pathlib.Path | str, *, metadata: dict | None = None,
            dpi: int = 200) -> pathlib.Path:
        figure = FigureSpec(
            panels=(PanelPlacement(specification, 0, 0),),
            rows=1, columns=1, size_inches=(8.0, 6.0), dpi=dpi)
        return self.render_figure(figure, output_path, metadata=metadata)

    def render_streamline_map(
            self, specification: StreamlineMapSpec,
            output_path: pathlib.Path | str, *, metadata: dict | None = None,
            dpi: int = 200) -> pathlib.Path:
        figure = FigureSpec(
            panels=(PanelPlacement(specification, 0, 0),),
            rows=1, columns=1, size_inches=(12.0, 6.0), dpi=dpi)
        return self.render_figure(figure, output_path, metadata=metadata)

    def render_figure(self, specification: FigureSpec,
                      output_path: pathlib.Path | str, *,
                      metadata: dict | None = None) -> pathlib.Path:
        import matplotlib.pyplot as plt

        output = pathlib.Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(_font_rc(specification.base_font_size)):
            return self._render_figure(specification, output, metadata)

    def _render_figure(self, specification: FigureSpec,
                       output: pathlib.Path, metadata: dict | None
                       ) -> pathlib.Path:
        import matplotlib.pyplot as plt

        figure = plt.figure(
            figsize=specification.size_inches,
            layout="constrained" if specification.constrained_layout else None)
        try:
            header_rows = sorted({group.row
                                  for group in specification.panel_groups})
            source_height_ratios = (
                specification.height_ratios or
                (1.0,) * specification.rows)
            mean_row_ratio = sum(source_height_ratios) / len(
                source_height_ratios)
            expanded_height_ratios = []
            for row, ratio in enumerate(source_height_ratios):
                if row in header_rows:
                    expanded_height_ratios.append(0.08 * mean_row_ratio)
                expanded_height_ratios.append(ratio)

            grid = figure.add_gridspec(
                specification.rows + len(header_rows), specification.columns,
                width_ratios=specification.width_ratios,
                height_ratios=expanded_height_ratios)
            rendered_groups = []
            for group in specification.panel_groups:
                header_row = group.row + sum(
                    candidate < group.row for candidate in header_rows)
                axes = figure.add_subplot(grid[
                    header_row,
                    group.column:group.column + group.column_span])
                axes.axis("off")
                axes.text(
                    0.5, 0.45, group.title, ha="center", va="center",
                    fontsize=11.0, fontweight="semibold", color="#333333")
                rendered_groups.append((group, axes))

            rendered_panels = []
            phase_panels = []
            for placement in specification.panels:
                first_row = placement.row + sum(
                    candidate <= placement.row for candidate in header_rows)
                last_source_row = placement.row + placement.row_span - 1
                last_row = last_source_row + sum(
                    candidate <= last_source_row for candidate in header_rows)
                axes = figure.add_subplot(grid[
                    first_row:last_row + 1,
                    placement.column:placement.column + placement.column_span])
                self._render_panel(figure, axes, placement.panel)
                rendered_panels.append((placement, axes))
                if (isinstance(placement.panel, SpectralCoefficientMapSpec) and
                        placement.panel.encoding is
                        SpectralEncoding.PHASE_MAGNITUDE):
                    phase_panels.append(placement.panel)
            if specification.tight_layout and not specification.constrained_layout:
                phase_margin = min(
                    0.14, 0.7 / figure.get_figheight()) if phase_panels else 0.0
                figure.tight_layout(
                    rect=(0.0, phase_margin, 1.0, 1.0)
                    if phase_panels else None)
            self._render_panel_groups(
                figure, specification, rendered_panels, rendered_groups)
            if phase_panels:
                self._render_phase_legend(figure, phase_panels)
            self._save_atomic(
                figure, output, dpi=specification.dpi, metadata=metadata)
        finally:
            plt.close(figure)
        return output

    @staticmethod
    def _render_panel_groups(
            figure, specification, rendered_panels, rendered_groups) -> None:
        """Draw requested generic inter-group dividers after layout."""
        if not specification.panel_groups:
            return

        from matplotlib.lines import Line2D
        from matplotlib.transforms import Bbox

        # Layout must be settled before axes positions can anchor separators.
        figure.canvas.draw()
        bounds = []
        heading_axes = dict(rendered_groups)
        for group in specification.panel_groups:
            axes = [axes for placement, axes in rendered_panels
                    if group.contains(placement)]
            if not axes:  # FigureSpec validation normally makes this impossible.
                continue
            if group in heading_axes:
                axes.append(heading_axes[group])
            box = Bbox.union([axes.get_position() for axes in axes])
            bounds.append((group, box))

        for group, box in bounds:
            if not group.separator_before:
                continue
            previous = [
                (other, other_box) for other, other_box in bounds
                if (other.row < group.row + group.row_span and
                    group.row < other.row + other.row_span and
                    other.column + other.column_span <= group.column)]
            if previous:
                _, previous_box = max(
                    previous, key=lambda member: member[0].column +
                    member[0].column_span)
                x = (previous_box.x1 + box.x0) / 2.0
                y0 = min(previous_box.y0, box.y0)
                y1 = max(previous_box.y1, box.y1)
            else:
                x = box.x0 - 0.012
                y0, y1 = box.y0, box.y1
            figure.add_artist(Line2D(
                (x, x), (y0, y1), transform=figure.transFigure,
                color="#666666", linewidth=0.8, alpha=0.4,
                solid_capstyle="round"))

    def _render_panel(self, figure, axes, panel) -> None:
        if isinstance(panel, ScalarMapSpec):
            self._render_scalar_panel(figure, axes, panel)
        elif isinstance(panel, SpectralCoefficientMapSpec):
            self._render_spectral_panel(figure, axes, panel)
        elif isinstance(panel, StreamlineMapSpec):
            self._render_streamline_panel(figure, axes, panel)
        elif isinstance(panel, TextPanelSpec):
            axes.axis("off")
            x = {"left": 0.0, "right": 1.0}.get(
                panel.horizontal_alignment, 0.5)
            axes.text(
                x, 0.5, panel.text, ha=panel.horizontal_alignment,
                va="center", fontfamily=panel.font_family,
                fontsize=panel.font_size, color=panel.color,
                fontweight=panel.font_weight, transform=axes.transAxes)
        elif isinstance(panel, LinePanelSpec):
            self._render_line_panel(axes, panel)
        elif isinstance(panel, LayeredMapSpec):
            self._render_layered_map(figure, axes, panel)
        elif isinstance(panel, ColorKeySpec):
            self._render_color_key(figure, axes, panel)
        elif isinstance(panel, LineWidthKeySpec):
            self._render_width_key(axes, panel)
        else:  # pragma: no cover - guarded by the specification union
            raise TypeError(f"unsupported panel specification {type(panel).__name__}")

    @staticmethod
    def _mpl_normalization(policy, values):
        from matplotlib.colors import LogNorm, Normalize

        resolved = policy.resolve(values)
        if resolved.kind is NormalizationKind.LOG_MAGNITUDE:
            return resolved, LogNorm(vmin=resolved.vmin, vmax=resolved.vmax)
        return resolved, Normalize(vmin=resolved.vmin, vmax=resolved.vmax)

    @staticmethod
    def _color_map(identifier: str):
        import matplotlib.pyplot as plt

        name = _SEMANTIC_COLORS.get(identifier, identifier)
        if ":" in name:
            # "name:low:high" keeps only that fraction of a colour map, e.g.
            # a lighter range under dark overlay lines.
            from matplotlib.colors import LinearSegmentedColormap
            base, low, high = name.split(":")
            samples = plt.get_cmap(base)(np.linspace(float(low), float(high),
                                                     256))
            return LinearSegmentedColormap.from_list(name, samples)
        return plt.get_cmap(name).copy()

    def _render_scalar_panel(self, figure, axes, spec: ScalarMapSpec) -> None:
        if spec.view != "equirectangular":
            raise NotImplementedError(
                f"the Matplotlib backend does not support map view {spec.view!r}")
        if spec.central_longitude not in (None, 0, 0.0):
            raise NotImplementedError(
                "the initial Matplotlib backend supports central_longitude=0 only")
        values = np.asarray(spec.field.values_at(spec.time_index))
        _, norm = self._mpl_normalization(spec.normalization, values)
        cmap = self._color_map(spec.color_policy)

        # Source rows are north-to-south.  imshow with origin='lower' expects
        # south-to-north, so reverse exactly once at the renderer boundary.
        south_to_north = np.flip(values, axis=0)
        lon = np.rad2deg(spec.field.longitudes)
        lat = np.rad2deg(spec.field.latitudes)
        lon_step = (lon[1] - lon[0]) if lon.size > 1 else 360.0
        image = axes.imshow(
            south_to_north,
            extent=(float(lon[0]), float(lon[-1] + lon_step),
                    float(lat[-1]), float(lat[0])),
            cmap=cmap, norm=norm, aspect="equal", origin="lower")
        axes.set_title(spec.title)
        axes.set_xlabel("Longitude (deg)")
        axes.set_ylabel("Latitude (deg)")
        figure.colorbar(
            image, ax=axes, orientation="horizontal", pad=0.1,
            fraction=0.05, aspect=30, label=spec.display_units)

    def _render_spectral_panel(
            self, figure, axes, spec: SpectralCoefficientMapSpec) -> None:
        coefficients = spec.field.coefficients_at(spec.time_index)
        magnitude = np.abs(coefficients)
        valid = spec.field.valid_mask

        if spec.encoding is SpectralEncoding.PHASE_MAGNITUDE:
            from matplotlib.colors import hsv_to_rgb

            hsv = phase_magnitude_hsv(
                coefficients, spec.normalization, valid_mask=valid,
                phase_offset_radians=spec.phase_offset_radians,
                magnitude_floor_db=spec.magnitude_floor_db)
            display = hsv_to_rgb(hsv)
            display[~valid] = (0.85, 0.85, 0.85)
            axes.imshow(
                display, origin="lower", interpolation="nearest",
                aspect="auto",
                extent=(-0.5, spec.field.l_max + 0.5,
                        -0.5, spec.field.l_max + 0.5))
            axes.set_facecolor("#f2f2f2")
            self._label_spectral_axes(axes, spec)
            return

        resolved, norm = self._mpl_normalization(
            spec.normalization, magnitude[valid])
        display = np.where(valid, np.maximum(magnitude, resolved.vmin), np.nan)
        display = np.ma.masked_where(~valid, display)
        cmap = self._color_map(spec.color_policy)
        cmap.set_bad("#d9d9d9")
        image = axes.imshow(
            display, origin="lower", interpolation="nearest", aspect="auto",
            extent=(-0.5, spec.field.l_max + 0.5,
                    -0.5, spec.field.l_max + 0.5),
            cmap=cmap, norm=norm)
        self._label_spectral_axes(axes, spec)
        unit_suffix = f" [{spec.display_units}]" if spec.display_units else ""
        figure.colorbar(
            image, ax=axes, orientation="vertical",
            label=f"Coefficient magnitude{unit_suffix}")

    @staticmethod
    def _label_spectral_axes(axes, spec: SpectralCoefficientMapSpec) -> None:
        axes.set_title(spec.title)
        axes.set_xlabel("Spherical-harmonic order m")
        axes.set_ylabel("Spherical-harmonic degree l")

    @staticmethod
    def _render_phase_legend(figure, panels) -> None:
        """Add one compact fixed-domain cyclic phase legend per figure."""
        from matplotlib.colors import hsv_to_rgb

        offsets = {panel.phase_offset_radians for panel in panels}
        if len(offsets) != 1:
            raise ValueError(
                "phase-magnitude panels in one figure must share a phase offset")
        offset = offsets.pop()
        phase = np.linspace(-np.pi, np.pi, 512, endpoint=False)[None, :]
        hue = ((phase + offset + np.pi) % (2.0 * np.pi)) / (2.0 * np.pi)
        hsv = np.stack(
            (hue, np.ones_like(hue), np.ones_like(hue)), axis=-1)
        legend_width = min(0.5, 3.0 / figure.get_figwidth())
        legend_height = min(0.04, 0.16 / figure.get_figheight())
        legend_bottom = min(0.04, 0.24 / figure.get_figheight())
        legend = figure.add_axes((
            0.5 - legend_width / 2.0, legend_bottom,
            legend_width, legend_height))
        legend.imshow(
            hsv_to_rgb(hsv), aspect="auto", origin="lower",
            extent=(-np.pi, np.pi, 0.0, 1.0), interpolation="nearest")
        legend.set_yticks(())
        legend.set_xticks((-np.pi, 0.0, np.pi), ("−π", "0", "π"))
        legend.tick_params(axis="x", labelsize=8, length=2, pad=1)
        label = panels[0].encoding_label
        if offset:
            label += f"; palette phase offset = {offset:.6g} rad"
        legend.set_xlabel(label, fontsize=8, labelpad=1)
        for spine in legend.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.5)

    def _render_streamline_panel(
            self, figure, axes, spec: StreamlineMapSpec) -> None:
        latitudes = spec.latitudes[::-1]
        u_grid = spec.zonal_velocity[::-1, :]
        v_grid = spec.meridional_velocity[::-1, :]
        longitudes = spec.longitudes

        stride = 1
        if latitudes.size > 300 or longitudes.size > 500:
            stride = 2
        if latitudes.size > 600 or longitudes.size > 1000:
            stride = 4
        if stride > 1:
            latitudes = latitudes[::stride]
            longitudes = longitudes[::stride]
            u_grid = u_grid[::stride, ::stride]
            v_grid = v_grid[::stride, ::stride]

        cos_lat = np.maximum(np.cos(latitudes), 1.0e-4)
        u_angular = u_grid / (spec.radius * cos_lat[:, None])
        v_angular = v_grid / spec.radius
        speed = np.sqrt(u_grid * u_grid + v_grid * v_grid)
        _, norm = self._mpl_normalization(spec.normalization, speed)
        stream = axes.streamplot(
            np.rad2deg(longitudes), np.rad2deg(latitudes),
            u_angular, v_angular, color=speed,
            cmap=_SEMANTIC_COLORS.get(spec.color_policy, spec.color_policy),
            norm=norm, density=spec.density, linewidth=1, arrowsize=1.2)
        figure.colorbar(
            stream.lines, ax=axes, label=f"Flow Speed ({spec.units})",
            orientation="horizontal", pad=0.1, fraction=0.05, aspect=30)
        axes.set_xlim(np.rad2deg(longitudes).min(), np.rad2deg(longitudes).max())
        axes.set_ylim(np.rad2deg(latitudes).min(), np.rad2deg(latitudes).max())
        axes.set_xlabel("Longitude (deg)")
        axes.set_ylabel("Latitude (deg)")
        axes.set_title(spec.title)
        axes.set_aspect("equal")

    @staticmethod
    def _render_line_panel(axes, spec: LinePanelSpec) -> None:
        for series in spec.series:
            axes.plot(
                series.x, series.y, color=series.color,
                linestyle=series.line_style if series.draw_line else "none",
                linewidth=series.line_width, marker=series.marker,
                markersize=series.marker_size, label=series.label)
        axes.set_xlabel(spec.x_label)
        axes.set_ylabel(spec.y_label)
        axes.set_title(spec.title)
        if spec.y_scale == "symlog":
            axes.set_yscale("symlog", linthresh=spec.y_linear_threshold)
        elif spec.y_scale == "log":
            axes.set_yscale("log")
        if spec.show_grid:
            axes.grid(True, alpha=0.3, linestyle="--")
        if spec.show_legend and spec.legend_location == "below":
            # Outside the axes, so no legend box can hide data.
            axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3),
                        fontsize="small", frameon=False, ncol=1,
                        borderaxespad=0.0)
        elif spec.show_legend:
            axes.legend(loc=spec.legend_location, fontsize="small")
        if spec.y_limits is not None:
            axes.set_ylim(*spec.y_limits)
        if spec.x_limits is not None:
            axes.set_xlim(*spec.x_limits)
        if spec.notes:
            axes.text(0.02, 0.96, "\n".join(spec.notes),
                      transform=axes.transAxes, ha="left", va="top",
                      fontsize="small", color="#444444")

    # -- layered maps ---------------------------------------------------

    def _render_layered_map(self, figure, axes, spec: LayeredMapSpec) -> None:
        mesh = None
        if spec.background is not None:
            mesh = self._draw_background(axes, spec.background)
        for layer in spec.contours:
            self._draw_contours(axes, layer)
        if spec.vectors is not None:
            if spec.vectors.style == "streamlines":
                self._draw_streamlines(axes, spec.vectors)
            else:
                self._draw_arrows(axes, spec.vectors)
        _format_map_axes(axes, label_longitudes=spec.label_longitudes,
                         label_latitudes=spec.label_latitudes)
        if spec.title:
            axes.set_title(spec.title)
        notes = []
        if spec.corner_text:
            notes.append(spec.corner_text)
        if spec.background is not None and not np.any(
                spec.background.field.values_at(spec.background.time_index)):
            notes.append("identically zero at this time")
        if notes:
            axes.text(0.012, 0.975, "\n".join(notes), transform=axes.transAxes,
                      ha="left", va="top", fontsize="small", zorder=5,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                            "edgecolor": "none", "alpha": 0.8})
        if mesh is not None and spec.colorbar:
            figure.colorbar(mesh, ax=axes, orientation="horizontal",
                            pad=0.12, fraction=0.05, aspect=30,
                            label=spec.background.display_label)

    def _draw_background(self, axes, layer: ScalarLayer):
        field = layer.field
        values = np.asarray(field.values_at(layer.time_index), dtype=np.float64)
        lat_s2n, lon_closed, closed = _closed_south_to_north(
            field.latitudes, field.longitudes, values)
        _, norm = self._mpl_normalization(layer.normalization, values)
        return axes.pcolormesh(
            _longitude_edges(lon_closed), _latitude_edges(lat_s2n), closed,
            cmap=self._color_map(layer.color_policy), norm=norm,
            shading="flat", rasterized=True, zorder=0)

    @staticmethod
    def _draw_contours(axes, layer: ContourLayer) -> None:
        import warnings

        field = layer.field
        values = np.asarray(field.values_at(layer.time_index), dtype=np.float64)
        lat_s2n, lon_closed, closed = _closed_south_to_north(
            field.latitudes, field.longitudes, values)
        with warnings.catch_warnings():
            # Levels outside the data range simply draw nothing.
            warnings.simplefilter("ignore", UserWarning)
            axes.contour(lon_closed, lat_s2n, closed, levels=layer.levels,
                         colors=layer.color, linewidths=layer.line_width,
                         linestyles=layer.line_style, zorder=3)

    def _vector_scale(self, layer: VectorLayer) -> float:
        resolved = layer.normalization.resolve(layer.speed)
        return max(float(resolved.vmax), np.finfo(np.float64).tiny)

    def _draw_streamlines(self, axes, layer: VectorLayer) -> None:
        lat, lon, u, v = _uniform_south_to_north_vectors(layer)
        speed = np.hypot(u, v)
        u_map, v_map = _map_direction(lat, u, v, layer.radius)
        polar = np.abs(lat) > layer.polar_limit_degrees
        u_map = np.ma.masked_array(u_map, mask=np.broadcast_to(
            polar[:, None], u_map.shape))
        v_map = np.ma.masked_array(v_map, mask=u_map.mask)
        options = {"density": layer.density, "arrowsize": layer.arrow_size,
                   "maxlength": layer.max_length, "zorder": 4}
        if layer.seed_count is not None:
            options["start_points"] = _r2_seed_points(
                layer.seed_count, layer.polar_limit_degrees)
        if layer.width_by == "speed":
            low, high = layer.line_width_range
            fraction = np.clip(speed / self._vector_scale(layer), 0.0, 1.0)
            options["linewidth"] = low + (high - low) * fraction
        else:
            options["linewidth"] = layer.line_width
        if layer.color_by == "speed":
            _, norm = self._mpl_normalization(layer.normalization, speed)
            options.update(color=speed, norm=norm,
                           cmap=self._color_map(layer.color_policy))
        else:
            options["color"] = layer.color
        axes.streamplot(lon, lat, u_map, v_map, **options)

    def _draw_arrows(self, axes, layer: VectorLayer) -> None:
        lat, lon, u, v = _uniform_south_to_north_vectors(layer)
        stride = layer.arrow_stride or max(1, int(round(lat.size / 18)))
        keep_lat = np.abs(lat) <= layer.polar_limit_degrees
        rows = np.flatnonzero(keep_lat)[::stride]
        columns = np.arange(0, lon.size - 1, stride)
        lat_s, lon_s = lat[rows], lon[columns]
        u_s, v_s = u[np.ix_(rows, columns)], v[np.ix_(rows, columns)]
        speed = np.hypot(u_s, v_s)
        u_map, v_map = _map_direction(lat_s, u_s, v_s, layer.radius)
        angle = np.arctan2(v_map, u_map)
        spacing = float(np.min(np.diff(lon_s))) if lon_s.size > 1 else 10.0
        length = 0.9 * spacing * np.clip(
            speed / self._vector_scale(layer), 0.0, 1.0)
        options = {"angles": "xy", "scale_units": "xy", "scale": 1.0,
                   "width": 0.0022, "zorder": 4}
        x, y = np.meshgrid(lon_s, lat_s)
        dx, dy = length * np.cos(angle), length * np.sin(angle)
        if layer.color_by == "speed":
            _, norm = self._mpl_normalization(layer.normalization, speed)
            axes.quiver(x, y, dx, dy, speed, norm=norm,
                        cmap=self._color_map(layer.color_policy), **options)
        else:
            axes.quiver(x, y, dx, dy, color=layer.color, **options)

    # -- keys -----------------------------------------------------------

    def _render_color_key(self, figure, axes, spec: ColorKeySpec) -> None:
        from matplotlib.cm import ScalarMappable

        axes.axis("off")
        resolved = spec.normalization.resolve(np.zeros(1))
        from matplotlib.colors import LogNorm, Normalize
        norm = (LogNorm if resolved.kind is NormalizationKind.LOG_MAGNITUDE
                else Normalize)(vmin=resolved.vmin, vmax=resolved.vmax)
        mappable = ScalarMappable(norm=norm,
                                  cmap=self._color_map(spec.color_policy))
        if spec.orientation == "horizontal":
            bounds = [0.04, 0.62 - spec.thickness / 2.0, 0.92, spec.thickness]
        else:
            bounds = [0.5 - spec.thickness / 2.0, 0.06, spec.thickness, 0.88]
        cax = axes.inset_axes(bounds)
        bar = figure.colorbar(mappable, cax=cax, orientation=spec.orientation,
                              ticks=spec.ticks)
        bar.set_label(spec.label)

    def _render_width_key(self, axes, spec: LineWidthKeySpec) -> None:
        axes.axis("off")
        vmax = max(float(spec.normalization.resolve(np.zeros(1)).vmax),
                   np.finfo(np.float64).tiny)
        low, high = spec.line_width_range
        count = len(spec.values)
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.text(0.5, 0.97, spec.label, ha="center", va="top",
                  transform=axes.transAxes)
        for index, value in enumerate(spec.values):
            x0 = 0.04 + index / count * 0.96
            width = low + (high - low) * min(max(value / vmax, 0.0), 1.0)
            axes.plot([x0, x0 + 0.12], [0.45, 0.45], color=spec.color,
                      linewidth=width, solid_capstyle="butt")
            axes.text(x0 + 0.14, 0.45, f"{value:g}", va="center",
                      ha="left", fontsize="small")

    @staticmethod
    def _save_atomic(figure, output: pathlib.Path, *, dpi: int,
                     metadata: dict | None) -> None:
        """Save to a same-directory temporary sibling, then atomically replace."""
        if not output.suffix:
            raise ValueError("render output path must have a file extension")
        temporary = output.with_name(
            f".{output.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}{output.suffix}")
        try:
            figure.savefig(temporary, dpi=dpi, metadata=metadata)
            # Windows' FlushFileBuffers requires a writable handle.
            with open(temporary, "rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        except BaseException:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise


# ---------------------------------------------------------------------------
# Map geometry helpers (NumPy only; unit-tested without drawing)
# ---------------------------------------------------------------------------

def _closed_south_to_north(latitudes, longitudes, values):
    """Degrees, south-to-north rows, plus one seam column at lon0 + 360.

    Inputs follow the field convention (radians, north-to-south rows,
    endpoint-exclusive periodic longitudes); the extra column repeats the
    first so contours and cells continue across the 0/360 seam.
    """
    lat = np.rad2deg(np.asarray(latitudes, dtype=np.float64))[::-1]
    lon = np.rad2deg(np.asarray(longitudes, dtype=np.float64))
    rows = np.asarray(values, dtype=np.float64)[::-1]
    lon_closed = np.append(lon, lon[0] + 360.0)
    closed = np.concatenate([rows, rows[:, :1]], axis=1)
    return lat, lon_closed, closed


def _longitude_edges(centres):
    """Cell edges centred on each (increasing) longitude sample."""
    centres = np.asarray(centres, dtype=np.float64)
    mids = 0.5 * (centres[1:] + centres[:-1])
    first = centres[0] - (mids[0] - centres[0])
    last = centres[-1] + (centres[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _latitude_edges(centres):
    """Cell edges centred between south-to-north samples, closed at the poles.

    Interior edges are midpoints; the outermost cells extend to +/-90 deg,
    so no sample is displaced and no polar strip is left unpainted.
    """
    centres = np.asarray(centres, dtype=np.float64)
    mids = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[-90.0], mids, [90.0]])


def _uniform_south_to_north_vectors(layer):
    """Vectors on an evenly spaced south-to-north grid with a closed seam.

    Matplotlib's streamplot needs evenly spaced coordinates. Longitudes
    already are; Gauss latitudes are not, so components are linearly
    interpolated along latitude onto as many evenly spaced rows. This
    resampling is for display only; no statistic is computed from it.
    """
    lat = np.rad2deg(layer.latitudes)[::-1]
    lon = np.rad2deg(layer.longitudes)
    u = layer.zonal[::-1]
    v = layer.meridional[::-1]
    if lat.size > 2 and not np.allclose(np.diff(lat), lat[1] - lat[0],
                                        rtol=1e-9, atol=1e-12):
        even = np.linspace(lat[0], lat[-1], lat.size)
        u = np.stack([np.interp(even, lat, u[:, j]) for j in range(lon.size)],
                     axis=1)
        v = np.stack([np.interp(even, lat, v[:, j]) for j in range(lon.size)],
                     axis=1)
        lat = even
    lon = np.append(lon, lon[0] + 360.0)
    u = np.concatenate([u, u[:, :1]], axis=1)
    v = np.concatenate([v, v[:, :1]], axis=1)
    return lat, lon, u, v


def _r2_seed_points(count: int, polar_limit: float) -> np.ndarray:
    """``count`` (lon, lat) points of the R2 low-discrepancy sequence.

    Evenly spread in longitude and latitude between +/- ``polar_limit``
    degrees, with no randomness: identical inputs give identical seeds.
    """
    g = 1.32471795724474602596          # plastic number
    alpha = np.array([1.0 / g, 1.0 / g ** 2])
    n = np.arange(1, count + 1)[:, None]
    unit = (0.5 + n * alpha) % 1.0
    return np.column_stack([360.0 * unit[:, 0],
                            polar_limit * (2.0 * unit[:, 1] - 1.0)])


def _map_direction(lat_degrees, u, v, radius):
    """Angular rates (dlon/dt, dlat/dt) giving the direction on the map."""
    cos_lat = np.maximum(np.cos(np.deg2rad(lat_degrees)), 1.0e-4)
    return u / (radius * cos_lat[:, None]), v / radius


def _format_map_axes(axes, *, label_longitudes: bool,
                     label_latitudes: bool) -> None:
    axes.set_xlim(0.0, 360.0)
    axes.set_ylim(-90.0, 90.0)
    axes.set_aspect("equal")
    axes.set_xticks(np.arange(0.0, 361.0, 60.0))
    axes.set_yticks(np.arange(-90.0, 91.0, 30.0))
    axes.tick_params(length=2.5, width=0.5, pad=1.5)
    if label_longitudes:
        axes.set_xticklabels([f"{int(x)}°" for x in axes.get_xticks()])
        axes.set_xlabel("longitude (°E)")
    else:
        axes.set_xticklabels([])
    if label_latitudes:
        axes.set_yticklabels([
            "0°" if y == 0 else f"{abs(int(y))}°{'N' if y > 0 else 'S'}"
            for y in axes.get_yticks()])
    else:
        axes.set_yticklabels([])
    for spine in axes.spines.values():
        spine.set_linewidth(0.5)


def _font_rc(base: float | None) -> dict:
    """Matplotlib rc overrides for one base font size (empty: defaults)."""
    if base is None:
        return {}
    return {"font.size": base, "axes.titlesize": base + 0.5,
            "axes.labelsize": base, "xtick.labelsize": base - 0.5,
            "ytick.labelsize": base - 0.5, "legend.fontsize": base - 0.5,
            "axes.linewidth": 0.5}
