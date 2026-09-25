"""Declarative visualization specifications with no backend objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TypeAlias

import numpy as np

from tropoi.representation.visual.fields import ScalarGridField, SphericalHarmonicField
from tropoi.representation.visual.normalization import NormalizationPolicy


class SpectralEncoding(str, Enum):
    """Supported visual encodings for complex spectral coefficients."""

    PHASE_MAGNITUDE = "phase-magnitude"
    MAGNITUDE = "magnitude"


@dataclass(frozen=True)
class ScalarMapSpec:
    field: ScalarGridField
    title: str
    time_index: int = 0
    units: str | None = None
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    view: str = "equirectangular"
    central_longitude: float | None = None
    color_policy: str = "viridis"
    normalization_group: str | None = None

    def __post_init__(self) -> None:
        self.field.values_at(self.time_index)
        if not self.title:
            raise ValueError("map title must be nonempty")
        if not isinstance(self.view, str) or not self.view:
            raise ValueError("map view identifier must be nonempty")
        if self.central_longitude is not None and not np.isfinite(
                self.central_longitude):
            raise ValueError("central longitude must be finite")
        if not self.color_policy:
            raise ValueError("color policy must be nonempty")
        if (self.normalization_group is not None and
                (not isinstance(self.normalization_group, str) or
                 not self.normalization_group.strip())):
            raise ValueError("normalization group must be a nonempty string")

    @property
    def display_units(self) -> str:
        return self.field.units if self.units is None else self.units


@dataclass(frozen=True)
class SpectralCoefficientMapSpec:
    """A fixed-range triangular coefficient map with explicit complex encoding.

    The canonical phase-magnitude encoding uses the argument of the stored
    coefficient with no implicit sign flip. Hue spans the fixed ``[-pi, pi)``
    phase domain; timeline-relative amplitude dB controls saturation.
    """

    field: SphericalHarmonicField
    title: str
    time_index: int = 0
    units: str | None = None
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.logarithmic_magnitude)
    color_policy: str = "viridis"
    normalization_group: str | None = None
    encoding: SpectralEncoding | str = SpectralEncoding.PHASE_MAGNITUDE
    phase_offset_radians: float = 0.0
    magnitude_floor_db: float = -60.0

    def __post_init__(self) -> None:
        self.field.coefficients_at(self.time_index)
        if not self.title:
            raise ValueError("coefficient-map title must be nonempty")
        try:
            encoding = SpectralEncoding(self.encoding)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"unsupported spectral encoding {self.encoding!r}") from err
        if not math.isfinite(self.phase_offset_radians):
            raise ValueError("spectral phase offset must be finite")
        if (not math.isfinite(self.magnitude_floor_db) or
                self.magnitude_floor_db >= 0.0):
            raise ValueError("spectral magnitude dB floor must be finite and negative")
        if (self.normalization_group is not None and
                (not isinstance(self.normalization_group, str) or
                 not self.normalization_group.strip())):
            raise ValueError("normalization group must be a nonempty string")
        object.__setattr__(self, "encoding", encoding)

    @property
    def display_units(self) -> str:
        return self.field.units if self.units is None else self.units

    @property
    def encoding_label(self) -> str:
        if self.encoding is SpectralEncoding.PHASE_MAGNITUDE:
            return (
                "Hue = phase; saturation = relative magnitude "
                f"[{self.magnitude_floor_db:g}, 0] dB")
        return "Color = |C_lm|"

    @property
    def convention_metadata(self) -> dict[str, object]:
        """Basis and palette conventions needed to interpret coefficient color."""
        return {
            "encoding": self.encoding.value,
            "phase_definition": "arg(C_lm)",
            "phase_domain": "[-pi, pi)",
            "phase_offset_radians": float(self.phase_offset_radians),
            "magnitude_floor_db": float(self.magnitude_floor_db),
            "coefficient_normalization": self.field.normalization,
            "coefficient_layout": self.field.layout,
            "longitude_origin_radians": float(
                self.field.longitude_origin_radians),
            "magnitude_mapping": (
                "amplitude dB relative to timeline maximum, mapped to saturation"),
        }


@dataclass(frozen=True)
class StreamlineMapSpec:
    latitudes: np.ndarray
    longitudes: np.ndarray
    zonal_velocity: np.ndarray
    meridional_velocity: np.ndarray
    radius: float
    title: str
    units: str = "m/s"
    density: float = 1.5
    color_policy: str = "viridis"
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    normalization_group: str | None = None

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitudes)
        lon = np.asarray(self.longitudes)
        u = np.asarray(self.zonal_velocity)
        v = np.asarray(self.meridional_velocity)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("streamline coordinates must be one-dimensional")
        if u.shape != v.shape or u.shape != (lat.size, lon.size):
            raise ValueError("streamline velocities must have shape (lat, lon)")
        if lat.size > 1 and not np.all(np.diff(lat) < 0.0):
            raise ValueError("streamline latitudes must be north-to-south")
        if lon.size > 1 and not np.all(np.diff(lon) > 0.0):
            raise ValueError("streamline longitudes must be increasing")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("streamline radius must be finite and positive")
        if (self.normalization_group is not None and
                (not isinstance(self.normalization_group, str) or
                 not self.normalization_group.strip())):
            raise ValueError("normalization group must be a nonempty string")
        object.__setattr__(self, "latitudes", lat)
        object.__setattr__(self, "longitudes", lon)
        object.__setattr__(self, "zonal_velocity", u)
        object.__setattr__(self, "meridional_velocity", v)


@dataclass(frozen=True)
class TextPanelSpec:
    text: str
    font_family: str = "monospace"
    font_size: float = 11.0
    horizontal_alignment: str = "center"
    color: str = "k"
    font_weight: str = "normal"


@dataclass(frozen=True)
class LineSeriesSpec:
    x: np.ndarray
    y: np.ndarray
    label: str
    color: str = "k"
    line_style: str = "-"
    line_width: float = 1.5
    #: A marker per sample; with ``draw_line=False`` the series is drawn as
    #: markers only, which is how values that exist only at saved snapshots
    #: must be shown (joining them would suggest a per-step measurement).
    marker: str | None = None
    marker_size: float = 5.0
    draw_line: bool = True

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("line-series x and y must be equal-length 1D arrays")
        if not self.draw_line and self.marker is None:
            raise ValueError("a line series without a line needs a marker")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@dataclass(frozen=True)
class LinePanelSpec:
    series: tuple[LineSeriesSpec, ...]
    title: str
    x_label: str
    y_label: str
    y_limits: tuple[float, float] | None = None
    show_grid: bool = True
    show_legend: bool = True
    #: ``"linear"``, ``"log"`` or ``"symlog"`` (signed values spanning many
    #: decades, such as relative drifts; ``y_linear_threshold`` sets the
    #: linear band around zero).
    y_scale: str = "linear"
    y_linear_threshold: float | None = None
    x_limits: tuple[float, float] | None = None
    #: Short notes drawn inside the axes (e.g. "mass drift is exactly 0").
    notes: tuple[str, ...] = ()
    #: A Matplotlib legend location, or ``"below"`` for a legend outside
    #: the axes (it can then never hide data).
    legend_location: str = "best"

    def __post_init__(self) -> None:
        if self.y_scale not in ("linear", "log", "symlog"):
            raise ValueError(f"unsupported y scale {self.y_scale!r}")
        if self.y_scale == "symlog" and (
                self.y_linear_threshold is None or
                not self.y_linear_threshold > 0.0):
            raise ValueError("a symlog axis needs a positive linear threshold")
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "notes", tuple(self.notes))


# ---------------------------------------------------------------------------
# Layered maps: a scalar background, contour lines and one vector overlay on
# one equirectangular axis, plus figure-level keys shared by a whole
# normalization group.
# ---------------------------------------------------------------------------

def _check_group(group) -> None:
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError("normalization group must be a nonempty string")


@dataclass(frozen=True)
class ScalarLayer:
    """A filled scalar background, drawn cell-centred on its own samples."""

    field: ScalarGridField
    time_index: int = 0
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    normalization_group: str | None = None
    color_policy: str = "viridis"
    #: Colour-key label; defaults to ``"<field name> (<units>)"``.
    label: str | None = None

    def __post_init__(self) -> None:
        self.field.values_at(self.time_index)
        _check_group(self.normalization_group)

    @property
    def display_label(self) -> str:
        if self.label is not None:
            return self.label
        return f"{self.field.name} ({self.field.units})"


@dataclass(frozen=True)
class ContourLayer:
    """Contour lines of a scalar at explicit levels (never automatic)."""

    field: ScalarGridField
    levels: tuple[float, ...]
    time_index: int = 0
    color: str = "k"
    line_width: float = 0.6
    line_style: str = "solid"
    #: Text describing what the lines are, for keys and captions.
    label: str | None = None

    def __post_init__(self) -> None:
        self.field.values_at(self.time_index)
        levels = tuple(float(level) for level in self.levels)
        if not levels or not all(np.isfinite(levels)):
            raise ValueError("contour levels must be a nonempty finite sequence")
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("contour levels must be strictly increasing")
        object.__setattr__(self, "levels", levels)


VECTOR_STYLES = ("streamlines", "arrows")


@dataclass(frozen=True)
class VectorLayer:
    """A horizontal wind drawn as instantaneous streamlines or as arrows.

    ``zonal`` and ``meridional`` are physical velocity components (m/s) on
    ``(latitudes, longitudes)`` (north-to-south, endpoint-exclusive
    ``[0, 2*pi)``). On the equirectangular map the direction of motion is
    ``(u / cos(lat), v)``; the renderer uses that geometry for both styles
    and keeps arrow length and line width proportional to the physical
    speed. Streamlines are instantaneous; they are not particle trajectories.
    """

    latitudes: np.ndarray
    longitudes: np.ndarray
    zonal: np.ndarray
    meridional: np.ndarray
    radius: float
    style: str = "streamlines"
    #: ``"speed"`` colours by speed through ``normalization``; ``None`` uses
    #: the fixed ``color``.
    color_by: str | None = None
    color: str = "k"
    color_policy: str = "viridis"
    #: ``"speed"`` scales line width with speed through ``normalization``;
    #: ``None`` uses ``line_width``.
    width_by: str | None = "speed"
    line_width: float = 0.8
    line_width_range: tuple[float, float] = (0.25, 1.6)
    density: float = 1.0
    arrow_size: float = 0.7
    #: Longest single streamline, in axes units (Matplotlib's ``maxlength``).
    #: Shorter segments stagger the direction arrows along long, straight
    #: streamlines instead of stacking them at one longitude.
    max_length: float = 4.0
    #: Seed streamlines from this many points of a fixed low-discrepancy
    #: (R2) sequence instead of Matplotlib's regular grid. With a short
    #: ``max_length`` this staggers the direction arrows; the sequence is
    #: deterministic, so renders are reproducible. ``None``: grid seeding.
    seed_count: int | None = None
    #: Streamlines are not integrated poleward of this latitude (degrees),
    #: where the map metric 1/cos(lat) is singular.
    polar_limit_degrees: float = 85.0
    arrow_stride: int | None = None
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    normalization_group: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitudes, dtype=np.float64)
        lon = np.asarray(self.longitudes, dtype=np.float64)
        u = np.asarray(self.zonal, dtype=np.float64)
        v = np.asarray(self.meridional, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("vector coordinates must be one-dimensional")
        if u.shape != v.shape or u.shape != (lat.size, lon.size):
            raise ValueError("vector components must have shape (lat, lon)")
        if lat.size > 1 and not np.all(np.diff(lat) < 0.0):
            raise ValueError("vector latitudes must be north-to-south")
        if lon.size > 1 and not np.all(np.diff(lon) > 0.0):
            raise ValueError("vector longitudes must be increasing")
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("vector components must be finite")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("vector radius must be finite and positive")
        if self.style not in VECTOR_STYLES:
            raise ValueError(f"vector style must be one of {VECTOR_STYLES}")
        if self.color_by not in (None, "speed"):
            raise ValueError("vectors can only be coloured by 'speed'")
        if self.width_by not in (None, "speed"):
            raise ValueError("vector width can only scale with 'speed'")
        low, high = self.line_width_range
        if not 0.0 < low <= high:
            raise ValueError("line-width range must be positive and ordered")
        if not 0.0 < self.polar_limit_degrees <= 90.0:
            raise ValueError("polar limit must lie in (0, 90] degrees")
        if self.seed_count is not None and self.seed_count < 1:
            raise ValueError("seed count must be a positive integer")
        if self.arrow_stride is not None and self.arrow_stride < 1:
            raise ValueError("arrow stride must be a positive integer")
        _check_group(self.normalization_group)
        for name, value in (("latitudes", lat), ("longitudes", lon),
                            ("zonal", u), ("meridional", v)):
            object.__setattr__(self, name, value)

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.zonal, self.meridional)


@dataclass(frozen=True)
class LayeredMapSpec:
    """One equirectangular map: optional background, contours and vectors."""

    title: str
    background: ScalarLayer | None = None
    contours: tuple[ContourLayer, ...] = ()
    vectors: VectorLayer | None = None
    #: Draw a colour bar for the background beside this panel. Figures that
    #: share one key across panels use a :class:`ColorKeySpec` instead.
    colorbar: bool = True
    label_longitudes: bool = True
    label_latitudes: bool = True
    #: Text drawn in the top-left corner inside the map.
    corner_text: str | None = None

    def __post_init__(self) -> None:
        contours = tuple(self.contours)
        if self.background is None and not contours and self.vectors is None:
            raise ValueError("a layered map needs at least one layer")
        if not isinstance(self.title, str):
            raise TypeError("map title must be a string")
        object.__setattr__(self, "contours", contours)


@dataclass(frozen=True)
class ColorKeySpec:
    """A colour key shared by every panel of one normalization group.

    Its limits are filled in from the group when normalizations are
    resolved, so the key always shows exactly the scale the panels used.
    """

    normalization_group: str
    label: str
    color_policy: str = "viridis"
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    orientation: str = "horizontal"
    ticks: tuple[float, ...] | None = None
    #: Fraction of the panel's short side the bar occupies.
    thickness: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.normalization_group, str):
            raise ValueError("a colour key must name its normalization group")
        _check_group(self.normalization_group)
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(
                "colour-key orientation must be horizontal or vertical")


@dataclass(frozen=True)
class LineWidthKeySpec:
    """A key for vector line width scaled by speed within one group."""

    normalization_group: str
    label: str
    values: tuple[float, ...]
    line_width_range: tuple[float, float] = (0.25, 1.6)
    color: str = "k"
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)

    def __post_init__(self) -> None:
        if not isinstance(self.normalization_group, str):
            raise ValueError("a line-width key must name its group")
        _check_group(self.normalization_group)
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError("a line-width key needs sample values")
        object.__setattr__(self, "values", values)


PanelSpec: TypeAlias = (
    ScalarMapSpec | SpectralCoefficientMapSpec | StreamlineMapSpec |
    TextPanelSpec | LinePanelSpec | LayeredMapSpec | ColorKeySpec |
    LineWidthKeySpec)


@dataclass(frozen=True)
class PanelPlacement:
    panel: PanelSpec
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1


@dataclass(frozen=True)
class PanelGroupSpec:
    """A labeled rectangular group of scientifically related panels.

    The group describes layout and presentation only. Scientific role names
    and the decision about which fields belong together remain adapter-owned.
    """

    title: str
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    separator_before: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("panel-group title must be a nonempty string")
        if (not isinstance(self.separator_before, bool) or
                any(not isinstance(value, int) or isinstance(value, bool)
                    for value in (self.row, self.column, self.row_span,
                                  self.column_span))):
            raise TypeError("panel-group placement values must be integers")
        if (self.row < 0 or self.column < 0 or self.row_span < 1 or
                self.column_span < 1):
            raise ValueError("panel-group placement must be positive")

    def contains(self, placement: PanelPlacement) -> bool:
        """Whether a panel placement lies completely inside this group."""
        return (
            placement.row >= self.row and
            placement.column >= self.column and
            placement.row + placement.row_span <= self.row + self.row_span and
            placement.column + placement.column_span <=
            self.column + self.column_span)


# Concise public spelling for callers that do not need the Spec suffix.
PanelGroup = PanelGroupSpec


@dataclass(frozen=True)
class FigureSpec:
    panels: tuple[PanelPlacement, ...]
    rows: int
    columns: int
    size_inches: tuple[float, float]
    dpi: int = 200
    width_ratios: tuple[float, ...] | None = None
    height_ratios: tuple[float, ...] | None = None
    tight_layout: bool = True
    panel_groups: tuple[PanelGroupSpec, ...] = ()
    #: Base font size in points for every text element (``None`` keeps the
    #: backend default, as all historical products do).
    base_font_size: float | None = None
    #: Use a constrained layout engine instead of ``tight_layout``.
    constrained_layout: bool = False

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("figure layout dimensions must be positive")
        if self.dpi < 1:
            raise ValueError("figure dpi must be positive")
        if self.width_ratios is not None and len(self.width_ratios) != self.columns:
            raise ValueError("width-ratio count must equal figure columns")
        if self.height_ratios is not None and len(self.height_ratios) != self.rows:
            raise ValueError("height-ratio count must equal figure rows")
        groups = tuple(self.panel_groups)
        occupied: set[tuple[int, int]] = set()
        for group in groups:
            if not isinstance(group, PanelGroupSpec):
                raise TypeError("figure panel_groups must contain PanelGroupSpec")
            if (group.row + group.row_span > self.rows or
                    group.column + group.column_span > self.columns):
                raise ValueError(
                    f"panel group is outside figure grid: {group}")
            cells = {
                (row, column)
                for row in range(group.row, group.row + group.row_span)
                for column in range(
                    group.column, group.column + group.column_span)}
            if occupied.intersection(cells):
                raise ValueError("panel groups must not overlap")
            occupied.update(cells)
        for placement in self.panels:
            if (placement.row < 0 or placement.column < 0 or
                    placement.row_span < 1 or placement.column_span < 1 or
                    placement.row + placement.row_span > self.rows or
                    placement.column + placement.column_span > self.columns):
                raise ValueError(f"panel placement is outside figure grid: {placement}")
        for group in groups:
            if not any(group.contains(placement) for placement in self.panels):
                raise ValueError(
                    f"panel group contains no panels: {group.title!r}")
        object.__setattr__(self, "panel_groups", groups)
