"""Public view objects for plotting saved runs.

A view says *what* to draw: which quantity fills a map, which contours and
vector overlay go on top, which saved times to show and which diagnostics
panels to add. It contains no data and no backend objects, and it is
immutable, so a figure recipe (for example the README's) can be written out
in full and pinned by a test: a later change of any default cannot alter a
recipe that spells out every field.

* :class:`Map`: one map (background quantity, contours, vectors).
* :class:`Overview`: one :class:`Map` repeated over saved times with shared
  colour and speed scales, an optional static map (terrain), and
  diagnostics panels (:class:`Drift`, :class:`Complexity`).
* :class:`Grid`: several different panels at one saved time.

``Simulation.plot`` and ``Snapshot.plot(view=...)`` accept these; the
``tropoi plot`` command builds them from its options.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import re

__all__ = [
    "Arrows",
    "Complexity",
    "Contours",
    "Drift",
    "Grid",
    "Map",
    "Overview",
    "Sigma",
    "Streamlines",
    "Style",
    "describe",
    "parse_time",
]


@dataclass(frozen=True)
class Sigma:
    """Select the full model level whose sigma is nearest ``value``.

    The chosen level's actual sigma is always shown. Sigma is p / p_s: it is
    not pressure and not height.
    """

    value: float

    def __post_init__(self) -> None:
        if not 0.0 < float(self.value) < 1.0:
            raise ValueError("sigma must lie strictly between 0 and 1")


@dataclass(frozen=True)
class Streamlines:
    """Instantaneous streamlines of a horizontal wind (not trajectories)."""

    vector: str = "wind"
    color_by: str | None = None          # None or "speed"
    color: str = "black"
    width_by: str | None = "speed"       # None or "speed"
    line_width_range: tuple[float, float] = (0.25, 1.6)
    density: float = 1.0
    arrow_size: float = 0.7
    #: Longest single streamline in axes units, and the number of fixed,
    #: evenly spread seed points (``None``: Matplotlib's grid seeding).
    #: Short streamlines from staggered seeds spread the direction arrows
    #: instead of stacking them where long straight lines have midpoints.
    max_length: float = 0.35
    seed_count: int | None = 220


@dataclass(frozen=True)
class Arrows:
    """Wind arrows: direction on the map, length proportional to speed."""

    vector: str = "wind"
    color_by: str | None = None
    color: str = "black"
    stride: int | None = None


@dataclass(frozen=True)
class Contours:
    """Contour lines of a quantity at explicit levels."""

    quantity: str
    levels: tuple[float, ...]
    color: str = "#3b3b3b"
    line_width: float = 0.6
    line_style: str = "solid"


@dataclass(frozen=True)
class Map:
    """One map: an optional filled quantity, contours and a vector overlay.

    ``background=None`` with ``vectors`` set gives a streamline-only (or
    arrow-only) map. ``color_policy``/``symmetric``/``limits`` override the
    quantity's display defaults; ``limits`` fixes the colour range.
    """

    background: str | None
    level: int | Sigma | None = None
    contours: tuple[Contours, ...] = ()
    vectors: Streamlines | Arrows | None = None
    color_policy: str | None = None
    symmetric: bool | None = None
    limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contours", tuple(self.contours))
        if self.background is None and not self.contours and (
                self.vectors is None):
            raise ValueError("a map needs a background, contours or vectors")


@dataclass(frozen=True)
class Drift:
    """Relative drift from the first saved time of conserved quantities.

    ``recorded`` names per-step columns of ``diagnostics/timeseries.csv``
    (drawn as lines); ``at_snapshots`` names quantities evaluated only at
    saved states (drawn as unconnected markers).
    """

    recorded: tuple[str, ...] = ()
    at_snapshots: tuple[str, ...] = ()
    #: Half-width of the linear band of the symmetric-log axis; ``None``
    #: chooses three decades below the largest drift.
    linear_threshold: float | None = None


@dataclass(frozen=True)
class Complexity:
    """Kinetic-energy spectral complexity at saved states (markers only)."""

    measures: tuple[str, ...] = ("mean_degree", "mean_order",
                                 "effective_modes")
    level: int | Sigma | None = None


@dataclass(frozen=True)
class Style:
    """Figure geometry and type size.

    The defaults target a README column: 7.2 in wide at 250 dpi (1800 px),
    8 pt base type, so text stays at least ~12 CSS px when the image is
    shown 880 px wide.
    """

    width_inches: float = 7.2
    base_font_size: float = 8.0
    dpi: int = 250
    map_columns: int = 2
    map_row_height_inches: float = 1.85
    static_height_inches: float = 1.6
    diagnostics_height_inches: float = 2.45


@dataclass(frozen=True)
class Overview:
    """A run at several saved times, plus static context and diagnostics.

    ``snapshots``: saved-output indices or exact saved times (``"5d"``,
    ``"12h"``, ``"3600s"``); ``None`` picks up to ``max_maps`` times spread
    evenly in physical time, always including the first and last.
    ``static``: a map drawn once (terrain), ``None`` for none, ``"auto"``
    for terrain when the run has it. ``diagnostics``: explicit panels, or
    ``"auto"`` for the solver's defaults (omitted when the run did not
    record them).
    """

    map: Map
    snapshots: tuple[int | str, ...] | None = None
    max_maps: int = 4
    static: Map | None | str = "auto"
    diagnostics: tuple[Drift | Complexity, ...] | str = "auto"
    title: str | None = None
    style: Style = field(default_factory=Style)

    def __post_init__(self) -> None:
        if self.snapshots is not None:
            object.__setattr__(self, "snapshots", tuple(self.snapshots))
        if not isinstance(self.diagnostics, str):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.static not in (None, "auto") and not isinstance(
                self.static, Map):
            raise TypeError("static must be a Map, None or 'auto'")
        if self.diagnostics != "auto" and isinstance(self.diagnostics, str):
            raise ValueError("diagnostics must be panels or 'auto'")
        if self.max_maps < 1:
            raise ValueError("max_maps must be at least 1")


@dataclass(frozen=True)
class Grid:
    """Different panels at one saved time; each map has its own scale."""

    rows: tuple[tuple[Map | Drift | Complexity, ...], ...]
    snapshot: int | str = -1
    title: str | None = None
    style: Style = field(default_factory=Style)

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.rows)
        if not rows or not all(rows):
            raise ValueError("a grid needs at least one panel in every row")
        object.__setattr__(self, "rows", rows)


_TIME_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([dhs]?)\s*$")
_TIME_UNITS = {"d": 86400.0, "h": 3600.0, "s": 1.0, "": 1.0}


def parse_time(text: str) -> float:
    """``"5d"``, ``"12h"``, ``"3600s"`` or ``"3600"`` -> seconds."""
    match = _TIME_RE.match(str(text))
    if not match:
        raise ValueError(f"cannot read a time from {text!r} "
                         "(use e.g. 5d, 12h, 3600s)")
    return float(match.group(1)) * _TIME_UNITS[match.group(2)]


def describe(view) -> dict:
    """A plain, JSON-ready description of a view (recorded in figures)."""
    def convert(value):
        if is_dataclass(value):
            return {"type": type(value).__name__,
                    **{f.name: convert(getattr(value, f.name))
                       for f in fields(value)}}
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value
    return convert(view)

