"""Latitude–sigma climate sections of a Held–Suarez experiment's saved states.

The six conventional Held–Suarez sections (Held & Suarez 1994; the set of
Kawai & Tomita 2025, Figure 10), all computed from ONE set of saved states
(``states/state-d<day>.npy``) on the run's own Gaussian state grid:

=================  =========================================  ============
key                statistic                                  units
=================  =========================================  ============
``u``              time mean of [u]                           m s^-1
``T``              time mean of [T]                           K
``usvs``           time mean of [u* v*]                       m^2 s^-2
``vsTs``           time mean of [v* T*]                       K m s^-1
``eke``            1/2 time mean of [u*^2 + v*^2]             m^2 s^-2
``TsTs``           time mean of [T*^2]                        K^2
=================  =========================================  ============

``[ ]`` is the plain (unweighted) mean over the ``nlon`` equally spaced
longitudes at one latitude and one native full sigma level, and ``*`` the
departure from that INSTANTANEOUS longitude mean of the same state. Eddy
products are formed for each state before the time mean, exactly as the
run-side accumulator (``run/held_suarez/experiment.py: daily_sample``)
forms ``[u*v*]``, ``[v*T*]``, ``[T*^2]``; ``eke`` also needs ``[v*^2]``,
which the accumulator lacks, so it is only available from saved states. The
time mean is the equal-weight arithmetic mean of the selected states (the
accumulator's sample-mean convention): no trapezoidal weights, no
interpolation in time and no filling of gaps.

Sample selection (:func:`select_samples`): the default window is
``spinup_days < day <= spinup_days + n_blocks * block_days`` (the
accumulator's averaging window); an explicit ``(start, end)`` uses the same
open-left, closed-right rule. Every saved state inside it is used; the
actual first/last days, count, cadence and any gap relative to the run's
nominal ``state_every_days`` are disclosed in :class:`SampleSelection`.

Reconstruction (:func:`reduce_sections`): one frame at a time, only the
needed u, v and T, by exact host synthesis
(:func:`~tropoi.spatial.transforms.host_synthesis.synthesize_latlon`,
:func:`~tropoi.spatial.transforms.host_synthesis.synthesize_wind_latlon`)
on the Gaussian latitudes stored in ``statistics.npz`` (checked against the
configured ``nlat`` Gauss–Legendre nodes, never replaced by another grid) and
``lon_j = 360 j / nlon`` degrees, the model's longitude convention. Memory is
bounded by one frame plus the (nlat, nlev) running sums.

Longitudinal aliasing: the longitude mean of a product of two fields with
zonal wavenumbers ``<= m_max`` is exact on ``nlon`` equally spaced points
when ``nlon > 2 m_max``. ``m_max`` is MEASURED from the coefficients (the
highest order with a nonzero coefficient in any vorticity, divergence or
temperature row of any selected state), so a populated buffer degree
``l_max`` of a retained-truncation layout counts. A violation refuses.
This makes the longitude quadrature exact; it says nothing about the size of
any GPU/host synthesis difference.

Orientation: arrays are ``(nlat, nlev)``, latitude north -> south as the
model stores it, level 0 = top (smallest sigma), the layout of
``statistics.npz``. Sigma means are NOT pressure-level means.

NumPy only; never imports CuPy, Matplotlib or the run layer.
"""
from __future__ import annotations

import hashlib
import io
import math
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tropoi.spatial.transforms.host_synthesis import (synthesize_latlon,
                                                      synthesize_wind_latlon)

__all__ = [
    "SECTION_FIELDS",
    "SECTION_KEYS",
    "Bandlimit",
    "DailySeries",
    "SampleSelection",
    "SectionAccumulator",
    "SectionError",
    "SectionResult",
    "daily_series",
    "default_window",
    "gaussian_latitudes_deg",
    "reduce_sections",
    "select_samples",
    "state_grid",
]

#: key -> (title, units, exact definition). Order = panel order.
SECTION_FIELDS: dict[str, tuple[str, str, str]] = {
    "u": ("Mean zonal wind", "m s⁻¹", "time mean of [u]"),
    "T": ("Mean temperature", "K", "time mean of [T]"),
    "usvs": ("Eddy momentum flux", "m² s⁻²", "time mean of [u* v*]"),
    "vsTs": ("Eddy heat flux", "K m s⁻¹", "time mean of [v* T*]"),
    "eke": ("Eddy kinetic energy", "m² s⁻²",
            "½ time mean of [u*² + v*²]"),
    "TsTs": ("Eddy temperature variance", "K²", "time mean of [T*²]"),
}
SECTION_KEYS = tuple(SECTION_FIELDS)

#: Absolute tolerance (degrees) when matching stored latitudes to the
#: configured Gauss–Legendre nodes (both are float64 arcsin/arccos values).
LATITUDE_MATCH_DEG = 1.0e-9


class SectionError(ValueError):
    """The saved states cannot give the requested sections."""


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def gaussian_latitudes_deg(nlat: int) -> np.ndarray:
    """Gauss–Legendre latitudes, north -> south, in degrees.

    The construction of ``GaussLatLonGridGeometry`` (nodes of
    ``leggauss(nlat)`` in sin(lat), sorted descending, ``lat = pi/2 -
    arccos(x)``), repeated here so this module never imports CuPy.
    """
    nlat = int(nlat)
    if nlat < 2:
        raise SectionError(f"nlat must be >= 2, got {nlat}")
    x, _ = np.polynomial.legendre.leggauss(nlat)
    x = x[np.argsort(-x)]
    return np.degrees(np.pi / 2.0 - np.arccos(x))


def state_grid(nlat: int, nlon: int, stored_lat_deg=None
               ) -> tuple[np.ndarray, np.ndarray, str]:
    """The run's state grid: (latitudes, longitudes, latitude source).

    ``stored_lat_deg`` (``statistics.npz['lat']``) is used when present, after
    checking it IS the configured ``nlat``-point Gaussian grid; otherwise the
    same grid is rebuilt. Longitudes are ``360 j / nlon`` degrees.
    """
    nlat, nlon = int(nlat), int(nlon)
    if nlon < 4:
        raise SectionError(f"nlon must be >= 4, got {nlon}")
    expected = gaussian_latitudes_deg(nlat)
    lon = 360.0 * np.arange(nlon, dtype=np.float64) / nlon
    if stored_lat_deg is None:
        return expected, lon, (f"Gauss–Legendre nodes rebuilt for the "
                               f"configured nlat = {nlat} (statistics.npz "
                               "has no 'lat')")
    lat = np.asarray(stored_lat_deg, dtype=np.float64).ravel()
    if lat.shape != expected.shape or not np.allclose(
            lat, expected, rtol=0.0, atol=LATITUDE_MATCH_DEG):
        raise SectionError(
            f"statistics.npz stores {lat.size} latitudes that are not the "
            f"configured {nlat}-point Gaussian state grid; refusing to "
            "reduce on a different grid")
    return lat, lon, "statistics.npz 'lat' (the model's Gaussian state grid)"


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------

def default_window(config: Mapping[str, Any]) -> tuple[float, float]:
    """``(spinup_days, spinup_days + n_blocks * block_days)`` of a config."""
    try:
        days = int(config["days"])
        spinup = int(config.get("spinup_days", 200))
        block = int(config.get("block_days", 200))
    except (KeyError, TypeError, ValueError) as err:
        raise SectionError(
            f"config.json lacks the averaging window fields: {err}") from err
    if block < 1:
        raise SectionError(f"block_days must be >= 1, got {block}")
    n_blocks = max(0, (days - spinup) // block)
    return float(spinup), float(spinup + n_blocks * block)


def _check_interval(interval) -> tuple[float, float]:
    try:
        start, end = interval
        start, end = float(start), float(end)
    except (TypeError, ValueError) as err:
        raise SectionError(
            f"interval must be a pair (start_day, end_day), got "
            f"{interval!r}") from err
    if not (math.isfinite(start) and math.isfinite(end)):
        raise SectionError(f"interval {interval!r} is not finite")
    if start < 0.0:
        raise SectionError(f"interval start {start:g} is before day 0")
    if not end > start:
        raise SectionError(
            f"interval ({start:g}, {end:g}] is empty: the window is "
            "start < day <= end, so end must exceed start")
    return start, end


def _fmt_day(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class SampleSelection:
    """Which saved states are averaged, and how that sample looks."""

    requested: tuple[float, float]
    window_source: str
    days: tuple[int, ...]
    indices: tuple[int, ...]
    nominal_every_days: int | None
    intervals: dict
    expected_days: tuple[int, ...] | None
    missing_days: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.days)

    @property
    def regular(self) -> bool:
        return len(self.intervals) <= 1

    @property
    def complete(self) -> bool:
        """Every nominal save day of the window is present."""
        return self.expected_days is not None and not self.missing_days

    def cadence_label(self) -> str:
        if self.count < 2:
            return "single state"
        if self.regular:
            (step,) = self.intervals
            return f"every {step} day{'s' if step != 1 else ''}"
        parts = ", ".join(f"{n}× {d} d" for d, n in sorted(
            self.intervals.items()))
        return f"irregular spacing ({parts})"

    def coverage_label(self) -> str:
        if self.expected_days is None:
            return ("coverage unknown: the run declares no state_every_days")
        expected = len(self.expected_days)
        if not self.missing_days:
            return (f"complete (all {expected} nominal save days of the "
                    f"window, every {self.nominal_every_days} d)")
        shown = ", ".join(str(d) for d in self.missing_days[:6])
        more = (f", … ({len(self.missing_days) - 6} more)"
                if len(self.missing_days) > 6 else "")
        return (f"INCOMPLETE: {self.count} of {expected} nominal save days "
                f"(every {self.nominal_every_days} d); missing days "
                f"{shown}{more}; gaps are not filled or interpolated")

    def summary(self) -> str:
        start, end = self.requested
        return (f"equal-weight mean of {self.count} saved states, "
                f"{_fmt_day(start)} < day ≤ {_fmt_day(end)} "
                f"({self.window_source}); samples day {self.days[0]}–"
                f"{self.days[-1]}, {self.cadence_label()}")

    def to_dict(self) -> dict:
        return {
            "rule": "equal-weight arithmetic mean of the selected saved "
                    "states; window start < day <= end",
            "requested_interval_days": list(self.requested),
            "window_source": self.window_source,
            "days": list(self.days),
            "count": self.count,
            "first_day": self.days[0] if self.days else None,
            "last_day": self.days[-1] if self.days else None,
            "intervals_days": {str(k): v for k, v in
                               sorted(self.intervals.items())},
            "cadence": self.cadence_label(),
            "nominal_every_days": self.nominal_every_days,
            "complete": self.complete,
            "coverage": self.coverage_label(),
            "missing_days": list(self.missing_days),
            "weights": "equal (1/N each)",
        }


def select_samples(saved_days: Sequence[int], config: Mapping[str, Any], *,
                   interval=None, min_states: int = 2) -> SampleSelection:
    """Select the saved states inside ``start < day <= end`` (see module doc).

    ``interval=None`` uses :func:`default_window`. Raises
    :class:`SectionError` for an invalid interval or when fewer than
    ``min_states`` saved states fall inside the window.
    """
    if int(min_states) < 1:
        raise SectionError(f"min_states must be >= 1, got {min_states}")
    if interval is None:
        start, end = default_window(config)
        if not end > start:
            raise SectionError(
                f"the run's averaging window ({_fmt_day(start)}, "
                f"{_fmt_day(end)}] is empty (days = {config.get('days')}, "
                f"spinup_days = {config.get('spinup_days')}, block_days = "
                f"{config.get('block_days')}); pass interval=(start, end)")
        source = ("default: spinup_days < day ≤ spinup_days + n_blocks × "
                  "block_days")
    else:
        start, end = _check_interval(interval)
        source = "explicit interval"
    days = [int(d) for d in saved_days]
    if any(b <= a for a, b in zip(days, days[1:])):
        raise SectionError("saved days must be strictly increasing")
    chosen = [(i, d) for i, d in enumerate(days) if start < d <= end]
    every = config.get("state_every_days")
    every = int(every) if every is not None else None
    if len(chosen) < int(min_states):
        found = (", ".join(str(d) for d in days[:8])
                 + (", …" if len(days) > 8 else "")) if days else "none"
        raise SectionError(
            f"only {len(chosen)} saved state(s) fall in {_fmt_day(start)} < "
            f"day ≤ {_fmt_day(end)}, fewer than the {int(min_states)} needed "
            f"for a time mean (saved days: {found}). Pass interval=(start, "
            f"end) covering more saved states, or run the experiment with "
            f"states saved inside the window (state_every_days). Plotting "
            f"from statistics.npz alone is not implemented yet.")
    sel_days = tuple(d for _, d in chosen)
    intervals: dict[int, int] = {}
    for a, b in zip(sel_days, sel_days[1:]):
        intervals[b - a] = intervals.get(b - a, 0) + 1
    if every is not None and every >= 1:
        first = math.floor(start / every) * every + every
        expected = tuple(d for d in range(int(first), int(math.floor(end)) + 1,
                                          every) if start < d <= end)
        missing = tuple(sorted(set(expected) - set(sel_days)))
    else:
        expected, missing = None, ()
    return SampleSelection(
        requested=(start, end), window_source=source, days=sel_days,
        indices=tuple(i for i, _ in chosen), nominal_every_days=every,
        intervals=intervals, expected_days=expected, missing_days=missing)


# ---------------------------------------------------------------------------
# bandlimit
# ---------------------------------------------------------------------------

@dataclass
class Bandlimit:
    """Measured spectral content of the selected states' u/v/T rows."""

    stored_l_max: int
    nominal_truncation: int | None
    max_degree: int = -1
    max_order: int = -1
    buffer_max_abs: float = 0.0      # largest |coefficient| above nominal

    def update(self, rows: np.ndarray) -> None:
        nonzero = np.abs(rows) > 0.0
        degrees = np.nonzero(nonzero.any(axis=(0, 2)))[0]
        orders = np.nonzero(nonzero.any(axis=(0, 1)))[0]
        if degrees.size:
            self.max_degree = max(self.max_degree, int(degrees[-1]))
            self.max_order = max(self.max_order, int(orders[-1]))
        top = self.nominal_truncation
        if top is not None and top < self.stored_l_max:
            above = float(np.abs(rows[:, top + 1:, :]).max())
            self.buffer_max_abs = max(self.buffer_max_abs, above)

    def check_longitudes(self, nlon: int) -> None:
        if not nlon > 2 * self.max_order:
            raise SectionError(
                f"nlon = {nlon} does not exceed 2 m_max = "
                f"{2 * self.max_order}: longitude means of quadratic "
                "products would alias")

    def to_dict(self, nlon: int) -> dict:
        return {"stored_l_max": self.stored_l_max,
                "nominal_truncation": self.nominal_truncation,
                "max_nonzero_degree": self.max_degree,
                "max_nonzero_order": self.max_order,
                "max_abs_above_nominal_truncation": self.buffer_max_abs,
                "nlon": nlon,
                "quadratic_products_alias_free": nlon > 2 * self.max_order,
                "rule": "longitude means of products are exact when "
                        "nlon > 2 m_max (m_max measured from the zeta, "
                        "delta and T coefficients of the selected states)"}


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------

class SectionAccumulator:
    """Running equal-weight sums of the six section statistics.

    :meth:`add` takes one state's gridpoint ``u, v, T``, each
    ``(nlev, nlat, nlon)``; longitude means and eddy products are formed for
    that state before they enter the sums.
    """

    _MOMENTS = ("u", "v", "T", "usvs", "vsTs", "usus", "vsvs", "TsTs")

    def __init__(self, nlat: int, nlev: int, nlon: int) -> None:
        self.shape = (int(nlev), int(nlat), int(nlon))
        self.sums = {k: np.zeros((int(nlat), int(nlev))) for k in self._MOMENTS}
        self.count = 0

    def add(self, u, v, T) -> None:
        u, v, T = (np.asarray(a, dtype=np.float64) for a in (u, v, T))
        for name, a in (("u", u), ("v", v), ("T", T)):
            if a.shape != self.shape:
                raise SectionError(
                    f"{name} has shape {a.shape}, expected {self.shape} "
                    "(nlev, nlat, nlon)")
            if not np.isfinite(a).all():
                raise SectionError(f"reconstructed {name} is not finite")
        zu, zv, zT = u.mean(axis=2), v.mean(axis=2), T.mean(axis=2)
        us, vs, Ts = u - zu[..., None], v - zv[..., None], T - zT[..., None]
        moments = {"u": zu, "v": zv, "T": zT,
                   "usvs": (us * vs).mean(axis=2),
                   "vsTs": (vs * Ts).mean(axis=2),
                   "usus": (us * us).mean(axis=2),
                   "vsvs": (vs * vs).mean(axis=2),
                   "TsTs": (Ts * Ts).mean(axis=2)}
        for key, value in moments.items():
            self.sums[key] += value.T                   # (nlat, nlev)
        self.count += 1

    def means(self) -> dict[str, np.ndarray]:
        """The six sections (plus [v], [u*^2], [v*^2]); equal weights 1/N."""
        if self.count == 0:
            raise SectionError("no states were accumulated")
        out = {k: v / self.count for k, v in self.sums.items()}
        out["eke"] = 0.5 * (out["usus"] + out["vsvs"])
        return out


@dataclass(frozen=True)
class SectionResult:
    """Reduced sections plus everything needed to label them."""

    fields: dict
    lat_deg: np.ndarray
    sigma: np.ndarray
    sigma_interfaces: np.ndarray
    lon_deg: np.ndarray
    selection: SampleSelection
    bandlimit: dict
    latitude_source: str
    state_files: tuple = field(default=())

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]

    def states_sha256(self) -> str:
        """SHA-256 over ``name:sha256`` lines of the selected state files."""
        h = hashlib.sha256()
        for name, digest in self.state_files:
            h.update(f"{name}:{digest}\n".encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# daily global series (series.json)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DailySeries:
    """The run's daily global scalars, one value per simulated day.

    Read from ``series.json`` (written by the run from every daily sample,
    day 0 = the initial state), NOT from the saved states: ``mean_ps`` is
    the Gaussian-weighted global mean of p_s (proportional to the total
    mass, which the dry Held–Suarez forcing neither adds nor removes),
    ``ke`` the global mean of sum_k dsigma_k |V_k|^2 / 2 and ``mean_T`` that
    of sum_k dsigma_k T_k. The forcing and dissipation make KE and T
    non-conserved; their series show whether the climate has settled.
    """

    days: np.ndarray
    mean_ps: np.ndarray
    ke: np.ndarray
    mean_T: np.ndarray

    @property
    def reference_day(self) -> int:
        return int(self.days[0])

    def mass_drift(self) -> np.ndarray:
        """(<p_s>(day) - <p_s>(first day)) / <p_s>(first day)."""
        return (self.mean_ps - self.mean_ps[0]) / self.mean_ps[0]

    def summary(self, window: tuple[float, float]) -> dict:
        """Mass drift over all days; KE and T inside ``start < day <= end``."""
        drift = self.mass_drift()
        worst = int(np.argmax(np.abs(drift)))
        start, end = window
        inside = (self.days > start) & (self.days <= end)
        out = {"reference_day": self.reference_day,
               "first_day": int(self.days[0]),
               "last_day": int(self.days[-1]),
               "count": int(self.days.size),
               "window_days": [float(start), float(end)],
               "days_in_window": int(inside.sum()),
               "mass_drift_max_abs": float(abs(drift[worst])),
               "mass_drift_max_abs_day": int(self.days[worst]),
               "mass_drift_last": float(drift[-1])}
        for key in ("ke", "mean_T"):
            values = getattr(self, key)[inside]
            out[f"{key}_window"] = (None if values.size == 0 else {
                "mean": float(values.mean()), "min": float(values.min()),
                "max": float(values.max())})
        return out

    def to_dict(self) -> dict:
        return {"day": self.days.astype(int).tolist(),
                "relative_mass_drift": self.mass_drift().tolist(),
                "ke": self.ke.tolist(), "mean_T": self.mean_T.tolist()}


def daily_series(doc: Mapping[str, Any]) -> DailySeries:
    """Validate the parsed ``series.json`` of a Held–Suarez run."""
    try:
        arrays = {k: np.asarray(doc[k], dtype=np.float64)
                  for k in ("day", "mean_ps", "ke", "mean_T")}
    except (KeyError, TypeError, ValueError) as err:
        raise SectionError(
            f"series.json lacks a usable day/mean_ps/ke/mean_T series: "
            f"{err}") from err
    days = arrays["day"]
    if days.ndim != 1 or days.size < 2 or any(
            a.shape != days.shape for a in arrays.values()):
        raise SectionError("series.json series must be one-dimensional, of "
                           "equal length and at least two days long")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise SectionError("series.json holds non-finite values")
    if np.any(days != np.round(days)) or np.any(np.diff(days) != 1.0):
        raise SectionError("series.json days must be consecutive whole days")
    if not np.all(arrays["mean_ps"] > 0.0):
        raise SectionError("series.json mean_ps must be positive")
    return DailySeries(days=days, mean_ps=arrays["mean_ps"],
                       ke=arrays["ke"], mean_T=arrays["mean_T"])


def _synthesize_uvT(frame: np.ndarray, nlev: int, lat, lon, radius_m: float):
    nlat, nlon = lat.size, lon.size
    u = np.empty((nlev, nlat, nlon))
    v = np.empty_like(u)
    T = np.empty_like(u)
    for k in range(nlev):
        T[k] = synthesize_latlon(frame[2 * nlev + k], lat, lon)
        u[k], v[k] = synthesize_wind_latlon(frame[k], frame[nlev + k],
                                            radius_m, lat, lon)
    return u, v, T


def load_state(path) -> tuple[np.ndarray, str]:
    """Read one saved state once: (array, sha256 of the file bytes)."""
    data = pathlib.Path(path).read_bytes()
    array = np.load(io.BytesIO(data), allow_pickle=False)
    return array, hashlib.sha256(data).hexdigest()


def reduce_sections(frames: Iterable, selection: SampleSelection, *,
                    nlev: int, l_max: int, nominal_truncation: int | None,
                    lat_deg, lon_deg, sigma, sigma_interfaces,
                    radius_m: float, latitude_source: str = "",
                    progress: Callable[[int, int], None] | None = None
                    ) -> SectionResult:
    """Reduce the selected frames to the six sections (see module doc).

    ``frames`` yields, in ``selection.days`` order, either packed PE frames
    ``(3 nlev + 1, l_max + 1, l_max + 1)`` or ``(frame, name, sha256)``
    triples; only one frame is held at a time.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    expected = (3 * nlev + 1, l_max + 1, l_max + 1)
    acc = SectionAccumulator(lat.size, nlev, lon.size)
    band = Bandlimit(stored_l_max=int(l_max),
                     nominal_truncation=(None if nominal_truncation is None
                                         else int(nominal_truncation)))
    files = []
    total = selection.count
    for n, item in enumerate(frames):
        if isinstance(item, tuple):
            frame, name, digest = item
            files.append((str(name), str(digest)))
        else:
            frame = item
        frame = np.asarray(frame)
        if frame.shape != expected or not np.iscomplexobj(frame):
            raise SectionError(
                f"saved state {n} has shape {frame.shape} / {frame.dtype}; "
                f"expected complex {expected}")
        band.update(frame[:3 * nlev])
        band.check_longitudes(lon.size)
        acc.add(*_synthesize_uvT(frame, nlev, lat, lon, radius_m))
        del frame
        if progress is not None:
            progress(n + 1, total)
    if acc.count != total:
        raise SectionError(
            f"expected {total} selected states, reduced {acc.count}")
    means = acc.means()
    return SectionResult(
        fields={k: means[k] for k in (*SECTION_KEYS, "v", "usus", "vsvs")},
        lat_deg=lat, sigma=np.asarray(sigma, dtype=np.float64),
        sigma_interfaces=np.asarray(sigma_interfaces, dtype=np.float64),
        lon_deg=lon, selection=selection, bandlimit=band.to_dict(lon.size),
        latitude_source=latitude_source, state_files=tuple(files))
