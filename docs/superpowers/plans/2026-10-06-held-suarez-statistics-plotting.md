# Held–Suarez Statistics Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the completed Held–Suarez climatology (zonal- and time-mean σ sections and daily series) from `statistics.npz` on any host, through the existing view/spec/renderer stack, and publish it in the README from a slim, hash-receipted evidence bundle.

**Architecture:** A host-only reader in `tropoi.representation.archive.statistics` turns an HS run directory (`statistics.npz` + `config.json`) into labelled `Product` objects that carry quantity id, reduction, weighting, coordinates, interval and source. A new `SectionSpec` panel (latitude × σ) joins the declarative spec family and is drawn by the Matplotlib renderer with the same normalization groups, keys and atomic write as maps. `Section`, `DailySeries`, `Panels`, `ZonalMean` and `TimeMean` view objects compose figures in a new `statistics_figures` module; `tropoi plot` recognises an HS directory and routes there. A pinned recipe draws the README figure from a published bundle under `docs/runs/`.

**Tech Stack:** Python 3.11+, NumPy, Matplotlib (Agg), pytest. No CuPy anywhere in this plan.

## Global Constraints

- `tropoi.representation` must not import `tropoi.run` or `tropoi.cli` (docs/ARCHITECTURE.md "Dependency direction"; enforced by `tests/test_layout_compat.py`). The reader therefore reads `config.json` as a plain dict and never imports `tropoi.run.held_suarez`.
- Nothing in this plan imports CuPy; every new test is CPU-only and runs in CI.
- Every panel states its reduction, window, sample count, cadence, weighting and coordinate system in its label or subtitle; no averaging is ever silent.
- Block spread is a *variability diagnostic*: sidecar key `block_spread`, label "block-to-block spread (N blocks of D days)". The words "uncertainty", "standard error", "confidence" and "significant" do not appear in code, labels or docs of this plan.
- "Not in the accumulators" and "not in any saved data of this run" are distinct messages, produced by one function (`sources_for`).
- Publication recipes pin `source="accumulated"` and spell out every field of every view object (the W5 recipe rule, `tests/test_readme_figure_recipe.py`).
- Figures go to `RUN/assets/` only; nothing else inside a run directory is ever written (same rule as `compose.render_view`).
- Whenever a module is added under `src/tropoi`, run `python -m tools.architecture.generate` and commit `docs/architecture/current/` in the same commit (CI runs `--check`).
- Run tests with the repository venv: `./venv/Scripts/python.exe -m pytest ...` (Windows). Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Existing behaviour of `Map`, `Overview`, `Grid`, `compose.py`, the capsule reader and all current tests stays byte-identical.

---

## File structure

| File | Responsibility |
|---|---|
| `src/tropoi/representation/archive/statistics.py` (new) | `Product`, `StatisticsError`, `AccumulatedStatistics`, `open_statistics`, `is_statistics_run`. Reads `statistics.npz` + `config.json`. NumPy + stdlib only. |
| `src/tropoi/representation/archive/__init__.py` | Re-export `open_statistics`, `is_statistics_run`, `AccumulatedStatistics`, `StatisticsError`. |
| `src/tropoi/representation/visual/fields.py` | Add `SectionField` (values on latitude × σ with interface edges). |
| `src/tropoi/representation/visual/specs.py` | Add `SectionContourLayer`, `SectionSpec`; extend `PanelSpec`. |
| `src/tropoi/representation/visual/timeline.py` | Teach `_normalization_slots` / `_normalizable_values` about `SectionSpec`. |
| `src/tropoi/representation/visual/matplotlib_renderer.py` | `_render_section`, `_sigma_edges` helper. |
| `src/tropoi/representation/visual/views.py` | Add `ZonalMean`, `TimeMean`, `Section`, `DailySeries`, `Panels`. |
| `src/tropoi/representation/visual/statistics_figures.py` (new) | `default_statistics_view`, `compose_statistics_view`, `render_statistics_view`. Composition only; no I/O beyond the renderer. |
| `src/tropoi/cli/main.py` | `tropoi plot` routes HS directories; `--section`, `--mean`; listing with sources. |
| `tests/statistics_fixture.py` (new) | `write_statistics_run(tmp_path, ...)`: a tiny deterministic HS-layout run. |
| `tests/test_accumulated_statistics.py` (new) | Reader tests. |
| `tests/test_section_panel.py` (new) | Field, spec, resolver and renderer tests. |
| `tests/test_statistics_views.py` (new) | Views, composition, sidecar, output-path rules. |
| `tests/cli/test_plot_statistics.py` (new) | CLI routing and listing. |
| `docs/figures/held_suarez_t42l20_climatology.py` (new) | The pinned README recipe. |
| `docs/runs/hs-T42L20-prod-002/` (new) | Slim evidence bundle + `SHA256SUMS` + `assets/`. |
| `tests/test_readme_figure_recipe.py`, `tests/test_published_layout.py` | Extend to the HS recipe and to receipts without a coefficients file. |
| `README.md`, `docs/SAVED_RUNS.md`, `docs/runs/README.md`, `docs/held_suarez/STATUS.md` | Documentation. |

---

### Task 1: Test fixture — a tiny HS-layout run

**Files:**
- Create: `tests/statistics_fixture.py`

**Interfaces:**
- Produces: `write_statistics_run(root: pathlib.Path, *, nlat=8, nlev=4, n_blocks=2, block_days=2, spinup_days=1, experiment_id="hs-fixture") -> pathlib.Path` returning the run directory. Block means are known by construction: `u[b, i, k] = 10*b + i + 0.1*k`, `T[b, i, k] = 250 + b + k`, every other zonal field `= b + 1`, counts all equal `block_days`, series linear in day.

- [ ] **Step 1: Write the fixture**

```python
"""A deterministic Held–Suarez-layout run directory for CPU tests.

Mirrors what ``tropoi.run.held_suarez.experiment._export`` writes
(``statistics.npz``, ``series.json``, ``config.json``) with tiny shapes and
hand-checkable numbers, so reader tests need neither a GPU nor the solver.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib

import numpy as np

CONFIG_SCHEMA = "palintropos.held_suarez.config/1"
ZONAL_KEYS = ("u", "v", "T", "TsTs", "usus", "vsTs", "usvs")


def _gauss(nlat: int):
    x, w = np.polynomial.legendre.leggauss(nlat)
    lat = np.degrees(np.arcsin(x))[::-1]          # north to south
    return lat, (w[::-1] / 2.0)


def write_statistics_run(root: pathlib.Path, *, nlat: int = 8, nlev: int = 4,
                         n_blocks: int = 2, block_days: int = 2,
                         spinup_days: int = 1,
                         experiment_id: str = "hs-fixture") -> pathlib.Path:
    run = pathlib.Path(root) / experiment_id
    run.mkdir(parents=True, exist_ok=True)
    days = spinup_days + n_blocks * block_days
    lat, weights = _gauss(nlat)
    sigma = (np.arange(nlev) + 0.5) / nlev
    dsigma = np.full(nlev, 1.0 / nlev)
    b = np.arange(n_blocks, dtype=float)[:, None, None]
    i = np.arange(nlat, dtype=float)[None, :, None]
    k = np.arange(nlev, dtype=float)[None, None, :]
    blocks = {key: np.broadcast_to(b + 1.0, (n_blocks, nlat, nlev)).copy()
              for key in ZONAL_KEYS}
    blocks["u"] = 10.0 * b + i + 0.1 * k + 0.0 * k
    blocks["T"] = 250.0 + b + k + 0.0 * i
    counts = np.full(n_blocks, block_days, dtype=np.int64)
    series_day = np.arange(days + 1, dtype=float)
    series = {"day": series_day, "mean_ps": 1.0e5 - series_day,
              "mean_lnps": np.log(1.0e5) - 1e-5 * series_day,
              "mean_T": 250.0 + 0.01 * series_day, "ke": 1.0 + series_day,
              "max_abs_u": 2.0 * series_day, "t_min": 200.0 + 0 * series_day,
              "t_max": 300.0 + 0 * series_day, "comp_mode": 0.01 + 0 * series_day}
    buf = io.BytesIO()
    np.savez(buf, counts=counts,
             **{f"block_{key}": value for key, value in blocks.items()},
             block_u_sq=blocks["u"] ** 2 + 1.0, block_T_sq=blocks["T"] ** 2 + 1.0,
             block_ke_spectrum=np.ones((n_blocks, 6)),
             **{f"series_{key}": value for key, value in series.items()},
             lat=lat, sigma=sigma, dsigma=dsigma, gauss_weights=weights)
    (run / "statistics.npz").write_bytes(buf.getvalue())
    (run / "series.json").write_text(json.dumps(
        {key: value.tolist() for key, value in series.items()}, sort_keys=True),
        encoding="utf-8")
    config = {"schema": CONFIG_SCHEMA, "experiment_id": experiment_id,
              "l_max": 5, "retained_truncation": 4, "nlat": nlat, "nlon": 2 * nlat,
              "nlev": nlev, "dt": 1200.0, "days": days, "spinup_days": spinup_days,
              "block_days": block_days, "state_every_days": 1,
              "damping_scheme": "centred", "radius": 6.371e6, "gravity": 9.8}
    doc = {"config": config,
           "config_sha256": hashlib.sha256(json.dumps(
               config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
           "protocol_sha256": "0" * 64,
           "code": {"git_commit": "f" * 40, "git_dirty": False},
           "created": "2026-01-01T00:00:00Z"}
    (run / "config.json").write_text(json.dumps(doc, indent=2, sort_keys=True),
                                     encoding="utf-8")
    (run / "events.jsonl").write_text(
        json.dumps({"event": "created", "day": 0, "step": 0}) + "\n"
        + json.dumps({"event": "completed", "day": days, "step": 1}) + "\n",
        encoding="utf-8")
    return run
```

- [ ] **Step 2: Smoke-check it imports and writes**

Run: `./venv/Scripts/python.exe -c "import pathlib,tempfile,sys; sys.path.insert(0,'tests'); from statistics_fixture import write_statistics_run as w; import numpy as np; r=w(pathlib.Path(tempfile.mkdtemp())); z=np.load(r/'statistics.npz'); print(sorted(z.files)[:4], z['block_u'].shape)"`
Expected: `['block_T', 'block_T_sq', 'block_TsTs', 'block_ke_spectrum'] (2, 8, 4)`

- [ ] **Step 3: Commit**

```bash
git add tests/statistics_fixture.py
git commit -m "test(statistics): deterministic Held–Suarez-layout run fixture

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `Product` and the accumulated-statistics reader

**Files:**
- Create: `src/tropoi/representation/archive/statistics.py`
- Modify: `src/tropoi/representation/archive/__init__.py`
- Test: `tests/test_accumulated_statistics.py`
- Regenerate: `docs/architecture/current/` (new module)

**Interfaces:**
- Produces:
  - `is_statistics_run(path) -> bool`
  - `open_statistics(path) -> AccumulatedStatistics`
  - `class StatisticsError(ValueError)`
  - `@dataclass(frozen=True) Product(quantity: str, long_name: str, units: str, values: np.ndarray, coordinates: dict[str, np.ndarray], reduction: tuple[str, ...], weighting: str, interval: dict | None, source: dict, definition: str)`
  - `AccumulatedStatistics` attributes: `run_dir`, `config` (dict), `config_sha256`, `statistics_sha256`, `experiment_id`, `latitudes_deg` (north→south, as stored), `gauss_weights`, `sigma_full`, `sigma_interfaces`, `dsigma`, `counts`, `block_windows: tuple[tuple[int, int], ...]` (first and last day of each block), `window: tuple[int, int]`; methods `product(quantity) -> Product`, `block_products(quantity) -> Product`, `block_spread(quantity) -> Product`, `series(name) -> Product`, `quantities() -> list[dict]`, `sources_for(quantity, reduction) -> dict`, `sha256s() -> dict`.
  - `ZONAL_QUANTITIES: dict[str, tuple[str, str, str, str]]` = id → (npz key, long name, units, definition); `SERIES_QUANTITIES` likewise for daily series.

- [ ] **Step 1: Write the failing tests**

```python
"""Host-only reader of Held–Suarez accumulated statistics."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from statistics_fixture import write_statistics_run  # noqa: E402

from tropoi.representation.archive.statistics import (  # noqa: E402
    AccumulatedStatistics, Product, StatisticsError, is_statistics_run,
    open_statistics)


@pytest.fixture
def run(tmp_path):
    return write_statistics_run(tmp_path)


def test_detects_a_statistics_run_and_not_a_capsule(run, tmp_path):
    assert is_statistics_run(run)
    other = tmp_path / "capsule"
    other.mkdir()
    (other / "config.json").write_text('{"solver": "bve"}', encoding="utf-8")
    assert not is_statistics_run(other)
    assert not is_statistics_run(tmp_path / "missing")


def test_open_reads_coordinates_window_and_hashes(run):
    stats = open_statistics(run)
    assert isinstance(stats, AccumulatedStatistics)
    assert stats.experiment_id == "hs-fixture"
    assert stats.sigma_full.shape == (4,)
    np.testing.assert_allclose(stats.sigma_interfaces, [0, .25, .5, .75, 1.0])
    assert stats.window == (2, 5)                      # spinup 1, 2 blocks of 2 days
    assert stats.block_windows == ((2, 3), (4, 5))
    assert stats.latitudes_deg[0] > stats.latitudes_deg[-1]   # north to south
    assert len(stats.statistics_sha256) == 64 and len(stats.config_sha256) == 64
    assert stats.config_sha256 == json.loads(
        (run / "config.json").read_text())["config_sha256"]


def test_product_is_the_count_weighted_mean_over_blocks(run):
    stats = open_statistics(run)
    u = stats.product("zonal_wind")
    assert isinstance(u, Product)
    assert u.quantity == "zonal_wind" and u.units == "m s^-1"
    i = np.arange(8.0)[:, None]
    k = np.arange(4.0)[None, :]
    np.testing.assert_allclose(u.values, 5.0 + i + 0.1 * k)   # mean of 10b over b=0,1
    assert u.reduction == ("zonal_mean", "time_mean")
    assert u.weighting == "unweighted on sigma surfaces"
    assert u.interval == {"days": (2, 5), "samples": 4, "cadence": "1 per day",
                          "blocks": 2, "block_days": 2}
    assert u.source["kind"] == "accumulated"
    assert u.source["file"] == "statistics.npz"
    assert u.source["sha256"] == stats.statistics_sha256
    assert set(u.coordinates) == {"latitude_deg", "sigma_full", "sigma_interfaces"}


def test_eddy_products_keep_the_accumulated_definition(run):
    flux = open_statistics(run).product("eddy_heat_flux")
    assert flux.units == "K m s^-1"
    assert "instantaneous zonal mean" in flux.definition
    assert "transient" not in flux.definition.lower()


def test_block_products_and_block_spread(run):
    stats = open_statistics(run)
    blocks = stats.block_products("zonal_wind")
    assert blocks.values.shape == (2, 8, 4)
    assert blocks.reduction == ("zonal_mean", "time_mean", "per_block")
    spread = stats.block_spread("zonal_wind")
    np.testing.assert_allclose(spread.values, np.full((8, 4), 10.0 / np.sqrt(2.0)))
    assert spread.reduction == ("zonal_mean", "time_mean", "block_spread")
    assert "block-to-block spread" in spread.long_name
    assert "uncertainty" not in spread.definition.lower()


def test_series_carry_day_coordinates_and_daily_cadence(run):
    ke = open_statistics(run).series("kinetic_energy")
    assert ke.values.shape == (6,)
    np.testing.assert_array_equal(ke.coordinates["day"], np.arange(6.0))
    assert ke.interval["cadence"] == "1 per day"
    assert ke.reduction == ("global_mean",)


def test_quantities_listing_names_sources(run):
    rows = {row["id"]: row for row in open_statistics(run).quantities()}
    assert rows["temperature"]["kind"] == "accumulated"
    assert rows["temperature"]["needs"] == "host"
    assert rows["temperature"]["available"] is True
    assert "zonal_mean" in rows["temperature"]["sources"]["accumulated"]
    assert rows["kinetic_energy"]["shape"] == "series"


def test_sources_for_distinguishes_accumulators_from_all_saved_data(run):
    stats = open_statistics(run)
    ok = stats.sources_for("temperature", ("zonal_mean", "time_mean"))
    assert ok["accumulated"] is True
    nope = stats.sources_for("temperature", ("time_mean",))
    assert nope["accumulated"] is False
    assert nope["saved_states"] == {"available": False,
                                     "reason": "no states/ directory in this run"}
    (stats.run_dir / "states").mkdir()
    (stats.run_dir / "states" / "state-d00002.npy").write_bytes(b"")
    (stats.run_dir / "states" / "state-d00004.npy").write_bytes(b"")
    again = open_statistics(run).sources_for("temperature", ("time_mean",))
    assert again["saved_states"]["available"] is True
    assert again["saved_states"]["count"] == 2
    assert again["saved_states"]["days"] == (2, 4)
    assert "CUDA" in again["saved_states"]["note"]
    with pytest.raises(StatisticsError, match="not available from any saved data"):
        stats.sources_for("mass_streamfunction", ("zonal_mean", "time_mean"))


def test_unknown_quantity_and_corrupt_layout_fail_clearly(run):
    stats = open_statistics(run)
    with pytest.raises(StatisticsError, match="unknown quantity"):
        stats.product("potential_enstrophy")
    z = dict(np.load(run / "statistics.npz"))
    z["sigma"] = z["sigma"] + 0.01
    np.savez(run / "statistics.npz", **z)
    with pytest.raises(StatisticsError, match="sigma"):
        open_statistics(run)


def test_reader_never_imports_the_solver_or_heavy_modules(run):
    code = ("import sys; from tropoi.representation.archive.statistics import "
            "open_statistics; s = open_statistics(sys.argv[1]); s.product('temperature'); "
            "bad = [m for m in sys.modules if m.startswith(('tropoi.run', 'tropoi.cli', "
            "'cupy', 'matplotlib', 'tropoi.temporal.tendencies', "
            "'tropoi.representation.visual'))]; print('BAD', bad)")
    out = subprocess.run([sys.executable, "-c", code, str(run)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "BAD []"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accumulated_statistics.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'tropoi.representation.archive.statistics'`

- [ ] **Step 3: Write the reader**

```python
"""Accumulated (online) statistics of a Held–Suarez run, read on the host.

A Held–Suarez experiment directory is not a snapshot capsule: its primary
product is ``statistics.npz``, written by the experiment driver with
per-block sums of daily zonal means and the daily scalar series. This
module reads that file and ``config.json`` as plain data (NumPy + stdlib;
it never imports the solver, the model or the stepper) and hands out
:class:`Product` objects that say exactly what they are: which catalogue
quantity, which reductions were applied, how the mean was weighted, on
which coordinates, over which days and samples, and from which file.

Definitions (``tropoi.run.held_suarez.statistics``, restated as data here):
every daily sample is the instantaneous state at the end of a day; zonal
means are unweighted means over the longitudes of the Gauss state grid, on
sigma surfaces; eddy products are zonal (co)variances about the
*instantaneous* zonal mean, averaged over the days of each block. No
stationary/transient split is available: the accumulators hold nothing in
longitude.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
import re

import numpy as np

__all__ = ["AccumulatedStatistics", "Product", "SERIES_QUANTITIES",
           "StatisticsError", "ZONAL_QUANTITIES", "is_statistics_run",
           "open_statistics"]

CONFIG_SCHEMA = "palintropos.held_suarez.config/1"
STATISTICS_FILE = "statistics.npz"
WEIGHTING = "unweighted on sigma surfaces"
_DAILY = "instantaneous state at the end of each day, 1 sample per day"

#: id -> (npz key, long name, units, definition)
ZONAL_QUANTITIES: dict[str, tuple[str, str, str, str]] = {
    "zonal_wind": ("u", "zonal wind", "m s^-1",
                   "[u]: unweighted zonal mean of u on sigma surfaces, " + _DAILY),
    "meridional_wind": ("v", "meridional wind", "m s^-1",
                        "[v]: unweighted zonal mean of v on sigma surfaces, " + _DAILY),
    "temperature": ("T", "temperature", "K",
                    "[T]: unweighted zonal mean of T on sigma surfaces, " + _DAILY),
    "eddy_temperature_variance": (
        "TsTs", "eddy temperature variance", "K^2",
        "[T*T*]: zonal variance of T about its instantaneous zonal mean, "
        "averaged over the daily samples of each block"),
    "eddy_zonal_wind_variance": (
        "usus", "eddy zonal-wind variance", "m^2 s^-2",
        "[u*u*]: zonal variance of u about its instantaneous zonal mean, "
        "averaged over the daily samples of each block"),
    "eddy_heat_flux": (
        "vsTs", "eddy heat flux", "K m s^-1",
        "[v*T*]: zonal covariance of v and T about their instantaneous zonal "
        "means, averaged over the daily samples of each block"),
    "eddy_momentum_flux": (
        "usvs", "eddy momentum flux", "m^2 s^-2",
        "[u*v*]: zonal covariance of u and v about their instantaneous zonal "
        "means, averaged over the daily samples of each block"),
}

#: id -> (npz key, long name, units, definition)
SERIES_QUANTITIES: dict[str, tuple[str, str, str, str]] = {
    "mean_surface_pressure": ("mean_ps", "global-mean surface pressure", "Pa",
                              "Gaussian-weighted global mean of p_s = exp(ln p_s)"),
    "mean_log_surface_pressure": ("mean_lnps", "global-mean ln p_s", "ln Pa",
                                  "Gaussian-weighted global mean of ln p_s"),
    "mean_temperature": ("mean_T", "global mass-weighted temperature", "K",
                         "sum_k dsigma_k of the Gaussian-weighted global mean of T_k"),
    "kinetic_energy": ("ke", "global kinetic energy per unit mass", "m^2 s^-2",
                       "global mean of sum_k dsigma_k |V_k|^2 / 2"),
    "max_wind_speed": ("max_abs_u", "largest |u| on the state grid", "m s^-1",
                       "max over levels and grid points of |u|"),
    "min_temperature": ("t_min", "lowest temperature", "K", "min over levels and points of T"),
    "max_temperature": ("t_max", "highest temperature", "K", "max over levels and points of T"),
    "computational_mode": ("comp_mode", "leapfrog computational-mode indicator", "1",
                           "|X^n - X^{n-1}_f| / |X^n| over the zeta and delta rows"),
}

_STATE_FILE = re.compile(r"^state-d(\d+)\.npy$")


class StatisticsError(ValueError):
    """The directory is not a readable Held–Suarez statistics run."""


@dataclass(frozen=True)
class Product:
    """One evaluated statistical product and everything that defines it."""

    quantity: str
    long_name: str
    units: str
    values: np.ndarray
    coordinates: dict
    reduction: tuple[str, ...]
    weighting: str
    interval: dict | None
    source: dict
    definition: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "reduction", tuple(self.reduction))

    def describe(self) -> dict:
        """Everything but the values, JSON-ready (for sidecars)."""
        return {"quantity": self.quantity, "long_name": self.long_name,
                "units": self.units, "shape": list(self.values.shape),
                "reduction": list(self.reduction), "weighting": self.weighting,
                "interval": None if self.interval is None else {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in self.interval.items()},
                "source": dict(self.source), "definition": self.definition}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_statistics_run(path) -> bool:
    """A directory holding ``statistics.npz`` and a Held–Suarez ``config.json``."""
    path = pathlib.Path(path)
    config = path / "config.json"
    if not (path / STATISTICS_FILE).is_file() or not config.is_file():
        return False
    try:
        doc = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("config"), dict) and (
        doc["config"].get("schema") == CONFIG_SCHEMA)


class AccumulatedStatistics:
    """Read-only view of one run's ``statistics.npz`` and ``config.json``."""

    def __init__(self, run_dir) -> None:
        self.run_dir = pathlib.Path(run_dir)
        if not is_statistics_run(self.run_dir):
            raise StatisticsError(
                f"{self.run_dir} is not a Held–Suarez statistics run (needs "
                f"{STATISTICS_FILE} and a config.json with schema {CONFIG_SCHEMA!r})")
        doc = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.config: dict = dict(doc["config"])
        self.config_sha256: str = str(doc.get("config_sha256") or "")
        self.code: dict = dict(doc.get("code") or {})
        self.protocol_sha256 = doc.get("protocol_sha256")
        self.statistics_path = self.run_dir / STATISTICS_FILE
        self.statistics_sha256 = _sha256(self.statistics_path)
        with np.load(self.statistics_path, allow_pickle=False) as archive:
            self._arrays = {name: np.array(archive[name]) for name in archive.files}
        for array in self._arrays.values():
            array.flags.writeable = False
        self._validate()

    # -- layout ---------------------------------------------------------
    def _validate(self) -> None:
        a = self._arrays
        needed = {"counts", "lat", "sigma", "dsigma", "gauss_weights", "series_day"}
        needed |= {f"block_{key}" for key, *_ in ZONAL_QUANTITIES.values()}
        missing = sorted(needed - set(a))
        if missing:
            raise StatisticsError(f"{self.statistics_path} lacks arrays {missing}")
        nlev = int(self.config["nlev"])
        nlat = int(self.config["nlat"])
        if a["sigma"].shape != (nlev,) or a["lat"].shape != (nlat,):
            raise StatisticsError(
                f"sigma/lat shapes {a['sigma'].shape}/{a['lat'].shape} disagree "
                f"with config nlev={nlev}, nlat={nlat}")
        interfaces = np.arange(nlev + 1) / nlev
        expected = 0.5 * (interfaces[1:] + interfaces[:-1])
        if not np.allclose(a["sigma"], expected, rtol=0.0, atol=1e-12):
            raise StatisticsError(
                "sigma full levels are not the midpoints of a uniform sigma "
                "grid; this reader only knows the uniform layout the Held–Suarez "
                "driver writes")
        self.sigma_interfaces = interfaces
        self.sigma_interfaces.flags.writeable = False
        if a["lat"].size > 1 and not np.all(np.diff(a["lat"]) < 0.0):
            raise StatisticsError("latitudes must be stored north to south")
        nb = a["counts"].shape[0]
        for key, *_ in ZONAL_QUANTITIES.values():
            if a[f"block_{key}"].shape != (nb, nlat, nlev):
                raise StatisticsError(f"block_{key} has shape "
                                      f"{a[f'block_{key}'].shape}, expected {(nb, nlat, nlev)}")

    # -- facts ------------------------------------------------------------
    @property
    def experiment_id(self) -> str:
        return str(self.config["experiment_id"])

    @property
    def latitudes_deg(self) -> np.ndarray:
        return self._arrays["lat"]

    @property
    def gauss_weights(self) -> np.ndarray:
        return self._arrays["gauss_weights"]

    @property
    def sigma_full(self) -> np.ndarray:
        return self._arrays["sigma"]

    @property
    def dsigma(self) -> np.ndarray:
        return self._arrays["dsigma"]

    @property
    def counts(self) -> np.ndarray:
        return self._arrays["counts"]

    @property
    def block_windows(self) -> tuple[tuple[int, int], ...]:
        spin, bd = int(self.config["spinup_days"]), int(self.config["block_days"])
        return tuple((spin + b * bd + 1, spin + (b + 1) * bd)
                     for b in range(self.counts.shape[0]))

    @property
    def window(self) -> tuple[int, int]:
        windows = self.block_windows
        return (windows[0][0], windows[-1][1]) if windows else (0, 0)

    @property
    def complete(self) -> bool:
        return bool(np.all(self.counts == int(self.config["block_days"])))

    def sha256s(self) -> dict:
        return {"statistics.npz": self.statistics_sha256,
                "config.json": _sha256(self.run_dir / "config.json")}

    def _source(self) -> dict:
        return {"kind": "accumulated", "file": STATISTICS_FILE,
                "sha256": self.statistics_sha256, "run_dir": str(self.run_dir),
                "experiment_id": self.experiment_id,
                "config_sha256": self.config_sha256}

    def _coordinates(self) -> dict:
        return {"latitude_deg": self.latitudes_deg, "sigma_full": self.sigma_full,
                "sigma_interfaces": self.sigma_interfaces}

    def _interval(self) -> dict:
        return {"days": self.window, "samples": int(self.counts.sum()),
                "cadence": "1 per day", "blocks": int(self.counts.shape[0]),
                "block_days": int(self.config["block_days"])}

    def _zonal(self, quantity: str):
        try:
            return ZONAL_QUANTITIES[quantity]
        except KeyError:
            raise StatisticsError(
                f"unknown quantity {quantity!r}; accumulated quantities: "
                f"{', '.join(ZONAL_QUANTITIES)}; series: {', '.join(SERIES_QUANTITIES)}"
            ) from None

    # -- products ---------------------------------------------------------
    def block_products(self, quantity: str) -> Product:
        """Per-block time means, shape (blocks, nlat, nlev)."""
        key, long_name, units, definition = self._zonal(quantity)
        return Product(quantity, long_name + " (per block)", units,
                       self._arrays[f"block_{key}"], self._coordinates(),
                       ("zonal_mean", "time_mean", "per_block"), WEIGHTING,
                       {**self._interval(), "block_windows": self.block_windows},
                       self._source(), definition)

    def product(self, quantity: str) -> Product:
        """The count-weighted mean over every block: the run's climatology."""
        key, long_name, units, definition = self._zonal(quantity)
        counts = self.counts.astype(np.float64)
        if not counts.sum() > 0:
            raise StatisticsError("no daily samples were accumulated")
        values = (self._arrays[f"block_{key}"] * counts[:, None, None]).sum(0) / counts.sum()
        return Product(quantity, long_name, units, values, self._coordinates(),
                       ("zonal_mean", "time_mean"), WEIGHTING, self._interval(),
                       self._source(),
                       definition + "; time mean = count-weighted mean of the block means")

    def block_spread(self, quantity: str) -> Product:
        """Block-to-block spread (sample std of the block means, ddof 1): a
        variability diagnostic of the time mean, not a bound on it."""
        blocks = self.block_products(quantity)
        nb = blocks.values.shape[0]
        if nb < 2:
            raise StatisticsError("block spread needs at least two blocks")
        values = np.std(blocks.values, axis=0, ddof=1)
        return Product(quantity, f"{blocks.long_name.replace(' (per block)', '')}: "
                       f"block-to-block spread ({nb} blocks of "
                       f"{self.config['block_days']} days)", blocks.units, values,
                       self._coordinates(),
                       ("zonal_mean", "time_mean", "block_spread"), WEIGHTING,
                       self._interval(), self._source(),
                       "sample standard deviation (ddof 1) of the per-block time means; "
                       "a variability diagnostic of this run's blocks")

    def series(self, name: str) -> Product:
        try:
            key, long_name, units, definition = SERIES_QUANTITIES[name]
        except KeyError:
            raise StatisticsError(
                f"unknown series {name!r}; series: {', '.join(SERIES_QUANTITIES)}") from None
        day = self._arrays["series_day"]
        return Product(name, long_name, units, self._arrays[f"series_{key}"],
                       {"day": day}, ("global_mean",), "Gaussian-weighted",
                       {"days": (int(day[0]), int(day[-1])), "samples": int(day.size),
                        "cadence": "1 per day"},
                       self._source(), definition)

    # -- discovery --------------------------------------------------------
    def saved_states(self) -> dict:
        states = self.run_dir / "states"
        if not states.is_dir():
            return {"available": False, "reason": "no states/ directory in this run"}
        days = sorted(int(m.group(1)) for p in states.iterdir()
                      if (m := _STATE_FILE.match(p.name)))
        if not days:
            return {"available": False, "reason": "states/ holds no state-dNNNNN.npy files"}
        return {"available": True, "count": len(days), "days": (days[0], days[-1]),
                "note": "full spectral states; evaluating them needs CUDA and is "
                        "not provided by this reader"}

    def sources_for(self, quantity: str, reduction) -> dict:
        """Which saved data can provide ``quantity`` under ``reduction``."""
        reduction = tuple(reduction)
        if quantity in SERIES_QUANTITIES:
            return {"accumulated": reduction in ((), ("global_mean",)),
                    "saved_states": self.saved_states()}
        if quantity not in ZONAL_QUANTITIES:
            raise StatisticsError(
                f"{quantity!r} is not available from any saved data of this run "
                f"(not in the accumulators: {', '.join(ZONAL_QUANTITIES)}; not derivable "
                "from the saved states by this reader)")
        accumulated = reduction in (("zonal_mean", "time_mean"),
                                    ("zonal_mean", "time_mean", "per_block"),
                                    ("zonal_mean", "time_mean", "block_spread"))
        return {"accumulated": accumulated, "saved_states": self.saved_states()}

    def quantities(self) -> list[dict]:
        """Rows in the shape ``Simulation.quantities`` uses, plus ``sources``."""
        states = self.saved_states()
        rows = []
        for ident, (_, long_name, units, definition) in ZONAL_QUANTITIES.items():
            rows.append({"id": ident, "long_name": long_name, "units": units,
                         "kind": "accumulated", "shape": "section",
                         "cadence": "1 per day, days %d-%d" % self.window,
                         "needs": "host", "levels": True, "available": True,
                         "note": "", "definition": definition,
                         "sources": {"accumulated": ["zonal_mean", "time_mean"],
                                     "saved_states": states}})
        for ident, (_, long_name, units, definition) in SERIES_QUANTITIES.items():
            rows.append({"id": ident, "long_name": long_name, "units": units,
                         "kind": "accumulated", "shape": "series",
                         "cadence": "1 per day", "needs": "host", "levels": False,
                         "available": True, "note": "", "definition": definition,
                         "sources": {"accumulated": ["global_mean"],
                                     "saved_states": states}})
        return rows

    def __repr__(self) -> str:
        return (f"AccumulatedStatistics({str(self.run_dir)!r}, "
                f"experiment={self.experiment_id!r}, blocks={self.counts.shape[0]})")


def open_statistics(run_path) -> AccumulatedStatistics:
    """Open a Held–Suarez run directory's accumulated statistics (host only)."""
    return AccumulatedStatistics(run_path)
```

Then export from `src/tropoi/representation/archive/__init__.py`: add

```python
from tropoi.representation.archive.statistics import (
    AccumulatedStatistics, Product, StatisticsError, is_statistics_run,
    open_statistics)
```

and the five names to `__all__`.

- [ ] **Step 4: Run the tests**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accumulated_statistics.py -q`
Expected: `11 passed`

- [ ] **Step 5: Check layering, regenerate architecture, run the layout tests**

Run: `./venv/Scripts/python.exe -m tools.architecture.generate && ./venv/Scripts/python.exe -m tools.architecture.generate --check && ./venv/Scripts/python.exe -m pytest tests/test_layout_compat.py -q`
Expected: `--check` exits 0; layout tests pass (the new module imports nothing from `run`/`cli`).

- [ ] **Step 6: Commit**

```bash
git add src/tropoi/representation/archive/statistics.py src/tropoi/representation/archive/__init__.py tests/test_accumulated_statistics.py docs/architecture/current
git commit -m "feat(archive): host-only reader of Held–Suarez accumulated statistics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `SectionField` and `SectionSpec`

**Files:**
- Modify: `src/tropoi/representation/visual/fields.py` (append)
- Modify: `src/tropoi/representation/visual/specs.py` (append after `LineWidthKeySpec`, extend `PanelSpec`)
- Modify: `src/tropoi/representation/visual/timeline.py:310-335`
- Test: `tests/test_section_panel.py`

**Interfaces:**
- Produces:
  - `SectionField(values, latitudes_deg, sigma_full, sigma_interfaces, *, name, units)` with `.values (nlat, nlev)`, `.latitudes_deg` (monotonic, either direction), `.sigma_full`, `.sigma_interfaces` (`nlev+1`, from 0 to 1).
  - `SectionContourLayer(field: SectionField, levels: tuple[float, ...], color="#3b3b3b", line_width=0.6, line_style="solid", label=None)`
  - `SectionSpec(title, field: SectionField, normalization=NormalizationPolicy.automatic(), normalization_group=None, color_policy="viridis", label=None, contours=(), colorbar=True, label_latitudes=True, label_sigma=True, corner_text=None)`; `.display_label`.
  - `_normalization_slots` yields a `"self"` slot for `SectionSpec`, so `resolve_figure_normalizations` and `ColorKeySpec` groups work unchanged.

- [ ] **Step 1: Write the failing tests**

```python
"""Latitude × sigma section panels: field, spec, group resolution, rendering."""
from __future__ import annotations

import numpy as np
import pytest

from tropoi.representation.visual.fields import SectionField
from tropoi.representation.visual.normalization import NormalizationPolicy
from tropoi.representation.visual.specs import (
    ColorKeySpec, FigureSpec, PanelPlacement, SectionContourLayer, SectionSpec)
from tropoi.representation.visual.timeline import resolve_figure_normalizations

LAT = np.array([60.0, 20.0, -20.0, -60.0])
SIG = np.array([0.125, 0.375, 0.625, 0.875])
EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def field(values=None, name="zonal wind", units="m s^-1"):
    if values is None:
        values = np.arange(16.0).reshape(4, 4) - 8.0
    return SectionField(values, LAT, SIG, EDGES, name=name, units=units)


def test_section_field_validates_shapes_and_coordinates():
    f = field()
    assert f.values.shape == (4, 4) and not f.values.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        SectionField(np.zeros((3, 4)), LAT, SIG, EDGES, name="x", units="1")
    with pytest.raises(ValueError, match="interfaces"):
        SectionField(np.zeros((4, 4)), LAT, SIG, EDGES[:-1], name="x", units="1")
    with pytest.raises(ValueError, match="monotonic"):
        SectionField(np.zeros((4, 4)), LAT[[0, 2, 1, 3]], SIG, EDGES, name="x", units="1")
    with pytest.raises(ValueError, match="finite"):
        SectionField(np.full((4, 4), np.nan), LAT, SIG, EDGES, name="x", units="1")


def test_section_spec_label_and_contours():
    spec = SectionSpec("u", field(), contours=(SectionContourLayer(
        field(name="temperature", units="K"), (-4.0, 0.0, 4.0)),))
    assert spec.display_label == "zonal wind (m s^-1)"
    assert spec.contours[0].levels == (-4.0, 0.0, 4.0)
    with pytest.raises(ValueError, match="increasing"):
        SectionContourLayer(field(), (1.0, 0.0))


def test_sections_in_one_group_share_a_scale_and_the_key_shows_it():
    a = SectionSpec("a", field(np.full((4, 4), 1.0)),
                    normalization=NormalizationPolicy.symmetric(),
                    normalization_group="g")
    b = SectionSpec("b", field(np.full((4, 4), -5.0)),
                    normalization=NormalizationPolicy.symmetric(),
                    normalization_group="g")
    key = ColorKeySpec("g", "shared", normalization=NormalizationPolicy.symmetric())
    figure = FigureSpec(panels=(PanelPlacement(a, 0, 0), PanelPlacement(b, 0, 1),
                                PanelPlacement(key, 1, 0, column_span=2)),
                        rows=2, columns=2, size_inches=(6.0, 4.0))
    resolved = resolve_figure_normalizations(figure)
    limits = {np.asarray(p.panel.normalization.resolve(np.zeros(1)).vmax).item()
              for p in resolved.panels}
    assert limits == {5.0}


def test_section_renders_to_png_with_metadata(tmp_path):
    from PIL import Image
    from tropoi.representation.visual.matplotlib_renderer import MatplotlibRenderer
    spec = SectionSpec("zonal wind, zonal and time mean", field(),
                       contours=(SectionContourLayer(field(), (0.0,)),),
                       corner_text="days 201–1200")
    figure = FigureSpec(panels=(PanelPlacement(spec, 0, 0),), rows=1, columns=1,
                        size_inches=(4.0, 3.0), dpi=100)
    out = MatplotlibRenderer().render_figure(figure, tmp_path / "s.png",
                                             metadata={"Software": "test"})
    with Image.open(out) as image:
        assert image.size == (400, 300)
        assert image.info["Software"] == "test"


def test_sigma_edges_helper_orders_surface_at_the_bottom():
    from tropoi.representation.visual.matplotlib_renderer import _section_edges
    lat_edges, sig_edges, values = _section_edges(field())
    assert lat_edges[0] == -90.0 and lat_edges[-1] == 90.0 and lat_edges.size == 5
    np.testing.assert_array_equal(sig_edges, EDGES)
    # rows are south-to-north after the helper, columns are sigma top-to-bottom
    assert values.shape == (4, 4)
    np.testing.assert_array_equal(values[0], field().values[-1])
```

- [ ] **Step 2: Run to verify failure**

Run: `./venv/Scripts/python.exe -m pytest tests/test_section_panel.py -q`
Expected: `ImportError: cannot import name 'SectionField'`

- [ ] **Step 3: Add `SectionField` to `fields.py`**

Append:

```python
class SectionField:
    """A scalar on latitude × sigma: a zonal-mean or single-meridian section.

    ``values`` has shape ``(nlat, nlev)``; ``latitudes_deg`` is strictly
    monotonic in either direction; ``sigma_full`` holds the full-level
    sigma = p / p_s values (top to bottom) and ``sigma_interfaces`` the
    ``nlev + 1`` interface values from exactly 0 to exactly 1, which are
    the cell edges a renderer draws. Sigma is neither pressure nor height.
    """

    __slots__ = ("values", "latitudes_deg", "sigma_full", "sigma_interfaces",
                 "name", "units")

    def __init__(self, values, latitudes_deg, sigma_full, sigma_interfaces,
                 *, name: str, units: str) -> None:
        values = np.array(values, dtype=np.float64)
        lat = np.array(latitudes_deg, dtype=np.float64)
        full = np.array(sigma_full, dtype=np.float64)
        edges = np.array(sigma_interfaces, dtype=np.float64)
        if values.ndim != 2 or values.shape != (lat.size, full.size):
            raise ValueError(
                f"section values must have shape (nlat, nlev) = "
                f"{(lat.size, full.size)}, got {values.shape}")
        if edges.shape != (full.size + 1,) or edges[0] != 0.0 or edges[-1] != 1.0 \
                or not np.all(np.diff(edges) > 0.0):
            raise ValueError("sigma interfaces must be nlev + 1 values increasing "
                             "from exactly 0 to exactly 1")
        if lat.size > 1 and not (np.all(np.diff(lat) > 0.0)
                                 or np.all(np.diff(lat) < 0.0)):
            raise ValueError("section latitudes must be strictly monotonic")
        if not (np.isfinite(values).all() and np.isfinite(lat).all()):
            raise ValueError("section values and latitudes must be finite")
        for array in (values, lat, full, edges):
            array.flags.writeable = False
        self.values, self.latitudes_deg = values, lat
        self.sigma_full, self.sigma_interfaces = full, edges
        self.name, self.units = str(name), str(units)
```

Add `"SectionField"` to the module's `__all__` if one exists (check with `grep -n __all__ src/tropoi/representation/visual/fields.py`; if none, skip).

- [ ] **Step 4: Add the specs**

In `specs.py`, import `SectionField` alongside `ScalarGridField` at the top, then append after `LineWidthKeySpec`:

```python
@dataclass(frozen=True)
class SectionContourLayer:
    """Contour lines of a section quantity at explicit levels."""

    field: SectionField
    levels: tuple[float, ...]
    color: str = "#3b3b3b"
    line_width: float = 0.6
    line_style: str = "solid"
    label: str | None = None

    def __post_init__(self) -> None:
        levels = tuple(float(level) for level in self.levels)
        if not levels or not all(np.isfinite(levels)):
            raise ValueError("contour levels must be a nonempty finite sequence")
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("contour levels must be strictly increasing")
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True)
class SectionSpec:
    """One latitude × sigma section: filled field, contours, colour key.

    Drawn with the surface (sigma = 1) at the bottom and cells spanning the
    sigma interfaces; latitude cells are centred on their samples and closed
    at the poles. The axis is sigma = p / p_s, never pressure or height.
    """

    title: str
    field: SectionField
    normalization: NormalizationPolicy = field(
        default_factory=NormalizationPolicy.automatic)
    normalization_group: str | None = None
    color_policy: str = "viridis"
    label: str | None = None
    contours: tuple[SectionContourLayer, ...] = ()
    colorbar: bool = True
    label_latitudes: bool = True
    label_sigma: bool = True
    corner_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str):
            raise TypeError("section title must be a string")
        _check_group(self.normalization_group)
        object.__setattr__(self, "contours", tuple(self.contours))

    @property
    def display_label(self) -> str:
        if self.label is not None:
            return self.label
        return f"{self.field.name} ({self.field.units})"
```

Extend the union:

```python
PanelSpec: TypeAlias = (
    ScalarMapSpec | SpectralCoefficientMapSpec | StreamlineMapSpec |
    TextPanelSpec | LinePanelSpec | LayeredMapSpec | ColorKeySpec |
    LineWidthKeySpec | SectionSpec)
```

- [ ] **Step 5: Teach the resolver**

In `timeline.py`, import `SectionSpec` from specs and in `_normalizable_values(panel)` (line ≈395) add before the first `if`:

```python
    if isinstance(panel, SectionSpec):
        return np.asarray(panel.field.values)
```

(`_normalization_slots` already falls through to the `"self"` slot, and `_with_normalization` already does `replace(panel, normalization=frozen)`.)

- [ ] **Step 6: Run the tests (renderer ones still fail)**

Run: `./venv/Scripts/python.exe -m pytest tests/test_section_panel.py -q`
Expected: 3 passed, 2 failed (`_section_edges` missing; renderer raises `TypeError: unsupported panel specification SectionSpec`).

- [ ] **Step 7: Commit the data side**

```bash
git add src/tropoi/representation/visual/fields.py src/tropoi/representation/visual/specs.py src/tropoi/representation/visual/timeline.py tests/test_section_panel.py
git commit -m "feat(visual): SectionField and SectionSpec for latitude-sigma sections

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Render sections

**Files:**
- Modify: `src/tropoi/representation/visual/matplotlib_renderer.py` (imports, `_render_panel`, new `_render_section`, new `_section_edges`)
- Test: `tests/test_section_panel.py` (written in Task 3)

**Interfaces:**
- Produces: `_section_edges(field: SectionField) -> (lat_edges_s2n, sigma_edges, values_s2n)` where rows run south→north, `lat_edges` closes at ±90 and `values_s2n[i, k]` is the cell between `lat_edges[i:i+2]` and `sigma_edges[k:k+2]`.

- [ ] **Step 1: Add the helper (beside `_latitude_edges`)**

```python
def _section_edges(field):
    """South-to-north latitude edges closed at the poles, sigma interface
    edges, and the values reordered to match (rows south to north)."""
    lat = np.asarray(field.latitudes_deg, dtype=np.float64)
    values = np.asarray(field.values, dtype=np.float64)
    if lat.size > 1 and lat[0] > lat[-1]:
        lat, values = lat[::-1], values[::-1]
    return _latitude_edges(lat), np.asarray(field.sigma_interfaces), values
```

- [ ] **Step 2: Add the renderer method and dispatch**

Import `SectionSpec` in the `specs` import list. In `_render_panel`, add before the final `else`:

```python
        elif isinstance(panel, SectionSpec):
            self._render_section(figure, axes, panel)
```

Add the method after `_render_layered_map`:

```python
    def _render_section(self, figure, axes, spec: SectionSpec) -> None:
        lat_edges, sigma_edges, values = _section_edges(spec.field)
        _, norm = self._mpl_normalization(spec.normalization, values)
        # pcolormesh wants (rows=y, columns=x): y is sigma, x is latitude.
        mesh = axes.pcolormesh(lat_edges, sigma_edges, values.T,
                               cmap=self._color_map(spec.color_policy), norm=norm,
                               shading="flat", rasterized=True, zorder=0)
        for layer in spec.contours:
            c_lat = np.asarray(layer.field.latitudes_deg, dtype=np.float64)
            c_values = np.asarray(layer.field.values, dtype=np.float64)
            if c_lat.size > 1 and c_lat[0] > c_lat[-1]:
                c_lat, c_values = c_lat[::-1], c_values[::-1]
            axes.contour(c_lat, np.asarray(layer.field.sigma_full), c_values.T,
                         levels=list(layer.levels), colors=layer.color,
                         linewidths=layer.line_width, linestyles=layer.line_style,
                         zorder=2)
        axes.set_xlim(-90.0, 90.0)
        axes.set_ylim(1.0, 0.0)                      # surface at the bottom
        axes.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        if spec.label_latitudes:
            axes.set_xticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"])
            axes.set_xlabel("latitude")
        else:
            axes.set_xticklabels([])
        if spec.label_sigma:
            axes.set_ylabel("σ = p / p_s")
        else:
            axes.set_yticklabels([])
        if spec.title:
            axes.set_title(spec.title)
        if spec.corner_text:
            axes.text(0.012, 0.975, spec.corner_text, transform=axes.transAxes,
                      ha="left", va="top", fontsize="small", zorder=5,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                            "edgecolor": "none", "alpha": 0.8})
        if spec.colorbar:
            figure.colorbar(mesh, ax=axes, orientation="vertical", pad=0.02,
                            fraction=0.05, aspect=25, label=spec.display_label)
```

- [ ] **Step 3: Run the tests**

Run: `./venv/Scripts/python.exe -m pytest tests/test_section_panel.py tests/test_saved_run_plotting.py -q -k "section or lazy"`
Expected: section tests `5 passed`; the laziness probe still passes (no new eager imports).

- [ ] **Step 4: Commit**

```bash
git add src/tropoi/representation/visual/matplotlib_renderer.py
git commit -m "feat(visual): render latitude-sigma sections with surface at the bottom

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Statistics views and composition

**Files:**
- Modify: `src/tropoi/representation/visual/views.py` (append five classes, extend `__all__`)
- Create: `src/tropoi/representation/visual/statistics_figures.py`
- Test: `tests/test_statistics_views.py`
- Regenerate: `docs/architecture/current/`

**Interfaces:**
- Consumes: `AccumulatedStatistics`, `Product` (Task 2); `SectionField`, `SectionSpec`, `SectionContourLayer` (Task 3); `Style`, `describe`, `Contours` (existing views); `LinePanelSpec`, `LineSeriesSpec`, `TextPanelSpec`, `ColorKeySpec`, `PanelPlacement`, `FigureSpec` (existing specs); `resolve_figure_normalizations`; `compose.pretty_units`, `compose.ASSETS_DIRNAME`.
- Produces (views):
  - `ZonalMean()` (frozen, no fields)
  - `TimeMean(days: tuple[int, int] | None = None, source: str = "accumulated")`; `source` ∈ {"accumulated"} for now (`"snapshots"`, `"auto"` raise `ValueError("not implemented in this version")` so recipes cannot pin them yet).
  - `Section(quantity: str, across: ZonalMean = ZonalMean(), time: TimeMean = TimeMean(), contours: tuple[Contours, ...] = (), color_policy: str | None = None, symmetric: bool | None = None, limits: tuple[float, float] | None = None, show_block_spread: bool = False)`
  - `DailySeries(quantities: tuple[str, ...], title: str | None = None, mark_blocks: bool = True)`
  - `Panels(rows: tuple[tuple[Section | DailySeries, ...], ...], title: str | None = None, style: Style = Style())`
- Produces (figures):
  - `default_statistics_view(stats) -> Panels`
  - `compose_statistics_view(stats, view) -> tuple[FigureSpec, dict]`
  - `render_statistics_view(stats, view=None, output_path=None, *, sidecar=False, renderer=None) -> pathlib.Path`
  - `AccumulatedStatistics.plot(output_path=None, view=None, *, sidecar=False, renderer=None)` added to Task 2's class, lazily importing `statistics_figures`.
  - `SECTION_DISPLAY_DEFAULTS: dict[str, tuple[str, bool]]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Views over accumulated statistics: composition, labels, sidecar, paths."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from statistics_fixture import write_statistics_run  # noqa: E402

from tropoi.representation.archive import open_statistics  # noqa: E402
from tropoi.representation.visual.specs import (  # noqa: E402
    ColorKeySpec, LinePanelSpec, SectionSpec, TextPanelSpec)
from tropoi.representation.visual.statistics_figures import (  # noqa: E402
    compose_statistics_view, default_statistics_view, render_statistics_view)
from tropoi.representation.visual.views import (  # noqa: E402
    Contours, DailySeries, Panels, Section, TimeMean, ZonalMean, describe)


@pytest.fixture
def stats(tmp_path):
    return open_statistics(write_statistics_run(tmp_path))


def test_time_mean_rejects_sources_this_version_cannot_pin():
    assert TimeMean().source == "accumulated"
    with pytest.raises(ValueError, match="not implemented"):
        TimeMean(source="snapshots")
    with pytest.raises(ValueError, match="not implemented"):
        TimeMean(source="auto")


def test_default_view_is_statistics_only_and_says_so(stats):
    view = default_statistics_view(stats)
    assert isinstance(view, Panels)
    kinds = {type(p).__name__ for row in view.rows for p in row}
    assert kinds == {"Section", "DailySeries"}
    figure, record = compose_statistics_view(stats, view)
    texts = [p.panel.text for p in figure.panels if isinstance(p.panel, TextPanelSpec)]
    assert any("longitudinal structure not shown" in t for t in texts)
    assert any("days 2–5" in t and "4 daily samples" in t for t in texts)
    assert record["omitted"] == [
        "maps: no longitude-resolved field is in the accumulators; "
        "this run has no states/ directory"]


def test_section_panel_labels_reduction_window_and_weighting(stats):
    view = Panels(rows=((Section("zonal_wind",
                                 contours=(Contours("temperature", (250.0, 252.0)),)),),))
    figure, record = compose_statistics_view(stats, view)
    sections = [p.panel for p in figure.panels if isinstance(p.panel, SectionSpec)]
    assert len(sections) == 1
    section = sections[0]
    assert section.title.startswith("zonal wind")
    assert "zonal mean" in section.title and "time mean" in section.title
    assert "days 2–5" in section.corner_text
    assert "unweighted on σ" in section.corner_text
    assert section.contours[0].levels == (250.0, 252.0)
    product = record["products"][0]
    assert product["quantity"] == "zonal_wind"
    assert product["reduction"] == ["zonal_mean", "time_mean"]
    assert product["source"]["kind"] == "accumulated"
    assert product["stats"]["max"] == pytest.approx(5.0 + 7 + 0.3)
    assert "contours" in product and product["contours"][0]["quantity"] == "temperature"


def test_block_spread_is_a_separate_labelled_layer(stats):
    view = Panels(rows=((Section("zonal_wind", show_block_spread=True),),))
    figure, record = compose_statistics_view(stats, view)
    section = next(p.panel for p in figure.panels if isinstance(p.panel, SectionSpec))
    layer = section.contours[-1]
    assert "block-to-block spread" in layer.label
    assert record["products"][0]["block_spread"]["definition"].startswith(
        "sample standard deviation")
    dumped = json.dumps(record).lower()
    assert "uncertainty" not in dumped and "confidence" not in dumped


def test_daily_series_panel_is_a_line_with_block_boundaries(stats):
    view = Panels(rows=((DailySeries(("kinetic_energy", "max_wind_speed")),),))
    figure, record = compose_statistics_view(stats, view)
    panel = next(p.panel for p in figure.panels if isinstance(p.panel, LinePanelSpec))
    assert [s.label for s in panel.series][0].startswith("global kinetic energy")
    assert all(s.draw_line for s in panel.series)
    assert "blocks" in " ".join(panel.notes)
    assert record["series"]["kinetic_energy"]["cadence"] == "1 per day"


def test_explicit_requests_the_run_cannot_serve_raise(stats):
    from tropoi.representation.archive import StatisticsError
    with pytest.raises(StatisticsError, match="not available from any saved data"):
        compose_statistics_view(stats, Panels(rows=((Section("mass_streamfunction"),),)))
    with pytest.raises(StatisticsError, match="unknown series"):
        compose_statistics_view(stats, Panels(rows=((DailySeries(("wind",)),),)))


def test_render_writes_png_sidecar_and_only_under_assets(stats, tmp_path):
    from PIL import Image
    out = render_statistics_view(stats, sidecar=True)
    assert out == stats.run_dir / "assets" / "climatology.png"
    assert out.with_suffix(".json").is_file()
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["run"]["statistics_sha256"] == stats.statistics_sha256
    assert side["run"]["config_sha256"] == stats.config_sha256
    assert side["view"] == describe(default_statistics_view(stats))
    with Image.open(out) as image:
        info = dict(image.info)
    assert info["StatisticsSHA256"] == stats.statistics_sha256
    assert info["ConfigSHA256"] == stats.config_sha256
    assert json.loads(info["View"]) == side["view"]
    with pytest.raises(ValueError, match="only assets/"):
        render_statistics_view(stats, output_path=stats.run_dir / "x.png")
    with pytest.raises(ValueError, match="output path"):
        render_statistics_view(stats, view=Panels(rows=((Section("temperature"),),)))
    elsewhere = render_statistics_view(
        stats, Panels(rows=((Section("temperature"),),)), tmp_path / "T.png")
    assert elsewhere.is_file()
    assert stats.plot(tmp_path / "again.png", Panels(rows=((Section("temperature"),),))).is_file()
```

- [ ] **Step 2: Run to verify failure**

Run: `./venv/Scripts/python.exe -m pytest tests/test_statistics_views.py -q`
Expected: `ImportError` on `statistics_figures` / `DailySeries`.

- [ ] **Step 3: Add the view objects**

Append to `views.py` (and add the five names to `__all__`):

```python
@dataclass(frozen=True)
class ZonalMean:
    """Average over longitude: the only ``across`` a section supports today.

    A zonal mean hides every longitudinal structure, stationary or
    transient; figures say "zonal mean" wherever one is drawn.
    """


@dataclass(frozen=True)
class TimeMean:
    """Average over days ``days`` (inclusive; ``None`` = the run's averaging
    window) from ``source``. Only ``"accumulated"`` exists in this version:
    the online block accumulators. ``"snapshots"`` (means of saved states)
    and ``"auto"`` are reserved and refused until they are implemented, so a
    recipe cannot pin a source that does not exist yet.
    """

    days: tuple[int, int] | None = None
    source: str = "accumulated"

    def __post_init__(self) -> None:
        if self.source in ("snapshots", "auto"):
            raise ValueError(f"TimeMean(source={self.source!r}) is not implemented "
                             "in this version; only 'accumulated' is available")
        if self.source != "accumulated":
            raise ValueError(f"unknown TimeMean source {self.source!r}")
        if self.days is not None:
            first, last = (int(v) for v in self.days)
            if first > last:
                raise ValueError("TimeMean days must be (first, last) with first <= last")
            object.__setattr__(self, "days", (first, last))


@dataclass(frozen=True)
class Section:
    """A latitude × sigma section of one quantity.

    ``across`` names the horizontal reduction (``ZonalMean()``), ``time``
    the temporal one; both are printed on the panel. ``show_block_spread``
    adds the block-to-block spread of the time mean as a labelled contour
    layer: a variability diagnostic, never a bound.
    """

    quantity: str
    across: ZonalMean = field(default_factory=ZonalMean)
    time: TimeMean = field(default_factory=TimeMean)
    contours: tuple[Contours, ...] = ()
    color_policy: str | None = None
    symmetric: bool | None = None
    limits: tuple[float, float] | None = None
    show_block_spread: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.across, ZonalMean):
            raise TypeError("only ZonalMean() sections exist in this version")
        object.__setattr__(self, "contours", tuple(self.contours))


@dataclass(frozen=True)
class DailySeries:
    """Daily scalar series of a run drawn as lines (cadence: 1 per day),
    with the averaging blocks marked when ``mark_blocks``."""

    quantities: tuple[str, ...]
    title: str | None = None
    mark_blocks: bool = True

    def __post_init__(self) -> None:
        quantities = tuple(self.quantities)
        if not quantities:
            raise ValueError("a daily-series panel needs at least one quantity")
        object.__setattr__(self, "quantities", quantities)


@dataclass(frozen=True)
class Panels:
    """Rows of statistics panels with no saved-time axis (each panel states
    its own window). Sections of one quantity share a colour scale."""

    rows: tuple[tuple[Section | DailySeries, ...], ...]
    title: str | None = None
    style: Style = field(default_factory=Style)

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.rows)
        if not rows or not all(rows):
            raise ValueError("Panels needs at least one panel in every row")
        object.__setattr__(self, "rows", rows)
```

- [ ] **Step 4: Write `statistics_figures.py`**

```python
"""Compose statistics views (``Panels`` of ``Section`` / ``DailySeries``)
into figures for a Held–Suarez run's accumulated statistics.

Everything drawn comes from :class:`~tropoi.representation.archive.
statistics.AccumulatedStatistics` products, so every panel carries its
reduction, window, sample count, cadence and weighting, and the sidecar
records each product's full description with the hashes of the files read.
Host only: no model, no CUDA.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from tropoi.representation.archive.statistics import (
    AccumulatedStatistics, Product, SERIES_QUANTITIES, StatisticsError,
    ZONAL_QUANTITIES)
from tropoi.representation.visual.compose import ASSETS_DIRNAME, pretty_units
from tropoi.representation.visual.fields import SectionField
from tropoi.representation.visual.normalization import NormalizationPolicy
from tropoi.representation.visual.specs import (
    ColorKeySpec, FigureSpec, LinePanelSpec, LineSeriesSpec, PanelPlacement,
    SectionContourLayer, SectionSpec, TextPanelSpec)
from tropoi.representation.visual.timeline import resolve_figure_normalizations
from tropoi.representation.visual.views import (
    DailySeries, Panels, Section, Style, describe)

__all__ = ["SECTION_DISPLAY_DEFAULTS", "compose_statistics_view",
           "default_statistics_view", "render_statistics_view"]

#: (colour policy, symmetric about zero) per quantity.
SECTION_DISPLAY_DEFAULTS = {
    "zonal_wind": ("RdBu_r", True),
    "meridional_wind": ("RdBu_r", True),
    "temperature": ("inferno:0.25:1.0", False),
    "eddy_temperature_variance": ("YlOrRd", False),
    "eddy_zonal_wind_variance": ("YlOrRd", False),
    "eddy_heat_flux": ("RdBu_r", True),
    "eddy_momentum_flux": ("RdBu_r", True),
}
_SERIES_COLORS = ("#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad", "#d35400")


def default_statistics_view(stats: AccumulatedStatistics) -> Panels:
    """Zonal-mean climatology sections and the daily series of a run."""
    return Panels(rows=(
        (Section("zonal_wind", contours=()), Section("temperature")),
        (Section("eddy_temperature_variance"), Section("eddy_heat_flux")),
        (DailySeries(("kinetic_energy",)), DailySeries(("max_wind_speed",))),
    ))


def _window_text(product: Product) -> str:
    first, last = product.interval["days"]
    return (f"days {first}–{last}, {product.interval['samples']} daily samples "
            f"in {product.interval['blocks']} blocks")


def _section_field(product: Product) -> SectionField:
    return SectionField(product.values, product.coordinates["latitude_deg"],
                        product.coordinates["sigma_full"],
                        product.coordinates["sigma_interfaces"],
                        name=product.long_name, units=product.units)


def _normalization(view: Section) -> NormalizationPolicy:
    if view.limits is not None:
        return NormalizationPolicy.fixed(*view.limits)
    symmetric = view.symmetric
    if symmetric is None:
        symmetric = SECTION_DISPLAY_DEFAULTS.get(view.quantity, ("viridis", False))[1]
    return NormalizationPolicy.symmetric() if symmetric else NormalizationPolicy.automatic()


def _stats(product: Product, weights: np.ndarray, dsigma: np.ndarray) -> dict:
    w = weights[:, None] * dsigma[None, :]
    return {"min": float(product.values.min()), "max": float(product.values.max()),
            "mass_weighted_mean": float((w * product.values).sum() / w.sum()),
            "units": product.units}


def _check_window(stats: AccumulatedStatistics, view: Section) -> None:
    days = view.time.days
    if days is not None and tuple(days) != stats.window:
        raise StatisticsError(
            f"TimeMean(days={days}) is not available from the accumulators, "
            f"whose window is days {stats.window[0]}–{stats.window[1]}; other "
            "windows are not available from any saved data through this reader")


def _section_panel(stats: AccumulatedStatistics, view: Section, group: str
                   ) -> tuple[SectionSpec, dict]:
    stats.sources_for(view.quantity, ("zonal_mean", "time_mean"))   # raises if unknown
    _check_window(stats, view)
    product = stats.product(view.quantity)
    record = {**product.describe(),
              "stats": _stats(product, stats.gauss_weights, stats.dsigma),
              "contours": []}
    contours = []
    for contour in view.contours:
        other = stats.product(contour.quantity)
        contours.append(SectionContourLayer(
            _section_field(other), tuple(contour.levels), color=contour.color,
            line_width=contour.line_width, line_style=contour.line_style,
            label=f"{other.long_name} at {', '.join(f'{v:g}' for v in contour.levels)} "
                  f"{pretty_units(other.units)}"))
        record["contours"].append({**other.describe(), "levels": list(contour.levels)})
    if view.show_block_spread:
        spread = stats.block_spread(view.quantity)
        level = float(np.percentile(spread.values, 75.0))
        contours.append(SectionContourLayer(
            _section_field(spread), (level,), color="#555555", line_width=0.5,
            line_style="dashed",
            label=f"{spread.long_name} at {level:.3g} {pretty_units(spread.units)}"))
        record["block_spread"] = {**spread.describe(), "contour_level": level,
                                  "max": float(spread.values.max())}
    color = view.color_policy or SECTION_DISPLAY_DEFAULTS.get(
        view.quantity, ("viridis", False))[0]
    spec = SectionSpec(
        title=f"{product.long_name} ({pretty_units(product.units)}): zonal mean, time mean",
        field=_section_field(product), normalization=_normalization(view),
        normalization_group=group, color_policy=color,
        label=f"{product.long_name} ({pretty_units(product.units)})",
        contours=tuple(contours), colorbar=True,
        corner_text=f"{_window_text(product)}\nunweighted on σ surfaces")
    return spec, record


def _series_panel(stats: AccumulatedStatistics, view: DailySeries) -> tuple[LinePanelSpec, dict]:
    series, record = [], {}
    for name, color in zip(view.quantities, _SERIES_COLORS):
        product = stats.series(name)
        series.append(LineSeriesSpec(
            product.coordinates["day"], product.values,
            f"{product.long_name} ({pretty_units(product.units)}), 1 per day",
            color=color, line_width=1.0))
        record[name] = {**product.describe(), "cadence": product.interval["cadence"],
                        "first": float(product.values[0]), "last": float(product.values[-1]),
                        "min": float(product.values.min()), "max": float(product.values.max())}
    notes = ()
    if view.mark_blocks:
        windows = stats.block_windows
        notes = (f"{len(windows)} averaging blocks: days "
                 + ", ".join(f"{a}–{b}" for a, b in windows),)
    title = view.title or ("daily series" if len(series) > 1 else series[0].label.split(" (")[0])
    return LinePanelSpec(tuple(series), title, "day", "", notes=notes,
                         legend_location="below"), record


def _header(stats: AccumulatedStatistics, title: str | None, omitted: list) -> tuple[str, str]:
    cfg = stats.config
    heading = title or (f"Held–Suarez climatology · {stats.experiment_id} · "
                        "dry primitive-equation core")
    commit = str(stats.code.get("git_commit") or "unknown")[:8]
    first, last = stats.window
    provenance = (
        f"Gauss–Legendre {cfg['nlat']} × {cfg['nlon']}, ℓ ≤ {cfg['l_max']}"
        + (f" (retained {cfg['retained_truncation']})" if cfg.get("retained_truncation") else "")
        + f", {cfg['nlev']} sigma levels · commit {commit} · config {stats.config_sha256[:8]}\n"
        f"accumulated statistics: days {first}–{last}, {int(stats.counts.sum())} daily samples; "
        "zonal and time means only — longitudinal structure not shown")
    states = stats.saved_states()
    omitted.append("maps: no longitude-resolved field is in the accumulators; "
                   + (f"{states['count']} saved states (days {states['days'][0]}–"
                      f"{states['days'][1]}) could be drawn with Map views once state "
                      "access exists (needs CUDA)" if states["available"]
                      else "this run has no states/ directory"))
    return heading, provenance


def compose_statistics_view(stats: AccumulatedStatistics, view: Panels
                            ) -> tuple[FigureSpec, dict]:
    if not isinstance(view, Panels):
        raise TypeError("statistics views are Panels of Section / DailySeries")
    style: Style = view.style
    record = {"view": describe(view),
              "run": {"experiment_id": stats.experiment_id,
                      "config_sha256": stats.config_sha256,
                      "statistics_sha256": stats.statistics_sha256,
                      "protocol_sha256": stats.protocol_sha256,
                      "git": stats.code, "window_days": list(stats.window),
                      "block_windows": [list(w) for w in stats.block_windows],
                      "counts": stats.counts.tolist()},
              "omitted": [], "products": [], "series": {}}
    columns = max(len(row) for row in view.rows)
    panels: list[PanelPlacement] = []
    heights: list[float] = []
    heading, provenance = _header(stats, view.title, record["omitted"])
    panels.append(PanelPlacement(TextPanelSpec(
        heading, font_family="sans-serif", font_size=style.base_font_size + 2,
        horizontal_alignment="left", font_weight="semibold"), 0, 0, column_span=columns))
    panels.append(PanelPlacement(TextPanelSpec(
        provenance, font_family="sans-serif", font_size=style.base_font_size - 1,
        horizontal_alignment="left", color="#555555"), 1, 0, column_span=columns))
    heights += [0.26, 0.42]
    for r, row in enumerate(view.rows, start=2):
        for c, panel in enumerate(row):
            if isinstance(panel, Section):
                spec, rec = _section_panel(stats, panel, group=f"section-{panel.quantity}")
                record["products"].append(rec)
            elif isinstance(panel, DailySeries):
                spec, rec = _series_panel(stats, panel)
                record["series"].update(rec)
            else:
                raise TypeError(f"unsupported statistics panel {panel!r}")
            panels.append(PanelPlacement(spec, r, c))
        heights.append(style.map_row_height_inches + 0.5
                       if any(isinstance(p, Section) for p in row)
                       else style.diagnostics_height_inches)
    figure = FigureSpec(panels=tuple(panels), rows=2 + len(view.rows), columns=columns,
                        size_inches=(style.width_inches, float(sum(heights))),
                        dpi=style.dpi, height_ratios=tuple(heights),
                        base_font_size=style.base_font_size, constrained_layout=True)
    return resolve_figure_normalizations(figure), record


def _png_metadata(stats: AccumulatedStatistics, record: dict) -> dict:
    return {"Software": "palintropos (tropoi.representation.visual.statistics_figures)",
            "RunId": stats.experiment_id,
            "Solver": "pe",
            "Commit": str(stats.code.get("git_commit") or ""),
            "StatisticsSHA256": stats.statistics_sha256,
            "ConfigSHA256": stats.config_sha256,
            "View": json.dumps(record["view"], sort_keys=True)}


def render_statistics_view(stats: AccumulatedStatistics, view: Panels | None = None,
                           output_path=None, *, sidecar: bool = False,
                           renderer=None) -> pathlib.Path:
    """Draw ``view`` (default: :func:`default_statistics_view`) and return the
    written image path. ``output_path=None`` is allowed for the default view
    only and writes ``RUN/assets/climatology.png``; inside the run directory
    only ``assets/`` is ever written."""
    if output_path is None:
        if view is not None:
            raise ValueError("give an output path for a custom view; only the "
                             "default view has a default location (RUN/assets/climatology.png)")
        output_path = stats.run_dir / ASSETS_DIRNAME / "climatology.png"
    output = pathlib.Path(output_path).resolve()
    run_dir = stats.run_dir.resolve()
    assets = run_dir / ASSETS_DIRNAME
    if (output == run_dir or run_dir in output.parents) and assets not in output.parents:
        raise ValueError(f"refusing to write {output}: inside the run {run_dir} only "
                         f"{ASSETS_DIRNAME}/ holds derived figures")
    if view is None:
        view = default_statistics_view(stats)
    figure, record = compose_statistics_view(stats, view)
    from tropoi.representation.visual.renderers import get_default_renderer
    backend = renderer or get_default_renderer()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = pathlib.Path(backend.render_figure(figure, output,
                                                 metadata=_png_metadata(stats, record)))
    if sidecar:
        written.with_suffix(".json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
    return written
```

Then add to `AccumulatedStatistics` (Task 2's class, after `quantities`):

```python
    def plot(self, output_path=None, view=None, *, sidecar: bool = False,
             renderer=None) -> pathlib.Path:
        """Draw a statistics view (lazy: Matplotlib is imported on the call)."""
        from tropoi.representation.visual.statistics_figures import (
            render_statistics_view)
        return render_statistics_view(self, view, output_path, sidecar=sidecar,
                                      renderer=renderer)
```

- [ ] **Step 5: Run the tests**

Run: `./venv/Scripts/python.exe -m pytest tests/test_statistics_views.py tests/test_accumulated_statistics.py -q`
Expected: all pass. If `test_default_view_is_statistics_only_and_says_so` fails on the `days 2–5` text, check that `_window_text` uses an en dash (`–`) exactly as the test does.

- [ ] **Step 6: Regenerate architecture and commit**

```bash
./venv/Scripts/python.exe -m tools.architecture.generate
git add src/tropoi/representation/visual/views.py src/tropoi/representation/visual/statistics_figures.py src/tropoi/representation/archive/statistics.py tests/test_statistics_views.py docs/architecture/current
git commit -m "feat(visual): Section, DailySeries and Panels views over accumulated statistics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `tropoi plot` on a Held–Suarez directory

**Files:**
- Modify: `src/tropoi/cli/main.py` (`_cmd_plot`, `_print_quantities`, plot parser, `_PLOT_EXAMPLES`)
- Test: `tests/cli/test_plot_statistics.py`

**Interfaces:**
- Consumes: `is_statistics_run`, `open_statistics` (Task 2); `Panels`, `Section`, `DailySeries`, `TimeMean`, `Contours` (Task 5).
- Produces: flags `--section QUANTITY` (repeatable; one row of sections), `--series Q1,Q2` (repeatable; one panel per flag), `--mean FIRST:LAST` (days; must equal the accumulators' window in this version), `--block-spread`. On an HS directory, `--map/--contours/--vectors/--level/--sigma/--snapshots/--at/--max-maps` are refused with the sources message.

- [ ] **Step 1: Write the failing tests**

```python
"""``tropoi plot`` on a Held–Suarez statistics directory (CPU only)."""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from statistics_fixture import write_statistics_run  # noqa: E402

from tropoi.cli.main import main  # noqa: E402


@pytest.fixture
def run(tmp_path):
    return write_statistics_run(tmp_path)


def test_list_quantities_names_sources_without_heavy_imports(run):
    code = ("import sys; from tropoi.cli.main import main; "
            "rc = main(['plot', sys.argv[1], '--list-quantities']); "
            "print('HEAVY', [m for m in sys.modules if m.startswith(('cupy', 'matplotlib', "
            "'tropoi.run', 'tropoi.temporal.tendencies'))]); sys.exit(rc)")
    out = subprocess.run([sys.executable, "-c", code, str(run)],
                         capture_output=True, text=True, check=True)
    assert "accumulated statistics" in out.stdout
    assert "eddy_heat_flux" in out.stdout and "zonal mean, time mean" in out.stdout
    assert "saved states: none" in out.stdout
    assert "HEAVY []" in out.stdout


def test_default_plot_writes_the_climatology_asset(run, capsys):
    assert main(["plot", str(run), "--no-open", "--sidecar"]) == 0
    assert (run / "assets" / "climatology.png").is_file()
    assert (run / "assets" / "climatology.json").is_file()
    assert "climatology.png" in capsys.readouterr().out


def test_section_and_series_flags_build_panels(run):
    assert main(["plot", str(run), "--section", "zonal_wind", "--section", "temperature",
                 "--series", "kinetic_energy,max_wind_speed", "--mean", "2:5",
                 "--block-spread", "--no-open"]) == 0
    written = list((run / "assets").glob("*.png"))
    assert len(written) == 1
    assert "section-zonal_wind" in written[0].name and "block-spread" in written[0].name


def test_map_options_are_refused_with_the_sources_message(run, capsys):
    assert main(["plot", str(run), "--map", "temperature", "--no-open"]) == 2
    err = capsys.readouterr().err
    assert "not in the accumulators" in err
    assert "no states/ directory" in err
    assert main(["plot", str(run), "--mean", "1:5", "--no-open"]) == 2
    assert "window is days 2–5" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify failure**

Run: `./venv/Scripts/python.exe -m pytest tests/cli/test_plot_statistics.py -q`
Expected: failures (`--section` unrecognised; default plot errors on `cannot open`).

- [ ] **Step 3: Implement the routing**

In `build_parser()`'s plot section add, after `--no-diagnostics`:

```python
    plot_parser.add_argument(
        "--section", action="append", default=None, metavar="QUANTITY",
        help="Held–Suarez statistics runs: a zonal-mean, time-mean latitude–sigma "
             "section of QUANTITY from the accumulators (repeatable; one row).")
    plot_parser.add_argument(
        "--series", action="append", default=None, metavar="Q1,Q2",
        help="Statistics runs: one panel of daily series (repeatable).")
    plot_parser.add_argument(
        "--mean", default=None, metavar="FIRST:LAST",
        help="Statistics runs: the averaging window in days; must be the "
             "accumulators' window in this version (it is printed on the figure).")
    plot_parser.add_argument(
        "--block-spread", action="store_true",
        help="Statistics runs: add each section's block-to-block spread as a "
             "labelled dashed contour (a variability diagnostic).")
```

Add to `_PLOT_EXAMPLES`:

```
  tropoi plot runs/hs-T42L20-prod-002                   Held–Suarez climatology from statistics.npz (no GPU)
  tropoi plot RUN --section zonal_wind --section temperature --series kinetic_energy
```

Add a helper and a branch at the top of `_cmd_plot`, right after `run_dir` is resolved:

```python
def _statistics_view(args):
    """Panels from the --section/--series/--mean/--block-spread options."""
    from tropoi.representation.visual.views import (
        DailySeries, Panels, Section, TimeMean)
    time = TimeMean()
    if args.mean is not None:
        first, _, last = args.mean.partition(":")
        time = TimeMean(days=(int(first), int(last)))
    rows = []
    if args.section:
        rows.append(tuple(Section(q, time=time, show_block_spread=args.block_spread)
                          for q in args.section))
    for text in args.series or ():
        rows.append((DailySeries(tuple(s.strip() for s in text.split(",") if s.strip())),))
    return Panels(rows=tuple(rows)) if rows else None


def _statistics_asset_name(args) -> str:
    import re
    parts = ["climatology"]
    for flag, value in (("section", "+".join(args.section or ()) or None),
                        ("series", "+".join(args.series or ()) or None),
                        ("mean", args.mean)):
        if value is not None:
            parts.append(f"{flag}-{value}")
    if args.block_spread:
        parts.append("block-spread")
    return re.sub(r"[^A-Za-z0-9_.,+=:-]", "-", "_".join(parts)).replace(":", "-") + ".png"


def _print_statistics_quantities(stats) -> None:
    states = stats.saved_states()
    print(f"  accumulated statistics: days {stats.window[0]}–{stats.window[1]}, "
          f"{int(stats.counts.sum())} daily samples, zonal mean, time mean, "
          "unweighted on sigma surfaces")
    print(f"  saved states: " + (f"{states['count']} (days {states['days'][0]}–"
                                 f"{states['days'][1]}); maps need state access and CUDA"
                                 if states["available"] else "none"))
    print(f"  {'quantity':<28}{'units':<12}{'shape':<9}available from")
    for row in stats.quantities():
        print(f"  {row['id']:<28}{row['units']:<12}{row['shape']:<9}"
              + ", ".join(row["sources"]["accumulated"]))
```

and in `_cmd_plot`, after the `run_dir is None` check:

```python
    from tropoi.representation.archive import StatisticsError, is_statistics_run
    if is_statistics_run(run_dir):
        from tropoi.representation.archive import open_statistics
        stats = open_statistics(run_dir)
        if note:
            print(note)
        if args.list_quantities:
            print(f"Run directory: {run_dir}  (Held–Suarez statistics, "
                  f"{stats.counts.shape[0]} blocks)")
            _print_statistics_quantities(stats)
            return 0
        rejected = [flag for flag, value in (
            ("--map", args.map), ("--contours", args.contours), ("--vectors", args.vectors),
            ("--level", args.level), ("--sigma", args.sigma), ("--snapshots", args.snapshots),
            ("--at", args.at), ("--max-maps", args.max_maps)) if value is not None]
        if rejected:
            states = stats.saved_states()
            return _error(
                f"{', '.join(rejected)}: longitude-resolved fields are not in the "
                "accumulators (zonal means only); "
                + (f"{states['count']} saved states exist but drawing them needs "
                   "state access and CUDA, which this version does not provide"
                   if states["available"] else states["reason"])
                + ". Use --section / --series, or --list-quantities.")
        try:
            view = _statistics_view(args)
            output = args.output
            if output is None and view is not None:
                output = run_dir / "assets" / _statistics_asset_name(args)
            written = stats.plot(output, view, sidecar=args.sidecar)
        except (StatisticsError, ValueError) as err:
            return _error(str(err))
        print(f"Wrote {written}")
        if args.sidecar:
            print(f"Wrote {pathlib.Path(written).with_suffix('.json')}")
        if args.open is True or (args.open is None and _can_open_viewer()):
            _open_in_viewer(pathlib.Path(written))
        return 0
```

- [ ] **Step 4: Run the CLI tests and the existing plot tests**

Run: `./venv/Scripts/python.exe -m pytest tests/cli/test_plot_statistics.py tests/cli/test_plot_command.py tests/cli/test_help.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/tropoi/cli/main.py tests/cli/test_plot_statistics.py
git commit -m "feat(cli): tropoi plot draws Held–Suarez statistics runs (--section, --series)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Publish the slim evidence bundle

**Files:**
- Create: `docs/runs/hs-T42L20-prod-002/{config.json,events.jsonl,series.json,statistics.npz,report/REPORT.md,report/report.json,SHA256SUMS}`
- Modify: `docs/runs/README.md` (index entry), `tests/test_published_layout.py:60-75`

**Interfaces:**
- Produces: the published bundle path `docs/runs/hs-T42L20-prod-002/` that Task 8's recipe reads; the published-layout test accepts a receipt with no `_coeffs.npy` when every figure's `StatisticsSHA256` matches `statistics.npz`.

- [ ] **Step 1: Extend the layout test first**

Replace `test_a_published_runs_figures_name_the_evidence_they_read` with:

```python
@pytest.mark.parametrize("run", RUNS, ids=lambda path: path.name[:40])
def test_a_published_runs_figures_name_the_evidence_they_read(run):
    from PIL import Image
    sums = _receipt(run)
    coefficients = [name for name in sums if name.endswith("_coeffs.npy")]
    statistics = "statistics.npz" in sums
    assert len(coefficients) == 1 or statistics, (
        "a published run is a snapshot capsule or a statistics run")
    for figure in sorted((run / "assets").glob("*.png")):
        with Image.open(figure) as image:
            info = dict(image.info)
        if coefficients:
            assert info.get("CoefficientsSHA256") == sums[coefficients[0]], figure.name
            if "diagnostics/timeseries.csv" in sums:
                assert info.get("DiagnosticsSHA256") == sums[
                    "diagnostics/timeseries.csv"], figure.name
        else:
            assert info.get("StatisticsSHA256") == sums["statistics.npz"], figure.name
            assert info.get("ConfigSHA256") == json.loads(
                (run / "config.json").read_text(encoding="utf-8"))["config_sha256"]
```

(add `import json` at the top). Run `./venv/Scripts/python.exe -m pytest tests/test_published_layout.py -q`: still passes for the two existing runs.

- [ ] **Step 2: Copy the evidence and write the receipt**

```bash
mkdir -p docs/runs/hs-T42L20-prod-002/report
cp runs/hs-T42L20-prod-002/config.json runs/hs-T42L20-prod-002/events.jsonl runs/hs-T42L20-prod-002/series.json runs/hs-T42L20-prod-002/statistics.npz docs/runs/hs-T42L20-prod-002/
cp runs/hs-T42L20-prod-002/report/REPORT.md runs/hs-T42L20-prod-002/report/report.json docs/runs/hs-T42L20-prod-002/report/
cd docs/runs/hs-T42L20-prod-002 && sha256sum config.json events.jsonl series.json statistics.npz report/REPORT.md report/report.json > SHA256SUMS && sha256sum -c SHA256SUMS && cd ../../..
git check-attr text -- docs/runs/hs-T42L20-prod-002/config.json
```

Expected: six `OK` lines; `text: unset` (the existing `docs/runs/** -text` rule covers it). Copy the Colab `report/` (the one whose tier A is INCOMPLETE), not `assets/report-with-tier-a/`: the published report is the run's own output, and STATUS.md already explains the tier-A evidence.

- [ ] **Step 3: Index entry in `docs/runs/README.md`**

Append:

```markdown
### `hs-T42L20-prod-002`

The Held–Suarez (1994) validation run of the dry primitive-equation core:
evolved T42 (stored ℓ ≤ 43, retained 42), 20 uniform σ levels, Gauss–Legendre
64 × 128, semi-implicit leapfrog with centred damping, Δt = 720 s, 1200 days,
statistics over days 201–1200 in five 200-day blocks. Produced at commit
`a52ea20` (clean) on Google Colab (Tesla T4), 2026-10-03/04, config
`d1fa645d…`, PROTOCOL.md amendment 1. Report verdicts: B PASS, C PASS
(A PASS with the local tier-A evidence; see
[docs/held_suarez/STATUS.md](../held_suarez/STATUS.md)).

This is a **statistics run**, not a snapshot capsule: its evidence is the
online accumulators (`statistics.npz`: per-block zonal means on σ surfaces of
u, v, T, [T*T*], [u*u*], [v*T*], [u*v*], squares of [u] and [T], the KE
spectrum, and the daily scalar series), the daily `series.json`, the frozen
`config.json`, the event log and the Colab `report/`. The 120 full spectral
states (217 MB) stay in the gitignored `runs/` copy and on Drive; nothing in
this bundle resolves longitude.

- **Assets:** `climatology.png` and `climatology.json` are the README figure
  and its numbers, drawn by the pinned recipe
  [docs/figures/held_suarez_t42l20_climatology.py](../figures/held_suarez_t42l20_climatology.py)
  from `statistics.npz` alone (no GPU). The PNG metadata records
  `StatisticsSHA256` and `ConfigSHA256`.
```

- [ ] **Step 4: Run the layout tests and commit**

Run: `./venv/Scripts/python.exe -m pytest tests/test_published_layout.py -q`
Expected: pass (the bundle has no `assets/` yet, so the figure loop is empty).

```bash
git add docs/runs/hs-T42L20-prod-002 docs/runs/README.md tests/test_published_layout.py
git commit -m "docs(runs): publish the hs-T42L20-prod-002 statistics bundle with its receipt

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: The pinned climatology recipe and its test

**Files:**
- Create: `docs/figures/held_suarez_t42l20_climatology.py`
- Create: `docs/runs/hs-T42L20-prod-002/assets/climatology.png`, `climatology.json` (generated)
- Modify: `tests/test_readme_figure_recipe.py` (add HS cases)

**Interfaces:**
- Consumes: `open_statistics`, `Panels`, `Section`, `DailySeries`, `TimeMean`, `ZonalMean`, `Contours`, `Style`, `describe`.
- Produces: `RECIPE: Panels`, `BUNDLE: pathlib.Path`, `OUTPUT: pathlib.Path`, `main()`.

- [ ] **Step 1: Write the recipe**

```python
"""Recipe for the README figure: the Held–Suarez climatology at T42 L20.

Run from the repository root (no GPU; about 3 s)::

    python docs/figures/held_suarez_t42l20_climatology.py

It draws the published statistics bundle (``docs/runs/hs-T42L20-prod-002``)
through the public interface (``AccumulatedStatistics.plot``) and writes

* ``assets/climatology.png``, whose PNG metadata records the SHA-256 of
  ``statistics.npz`` and of the configuration, and this recipe;
* ``assets/climatology.json``: every number the figure shows and the full
  description of every product (reduction, weighting, window, source).

Every field of every view object is spelled out, and ``source`` is pinned to
the accumulators, so no later default can change this figure silently
(``tests/test_readme_figure_recipe.py``).
"""
from __future__ import annotations

import pathlib

from tropoi.representation.archive import open_statistics
from tropoi.representation.visual.views import (
    Contours, DailySeries, Panels, Section, Style, TimeMean, ZonalMean)

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "docs" / "runs" / "hs-T42L20-prod-002"
OUTPUT = BUNDLE / "assets" / "climatology.png"

_WINDOW = TimeMean(days=(201, 1200), source="accumulated")

RECIPE = Panels(
    rows=(
        (Section(quantity="zonal_wind", across=ZonalMean(), time=_WINDOW,
                 contours=(Contours(quantity="temperature",
                                    levels=(210.0, 230.0, 250.0, 270.0, 290.0),
                                    color="#3b3b3b", line_width=0.6,
                                    line_style="solid"),),
                 color_policy="RdBu_r", symmetric=True, limits=None,
                 show_block_spread=False),
         Section(quantity="temperature", across=ZonalMean(), time=_WINDOW,
                 contours=(), color_policy="inferno:0.25:1.0", symmetric=False,
                 limits=None, show_block_spread=False)),
        (Section(quantity="eddy_temperature_variance", across=ZonalMean(),
                 time=_WINDOW, contours=(), color_policy="YlOrRd",
                 symmetric=False, limits=None, show_block_spread=False),
         Section(quantity="eddy_heat_flux", across=ZonalMean(), time=_WINDOW,
                 contours=(), color_policy="RdBu_r", symmetric=True, limits=None,
                 show_block_spread=False)),
        (DailySeries(quantities=("kinetic_energy",), title=None, mark_blocks=True),
         DailySeries(quantities=("max_wind_speed",), title=None, mark_blocks=True)),
    ),
    title=("Held–Suarez (1994) climatology, run hs-T42L20-prod-002 · "
           "dry primitive-equation core"),
    style=Style(width_inches=7.2, base_font_size=8.0, dpi=250, map_columns=2,
                map_row_height_inches=1.85, static_height_inches=1.6,
                diagnostics_height_inches=2.45),
)


def main() -> None:
    statistics = open_statistics(BUNDLE)
    written = statistics.plot(OUTPUT, RECIPE, sidecar=True)
    print(f"Wrote {written}")
    print(f"Wrote {written.with_suffix('.json')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the assets and look at them**

Run: `./venv/Scripts/python.exe docs/figures/held_suarez_t42l20_climatology.py`
Expected: two `Wrote ...` lines. Open the PNG; check: jets near 45° at σ ≈ 0.25 of about 33 m/s with surface easterlies in the tropics, T decreasing upward, [T*T*] peaking in mid-latitudes near the surface, [v*T*] positive in the NH and negative in the SH, every panel subtitle showing "days 201–1200, 1000 daily samples in 5 blocks / unweighted on σ surfaces", and text legible at 880 px width. If text overlaps, adjust only `Style` numbers in the recipe (they are pinned fields, so the test still passes).

- [ ] **Step 3: Extend the recipe test**

Append to `tests/test_readme_figure_recipe.py`:

```python
HS_RECIPE_PATH = ROOT / "docs" / "figures" / "held_suarez_t42l20_climatology.py"
HS_BUNDLE = ROOT / "docs" / "runs" / "hs-T42L20-prod-002"
HS_FIGURE = HS_BUNDLE / "assets" / "climatology.png"


def _hs_recipe_module():
    spec = importlib.util.spec_from_file_location("hs_recipe", HS_RECIPE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_hs_recipe_spells_out_every_field_and_pins_the_source():
    tree = ast.parse(HS_RECIPE_PATH.read_text(encoding="utf-8"))
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and hasattr(views, node.func.id)):
            continue
        cls = getattr(views, node.func.id)
        if not dataclasses.is_dataclass(cls):
            continue
        given = {keyword.arg for keyword in node.keywords}
        required = {field.name for field in dataclasses.fields(cls)}
        assert not node.args, f"{cls.__name__}: use keywords only"
        assert given == required, f"{cls.__name__} leaves {sorted(required - given)} to defaults"
        if cls is views.TimeMean:
            source = next(k.value for k in node.keywords if k.arg == "source")
            assert isinstance(source, ast.Constant) and source.value == "accumulated"
        checked += 1
    assert checked >= 8


def test_committed_hs_figure_was_made_by_this_recipe_from_the_bundle():
    from PIL import Image
    recipe = _hs_recipe_module()
    with Image.open(HS_FIGURE) as image:
        info = dict(image.info)
        assert image.size[0] == 1800
    assert json.loads(info["View"]) == describe(recipe.RECIPE)
    sums = dict(reversed(line.split(maxsplit=1)) for line in
                (HS_BUNDLE / "SHA256SUMS").read_text().splitlines())
    assert info["StatisticsSHA256"] == sums["statistics.npz"]
    assert info["ConfigSHA256"] == json.loads(
        (HS_BUNDLE / "config.json").read_text(encoding="utf-8"))["config_sha256"]
    assert recipe.OUTPUT == HS_FIGURE


def test_committed_hs_sidecar_agrees_with_numbers_recomputed_here():
    record = json.loads(HS_FIGURE.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(HS_BUNDLE / "statistics.npz", allow_pickle=False) as z:
        counts = z["counts"].astype(float)
        u = (z["block_u"] * counts[:, None, None]).sum(0) / counts.sum()
        w = z["gauss_weights"][:, None] * z["dsigma"][None, :]
    product = next(p for p in record["products"] if p["quantity"] == "zonal_wind")
    assert product["stats"]["max"] == pytest.approx(float(u.max()), rel=1e-12)
    assert product["stats"]["mass_weighted_mean"] == pytest.approx(
        float((w * u).sum() / w.sum()), rel=1e-12)
    assert product["interval"] == {"days": [201, 1200], "samples": 1000,
                                   "cadence": "1 per day", "blocks": 5, "block_days": 200}
    assert product["weighting"] == "unweighted on sigma surfaces"
    assert record["run"]["window_days"] == [201, 1200]
    dumped = json.dumps(record).lower()
    assert "uncertainty" not in dumped and "significan" not in dumped
```

- [ ] **Step 4: Run the recipe, layout and recipe tests; check determinism**

Run: `./venv/Scripts/python.exe -m pytest tests/test_readme_figure_recipe.py tests/test_published_layout.py -q`
Expected: pass. Then regenerate once more and confirm the sidecar is byte-identical: `./venv/Scripts/python.exe docs/figures/held_suarez_t42l20_climatology.py && git status --short docs/runs/hs-T42L20-prod-002/assets/climatology.json` → no change listed (the PNG may differ by antialiasing; that is documented).

- [ ] **Step 5: Commit**

```bash
git add docs/figures/held_suarez_t42l20_climatology.py docs/runs/hs-T42L20-prod-002/assets tests/test_readme_figure_recipe.py
git commit -m "docs(figures): pinned Held–Suarez climatology recipe and its README assets

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: README and documentation

**Files:**
- Modify: `README.md` (model-core table row for PE at line 73; a new figure block after the BVE example near line 258; a sentence in "Outputs and saved runs")
- Modify: `docs/SAVED_RUNS.md` (new section "Statistics runs"; update "What this interface does not do")
- Modify: `docs/held_suarez/STATUS.md` (one line pointing at the published bundle and recipe)

- [ ] **Step 1: README model-core row**

Replace the PE row (line 73) with:

```markdown
| **Dry hydrostatic primitive equations (PE)** | `tropoi run pe`, `python -m tropoi.run.held_suarez` | per-level vorticity, divergence, temperature; ln surface pressure | Exact rest; analytic orographic balance; **Held–Suarez (1994) climatology at T42 L20 validated against a Dinosaur reference** (tiers B and C PASS; [STATUS](docs/held_suarez/STATUS.md)) | Semi-implicit leapfrog with HS94 forcing for the validation experiment only; the `tropoi run pe` demo keeps a fixed explicit step and no forcing; no moist physics; no energy-conservation claim |
```

- [ ] **Step 2: README figure block**

Insert after the BVE example caption (before "## Validation: measured, not guaranteed"):

```markdown
![Held–Suarez climatology at T42 L20: zonal- and time-mean zonal wind with temperature contours, temperature, eddy temperature variance and eddy heat flux on latitude–sigma sections; daily kinetic energy and peak wind](docs/runs/hs-T42L20-prod-002/assets/climatology.png)

*The dry primitive-equation core under Held–Suarez (1994) forcing, run
[`hs-T42L20-prod-002`](docs/runs/hs-T42L20-prod-002/): 1200 days at evolved
T42 with 20 σ levels, statistics over days 201–1200 (1000 daily samples in
five 200-day blocks). Every section is a **zonal and time mean on σ
surfaces**, read from the run's online accumulators with no GPU
(`tropoi plot docs/runs/hs-T42L20-prod-002`); longitudinal structure is not
shown, and the eddy products are zonal (co)variances about each day's zonal
mean as the run accumulated them. Zonal-mean u, T, [T*T*] and [v*T*] agree
with a Dinosaur reference within the report's criteria (tier C PASS); the
comparison is in [docs/held_suarez/STATUS.md](docs/held_suarez/STATUS.md).*
```

- [ ] **Step 3: README "Outputs and saved runs" sentence**

After the paragraph ending "`--plot summary` or `--plot all`.", add:

```markdown
A Held–Suarez experiment directory is a *statistics run*: its evidence is
`statistics.npz` (online zonal- and time-mean accumulators) and the daily
series. `tropoi plot RUN` draws its climatology sections on any host, and
`--list-quantities` says which quantities the accumulators provide and whether
saved full states exist (they need CUDA and a later version to draw).
```

- [ ] **Step 4: `docs/SAVED_RUNS.md` section**

Add before "## Where the implementation lives":

```markdown
## Statistics runs: `open_statistics` and `Panels` views

A Held–Suarez experiment directory (`python -m tropoi.run.held_suarez`) is not
a snapshot capsule: `open_simulation` refuses it. Its primary product is
`statistics.npz`, the online accumulators, read on the host by

```python
from tropoi.representation.archive import open_statistics
from tropoi.representation.visual.views import (
    Contours, DailySeries, Panels, Section, TimeMean, ZonalMean)

stats = open_statistics("runs/hs-T42L20-prod-002")
stats.product("zonal_wind")          # Product: values (nlat, nlev) + reduction, weighting, window, source
stats.block_spread("temperature")    # block-to-block spread of the time mean (a variability diagnostic)
stats.series("kinetic_energy")       # daily series, 1 sample per day
stats.sources_for("temperature", ("time_mean",))   # not in the accumulators; saved states?
stats.plot()                         # default climatology -> RUN/assets/climatology.png
stats.plot("u.png", Panels(rows=((Section("zonal_wind",
                                          contours=(Contours("temperature", (250.0, 270.0)),)),),)))
```

- **What a `Product` says.** Its `quantity` is the catalogue id
  (`temperature` is the same quantity in a snapshot map and in an accumulated
  mean); `reduction` lists what was applied (`("zonal_mean", "time_mean")`),
  `weighting` says how (`unweighted on sigma surfaces`), `interval` gives the
  days, sample count and cadence, `coordinates` the latitudes and σ levels,
  `source` the file and its SHA-256, `definition` the exact formula.
- **Accumulated means are on σ surfaces, unweighted by p_s.** A mass-weighted
  mean ([p_s X]/[p_s]) is not available from any saved data of these runs.
- **Eddy products** keep the definition the run accumulated: zonal
  (co)variances about the *instantaneous* zonal mean, averaged over daily
  samples. No stationary/transient split exists, because nothing in
  longitude was recorded.
- **Block spread** (`block_spread`, `Section(show_block_spread=True)`) is the
  sample standard deviation of the per-block time means. It describes how
  much this run's blocks differ; it is not a bound on the time mean.
- **Sources.** `TimeMean(source=...)` accepts only `"accumulated"` in this
  version. `"snapshots"` and `"auto"` are reserved and refused, so a
  publication recipe cannot pin a source that does not exist. When saved
  states exist, `sources_for` and `--list-quantities` say so, and say that
  drawing them needs CUDA and state access, which a later version adds.
- **Two kinds of unavailable.** "Not in the accumulators" (a time-mean map, a
  different window) names what else could serve the request; "not available
  from any saved data of this run" (a mass streamfunction) does not.
- **Figures** go to `RUN/assets/` only; PNG metadata records
  `StatisticsSHA256`, `ConfigSHA256` and the full view; `sidecar=True` writes
  every product description and number to `<output>.json`.
```

And in "What this interface does not do", change "no vertical profiles or sections, zonal means" to "no vertical profiles, and sections only as zonal- and time-mean products of statistics runs (no single-longitude sections, no snapshot-based time means yet)".

- [ ] **Step 5: STATUS.md pointer**

In `docs/held_suarez/STATUS.md`, under the run-002 "Local copy" bullet, append:

```markdown
- **Published (2026-10-06).** A slim evidence bundle (`config.json`, `events.jsonl`,
  `series.json`, `statistics.npz`, the Colab `report/`, `SHA256SUMS`) is committed
  at `docs/runs/hs-T42L20-prod-002/`; the README climatology figure is drawn
  from it by `docs/figures/held_suarez_t42l20_climatology.py`. States stay in
  `runs/` and on Drive.
```

- [ ] **Step 6: Full CPU suite and commit**

Run: `./venv/Scripts/python.exe -m pytest tests -q -x --ignore=tests/test_w5_canonical_capsule.py -p no:cacheprovider -k "not cuda"` (then the full local suite if the GPU is free: `./venv/Scripts/python.exe -m pytest tests -q`).
Expected: no failures; the pre-existing 5 skips only.

```bash
git add README.md docs/SAVED_RUNS.md docs/held_suarez/STATUS.md
git commit -m "docs: Held–Suarez climatology in the README; statistics runs in SAVED_RUNS

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage.** Reader without solver import (Task 2, laziness test); product separate from quantity (Task 2 `Product`); linear-vs-nonlinear averaging rule is not needed yet because no snapshot source exists; `TimeMean` refuses `"snapshots"`/`"auto"` so recipes cannot pin them (Task 5); block spread as a variability diagnostic with banned words checked in tests (Tasks 2, 5, 8); statistics-only default with the "longitudinal structure not shown" line and `omitted` naming the map options (Task 5) and `--list-quantities` naming saved states (Task 6); "not in the accumulators" vs "not available from any saved data" from one `sources_for` (Tasks 2, 5, 6); dependency direction verified and the reader placed in `representation.archive` (Task 2 Step 5); slim bundle + README + validation-doc link (Tasks 7, 9).
- **Not in this plan (deferred by decision):** HS state access, snapshot time means, `Longitude`/`Latitude` sections, `departure=`, pressure/height, the Dinosaur comparison *figure* (the README links to STATUS.md's existing numbers), new recording.
- **Type consistency.** `Product.interval["days"]` is a tuple in Python and a list in JSON (`describe()` converts; Task 8's test compares lists). `Section.contours` reuses the existing `Contours` view (quantity, levels, color, line_width, line_style). `AccumulatedStatistics.plot` signature matches `render_statistics_view`. `_section_edges` name is used consistently in Tasks 3 and 4.
- **Known risk.** Task 8 Step 2's expected physical features are what STATUS.md reports (jet 32.6 m/s at 43.3°, surface easterlies); if the figure disagrees, the reader or the σ orientation is wrong, not the run.
