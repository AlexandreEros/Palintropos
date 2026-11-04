"""The climate-figure reduction against a real GPU production run (CPU only).

``tests/fixtures/hs-T21L10-dev4`` is a production run of exactly
``development_config(days=4, retained=True)`` made on a GPU by
``tools/held_suarez/climate_fixture.py`` (see the fixture's README.md). Its
``statistics.npz`` and ``gpu_daily_samples.npz`` come from the production
``daily_sample`` (GPU synthesis, ``model.wind_on_state_grid`` /
``temperature_on_state_grid``) -> ``OnlineStatistics`` path; the saved states
are the very states those samples were taken from. These tests reduce the
saved days 1-4 with the host synthesis and compare every statistic both
paths compute, on the same Gaussian grid, without CUDA.

Tolerance (derived, not fitted to pass): each synthesized field ``a`` is
granted an absolute error ``delta_a = SYNTHESIS_BUDGET * max|a|`` per state.
``SYNTHESIS_BUDGET`` = 1e-12 covers the measured worst GPU-versus-host
gridpoint difference recorded in ``generation.json`` (1.7e-14 of max|u|)
more than fiftyfold, and the test checks that it does. Propagated to the
statistics: a zonal mean ``[a]`` may differ by ``delta_a``; a departure
``a*`` by ``2 delta_a``; an eddy product ``[a* b*]`` by
``2 (A* delta_b + B* delta_a) + 4 delta_a delta_b`` with ``A* = max|a*|``.
A time mean may differ by at most the largest per-state bound.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from tropoi.representation.archive import open_simulation
from tropoi.representation.diagnostics.held_suarez_sections import (
    _synthesize_uvT, gaussian_latitudes_deg)
from tropoi.representation.visual.held_suarez_climate import (
    reduce_experiment)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hs-T21L10-dev4"
#: Statistics both paths compute (the accumulator has no [v*^2]).
COMPARED = ("u", "v", "T", "usvs", "vsTs", "usus", "TsTs")
PRODUCTS = {"usvs": ("u", "v"), "vsTs": ("v", "T"), "usus": ("u", "u"),
            "TsTs": ("T", "T")}
SYNTHESIS_BUDGET = 1.0e-12


@pytest.fixture(scope="module")
def storage():
    return open_simulation(FIXTURE).storage


@pytest.fixture(scope="module")
def generation():
    return json.loads((FIXTURE / "generation.json").read_text(
        encoding="utf-8"))


@pytest.fixture(scope="module")
def gpu():
    with np.load(FIXTURE / "gpu_daily_samples.npz") as npz:
        return {k: np.array(npz[k]) for k in npz.files}


@pytest.fixture(scope="module")
def stats():
    with np.load(FIXTURE / "statistics.npz") as npz:
        return {k: np.array(npz[k]) for k in npz.files}


@pytest.fixture(scope="module")
def bounds(storage):
    """Per-day propagated tolerance of every compared statistic."""
    cfg = storage.experiment_config
    ctx = storage.pe_context()
    lat = gaussian_latitudes_deg(cfg["nlat"])
    lon = 360.0 * np.arange(cfg["nlon"]) / cfg["nlon"]
    out = {}
    for index, day in enumerate(storage.days):
        frame = np.load(storage.state_path(index))
        u, v, T = _synthesize_uvT(frame, cfg["nlev"], lat, lon,
                                  ctx.radius_m)
        fields = {"u": u, "v": v, "T": T}
        delta = {k: SYNTHESIS_BUDGET * np.abs(a).max()
                 for k, a in fields.items()}
        eddy = {k: np.abs(a - a.mean(axis=2, keepdims=True)).max()
                for k, a in fields.items()}
        tol = {k: delta[k] for k in fields}
        for key, (a, b) in PRODUCTS.items():
            tol[key] = (2.0 * (eddy[a] * delta[b] + eddy[b] * delta[a])
                        + 4.0 * delta[a] * delta[b])
        out[int(day)] = tol
    return out


def test_the_fixture_is_exactly_the_production_development_config(storage):
    from tropoi.run.held_suarez.config import (HeldSuarezConfig,
                                               development_config)
    doc = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    expected = development_config(days=4, retained=True)
    assert HeldSuarezConfig.from_dict(doc["config"]) == expected
    assert doc["config_sha256"] == expected.sha256()
    # the production retained layout, not the legacy retained=False default
    cfg = storage.experiment_config
    assert (cfg["l_max"], cfg["retained_truncation"]) == (22, 21)
    assert (cfg["nlat"], cfg["nlon"], cfg["nlev"]) == (32, 64, 10)
    assert (cfg["days"], cfg["spinup_days"], cfg["block_days"],
            cfg["state_every_days"]) == (4, 0, 2, 1)
    assert doc["code"]["git_dirty"] is False and doc["code"]["git_commit"]


def test_the_receipt_covers_every_fixture_file():
    sums = {}
    for line in (FIXTURE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    files = {p.relative_to(FIXTURE).as_posix() for p in FIXTURE.rglob("*")
             if p.is_file()} - {"SHA256SUMS", "README.md"}
    assert set(sums) == files
    for name, digest in sums.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == (
            digest), name


def test_the_saved_states_have_the_retained_layout(storage, generation):
    assert storage.days == (1, 2, 3, 4)
    for index in range(4):
        frame = np.load(storage.state_path(index))
        assert frame.shape == (31, 23, 23) and frame.dtype == np.complex128
        rows = frame[:30]
        # the stored buffer degree l = 22 is exactly zero in every state
        assert not np.any(rows[:, 22, :])
        assert np.abs(rows[:, 21, :]).max() > 0.0
    measured = generation["gpu_vs_host_synthesis"]
    assert all(e["max_abs_buffer_degree"] == 0.0
               for e in measured["per_state"])


def test_the_synthesis_budget_covers_the_measured_gpu_host_difference(
        generation):
    worst = generation["gpu_vs_host_synthesis"]["max_relative"]
    assert set(worst) == {"u", "v", "T"}
    assert max(worst.values()) * 50.0 < SYNTHESIS_BUDGET


@pytest.mark.parametrize("day", [1, 2, 3, 4])
def test_each_saved_day_matches_its_production_daily_sample(
        storage, gpu, bounds, day):
    result = reduce_experiment(storage, interval=(day - 1, day),
                               min_states=1)
    assert result.selection.days == (day,)
    i = list(gpu["day"]).index(day)
    for key in COMPARED:
        diff = np.abs(result[key] - gpu[key][i]).max()
        assert diff <= bounds[day][key], (key, diff, bounds[day][key])
    assert result.fields["eke"] == pytest.approx(
        0.5 * (result["usus"] + result["vsvs"]), abs=0.0)


@pytest.mark.parametrize("block, interval", [(0, (0, 2)), (1, (2, 4))])
def test_each_block_matches_the_production_accumulator(
        storage, stats, bounds, block, interval):
    assert list(stats["counts"]) == [2, 2]
    result = reduce_experiment(storage, interval=interval)
    days = result.selection.days
    assert days == tuple(d for d in (1, 2, 3, 4)
                         if interval[0] < d <= interval[1])
    for key in COMPARED:
        tol = max(bounds[d][key] for d in days)
        diff = np.abs(result[key] - stats[f"block_{key}"][block]).max()
        assert diff <= tol, (key, diff, tol)


def test_the_default_window_is_all_four_days_and_matches(storage, stats,
                                                         bounds):
    result = reduce_experiment(storage)
    sel = result.selection
    assert sel.days == (1, 2, 3, 4) and sel.complete
    assert sel.requested == (0.0, 4.0)
    for key in COMPARED:
        tol = max(b[key] for b in bounds.values())
        # equal counts (2, 2): the four-day mean is the mean of the blocks
        target = stats[f"block_{key}"].mean(axis=0)
        assert np.abs(result[key] - target).max() <= tol, key
    # same grid and orientation as the accumulator: north -> south, top first
    assert np.allclose(result.lat_deg, stats["lat"], rtol=0.0, atol=1e-12)
    assert np.allclose(result.sigma, stats["sigma"], rtol=0.0, atol=1e-15)
    assert result.bandlimit["max_nonzero_order"] == 21
    assert result.bandlimit["max_abs_above_nominal_truncation"] == 0.0


def test_plot_draws_the_fixture_without_cuda(tmp_path, storage):
    from PIL import Image
    sim = open_simulation(FIXTURE)
    before = (FIXTURE / "SHA256SUMS").read_bytes()
    png = sim.plot(tmp_path / "fixture.png", sidecar=True, dpi=72)
    assert png.is_file() and png.with_suffix(".pdf").is_file()
    with Image.open(png) as image:
        assert image.text["PDF"] == "fixture.pdf"
    record = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["samples"]["days"] == [1, 2, 3, 4]
    # The strip draws the run's own daily series: every day 0-4, the
    # production values, the mass drift relative to day 0, and the sections'
    # window (0, 4] shaded.
    raw = (FIXTURE / "series.json").read_bytes()
    daily = json.loads(raw)
    series = record["series"]
    assert series["sha256"] == hashlib.sha256(raw).hexdigest()
    assert series["day"] == [0, 1, 2, 3, 4]
    assert series["ke"] == daily["ke"] and series["mean_T"] == daily["mean_T"]
    ps = np.asarray(daily["mean_ps"])
    assert np.array_equal(series["relative_mass_drift"], (ps - ps[0]) / ps[0])
    assert series["summary"]["window_days"] == [0.0, 4.0]
    assert series["summary"]["days_in_window"] == 4
    assert not (FIXTURE / "assets").exists()
    assert (FIXTURE / "SHA256SUMS").read_bytes() == before
