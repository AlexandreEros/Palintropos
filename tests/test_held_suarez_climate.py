"""Six-panel Held–Suarez climate sections from saved states (CPU only).

Manufactured fields with closed-form statistics. Spectral coefficients of
degree 1 give, in the repository's orthonormal basis (Condon–Shortley
phase, ``Pbar_11 = -sqrt(3 / 8 pi) cos(lat)``):

* ``zeta_10 = 2 U / (R sqrt(3 / 4 pi))``: solid-body zonal wind U cos(lat);
* ``psi = A cos(lat) cos(lon)``, ``chi = B cos(lat) cos(lon)`` (zeta = -2 psi
  / R^2, delta = -2 chi / R^2) give u* = (A sin(lat) cos(lon) - B sin(lon))/R
  and v* = -(A sin(lon) + B sin(lat) cos(lon)) / R, hence
  [u*v*] = A B cos^2(lat) / (2 R^2),
  [u*^2 + v*^2] / 2 = (A^2 + B^2)(1 + sin^2 lat) / (4 R^2);
* ``T = T0 + c cos(lat) cos(lon)``: [T*^2] = c^2 cos^2(lat) / 2 and
  [v* T*] = -c B sin(lat) cos(lat) / (2 R).

These are independent of the reduction code. They exercise the host path on
CPU-constructible states only; the comparison against a production
``daily_sample``/``OnlineStatistics`` archive from a real GPU run is NOT
here (deferred: docs/held_suarez/CLIMATE_FIGURE.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import pytest

from tropoi.representation.diagnostics.held_suarez_sections import (
    SECTION_KEYS, SectionAccumulator, SectionError, default_window,
    gaussian_latitudes_deg, select_samples, state_grid)

R = 6.371e6
C11 = -1.0 / (2.0 * math.sqrt(3.0 / (8.0 * math.pi)))   # cos(lat)cos(lon)
C10 = 1.0 / math.sqrt(3.0 / (4.0 * math.pi))           # sin(lat)
C00 = math.sqrt(4.0 * math.pi)                         # 1


def _state(l_max, nlev, *, U, A, B, T0, c):
    """Packed PE frame with per-level amplitudes (sequences of length nlev)."""
    x = np.zeros((3 * nlev + 1, l_max + 1, l_max + 1), dtype=np.complex128)
    for k in range(nlev):
        x[k, 1, 0] = 2.0 * U[k] / R * C10
        x[k, 1, 1] = -2.0 * A[k] / R ** 2 * C11
        x[nlev + k, 1, 1] = -2.0 * B[k] / R ** 2 * C11
        x[2 * nlev + k, 0, 0] = T0[k] * C00
        x[2 * nlev + k, 1, 1] = c[k] * C11
    x[3 * nlev, 0, 0] = math.log(1.0e5) * C00
    return x


def _expected(lat_deg, *, U, A, B, T0, c):
    """Closed-form sections (nlat, nlev) for one state's amplitudes."""
    phi = np.radians(np.asarray(lat_deg))[:, None]
    U, A, B, T0, c = (np.asarray(a, dtype=float)[None, :]
                      for a in (U, A, B, T0, c))
    s, co = np.sin(phi), np.cos(phi)
    return {
        "u": U * co + 0.0 * A,
        "T": T0 + 0.0 * phi,
        "usvs": A * B * co ** 2 / (2 * R ** 2),
        "vsTs": -c * B * s * co / (2 * R),
        "eke": (A ** 2 + B ** 2) * (1 + s ** 2) / (4 * R ** 2),
        "TsTs": c ** 2 * co ** 2 / 2 + 0.0 * A,
    }


def write_experiment(root: pathlib.Path, states: dict, *, nlat=8, nlon=16,
                     l_max=3, retained=2, nlev=3, days=4, spinup=0,
                     block_days=2, every=1, store_lat=True, lat=None,
                     series=None):
    """A Held–Suarez experiment directory (run/held_suarez layout)."""
    root.mkdir(parents=True, exist_ok=True)
    if series is not None:
        (root / "series.json").write_text(json.dumps(series),
                                          encoding="utf-8")
    config = {"schema": "palintropos.held_suarez.config/1",
              "experiment_id": "hs-manufactured", "l_max": l_max,
              "retained_truncation": retained, "nlat": nlat, "nlon": nlon,
              "nlev": nlev, "dt": 1200.0, "days": days,
              "spinup_days": spinup, "block_days": block_days,
              "state_every_days": every, "forcing": {}, "grid": "latlon",
              "radius": R, "gravity": 9.8, "cp_dry": 1004.0,
              "kappa": 2.0 / 7.0, "damping_scheme": "centred"}
    doc = {"config": config, "config_sha256": "c" * 64,
           "protocol_sha256": "p" * 64,
           "code": {"git_commit": "a" * 40, "git_dirty": False},
           "created": "2026-11-03T00:00:00Z"}
    (root / "config.json").write_text(json.dumps(doc), encoding="utf-8")
    extra = {}
    if store_lat:
        extra["lat"] = (gaussian_latitudes_deg(nlat) if lat is None
                        else np.asarray(lat))
    np.savez(root / "statistics.npz", counts=np.zeros(1), **extra)
    (root / "states").mkdir(exist_ok=True)
    for day, frame in states.items():
        np.save(root / "states" / f"state-d{day:05d}.npy", frame)
    return root


AMPS = dict(U=[30.0, 20.0, 5.0], A=[3.0e6, 1.0e6, 2.0e6],
            B=[1.0e6, -2.0e6, 0.5e6], T0=[220.0, 260.0, 290.0],
            c=[2.0, 5.0, -3.0])


def _scaled(scale):
    """Eddy amplitudes times ``scale``; zonal parts unchanged."""
    return {k: (list(v) if k in ("U", "T0") else [scale * a for a in v])
            for k, v in AMPS.items()}


def _fingerprint(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size,
                   p.stat().st_mtime_ns) for p in root.rglob("*")
                  if p.is_file())


def _open(root):
    from tropoi.representation.archive import open_simulation
    return open_simulation(root)


def _reduce(root, **kw):
    from tropoi.representation.visual.held_suarez_climate import (
        reduce_experiment)
    return reduce_experiment(_open(root).storage, **kw)


# ---------------------------------------------------------------------------
# accumulator on gridpoint fields
# ---------------------------------------------------------------------------

def test_accumulator_forms_products_before_the_equal_weight_mean():
    nlev, nlat, nlon = 2, 3, 8
    lon = 2 * np.pi * np.arange(nlon) / nlon
    acc = SectionAccumulator(nlat, nlev, nlon)
    for sign in (1.0, -1.0, 1.0):            # eddies cancel in the time mean
        u = 10.0 + sign * np.cos(lon) * np.ones((nlev, nlat, 1))
        v = sign * 2.0 * np.cos(lon) * np.ones((nlev, nlat, 1))
        T = 250.0 + sign * 3.0 * np.sin(lon) * np.ones((nlev, nlat, 1))
        acc.add(u, v, T)
    out = acc.means()
    np.testing.assert_allclose(out["u"], 10.0)
    np.testing.assert_allclose(out["usvs"], 1.0)       # [cos * 2cos] = 1
    np.testing.assert_allclose(out["vsTs"], 0.0, atol=1e-13)  # quadrature
    np.testing.assert_allclose(out["TsTs"], 4.5)
    np.testing.assert_allclose(out["eke"], 0.5 * (0.5 + 2.0))
    assert out["u"].shape == (nlat, nlev)


def test_accumulator_weights_are_equal_not_by_spacing():
    acc = SectionAccumulator(1, 1, 4)
    for value in (1.0, 2.0, 10.0):          # e.g. days 1, 2, 10
        acc.add(np.full((1, 1, 4), value), np.zeros((1, 1, 4)),
                np.zeros((1, 1, 4)))
    assert acc.means()["u"][0, 0] == pytest.approx(13.0 / 3.0)


def test_accumulator_rejects_wrong_shapes_and_nonfinite_values():
    acc = SectionAccumulator(2, 1, 4)
    with pytest.raises(SectionError, match="shape"):
        acc.add(np.zeros((1, 4, 2)), np.zeros((1, 4, 2)), np.zeros((1, 4, 2)))
    bad = np.zeros((1, 2, 4))
    bad[0, 0, 0] = np.nan
    with pytest.raises(SectionError, match="not finite"):
        acc.add(bad, np.zeros_like(bad), np.zeros_like(bad))
    with pytest.raises(SectionError, match="no states"):
        acc.means()


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------

CFG = {"days": 1200, "spinup_days": 200, "block_days": 200,
       "state_every_days": 10}


def test_default_window_is_the_accumulator_window():
    assert default_window(CFG) == (200.0, 1200.0)
    assert default_window({**CFG, "days": 1150}) == (200.0, 1000.0)


def test_window_is_open_left_closed_right():
    days = list(range(10, 1201, 10))
    sel = select_samples(days, CFG)
    assert sel.days[0] == 210 and sel.days[-1] == 1200 and sel.count == 100
    assert sel.complete and sel.regular
    assert sel.cadence_label() == "every 10 days"
    sel = select_samples(days, CFG, interval=(1100, 1150))
    assert sel.days == (1110, 1120, 1130, 1140, 1150)
    assert sel.indices == tuple(days.index(d) for d in sel.days)
    assert sel.window_source == "explicit interval"


def test_irregular_and_incomplete_samples_are_disclosed_not_filled():
    sel = select_samples([1, 2, 4, 5], {"days": 5, "spinup_days": 0,
                                        "block_days": 5,
                                        "state_every_days": 1})
    assert sel.days == (1, 2, 4, 5)
    assert not sel.regular and not sel.complete
    assert sel.missing_days == (3,)
    assert "irregular" in sel.cadence_label()
    assert "INCOMPLETE" in sel.coverage_label()
    assert "3" in sel.coverage_label()
    record = sel.to_dict()
    assert record["weights"] == "equal (1/N each)"
    assert record["intervals_days"] == {"1": 2, "2": 1}


@pytest.mark.parametrize("interval", [(5, 5), (5, 2), (-1, 3),
                                      (float("nan"), 2), ("a", "b"), (1,),
                                      (0, float("inf"))])
def test_invalid_intervals_refuse(interval):
    with pytest.raises(SectionError):
        select_samples([1, 2, 3], CFG, interval=interval)


def test_insufficient_states_explain_what_to_do():
    with pytest.raises(SectionError, match="only 1 saved state") as err:
        select_samples([10, 1200], CFG)
    assert "interval=(start, end)" in str(err.value)
    assert "statistics.npz alone is not implemented" in str(err.value)
    assert select_samples([10, 1200], CFG, min_states=1).count == 1
    with pytest.raises(SectionError, match="empty"):
        select_samples([1], {"days": 5, "spinup_days": 10, "block_days": 2})


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_state_grid_uses_stored_gaussian_latitudes_or_rebuilds_them():
    lat, lon, source = state_grid(8, 16, gaussian_latitudes_deg(8))
    assert "statistics.npz" in source
    assert lat[0] > 0 > lat[-1]                       # north -> south
    x, _ = np.polynomial.legendre.leggauss(8)
    np.testing.assert_allclose(np.sort(np.sin(np.radians(lat))), np.sort(x),
                               atol=1e-15)
    np.testing.assert_array_equal(lon, 22.5 * np.arange(16))
    rebuilt, _, source = state_grid(8, 16, None)
    np.testing.assert_array_equal(rebuilt, lat)
    assert "rebuilt" in source
    with pytest.raises(SectionError, match="not the configured"):
        state_grid(8, 16, np.linspace(80, -80, 8))
    with pytest.raises(SectionError, match="not the configured"):
        state_grid(8, 16, gaussian_latitudes_deg(10))


# ---------------------------------------------------------------------------
# end to end on manufactured spectral states
# ---------------------------------------------------------------------------

def test_manufactured_states_give_the_closed_form_sections(tmp_path):
    days = {1: 1.0, 2: -1.0, 3: 2.0, 4: 0.5}   # eddy amplitude per state
    states = {d: _state(3, 3, **_scaled(s)) for d, s in days.items()}
    root = write_experiment(tmp_path / "hs", states)
    result = _reduce(root)
    assert result.selection.days == (1, 2, 3, 4)
    lat = result.lat_deg
    assert lat[0] > 0 > lat[-1]
    mean_sq = np.mean(np.square(list(days.values())))
    one = _expected(lat, **AMPS)
    for key in SECTION_KEYS:
        want = one[key] * (mean_sq if key not in ("u", "T") else 1.0)
        np.testing.assert_allclose(result[key], want, rtol=1e-9,
                                   atol=1e-9 * np.abs(want).max(),
                                   err_msg=key)
    # products before the mean: the time-mean eddy is not zero here, but
    # its square would be (1 - 1 + 2 + 0.5)^2 / 16 != mean_sq
    assert mean_sq != pytest.approx(np.mean(list(days.values())) ** 2)
    np.testing.assert_allclose(result["v"], 0.0, atol=1e-12)


def test_sign_and_level_orientation(tmp_path):
    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2)})
    r = _reduce(root)
    north = r.lat_deg > 0
    # level 0 is the top (smallest sigma) and carries level-0 amplitudes
    assert r.sigma[0] < r.sigma[-1]
    np.testing.assert_allclose(r["T"][0], AMPS["T0"])
    # [v*T*] = -c B sin cos / 2R: level 0 has c B > 0 -> negative in the
    # north, positive in the south; level 1 has c B < 0 -> the reverse
    assert (r["vsTs"][north, 0] < 0).all() and (r["vsTs"][~north, 0] > 0).all()
    assert (r["vsTs"][north, 1] > 0).all()
    # [u*v*] = A B cos^2 / 2R^2 keeps the sign of A B per level
    assert (r["usvs"][:, 0] > 0).all() and (r["usvs"][:, 1] < 0).all()
    np.testing.assert_allclose(r["u"][:, 0],
                               30.0 * np.cos(np.radians(r.lat_deg)))


def test_full_eke_includes_the_meridional_variance(tmp_path):
    amps = dict(AMPS, A=[0.0] * 3, B=[2.0e6] * 3)      # u*, v* from chi
    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **amps) for d in (1, 2)})
    r = _reduce(root)
    s2 = np.sin(np.radians(r.lat_deg)) ** 2
    usus = (2.0e6) ** 2 / (2 * R ** 2)                    # [u*^2] = B^2/2R^2
    vsvs = (2.0e6) ** 2 * s2 / (2 * R ** 2)
    np.testing.assert_allclose(r["usus"][:, 0], usus, rtol=1e-9)
    np.testing.assert_allclose(r["vsvs"][:, 0], vsvs, rtol=1e-9)
    np.testing.assert_allclose(r["eke"][:, 0], 0.5 * (usus + vsvs),
                               rtol=1e-9)


def test_zero_fields_give_zero_sections_and_still_render(tmp_path):
    zero = np.zeros((10, 4, 4), dtype=np.complex128)
    root = write_experiment(tmp_path / "hs", {1: zero, 2: zero})
    r = _reduce(root)
    for key in SECTION_KEYS:
        np.testing.assert_array_equal(r[key], 0.0)
    path = _open(root).plot()
    assert path.is_file() and path.with_suffix(".pdf").is_file()


def test_buffer_degree_counts_in_the_measured_bandlimit(tmp_path):
    clean = _state(3, 3, **AMPS)
    r = _reduce(write_experiment(tmp_path / "clean", {1: clean, 2: clean}))
    assert r.bandlimit["max_nonzero_order"] == 1
    assert r.bandlimit["max_abs_above_nominal_truncation"] == 0.0
    assert r.bandlimit["nominal_truncation"] == 2
    assert r.bandlimit["stored_l_max"] == 3
    filled = clean.copy()
    filled[6, 3, 3] = 1.0                    # T at the buffer degree l_max
    r = _reduce(write_experiment(tmp_path / "buffer",
                                 {1: filled, 2: clean}))
    assert r.bandlimit["max_nonzero_order"] == 3
    assert r.bandlimit["max_abs_above_nominal_truncation"] == 1.0
    assert r.bandlimit["quadratic_products_alias_free"] is True   # 16 > 6
    # the buffer content is reduced, not dropped: [T*^2] grows at level 0
    assert (r["TsTs"][:, 0] >= _expected(r.lat_deg, **AMPS)["TsTs"][:, 0]
            - 1e-12).all()
    aliased = write_experiment(tmp_path / "aliased", {1: filled, 2: clean},
                               nlon=6, l_max=3)          # 6 = 2 m_max
    with pytest.raises(SectionError, match="alias"):
        _reduce(aliased)


def test_states_are_reduced_one_at_a_time(tmp_path, monkeypatch):
    import tropoi.representation.diagnostics.held_suarez_sections as mod

    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2, 3)})
    loaded, live, peak = [], [0], [0]
    real_load, real_synth = mod.load_state, mod._synthesize_uvT

    def load(path):
        loaded.append(pathlib.Path(path).name)
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        return real_load(path)

    def synth(*args, **kw):
        out = real_synth(*args, **kw)
        live[0] -= 1
        return out

    monkeypatch.setattr(mod, "load_state", load)
    monkeypatch.setattr(mod, "_synthesize_uvT", synth)
    import tropoi.representation.visual.held_suarez_climate as climate
    monkeypatch.setattr(climate, "load_state", load)
    _reduce(root)
    assert loaded == ["state-d00001.npy", "state-d00002.npy",
                      "state-d00003.npy"]
    assert peak[0] == 1


def test_only_u_v_and_T_are_synthesized(tmp_path, monkeypatch):
    import tropoi.representation.diagnostics.held_suarez_sections as mod

    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2)})
    calls = {"scalar": 0, "wind": 0}
    real_s, real_w = mod.synthesize_latlon, mod.synthesize_wind_latlon

    def scalar(*a, **k):
        calls["scalar"] += 1
        return real_s(*a, **k)

    def wind(*a, **k):
        calls["wind"] += 1
        return real_w(*a, **k)

    monkeypatch.setattr(mod, "synthesize_latlon", scalar)
    monkeypatch.setattr(mod, "synthesize_wind_latlon", wind)
    _reduce(root)
    assert calls == {"scalar": 2 * 3, "wind": 2 * 3}   # T and (u, v) per level


def test_missing_stored_latitudes_rebuild_the_same_grid(tmp_path):
    states = {d: _state(3, 3, **AMPS) for d in (1, 2)}
    a = _reduce(write_experiment(tmp_path / "a", states))
    b = _reduce(write_experiment(tmp_path / "b", states, store_lat=False))
    np.testing.assert_array_equal(a.lat_deg, b.lat_deg)
    np.testing.assert_array_equal(a["usvs"], b["usvs"])
    with pytest.raises(SectionError, match="not the configured"):
        _reduce(write_experiment(tmp_path / "c", states,
                                 lat=np.linspace(70, -70, 8)))


# ---------------------------------------------------------------------------
# public plotting API
# ---------------------------------------------------------------------------

def _png_text(path):
    from PIL import Image
    with Image.open(path) as image:
        return dict(image.text)


def test_plot_writes_png_and_pdf_to_assets_and_returns_the_png(tmp_path):
    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **_scaled(s))
                             for d, s in ((1, 1.0), (2, -1.0), (4, 2.0))})
    before = _fingerprint(root)
    sim = _open(root)
    path = sim.plot(sidecar=True)
    assert path == (root / "assets" / "overview.png").resolve()
    pdf = path.with_suffix(".pdf")
    assert pdf.read_bytes()[:5] == b"%PDF-"
    meta = _png_text(path)
    assert meta["PDF"] == "overview.pdf"
    assert meta["RunId"] == "hs-manufactured"
    samples = json.loads(meta["Samples"])
    assert samples["days"] == [1, 2, 4] and samples["complete"] is False
    assert samples["missing_days"] == [3]
    assert json.loads(meta["View"])["view"] == "held_suarez_climate"
    record = json.loads(path.with_suffix(".json").read_text("utf-8"))
    assert record["outputs"] == {"png": "overview.png", "pdf": "overview.pdf"}
    assert set(record["panels"]) == set(SECTION_KEYS)
    assert record["grid"]["latitude_order"].startswith("north")
    # everything new is under assets/; primary files untouched
    after = [e for e in _fingerprint(root) if not e[0].startswith("assets")]
    assert after == before


def test_explicit_png_path_and_overrides(tmp_path):
    from tropoi.representation.visual.held_suarez_climate import SectionScale

    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2, 3, 4)})
    sim = _open(root)
    out = tmp_path / "figs" / "hs_climate.png"
    path = sim.plot(out, interval=(1, 3), sidecar=True,
                    scales={"u": (-5.0, 0.0, 5.0),
                            "TsTs": SectionScale((0.0, 1.0, 2.0), "Greys")})
    assert path == out.resolve() and out.with_suffix(".pdf").is_file()
    record = json.loads(out.with_suffix(".json").read_text("utf-8"))
    assert record["samples"]["days"] == [2, 3]
    assert record["samples"]["requested_interval_days"] == [1.0, 3.0]
    assert record["panels"]["u"]["fraction_above_scale"] > 0.0
    assert record["view"]["scales"]["TsTs"]["cmap"] == "Greys"
    with pytest.raises(ValueError, match="unknown panel"):
        sim.plot(out, scales={"omega": (0, 1)})


def test_plot_refuses_bad_paths_views_and_insufficient_data(tmp_path):
    from tropoi.representation.archive.held_suarez import HeldSuarezPlotError
    from tropoi.representation.visual.views import Map, Overview

    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2)})
    sim = _open(root)
    with pytest.raises(ValueError, match="only assets/"):
        sim.plot(root / "figure.png")
    with pytest.raises(ValueError, match=r"\.png"):
        sim.plot(tmp_path / "figure.pdf")
    with pytest.raises(HeldSuarezPlotError, match="explicit views"):
        sim.plot(tmp_path / "v.png", Overview(map=Map("temperature")))
    with pytest.raises(HeldSuarezPlotError, match="explicit views"):
        sim[0].plot(tmp_path / "v.png", view=Map("temperature"))
    with pytest.raises(Exception, match="no snapshot figure"):
        sim[0].plot(tmp_path / "s.png")                 # unchanged
    with pytest.raises(SectionError, match="only 1 saved state"):
        sim.plot(tmp_path / "x.png", interval=(1, 2))
    with pytest.raises(SectionError, match="empty"):
        sim.plot(tmp_path / "x.png", interval=(2, 2))
    empty = write_experiment(tmp_path / "stats", {})
    (empty / "states").rmdir()
    with pytest.raises(SectionError, match="no saved states"):
        _open(empty).plot()
    assert not (root / "figure.png").exists()


def _series(days=5, ps0=1.0e5):
    day = list(range(days))
    return {"day": [float(d) for d in day],
            "mean_ps": [ps0 * (1.0 - 2.0e-7 * d) for d in day],
            "ke": [100.0 + 10.0 * d for d in day],
            "mean_T": [250.0 - 0.5 * d for d in day],
            "max_abs_u": [0.0] * days}


def test_daily_series_validates_and_measures():
    from tropoi.representation.diagnostics.held_suarez_sections import (
        daily_series)
    series = daily_series(_series())
    assert np.allclose(series.mass_drift(), [-2.0e-7 * d for d in range(5)],
                       rtol=1e-9, atol=0.0)
    summary = series.summary((2.0, 4.0))          # days 3 and 4 only
    assert summary["days_in_window"] == 2
    assert summary["ke_window"] == {"mean": 135.0, "min": 130.0, "max": 140.0}
    assert summary["mean_T_window"]["mean"] == pytest.approx(248.25)
    assert summary["mass_drift_max_abs_day"] == 4
    assert summary["mass_drift_max_abs"] == pytest.approx(8.0e-7)
    assert daily_series(_series()).summary((10.0, 20.0))["ke_window"] is None
    for broken in ({**_series(), "day": [0.0, 1.0, 3.0, 4.0, 5.0]},
                   {**_series(), "ke": [1.0, 2.0]},
                   {**_series(), "mean_ps": [0.0] * 5},
                   {**_series(), "mean_T": [float("nan")] * 5},
                   {"day": [0.0, 1.0]}):
        with pytest.raises(SectionError):
            daily_series(broken)


def test_series_strip_uses_series_json_and_shades_the_window(tmp_path):
    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (3, 4)},
                            spinup=2, series=_series())
    png = _open(root).plot(tmp_path / "s.png", sidecar=True, dpi=60)
    record = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    series = record["series"]
    raw = (root / "series.json").read_bytes()
    assert series["sha256"] == hashlib.sha256(raw).hexdigest()
    assert _png_text(png)["SeriesSHA256"] == series["sha256"]
    assert series["day"] == [0, 1, 2, 3, 4]
    assert series["ke"] == _series()["ke"]
    assert series["relative_mass_drift"] == pytest.approx(
        [-2.0e-7 * d for d in range(5)], rel=1e-9)
    # the shaded window is the sections' own window, (spinup, end]
    assert record["samples"]["requested_interval_days"] == [2.0, 4.0]
    assert series["summary"]["window_days"] == [2.0, 4.0]
    assert series["summary"]["ke_window"]["mean"] == 135.0
    # an explicit interval moves both
    png = _open(root).plot(tmp_path / "e.png", sidecar=True, dpi=60,
                           interval=(0, 4))
    record = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["series"]["summary"]["window_days"] == [0.0, 4.0]


def test_no_series_json_omits_the_strip_and_still_draws(tmp_path):
    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2)})
    png = _open(root).plot(tmp_path / "n.png", sidecar=True, dpi=60)
    assert png.is_file() and png.with_suffix(".pdf").is_file()
    record = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["series"] is None
    assert _png_text(png)["SeriesSHA256"] == ""


def test_a_broken_series_json_refuses_before_writing(tmp_path):
    root = write_experiment(tmp_path / "hs",
                            {d: _state(3, 3, **AMPS) for d in (1, 2)},
                            series={**_series(), "day": [0, 2, 3, 4, 5]})
    with pytest.raises(SectionError, match="consecutive"):
        _open(root).plot(tmp_path / "b.png")
    assert not (tmp_path / "b.png").exists()
    (root / "series.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SectionError, match="not valid JSON"):
        _open(root).plot(tmp_path / "b.png")


def test_single_level_experiment_refuses_the_figure_but_reduces(tmp_path):
    one = {k: v[:1] for k, v in AMPS.items()}
    root = write_experiment(tmp_path / "hs1",
                            {d: _state(3, 1, **one) for d in (1, 2)},
                            nlev=1)
    sim = _open(root)
    with pytest.raises(SectionError, match="at least two σ levels"):
        sim.plot(tmp_path / "one.png")
    assert not (tmp_path / "one.png").exists()
    assert not (tmp_path / "one.pdf").exists()
    from tropoi.representation.visual.held_suarez_climate import (
        reduce_experiment)
    result = reduce_experiment(sim.storage)
    assert result["u"].shape == (8, 1)


def test_assets_name_matches_compose():
    from tropoi.representation.visual import compose, held_suarez_climate
    assert held_suarez_climate.ASSETS_DIRNAME == compose.ASSETS_DIRNAME


def test_default_scales_are_documented_and_valid():
    from tropoi.representation.visual.held_suarez_climate import (
        HS_SECTION_SCALES)

    assert tuple(HS_SECTION_SCALES) == SECTION_KEYS
    for key in ("u", "usvs", "vsTs"):                 # signed: symmetric
        levels = HS_SECTION_SCALES[key].levels
        assert levels[0] == -levels[-1] and 0.0 in levels
    for key in ("eke", "TsTs"):
        assert HS_SECTION_SCALES[key].levels[0] == 0.0
    doc = (pathlib.Path(__file__).resolve().parents[1] / "docs" /
           "held_suarez" / "CLIMATE_FIGURE.md").read_text(encoding="utf-8")
    for key, scale in HS_SECTION_SCALES.items():
        assert f"`{key}`" in doc
        assert f"{scale.levels[0]:g} to {scale.levels[-1]:g}" in doc, key
