"""The path from a radiative configuration to layer heating through named
k-tables. CPU only.

The table is the synthetic Lorentz-line fixture of
``test_radiation_kdist.py`` (random lines, no real gas): what is tested here
is the wiring - a ``correlated_k`` entry of :class:`RadiativeConfig`
resolves a table by name, checks its content hash, builds the providers,
and the column physics returns interface fluxes whose divergence is the
heating - and the strictness of that path, not any opacity.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_radiation_kdist import BANDS, NU, absorption
from tropoi.physics.radiation.kdist import (CorrelatedKLongwave, KTable,
                                            ktable_from_spectrum)
from tropoi.physics.radiation.scheme import Column, LowerBoundary, Radiation
from tropoi.physics.radiation.tables import (OPACITY_PATH_ENV, KTableError, load_table,
                                             table_spec)
from tropoi.run.radiative.config import RadiativeConfig
from tropoi.run.radiative.model import build_physics, build_radiation, build_sigma

G, CP, R = 9.8, 1004.0, 287.0
Q = 2.0e-2


@pytest.fixture(scope="module")
def table() -> KTable:
    return ktable_from_spectrum("synthetic", NU, absorption, np.logspace(0.5, 5.1, 8),
                                np.array([180.0, 240.0, 300.0, 360.0]), BANDS, 8,
                                provenance="random Lorentz lines, seed 11")


@pytest.fixture()
def stored(table, tmp_path, monkeypatch):
    table.save(tmp_path / "synthetic.npz")
    monkeypatch.setenv(OPACITY_PATH_ENV, str(tmp_path))
    return table_spec(tmp_path / "synthetic.npz")


def _config(spec, **changes) -> RadiativeConfig:
    base = dict(
        experiment_id="dev-ktable-wiring", radius=6.371e6, omega=7.292e-5, gravity=G,
        gas_constant=R, cp=CP, reference_pressure=1.0e5, l_max=5,
        retained_truncation=None, nlat=8, nlon=16,
        vertical={"kind": "stretched", "nlev": 20, "sigma_top": 1.0e-4,
                  "linear_scale": 0.2},
        dt=1200.0, days=1,
        longwave={"kind": "correlated_k", "tables": [spec],
                  "mass_fractions": {"synthetic": Q}},
        shortwave={"kind": "correlated_k", "tables": [spec],
                   "mass_fractions": {"synthetic": Q}, "stellar_temperature": 2600.0},
        insolation={"kind": "uniform", "flux": 225.0},
        surface={"heat_capacity": 2.0e6, "albedo": 0.3, "initial_temperature": 290.0},
        initial={"kind": "isothermal", "temperature": 250.0})
    base.update(changes)
    return RadiativeConfig(**base)


def _columns(cfg, ncol=3):
    sig = build_sigma(cfg).interfaces_array()
    p = sig[:, None] * np.array([1.0e5, 0.97e5, 1.02e5])[None, :ncol]
    mid = 0.5 * (p[1:] + p[:-1])
    t = np.maximum(285.0 * (mid / 1.0e5) ** 0.19, 200.0)
    return p, t, np.array([292.0, 280.0, 300.0])[:ncol]


def test_configuration_reaches_heating_and_closes(stored):
    cfg = _config(stored)
    physics = build_physics(cfg)
    p, t, ts = _columns(cfg)
    lat = np.zeros(3)
    out = physics.compute(p, t, 0.0, latitude=lat, longitude=lat, surface_temperature=ts)
    fl = out.fluxes
    assert fl.lw_up.shape == p.shape and out.heating.shape == t.shape
    assert np.all(fl.olr > 0) and np.all(fl.absorbed_stellar > 0)
    # the pressure-weighted heating is the convergence of the net flux
    column_heating = (CP * (p[1:] - p[:-1]) / G * out.heating).sum(axis=0)
    convergence = fl.net_down[0] - fl.net_down[-1]
    np.testing.assert_allclose(column_heating, convergence, rtol=1e-12, atol=1e-10)
    # and it is the same answer as providers built by hand from the table
    table = load_table(stored)
    direct = Radiation(CorrelatedKLongwave([table], {"synthetic": Q}, gravity=G))
    ref = direct.fluxes(Column(p, t, G, CP), LowerBoundary("surface", temperature=ts))
    np.testing.assert_array_equal(fl.lw_up, ref.lw_up)
    np.testing.assert_array_equal(fl.lw_down, ref.lw_down)


def test_radiation_refresh_holds_and_renews_fluxes(stored):
    cfg = _config(stored, radiation_interval=3)
    physics = build_physics(cfg)
    p, t, ts = _columns(cfg)
    lat = np.zeros(3)
    kw = dict(latitude=lat, longitude=lat, surface_temperature=ts)
    first = physics.compute(p, t, 0.0, **kw).fluxes.olr.copy()
    warmer = t + 5.0
    held = [physics.compute(p, warmer, 0.0, **kw).fluxes.olr.copy() for _ in range(2)]
    for olr in held:                      # calls 2 and 3 reuse the first solve
        np.testing.assert_array_equal(olr, first)
    renewed = physics.compute(p, warmer, 0.0, **kw).fluxes.olr
    assert np.all(renewed > first)        # call 4 recomputes from the warmer state


def test_table_hash_is_part_of_the_configuration(stored, table, tmp_path):
    cfg = _config(stored)
    other = {**stored, "sha256": "0" * 64}
    assert _config(other).sha256() != cfg.sha256()
    with pytest.raises(KTableError, match="content hash"):
        build_radiation(_config(other))
    with pytest.raises(KTableError, match="not found"):
        build_radiation(_config({"file": "absent.npz", "sha256": stored["sha256"]}))
    # the hash is of the numbers: metadata and the container do not move it
    relabelled = KTable(table.species, table.pressure, table.temperature, table.band_edges,
                        table.g_weights, table.k, "other text", {"note": "x"})
    assert relabelled.content_sha256() == table.content_sha256() == stored["sha256"]
    relabelled.save(tmp_path / "again.npz")
    back = KTable.load(tmp_path / "again.npz")
    assert back.metadata == {"note": "x"} and back.content_sha256() == stored["sha256"]


def test_out_of_range_state_is_an_error_by_default(stored):
    cfg = _config(stored)
    rad = build_radiation(cfg)
    p, t, ts = _columns(cfg)
    hot = Column(p, t + 200.0, G, CP)                      # above the table's 360 K
    with pytest.raises(ValueError, match="outside the k-table"):
        rad.fluxes(hot, LowerBoundary("surface", temperature=ts))
    lw = {**cfg.longwave, "bounds": "clamp"}
    clamped = build_radiation(cfg.with_(longwave=lw, shortwave=None, insolation=None))
    clamped.fluxes(hot, LowerBoundary("surface", temperature=ts))
    assert clamped.longwave.clamped_layers > 0


def test_unknown_or_missing_options_are_rejected(stored):
    cfg = _config(stored)
    with pytest.raises(ValueError, match="unknown keys"):
        build_radiation(cfg.with_(longwave={**cfg.longwave, "extrapolate": True}))
    with pytest.raises(ValueError, match="no table provides"):
        build_radiation(cfg.with_(longwave={**cfg.longwave, "mass_fractions": {
            "synthetic": Q, "CO2": 1.0}}))
    with pytest.raises(ValueError, match="stellar_temperature"):
        sw = {k: v for k, v in cfg.shortwave.items() if k != "stellar_temperature"}
        build_radiation(cfg.with_(shortwave=sw))


def test_per_band_padded_quadrature_equals_the_shared_one(table):
    """A table whose bands carry their own (zero-padded) weights gives the
    fluxes of the shared quadrature, and padded points are never solved."""
    pad = 3
    w = np.zeros((table.n_band, table.n_g + pad))
    k = np.full(table.k.shape[:3] + (table.n_g + pad,), 123.0)      # padding is ignored
    for b in range(table.n_band):
        lo = b % (pad + 1)                                         # padding position varies
        w[b, lo:lo + table.n_g] = table.g_weights
        k[:, :, b, lo:lo + table.n_g] = table.k[:, :, b]
    padded = KTable("synthetic", table.pressure, table.temperature, table.band_edges, w, k)
    cfg = _config({"file": "unused", "sha256": "unused"})
    p, t, ts = _columns(cfg)
    col, low = Column(p, t, G, CP), LowerBoundary("surface", temperature=ts)
    a = CorrelatedKLongwave([table], {"synthetic": Q}, gravity=G)
    b = CorrelatedKLongwave([padded], {"synthetic": Q}, gravity=G)
    assert b.n_points == a.n_points == table.n_band * table.n_g
    fa, fb = Radiation(a).fluxes(col, low), Radiation(b).fluxes(col, low)
    np.testing.assert_allclose(fb.lw_up, fa.lw_up, rtol=1e-13)
    np.testing.assert_allclose(fb.heating, fa.heating, rtol=1e-10, atol=1e-18)
    with pytest.raises(ValueError, match="each row sum to 1"):
        KTable("synthetic", table.pressure, table.temperature, table.band_edges, 0.5 * w, k)


def test_premixed_table_is_held_to_its_gas(table, tmp_path, monkeypatch):
    mixed = KTable("mix", table.pressure, table.temperature, table.band_edges,
                   table.g_weights, table.k, "", {"mass_basis": {
                       "kind": "premixed", "gas_constant": R, "composition": {"X": 1.0}}})
    mixed.save(tmp_path / "mix.npz")
    monkeypatch.setenv(OPACITY_PATH_ENV, str(tmp_path))
    spec = table_spec(tmp_path / "mix.npz")
    lw = {"kind": "correlated_k", "tables": [spec], "mass_fractions": {"mix": 1.0}}
    cfg = _config(spec, longwave=lw, shortwave=None, insolation=None,
                  internal_heat_flux=0.0)
    build_radiation(cfg)                                            # matching gas: accepted
    with pytest.raises(KTableError, match="radiative and the bulk composition"):
        build_radiation(cfg.with_(gas_constant=188.9))
    with pytest.raises(KTableError, match="must be 1.0"):
        build_radiation(cfg.with_(longwave={**lw, "mass_fractions": {"mix": 0.5}}))


# ---------------------------------------------------------------------------
# The THAI dry development presets
# ---------------------------------------------------------------------------

def test_thai_presets_differ_only_through_inputs():
    """Ben 1 and Ben 2 are one configuration with two sets of inputs: the
    composition (hence R and c_p) and the k-table mixed for it."""
    from tropoi.physics.composition import COMPOSITIONS
    from tropoi.physics.thermodynamics import heat_capacity
    from tropoi.run.radiative.config import PRESETS, THAI_CP_TEMPERATURE
    ben1, ben2 = PRESETS["dev-thai-ben1"](), PRESETS["dev-thai-ben2"]()
    d1, d2 = ben1.to_dict(), ben2.to_dict()
    assert {k for k in d1 if d1[k] != d2[k]} == {"experiment_id", "gas_constant", "cp",
                                                 "longwave"}
    for cfg, name in ((ben1, "thai_ben1"), (ben2, "thai_ben2")):
        comp = COMPOSITIONS[name]
        assert cfg.gas_constant == comp.gas_constant
        assert cfg.cp == float(heat_capacity(comp, THAI_CP_TEMPERATURE))
        assert cfg.longwave["mass_fractions"] == {name: 1.0}
        assert cfg.longwave["bounds"] == "error"
        assert cfg.experiment_id.startswith("dev-thai-")
        assert RadiativeConfig.from_dict(cfg.to_dict()).sha256() == cfg.sha256()
    assert ben1.gas_constant == pytest.approx(296.729, abs=1e-3)      # N2 + 400 ppmv CO2
    assert ben2.gas_constant == pytest.approx(188.926, abs=1e-3)      # CO2


@pytest.mark.parametrize("case", ["dev-thai-ben1", "dev-thai-ben2"])
def test_thai_imported_table_when_present(case):
    """With the imported tables in ``opacity_cache/tables`` (not shipped;
    docs/radiation/OPACITY.md), the preset loads the pinned table, the
    table was mixed for the preset's gas, an isothermal column emits
    sigma T^4 over the tabulated bands whatever the gas, and a state
    outside the table is refused."""
    from tropoi.physics.radiation.scheme import STEFAN_BOLTZMANN
    from tropoi.run.radiative.config import PRESETS
    cfg = PRESETS[case]()
    try:
        table = load_table(cfg.longwave["tables"][0])
    except KTableError as exc:
        if "not found" in str(exc):
            pytest.skip("imported THAI k-tables are not present")
        raise
    basis = table.metadata["mass_basis"]
    assert basis["kind"] == "premixed" and basis["gas_constant"] == cfg.gas_constant
    assert table.metadata["coverage"]["temperature_k"] == [100.0, 400.0]
    assert len(table.metadata["source"]["files"]) == 2
    rad = build_radiation(cfg)
    p = (build_sigma(cfg).interfaces_array() * cfg.reference_pressure)[:, None]
    t = np.full((p.shape[0] - 1, 1), 300.0)
    fl = rad.fluxes(Column(p, t, cfg.gravity, cfg.cp),
                    LowerBoundary("surface", temperature=np.full(1, 300.0)))
    expected = STEFAN_BOLTZMANN * 300.0 ** 4 * rad.longwave.spectral_coverage(300.0)
    assert fl.olr[0] == pytest.approx(expected, rel=1e-6)
    column_heating = (cfg.cp * (p[1:] - p[:-1]) / cfg.gravity * fl.heating).sum()
    assert column_heating == pytest.approx(float(fl.net_down[0, 0] - fl.net_down[-1, 0]),
                                           rel=1e-12)
    with pytest.raises(ValueError, match="outside the k-table"):
        rad.fluxes(Column(p, t - 210.0, cfg.gravity, cfg.cp),
                   LowerBoundary("surface", temperature=np.full(1, 300.0)))


def test_recorded_thai_configurations_keep_their_hashes():
    """The configurations of the recorded development runs
    (docs/radiation/evidence/dev-thai-ben*-lwck-heated-T21L40-9f8c776)."""
    from tropoi.run.radiative.config import PRESETS
    assert PRESETS["dev-thai-ben2"]().sha256() == (
        "80e5a1824e5b378f4d97491f13e6d0aa0479fb2d41b5ad5647cdc40195dd1d28")
    assert PRESETS["dev-thai-ben1"]().sha256() == (
        "72494a3143b330c9221721257d5aa4e9e3ef2379ff99723bda8520b5b0dad0c1")


def test_ben2_layer_linear_preset_is_a_distinct_opt_in_experiment():
    """The corrected Ben 2 configuration (issue #47) selects the
    layer-centered source under its own experiment id, leaving the recorded
    ``dev-thai-ben2-ck`` (fdbcf1cf...) untouched. With the historical id it
    is exactly the configuration of the archived replay
    (docs/radiation/evidence/dev-thai-ben2-ck-heated-T21L40-9b2060c-layer-linear,
    3f2de9d8...), which was run as ``dev-thai-ben2-ck`` with the source
    overridden."""
    from tropoi.run.radiative.config import PRESETS
    ck, ll = PRESETS["dev-thai-ben2-ck"](), PRESETS["dev-thai-ben2-ck-layer-linear"]()
    assert ck.thermal_source == "linear"
    assert ck.sha256() == "fdbcf1cfd7ebcfaaf3ca887896c3cc6a049a371abdadc5985c78131304c209f3"
    assert ll.thermal_source == "layer-linear"
    assert ll.experiment_id == "dev-thai-ben2-ck-layer-linear-heated-T21L40"
    assert ll.sha256() == "08bc87147ef557ae3161d57b9caa533c3f89caf680151b7dff5dfdc17c374af2"
    d0, d1 = ck.to_dict(), ll.to_dict()
    assert {k for k in d0 if d0[k] != d1[k]} == {"experiment_id", "thermal_source"}
    replay = ll.with_(experiment_id=ck.experiment_id)
    assert replay == ck.with_(thermal_source="layer-linear")
    assert replay.sha256() == (
        "3f2de9d8985c2bc1405cced826bb84755860ddf6ed68986aeb702186d2536eff")
    # Ben 1 keeps the historical source; switching it is a separate decision
    assert "dev-thai-ben1-ck-layer-linear" not in PRESETS


@pytest.mark.parametrize("case", ["ben1", "ben2"])
def test_thai_stellar_absorption_is_one_more_input(case):
    """The ``-ck`` presets are the default ones with a stellar k-table and
    the star's band fractions in place of the transparent shortwave."""
    from tropoi.run.radiative.config import PRESETS
    base, ck = PRESETS[f"dev-thai-{case}"](), PRESETS[f"dev-thai-{case}-ck"]()
    d0, d1 = base.to_dict(), ck.to_dict()
    assert {k for k in d0 if d0[k] != d1[k]} == {"experiment_id", "shortwave"}
    sw = ck.shortwave
    assert sw["kind"] == "correlated_k" and sw["bounds"] == "error"
    assert sw["mass_fractions"] == ck.longwave["mass_fractions"]
    assert sum(sw["band_fractions"]) == pytest.approx(1.0, abs=1e-9)
    assert "-ck-" in ck.experiment_id and "-lwck-" in base.experiment_id
    try:
        rad = build_radiation(ck)
    except KTableError as exc:
        if "not found" in str(exc):
            pytest.skip("imported THAI k-tables are not present")
        raise
    assert len(sw["band_fractions"]) == rad.shortwave.n_band
    p = (build_sigma(ck).interfaces_array() * ck.reference_pressure)[:, None]
    mid = 0.5 * (p[1:] + p[:-1])
    t = np.maximum(250.0 * (mid / ck.reference_pressure) ** ck.kappa, 160.0)
    fl = rad.fluxes(Column(p, t, ck.gravity, ck.cp),
                    LowerBoundary("surface", temperature=np.full(1, 250.0), albedo=0.3),
                    toa_flux=np.full(1, 225.0), mu0=np.full(1, 0.5))
    absorbed = float(fl.absorbed_stellar[0])
    at_surface = float(fl.sw_down[-1, 0] - fl.sw_up[-1, 0])
    # absorption only: the gas can only darken the planet below its surface albedo
    assert 0.7 * 225.0 < absorbed < 225.0 and 0.0 < at_surface < 0.7 * 225.0
    column_heating = (ck.cp * (p[1:] - p[:-1]) / ck.gravity * fl.heating).sum()
    assert column_heating == pytest.approx(float(fl.net_down[0, 0] - fl.net_down[-1, 0]),
                                           rel=1e-12)
