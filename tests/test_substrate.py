"""Fast CPU regression tests for the conductive substrate.

The original acceptance gates (G1-G5b) live in ``audit_substrate_gates.py``.
The nonlinear, stratified gates (G-B) live in ``test_substrate_nonlinear.py``.
These are quick checks of layering, sign convention, the energy identity,
the one-layer slab limit and the Robin/skin response, written against the
generalized kernel (Planet One). The layer-mean skin mode was removed with
the owner's choice of the diagnosed skin, so its checks are gone.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

from tropoi.physics.substrate import (ConductiveSubstrate, ENERGY_REFERENCE_TEMPERATURE,
                                      geometric_interfaces)
from tropoi.physics.surface import SlabSurface


def _prescribed(**kw):
    """A constant-property column with a non-radiating surface: the forcing
    is the prescribed flux alone."""
    return ConductiveSubstrate.constant(emissivity=0.0, **kw)


def test_default_layering_is_thai_column():
    sub = _prescribed()
    assert sub.layers == 30
    assert sub.interfaces[0] == 0.0 and sub.interfaces[-1] == 3.5
    assert math.isclose(sub.thickness[0], 0.02, rel_tol=1e-12)
    ratios = sub.thickness[1:] / sub.thickness[:-1]
    assert np.all(ratios <= 1.5) and np.all(ratios >= 1.0)
    st = sub.initial_state(1)
    assert np.allclose(sub.conductivity(st), 2.0, rtol=1e-15, atol=0)
    assert sub.signature()["kind"] == "conductive"


def _ratios(edges):
    dz = np.diff(edges)
    return dz[1:] / dz[:-1]


def test_stretch_layering_merges_thin_remainder():
    # 0.1, 0.12, 0.144, 0.1728, 0.20736 fit to 0.74416, then 0.248832 to 0.992992.
    # The 0.007008 m gap is merged into the last layer: (0.248832 + 0.007008) / 0.20736
    # = 1.2338 lies in [1, 1.5], while keeping it as a layer would give 0.028.
    e = geometric_interfaces(0.1, 1.0, stretch=1.2)
    expect = [0.0, 0.1, 0.22, 0.364, 0.5368, 0.74416, 1.0]
    assert len(e) == 7 and e[-1] == 1.0
    np.testing.assert_allclose(e, expect, rtol=0, atol=1e-12)
    assert math.isclose(np.diff(e)[-1], 0.25584, rel_tol=1e-9)
    assert np.all((_ratios(e) >= 1.0) & (_ratios(e) <= 1.5))


def test_stretch_layering_keeps_remainder_as_layer_when_merge_is_too_steep():
    # The gap after 0.74416 is 0.22 m (the next 0.248832 layer does not fit in 0.96416).
    # Merging would give (0.20736 + 0.22) / 0.1728 = 2.47; keeping it gives 0.22 / 0.20736 = 1.061.
    e = geometric_interfaces(0.1, 0.96416, stretch=1.2)
    np.testing.assert_allclose(e, [0.0, 0.1, 0.22, 0.364, 0.5368, 0.74416, 0.96416], atol=1e-12)
    assert math.isclose(_ratios(e)[-1], 0.22 / 0.20736, rel_tol=1e-9)


def test_stretch_layering_rejects_incompatible_constraints():
    # 0.1 + 0.15 fit in 0.3; the 0.05 m gap gives a final ratio of 2.0 (merge) or 0.33 (layer).
    with pytest.raises(ValueError, match="incompatible"):
        geometric_interfaces(0.1, 0.3, stretch=1.5)
    # The reviewer's example: the old code returned a final ratio of 2.333 here.
    with pytest.raises(ValueError, match="incompatible"):
        geometric_interfaces(0.1, 1.0, stretch=1.5)


def test_depth_as_minimum_keeps_every_ratio_exact():
    e = geometric_interfaces(0.1, 0.3, stretch=1.5, depth_is_minimum=True)
    np.testing.assert_allclose(e, [0.0, 0.1, 0.25, 0.475], atol=1e-12)
    np.testing.assert_allclose(_ratios(e), 1.5, rtol=1e-12)
    exact = geometric_interfaces(0.1, 0.25, stretch=1.5, depth_is_minimum=True)
    assert exact[-1] == 0.25 and len(exact) == 3          # already a geometric interface


def test_layering_boundary_and_argument_errors():
    assert geometric_interfaces(0.1, 1.0, stretch=1.0)[-1] == 1.0     # uniform, exact fit
    e = geometric_interfaces(0.1, 0.475, stretch=1.5)                 # exact fit at ratio 1.5
    assert e[-1] == 0.475 and np.all(_ratios(e) <= 1.5 + 1e-12)
    for bad in (dict(stretch=1.6), dict(stretch=0.9), dict(stretch=float("nan")),
                dict(layers=5),                                     # needs ratio > 1.5
                dict(layers=30, stretch=1.1), dict(),
                dict(layers=30, depth_is_minimum=True)):
        with pytest.raises(ValueError):
            geometric_interfaces(0.02, 3.5, **bad)
    for top, depth in ((float("nan"), 1.0), (0.1, float("inf")), (0.0, 1.0), (2.0, 1.0)):
        with pytest.raises(ValueError):
            geometric_interfaces(top, depth, stretch=1.2)


def test_layers_mode_near_uniform_and_argument_checks():
    # Near ratio 1 the geometric sum is evaluated without cancellation.
    e = geometric_interfaces(0.1, 1.0 + 1e-9, layers=10)
    assert e[-1] == 1.0 + 1e-9 and np.all(_ratios(e) >= 1.0 - 1e-9)
    with pytest.raises(ValueError, match="cannot reach depth"):
        geometric_interfaces(0.02, 3.5, layers=5)          # needs a ratio of about 2.7
    for bad in (2.9, True, "30"):
        with pytest.raises((ValueError, TypeError)):
            geometric_interfaces(0.02, 3.5, layers=bad)
    with pytest.raises(ValueError):
        ConductiveSubstrate.constant(layers=29.9)


def test_generated_grids_never_leave_the_ratio_bounds():
    rng = np.random.default_rng(3)
    accepted = rejected = 0
    for _ in range(3000):
        top = 10 ** rng.uniform(-3, -1)
        depth = top * 10 ** rng.uniform(0.5, 3.5)
        ratio = rng.uniform(1.0, 1.5)
        minimum = bool(rng.integers(2))
        try:
            e = geometric_interfaces(top, depth, stretch=ratio, depth_is_minimum=minimum)
        except ValueError:
            rejected += 1
            assert not minimum
            continue
        accepted += 1
        assert e[0] == 0.0 and math.isclose(e[1], top, rel_tol=1e-12)
        r = _ratios(e)
        assert np.all(r >= 1.0 - 1e-12) and np.all(r <= 1.5 + 1e-12)
        if minimum:
            assert e[-1] >= depth * (1 - 1e-12)
        else:
            assert e[-1] == depth
    assert accepted > 0 and rejected > 0


def test_constructor_rejects_conflicting_geometry():
    for kwargs in (dict(interfaces=(0.0, 1.0), depth=3.5), dict(interfaces=(0.0, 1.0), layers=1),
                   dict(interfaces=(0.0, 1.0), top_thickness=0.02),
                   dict(interfaces=(0.0, 1.0), stretch=1.1), dict(layers=18, stretch=1.2)):
        with pytest.raises(ValueError):
            ConductiveSubstrate.constant(**kwargs)
    sub = ConductiveSubstrate.constant(stretch=1.2, depth=1.0, top_thickness=0.1)
    assert sub.layers == 6 and sub.depth == 1.0


def test_constructor_rejects_nonfinite_parameters():
    for kwargs in (dict(bottom_heat_flux=float("nan")), dict(initial_temperature=float("nan")),
                   dict(initial_temperature=0.0), dict(interfaces=(0.0, float("nan"))),
                   dict(albedo=float("nan")), dict(volumetric_heat_capacity=float("inf")),
                   dict(thermal_inertia=2000.0, conductivity=2.0)):
        with pytest.raises(ValueError):
            ConductiveSubstrate.constant(**kwargs)


def test_step_rejects_invalid_inputs():
    sub = _prescribed()
    st = sub.initial_state(2)
    f = np.array([10.0, -10.0])
    for dt in (0.0, -600.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sub.step(st, dt, flux=f)
    with pytest.raises(ValueError):
        sub.step(st, 600.0, flux=np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        sub.step(st, 600.0, flux=f, bottom_heat_flux=np.array([0.0, np.inf]))
    for a in (np.array([-1.0, 200.0]), np.array([-1.0, np.nan]), 1e-9):
        with pytest.raises(ValueError):
            sub.step(st, 600.0, flux=f, flux_derivative=a)
    ok, _ = sub.step(st, 600.0, flux=f, flux_derivative=np.array([-5.0, 0.0]))
    assert np.all(np.isfinite(ok.anomaly))


def _fixture():
    path = pathlib.Path(__file__).resolve().parent / "fixtures" / "substrate_m1_historical_grids.json"
    return json.loads(path.read_text())


def test_historical_grids_fixture():
    fx = _fixture()
    assert fx["source_commit"].startswith("cc33bfb")
    grids = fx["grids"]
    # The production default is bitwise the grid the original run used.
    assert list(ConductiveSubstrate.constant().interfaces) == grids["default"]
    for key, e in grids.items():
        e = np.asarray(e)
        assert e[0] == 0.0 and np.all(np.diff(e) > 0), key
        ConductiveSubstrate.constant(interfaces=e)   # explicit grids are accepted as given
    # Every gate grid of the original run is present. 'uniform:N=400' is the extra grid of
    # the post hoc G1-C supplementary study (not a gate grid), kept so it is documented.
    expected = {"default", "G4", "uniform:N=400"}
    expected |= {f"G1M:P={p}" for p in (1.0, 2.0, 6.1, 20.0, 61.0, 200.0)}
    expected |= {f"uniform:N={n}" for n in (25, 50, 100, 200)}
    expected |= {f"G5b:periodic:P={p}:top={t}" for p in (1.0, 6.1, 61.0) for t in (0.005, 0.01, 0.02, 0.04)}
    expected |= {f"G5b:step:top={t}" for t in (0.005, 0.01, 0.02, 0.04)}
    assert set(grids) == expected
    # The documented historical defect: the 4 cm G5b grid ends with a ratio of 1.657.
    assert math.isclose(_ratios(grids["G5b:step:top=0.04"])[-1], 1.6565, abs_tol=1e-4)


def test_gate_harness_rejects_unknown_gate_names(tmp_path, capsys):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        import audit_substrate_gates as gates
    finally:
        sys.path.pop(0)
    assert gates.main([str(tmp_path / "out"), "--only", "G1-M"]) == 2
    assert "unknown gate name" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_bottom_flux_enters_bottom_row():
    sub = _prescribed()
    st, _ = sub.step(sub.initial_state(1), 600.0, flux=0.0, bottom_heat_flux=5.0)
    theta = st.anomaly[:, 0]
    assert theta[-1] > 0.0
    assert np.all(np.diff(theta) >= 0.0)                # warms from below
    assert theta[0] < 1e-6 * theta[-1]


def test_energy_identity_with_signs():
    """G5's criterion: cumulative |E - sum dt (F_s + F_b)| <= 1e-12 sum dt |F_s + F_b|."""
    sub = _prescribed()
    rng = np.random.default_rng(1)
    fb = np.array([0.0, 0.1, -0.1])
    st = sub.initial_state(3)
    inflow = np.zeros(3)
    denom = np.zeros(3)
    for _ in range(2000):
        f = rng.uniform(-200.0, 200.0, size=3)
        st, rep = sub.step(st, 600.0, flux=f, bottom_heat_flux=fb)
        assert rep.iterations == 1                       # linear problem: one Newton step
        inflow += 600.0 * (rep.conducted + fb)
        denom += 600.0 * np.abs(rep.conducted + fb)
        resid = np.abs(np.asarray(sub.energy(st)) - inflow)
        assert np.all(resid <= 1e-12 * denom + 1e-30)


def test_one_layer_matches_slab():
    sub = _prescribed(interfaces=(0.0, 1.0))
    slab = SlabSurface(2.0e6)
    rng = np.random.default_rng(2)
    st = sub.initial_state(1)
    T = np.array([ENERGY_REFERENCE_TEMPERATURE])
    for _ in range(1000):
        f = rng.uniform(-200.0, 200.0, size=1)
        st, _ = sub.step(st, 600.0, flux=f)
        T = slab.advance(T, f, 600.0)
    assert abs(float(sub.temperature(st)[0, 0]) - float(T[0])) < 1000 * np.spacing(300.0)


def test_robin_linearization_is_exact():
    sub = _prescribed()
    h = 8.14
    st = sub.initial_state(1)
    st = st._replace(anomaly=st.anomaly + 5.0, surface_anomaly=st.surface_anomaly + 5.0)
    ts_old = float(sub.surface_temperature(st)[0]) - 300.0
    new, rep = sub.step(st, 600.0, flux=np.array([-h * ts_old]), flux_derivative=-h)
    t1 = float(new.anomaly[0, 0])
    r = sub.thickness[0] / (2.0 * 2.0)
    assert math.isclose(float(rep.conducted[0]), -h * t1 / (1.0 + h * r), rel_tol=1e-12)
    ts_new = float(sub.surface_temperature(new)[0]) - 300.0
    assert math.isclose(float(rep.conducted[0]), -h * ts_new, rel_tol=1e-12)


def test_step_flux_surface_response():
    """G2 at the default layering (the gate's own criterion, as a regression)."""
    sub = _prescribed()
    st = sub.initial_state(1)
    F, dt, k, kappa = 100.0, 600.0, 2.0, 1e-6
    z = 0.5 * sub.thickness[0]
    for n in range(1, 400):
        st, _ = sub.step(st, dt, flux=np.array([F]))
    t = 399 * dt
    s = math.sqrt(kappa * t)
    x = z / (2 * s)
    ana = 2 * F / k * s * (math.exp(-x * x) / math.sqrt(math.pi) - x * math.erfc(x))
    # layer mean against point value at the centre differs at O(dz^2); 1 % covers it
    assert abs(float(st.anomaly[0, 0]) / ana - 1.0) < 0.01
