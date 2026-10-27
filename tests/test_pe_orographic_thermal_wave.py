"""``orographic_thermal_wave``: a small perturbation of the balanced base.

The preset composes the unchanged ``orographic_isothermal_rest`` constructor
with the ``thermal_wave`` temperature perturbation (one real coefficient on
the (2, 2) mode at every full level). Checks:

* configuration (CPU): registered; an additive ``initial_perturbation``
  record (base state, mode, real/imaginary coefficient, normalization,
  pointwise peak, vertical weights) emitted for this preset only, so every
  other scenario's run dictionary, and therefore its run id, is unchanged;
  the amplitude is part of the scientific identity; the thermal-wave
  support boundary applies, naming this scenario;
* the recorded pointwise peak equals the host synthesis of the mode;
* construction (GPU): zero amplitude is bitwise ``orographic_isothermal_rest``;
  a nonzero amplitude changes exactly the (2, 2) temperature coefficient of
  every level by the amplitude, leaving ln p_s, vorticity and divergence
  bitwise at their base values;
* evolution (GPU): a short fixed-step run over a mountain stays finite and
  physical, generates winds, and differs from the orographic-rest control by
  far more than the control's own drift.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

T0 = 210.0
PS0 = 610.0
GRAVITY = 3.71
R_DRY, CP_DRY = 191.8, 735.0
NLEV = 4
MOUNTAIN = dict(height_m=6000.0, lat_deg=10.0, lon_deg=-120.0, width_deg=25.0)


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(),
                                   reason="CUDA/CuPy not available")


# ---------------------------------------------------------------------------
# Configuration (CPU)
# ---------------------------------------------------------------------------

def _cfg(**extra):
    from tropoi.run.pe.config import PERunConfig
    base = {"scenario": "orographic_thermal_wave", "grid": "latlon",
            "lmax": 15, "nlat": 32, "nlon": 64, "nlev": NLEV,
            "topography": "mountain"}
    base.update(extra)
    return PERunConfig.resolve(base)


def test_preset_is_registered_everywhere():
    from tropoi.run.pe.config import PE_SCENARIOS
    from tropoi.cli.main import build_parser

    assert "orographic_thermal_wave" in PE_SCENARIOS
    args = build_parser().parse_args(
        ["run", "pe", "--scenario", "orographic_thermal_wave"])
    assert args.scenario == "orographic_thermal_wave"


def test_perturbation_record_is_additive_and_exact():
    from tropoi.run.pe.config import PERunConfig

    cfg = _cfg(thermal_amplitude=1.0)
    rec = cfg.to_run_config_dict()["initial_perturbation"]
    assert rec == {
        "base_state": "orographic_isothermal_rest", "field": "temperature",
        "degree": 2, "order": 2, "coefficient_real_K": 1.0,
        "coefficient_imag_K": 0.0,
        "normalization": "orthonormal complex Y_lm; real field = a_l0 Y_l0 "
                         "+ 2 Re sum_{m>0} a_lm Y_lm",
        "pointwise_peak_K": pytest.approx(0.7725484040463791, rel=1e-15),
        "vertical_profile": "uniform", "vertical_weights": [1.0] * NLEV}
    assert "initial_perturbation" in cfg.scientific_config_dict()
    assert "pointwise peak 0.7725 K" in "\n".join(cfg.summary_lines())

    # every other scenario keeps its historical dictionary
    for scenario in ("isothermal_rest", "thermal_wave",
                     "orographic_isothermal_rest"):
        d = PERunConfig.resolve({"scenario": scenario}).to_run_config_dict()
        assert "initial_perturbation" not in d, scenario
    rest = _cfg(scenario="orographic_isothermal_rest").to_run_config_dict()
    wave = _cfg().to_run_config_dict()
    assert set(wave) - set(rest) == {"initial_perturbation"}


def test_amplitude_is_part_of_the_scientific_identity():
    a = _cfg(thermal_amplitude=1.0).scientific_config_dict()
    b = _cfg(thermal_amplitude=0.5).scientific_config_dict()
    assert a != b
    assert b["initial_perturbation"]["coefficient_real_K"] == 0.5


def test_thermal_wave_support_boundary_applies():
    with pytest.raises(ValueError, match="orographic_thermal_wave"):
        _cfg(lmax=2, nlat=16, nlon=32, thermal_amplitude=1.0)
    with pytest.raises(ValueError, match="orographic_thermal_wave"):
        _cfg(lmax=1, nlat=16, nlon=32, topography="flat",
             thermal_amplitude=0.0)
    _cfg(lmax=3, nlat=16, nlon=32, topography="flat")      # retains l = 2


def test_recorded_peak_is_the_synthesized_maximum():
    from tropoi.run.pe.config import THERMAL_WAVE_PEAK_PER_COEFFICIENT
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    lm = np.zeros((6, 6), dtype=np.complex128)
    lm[2, 2] = 1.0
    field = synthesize_latlon(lm, np.linspace(-90, 90, 361),
                              np.linspace(-180, 180, 721))
    assert field.max() == pytest.approx(THERMAL_WAVE_PEAK_PER_COEFFICIENT,
                                        rel=1e-12)
    assert -field.min() == pytest.approx(THERMAL_WAVE_PEAK_PER_COEFFICIENT,
                                         rel=1e-12)


# ---------------------------------------------------------------------------
# Construction and evolution (GPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def terrain_model():
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.spatial.terrain.topography import Topography
    from tropoi.spatial.truncation import product_truncation_cut
    from tropoi.temporal.tendencies.primitive_equations import (
        PrimitiveEquationsModel)

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=24.6229, radius_earth_units=0.53307),
        grid_type="latlon", nlat=24, nlon=48, l_max=15, grid_resolution=3,
        product_quadrature="fine")
    topo = Topography.mountain(planet, l_cut=product_truncation_cut(15),
                               **MOUNTAIN)
    return PrimitiveEquationsModel(
        planet, SigmaGrid.uniform(NLEV), r_dry=R_DRY, cp_dry=CP_DRY,
        surface_geopotential_lm=topo.surface_geopotential_lm(GRAVITY))


def _state(model, scenario, amplitude=0.0):
    from tropoi.spatial.initialization.pe import make_pe_ic
    return make_pe_ic(scenario, model, temperature=T0, surface_pressure=PS0,
                      thermal_amplitude=amplitude)


@requires_cuda
def test_zero_amplitude_is_bitwise_orographic_rest(terrain_model):
    import cupy as cp
    rest = _state(terrain_model, "orographic_isothermal_rest")
    wave = _state(terrain_model, "orographic_thermal_wave", 0.0)
    assert cp.asnumpy(wave.coeffs).tobytes() == \
        cp.asnumpy(rest.coeffs).tobytes()


@requires_cuda
def test_perturbation_is_one_mode_on_every_level_and_nothing_else(
        terrain_model):
    import cupy as cp
    from tropoi.spatial.initialization.pe import thermal_wave_peak

    K = NLEV
    rest = cp.asnumpy(_state(terrain_model,
                             "orographic_isothermal_rest").coeffs)
    wave = cp.asnumpy(_state(terrain_model, "orographic_thermal_wave",
                             1.0).coeffs)
    diff = wave - rest
    expected = np.zeros_like(diff)
    expected[2 * K:3 * K, 2, 2] = 1.0
    assert np.array_equal(diff, expected)
    # vorticity, divergence and ln p_s are the base's, bit for bit
    for block in (slice(0, K), slice(K, 2 * K), slice(3 * K, 3 * K + 1)):
        assert wave[block].tobytes() == rest[block].tobytes()
    # the base surface pressure really follows the terrain
    assert np.any(rest[3 * K, 1:])
    assert thermal_wave_peak(1.0) == pytest.approx(0.7725484040463791)


def _run(model, scenario, out_dir, *, amplitude, dt=300.0, n_steps=24):
    from tropoi.run.pe.runner import run_pe
    from tropoi.temporal.integration import count_snapshot_times

    t_end = n_steps * dt
    run_pe(model, _state(model, scenario, amplitude), dt_seconds=dt,
           t_end_days=t_end / 86400.0, out_dir=pathlib.Path(out_dir),
           snapshot_times=count_snapshot_times(3, t_end),
           snapshot_mode="count", dt_snapshots=t_end / 2, plots=(),
           scenario=scenario)
    import csv
    with open(pathlib.Path(out_dir) / "diagnostics" / "timeseries.csv",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.load(pathlib.Path(out_dir) / "pe_coeffs.npy"), rows


@requires_cuda
def test_short_run_over_terrain_evolves_and_stays_physical(terrain_model,
                                                           tmp_path):
    wave, rows = _run(terrain_model, "orographic_thermal_wave",
                      tmp_path / "wave", amplitude=1.0)
    rest, rest_rows = _run(terrain_model, "orographic_isothermal_rest",
                           tmp_path / "rest", amplitude=0.0)
    assert np.all(np.isfinite(wave))
    t_min = min(float(r["t_min"]) for r in rows)
    ps_min = min(float(r["ps_min"]) for r in rows)
    assert t_min > 0.0 and ps_min > 0.0
    # total_mass integrates p_s = exp(ln p_s) on the grid. The continuous
    # equations conserve it, but the scheme advances ln p_s, whose tendency
    # (-V.grad ln p_s - delta, column-integrated) is evaluated on the grid
    # and truncated, so mass is conserved only to the spatial truncation
    # (the ln p_s monopole itself is not an invariant over terrain). The
    # 2 h run drifts by ~3e-10; the 1-sol MOLA run by up to 6e-7, the same
    # at 60 s and 30 s steps
    assert max(abs(float(r["mass_rel_drift"])) for r in rows) < 1e-8

    wind = float(rows[-1]["max_wind_ms"])
    rest_wind = max(float(r["max_wind_ms"]) for r in rest_rows)
    assert wind > 1e-3                              # it moves
    assert rest_wind < 1e-9                         # the control does not
    evolution = np.abs(wave[-1] - wave[0]).max()
    control = np.abs(rest[-1] - rest[0]).max()
    assert evolution > 1e3 * max(control, 1e-30)
