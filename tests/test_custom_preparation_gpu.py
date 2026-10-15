"""Preparation from physical fields and callables (projection, terrain,
positivity) and equivalence with the preset constructions. Needs CUDA.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU")

L = 15
NLAT, NLON = 24, 48


def _disc(lmax=L):
    from tropoi.run.custom import DiscretizationSpec
    return DiscretizationSpec.gauss_latlon(lmax, NLAT, NLON)


def test_state_sampling_is_the_model_grid():
    from tropoi.run.custom import DiscretizationSpec, PlanetSpec, state_sampling

    planet = PlanetSpec.earth_like()
    s = state_sampling(planet, _disc())
    assert s.shape == (NLAT, NLON)
    assert s.latitudes[0, 0] > 0 > s.latitudes[-1, 0]          # north first
    assert s.longitudes[0, 1] == pytest.approx(2 * math.pi / NLON)
    assert s.weights.sum() == pytest.approx(4 * math.pi, rel=1e-12)
    geo = state_sampling(planet, DiscretizationSpec.geodesic(10, 3))
    assert geo.shape == (geo.latitudes.size,) and geo.grid == "geodesic"


def test_band_limited_callable_projects_exactly_on_the_gauss_grid():
    from tropoi.run.custom import (PlanetSpec, SampledField, SampledFunction,
                                   prepare_bve_state, state_sampling)

    planet = PlanetSpec.ideal_sphere(3.0e6, 40000.0)
    c = 3e-5

    def zeta(lat, lon):
        assert isinstance(lat, np.ndarray) and lat.dtype == np.float64
        return c * np.sin(lat) + 1e-6 * np.cos(lat) ** 3 * np.cos(3 * lon)

    prepared = prepare_bve_state(planet, _disc(), zeta=zeta)
    entry = prepared.report.fields[0]
    assert entry.source == "function" and entry.function.endswith("zeta")
    assert entry.relative_residual < 1e-13
    a = prepared.state.coeffs
    assert a[1, 0].real == pytest.approx(c * math.sqrt(4 * math.pi / 3),
                                         rel=1e-13)
    assert a[1, 0].imag == 0.0 and a[0, 0] == pytest.approx(0, abs=1e-20)
    # The same samples given as an array produce the same coefficients.
    s = state_sampling(planet, _disc())
    again = prepare_bve_state(planet, _disc(), zeta=SampledField(
        zeta(s.latitudes, s.longitudes))).state
    assert again.coeffs.tobytes() == a.tobytes()
    assert again.digest == prepared.state.digest


def test_unrepresentable_input_fails_with_its_residual_and_never_smooths():
    from tropoi.run.custom import (PlanetSpec, PreparationError,
                                   ProjectionPolicy, prepare_bve_state)

    def narrow_vortex(lat, lon):          # far narrower than l_max=15 resolves
        d = np.arccos(np.clip(np.sin(lat) * np.sin(0.5) + np.cos(lat)
                              * np.cos(0.5) * np.cos(lon - 1.0), -1, 1))
        return 1e-4 * np.exp(-(d / 0.08) ** 2)

    planet = PlanetSpec.earth_like()
    with pytest.raises(PreparationError, match="residual") as info:
        prepare_bve_state(planet, _disc(), zeta=narrow_vortex,
                          policy=ProjectionPolicy(subtract_mean=("zeta",)))
    residual = info.value.report.fields[0].relative_residual
    assert residual > 1e-2
    relaxed = prepare_bve_state(
        planet, _disc(), zeta=narrow_vortex,
        policy=ProjectionPolicy(max_relative_residual=1.0,
                                subtract_mean=("zeta",)))
    record = relaxed.state.provenance["preparation"]["fields"][0]
    assert record["relative_residual"] == pytest.approx(residual)
    assert record["subtracted_mean"] != 0.0


@pytest.mark.parametrize("bad,match", [
    (lambda lat, lon: np.zeros(3), "shape"),
    (lambda lat, lon: [0.0] * lat.size, "numpy.ndarray"),
    (lambda lat, lon: np.full(lat.shape, np.nan), "finite"),
    (lambda lat, lon: 1 / 0, "ZeroDivisionError"),
])
def test_bad_callables_are_reported(bad, match):
    from tropoi.run.custom import (PlanetSpec, PreparationError,
                                   prepare_bve_state)

    with pytest.raises(PreparationError, match=match):
        prepare_bve_state(PlanetSpec.earth_like(), _disc(), zeta=bad)


def test_cupy_results_are_rejected():
    import cupy as cp
    from tropoi.run.custom import (PlanetSpec, PreparationError,
                                   prepare_bve_state)

    with pytest.raises(PreparationError, match="cupy.asnumpy"):
        prepare_bve_state(PlanetSpec.earth_like(), _disc(),
                          zeta=lambda lat, lon: cp.zeros(lat.shape))


def test_two_vortices_from_a_callable_matches_the_preset_projection():
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.custom import (ProjectionPolicy,
                                   initial_state_from_config,
                                   planet_spec_for_run_config,
                                   prepare_bve_state)

    cfg = BVERunConfig.resolve({"scenario": "two_vortices", "grid": "latlon",
                                "nlat": NLAT, "nlon": NLON, "lmax": L,
                                "day_hours": 24.0})
    preset = initial_state_from_config(cfg)

    def gaussian(lat, lon, lat0, lon0, sigma):
        cosd = (np.sin(lat) * np.sin(lat0)
                + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0))
        d = np.arccos(np.clip(cosd, -1.0, 1.0))
        return np.exp(-(d * d) / (2 * sigma * sigma))

    def two_vortices(lat, lon):
        s = 10.0 * np.pi / 180.0
        return (5e-5 * gaussian(lat, lon, 33 * np.pi / 180, np.pi / 2, s)
                - 5e-5 * gaussian(lat, lon, -33 * np.pi / 180, -np.pi / 2, s))

    # The Gaussians are not band-limited at L=15: the residual is reported
    # (and must be accepted explicitly), exactly as the preset projects it.
    prepared = prepare_bve_state(planet_spec_for_run_config(
        cfg.to_run_config_dict()), preset.discretization, zeta=two_vortices,
        policy=ProjectionPolicy(max_relative_residual=0.5))
    assert prepared.report.fields[0].relative_residual > 0
    scale = np.abs(preset.coeffs).max()
    # CPU (NumPy) and GPU (CuPy) trigonometry differ in the last ulp only.
    np.testing.assert_allclose(prepared.state.coeffs, preset.coeffs,
                               rtol=0, atol=1e-12 * scale)


def test_williamson2_from_potentials_matches_the_preset_state():
    from tropoi.run.custom import initial_state_from_preset, prepare_swe_state

    preset = initial_state_from_preset("swe", {
        "scenario": "williamson2", "grid": "latlon", "nlat": NLAT,
        "nlon": NLON, "lmax": L})
    planet, fluid = preset.planet, preset.fluid
    a, omega, g = (planet.dynamical_radius_m, planet.angular_velocity_rad_s,
                   fluid.gravity_m_s2)
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    C = a * omega * u0 + 0.5 * u0 * u0

    prepared = prepare_swe_state(
        planet, preset.discretization, fluid,
        streamfunction=lambda lat, lon: -a * u0 * np.sin(lat),
        free_surface_anomaly=lambda lat, lon:
            (C / g) * (1.0 / 3.0 - np.sin(lat) ** 2))
    assert prepared.report.min_thickness_m > 0
    for k in range(3):                 # zeta, delta (zero), phi
        scale = max(float(np.abs(preset.coeffs[k]).max()), 1e-300)
        np.testing.assert_allclose(prepared.state.coeffs[k], preset.coeffs[k],
                                   rtol=0, atol=1e-12 * scale)


def test_lake_at_rest_over_a_mountain_matches_the_preset_trajectory(tmp_path):
    from tropoi.cli.swe import execute_run
    from tropoi.representation.archive import resolve_run_directory
    from tropoi.run.custom import (SpectralCoefficients, TopographySpec,
                                   initial_state_from_preset,
                                   prepare_swe_state, run_initial_state)
    from tropoi.run.swe.config import SWERunConfig

    settings = {"scenario": "rest", "grid": "latlon", "nlat": NLAT,
                "nlon": NLON, "lmax": L, "topography": "mountain",
                "mountain_width_deg": 30.0, "mean_depth_m": 4000.0}
    preset_state = initial_state_from_preset("swe", settings)
    zero = np.zeros((L + 1, L + 1), dtype=np.complex128)
    prepared = prepare_swe_state(
        preset_state.planet, preset_state.discretization, preset_state.fluid,
        TopographySpec.mountain(height_m=2000.0, lat_deg=30.0, lon_deg=90.0,
                                width_deg=30.0),
        free_surface_anomaly=SpectralCoefficients(zero))
    # Same terrain projection and the same phi = -phi_s' construction: equal
    # values (the preset's negation leaves -0.0 where this leaves +0.0).
    np.testing.assert_array_equal(prepared.state.coeffs, preset_state.coeffs)
    np.testing.assert_array_equal(prepared.state.topography_lm,
                                  preset_state.topography_lm)

    cfg = SWERunConfig.resolve({**settings, "duration_days": 0.05,
                                "n_snapshots": 2,
                                "out": str(tmp_path / "preset")})
    assert execute_run(cfg) == 0
    preset_run = resolve_run_directory(tmp_path / "preset")
    custom_run = run_initial_state(prepared.state, duration_days=0.05,
                                   n_snapshots=2, out=str(tmp_path / "c"))
    expected = np.load(preset_run / "swe_coeffs.npy")
    np.testing.assert_allclose(np.load(custom_run / "swe_coeffs.npy"),
                               expected, rtol=0,
                               atol=1e-12 * np.abs(expected).max())


def test_negative_thickness_fails_preparation_with_its_margin():
    from tropoi.run.custom import (FluidLayerSpec, PlanetSpec,
                                   PreparationError, TopographySpec,
                                   prepare_swe_state)

    with pytest.raises(PreparationError, match="not strictly positive") as e:
        prepare_swe_state(
            PlanetSpec.earth_like(), _disc(), FluidLayerSpec(9.80616, 1500.0),
            TopographySpec.mountain(height_m=2000.0, lat_deg=30.0,
                                    lon_deg=90.0, width_deg=30.0),
            free_surface_anomaly=lambda lat, lon: np.zeros(lat.shape))
    assert e.value.report.min_thickness_m < 0


def test_sampled_mean_is_reported_and_only_removed_on_request():
    from tropoi.run.custom import (FluidLayerSpec, PlanetSpec,
                                   PreparationError, ProjectionPolicy,
                                   prepare_swe_state)

    args = (PlanetSpec.earth_like(), _disc(), FluidLayerSpec(9.8, 1000.0))
    bump = lambda lat, lon: 98.0 + 50.0 * np.cos(lat) ** 2 * np.cos(2 * lon)
    with pytest.raises(PreparationError, match="mean_depth_m") as info:
        prepare_swe_state(*args, phi=bump)
    assert info.value.report.fields[2].mean == pytest.approx(98.0, rel=1e-12)
    prepared = prepare_swe_state(
        *args, phi=bump, policy=ProjectionPolicy(subtract_mean=("phi",)))
    assert prepared.state.coeffs[2, 0, 0] == 0.0
    assert prepared.report.fields[2].subtracted_mean == pytest.approx(98.0)
