"""Initial-state preparation interfaces on the CPU.

Coefficient inputs need no transform, so BVE coefficient preparation and
every argument/policy check run here without CuPy. Projection of physical
fields, callables, terrain resolution and SWE positivity are GPU-gated in
tests/test_custom_preparation_gpu.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec, PlanetSpec,
                               PreparationError, ProjectionPolicy,
                               SampledField, SampledFunction,
                               SpectralCoefficients, load_initial_state,
                               pad_coefficients, prepare_bve_state,
                               prepare_swe_state)

L = 8
PLANET = PlanetSpec.ideal_sphere(4.0e6, 50000.0)
DISC = DiscretizationSpec.gauss_latlon(L, 12, 24)


def _zonal_jet(l_max=L, amplitude=1e-5):
    a = np.zeros((l_max + 1, l_max + 1), dtype=np.complex128)
    a[1, 0] = amplitude
    a[3, 0] = -0.3 * amplitude
    a[4, 2] = 0.1 * amplitude * (1 + 1j)
    return a


def test_bve_from_vorticity_coefficients_is_exact():
    zeta = _zonal_jet()
    prepared = prepare_bve_state(PLANET, DISC,
                                 zeta=SpectralCoefficients(zeta), label="jet")
    assert prepared.state.coeffs.tobytes() == zeta.tobytes()
    (entry,) = prepared.report.fields
    assert entry.source == "coefficients" and entry.relative_residual is None
    assert entry.mean == 0.0
    # Degree 8 > cut 5 holds no power here; the share is reported, not cut.
    assert entry.power_above_product_cut == 0.0
    provenance = prepared.state.provenance["preparation"]
    json.dumps(provenance, allow_nan=False)
    assert provenance["policy"]["max_relative_residual"] == 1e-2
    assert "not necessarily balanced" in provenance["balanced"]


def test_bve_coefficient_preparation_does_not_import_cupy():
    # Other GPU tests may already have imported CuPy in this pytest process.
    # A fresh interpreter checks imports and preparation without changing the
    # parent's module cache or requiring CuPy to be installed.
    code = textwrap.dedent("""
        import json
        import runpy
        import sys

        sys.path[:] = json.loads(sys.argv[2])
        assert "cupy" not in sys.modules
        namespace = runpy.run_path(sys.argv[1])
        namespace["test_bve_from_vorticity_coefficients_is_exact"]()
        assert "cupy" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), json.dumps(sys.path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_power_above_the_product_cut_is_reported_never_removed():
    zeta = _zonal_jet()
    zeta[L, 3] = 2e-6 + 1e-6j          # above the 2/3 cut (5 for L=8)
    prepared = prepare_bve_state(PLANET, DISC,
                                 zeta=SpectralCoefficients(zeta))
    assert prepared.state.coeffs[L, 3] == zeta[L, 3]
    assert prepared.report.fields[0].power_above_product_cut > 0.0


def test_bve_from_streamfunction_coefficients_uses_the_exact_laplacian():
    psi = np.zeros((L + 1, L + 1), dtype=np.complex128)
    psi[0, 0] = 1e9                    # an inert mean: allowed for psi
    psi[2, 1] = 3e6 - 1e6j
    a = PLANET.dynamical_radius_m
    state = prepare_bve_state(PLANET, DISC,
                              streamfunction=SpectralCoefficients(psi)).state
    assert state.coeffs[0, 0] == 0.0
    assert state.coeffs[2, 1] == -6.0 / a**2 * psi[2, 1]


def test_mean_modes_are_rejected_unless_subtracted_explicitly():
    zeta = _zonal_jet()
    zeta[0, 0] = 1e-7
    with pytest.raises(PreparationError, match="global mean") as info:
        prepare_bve_state(PLANET, DISC, zeta=SpectralCoefficients(zeta))
    assert info.value.report.fields[0].mean == pytest.approx(
        1e-7 / math.sqrt(4 * math.pi))
    prepared = prepare_bve_state(
        PLANET, DISC, zeta=SpectralCoefficients(zeta),
        policy=ProjectionPolicy(subtract_mean=("zeta",)))
    assert prepared.state.coeffs[0, 0] == 0.0
    assert prepared.report.fields[0].subtracted_mean == pytest.approx(
        1e-7 / math.sqrt(4 * math.pi))


@pytest.mark.parametrize("edit,match", [
    (lambda a: a.__setitem__((2, 5), 1e-6), "padding"),
    (lambda a: a.__setitem__((3, 0), 1e-6 + 1e-9j), "Im"),
    (lambda a: a.__setitem__((3, 1), np.inf), "non-finite"),
])
def test_coefficient_conventions_are_checked_not_repaired(edit, match):
    zeta = _zonal_jet()
    edit(zeta)
    with pytest.raises(PreparationError, match=match):
        prepare_bve_state(PLANET, DISC, zeta=SpectralCoefficients(zeta))


def test_coefficient_truncation_is_never_implicit():
    with pytest.raises(PreparationError, match="pad_coefficients"):
        prepare_bve_state(PLANET, DISC,
                          zeta=SpectralCoefficients(_zonal_jet(l_max=5)))
    padded = pad_coefficients(_zonal_jet(l_max=5), L)
    assert padded.shape == (L + 1, L + 1)
    assert prepare_bve_state(PLANET, DISC, zeta=SpectralCoefficients(
        padded)).state.coeffs[4, 2] == _zonal_jet(l_max=5)[4, 2]
    with pytest.raises(PreparationError, match="truncation"):
        pad_coefficients(_zonal_jet(l_max=10), L)
    with pytest.raises(PreparationError, match="l_max=5"):
        prepare_swe_state(PLANET, DISC, FluidLayerSpec(9.8, 1000.0),
                          phi=SpectralCoefficients(_zonal_jet(l_max=5)))


def test_sources_are_explicit_about_physical_versus_spectral():
    with pytest.raises(PreparationError, match="SampledField"):
        SpectralCoefficients(np.zeros((L + 1, L + 1)))      # real => physical
    with pytest.raises(PreparationError, match="SpectralCoefficients"):
        SampledField(np.zeros((3, 3), dtype=np.complex128))
    with pytest.raises(PreparationError, match="wrap the input"):
        prepare_bve_state(PLANET, DISC, zeta=np.zeros((12, 24)))
    assert SampledFunction(np.sin).name == "sin"
    assert SpectralCoefficients(
        np.zeros((3, 3), dtype=np.complex64)).coeffs.dtype == np.complex128


@pytest.mark.parametrize("call,match", [
    (lambda: prepare_bve_state(PLANET, DISC), "zeta or streamfunction"),
    (lambda: prepare_bve_state(
        PLANET, DISC, zeta=SpectralCoefficients(_zonal_jet()),
        streamfunction=SpectralCoefficients(_zonal_jet())), "not both"),
    (lambda: prepare_bve_state({"radius": 1}, DISC), "PlanetSpec"),
    (lambda: prepare_swe_state(PLANET, DISC, None), "FluidLayerSpec"),
    (lambda: prepare_swe_state(
        PLANET, DiscretizationSpec(lmax=L, product_quadrature="coarse"),
        FluidLayerSpec(9.8, 1000.0)), "fine"),
    (lambda: prepare_swe_state(
        PLANET, DISC, FluidLayerSpec(9.8, 1000.0),
        phi=SpectralCoefficients(_zonal_jet()),
        free_surface_anomaly=SpectralCoefficients(_zonal_jet())), "not both"),
    (lambda: ProjectionPolicy(max_relative_residual=-1), "residual"),
    (lambda: ProjectionPolicy(subtract_mean=("streamfunction",)),
     "subtract_mean"),
])
def test_argument_errors_fail_before_cuda(call, match):
    with pytest.raises(PreparationError, match=match):
        call()


def test_prepared_bve_state_saves_and_reloads(tmp_path):
    prepared = prepare_bve_state(PLANET, DISC,
                                 zeta=SpectralCoefficients(_zonal_jet()),
                                 label="jet")
    loaded = load_initial_state(prepared.save(tmp_path / "jet"))
    assert loaded.digest == prepared.state.digest
    assert loaded.provenance["preparation"]["fields"][0]["name"] == "zeta"
