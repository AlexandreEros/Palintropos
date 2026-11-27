"""The native (C++) substrate step across its two backends, and the native
library's guards.

The step's equations are verified independently in test_substrate.py and
test_substrate_nonlinear.py (analytical, manufactured, finite-difference
Jacobian and conservation tests), and its Thomas elimination here against
numpy.linalg.solve. Comparisons of the CUDA and the CPU builds of the same
code check consistency across backends, not correctness: they agree
within the G-GPU criteria (|dT| <= 1e-6 K, identical Newton counts), and bit
for bit only where the host's math library rounds as libdevice does.

The bitwise migration gates against the former Python implementation
(audit N1) are recorded in docs/validation/consolidation-evidence/.
"""
import re

import numpy as np
import pytest

from tropoi.physics.materials import DepthProfile, Stratum, ledlow_1992
from tropoi.physics.substrate import ConductiveSubstrate
from tropoi.spatial.vertical import stratified_interfaces
from tropoi.temporal import substrate_native
from tropoi.temporal.native import build_identity
from tropoi.temporal.substrate import BackwardEulerSubstrate, SubstrateSolveError

#: the existing G-GPU bound (docs/validation/PLANET-ONE-GATES.md, A2)
G_GPU_K = 1e-6

def _three_strata():
    """Three strata of three law families: constant c_p, the Ledlow piecewise
    law with a radiative term, and a polynomial with depth profiles."""
    strata = [Stratum(0.02, 1300.0, 9.2e-4, ledlow_1992(), radiative_coefficient=1.48,
                      radiative_reference_temperature=350.0, name="top"),
              Stratum.volumetric(0.2, 1.6e6, 0.4, name="constant middle"),
              Stratum(0.8, DepthProfile(1500.0, 1900.0, 0.3), DepthProfile(2e-3, 6e-3, 0.3),
                      (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9),
                      radiative_coefficient=2.7, radiative_reference_temperature=350.0,
                      name="deep")]
    faces = stratified_interfaces([s.bottom for s in strata], 1e-3, 1.15)
    return ConductiveSubstrate(strata, interfaces=faces, albedo=0.12, emissivity=0.95,
                               bottom_heat_flux=0.02, initial_temperature=260.0)


def _forcings(rng, n, L, steps):
    """Randomized forcing over every input of step(), varying its form."""
    for k in range(steps):
        kw = {"shortwave_down": rng.uniform(0.0, 9000.0, n) * (rng.random(n) < 0.6),
              "longwave_down": rng.uniform(0.0, 40.0, n),
              "flux": rng.uniform(-30.0, 30.0, n), "flux_derivative": -rng.uniform(0.0, 4.0, n)}
        if k % 2:
            kw["flux_reference_temperature"] = rng.uniform(150.0, 500.0, n)
        if k % 3 == 1:
            kw["bottom_heat_flux"] = rng.uniform(0.0, 0.1, n)
        form = k % 4
        if form == 1:
            kw["source"] = rng.uniform(-0.5, 0.5, (L, n))
        elif form == 2:
            kw["source"] = rng.uniform(-0.5, 0.5, (L, 1))
        elif form == 3:
            kw["source"] = 0.05
        yield kw


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    return cp


def _host(v):
    return v.get() if hasattr(v, "get") else np.asarray(v)


def test_cuda_and_cpu_builds_agree_within_the_g_gpu_criteria():
    """The same randomized forcing (every input of step(), every form) on
    both backends, from the same state: the G-GPU criteria every step, and
    the energy identity on each."""
    cp = _cupy()
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_substrate
    for sub in (_three_strata(), build_substrate(mercury_vpw99())):
        rng = np.random.default_rng(5)
        n = 37
        a, b = sub.initial_state(n, xp=cp), sub.initial_state(n)
        stepper = BackwardEulerSubstrate(sub)
        bitwise = True
        for kw in _forcings(rng, n, sub.layers, 12):
            a, ra = stepper.step(a, 3600.0, **{k: (cp.asarray(v) if isinstance(v, np.ndarray)
                                                   else v) for k, v in kw.items()})
            b, rb = stepper.step(b, 3600.0, **kw)
            assert ra.iterations == rb.iterations
            assert np.max(np.abs(_host(a.anomaly) - b.anomaly)) <= G_GPU_K
            assert np.max(np.abs(_host(a.surface_anomaly) - b.surface_anomaly)) <= G_GPU_K
            for r in (ra, rb):
                assert np.all(np.abs(_host(r.residual_sum)) <= _host(r.tolerance_sum))
            bitwise &= bool(np.array_equal(_host(a.anomaly), b.anomaly))
        print(sub.layers, "layers: CUDA and CPU bitwise:", bitwise)


def test_native_thomas_matches_an_independent_solver():
    """pt::thomas against numpy.linalg.solve on diagonally dominant systems
    of 1 to the full capacity of rows; larger systems are refused."""
    rng = np.random.default_rng(11)
    cap = substrate_native.host_library().layer_capacity + 1
    for n in (1, 2, 3, 17, 52, cap):
        lower, upper = rng.uniform(-1, 1, n - 1), rng.uniform(-1, 1, n - 1)
        diag = (np.abs(np.r_[lower, 0.0]) + np.abs(np.r_[0.0, upper])
                + rng.uniform(0.5, 2.0, n)) * rng.choice([-1.0, 1.0], n)
        rhs = rng.uniform(-10, 10, n)
        A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
        x = substrate_native.thomas_host(lower, diag, upper, rhs)
        np.testing.assert_allclose(x, np.linalg.solve(A, rhs), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(A @ x, rhs, rtol=1e-12, atol=1e-11)
    with pytest.raises(ValueError, match="rows"):
        substrate_native.thomas_host(np.ones(cap), np.full(cap + 1, 4.0), np.ones(cap),
                                     np.ones(cap + 1))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_convergence_failure_raises_and_names_the_columns(backend):
    xp = np if backend == "numpy" else _cupy()
    sub = _three_strata()
    sub.max_iterations = 1
    st = sub.initial_state(5, xp=xp)
    sw = xp.asarray([9000.0, 0.0, 9000.0, 0.0, 9000.0])
    with pytest.raises(SubstrateSolveError) as exc:
        BackwardEulerSubstrate(sub).step(st, 86400.0, shortwave_down=sw)
    assert re.fullmatch(r"Newton did not converge in 1 iterations for \d+ columns "
                        r"\(first \[[\d, ]+\]\); worst residual/tolerance \d\.\d{3}e[+-]\d+",
                        str(exc.value)), str(exc.value)


def test_host_capacity_and_law_limits_are_enforced_never_truncated():
    lib = substrate_native.host_library()
    deep = ConductiveSubstrate([Stratum.volumetric(1.0, 2e6, 2.0)],
                               interfaces=np.linspace(0.0, 1.0, lib.layer_capacity + 2),
                               albedo=0.0, emissivity=1.0, bottom_heat_flux=0.0,
                               initial_temperature=300.0)
    with pytest.raises(ValueError, match="never truncated"):
        BackwardEulerSubstrate(deep).step(deep.initial_state(2), 600.0, flux=1.0)
    from tropoi.physics.materials import PiecewiseHeatCapacity, PolynomialHeatCapacity
    nested = PiecewiseHeatCapacity(PiecewiseHeatCapacity(PolynomialHeatCapacity((700.0,)),
                                                         PolynomialHeatCapacity((800.0,)), 200.0),
                                   PolynomialHeatCapacity((900.0,)), 400.0)
    sub = ConductiveSubstrate([Stratum(0.5, 1500.0, 1e-3, nested)],
                              interfaces=np.linspace(0.0, 0.5, 6), albedo=0.0, emissivity=1.0,
                              bottom_heat_flux=0.0, initial_temperature=300.0)
    with pytest.raises(ValueError, match="nested piecewise"):
        BackwardEulerSubstrate(sub).step(sub.initial_state(2), 600.0, flux=1.0)


def test_a_stale_or_missing_library_is_refused(monkeypatch):
    substrate_native.host_library()                    # the real one loads
    monkeypatch.setattr(substrate_native, "_HOST", None)
    monkeypatch.setattr(substrate_native, "_DIGESTS", {})
    monkeypatch.setattr(build_identity, "digest", lambda flags=None: "0" * 64)
    with pytest.raises(substrate_native.NativeLibraryError, match="other sources"):
        substrate_native.host_library()
    monkeypatch.undo()
    monkeypatch.setattr(substrate_native, "_HOST", None)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(substrate_native.NativeLibraryError, match="not built"):
        substrate_native.host_library()


def test_an_older_library_is_refused_before_its_newer_exports_are_resolved(monkeypatch):
    """A library of an earlier ABI (without, say, pt_numerics_thomas) is
    reported as stale with the rebuild instructions, not as a missing
    symbol; provenance then records it as unavailable."""
    class Older:
        def __init__(self):
            self.pt_substrate_abi_version = lambda: substrate_native.ABI_VERSION - 1
            self.pt_substrate_build_digest = lambda: b"0" * 64

    monkeypatch.setattr(substrate_native, "_HOST", None)
    monkeypatch.setattr(substrate_native.ctypes, "CDLL", lambda path: Older())
    with pytest.raises(substrate_native.NativeLibraryError, match="other sources"):
        substrate_native.host_library()
    d = substrate_native.describe(_three_strata(), "numpy")
    assert d["host_library"].startswith("unavailable")
    monkeypatch.setattr(substrate_native.ctypes, "CDLL", lambda path: object())
    with pytest.raises(substrate_native.NativeLibraryError, match="not a native"):
        substrate_native.host_library()


def test_native_include_direction_follows_the_layers():
    """numerics headers include only numerics; physics headers include
    numerics and physics; nothing includes upward."""
    allowed = {"numerics": ("pt/portable.h", "thomas.h", "../portable.h"),
               "physics": ("pt/portable.h", "materials.h")}
    for layer, ok in allowed.items():
        root = build_identity.TROPOI / layer / "native"
        for h in root.rglob("*.h"):
            for inc in re.findall(r'#include\s+"([^"]+)"', h.read_text()):
                assert inc in ok, (h, inc)
    # the temporal composition includes physics only through its template parameter
    be = (build_identity.TROPOI / "temporal" / "native" / "pt" / "temporal" /
          "backward_euler.h").read_text()
    assert re.findall(r'#include\s+"([^"]+)"', be) == ["pt/portable.h"]


def test_provenance_describes_the_native_build():
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_substrate
    d = BackwardEulerSubstrate(build_substrate(mercury_vpw99())).describe("numpy")
    assert d["source_digest_host"] == build_identity.digest(build_identity.HOST_FLAGS)
    assert "-ffp-contract=off" in d["host_build"]
    assert d["host_layer_capacity"] >= 51


def _backend(name):
    return np if name == "numpy" else _cupy()


BAD_FORCING = [
    ({"shortwave_down": np.inf}, "shortwave_down must be finite"),
    ({"longwave_down": "column-nan"}, "longwave_down must be finite"),
    ({"flux": -np.inf}, "flux must be finite"),
    ({"flux_derivative": np.nan}, "flux_derivative must be finite"),
    ({"flux_derivative": 0.5}, r"flux_derivative must be finite and <= 0"),
    ({"bottom_heat_flux": "column-nan"}, "bottom_heat_flux must be finite"),
    ({"source": "cell-nan"}, "source must be finite"),
    ({"flux_reference_temperature": "column-inf"}, "flux_reference_temperature must be finite"),
    ({"shortwave_down": np.nan, "flux_derivative": 1.0}, "shortwave_down must be finite"),
]


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("bad, message", BAD_FORCING)
def test_invalid_forcing_is_refused_in_the_documented_order(backend, bad, message):
    """The native step validates the forcing itself; the refusal names the
    first invalid input in the documented order, and no state is returned."""
    xp = _backend(backend)
    sub = _three_strata()
    n = 6
    kw = {"shortwave_down": xp.full(n, 500.0), "flux_reference_temperature": xp.full(n, 280.0)}
    for k, v in bad.items():
        if v == "column-nan" or v == "column-inf":
            arr = np.full(n, 1.0)
            arr[3] = np.nan if v == "column-nan" else np.inf
            kw[k] = xp.asarray(arr)
        elif v == "cell-nan":
            arr = np.zeros((sub.layers, n))
            arr[sub.layers - 1, 4] = np.nan
            kw[k] = xp.asarray(arr)
        else:
            kw[k] = v
    st = sub.initial_state(n, xp=xp)
    with pytest.raises(ValueError, match=message):
        BackwardEulerSubstrate(sub).step(st, 3600.0, **kw)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_extent_bounds_are_the_extent(backend):
    xp = _backend(backend)
    sub = _three_strata()
    st, rep = BackwardEulerSubstrate(sub).step(sub.initial_state(9, xp=xp), 3600.0,
                                               shortwave_down=xp.linspace(0.0, 8000.0, 9))
    lows, highs = rep.extent_bounds
    assert [(float(a), float(b)) for a, b in rep.extent] == \
           list(zip(_host(lows).tolist(), _host(highs).tolist()))
    assert [tuple(map(float, e)) for e in rep.extent] == sub.extent(st)
