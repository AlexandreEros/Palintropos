"""Migration gates of the native (C++) substrate step against the Python
reference implementation, plus the native library's guards.

* GPU: the native kernel must agree with the reference on CuPy **bit for
  bit** in the state and every StepReport field (same arithmetic, same
  reductions; audit N1).
* CPU: the native library must agree with the reference on NumPy within the
  existing G-GPU criteria (|dT| <= 1e-6 K and identical Newton counts) and
  close the energy identity. Bit-for-bit agreement depends on the host's
  math library and compiler (it holds with clang on the development
  machine, not with every g++/glibc pair) and is reported, not required.

These are comparisons of two implementations; the independent verification
of the shared equations is in test_substrate.py and
test_substrate_nonlinear.py (analytical, manufactured, finite-difference
Jacobian and conservation tests).
"""
import re

import numpy as np
import pytest

from tropoi.physics.materials import DepthProfile, Stratum, ledlow_1992
from tropoi.physics.substrate import ConductiveSubstrate
from tropoi.spatial.vertical import stratified_interfaces
from tropoi.temporal import substrate_native
from tropoi.temporal._substrate_reference import ReferenceBackwardEulerSubstrate
from tropoi.temporal.native import build_identity
from tropoi.temporal.substrate import BackwardEulerSubstrate, SubstrateSolveError

#: the existing G-GPU bound (docs/validation/PLANET-ONE-GATES.md, A2)
G_GPU_K = 1e-6
FIELDS = ("energy_tendency", "absorbed", "emitted", "conducted", "bottom", "source",
          "residual_sum", "tolerance_sum", "max_residual_ratio")


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


def _run_both(sub, xp, steps=12, n=37, seed=5, dt=3600.0):
    rng = np.random.default_rng(seed)
    a = sub.initial_state(n, xp=xp)
    b = sub.initial_state(n, xp=xp)
    native, ref = BackwardEulerSubstrate(sub), ReferenceBackwardEulerSubstrate(sub)
    for kw in _forcings(rng, n, sub.layers, steps):
        kw = {k: (xp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
        a, ra = native.step(a, dt, **kw)
        b, rb = ref.step(b, dt, **kw)
        yield a, ra, b, rb


def _host(v):
    return v.get() if hasattr(v, "get") else np.asarray(v)


def test_cpu_native_matches_the_reference_within_the_g_gpu_criteria():
    sub = _three_strata()
    bitwise = True
    for a, ra, b, rb in _run_both(sub, np):
        assert ra.iterations == rb.iterations
        assert np.max(np.abs(a.anomaly - b.anomaly)) <= G_GPU_K
        assert np.max(np.abs(a.surface_anomaly - b.surface_anomaly)) <= G_GPU_K
        assert np.all(np.abs(ra.residual_sum) <= ra.tolerance_sum)      # energy identity
        bitwise &= (np.array_equal(a.anomaly, b.anomaly)
                    and np.array_equal(a.surface_anomaly, b.surface_anomaly)
                    and all(np.array_equal(getattr(ra, f), getattr(rb, f)) for f in FIELDS))
    print("CPU native vs reference bitwise:", bitwise)


def test_gpu_native_matches_the_reference_bit_for_bit():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import build_substrate
    for sub in (_three_strata(), build_substrate(mercury_vpw99())):
        for a, ra, b, rb in _run_both(sub, cp, steps=10, n=129):
            assert ra.iterations == rb.iterations
            assert bool(cp.array_equal(a.anomaly, b.anomaly))
            assert bool(cp.array_equal(a.surface_anomaly, b.surface_anomaly))
            for f in FIELDS:
                assert bool(cp.array_equal(getattr(ra, f), getattr(rb, f))), f
            assert [tuple(map(float, e)) for e in ra.extent] == \
                   [tuple(map(float, e)) for e in rb.extent]


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_convergence_failure_is_raised_as_by_the_reference(backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception:
            pytest.skip("no usable CUDA device")
        xp = cp
    else:
        xp = np
    sub = _three_strata()
    sub.max_iterations = 1
    st = sub.initial_state(5, xp=xp)
    sw = xp.asarray([9000.0, 0.0, 9000.0, 0.0, 9000.0])
    messages = []
    for impl in (BackwardEulerSubstrate(sub), ReferenceBackwardEulerSubstrate(sub)):
        with pytest.raises(SubstrateSolveError) as exc:
            impl.step(st, 86400.0, shortwave_down=sw)
        messages.append(str(exc.value))
    assert re.match(r"Newton did not converge in 1 iterations for \d+ columns", messages[0])
    if backend == "cupy":
        assert messages[0] == messages[1]
    else:                                  # the worst ratio's last digits may differ on a host
        assert messages[0].split("; worst")[0] == messages[1].split("; worst")[0]


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
    if name == "numpy":
        return np
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    return cp


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
def test_invalid_forcing_is_refused_as_by_the_reference(backend, bad, message):
    """The native step validates the forcing itself; the refusal (type and
    message, in the documented order) is the reference's, and no state is
    returned."""
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
    for impl in (BackwardEulerSubstrate(sub), ReferenceBackwardEulerSubstrate(sub)):
        with pytest.raises(ValueError, match=message):
            impl.step(st, 3600.0, **kw)


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
