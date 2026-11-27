"""The airless runner's fused per-step accumulation (one kernel on CuPy)
against its reference form, bit for bit (audit N1)."""
import numpy as np
import pytest

from tropoi.run.airless.experiment import SERIES, accumulate, accumulate_reference
from tropoi.temporal.substrate import StepReport


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    return cp


def _report(xp, rng, n):
    v = lambda lo, hi: xp.asarray(rng.uniform(lo, hi, n))                   # noqa: E731
    return StepReport(3, v(-50, 50), v(0, 9000), v(0, 3000), v(-400, 400), v(0, 0.1),
                      v(-1, 1), v(-1e-9, 1e-9), v(0, 1e-8), v(0, 1), ())


@pytest.mark.parametrize("per_point", [False, True])
def test_fused_accumulation_matches_the_reference_bit_for_bit(per_point):
    cp = _cupy()
    rng = np.random.default_rng(3)
    n, steps, dt = 257, 7, 1234.5678
    keys = ("absorbed", "emitted", "conducted", "bottom", "residual", "tolerance",
            "abs_conducted")
    acc = [({k: cp.asarray(rng.uniform(-1e6, 1e6, n)) for k in keys}, cp.zeros(n),
            {k: cp.zeros((steps, n)) for k in SERIES}) for _ in range(2)]
    acc[1] = ({k: v.copy() for k, v in acc[0][0].items()}, acc[0][1].copy(),
              {k: v.copy() for k, v in acc[0][2].items()})
    for step in range(steps):
        rep = _report(cp, rng, n)
        got = []
        for fn, (sums, ratio, series) in zip((accumulate, accumulate_reference), acc):
            rows = {k: series[k][step] for k in SERIES} if per_point else None
            got.append(fn(cp, dt, rep, sums, ratio, rows))
        for k in SERIES:
            assert bool(cp.array_equal(got[0][k], got[1][k])), k
    (s0, r0, g0), (s1, r1, g1) = acc
    for k in keys:
        assert bool(cp.array_equal(s0[k], s1[k])), k
    assert bool(cp.array_equal(r0, r1))
    if per_point:
        for k in SERIES:
            assert bool(cp.array_equal(g0[k], g1[k])), k


def test_the_numpy_accumulation_is_the_reference():
    rng = np.random.default_rng(4)
    rep = _report(np, rng, 5)
    keys = ("absorbed", "emitted", "conducted", "bottom", "residual", "tolerance",
            "abs_conducted")
    a = {k: np.zeros(5) for k in keys}
    b = {k: np.zeros(5) for k in keys}
    ra, rb = np.zeros(5), np.zeros(5)
    va = accumulate(np, 60.0, rep, a, ra)
    vb = accumulate_reference(np, 60.0, rep, b, rb)
    assert all(np.array_equal(a[k], b[k]) for k in keys) and np.array_equal(ra, rb)
    assert all(np.array_equal(va[k], vb[k]) for k in SERIES)
