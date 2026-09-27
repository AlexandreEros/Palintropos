"""TimeStepper protocol and the RK4 reference stepper (CPU-only, NumPy).

The stepper owns *how* one step advances a coefficient array; the scheduler
owns *when*. ``RK4Stepper`` must reproduce ``rk4_step_array`` bit for bit so
existing BVE/SWE/PE runs are unchanged when they adopt it.
"""
import numpy as np
import pytest

from tropoi.temporal.integration import rk4_step_array


def _decay(y):
    return -0.7 * y


def test_rk4_stepper_matches_rk4_step_array_bitwise():
    from tropoi.temporal.steppers import RK4Stepper
    y0 = np.arange(6, dtype=np.float64).reshape(2, 3) + 0.25
    st = RK4Stepper(_decay, dt=0.1)
    st.initialize(y0, t0=0.0)
    st.step()
    st.step()
    ref = rk4_step_array(_decay, rk4_step_array(_decay, y0, 0.0, 0.1), 0.1, 0.1)
    assert st.state.tobytes() == ref.tobytes()
    assert st.t == 0.2 and st.step_count == 2


def test_rk4_stepper_state_dict_round_trip_is_bit_identical():
    from tropoi.temporal.steppers import RK4Stepper
    y0 = np.array([1.0, -2.0, 0.5])
    a = RK4Stepper(_decay, dt=0.05)
    a.initialize(y0)
    a.step()
    saved = a.state_dict()
    b = RK4Stepper(_decay, dt=0.05)
    b.load_state_dict(saved)
    a.step(); b.step()
    assert a.state.tobytes() == b.state.tobytes()
    assert a.t == b.t and a.step_count == b.step_count


def test_rk4_stepper_is_fourth_order():
    from tropoi.temporal.steppers import RK4Stepper
    exact = np.exp(-0.7 * 1.0)
    errs = []
    for n in (10, 20, 40):
        st = RK4Stepper(_decay, dt=1.0 / n)
        st.initialize(np.array([1.0]))
        for _ in range(n):
            st.step()
        errs.append(abs(st.state[0] - exact))
    p1 = np.log2(errs[0] / errs[1])
    p2 = np.log2(errs[1] / errs[2])
    assert 3.8 < p1 < 4.2 and 3.8 < p2 < 4.2


def test_rk4_stepper_rejects_step_before_initialize():
    from tropoi.temporal.steppers import RK4Stepper
    st = RK4Stepper(_decay, dt=0.1)
    with pytest.raises(RuntimeError):
        st.step()


def test_rk4_stepper_forwards_stage_validator():
    from tropoi.temporal.steppers import RK4Stepper
    seen = []
    st = RK4Stepper(_decay, dt=0.1, stage_validator=lambda y: seen.append(y.copy()))
    st.initialize(np.array([1.0]))
    st.step()
    assert len(seen) == 3
