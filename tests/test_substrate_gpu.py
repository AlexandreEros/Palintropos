"""G-GPU (docs/validation/PLANET-ONE-GATES.md, amendment A2): CPU/GPU parity
of the one conduction implementation. Skipped without CuPy and a GPU."""
from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:                                   # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

from tropoi.temporal.substrate import BackwardEulerSubstrate
from tropoi.physics.insolation import insolation_from_spec
from tropoi.physics.surface import SlabSurface
from tropoi.physics.substrate import ConductiveSubstrate
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import build_substrate

COLS = ([0.0, 0.0, 0.0, 0.0, 45.0, 85.0], [0.0, 90.0, 180.0, 270.0, 0.0, 0.0])


def _mercury_cycle(xp, steps=4320, cycles=1):
    cfg = mercury_vpw99(model="TWO", geothermal_flux=0.02,
                        columns={"kind": "points", "lat_deg": COLS[0], "lon_deg": COLS[1]})
    sub = build_substrate(cfg)
    ins = insolation_from_spec(cfg.illumination)
    lat, lon = xp.asarray(np.radians(COLS[0])), xp.asarray(np.radians(COLS[1]))
    n = (xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat))
    st = sub.initial_state(len(COLS[0]), xp=xp)
    dt = cfg.cycle_s / steps
    out, iters = [], []
    for k in range(steps * cycles):
        sw, _ = ins.incident(*n, (k + 1) * dt)
        st, rep = BackwardEulerSubstrate(sub).step(st, dt, shortwave_down=sw)
        out.append(cp.asnumpy(xp.concatenate([st.surface_anomaly[None], st.anomaly]))
                   if xp is cp else np.concatenate([st.surface_anomaly[None], st.anomaly]))
        iters.append(rep.iterations)
    return np.stack(out), np.array(iters)


def test_mercury_columns_one_cycle_parity():
    cpu, it_c = _mercury_cycle(np)
    gpu, it_g = _mercury_cycle(cp)
    diff = float(np.max(np.abs(gpu - cpu)))
    print("max |T_GPU - T_CPU| =", diff, "K; iteration-count differences:",
          int(np.sum(it_c != it_g)), "of", it_c.size)
    assert diff <= 1e-6


def test_energy_closure_on_the_gpu():
    rng = np.random.default_rng(5)
    cfg = mercury_vpw99(model="TWO")
    sub = build_substrate(cfg)
    st = sub.initial_state(3, xp=cp)
    E0 = sub.energy(st).copy()
    net_sum, bound = cp.zeros(3), cp.zeros(3)
    for _ in range(300):
        sw = cp.asarray(rng.uniform(0, 12000, 3) * (rng.uniform(size=3) > 0.4))
        st, rep = BackwardEulerSubstrate(sub).step(st, 900.0, shortwave_down=sw, bottom_heat_flux=0.02)
        net = rep.absorbed - rep.emitted + rep.bottom + rep.source
        assert bool(cp.all(cp.abs(rep.energy_tendency - net - rep.residual_sum)
                           <= 1e-12 * (cp.abs(rep.energy_tendency) + cp.abs(net)) + 1e-12))
        assert bool(cp.all(cp.abs(rep.residual_sum) <= rep.tolerance_sum))
        net_sum += 900.0 * net
        bound += 900.0 * rep.tolerance_sum
    E = sub.energy(st)
    slack = 4 * np.finfo(float).eps * float(cp.max(cp.abs(E))) + 1e-6
    assert bool(cp.all(cp.abs(E - E0 - net_sum) <= bound + slack))


def test_g4_and_g5_on_the_gpu():
    sub = ConductiveSubstrate.constant(emissivity=0.0, interfaces=(0.0, 1.0), albedo=0.0, bottom_heat_flux=0.0, initial_temperature=300.0, thermal_inertia=2000.0, volumetric_heat_capacity=2.0e6)
    slab = SlabSurface(2.0e6, albedo=0.0, emissivity=1.0)
    rng = np.random.default_rng(2)
    st = sub.initial_state(1, xp=cp)
    T = np.array([300.0])
    for _ in range(1000):
        f = rng.uniform(-200, 200, size=1)
        st, _ = BackwardEulerSubstrate(sub).step(st, 600.0, flux=cp.asarray(f))
        T = slab.advance(T, f, 600.0)
    assert abs(float(sub.temperature(st)[0, 0]) - float(T[0])) < 1000 * np.spacing(300.0)
    from tropoi.spatial.vertical import geometric_interfaces
    sub = ConductiveSubstrate.constant(emissivity=0.0,
                                       interfaces=geometric_interfaces(0.02, 3.5, layers=30), albedo=0.0, bottom_heat_flux=0.0, initial_temperature=300.0, thermal_inertia=2000.0, volumetric_heat_capacity=2.0e6)
    fb = cp.asarray([0.0, 0.1, -0.1])
    st = sub.initial_state(3, xp=cp)
    inflow, denom = np.zeros(3), np.zeros(3)
    for _ in range(2000):
        f = cp.asarray(rng.uniform(-200, 200, size=3))
        st, rep = BackwardEulerSubstrate(sub).step(st, 600.0, flux=f, bottom_heat_flux=fb)
        a = cp.asnumpy(rep.conducted + fb)
        inflow += 600.0 * a
        denom += 600.0 * np.abs(a)
        assert np.all(np.abs(cp.asnumpy(sub.energy(st)) - inflow) <= 1e-12 * denom + 1e-30)


def test_manufactured_solution_errors_match_the_cpu():
    import test_substrate_nonlinear as m
    strata = m._two_strata(kc=(0.004, 0.02))
    for nu, nl, dt in ((8, 12, 500.0), (16, 24, 125.0)):
        edges = m._faces(m.ZI, m.DEPTH, nu, nl)
        cpu = m._manufactured_run(edges, dt, 4.0e4, strata)
        gpu = m._manufactured_run(edges, dt, 4.0e4, strata, xp=cp)
        e_c = np.max(np.abs(cpu[0] - cpu[1]))
        e_g = np.max(np.abs(gpu[0] - gpu[1]))
        assert e_g == pytest.approx(e_c, rel=1e-9)
        assert abs(gpu[2] - gpu[3]) == pytest.approx(abs(cpu[2] - cpu[3]), rel=1e-9)
