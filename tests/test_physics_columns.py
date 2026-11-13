"""Column physics on the CPU: thermodynamics, gray schemes against analytic
solutions and published closed forms, insolation, convection, energy
closure and radiative(-convective) equilibrium."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.physics.column_model import ColumnPhysics, RadiativeConvectiveColumn
from tropoi.physics.convection import DryConvectiveAdjustment, neutral_profile
from tropoi.physics.composition import COMPOSITIONS, Composition, CompositionError
from tropoi.physics.insolation import (DiurnalMeanInsolation, InstantaneousInsolation,
                                       PrescribedInsolation, StellarIllumination,
                                       TidallyLockedInsolation,
                                       frierson_net_surface_flux)
from tropoi.physics.planets import PLANETS
from tropoi.physics.radiation.gray import (GrayLongwave, GrayShortwave, OpticalDepthProfile,
                                           SemiInfiniteScatteringShortwave, frierson_longwave,
                                           isca_mars_longwave, isca_mars_shortwave,
                                           schneider_liu_longwave)
from tropoi.physics.radiation.scheme import (STEFAN_BOLTZMANN, Column, LowerBoundary,
                                             Radiation)
from tropoi.physics.surface import SlabSurface, drag_coefficient, sensible_heat_flux
from tropoi.physics.thermodynamics import (R_UNIVERSAL, cp_molar, heat_capacity,
                                           heat_capacity_range)
from tropoi.spatial.sigma_coordinate import SigmaGrid

SB = STEFAN_BOLTZMANN


# ---------------------------------------------------------------- gases
def test_heat_capacities_match_reference_values_at_298K():
    reference = {"N2": 29.12, "O2": 29.38, "CO2": 37.13, "CH4": 35.70, "H2": 28.84,
                 "H2O": 33.59, "He": 20.786, "Ar": 20.786, "NH3": 35.65, "CO": 29.14}
    for name, value in reference.items():
        assert float(cp_molar(name, 298.15)) == pytest.approx(value, rel=6e-3), name


def test_mixture_constants_and_the_published_thai_heat_capacities():
    air = COMPOSITIONS["earth_dry_air"]
    assert air.gas_constant == pytest.approx(287.05, abs=0.1)
    assert float(heat_capacity(air, 300.0)) == pytest.approx(1004.6, rel=2e-3)
    # absorber amounts for gas optics are per kg of air: 400 ppmv CO2 = 608 ppm by mass
    assert air.mass_fraction("CO2") == pytest.approx(6.08e-4, rel=2e-3)
    assert air.mass_fraction("CH4") == 0.0
    # Turbet et al. (2022) quote ~1040 and ~850 J/kg/K for Ben 1 and Ben 2
    assert float(heat_capacity(COMPOSITIONS["thai_ben1"], 300.0)) == pytest.approx(1040.0, rel=5e-3)
    assert float(heat_capacity(COMPOSITIONS["thai_ben2"], 300.0)) == pytest.approx(850.0, rel=1e-2)
    with pytest.raises(CompositionError):
        Composition.normalized({"N2": 0.5}, label="incomplete", source="test")
    with pytest.raises(CompositionError):
        Composition.of(Xe=1.0)


def test_hydrogen_heat_capacity_depends_on_the_ortho_para_assumption():
    normal = float(cp_molar("H2", 60.0)) / R_UNIVERSAL
    equilibrium = float(cp_molar("H2", 60.0, ortho_para="equilibrium")) / R_UNIVERSAL
    assert normal == pytest.approx(2.52, abs=0.02)        # rotation nearly frozen out
    assert equilibrium > normal + 1.5                     # conversion heat
    assert float(cp_molar("H2", 100.0)) / R_UNIVERSAL == pytest.approx(2.71, abs=0.02)
    assert float(cp_molar("H2", 600.0)) / R_UNIVERSAL == pytest.approx(3.5, abs=0.03)
    spread = heat_capacity_range(COMPOSITIONS["jupiter"], 105.0, 700.0)["relative_spread"]
    assert 0.15 < spread < 0.30      # constant c_p is a ~20 % statement on Jupiter


def test_every_planet_preset_is_internally_consistent():
    for body in PLANETS.values():
        reg = body.regime()
        assert 0.0 < body.kappa < 0.4
        assert reg["depth_over_radius"] < 0.2
        grid = SigmaGrid.stretched(body.vertical_grid.nlev, body.vertical_grid.sigma_top,
                                   body.vertical_grid.linear_scale)
        assert grid.nlev == body.vertical_grid.nlev
        assert body.has_surface or body.internal_heat_flux > 0.0


# ---------------------------------------------------------------- grids
def test_stretched_grid_is_logarithmic_aloft_and_bounded_near_the_surface():
    grid = SigmaGrid.stretched(40, 1e-5, 0.15)
    s = np.array(grid.interfaces)
    assert s[0] == 0.0 and s[1] == 1e-5 and s[-1] == 1.0
    ratio = s[2:] / s[1:-1]
    assert ratio[:5].std() / ratio[:5].mean() < 0.02       # constant ratio aloft
    assert np.diff(s).max() < 0.1                          # no thick bottom layer
    log_only = np.array(SigmaGrid.stretched(40, 1e-5).interfaces)
    assert np.allclose(log_only[2:] / log_only[1:-1], (1e5) ** (1 / 39), rtol=1e-9)
    # a uniform grid cannot see above p_s / (2 K)
    assert SigmaGrid.uniform(30).full_levels[0] == pytest.approx(1 / 60)
    fhz = SigmaGrid.frierson(25)
    assert fhz.nlev == 25 and fhz.interfaces[1] == pytest.approx(math.exp(-5 * (0.05 * 0.96 + 0.95 * 0.96 ** 3)))


# ---------------------------------------------------------------- radiation
def _column(K, ps=1.0e5, temperature=250.0, ncol=2, gravity=9.8, cp=1004.0, lat=0.0):
    p = np.linspace(0.0, ps, K + 1)[:, None] * np.ones((1, ncol))
    t = np.full((K, ncol), temperature) if np.isscalar(temperature) else temperature
    return Column(p, t, gravity, cp, latitude=np.full(ncol, lat))


def test_gray_radiative_equilibrium_is_reproduced_at_second_order():
    """Analytic gray equilibrium: sigma T^4 = OLR (1 + tau) / 2, surface
    sigma T_s^4 = OLR (1 + tau_s / 2). The net flux must then equal -OLR at
    every interface; the error falls as K^-2."""
    olr, tau_s = 240.0, 3.0
    lw = GrayLongwave(OpticalDepthProfile([(1.0, 1.0)], equator=tau_s))
    errors = []
    for K in (20, 40, 80):
        mid = tau_s * (np.arange(K) + 0.5) / K
        t = ((olr * (1 + mid) / 2 / SB) ** 0.25)[:, None] * np.ones((1, 2))
        ts_ = np.full(2, (olr * (1 + tau_s / 2) / SB) ** 0.25)
        f = Radiation(lw, source="isothermal").fluxes(_column(K, temperature=t),
                                                      LowerBoundary(temperature=ts_))
        errors.append(np.abs(-f.net_down - olr).max())
    assert errors[0] < 0.4
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05)


def test_heating_is_the_flux_divergence_and_closes_the_column_budget():
    rng = np.random.default_rng(5)
    K, ncol = 17, 6
    sig = np.array(SigmaGrid.stretched(K, 1e-3, 0.2).interfaces)
    p = sig[:, None] * rng.uniform(0.8e5, 1.1e5, ncol)[None, :]
    col = Column(p, rng.uniform(180, 310, (K, ncol)), 9.8, 1004.0,
                 latitude=rng.uniform(-1.5, 1.5, ncol))
    rad = Radiation(frierson_longwave(), GrayShortwave(
        OpticalDepthProfile([(1.0, 1.0)], equator=0.4), ssa=0.5, asym=0.3))
    f = rad.fluxes(col, LowerBoundary(temperature=rng.uniform(250, 310, ncol), albedo=0.2,
                                      emissivity=0.95),
                   toa_flux=rng.uniform(0, 400, ncol), mu0=rng.uniform(0.05, 1, ncol))
    column_heating = (f.heating * col.cp * col.layer_pressure_thickness / col.gravity).sum(axis=0)
    assert np.abs(column_heating - (f.toa_net_down - f.bottom_net_down)).max() < 1e-10
    assert (f.sw_down >= f.sw_down_direct - 1e-12).all()


def test_schneider_liu_solar_flux_is_the_semi_infinite_two_stream_solution():
    profile = OpticalDepthProfile([(1.0, 1.0)], equator=40.0, reference=3.0e5)
    analytic = SemiInfiniteScatteringShortwave(profile, ssa=0.8, asym=0.204)
    assert analytic.bond_albedo == pytest.approx(0.343, abs=5e-4)     # SL09 Table 1
    general = GrayShortwave(profile, ssa=0.8, asym=0.204, closure="hemispheric",
                            diffuse_incidence=True)
    col = _column(60, ps=3.0e5, temperature=150.0, gravity=26.0, cp=12619.0)
    a = Radiation(None, analytic).fluxes(col, LowerBoundary("closed"), toa_flux=np.full(2, 10.0))
    b = Radiation(None, general).fluxes(col, LowerBoundary("closed"), toa_flux=np.full(2, 10.0))
    assert np.abs(a.net_down - b.net_down).max() < 1e-12
    # SL09: less than 9 % of the incident flux reaches 3 bar (tau_s = 3)
    shallow = SemiInfiniteScatteringShortwave(
        OpticalDepthProfile([(1.0, 1.0)], equator=3.0, reference=3.0e5), ssa=0.8, asym=0.204)
    f = Radiation(None, shallow).fluxes(col, LowerBoundary("closed"), toa_flux=np.full(2, 1.0))
    assert 0.05 < f.net_down[-1, 0] < 0.09


def test_closed_bottom_returns_every_watt():
    col = _column(30, ps=3.0e5, temperature=np.linspace(110, 230, 30)[:, None] * np.ones((1, 2)),
                  gravity=26.0, cp=12619.0)
    rad = Radiation(schneider_liu_longwave(), SemiInfiniteScatteringShortwave(
        OpticalDepthProfile([(1.0, 1.0)], equator=3.0, reference=3.0e5), ssa=0.8, asym=0.204))
    f = rad.fluxes(col, LowerBoundary("closed"), toa_flux=np.full(2, 12.0))
    assert np.abs(f.bottom_net_down).max() < 1e-12
    assert f.sw_down[-1, 0] > 0.0


def test_published_gray_optical_depths():
    col = _column(10, lat=0.0)
    assert frierson_longwave().profile.cumulative(col)[-1, 0] == pytest.approx(6.0)
    pole = _column(10, lat=math.pi / 2)
    assert frierson_longwave().profile.cumulative(pole)[-1, 0] == pytest.approx(1.5)
    mid = frierson_longwave().profile.cumulative(col)[5, 0]
    assert mid == pytest.approx(6.0 * (0.1 * 0.5 + 0.9 * 0.5 ** 4))
    jup = _column(10, ps=3.0e5)
    assert schneider_liu_longwave().profile.cumulative(jup)[5, 0] == pytest.approx(80 * 0.25)
    assert isca_mars_longwave().profile.cumulative(col)[-1, 0] == pytest.approx(0.2)
    assert isca_mars_shortwave().profile.cumulative(col)[5, 0] == pytest.approx(0.2 / 16)


# ---------------------------------------------------------------- insolation
def test_insolation_geometries_integrate_to_the_right_global_means():
    lat = np.arcsin(np.linspace(-1, 1, 4001)[1::2])          # equal-area latitudes
    s0 = 1361.0
    equinox, mu = DiurnalMeanInsolation(StellarIllumination.prescribed(s0))(lat, 0 * lat, 0.0)
    assert np.allclose(equinox, s0 * np.cos(lat) / math.pi)          # SL09 form
    assert equinox.mean() == pytest.approx(s0 / 4, rel=1e-5)
    orbit = StellarIllumination.prescribed(s0, 100.0, 0.0, 25.0)
    for t in (0.0, 25.0, 60.0):          # any season: global mean stays S0 / 4
        flux, mu = DiurnalMeanInsolation(orbit)(lat, 0 * lat, t)
        assert flux.mean() == pytest.approx(s0 / 4, rel=1e-5)
        assert (mu >= 0).all() and (mu <= 1).all()
    lon = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    la, lo = np.meshgrid(lat[::40], lon, indexing="ij")
    inst, _ = InstantaneousInsolation(orbit, 10.0)(la, lo, 25.0)
    daily, _ = DiurnalMeanInsolation(orbit)(lat[::40], 0 * lat[::40], 25.0)
    assert np.allclose(inst.mean(axis=1), daily, atol=1e-3 * s0)
    locked, _ = TidallyLockedInsolation(900.0)(la, lo, 0.0)
    weights = np.ones_like(la) / la.size
    assert (locked * weights).sum() == pytest.approx(225.0, rel=5e-3)   # coarse test mesh
    ecc = StellarIllumination.prescribed(s0, 100.0, 0.3)
    mean_flux = np.mean([ecc.flux_at(t) for t in np.linspace(0, 100, 20001)[:-1]])
    assert mean_flux == pytest.approx(s0 / math.sqrt(1 - 0.09), rel=1e-4)
    fhz, _ = frierson_net_surface_flux()(lat, 0 * lat, 0.0)
    assert fhz.mean() == pytest.approx(938.4 / 4, rel=1e-6)


# ---------------------------------------------------------------- convection
def _pava(y, w):
    blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi * wi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], blocks[-1][2] + b[2]]
    return np.concatenate([[b[0] / b[1]] * b[2] for b in blocks])


def test_convective_adjustment_is_exact_neutral_and_conserves_enthalpy():
    rng = np.random.default_rng(1)
    K, n, kappa = 25, 60, 0.2857
    interior = np.sort(rng.uniform(0, 1e5, (K - 1, n)), axis=0)
    p = np.vstack([np.zeros((1, n)), interior, np.full((1, n), 1e5)])
    dp, mid = np.diff(p, axis=0), 0.5 * (p[1:] + p[:-1])
    t = rng.uniform(150, 320, (K, n))
    adj = neutral_profile(t, mid, dp, kappa)
    assert np.abs((adj * dp).sum(0) / (t * dp).sum(0) - 1).max() < 1e-13
    theta = adj / (mid / 1e5) ** kappa
    assert (theta[1:] - theta[:-1]).max() < 1e-9          # never increases downward
    for c in range(n):                                    # sequential reference
        ex = (mid[::-1, c] / mid[-1, c]) ** kappa
        ref = (_pava(t[::-1, c] / ex, ex * dp[::-1, c]) * ex)[::-1]
        assert np.abs(ref - adj[:, c]).max() < 1e-7
    stable = 300.0 * (mid / 1e5) ** (0.5 * kappa)
    assert np.abs(neutral_profile(stable, mid, dp, kappa) - stable).max() < 1e-7
    relaxed = DryConvectiveAdjustment(kappa, timescale=6 * 3600.0).adjust(t, mid, dp, 3600.0)
    frac = 1 - math.exp(-1 / 6)
    assert np.allclose(relaxed, t + frac * (adj - t))


# ---------------------------------------------------------------- surface
def test_bulk_exchange_follows_the_published_drag_law():
    # FHZ06: z0 = 3.21e-5 m gives C = 0.001 at 10 m for neutral or unstable air
    assert float(drag_coefficient(np.array(10.0), 3.21e-5, np.array(-1.0))) == pytest.approx(1e-3, rel=2e-3)
    assert float(drag_coefficient(np.array(10.0), 3.21e-5, np.array(0.5))) == pytest.approx(0.25e-3, rel=2e-3)
    assert float(drag_coefficient(np.array(10.0), 3.21e-5, np.array(2.0))) == 0.0
    up, _ = sensible_heat_flux(np.array(300.0), np.array(290.0), np.array(50.0), np.array(8.0),
                               np.array(1.2), 1004.0, 9.8)
    down, _ = sensible_heat_flux(np.array(280.0), np.array(290.0), np.array(50.0), np.array(8.0),
                                 np.array(1.2), 1004.0, 9.8)
    assert up > 0.0 and down <= 0.0 and abs(down) < up     # stable side is throttled
    assert float(SlabSurface(1e7).advance(np.array(280.0), np.array(100.0), 1e5)) == pytest.approx(281.0)


# ---------------------------------------------------------------- equilibria
def test_radiative_equilibrium_column_converges_to_the_analytic_solution():
    K, tau_s, flux = 40, 2.0, 240.0
    sig = np.array(SigmaGrid.uniform(K).interfaces)
    physics = ColumnPhysics(
        Radiation(GrayLongwave(OpticalDepthProfile([(1.0, 1.0)], equator=tau_s)),
                  GrayShortwave(OpticalDepthProfile([(1.0, 1.0)], equator=1e-12)),
                  source="isothermal"),
        gravity=9.8, cp=1004.0, gas_constant=287.0, surface=SlabSurface(1e6),
        insolation=PrescribedInsolation(lambda lat: flux + 0 * lat, "uniform", {}))
    model = RadiativeConvectiveColumn(physics, (sig * 1e5)[:, None], np.full((K, 1), 250.0),
                                      latitude=np.zeros(1), longitude=np.zeros(1),
                                      surface_temperature=np.full(1, 280.0))
    result = model.equilibrate(tolerance=1e-3)
    assert result["converged"]
    mid = tau_s * 0.5 * (sig[1:] + sig[:-1])
    analytic = (flux * (1 + mid) / 2 / SB) ** 0.25
    assert np.abs(model.temperature[:, 0] - analytic).max() < 0.2
    assert model.surface_temperature[0] == pytest.approx((flux * (1 + tau_s / 2) / SB) ** 0.25, abs=0.05)
    assert model.evaluate().fluxes.olr[0] == pytest.approx(flux, abs=0.05)


def test_schneider_liu_column_reproduces_the_published_energy_balance_and_structure():
    """Radiative-convective equilibrium with the SL09 radiation, convection
    and interior heat (their Table 1 constants, 30 uniform sigma levels to
    3 bar). Published: energy uptake 14.0 W m^-2, statically stable above
    ~0.5 bar, tropopause near 0.15 bar."""
    K, g, r, cp = 30, 26.0, 3605.38, 12619.0
    p = (np.array(SigmaGrid.uniform(K).interfaces) * 3.0e5)[:, None]
    rad = Radiation(schneider_liu_longwave(), SemiInfiniteScatteringShortwave(
        OpticalDepthProfile([(1.0, 1.0)], equator=3.0, reference=3.0e5), ssa=0.8, asym=0.204))
    physics = ColumnPhysics(rad, gravity=g, cp=cp, gas_constant=r, internal_heat_flux=5.7,
                            convection=DryConvectiveAdjustment(r / cp),
                            insolation=PrescribedInsolation(lambda lat: 50.7 / 4 + 0 * lat,
                                                            "global_mean", {}))
    model = RadiativeConvectiveColumn(physics, p, np.full((K, 1), 150.0),
                                      latitude=np.zeros(1), longitude=np.zeros(1))
    assert model.equilibrate(tolerance=5e-4)["converged"]
    out = model.evaluate()
    assert out.fluxes.olr[0] == pytest.approx(14.0, abs=0.05)
    # the tolerance is per layer; 30 layers may hold 30 x 5e-4 W m^-2 between them
    assert out.fluxes.absorbed_stellar[0] + 5.7 == pytest.approx(out.fluxes.olr[0], abs=0.02)
    mid = 0.5 * (p[1:, 0] + p[:-1, 0])
    theta = model.temperature[:, 0] * (3.0e5 / mid) ** (r / cp)
    convecting = mid[np.abs(theta - theta[-1]) < 1e-6]
    assert 0.3e5 < convecting.min() < 0.6e5               # neutral below ~0.35-0.45 bar
    assert mid[np.argmin(model.temperature[:, 0])] == pytest.approx(0.15e5)
    t_1bar = np.interp(1.0e5, mid, model.temperature[:, 0])
    assert 160.0 < t_1bar < 175.0                         # Jupiter: ~165 K at 1 bar


def test_held_fluxes_heat_the_current_layer_masses():
    """With ``radiation_interval > 1`` the interface fluxes are held. If the
    surface pressure changes meanwhile, the heating must still integrate to
    the held boundary fluxes over the *new* layer masses."""
    from tropoi.physics.column_model import ColumnPhysics
    from tropoi.physics.radiation.gray import schneider_liu_longwave
    gravity, cp = 9.8, 1004.0
    sigma = np.linspace(0.0, 1.0, 9)[:, None]
    temperature = np.linspace(210.0, 285.0, 8)[:, None] * np.ones((1, 2))

    def physics(interval):
        radiation = Radiation(schneider_liu_longwave(tau0=3.0, p0=1.0e5))
        return ColumnPhysics(radiation, gravity=gravity, cp=cp, gas_constant=287.0,
                             internal_heat_flux=4.0, radiation_interval=interval)

    held = physics(4)
    p0 = sigma * np.array([[1.0e5, 0.9e5]])
    p1 = sigma * np.array([[1.01e5, 0.93e5]])            # the dynamics moved mass
    first = held.compute(p0, temperature, 0.0)
    second = held.compute(p1, temperature, 0.0)
    assert second.fluxes is first.fluxes                  # fluxes held, not recomputed
    for out, p in ((first, p0), (second, p1)):
        column_heating = (cp * (p[1:] - p[:-1]) / gravity * out.heating).sum(axis=0)
        np.testing.assert_allclose(column_heating, out.column_energy_tendency, rtol=1e-12)
    # with the default interval nothing changes: same bits as the scheme's own heating
    every = physics(1).compute(p0, temperature, 0.0)
    expected = every.fluxes.heating.copy()
    expected[-1] += gravity / (cp * (p0[-1] - p0[-2])) * 4.0
    assert np.array_equal(every.heating, expected)



def test_vertical_path_shortwave_is_iscas_frierson_scheme():
    """Isca two_stream_gray_rad (frierson case, commit d1321ac3):
    ``sw_down(k) = insolation exp(-sw_tau(k))`` with no slant path, and
    ``sw_up(k) = albedo sw_down(surface)`` at every level, so the reflected
    beam leaves unabsorbed. With tau = 0.2 (p / 610 Pa)^4 (Thomson & Vallis
    2019 eq. 15) the atmosphere absorbs 1 - exp(-0.2) of the insolation
    whatever the zenith angle."""
    from tropoi.physics.radiation.gray import VerticalPathShortwave
    sw = VerticalPathShortwave(OpticalDepthProfile([(1.0, 4.0)], equator=0.2, reference=610.0))
    col = _column(25, ps=610.0, gravity=3.71, cp=844.8)
    rad = Radiation(isca_mars_longwave(), sw)
    toa = np.array([300.0, 40.0])
    mu0 = np.array([0.9, 0.05])                   # must not matter
    f = rad.fluxes(col, LowerBoundary("surface", temperature=np.full(2, 220.0), emissivity=1.0,
                                      albedo=0.3), toa_flux=toa, mu0=mu0)
    tau = 0.2 * (col.pressure_interfaces / 610.0) ** 4
    np.testing.assert_allclose(f.sw_down, toa[None, :] * np.exp(-tau), rtol=1e-14)
    np.testing.assert_allclose(f.sw_up, 0.3 * f.sw_down[-1][None, :] * np.ones_like(tau),
                               rtol=1e-14)
    absorbed_atm = (f.sw_down[0] - f.sw_up[0]) - (f.sw_down[-1] - f.sw_up[-1])
    np.testing.assert_allclose(absorbed_atm, toa * (1.0 - math.exp(-0.2)), rtol=1e-13)
    assert sw.signature()["kind"] == "vertical_path_shortwave"


def _mars_bottom_exchange(implicit: bool):
    """A column with Isca's 0.73 Pa bottom layer over the M-I slab, a 130 m/s
    lowest-level wind (k w / Ca ~ 13, the value measured in MARS_MI_PLAN §2c)."""
    sensible = {"roughness_length": 3.21e-5, "critical_richardson": 1.0, "gustiness": 0.0}
    if implicit:
        sensible["implicit"] = True
    physics = ColumnPhysics(Radiation(isca_mars_longwave(), None), gravity=3.71, cp=844.8,
                            gas_constant=192.0, surface=SlabSurface(8.257737e6, albedo=0.3),
                            sensible=sensible)
    sigma = np.array(SigmaGrid.stretched(25, 1e-4, None).interfaces)
    sigma[-2] = 1.0 - 0.73 / 610.0                     # Isca's bottom layer
    p = (sigma * 610.0)[:, None]
    return physics, p


def _exchange_numbers(physics, p, t_air, t_sfc, wind, window):
    out = physics.compute(p, np.vstack([np.full((24, 1), 200.0), [[t_air]]]), 0.0,
                          latitude=np.zeros(1), longitude=np.zeros(1),
                          surface_temperature=np.array([t_sfc]), wind_speed=np.array([wind]),
                          exchange_window=window, surface_step=window / 2.0)
    ca = physics.cp * (p[-1, 0] - p[-2, 0]) / physics.gravity
    return float(out.sensible[0]), ca


def test_implicit_surface_exchange_is_stable_where_the_explicit_one_diverges():
    """MARS_MI_PLAN §2d, I1. A leapfrog-like host: the air gets the flux over
    w = 2 dt, the slab over dt, from the same state."""
    w = 1200.0
    histories = {}
    for implicit in (False, True):
        physics, p = _mars_bottom_exchange(implicit)
        ta, ts = 200.0, 230.0
        hist = []
        for _ in range(20):
            f, ca = _exchange_numbers(physics, p, ta, ts, 130.0, w)
            ta += w * f / ca
            ts -= (w / 2.0) * f / physics.surface.heat_capacity
            hist.append(ta)
        histories[implicit] = np.array(hist)
    explicit, implicit = histories[False], histories[True]
    # explicit: overshoots past equilibrium to a non-physical temperature at once
    assert np.abs(explicit[0] - 200.0) > 100.0
    assert not np.all(np.isfinite(explicit)) or explicit.min() <= 0.0
    # implicit: finite, monotone towards the slab, never past it
    assert np.all(np.isfinite(implicit))
    assert np.all(np.diff(implicit) >= -1e-9) and implicit.max() < 230.0 + 1e-9


def test_implicit_surface_exchange_conserves_energy_on_every_call():
    """I1: with one step for air and slab, Ca dT_a + Cs dT_s = 0 to round-off."""
    physics, p = _mars_bottom_exchange(True)
    dt = 600.0
    for t_air, t_sfc, wind in ((200.0, 230.0, 130.0), (250.0, 210.0, 5.0), (215.0, 215.5, 40.0)):
        out = physics.compute(p, np.vstack([np.full((24, 1), 200.0), [[t_air]]]), 0.0,
                              latitude=np.zeros(1), longitude=np.zeros(1),
                              surface_temperature=np.array([t_sfc]),
                              wind_speed=np.array([wind]), exchange_window=dt, surface_step=dt)
        f = float(out.sensible[0])
        ca = physics.cp * (p[-1, 0] - p[-2, 0]) / physics.gravity
        d_air = dt * f / ca
        d_sfc = -dt * f / physics.surface.heat_capacity
        assert abs(ca * d_air + physics.surface.heat_capacity * d_sfc) <= 1e-12 * abs(dt * f) + 1e-300
        # and it never overshoots the shared equilibrium of the two reservoirs
        assert (t_air + d_air - (t_sfc + d_sfc)) * (t_air - t_sfc) >= -1e-9


def test_the_implicit_flag_is_off_by_default_and_leaves_the_explicit_flux_unchanged():
    """I2: no flag, no window -> the explicit flux, bit for bit."""
    a, p = _mars_bottom_exchange(False)
    b, _ = _mars_bottom_exchange(True)
    fa, _ = _exchange_numbers(a, p, 200.0, 230.0, 30.0, 1200.0)
    out = b.compute(p, np.vstack([np.full((24, 1), 200.0), [[200.0]]]), 0.0,
                    latitude=np.zeros(1), longitude=np.zeros(1),
                    surface_temperature=np.array([230.0]), wind_speed=np.array([30.0]))
    assert float(out.sensible[0]) == fa                 # implicit without a window = explicit
    assert "implicit" not in a.signature()["sensible"]
