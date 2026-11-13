"""The shared physical world of ``tropoi.physics``: packaged species and
property data, strict compositions with explicit normalization, Kepler
orbits against analytic cases, stellar consistency and distance scaling,
rotation/illumination geometry, and the frozen configuration contract of
the radiative experiments."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.physics.composition import (COMPOSITIONS, Composition,
                                        CompositionError,
                                        NORMALIZATION_GUARD, SUM_TOLERANCE)
from tropoi.physics.constants import (AU_M, G, L_SUN_W, R_SUN_M, R_UNIVERSAL,
                                      SIGMA_SB, SOLAR_CONSTANT, T_SUN_K)
from tropoi.physics.insolation import (DiurnalMeanInsolation,
                                       StellarIllumination)
from tropoi.physics.orbit import (KeplerOrbit, mean_from_true, true_from_mean)
from tropoi.physics.planets import PLANETS
from tropoi.physics.rayleigh import OPTICS, composition_cross_section
from tropoi.physics.rotation import Rotation
from tropoi.physics.species import (ATOMIC_WEIGHTS, DATASETS, SPECIES,
                                    dataset_identities)
from tropoi.physics.star import STARS, ApparentStar, Star
from tropoi.physics.thermodynamics import THERMO, heat_capacity


# ---------------------------------------------------------------- packaged data

def test_packaged_datasets_load_through_package_resources_and_identify():
    ids = dataset_identities()
    assert set(ids) == set(DATASETS)
    for name, ident in ids.items():
        assert len(ident["sha256"]) == 64, name
        assert ident["schema"].startswith("palintropos."), name


def test_molar_masses_follow_from_the_one_atomic_weight_table():
    # computed per molecule from the CIAAW 2021 atomic weights, never
    # transcribed per species
    assert SPECIES["N2"].molar_mass == pytest.approx(2 * ATOMIC_WEIGHTS["N"] / 1000, rel=1e-15)
    assert SPECIES["CO2"].molar_mass == pytest.approx(
        (ATOMIC_WEIGHTS["C"] + 2 * ATOMIC_WEIGHTS["O"]) / 1000, rel=1e-15)
    # the canonical values sit within 1e-4 of the retired per-molecule table
    legacy = {"H2": 2.01588e-3, "He": 4.002602e-3, "Ar": 39.948e-3,
              "N2": 28.0134e-3, "O2": 31.9988e-3, "CO": 28.0101e-3,
              "CO2": 44.0095e-3, "H2O": 18.01528e-3, "CH4": 16.04246e-3,
              "NH3": 17.03052e-3, "SO2": 64.0638e-3}
    for name, old in legacy.items():
        assert SPECIES[name].molar_mass == pytest.approx(old, rel=1e-4), name


def test_species_formula_reports_atom_counts_without_mutating_the_registry():
    carbon_dioxide = SPECIES["CO2"]
    formula = carbon_dioxide.formula_dict()
    assert formula == {"C": 1, "O": 2}
    formula["O"] = 1
    assert carbon_dioxide.formula_dict() == {"C": 1, "O": 2}


def test_species_identity_is_separate_from_property_coverage():
    # every identity has thermodynamic data today; only four have optics
    assert set(THERMO) == set(SPECIES)
    assert set(OPTICS) == {"N2", "O2", "Ar", "CO2"}
    co = Composition.of(CO=1.0)                        # known identity
    assert float(heat_capacity(co, 300.0)) > 0.0       # thermo covered
    with pytest.raises(CompositionError, match="no optical data"):
        composition_cross_section(co, 550.0)           # optics not covered
    with pytest.raises(CompositionError, match="no thermodynamic data"):
        co.require_coverage([], "thermodynamic")


def test_every_packaged_dataset_is_declared_as_package_data():
    """Every data file the loaders read must be covered by a pyproject
    package-data glob, or an installed wheel would silently lack it (the
    wheel built at integration time was inspected and carries all of them;
    docs/radiation/SHARED_PHYSICS.md records the listing)."""
    import pathlib
    import tomllib
    repo = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text("utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "data/*.toml" in package_data["tropoi.physics"]
    assert "data/*.csv" in package_data["tropoi.representation.visual"]
    physics_data = repo / "src" / "tropoi" / "physics" / "data"
    on_disk = {p.name for p in physics_data.glob("*.toml")}
    assert on_disk == set(DATASETS)
    stray = [p.name for p in physics_data.iterdir()
             if p.is_file() and p.suffix != ".toml"]
    assert not stray, f"undeclared data files would not ship: {stray}"


# ---------------------------------------------------------------- composition

def test_strict_validation_and_explicit_normalization():
    with pytest.raises(CompositionError, match="never renormalized"):
        Composition.of(N2=0.7809, O2=0.2095, Ar=0.0093, CO2=0.0004)  # 1.0001
    # float round-off within SUM_TOLERANCE is accepted
    assert Composition.of(N2=0.3, O2=0.3, Ar=0.2, CO2=0.2).molar_mass > 0
    norm = Composition.normalized(
        {"N2": 0.7809, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004},
        label="round numbers", source="test")
    assert sum(x for _, x in norm.mole_fractions) == pytest.approx(1.0, abs=1e-15)
    assert norm.normalization["stated_total"] == pytest.approx(1.0001)
    assert norm.normalization["stated_mole_fractions"][0] == ["N2", 0.7809]
    assert norm.to_dict()["normalization"]["stated_total"] == pytest.approx(1.0001)
    with pytest.raises(CompositionError, match="not a rounding remainder"):
        Composition.normalized({"N2": 0.5}, label="incomplete", source="test")
    assert NORMALIZATION_GUARD > SUM_TOLERANCE


def test_mole_mass_conversion_round_trips_both_ways():
    c = Composition.of(H2=0.85, He=0.13, CH4=0.02)
    w = dict(c.mass_fractions)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-15)
    back = Composition.from_mass_fractions(**w)
    for (k, x), (k2, x2) in zip(back.mole_fractions, c.mole_fractions):
        assert k == k2 and x == pytest.approx(x2, rel=1e-14)
    # mass fraction of the heavy trace exceeds its mole fraction
    assert w["CH4"] > 0.02


def test_preset_families_stay_distinct_with_their_own_provenance():
    climate, sky_ = COMPOSITIONS["mars"], COMPOSITIONS["mars_illustrative"]
    assert dict(climate.mole_fractions) != dict(sky_.mole_fractions)
    assert climate.source != sky_.source
    assert climate.normalization is not None          # 0.999 stated total
    assert climate.normalization["stated_total"] == pytest.approx(0.999)
    assert sky_.normalization is None                 # sums to 1 as stated
    assert "NSSDCA" in sky_.source
    # the normalized climate mixture reproduces the retired GasMixture
    assert climate.mole_fraction("CO2") == pytest.approx(0.951 / 0.999, rel=1e-14)
    assert climate.mole_fraction("CH4") == 0.0


# ---------------------------------------------------------------- orbits

def test_kepler_equation_round_trips_and_period_factory():
    for e in (0.0, 0.0934, 0.3, 0.9):
        for nu in np.linspace(0.05, 6.2, 25):
            back = true_from_mean(mean_from_true(float(nu), e), e)
            assert back % (2 * math.pi) == pytest.approx(
                float(nu) % (2 * math.pi), abs=1e-9), (e, nu)
    mu = 1.3271244e20
    period_s = 365.256 * 86400.0
    orbit = KeplerOrbit.from_period(period_s, mu)
    assert orbit.period(mu) == pytest.approx(period_s, rel=1e-14)
    assert orbit.mean_motion(mu) == pytest.approx(math.tau / period_s, rel=1e-14)
    assert orbit.semimajor_axis_m == pytest.approx(AU_M, rel=1e-3)  # ~1 au


def test_circular_orbit_propagates_uniformly_and_equatorially():
    mu = 1.0e20
    orbit = KeplerOrbit(1.0e11)                       # e = 0, i = 0
    period = orbit.period(mu)
    for f in (0.1, 0.35, 0.75):
        nu = orbit.true_anomaly_at(f * period, gravitational_parameter=mu)
        assert nu == pytest.approx(2 * math.pi * f, abs=1e-12)
        assert orbit.radius_at(nu) == pytest.approx(1.0e11, rel=1e-15)
        assert orbit.position_at(nu)[2] == 0.0        # equatorial convention


def test_eccentric_orbit_matches_the_analytic_ellipse():
    orbit = KeplerOrbit.from_apsides(0.7e11, 1.3e11)
    assert orbit.semimajor_axis_m == pytest.approx(1.0e11)
    assert orbit.eccentricity == pytest.approx(0.3)
    assert orbit.radius_at(0.0) == pytest.approx(0.7e11, rel=1e-15)
    assert orbit.radius_at(math.pi) == pytest.approx(1.3e11, rel=1e-15)
    # Kepler's second law: numerically, r^2 dnu/dt is constant
    mu = 3.0e19
    period = orbit.period(mu)
    t = np.linspace(0.0, period, 2001)
    nu = np.array([orbit.true_anomaly_at(float(ti), gravitational_parameter=mu)
                   for ti in t])
    nu = np.unwrap(nu)
    r = np.array([orbit.radius_at(float(n)) for n in nu])
    rate = (r[:-1] ** 2 + r[1:] ** 2) / 2 * np.diff(nu) / np.diff(t)
    assert rate.std() / rate.mean() < 1e-5
    # one full revolution per period
    assert nu[-1] - nu[0] == pytest.approx(2 * math.pi, abs=1e-9)


def test_inclined_orbit_positions_follow_the_stated_frame():
    orbit = KeplerOrbit(1.0, inclination_rad=math.pi / 2,
                        ascending_node_rad=0.0, periapsis_longitude_rad=0.0)
    np.testing.assert_allclose(orbit.position_at(0.0), [1.0, 0.0, 0.0],
                               atol=1e-15)
    np.testing.assert_allclose(orbit.position_at(math.pi / 2),
                               [0.0, 0.0, 1.0], atol=1e-15)


def test_irradiance_factory_requires_an_explicit_reference():
    lum = L_SUN_W
    with pytest.raises(TypeError):
        KeplerOrbit.from_irradiance(1361.0, lum)            # no reference
    with pytest.raises(ValueError, match="say explicitly"):
        KeplerOrbit.from_irradiance(1361.0, lum, reference="mean")
    a = KeplerOrbit.from_irradiance(SOLAR_CONSTANT, lum,
                                    reference="semimajor_axis")
    assert a.semimajor_axis_m == pytest.approx(AU_M, rel=2e-4)
    e = 0.3
    peri = KeplerOrbit.from_irradiance(1000.0, lum, reference="periapsis",
                                       eccentricity=e)
    assert peri.periapsis_m == pytest.approx(
        math.sqrt(lum / (4 * math.pi * 1000.0)), rel=1e-12)
    # time_mean: <r^-2> over one orbit equals 1 / (a^2 sqrt(1 - e^2))
    tm = KeplerOrbit.from_irradiance(1000.0, lum, reference="time_mean",
                                     eccentricity=e)
    mu = 1.0e20
    period = tm.period(mu)
    t = np.linspace(0.0, period, 20001)[:-1]
    inv_r2 = np.array([tm.radius_at_time(float(ti), gravitational_parameter=mu)
                       for ti in t]) ** -2.0
    d2 = lum / (4 * math.pi * 1000.0)
    assert inv_r2.mean() == pytest.approx(1.0 / d2, rel=1e-4)


def test_orbit_validation_rejects_unbound_and_nonsense_elements():
    with pytest.raises(ValueError, match="bound ellipse"):
        KeplerOrbit(1.0e11, eccentricity=1.0)
    with pytest.raises(ValueError, match="> 0"):
        KeplerOrbit(0.0)
    with pytest.raises(ValueError, match="inclination"):
        KeplerOrbit(1.0e11, inclination_rad=4.0)
    with pytest.raises(ValueError, match="exactly one"):
        KeplerOrbit(1.0e11).true_anomaly_at(0.0)


# ---------------------------------------------------------------- stars

def test_blackbody_star_derives_the_missing_quantity_and_rejects_contradictions():
    sun_tr = Star.blackbody("tr", temperature_k=T_SUN_K, radius_rsun=1.0)
    sun_tl = Star.blackbody("tl", temperature_k=T_SUN_K,
                            luminosity_lsun=sun_tr.luminosity_lsun)
    sun_rl = Star.blackbody("rl", radius_rsun=1.0,
                            luminosity_lsun=sun_tr.luminosity_lsun)
    assert sun_tl.radius_m == pytest.approx(R_SUN_M, rel=1e-12)
    assert sun_rl.temperature_k == pytest.approx(T_SUN_K, rel=1e-12)
    assert sun_tr.luminosity_w == pytest.approx(
        4 * math.pi * R_SUN_M ** 2 * SIGMA_SB * T_SUN_K ** 4, rel=1e-15)
    with pytest.raises(ValueError, match="needs two"):
        Star.blackbody("t", temperature_k=5000.0)
    with pytest.raises(ValueError, match="inconsistent blackbody"):
        Star("bad", T_SUN_K, R_SUN_M, 2 * sun_tr.luminosity_w)


def test_irradiance_scales_with_inverse_square_distance():
    sun = STARS["sun"]
    s1 = sun.irradiance_at(AU_M)
    assert s1 == pytest.approx(SOLAR_CONSTANT, rel=1.5e-3)  # nominal rounding
    assert sun.irradiance_at(2 * AU_M) == pytest.approx(s1 / 4, rel=1e-15)
    assert sun.angular_radius_deg_at(AU_M) == pytest.approx(
        math.degrees(math.asin(R_SUN_M / AU_M)), rel=1e-12)


def test_seen_from_reduces_to_the_apparent_star_of_the_sky_page():
    apparent = STARS["sun"].seen_from(AU_M)
    nominal = ApparentStar.sun_at_1au()
    assert apparent.angular_radius_deg == pytest.approx(
        nominal.angular_radius_deg, rel=1e-12)
    assert apparent.temperature_k == nominal.temperature_k
    # viewing geometry ties the two descriptions together
    assert apparent.bolometric_irradiance_w_m2 == pytest.approx(
        STARS["sun"].irradiance_at(AU_M), rel=1e-5)
    # an apparent star never needs an invented mass or distance
    bare = ApparentStar(3200.0, 0.87)
    assert bare.radius_rsun is None and bare.distance_au is None


def test_sun_mass_comes_from_the_nominal_gm():
    sun = STARS["sun"]
    assert sun.gravitational_parameter == pytest.approx(1.3271244e20, rel=1e-14)
    assert sun.mass_kg == pytest.approx(1.3271244e20 / G, rel=1e-14)
    assert STARS["sirius_a"].gravitational_parameter is None


def test_a_supplied_spectrum_overrides_the_blackbody():
    class Flat:
        def radiance(self, wavelength_nm):
            return np.full(np.shape(wavelength_nm), 7.0)
    star = Star("flat", 5000.0, R_SUN_M,
                4 * math.pi * R_SUN_M ** 2 * SIGMA_SB * 5000.0 ** 4,
                spectral_model="tabulated", spectrum=Flat())
    np.testing.assert_array_equal(star.radiance([500.0, 600.0]), [7.0, 7.0])


# ---------------------------------------------------------------- rotation and illumination

def test_rotation_owns_the_spin_state():
    r = Rotation.of(7.292115e-5, obliquity_deg=23.44)
    assert r.sidereal_period_s == pytest.approx(2 * math.pi / 7.292115e-5)
    assert r.obliquity_deg == pytest.approx(23.44)
    retro = Rotation.from_period(-243.0 * 86400.0)
    assert retro.angular_velocity < 0.0
    with pytest.raises(ValueError, match="obliquity"):
        Rotation.of(1e-5, obliquity_deg=200.0)
    assert Rotation(0.0).sidereal_period_s == math.inf


def test_illumination_from_components_matches_the_prescribed_form():
    sun = STARS["sun"]
    orbit = KeplerOrbit.from_irradiance(
        1361.0, sun.luminosity_w, reference="semimajor_axis",
        eccentricity=0.0167, periapsis_longitude_rad=math.radians(282.9))
    rotation = Rotation.of(7.292115e-5, obliquity_deg=23.44)
    combined = StellarIllumination.from_components(sun, orbit, rotation)
    stated = StellarIllumination.prescribed(
        1361.0, combined.period_s, 0.0167, 23.44, 282.9)
    for t in np.linspace(0.0, combined.period_s, 13):
        assert combined.flux_at(float(t)) == pytest.approx(
            stated.flux_at(float(t)), rel=1e-12)
        assert combined.declination(float(t)) == pytest.approx(
            stated.declination(float(t)), abs=1e-12)
    assert combined.irradiance_at_semimajor == pytest.approx(1361.0, rel=1e-12)
    # period from mu = the nominal solar GM: a sidereal year to ~0.1 %
    assert combined.period_s == pytest.approx(365.25 * 86400.0, rel=1e-3)
    d = combined.as_dict()
    assert d["stellar_constant"] == pytest.approx(1361.0)
    assert d["components"]["star"]["key"] == "sun"
    with pytest.raises(ValueError, match="gravitational parameter"):
        StellarIllumination.from_components(STARS["sirius_a"], orbit, rotation)


def test_illumination_honours_the_orbits_epoch():
    # An orbit stated at apoapsis (nu_0 = pi): the illumination's flux at any
    # time must agree with the star's irradiance at the orbit's own position.
    sun = STARS["sun"]
    orbit = KeplerOrbit.from_au(1.0, eccentricity=0.3,
                                true_anomaly_epoch_rad=math.pi)
    rotation = Rotation.of(7.292115e-5)
    ill = StellarIllumination.from_components(sun, orbit, rotation)
    mu = sun.gravitational_parameter
    for t in (0.0, 0.25 * ill.period_s, 0.6 * ill.period_s):
        r = orbit.radius_at_time(t, gravitational_parameter=mu)
        assert ill.flux_at(t) == pytest.approx(sun.irradiance_at(r), rel=1e-9)
    # An explicit time_of_periapsis still overrides the orbit's epoch.
    forced = StellarIllumination.from_components(sun, orbit, rotation,
                                                 time_of_periapsis=0.0)
    assert forced.time_of_periapsis_s == 0.0
    assert forced.flux_at(0.0) == pytest.approx(
        forced.irradiance_at_semimajor / (1.0 - 0.3) ** 2, rel=1e-12)


def test_forcing_conventions_are_preserved():
    ill = StellarIllumination.prescribed(1000.0, 100.0, 0.3, 25.0, 90.0)
    # at perihelion (t = 0): flux = S / (1 - e)^2
    assert ill.flux_at(0.0) == pytest.approx(1000.0 / 0.7 ** 2, rel=1e-12)
    # L_s(perihelion) = 90 deg: northern summer solstice => dec = obliquity
    assert ill.declination(0.0) == pytest.approx(math.radians(25.0), abs=1e-12)
    # legacy signature keys stay put for stored manifests
    sig = DiurnalMeanInsolation(ill).signature()
    assert sig["kind"] == "diurnal_mean"
    assert set(sig["orbit"]) == {"stellar_constant", "period", "eccentricity",
                                 "obliquity_deg", "perihelion_longitude_deg",
                                 "time_of_perihelion"}


def test_planet_presets_carry_the_shared_objects_with_stated_scalars():
    earth = PLANETS["earth"]
    assert earth.star.key == "sun"
    assert earth.orbit.eccentricity == 0.0167
    assert earth.rotation.obliquity_deg == pytest.approx(23.44)
    assert earth.illumination.irradiance_at_semimajor == 1361.0   # stated
    assert earth.rotation_rate == 7.292115e-5
    # the stated constant and the shared objects agree to fact-sheet rounding
    derived = earth.star.irradiance_at(earth.orbit.semimajor_axis_m)
    assert derived == pytest.approx(1361.0, rel=2e-3)
    venus = PLANETS["venus"]
    assert venus.rotation.angular_velocity < 0.0      # retrograde


# ---------------------------------------------------------------- frozen config contract

def test_radiative_config_hash_still_matches_the_committed_evidence():
    """The canonical JSON hash is the checkpoint/resume contract: it must
    reproduce the config_sha256 of the committed evidence runs."""
    import json
    import pathlib
    from tropoi.run.radiative.config import PRESETS
    evidence = pathlib.Path(__file__).resolve().parents[1] / "docs" / \
        "radiation" / "evidence" / "earth-gray-dry-T21L25-300d" / "config.json"
    recorded = json.loads(evidence.read_text(encoding="utf-8"))
    assert PRESETS["earth-gray-dry"](days=300).sha256() == \
        recorded["config_sha256"] == \
        "e31ae183937da3bc9c5a63cbaf5c889bbeefc384a2f70b81e529b7a6965890d5"


def test_insolation_specs_resolve_through_the_shared_illumination():
    from tropoi.run.radiative.model import build_insolation
    ins = build_insolation({"kind": "diurnal_mean", "stellar_constant": 50.7})
    assert isinstance(ins, DiurnalMeanInsolation)
    assert ins.illumination.irradiance_at_semimajor == 50.7
    lat = np.array([0.0])
    flux, mu0 = ins(lat, 0 * lat, 0.0)
    assert flux[0] == pytest.approx(50.7 / math.pi, rel=1e-12)   # SL09 form
    inst = build_insolation({"kind": "instantaneous", "stellar_constant": 100.0,
                             "solar_day": 86400.0})
    assert inst.solar_day == 86400.0
