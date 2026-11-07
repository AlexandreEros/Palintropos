"""Sky optics laboratory: optics, composition, column, reference, page contracts.

CPU only. No assertion compares a sky colour or an optical depth with an
expected "look": the checks are internal physics and contracts --
analytic hydrostatics, exact inverses, refusals, the no-inference rule,
column counts against p_s R/(k_B g), numerical convergence, the King
factors and air cross section against Bodhaine et al. (1999) as printed,
and the page boundary (a standalone page that needs no simulation
infrastructure; globe pages carry no sky; saved runs untouched).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from tropoi.spatial.optics import (COMPOSITION_PRESETS, SPECIES,
                                   AtmosphericColumn, ColumnError,
                                   Composition, CompositionError, Star,
                                   gas_constant_report)
from tropoi.spatial.optics.colour import (CIE_SHA256, bin_weights, cie_1931,
                                          omitted_fraction)
from tropoi.spatial.optics.constants import K_B, R_MOLAR
from tropoi.representation.visual import sky

AIR = COMPOSITION_PRESETS["earth_dry_air_illustrative"]
REPO = pathlib.Path(__file__).resolve().parents[1]


# -- optics ------------------------------------------------------------------

def test_cross_sections_positive_and_refuse_extrapolation():
    lam = sky.WAVELENGTHS_NM
    for name, species in SPECIES.items():
        s = species.cross_section(lam)
        assert np.all(s > 0) and np.all(np.diff(s) < 0), name   # falls with lambda
        with pytest.raises(ValueError, match="measured range"):
            species.cross_section([2000.0])      # beyond every record
    with pytest.raises(ValueError, match="measured range"):
        SPECIES["N2"].cross_section([390.0])
    with pytest.raises(ValueError, match="no optical data"):
        sky.SkySpec(Composition.of(He=1.0))


def test_cross_section_scales_as_inverse_square_reference_density():
    # sigma ~ ((n^2-1)/(n^2+2) / N_ref)^2: the same refractivity attributed
    # to twice the density means a quarter of the cross section, so the
    # recorded measurement conditions matter and are carried with the data.
    import dataclasses
    n2 = SPECIES["N2"]
    s0 = n2.cross_section(550.0)
    twice = dataclasses.replace(n2, ref_pressure_pa=2 * n2.ref_pressure_pa)
    assert float(twice.cross_section(550.0)) == pytest.approx(float(s0) / 4)


def test_every_species_records_its_sources_and_checked_king_factor():
    for species in SPECIES.values():
        text = species.describe()
        assert "refractiveindex.info" in text and "Bates (1984)" in text
        assert "eqs. 5-6" in text
        assert species.king_verified is True
        assert "not yet checked" not in text


def test_king_factors_are_those_printed_by_bodhaine_1999():
    # Bodhaine et al. (1999), J. Atmos. Oceanic Technol. 16, 1854, as
    # printed: eq. 5 F(N2) = 1.034 + 3.17e-4 / lambda^2, eq. 6 F(O2) = 1.096
    # + 1.385e-3 / lambda^2 + 1.448e-4 / lambda^4, then "F(Ar) = 1.00,
    # F(CO2) = 1.15" (checked against the paper 2026-11-07).
    assert SPECIES["N2"].king == (1.034, 3.17e-4, 0.0)
    assert SPECIES["O2"].king == (1.096, 1.385e-3, 1.448e-4)
    assert SPECIES["Ar"].king == (1.0, 0.0, 0.0)
    assert SPECIES["CO2"].king == (1.15, 0.0, 0.0)

    # lambda in micrometres: the paper's eq. 23 for 360 ppm CO2 air
    # reproduces its Fig. 1 (about 1.080 at 0.2 um, 1.047 at 1 um)
    def f_air(um):
        n2, o2 = (SPECIES[k].king_factor(um) for k in ("N2", "O2"))
        return float((78.084 * n2 + 20.946 * o2 + 0.934 * 1.00
                      + 0.036 * 1.15) / (78.084 + 20.946 + 0.934 + 0.036))
    assert f_air(0.2) == pytest.approx(1.080, abs=1.5e-3)
    assert f_air(1.0) == pytest.approx(1.047, abs=1.5e-3)


def test_dry_air_cross_section_agrees_with_bodhaine_eq_29():
    # End to end (refractive indices, King factors, constants): eq. 29 is
    # the paper's fit (better than 0.01 %) to its sigma of 360 ppm CO2 air,
    # built from Peck & Reeder's air index, not from our per-gas records.
    lam = sky.WAVELENGTHS_NM
    um = lam / 1000.0
    eq29 = ((1.0455996 - 341.29061 * um ** -2 - 0.90230850 * um ** 2)
            / (1 + 0.0027059889 * um ** -2 - 85.968563 * um ** 2)) * 1e-32
    rel = AIR.cross_section(lam) / eq29 - 1.0
    # measured 2026-11-07: -0.094 % .. -0.054 % over 410-750 nm
    assert np.max(np.abs(rel)) < 1.5e-3


def test_cie_table_is_the_verbatim_cie_dataset():
    raw = (pathlib.Path(sky.__file__).parents[2] / "spatial" / "optics"
           / "data" / "CIE_xyz_1931_2deg.csv").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CIE_SHA256
    lam, cmf = cie_1931()
    assert lam[0] == 360 and lam[-1] == 830 and cmf.shape == (471, 3)
    assert cmf[lam == 555, 1][0] == 1.0
    w = bin_weights(sky.BIN_EDGES_NM[:-1], sky.BIN_EDGES_NM[1:])
    assert w.shape == (18, 3)
    # what the 400-760 nm band leaves out is reported, and it is small
    assert np.all(omitted_fraction(400, 760) < 0.01)


def test_star_irradiance_is_planck_times_solid_angle():
    sun = Star.sun_at_1au()
    lam = np.array([500.0])
    assert sun.irradiance(lam)[0] == pytest.approx(
        sun.radiance(lam)[0] * 2 * math.pi
        * (1 - math.cos(math.radians(sun.angular_radius_deg))))
    with pytest.raises(ValueError):
        Star(5000.0, 0.0)


# -- composition --------------------------------------------------------------

def test_composition_is_explicit_and_never_renormalized():
    with pytest.raises(CompositionError, match="never renormalized"):
        Composition.of(N2=0.78, O2=0.21)
    with pytest.raises(CompositionError, match="no optical data"):
        Composition.of(N2=0.9, CO=0.1)
    with pytest.raises(CompositionError, match="> 0"):
        Composition.of(N2=1.0, O2=0.0)
    with pytest.raises(TypeError, match="ever inferred"):
        sky.SkySpec(composition=None)
    c = Composition.of(CO2=1.0)
    assert c.molar_mass_kg_per_mol == pytest.approx(0.044009)
    assert c.gas_constant == pytest.approx(R_MOLAR / 0.044009)
    mix = Composition.of(N2=0.5, Ar=0.5)
    assert float(mix.cross_section(500.0)) == pytest.approx(
        0.5 * float(SPECIES["N2"].cross_section(500.0))
        + 0.5 * float(SPECIES["Ar"].cross_section(500.0)))


def test_gas_constant_report_compares_without_changing_anything():
    r = gas_constant_report(AIR, 287.04)
    assert r["status"] == "consistent" and r["r_dry"] == 287.04
    co2 = Composition.of(CO2=1.0)
    bad = gas_constant_report(co2, 287.04)
    assert bad["status"] == "inconsistent" and "DISAGREE" in bad["text"]
    assert bad["r_dry"] == 287.04                       # reported, not replaced
    assert gas_constant_report(co2, None)["status"] == "no_r_dry"


# -- column -------------------------------------------------------------------

def _iso(**kw):
    args = dict(surface_pressure_pa=1.0e5, temperature_k=250.0,
                gravity=9.80665, gas_constant=287.04)
    args.update(kw)
    return AtmosphericColumn.hypothetical(**args)


def test_hypothetical_isothermal_matches_analytic_hydrostatics():
    c = _iso()
    z = np.linspace(0.0, c.top_m, 37)
    h = 287.04 * 250.0 / 9.80665
    np.testing.assert_allclose(c.pressure_at(z), 1.0e5 * np.exp(-z / h),
                               rtol=1e-12)
    np.testing.assert_allclose(c.number_density(z),
                               1.0e5 * np.exp(-z / h) / (K_B * 250.0),
                               rtol=1e-12)
    assert c.kind == "hypothetical" and "HYPOTHETICAL" in c.notes[0]


@pytest.mark.parametrize("column", [
    _iso(),
    _iso(lapse_rate_k_per_m=0.0065, temperature_floor_k=200.0),
])
def test_altitude_pressure_round_trip(column):
    z = np.linspace(column.base_m, column.ceiling_m, 25)
    np.testing.assert_allclose(column.altitude_at(column.pressure_at(z)), z,
                               rtol=1e-11, atol=1e-6)
    obs = column.observer_altitude(pressure_pa=60000.0)
    back = column.observer_altitude(altitude_m=obs["altitude_m"])
    assert back["pressure_pa"] == pytest.approx(60000.0, rel=1e-12)


def test_lapse_rate_profile_temperature_and_pressure():
    c = _iso(lapse_rate_k_per_m=0.0065, temperature_floor_k=200.0)
    z_f = (250.0 - 200.0) / 0.0065
    z = np.array([0.0, 1000.0, 5000.0, z_f, z_f + 3000.0])
    np.testing.assert_allclose(c.temperature_at(z),
                               [250.0, 243.5, 217.5, 200.0, 200.0],
                               rtol=1e-12)
    # node values are the analytic polytrope p = p_s (T/T0)^(g/(R Gamma))
    expo = 9.80665 / (287.04 * 0.0065)
    nodes = c.z_m[:-1]
    np.testing.assert_allclose(
        c.pressure_at(nodes), 1.0e5 * ((250.0 - 0.0065 * nodes) / 250.0)
        ** expo, rtol=1e-12)


def test_observer_refusals():
    c = _iso()
    with pytest.raises(ColumnError, match="exactly one"):
        c.observer_altitude(altitude_m=1.0, pressure_pa=5e4)
    with pytest.raises(ColumnError, match="underground"):
        c.observer_altitude(altitude_m=100.0, terrain_m=500.0)
    with pytest.raises(ColumnError, match="underground"):
        c.observer_altitude(pressure_pa=1.2e5)
    with pytest.raises(ColumnError, match="above the top"):
        c.observer_altitude(altitude_m=c.ceiling_m + 10.0)
    with pytest.raises(ColumnError, match="floor"):
        _iso(lapse_rate_k_per_m=0.0065)


def test_column_from_levels_nodes_and_ceiling():
    sig = np.array([0.1, 0.4, 0.7, 0.95])
    ps, hs, t = 9.0e4, 800.0, np.array([210.0, 240.0, 265.0, 280.0])
    r, g = 287.0, 9.81
    p = sig * ps
    z = hs + r * t / g * np.log(ps / p)            # isothermal-layer heights
    c = AtmosphericColumn.from_levels(z, p, t, surface_height_m=hs,
                                      surface_pressure_pa=ps,
                                      gas_constant=r, gravity=g)
    assert c.base_m == hs and c.surface_pressure_pa == pytest.approx(ps)
    assert c.ceiling_m == pytest.approx(z.max())
    assert c.pressure_at(c.top_m) == pytest.approx(ps * 1e-5)
    assert c.temperature_k[0] == t[-1] and c.temperature_k[-1] == t[0]
    np.testing.assert_allclose(c.pressure_at(np.sort(z)), np.sort(p)[::-1])
    with pytest.raises(ColumnError, match="above the top saved"):
        c.observer_altitude(altitude_m=z.max() + 1.0)
    with pytest.raises(ColumnError, match="not above the surface"):
        AtmosphericColumn.from_levels(z, p, t, surface_height_m=z.min() + 1,
                                      surface_pressure_pa=ps,
                                      gas_constant=r, gravity=g)


# -- reference radiance -----------------------------------------------------------

def _irradiance():
    return Star.sun_at_1au().irradiance(sky.WAVELENGTHS_NM)


def _table(column, **kw):
    args = dict(radius_m=6.371e6, observer_z_m=2.0, star_elevation_deg=30.0,
                cross_section_m2=AIR.cross_section(sky.WAVELENGTHS_NM),
                irradiance=_irradiance(), n_zenith=31, n_azimuth=13)
    args.update(kw)
    return sky.radiance_table((column.z_m, column.ln_p, column.temperature_k),
                              **args)


def test_vertical_column_count_matches_hydrostatics():
    c = _iso()
    t = _table(c, observer_z_m=0.0)
    expected = 1.0e5 * (1 - 1e-5) * 287.04 / (K_B * 9.80665)  # = int n dz
    assert t["column"][0, 0] == pytest.approx(expected, rel=2e-3)


def test_no_scatterer_no_sky_and_linearity_in_irradiance():
    c = _iso()
    zero = _table(c, cross_section_m2=np.zeros(18))
    assert np.all(zero["radiance"] == 0.0)
    a = _table(c)
    b = _table(c, irradiance=2.0 * np.ones(18) * 3.0)
    ratio = b["radiance"] / np.where(a["radiance"] > 0, a["radiance"], 1)
    irr = _irradiance()
    np.testing.assert_allclose(ratio[a["radiance"] > 0],
                               np.broadcast_to(6.0 / irr, ratio.shape)
                               [a["radiance"] > 0], rtol=1e-12)


def test_star_below_the_solid_sphere_lights_only_through_high_paths():
    c = _iso()
    deep = _table(c, star_elevation_deg=-60.0)
    assert np.all(deep["radiance"] == 0.0)     # whole column in shadow


def test_numerical_convergence_of_the_default_steps():
    c = _iso()
    coarse = _table(c, n_zenith=91, n_azimuth=7)
    fine = _table(c, n_zenith=91, n_azimuth=7, view_steps=4 * sky.VIEW_STEPS,
                  sun_steps=4 * sky.SUN_STEPS,
                  density_nodes=4 * sky.DENSITY_NODES)
    up = slice(0, 90)          # above the horizon
    rel = np.abs(coarse["radiance"][up] - fine["radiance"][up]) / \
        fine["radiance"][up]
    # measured at commit time: 2.8e-3 per spectral bin (luminance 5e-4)
    assert rel.max() < 5e-3


def test_star_path_column_overhead_is_the_zenith_column():
    c = _iso()
    nodes = (c.z_m, c.ln_p, c.temperature_k)
    t = _table(c, star_elevation_deg=90.0)
    overhead = sky.star_path_column(nodes, radius_m=6.371e6, observer_z_m=2.0,
                                    star_elevation_deg=90.0)
    assert overhead == pytest.approx(float(t["column"][0, 0]), rel=1e-12)
    low = sky.star_path_column(nodes, radius_m=6.371e6, observer_z_m=2.0,
                               star_elevation_deg=5.0)
    assert low > 5 * overhead                       # a long slant path
    assert sky.star_path_column(nodes, radius_m=6.371e6, observer_z_m=2.0,
                                star_elevation_deg=-30.0) is None


def test_optical_depth_diagnostic_levels():
    sig = AIR.cross_section(sky.WAVELENGTHS_NM)
    unit = 1.0 / float(sig.max())                   # tau = 1 at 410 nm
    for column, level in ((0.05 * unit, "ok"), (0.5 * unit, "caution"),
                          (2.0 * unit, "unreliable")):
        d = sky.optical_depth_diagnostic(sig, column, None)
        assert d["level"] == level
        assert d["peak_zenith"] == pytest.approx(column / unit)
        assert d["tau_star"] is None
    d = sky.optical_depth_diagnostic(sig, unit, 3 * unit)
    np.testing.assert_allclose(d["tau_star"], 3 * np.asarray(d["tau_zenith"]))
    assert sky.TAU_CAUTION < sky.TAU_UNRELIABLE


# -- page contracts ---------------------------------------------------------------

def _hyp(**kw):
    args = dict(surface_pressure_pa=1.0e4, temperature_k=250.0,
                gravity=9.80665, gas_constant="from_composition",
                radius_m=6.371e6)
    args.update(kw)
    return sky.HypotheticalColumn(**args)


def _hyp_spec(**kw):
    args = dict(composition=Composition.of(N2=1.0, label="N2"),
                star=Star(3200.0, 0.87), hypothetical=_hyp())
    args.update(kw)
    return sky.SkySpec(**args)


def _block(path) -> dict:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    raw = re.search(r'id="sky-data" type="application/json">(.*?)</script>',
                    text, re.S).group(1)
    return json.loads(raw.replace("\\u003c", "<").replace("\\u003e", ">"))


def test_spec_refusals():
    with pytest.raises(TypeError, match="ever inferred"):
        sky.SkySpec("N2", hypothetical=_hyp())
    with pytest.raises(ValueError, match="horizontally uniform"):
        _hyp_spec(observer_lat_deg=10.0, observer_lon_deg=0.0)
    with pytest.raises(ValueError, match="or neither"):
        sky.SkySpec(AIR, observer_lat_deg=10.0)
    with pytest.raises(ValueError, match="exposure_reference_cd_m2 > 0"):
        _hyp_spec(exposure_mode="fixed")
    with pytest.raises(ValueError, match="'auto' uses"):
        _hyp_spec(exposure_reference_cd_m2=100.0)
    with pytest.raises(ValueError, match="exposure_mode"):
        _hyp_spec(exposure_mode="bright")
    with pytest.raises(ValueError, match="radius_m"):
        _hyp(radius_m=0.0)
    with pytest.raises(ValueError, match="from_composition"):
        _hyp(gas_constant="guess")


def test_write_sky_page_hypothetical_block(tmp_path):
    path = sky.write_sky_page(_hyp_spec(pressure_pa=5000.0),
                              tmp_path / "s.html")
    block = _block(path)
    assert block["label"].startswith("Visual prototype")
    assert block["column_kind"] == "hypothetical"
    assert block["hypothetical"]["radius_m"] == 6.371e6
    assert block["radius_m"] == 6.371e6
    assert block["observer"]["mode"] == "pressure"
    assert block["observer"]["pressure_pa"] == pytest.approx(5000.0)
    assert block["consistency"]["status"] == "consistent"   # R from it
    assert block["exposure"] == {"mode": "auto", "ev": 0.0,
                                 "reference_cd_m2": None}
    assert set(block["species"]) == set(SPECIES)
    for name, sp in block["species"].items():
        np.testing.assert_array_equal(
            sp["cross_section_m2"],
            SPECIES[name].cross_section(sky.WAVELENGTHS_NM))
    assert block["diagnostic"] == {"tau_caution": sky.TAU_CAUTION,
                                   "tau_unreliable": sky.TAU_UNRELIABLE}
    assert any("aerosols" in u for u in block["unsupported"])
    assert any("CIE" in s for s in block["sources"])
    assert any("checked 2026-11-07" in s for s in block["sources"])
    with pytest.raises(ValueError, match="needs spec.hypothetical"):
        sky.write_sky_page(sky.SkySpec(AIR), tmp_path / "x.html")
    with pytest.raises(ColumnError, match="exceeds the surface pressure"):
        sky.write_sky_page(_hyp_spec(pressure_pa=2.0e4), tmp_path / "y.html")
    fixed = _block(sky.write_sky_page(
        _hyp_spec(exposure_mode="fixed", exposure_reference_cd_m2=800.0),
        tmp_path / "f.html"))
    assert fixed["exposure"]["mode"] == "fixed"
    assert fixed["exposure"]["reference_cd_m2"] == 800.0


def test_numeric_gas_constant_is_reported_not_replaced(tmp_path):
    block = _block(sky.write_sky_page(
        _hyp_spec(hypothetical=_hyp(gas_constant=287.0)),
        tmp_path / "r.html"))
    assert block["column"]["gas_constant"] == 287.0     # kept as given
    assert block["consistency"]["status"] == "inconsistent"
    assert block["consistency"]["r_mix"] == pytest.approx(R_MOLAR / 0.028014)


def test_globe_pages_carry_no_sky():
    from tropoi.representation.visual.terrain import write_globe_html
    from tropoi.run.custom.planet import PreparedPlanet
    template = (pathlib.Path(sky.__file__).with_name("terrain_globe.html")
                .read_text(encoding="utf-8"))
    assert "sky" not in template.lower()
    assert "sky" not in inspect.signature(write_globe_html).parameters
    assert "sky" not in inspect.signature(PreparedPlanet.view).parameters


def test_example_needs_no_simulation_infrastructure(tmp_path):
    # The standalone example runs with CuPy, Matplotlib, the archive, runs,
    # solvers, diagnostics, terrain, transforms and the globe viewer all
    # made unimportable, and none of them is loaded afterwards.
    out = tmp_path / "n2.html"
    banned = ["cupy", "cupyx", "matplotlib", "tropoi.temporal",
              "tropoi.run", "tropoi.representation.archive",
              "tropoi.representation.diagnostics",
              "tropoi.representation.visual.terrain",
              "tropoi.representation.visual.atmosphere",
              "tropoi.spatial.terrain", "tropoi.spatial.transforms"]
    example = REPO / "examples" / "sky_hypothetical.py"
    probe = "\n".join([
        "import sys, runpy",
        f"banned = {banned!r}",
        "def hit(name):",
        "    return any(name == b or name.startswith(b + '.') for b in banned)",
        "class Block:",
        "    def find_spec(self, name, path=None, target=None):",
        "        if hit(name):",
        "            raise ImportError('blocked for this test: ' + name)",
        "sys.meta_path.insert(0, Block())",
        f"sys.argv = ['sky_hypothetical.py', '--no-open', '--out', {str(out)!r}]",
        f"runpy.run_path({str(example)!r}, run_name='__main__')",
        "loaded = [m for m in sys.modules if hit(m)]",
        "assert not loaded, loaded",
    ])
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    block = _block(out)
    assert block["column_kind"] == "hypothetical"
    assert block["composition"]["mole_fractions"] == [["N2", 1.0]]
    assert block["hypothetical"]["surface_pressure_pa"] == 1.0e4
    assert block["hypothetical"]["radius_m"] == 6.371e6
    assert block["star"]["temperature_k"] == 3200.0
    # bolometric irradiance of the IAU Sun at 1 au: T^4 sin^2(a) conserved
    sin_a = math.sin(math.radians(block["star"]["angular_radius_deg"]))
    assert 3200.0 ** 4 * sin_a ** 2 == pytest.approx(
        5772.0 ** 4 * (695700.0 / 149597870.7) ** 2, rel=1e-12)


def _tree_digest(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_saved_run_sky_leaves_the_run_untouched(tmp_path):
    from test_atmosphere_view import write_pe_capsule
    from tropoi.representation.archive import open_simulation
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=2)
    before = _tree_digest(root)
    sim = open_simulation(root)
    spec = sky.SkySpec(AIR, observer_lat_deg=20.0, observer_lon_deg=10.0)
    block = _block(sim.sky(spec, tmp_path / "p.html", open_browser=False))
    assert block["column_kind"] == "saved_state"
    assert block["hypothetical"] is None
    assert block["context"]["time_index"] == 1          # the last by default
    assert block["context"]["lat_deg"] == 20.0
    assert block["consistency"]["status"] in ("consistent", "inconsistent")
    first = _block(sim.sky(spec, tmp_path / "p0.html", time=0,
                           open_browser=False))
    assert first["context"]["time_index"] == 0
    with pytest.raises(ValueError, match="needs observer_lat_deg"):
        sim.sky(sky.SkySpec(AIR), tmp_path / "q.html", open_browser=False)
    with pytest.raises(ValueError, match="own"):
        sim.sky(_hyp_spec(), tmp_path / "r.html", open_browser=False)
    with pytest.raises(IndexError):
        sim.sky(spec, tmp_path / "s.html", time=5, open_browser=False)
    with pytest.raises(ValueError, match="inside the run directory"):
        sim.sky(spec, root / "sky.html", open_browser=False)
    assert _tree_digest(root) == before
