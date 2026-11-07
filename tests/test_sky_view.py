"""Sky-from-the-surface prototype: optics, composition, column, reference.

CPU only. No assertion compares a sky colour or an optical depth with an
expected "look": the checks are internal physics and contracts --
analytic hydrostatics, exact inverses, refusals, the no-inference rule,
column counts against p_s R/(k_B g), numerical convergence, and the
viewer's additive behaviour (no sky block unless asked; runs untouched).
"""
from __future__ import annotations

import hashlib
import math
import pathlib

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


def test_every_species_records_its_sources_and_unverified_king_factor():
    for species in SPECIES.values():
        text = species.describe()
        assert "refractiveindex.info" in text and "Bates (1984)" in text
        assert species.king_verified is False
        assert "not yet checked" in text


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

def _table(column, **kw):
    o = sky.optics_block(AIR, Star.sun_at_1au())
    args = dict(radius_m=6.371e6, observer_z_m=2.0, star_elevation_deg=30.0,
                cross_section_m2=o["cross_section_m2"],
                irradiance=np.array(o["star_radiance"])
                * o["star_solid_angle_sr"],
                n_zenith=31, n_azimuth=13)
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
    o = sky.optics_block(AIR, Star.sun_at_1au())
    irr = np.array(o["star_radiance"]) * o["star_solid_angle_sr"]
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


# -- viewer contracts ---------------------------------------------------------------

def _preview():
    from tropoi.representation.visual.terrain import build_terrain_preview
    lm = np.zeros((11, 11), complex)
    lm[0, 0] = 500.0 * math.sqrt(4 * math.pi)
    return build_terrain_preview(lm, radius_m=3.3895e6, title="t")


def _hyp_spec(**kw):
    args = dict(composition=Composition.of(CO2=1.0), observer_lat_deg=10.0,
                hypothetical=sky.HypotheticalColumn(
                    surface_pressure_pa=610.0, temperature_k=210.0,
                    gravity=3.71, gas_constant="from_composition"))
    args.update(kw)
    return sky.SkySpec(**args)


def test_terrain_page_sky_is_additive(tmp_path):
    from tropoi.representation.visual.terrain import (globe_payload,
                                                      write_globe_html)
    preview = _preview()
    plain = write_globe_html(preview, tmp_path / "a.html")
    import json, re
    data = re.search(r'id="terrain-data" type="application/json">(.*?)</script>',
                     plain.read_text(encoding="utf-8"), re.S).group(1)
    assert "sky" not in json.loads(data)
    assert "sky" not in globe_payload(preview)
    page = write_globe_html(preview, tmp_path / "b.html", sky=_hyp_spec())
    assert '"column_kind":"hypothetical"' in page.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="no saved atmosphere"):
        write_globe_html(preview, tmp_path / "c.html",
                         sky=_hyp_spec(hypothetical=None))
    with pytest.raises(ColumnError, match="underground"):
        write_globe_html(preview, tmp_path / "d.html",
                         sky=_hyp_spec(altitude_m=100.0))


def test_hypothetical_block_contents():
    from tropoi.representation.visual.terrain import hypothetical_sky_block
    block = hypothetical_sky_block(_hyp_spec(pressure_pa=300.0), _preview())
    assert block["label"].startswith("Visual prototype")
    assert block["observer"]["mode"] == "pressure"
    assert block["observer"]["pressure_pa"] == pytest.approx(300.0)
    assert block["consistency"]["status"] == "consistent"   # R from it
    assert len(block["optics"]["cross_section_m2"]) == 18
    assert any("aerosols" in u for u in block["unsupported"])
    assert any("CIE" in s for s in block["sources"])


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
    path = sim.view(tmp_path / "p.html", open_browser=False,
                    resolution_deg=6.0, times="all", sky=spec)
    assert '"column_kind":"saved_state"' in path.read_text(encoding="utf-8")
    assert _tree_digest(root) == before
    with pytest.raises(ValueError, match="every native level"):
        sim.view(tmp_path / "q.html", open_browser=False, levels=[0, 1],
                 resolution_deg=6.0, sky=spec)
    with pytest.raises(ValueError, match="own column"):
        sim.view(tmp_path / "r.html", open_browser=False, resolution_deg=6.0,
                 sky=_hyp_spec())
