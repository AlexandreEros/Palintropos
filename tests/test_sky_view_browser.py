"""Standalone sky page in a real headless browser (Chrome, Chromium or Edge).

* The page's float64 JavaScript equals the NumPy reference
  (``representation.visual.sky``, ``spatial.optics``) to a relative 1e-9:
  the hypothetical column, the mixture cross section, the spectral
  radiance in sampled directions, the zenith and star-path columns and the
  optical-depth diagnostic -- for the acceptance case (0.1 bar N2, 3200 K
  star) and after the atmosphere, composition and star are edited in the
  page, and for a saved primitive-equation column.
* Exposure: AUTO follows the brightest sky luminance; FIXED holds its
  reference when pressure or illumination change, and the page says which.
* Optical depth: the caution and unreliable warnings appear when the
  zenith optical depth crosses the stated thresholds.
* Refusals (non-unit mole-fraction sums, underground observers) are
  messages, not exceptions; one request (the file itself), no console
  errors.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402
from test_atmosphere_view import write_pe_capsule  # noqa: E402

from tropoi.representation.visual import sky  # noqa: E402
from tropoi.spatial.optics import (COMPOSITION_PRESETS, Composition,  # noqa: E402
                                   Star)
from tropoi.spatial.optics.colour import bin_weights  # noqa: E402
from tropoi.spatial.optics.constants import LUMINOUS_EFFICACY  # noqa: E402

BROWSER = find_browser()
pytestmark = pytest.mark.skipif(BROWSER is None,
                                reason="needs Chrome, Chromium or Edge")
API = "window.__sky"
WL = sky.WAVELENGTHS_NM
DIRECTIONS = [(0, 0), (10, 5), (45, 0), (45, 30), (60, 60), (80, 20),
              (88, 0), (89, 45), (90, 10), (91, 0), (95, 30), (120, 60),
              (178, 2)]


def _assert_clean(page, url):
    assert page.requests == [url], page.requests
    assert page.console_errors == [], page.console_errors
    assert page.exceptions == [], page.exceptions


def _block(path) -> dict:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    raw = re.search(r'id="sky-data" type="application/json">(.*?)</script>',
                    text, re.S).group(1)
    return json.loads(raw.replace("\\u003c", "<").replace("\\u003e", ">"))


def _text(page, element_id):
    return page.eval(f"document.getElementById({element_id!r}).textContent")


def _parity(page, *, hypothetical: dict | None):
    """Page == NumPy reference for whatever the page now shows."""
    info = page.eval(f"{API}.info()")
    assert info["error"] is None, info["error"]
    col = page.eval(f"{API}.column()")
    mix = page.eval(f"{API}.mixture()")
    composition = Composition(tuple((k, x) for k, x in mix["rows"]))
    sigma = composition.cross_section(WL)
    np.testing.assert_allclose(mix["sigma"], sigma, rtol=1e-12)
    assert mix["R"] == pytest.approx(composition.gas_constant, rel=1e-14)
    if hypothetical is not None:
        ref_col = sky.HypotheticalColumn(**hypothetical).build(composition)
        for got, want in ((col["z"], ref_col.z_m), (col["lnp"], ref_col.ln_p),
                          (col["T"], ref_col.temperature_k)):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9)
    nodes = (col["z"], col["lnp"], col["T"])
    star = Star(info["star"]["T"], info["star"]["alpha"])
    kw = dict(radius_m=info["radius"], observer_z_m=info["place"]["z"],
              star_elevation_deg=info["star"]["el"])
    ref = sky.radiance_table(nodes, cross_section_m2=sigma,
                             irradiance=star.irradiance(WL), **kw)
    for iz, ia in DIRECTIONS:
        got = np.array(page.eval(f"{API}.spectral({iz}, {ia})"))
        np.testing.assert_allclose(got, ref["radiance"][iz, ia], rtol=1e-9,
                                   atol=0.0, err_msg=f"direction {iz}, {ia}")
    assert info["zenithColumn"] == pytest.approx(float(ref["column"][0, 0]),
                                                 rel=1e-9)
    star_col = sky.star_path_column(nodes, **kw)
    if star_col is None:
        assert info["starColumn"] is None and info["starXYZ"] is None
    else:
        assert info["starColumn"] == pytest.approx(star_col, rel=1e-9)
        w = bin_weights(sky.BIN_EDGES_NM[:-1], sky.BIN_EDGES_NM[1:])
        want = LUMINOUS_EFFICACY * (star.radiance(WL)
                                    * np.exp(-sigma * star_col)) @ w
        np.testing.assert_allclose(info["starXYZ"], want, rtol=1e-9)
    d = sky.optical_depth_diagnostic(sigma, float(ref["column"][0, 0]),
                                     star_col)
    assert info["diagnostic"]["level"] == d["level"]
    np.testing.assert_allclose(info["diagnostic"]["tau_zenith"],
                               d["tau_zenith"], rtol=1e-9)
    return info


@pytest.fixture(scope="module")
def acceptance_page(tmp_path_factory):
    """The documented example, run as a script: 0.1 bar N2, 3200 K star."""
    out = tmp_path_factory.mktemp("sky") / "n2.html"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "sky_hypothetical.py"),
         "--no-open", "--out", str(out)], capture_output=True, text=True,
        timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def test_acceptance_case_parity_and_editing(acceptance_page):
    block = _block(acceptance_page)
    hyp = {k: block["hypothetical"][k] for k in (
        "surface_pressure_pa", "temperature_k", "gravity", "gas_constant",
        "radius_m", "lapse_rate_k_per_m", "temperature_floor_k")}
    with OfflinePage(BROWSER) as page:
        url = page.open(acceptance_page)
        # opens straight at the observer's viewpoint: no mode switch
        assert page.eval("document.getElementById('sky').hidden") is False
        assert "HYPOTHETICAL" in _text(page, "stageLabel")
        assert "Visual prototype" in _text(page, "stageLabel")
        info = _parity(page, hypothetical=hyp)
        assert info["diagnostic"]["level"] == "ok"      # tau(410) ~ 0.03
        assert info["place"]["z"] == pytest.approx(2.0)

        # atmosphere edited in the page: thicker, warmer, a lapse rate, a
        # different planet, a fixed gas constant
        edit = dict(hyp, surface_pressure_pa=6.0e4, temperature_k=280.0,
                    lapse_rate_k_per_m=0.004, temperature_floor_k=200.0,
                    gravity=7.5, radius_m=4.0e6, gas_constant=290.0)
        page.eval(f"{API}.set({{atmosphere: {json.dumps(edit)}}})")
        _parity(page, hypothetical=edit)
        assert "DISAGREE" in _text(page, "compNote")    # 290 vs N2's R*/M

        # composition edited: a four-gas mixture (sums to 1 exactly)
        page.eval(f"{API}.set({{composition: {{N2: 0.5, O2: 0.25, Ar: 0.125,"
                  " CO2: 0.125}})")
        _parity(page, hypothetical=edit)
        assert "edited in the page" in _text(page, "compNote")

        # star edited: hotter and larger, lower in the sky
        page.eval(f"{API}.set({{star: {{T: 9000, alpha: 1.5, el: 8}}}})")
        info = _parity(page, hypothetical=edit)
        assert info["star"]["label"] == "blackbody star set in the page"

        # observer by pressure
        page.eval(f"{API}.set({{observer: {{mode: 'pressure', "
                  "pressure_pa: 30000}})")
        info = _parity(page, hypothetical=edit)
        assert info["place"]["p"] == pytest.approx(30000.0, rel=1e-12)
        _assert_clean(page, url)


def test_exposure_modes_auto_and_fixed(acceptance_page):
    with OfflinePage(BROWSER) as page:
        url = page.open(acceptance_page)
        info = page.eval(f"{API}.info()")
        assert info["exposure"]["mode"] == "auto"
        assert info["white"] == info["brightest"]
        assert "AUTO" in _text(page, "expNote")
        # thicker atmosphere: AUTO rescales the brightness change away
        thick = page.eval(f"{API}.set({{atmosphere: "
                          "{surface_pressure_pa: 4e4}})")
        assert thick["white"] == thick["brightest"] != info["brightest"]
        # switch to FIXED through the control: the reference starts at the
        # brightest luminance now, then stays put
        page.eval("const m = document.getElementById('expMode');"
                  " m.value = 'fixed'; m.dispatchEvent(new Event('change'))")
        fixed = page.eval(f"{API}.info()")
        assert fixed["exposure"]["mode"] == "fixed"
        assert fixed["white"] == pytest.approx(thick["brightest"], rel=1e-3)
        assert "FIXED" in _text(page, "expNote")
        assert "meaningful" in _text(page, "expNote")
        thin = page.eval(f"{API}.set({{atmosphere: "
                         "{surface_pressure_pa: 1e4}})")
        assert thin["white"] == fixed["white"]          # held
        assert thin["brightest"] < 0.5 * fixed["white"]  # so the sky darkens
        dim = page.eval(f"{API}.set({{star: {{alpha: 0.3}}}})")
        assert dim["white"] == fixed["white"]
        assert dim["brightest"] < thin["brightest"]
        # the clipping figures describe the image just drawn
        page.eval("const e = document.getElementById('ev'); e.value = 9;"
                  " e.dispatchEvent(new Event('input'))")
        clip = page.eval(f"{API}.info()")["clip"]
        assert f"{100 * clip['over']:.1f} % over-exposed" in _text(
            page, "clipNote")
        _assert_clean(page, url)


def test_optical_depth_warnings_and_refusals(acceptance_page):
    with OfflinePage(BROWSER) as page:
        url = page.open(acceptance_page)
        assert page.eval("document.getElementById('stageWarn').hidden") is True
        # every control before anything else: no exceptions
        for cid, event in (("starEl", "input"), ("starAz", "input"),
                           ("fov", "input"), ("ev", "input"),
                           ("expRef", "change"), ("altVal", "change"),
                           ("atmPs", "change"), ("comp_O2", "change"),
                           ("starT", "change")):
            page.eval(f"document.getElementById({cid!r})"
                      f".dispatchEvent(new Event({event!r}))")
        page.eval(f"{API}.set({{}})")
        assert page.exceptions == [], page.exceptions
        # Earth-like column: zenith tau(410 nm) ~ 0.32 -> caution
        air = dict(COMPOSITION_PRESETS["earth_dry_air_illustrative"]
                   .mole_fractions)
        info = page.eval(f"{API}.set({{composition: {json.dumps(air)}, "
                         "atmosphere: {surface_pressure_pa: 101325}})")
        assert info["diagnostic"]["level"] == "caution"
        assert page.eval("document.getElementById('stageWarn').hidden") is False
        assert "Caution" in _text(page, "tauNote")
        # a thick CO2 column -> unreliable, said on the sky itself
        info = page.eval(f"{API}.set({{composition: {{CO2: 1}}, atmosphere: "
                         "{surface_pressure_pa: 2e6}})")
        assert info["diagnostic"]["level"] == "unreliable"
        assert info["diagnostic"]["peak_zenith"] >= sky.TAU_UNRELIABLE
        assert "UNRELIABLE" in _text(page, "stageWarn")
        # never renormalized: a sum of 0.9 is refused, the explicit
        # "set to the remainder" action repairs it
        info = page.eval(f"{API}.set({{composition: {{N2: 0.6, O2: 0.3}}}})")
        assert "never renormalized" in info["error"]
        assert "never renormalized" in _text(page, "obsNote")
        page.eval("const w = document.getElementById('compFillWhich');"
                  " w.value = 'N2'; document.getElementById('compFill')"
                  ".click()")
        info = page.eval(f"{API}.info()")
        assert info["error"] is None
        assert page.eval(f"{API}.mixture()")["rows"] == [["N2", 0.7],
                                                       ["O2", 0.3]]
        # underground is a message
        info = page.eval(f"{API}.set({{observer: {{mode: 'pressure', "
                         "pressure_pa: 5e6}})")
        assert "underground" in info["error"]
        _assert_clean(page, url)


def test_saved_state_sky_page(tmp_path):
    from tropoi.representation.archive import open_simulation
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=2)
    spec = sky.SkySpec(COMPOSITION_PRESETS["earth_dry_air_illustrative"],
                       observer_lat_deg=20.0, observer_lon_deg=10.0)
    path = open_simulation(root).sky(spec, tmp_path / "saved.html",
                                     open_browser=False)
    block = _block(path)
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        assert page.eval("document.getElementById('atmHyp').hidden") is True
        # no presets on a fixed column, and the address carries no atmosphere
        assert block["presets"] == []
        assert page.eval("document.getElementById('presetSection').hidden")
        assert not re.search(r"(^|&)(ps|T|g|radius)=",
                             page.eval(f"{API}.state()"))
        assert "saved state" in _text(page, "stageLabel")
        assert "cannot be edited" in _text(page, "atmNote")
        info = _parity(page, hypothetical=None)
        col = page.eval(f"{API}.column()")
        np.testing.assert_array_equal(col["z"], block["column"]["z_m"])
        assert info["place"]["floor"] == block["terrain_m"]
        # composition and star stay view settings on a saved column
        page.eval(f"{API}.set({{composition: {{N2: 1}}, star: {{T: 4000}}}})")
        _parity(page, hypothetical=None)
        assert "r_dry" in _text(page, "compNote")
        with pytest.raises(RuntimeError, match="cannot be edited"):
            page.eval(f"{API}.set({{atmosphere: {{temperature_k: 300}}}})")
        _assert_clean(page, url)


# -- the standalone page, presets, and settings kept in the address -------------

STANDALONE = REPO_ROOT / "docs" / "sky.html"


def _settings(info) -> dict:
    """What the address and a saved page must reproduce exactly."""
    return {k: info[k] for k in ("atmosphere", "composition", "observer",
                                 "exposure", "preset")} | {
        "star": {k: info["star"][k] for k in ("T", "alpha", "az", "el")},
        "look": {k: round(info["look"][k], 2) for k in ("az", "el", "fov")}}


def test_standalone_page_opens_without_python_and_offers_presets():
    """docs/sky.html, opened as a file: the Earth preset, then each preset,
    each equal to the NumPy reference of that preset's sky."""
    with OfflinePage(BROWSER) as page:
        url = page.open(STANDALONE)
        info = page.eval(f"{API}.info()")
        assert info["error"] is None and info["preset"] == "earth"
        assert page.eval("document.getElementById('preset').value") == "earth"
        assert page.eval(f"{API}.presets()") == list(sky.SKY_PRESETS)
        for key, preset in sky.SKY_PRESETS.items():
            info = page.eval(f"{API}.preset({key!r})")
            assert info["preset"] == key
            assert info["compLabel"] == preset.spec.composition.label
            assert info["observer"]["mode"] == "ground"
            assert info["exposure"] == {"mode": "auto", "ev": 0,
                                        "ref": None}
            _parity(page, hypothetical={
                k: v for k, v in preset.spec.hypothetical.to_dict().items()
                if k not in ("top_ratio", "nodes")})
            assert preset.description[:20] in _text(page, "presetNote")
        # an edit leaves the preset: the menu says custom
        info = page.eval(f"{API}.set({{atmosphere: {{temperature_k: 260}}}})")
        assert info["preset"] is None
        assert page.eval("document.getElementById('preset').value") == ""
        assert "Custom settings" in _text(page, "presetNote")
        _assert_clean(page, url)


def test_settings_survive_the_address_and_a_saved_page(tmp_path):
    edit = ("{atmosphere: {surface_pressure_pa: 54321.5, temperature_k: "
            "271.25, lapse_rate_k_per_m: 0.004, temperature_floor_k: 201.5, "
            "gas_constant: 290.5, gravity: 7.125, radius_m: 4.5e6}, "
            "composition: {N2: 0.5, O2: 0.25, Ar: 0.125, CO2: 0.125}, "
            "star: {T: 6100.5, alpha: 0.41, az: 200, el: 12.5}, "
            "observer: {mode: 'pressure', pressure_pa: 30000}, "
            "look: {az: 201.234, el: 10.5, fov: 80}, "
            "exposure: {mode: 'fixed', ref: 1234.5, ev: -1.25}}")
    with OfflinePage(BROWSER) as page:
        url = page.open(STANDALONE)
        want = _settings(page.eval(f"{API}.set({edit})"))
        assert want["preset"] is None
        fragment = page.eval(f"{API}.flush()")
        assert page.eval("location.hash") == fragment
        assert page.eval("document.getElementById('shareUrl').value"
                         ).endswith(fragment)
        html = page.eval(f"{API}.savedPage()")
        _assert_clean(page, url)
    # a reload (or a bookmark) of that address reopens the same sky
    copy = tmp_path / "copy.html"
    copy.write_text(STANDALONE.read_text(encoding="utf-8"), encoding="utf-8")
    with OfflinePage(BROWSER) as page:
        url = page.open(copy, fragment=fragment[1:])
        info = page.eval(f"{API}.info()")
        assert info["error"] is None
        assert _settings(info) == want
        assert "from the page address" in info["compLabel"]
        assert "from the page address" in info["star"]["label"]
        _parity(page, hypothetical={
            "surface_pressure_pa": 54321.5, "temperature_k": 271.25,
            "lapse_rate_k_per_m": 0.004, "temperature_floor_k": 201.5,
            "gas_constant": 290.5, "gravity": 7.125, "radius_m": 4.5e6})
        # Reset: back to what the page was written with
        assert page.eval(f"{API}.reset()")["preset"] == "earth"
        _assert_clean(page, url)
    # "Save as page": one file that opens with those settings, no address
    saved = tmp_path / "saved.html"
    saved.write_text(html, encoding="utf-8")
    with OfflinePage(BROWSER) as page:
        url = page.open(saved)
        info = page.eval(f"{API}.info()")
        assert info["error"] is None and _settings(info) == want
        assert "from the saved page" in info["compLabel"]
        assert page.eval(f"{API}.reset()")["preset"] == "earth"
        _assert_clean(page, url)


def test_a_bad_address_is_reported_not_obeyed():
    with OfflinePage(BROWSER) as page:
        url = page.open(STANDALONE, fragment="ps=abc&foo=1&N2=1&obs=moon")
        info = page.eval(f"{API}.info()")
        assert info["error"] is None
        assert info["atmosphere"]["surface_pressure_pa"] == 101325.0
        assert info["observer"]["mode"] == "ground"
        assert info["composition"] == [["N2", 1], ["O2", 0], ["Ar", 0],
                                       ["CO2", 0]]
        note = _text(page, "shareNote")
        for key in ("ps", "foo", "obs"):
            assert key in note
        # a preset's exact numbers in the address bring back its labels
        page.eval(f"{API}.preset('mars')")
        mars = page.eval(f"{API}.state()")
        page.eval(f"{API}.reset()")
        info = page.eval(f"{API}.restore({mars!r}) && {API}.info()")
        assert info["preset"] == "mars"
        assert info["compLabel"] == sky.SKY_PRESETS["mars"].spec \
            .composition.label
        _assert_clean(page, url)
