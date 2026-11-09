"""The isobaric-surface export of ``Simulation.view`` (issue #13, first slice).

Payload: the ``surface`` option of ``representation.visual.atmosphere``
(fields, version, masks, colour limits over valid nodes only, estimate
before reconstruction, strict ``times="all"``, exclusivity with ``levels``),
and the native-level payload left as it was. Page: the written file opened
from ``file://`` in headless Chrome with networking blocked (skipped without
a browser): no errors, one request, distinct frames, triangles dropped at
masked nodes, hover naming the mask reason, streamlines outside masked cells.
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropoi.representation.diagnostics.pe_surfaces import (  # noqa: E402
    MASK_BELOW_BOTTOM_LEVEL, MASK_BELOW_SURFACE, MASK_VALID)
from tropoi.representation.visual.atmosphere import (  # noqa: E402
    FIELDS, SURFACE_FIELDS, AtmosphereViewWarning, PayloadTooLargeError,
    atmosphere_payload, estimate_payload_bytes)
from test_atmosphere_view import (GRAVITY, L_MAX, NLEV, R_DRY,  # noqa: E402
                                  SQ4PI, _band_limited_terrain,
                                  write_pe_capsule)
from test_pe_pressure_surface import (_balanced_ln_ps,  # noqa: E402
                                      _isothermal_frame)

API = "window.__terrainGlobe.atmosphere"


def _storage(root):
    from tropoi.representation.archive import open_simulation
    sim = open_simulation(root)
    return sim, (getattr(sim, "_storage", None) or sim.storage)


def _u16(block: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(block), dtype="<u2")


def _u8(block: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(block), dtype="u1")


def write_holed_capsule(root, *, frames=3, p_ref=1.0e5, warming=3.0):
    """Isothermal states over tall terrain whose 930 hPa isobar is under
    ground on the peaks, with a warming of ``warming`` K per saved state so
    the isobar rises between times."""
    root, coeffs, lm = write_pe_capsule(root, frames=frames)
    terrain = _band_limited_terrain(scale=6000.0)
    ln_ps = _balanced_ln_ps(terrain, 230.0, p_ref)      # the same p_s always
    for i in range(frames):
        coeffs[i] = _isothermal_frame(230.0 + warming * i, ln_ps)
    np.save(root / "pe_coeffs.npy", coeffs)
    # the prepared terrain must be the one the balance used
    from tropoi.representation.archive.prepared_terrain import (
        SavedTerrain, save_prepared_terrain)
    from tropoi.spatial.specification import (DiscretizationSpec, PlanetSpec,
                                              TopographySpec)
    import shutil
    shutil.rmtree(root / "terrain")
    saved = SavedTerrain(
        PlanetSpec.earth_like(), DiscretizationSpec.gauss_latlon(L_MAX, 16, 32),
        TopographySpec.imported({"source_name": "fixture.nc",
                                 "source_sha256": "b" * 64,
                                 "format": "netcdf"}),
        terrain, l_cut=6, label="tall fixture terrain")
    save_prepared_terrain(saved, root / "terrain")
    for name in ("manifest.json", "config.json"):
        doc = json.loads((root / name).read_text(encoding="utf-8"))
        cfg = doc["run_config"] if name == "manifest.json" else doc
        cfg["terrain_digest"] = saved.digest
        (root / name).write_text(json.dumps(doc), encoding="utf-8")
    return root, coeffs, terrain


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_surface_payload_carries_masks_fields_and_version(tmp_path):
    root, _, _ = write_holed_capsule(tmp_path / "cap")
    _, storage = _storage(root)
    page, report = atmosphere_payload(storage, surface=("pressure", 93_000.0),
                                      times="all", resolution_deg=6.0)
    A = page["atmosphere"]
    assert A["version"] == 2
    assert [f["id"] for f in A["fields"]] == list(SURFACE_FIELDS)
    assert "pressure" not in {f["id"] for f in A["fields"]}
    assert "zeta" not in {f["id"] for f in A["fields"]}
    assert A["surface"] == {
        "kind": "pressure", "value": 93_000.0, "units": "Pa",
        "label": "930 hPa", "level_sigma": "hydrostatic",
        "level_positions": [pytest.approx(s) for s in
                            report["context"].sigma.hydrostatic_full_levels],
        "mask_labels": A["surface"]["mask_labels"],
        "valid_counts": A["surface"]["valid_counts"]}
    assert A["levels"] == [{"index": None, "number": None, "sigma": None,
                            "label": "930 hPa"}]
    n = A["mesh"]["nlat"] * A["mesh"]["nlon"]
    mask = _u8(A["mask"]["data"]).reshape(len(A["times"]), n)
    for t, counts in enumerate(report["mask_counts"]):
        assert counts[MASK_VALID] > 0 and counts[MASK_BELOW_SURFACE] > 0
        assert counts[MASK_BELOW_BOTTOM_LEVEL] > 0
        for code, number in counts.items():
            assert int((mask[t] == code).sum()) == number
        assert A["surface"]["valid_counts"][t] == {str(k): v for k, v in
                                                   counts.items()}
    # the masks do not change with the uniform warming; the isobar rises
    assert (mask[0] == mask[-1]).all()
    z = A["height"]
    lo0, lo1 = z["bounds"][0][0], z["bounds"][-1][0]
    hi0, hi1 = z["bounds"][0][1], z["bounds"][-1][1]
    assert lo1 > lo0 and hi1 > hi0
    assert report["levels"] == [] and report["surface"].pressure_pa == 93e3
    assert any("not solver state" in line for line in A["interpretation"])
    assert any("vorticity and divergence" in line
               for line in A["interpretation"])


def test_colour_limits_and_quantization_use_valid_nodes_only(tmp_path):
    root, _, _ = write_holed_capsule(tmp_path / "cap", frames=2)
    _, storage = _storage(root)
    from tropoi.representation.diagnostics.pe_surfaces import pressure_surface
    page, report = atmosphere_payload(storage, surface=("pressure", 93_000.0),
                                      times="all", resolution_deg=6.0)
    A = page["atmosphere"]
    n = A["mesh"]["nlat"] * A["mesh"]["nlon"]
    lat = np.linspace(-90, 90, A["mesh"]["nlat"])
    lon = np.linspace(-180, 180, A["mesh"]["nlon"])
    context = report["context"]
    fields = {f["id"]: f for f in A["fields"]}
    for t, index in enumerate(report["times"]):
        s = pressure_surface(storage.frame(index), context, lat, lon,
                             ("pressure", 93_000.0))
        ok = s.valid.ravel()
        values = {"temperature": s["temperature"].ravel(),
                  "geopotential_height": s["geopotential_height"].ravel(),
                  "wind_speed": np.hypot(s["u"], s["v"]).ravel()}
        for name, exact in values.items():
            f = fields[name]
            lo, hi = f["bounds"][t]
            assert lo == pytest.approx(exact[ok].min()) \
                and hi == pytest.approx(exact[ok].max())
            q = _u16(f["data"])[t * n:(t + 1) * n]
            back = lo + q * (hi - lo) / 65535.0
            assert np.abs(back[ok] - exact[ok]).max() <= (hi - lo) / 131070 \
                + 1e-9
            assert (q[~ok] == 0).all()            # placeholders, never read
            whole = f["range"]
            assert whole[0] <= lo and whole[1] >= hi
        # the range is set by valid nodes only: the under-ground placeholder
        # 0 K would otherwise pull the temperature range down
        assert fields["temperature"]["range"][0] > 200.0


def test_estimate_precedes_reconstruction_and_bounds_the_file(tmp_path):
    root, _, _ = write_holed_capsule(tmp_path / "cap", frames=3)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(storage, surface=("pressure", 93_000.0),
                                      times="all", resolution_deg=6.0)
    nlat, nlon = page["atmosphere"]["mesh"]["nlat"], \
        page["atmosphere"]["mesh"]["nlon"]
    per_time = estimate_payload_bytes(1, 1, nlat, nlon, n_fields=3,
                                      mask=True) - 200_000
    assert report["estimate_bytes"] >= per_time * 3
    path = sim.view(tmp_path / "holes.html", surface=("pressure", 93_000.0),
                    times="all", resolution_deg=6.0, open_browser=False)
    assert path.stat().st_size <= report["estimate_bytes"] + 300_000
    # the native estimate is what it was
    assert estimate_payload_bytes(2, 3, 31, 61) == estimate_payload_bytes(
        2, 3, 31, 61, n_fields=len(FIELDS), mask=False)


def test_all_times_never_subsamples_and_the_refusal_names_alternatives(
        tmp_path):
    root, _, _ = write_holed_capsule(tmp_path / "cap", frames=3)
    _, storage = _storage(root)
    with pytest.raises(PayloadTooLargeError, match="coarser resolution_deg"):
        atmosphere_payload(storage, surface=("pressure", 93_000.0),
                           times="all", resolution_deg=6.0, max_mb=0.2)
    # the automatic selection still subsamples with a warning, as for levels:
    # a budget with room for two states and a half takes the endpoints
    _, one = atmosphere_payload(storage, surface=("pressure", 93_000.0),
                                times=[0], resolution_deg=6.0)
    _, three = atmosphere_payload(storage, surface=("pressure", 93_000.0),
                                  times="all", resolution_deg=6.0)
    per_time = (three["estimate_bytes"] - one["estimate_bytes"]) / 2
    budget = one["estimate_bytes"] + 1.5 * per_time
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        page, report = atmosphere_payload(
            storage, surface=("pressure", 93_000.0), resolution_deg=6.0,
            max_mb=budget / 1e6)
    assert any(issubclass(w.category, AtmosphereViewWarning) for w in caught)
    assert report["times"] == [0, 2]


def test_surface_options_are_validated(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    sim, storage = _storage(root)
    with pytest.raises(ValueError, match="exclusive"):
        atmosphere_payload(storage, surface=("pressure", 7e4), levels=[0])
    with pytest.raises(ValueError, match="level= selects a native level"):
        atmosphere_payload(storage, surface=("pressure", 7e4), level=1)
    with pytest.raises(ValueError, match="field must be one of"):
        atmosphere_payload(storage, surface=("pressure", 7e4), field="zeta")
    with pytest.raises(ValueError, match="field must be one of"):
        atmosphere_payload(storage, field="wind_speed")
    # heights are exported now (test_interactive_height_view); a bad one
    # is refused by the specification
    with pytest.raises(ValueError, match="finite"):
        atmosphere_payload(storage, surface=("geopotential_height",
                                             float("nan")))
    with pytest.raises(TypeError, match="'pressure'"):
        atmosphere_payload(storage, surface=70000.0)
    with pytest.raises(ValueError, match="level_sigma must be one of"):
        atmosphere_payload(storage, surface=("pressure", 7e4),
                           level_sigma="middle")
    from tropoi.representation.diagnostics.pe_surfaces import (
        SurfaceUnavailableError)
    with pytest.raises(SurfaceUnavailableError, match="no valid node"):
        atmosphere_payload(storage, surface=("pressure", 5e5))
    # the one-snapshot view goes through the same option
    path = sim[0].view(tmp_path / "one.html", surface=("pressure", 7e4),
                       resolution_deg=6.0, open_browser=False)
    assert path.is_file()


def test_native_level_payload_is_unchanged_by_the_surface_option(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=2)
    _, storage = _storage(root)
    page, report = atmosphere_payload(storage, times="all", resolution_deg=6.0)
    A = page["atmosphere"]
    assert A["version"] == 1
    assert "surface" not in A and "mask" not in A
    assert [f["id"] for f in A["fields"]] == list(FIELDS)
    assert "surface" not in report
    # the levels keep their native description
    assert A["levels"][0]["sigma"] == pytest.approx(
        report["context"].full_levels[0])


def test_default_output_names_the_isobar(tmp_path):
    from tropoi.representation.visual.atmosphere import _default_path
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    _, native = atmosphere_payload(storage, resolution_deg=6.0)
    _, derived = atmosphere_payload(storage, surface=("pressure", 7e4),
                                    resolution_deg=6.0)
    a, b = _default_path(storage, native), _default_path(storage, derived)
    assert "-atmosphere-" in a.name and "-isobar-" in b.name
    assert a != b


# ---------------------------------------------------------------------------
# The page, in a browser, offline
# ---------------------------------------------------------------------------

from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402

BROWSER = find_browser()
needs_browser = pytest.mark.skipif(BROWSER is None,
                                   reason="needs Chrome, Chromium or Edge")


@pytest.fixture(scope="module")
def holed_page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("isobar")
    root, _, _ = write_holed_capsule(tmp / "cap", frames=3)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(storage, surface=("pressure", 93_000.0),
                                      times="all", resolution_deg=6.0)
    path = sim.view(tmp / "holes.html", surface=("pressure", 93_000.0),
                    times="all", resolution_deg=6.0, open_browser=False)
    return path, page, report


@needs_browser
def test_page_drops_masked_triangles_and_names_the_reason(holed_page):
    path, payload, report = holed_page
    A = payload["atmosphere"]
    nlat, nlon = A["mesh"]["nlat"], A["mesh"]["nlon"]
    n = nlat * nlon
    mask = _u8(A["mask"]["data"]).reshape(len(A["times"]), nlat, nlon)
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        page.wait_for(f"{API} && document.getElementById('timeLabel')"
                      ".textContent")
        assert page.eval("window.__terrainGlobe.glOK")
        assert page.eval("document.getElementById('levelSlider').hidden")
        assert page.eval("document.getElementById('levelTitle').textContent"
                         ) == "Surface"
        assert "930 hPa" in page.eval(
            "document.getElementById('levelLabel').textContent")
        digests = []
        for t in range(len(A["times"])):
            page.eval(f"{API}.setTime({t})")
            counts = page.eval(f"{API}.maskCounts()")
            m = mask[t]
            assert counts["counts"] == {str(k): int((m == k).sum())
                                        for k in range(4)}
            # exactly the triangles with no masked vertex survive
            quads = m[:-1, :-1] | m[1:, :-1] | m[:-1, 1:] | m[1:, 1:]
            masked_tri = 0
            for j in range(nlat - 1):
                for i in range(nlon - 1):
                    a, b, c, d = m[j, i], m[j, i + 1], m[j + 1, i], m[j + 1,
                                                                      i + 1]
                    masked_tri += bool(a or b or d) + bool(a or d or c)
            assert counts["triangles"] == counts["trianglesAll"] - masked_tri
            assert counts["trianglesAll"] == 2 * (nlat - 1) * (nlon - 1)
            del quads
            digests.append(page.eval(f"{API}.digest()"))
            note = page.eval("document.getElementById('atmNote').textContent")
            assert "nodes valid" in note and "under ground" in note
        assert len(set(digests)) == len(A["times"])      # frames differ
        # hover: a masked node says why, a valid one gives values
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(-180, 180, nlon)
        j, i = np.argwhere(mask[-1] == MASK_BELOW_SURFACE)[0]
        under = page.eval(f"{API}.sample({lat[j]}, {lon[i]})")
        assert under["mask"] == MASK_BELOW_SURFACE
        assert "below the surface" in under["readout"]
        j, i = np.argwhere(mask[-1] == MASK_BELOW_BOTTOM_LEVEL)[0]
        gap = page.eval(f"{API}.sample({lat[j]}, {lon[i]})")
        assert "bottom full level" in gap["readout"]
        j, i = np.argwhere(mask[-1] == MASK_VALID)[0]
        good = page.eval(f"{API}.sample({lat[j]}, {lon[i]})")
        assert good["mask"] == 0 and "interpolated" in good["readout"]
        # between a valid node and a masked neighbour the bilinear sample
        # would mix in a placeholder: the readout must refuse it even where
        # the nearest node is valid
        m = mask[-1]
        pair = None
        for (jj, ii) in np.argwhere(m == MASK_VALID):
            if ii + 1 < nlon and m[jj, ii + 1] == MASK_BELOW_SURFACE:
                pair = (jj, ii)
                break
        assert pair is not None
        jj, ii = pair
        near = page.eval(f"{API}.sample({lat[jj]}, {lon[ii] + 0.4 * d})")
        assert near["mask"] == MASK_BELOW_SURFACE
        assert near["readout"].startswith("no sample")
        # streamline points never fall in a cell with a masked node
        pts = np.frombuffer(base64.b64decode(A["streamlines"]["points"]),
                            dtype="<i2").reshape(-1, 2) / 100.0
        starts = np.frombuffer(base64.b64decode(
            A["streamlines"]["line_start"]), dtype="<u4")
        slices = A["streamlines"]["slice_line_start"]
        d = A["mesh"]["dlat"]
        for t in range(len(A["times"])):
            cells = (mask[t][:-1, :-1] | mask[t][1:, :-1]
                     | mask[t][:-1, 1:] | mask[t][1:, 1:]) != 0
            for line in range(slices[t], slices[t + 1]):
                seg = pts[starts[line]:starts[line + 1]]
                jj = np.clip(np.floor((seg[:, 0] + 90.0) / d).astype(int), 0,
                             nlat - 2)
                ii = np.floor(((seg[:, 1] + 180.0) % 360.0) / d).astype(int)
                ii = np.clip(ii, 0, nlon - 2)
                assert not cells[jj, ii].any()
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []
        assert n == nlat * nlon
