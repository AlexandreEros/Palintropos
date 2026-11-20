"""Scientific and archive-boundary checks for the Viewer 2 frame provider."""
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.archive.capsule import open_storage
from tropoi.representation.archive.viewer_manifest import viewer_manifest
from tropoi.representation.diagnostics.pe_surfaces import pressure_surface
from tropoi.representation.visual.viewer_fields import FrameProvider, _cone_coefficients, _slab_map
from tropoi.spatial.transforms.host_synthesis import synthesize_latlon


from test_viewer_manifest import BVE as BVE_ID, MARS as MARS_ID, SWE as SWE_ID

ROOT = Path(__file__).resolve().parents[1]
W5, MARS, BVE = [ROOT / key for key in (SWE_ID, MARS_ID, BVE_ID)]


def test_bve_nondivergent_zonal_flow(tmp_path):
    m = viewer_manifest(BVE)
    n = m["axes"]["support"]["l_max"] + 1
    coeff = np.zeros((1, n, n), dtype=complex)
    speed = 13.0
    radius = m["geometry"]["dynamical_radius_m"]
    # Solid-body rotation u=U cos(lat), zeta=2U sin(lat)/a.
    coeff[0, 1, 0] = 2 * speed / radius * np.sqrt(4*np.pi / 3)
    np.save(tmp_path / m["resources"]["state"]["path"], coeff)
    result = FrameProvider(tmp_path, m).slice(0, "u")
    expected = speed * np.cos(np.radians(result["lat"]))[:, None]
    # The geographic basis at a pole is evaluated just off the pole.
    np.testing.assert_allclose(result["values"][1:-1] - expected[1:-1], 0, atol=1e-11)
    np.testing.assert_array_equal(result["values"], result["u"])
    np.testing.assert_allclose(result["v"], 0, atol=1e-12)
    assert not result["mask"].any() and not result["height"].any()


def test_w5_solver_projection_on_native_quadrature():
    m = viewer_manifest(W5)
    coeff = _cone_coefficients(m["physics"], m["geometry"], m["axes"]["support"]["l_max"])
    with np.load(ROOT / "docs/validation/williamson_5/aeolus_w5_t63.npz") as reference:
        lat, lon = reference["latitude"], reference["longitude"]
        terrain = synthesize_latlon(coeff, lat, lon)
        np.testing.assert_allclose(terrain, reference["topography_height_bandlimited"], rtol=0, atol=1e-9)
        # Compare the actual saved final state against an independent export.
        state = open_storage(W5).frame(3)
        height = terrain + m["physics"]["mean_depth_m"] + synthesize_latlon(state[2], lat, lon) / m["physics"]["gravity"]
        np.testing.assert_allclose(height, reference["free_surface_height"][-1], rtol=0, atol=1e-8)


@pytest.mark.parametrize("flat", [False, True])
def test_constant_free_surface_and_below_terrain(tmp_path, flat):
    m = viewer_manifest(W5)
    p = m["physics"]
    gravity = p["gravity"]
    coeff = _cone_coefficients(p, m["geometry"], 63)
    if flat:
        p["topography"] = "flat"
        coeff[:] = 0
    # A level lake at 500 m elevation. High mountain cells are dry.
    state = np.zeros((1, 3, 64, 64), complex)
    state[0, 2] = -gravity * coeff
    state[0, 2, 0, 0] += (500 - p["mean_depth_m"]) * gravity * np.sqrt(4*np.pi)
    np.save(tmp_path / m["resources"]["state"]["path"], state)
    provider = FrameProvider(tmp_path, m)
    for field in ("zeta", "phi", "layer_depth", "free_surface_height", "free_surface_geopotential",
                  "terrain", "u", "v"):
        selection = provider.slice(0, field)
        np.testing.assert_allclose(selection["height"], 500, atol=2e-10, rtol=0)
        expected_mask = selection["terrain"] >= 500
        assert expected_mask.any() != flat
        np.testing.assert_array_equal(selection["mask"], expected_mask)
        if field == "layer_depth":
            np.testing.assert_allclose(selection["values"] + selection["terrain"], 500, atol=2e-10)
        elif field == "free_surface_geopotential":
            np.testing.assert_allclose(selection["values"], gravity * 500, atol=2e-9)
        elif field == "terrain" and flat:
            assert not selection["values"].any()


@pytest.mark.parametrize("coordinate,value", [("pressure", 500), ("sigma", .625),
                                               ("geopotential_height", 3500)])
def test_mars_matches_atmosphere_evaluator(coordinate, value):
    manifest = viewer_manifest(MARS)
    provider = FrameProvider(MARS, manifest)
    result = provider.slice(8, "temperature", coordinate, value)
    storage = open_storage(MARS)
    expected = pressure_surface(storage.frame(8), storage.pe_context(), result["lat"],
                                result["lon"], (coordinate, value))
    np.testing.assert_array_equal(result["mask"], expected.mask)
    for key, name in (("values", "temperature"), ("height", "geopotential_height"),
                      ("u", "u"), ("v", "v")):
        np.testing.assert_allclose(result[key], expected[name], atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(result["terrain"], expected.surface_height, atol=1e-9, rtol=0)
    if coordinate != "sigma":
        assert result["mask"].any() and (result["mask"] == 0).any()
        with pytest.raises(ValueError):
            provider.slice(8, "zeta", coordinate, value)
    else:
        native = provider.slice(8, "temperature", value=2)
        np.testing.assert_allclose(result["values"], native["values"], atol=1e-10)
        np.testing.assert_allclose(result["height"], native["height"], atol=1e-8)


def test_surface_pressure_display_is_exponential_in_pascals():
    provider = FrameProvider(MARS, viewer_manifest(MARS))
    selection = provider.slice(8, "ln_ps")
    saved = open_storage(MARS).frame(8)
    expected = np.exp(synthesize_latlon(saved[-1], selection["lat"], selection["lon"]))
    np.testing.assert_array_equal(expected, selection["values"])
    np.testing.assert_array_equal(selection["height"], selection["terrain"])
    assert "u" not in selection


@pytest.mark.parametrize("coordinate,value", [("native", 2), ("sigma", .625),
                                               ("pressure", 500), ("geopotential_height", 3500)])
def test_pressure_and_geopotential_colours_share_slice_geometry(coordinate, value):
    m = viewer_manifest(MARS)
    p = FrameProvider(MARS, m)
    height = p.slice(8, "geopotential_height", coordinate, value)
    phi = p.slice(8, "geopotential", coordinate, value)
    pressure = p.slice(8, "pressure", coordinate, value)
    valid = height["mask"] == 0
    for a in (phi, pressure):
        np.testing.assert_array_equal(a["mask"], height["mask"])
        np.testing.assert_array_equal(a["height"], height["height"])
    np.testing.assert_allclose(height["values"][valid], height["height"][valid])
    np.testing.assert_allclose(phi["values"][valid], height["height"][valid] * m["physics"]["gravity"])
    if coordinate == "pressure":
        np.testing.assert_array_equal(pressure["values"][valid], value)
    elif coordinate in ("sigma", "native"):
        ps = p.slice(8, "ln_ps")["values"]
        np.testing.assert_allclose(pressure["values"][valid], .625 * ps[valid])
    else:
        assert pressure["values"][valid].min() > 0


def test_slab_mapping_preserves_native_values_and_wrap():
    nlat, nlon = 8, 16
    geometry = dict(nlat=nlat, nlon=nlon, latitude="Gaussian north to south",
                    longitude="360*j/nlon degrees east", slab_order="latitude, longitude")
    north_first = np.arange(nlat * nlon).reshape(nlat, nlon)
    lat = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))
    lon = np.r_[360 * np.arange(nlon) / nlon, 360 - 180 / nlon]
    mapped = _slab_map(north_first, geometry, lat, lon)
    np.testing.assert_allclose(mapped[:, :-1], north_first[::-1], atol=1e-9)
    np.testing.assert_allclose(mapped[:, -1], (north_first[::-1, 0] + north_first[::-1, -1]) / 2)


@pytest.mark.parametrize("directory", ["docs/runs/MOST_H",
                         "runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-30d",
                         "runs/hs-T42L20-prod-002"])
def test_jump_to_final_day_reads_only_requested_members(directory, monkeypatch):
    path = (ROOT / directory).resolve()
    if not path.is_dir():
        pytest.skip(f"Missing archive: {path}")
    m = viewer_manifest(path)
    provider = FrameProvider(path, m)
    reads = []
    original = np.load

    def audit(file, **kwargs):
        reads.append(Path(file).relative_to(path).as_posix())
        assert kwargs == {"mmap_mode": "r", "allow_pickle": False}
        return original(file, **kwargs)

    monkeypatch.setattr(np, "load", audit)
    last = len(m["time_axes"]["snapshots"]["values"]) - 1
    air = provider.slice(last, "temperature", value=len(m["axes"]["vertical"]["full_levels"])-1)
    assert reads == [m["resources"]["state"]["paths"][-1]]
    if "slab" in m["resources"]:
        slab = provider.slice(last, "surface_temperature")
        assert reads[1:] == [m["resources"]["slab"]["paths"][-1]]
        assert not np.allclose(slab["values"], air["values"])
        assert "u" not in slab and (slab["height"] == 0).all()


def test_redirected_path_invalid_time_and_coordinate():
    m = viewer_manifest(BVE)
    p = FrameProvider(BVE, m)
    with pytest.raises(ValueError, match="time index"):
        p.slice(-1, "zeta")
    with pytest.raises(ValueError, match="coordinate"):
        p.slice(0, "zeta", "pressure", 500)
    m["resources"]["state"]["path"] = "../elsewhere.npy"
    with pytest.raises(ValueError, match="resource"):
        p.slice(0, "zeta")


def test_cuda_selection_matches_reference_if_available():
    m = viewer_manifest(W5)
    host = FrameProvider(W5, m)
    accelerated = FrameProvider(W5, m, backend="cuda")
    reference = host.slice(3, "zeta")
    result = accelerated.slice(3, "zeta")
    if accelerated.backend == "cpu":
        assert accelerated.backend_reason
    for key, atol in (("height", 1e-8), ("terrain", 1e-8), ("values", 1e-15),
                      ("u", 1e-10), ("v", 1e-10)):
        np.testing.assert_allclose(result[key], reference[key], rtol=1e-12, atol=atol)
    np.testing.assert_array_equal(result["mask"], reference["mask"])


def test_mars_column_and_section_follow_the_terrain():
    """Columns and sections over prepared terrain use the same datum as the map."""
    if not MARS.is_dir():
        pytest.skip(f"Missing archive: {MARS}")
    m = viewer_manifest(MARS)
    provider = FrameProvider(MARS, m)
    last = len(m["time_axes"]["snapshots"]["values"]) - 1
    peak = np.unravel_index(provider.terrain.argmax(), provider.terrain.shape)
    lat, lon = float(provider.lat[peak[0]]) + 0.7, float(provider.lon[peak[1]]) + 0.3
    column = provider.column(last, lat, lon, snap=False)
    terrain = synthesize_latlon(provider._terrain_lm, [lat], [lon % 360])[0, 0]
    assert column["surface_height"] == pytest.approx(terrain, abs=1e-9) and terrain > 1000
    heights = np.array(column["geopotential_height"])
    assert heights[-1] > column["surface_height"] and np.all(np.diff(heights) < 0)
    section = provider.section(last, "meridional", lon)
    row = int(np.abs(np.array(section["points"]) - lat).argmin())
    expected = synthesize_latlon(provider._terrain_lm, [section["points"][row]], [lon % 360])[0, 0]
    assert section["surface_height"][row] == pytest.approx(expected, abs=1e-9)
    assert all(np.array(section["geopotential_height"])[-1] > np.array(section["surface_height"]))


def mars_variant(tmp_path, *, flat=False, geodesic=False):
    """A copy of the published Mars capsule, relabelled as a capsule the viewer must also open.

    ``flat``: a flat run that records no gravity (the ``tropoi run pe`` default).
    ``geodesic``: the same coefficients on a geodesic state grid (no Gaussian nodes).
    """
    import json
    import shutil
    if not MARS.is_dir():
        pytest.skip(f"Missing archive: {MARS}")
    root = tmp_path / MARS.name
    root.mkdir(parents=True)
    for name in ("config.json", "manifest.json", "pe_coeffs.npy", "pe_snapshot_times.npy"):
        shutil.copy(MARS / name, root / name)
    if not flat:
        shutil.copytree(MARS / "terrain", root / "terrain")
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if flat:
        for block in (config, manifest["run_config"], manifest["state_schema"]["environment"]):
            block.update(topography="flat", gravity=None)
            block.pop("terrain_digest", None)
    if geodesic:
        for block in (config, manifest["run_config"], manifest["state_schema"]["geometry"]):
            block.update(grid="geodesic", resolution=3)
            block.pop("nlat", None)
            block.pop("nlon", None)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "config.json").write_text(json.dumps(config))
    return root


def test_flat_capsule_without_gravity_keeps_native_level_maps(tmp_path):
    """Native T, wind, ζ, δ and p need no gravity; Z and interpolated surfaces are withheld."""
    flat = mars_variant(tmp_path, flat=True)
    m = viewer_manifest(flat)
    assert m["physics"]["gravity"] is None
    assert {"geopotential", "geopotential_height"}.isdisjoint(m["fields"])
    assert {"sigma", "pressure", "geopotential_height", "structure"}.isdisjoint(m["operations"])
    assert "no gravity" in m["unavailable"]["vertical_surfaces"]
    provider = FrameProvider(flat, m)
    reference = FrameProvider(MARS, viewer_manifest(MARS))
    for field in ("temperature", "u", "v", "wind_speed", "zeta", "delta", "pressure"):
        result = provider.slice(8, field, "native", 2)
        expected = reference.slice(8, field, "native", 2)
        # T, the winds, ζ, δ and σ pₛ are the same synthesis whatever the terrain.
        np.testing.assert_allclose(result["values"], expected["values"], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(result["height"], np.zeros(provider.shape))
    with pytest.raises(ValueError, match="Unavailable coordinate"):
        provider.slice(8, "temperature", "sigma", 0.5)


def test_geodesic_column_uses_exact_point_synthesis(tmp_path):
    """No Gaussian nodes to snap to: a snapped request is evaluated at the point itself."""
    geodesic = mars_variant(tmp_path, geodesic=True)
    m = viewer_manifest(geodesic)
    assert m["geometry"]["grid"] == "geodesic" and "nlat" not in m["geometry"]
    assert m["operations"]["structure"] == ["column", "meridional_section", "zonal_section"]
    provider = FrameProvider(geodesic, m)
    snapped = provider.column(8, 12.3, 201.7, snap=True)
    exact = provider.column(8, 12.3, 201.7, snap=False)
    assert (snapped["latitude"], snapped["longitude"]) == (12.3, 201.7)
    assert snapped["sampling"] == "exact band-limited synthesis at the requested point"
    np.testing.assert_array_equal(snapped["temperature"], exact["temperature"])
    reference = FrameProvider(MARS, viewer_manifest(MARS)).column(8, 12.3, 201.7, snap=False)
    np.testing.assert_allclose(exact["temperature"], reference["temperature"], rtol=1e-12)
