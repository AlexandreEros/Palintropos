"""Fixed geopotential-height surfaces of a saved PE state: the Python
reference of :mod:`tropoi.representation.diagnostics.pe_surfaces`.

``("geopotential_height", Z0)`` inverts each column's piecewise-linear
``Phi(ln sigma)`` at ``Phi0 = g Z0``. Checked here: the specification and
its labels (any finite sign); the analytic isothermal column (pressure in
closed form, every mask reason, signed heights); pressure recovered by
inverting a pressure surface node by node; level coincidence; the end
tolerance on both ends and at negative geopotential; refused geometry (no
division by a zero or inverted interval); no extrapolation; and the stored
(16-bit) columns, their error measured apart from the interpolation and
bounded by the quantization.
"""
from __future__ import annotations

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
    END_TOLERANCE, HEIGHT_MASK_LABELS, MASK_ABOVE_TOP_LEVEL,
    MASK_BELOW_BOTTOM_LEVEL, MASK_BELOW_SURFACE, MASK_VALID,
    GeopotentialHeightSurface, PressureSurface, SurfaceUnavailableError,
    interpolate_columns, interpolate_geopotential, parse_surface,
    pressure_columns, pressure_surface)
from tropoi.representation.visual.atmosphere import (  # noqa: E402
    COLUMN_FIELDS, _mesh, dequantize_columns, quantize_columns)
from test_atmosphere_view import R_DRY, write_pe_capsule  # noqa: E402
from test_pressure_surface_view import (_storage,  # noqa: E402
                                        write_holed_capsule)

RES = 6.0


def _columns(root, index=0):
    _, storage = _storage(root)
    lat, lon = _mesh(RES)
    return pressure_columns(storage.frame(index), storage.pe_context(), lat,
                            lon), storage


def test_specification_labels_and_refusals():
    s = parse_surface(("geopotential_height", 5500))
    assert isinstance(s, GeopotentialHeightSurface) and s.height_m == 5500.0
    assert s.kind == "geopotential_height" and s.units == "m"
    assert s.value == 5500.0 and s.label == "5.5 km"
    assert parse_surface(("geopotential_height", -250.0)).label == "-250 m"
    assert parse_surface(("geopotential_height", 0)).label == "0 m"
    assert parse_surface(("geopotential_height", -7123.4)).label \
        == "-7.1234 km"
    assert parse_surface(s) is s
    # pressure is what it was, with the same generic attributes
    p = parse_surface(("pressure", 5e4))
    assert isinstance(p, PressureSurface) and p.kind == "pressure"
    assert p.value == 5e4 and p.units == "Pa" and p.label == "500 hPa"
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            parse_surface(("geopotential_height", bad))
    for bad in ("5000", True, None):
        with pytest.raises(TypeError, match="number in m"):
            parse_surface(("geopotential_height", bad))
    with pytest.raises(ValueError, match="> 0 Pa"):
        parse_surface(("pressure", -1.0))      # heights may be negative,
    with pytest.raises(ValueError, match="unknown surface coordinate"):
        parse_surface(("height", 10.0))       # pressures may not
    with pytest.raises(TypeError, match="geopotential_height"):
        parse_surface(5000.0)
    assert set(HEIGHT_MASK_LABELS) == {0, 1, 2, 3}
    assert "terrain" in HEIGHT_MASK_LABELS[MASK_BELOW_SURFACE]


def test_isothermal_column_closed_form_signed_heights_every_mask(tmp_path):
    """An isothermal column with hydrostatic level positions has Phi_k =
    Phi_s + R_d T ln(1/sigma^h_k) exactly, so inverting it at Phi0 = g Z0
    gives p = p_s exp(-(g Z0 - g h_s)/(R_d T)) to round-off wherever a
    bracket exists; the masks follow the terrain (physical, unexaggerated)
    and the sampled range. The tall fixture terrain spans -12.8 to +23.5
    km, so negative, zero and positive heights all cut it."""
    root, _, _ = write_holed_capsule(tmp_path / "cap", frames=1)
    cols, _ = _columns(root)
    T, g = 230.0, cols.gravity
    hs, ps = cols.surface_height, cols.surface_pressure
    assert hs.min() < -10_000 and hs.max() > 20_000
    phi = cols.geopotential
    seen = set()
    for z0 in (-9000.0, -2500.0, 0.0, 1234.5, 8000.0, 20_000.0, 30_000.0):
        s = interpolate_columns(cols, ("geopotential_height", z0))
        assert s.surface.height_m == z0
        ok = s.valid
        seen |= set(np.unique(s.mask).tolist())
        closed = ps * np.exp(-(g * z0 - g * hs) / (R_DRY * T))
        assert np.abs(s["pressure"] / closed - 1.0)[ok].max() < 1e-12
        with np.errstate(divide="ignore"):         # 0 placeholders off ok
            lp = np.log(s["pressure"] / ps)
        assert np.abs(s["log_sigma"] - lp)[ok].max() < 1e-12
        assert np.abs(s["temperature"] - T)[ok].max() < 1e-9
        assert (s["geopotential_height"] == z0).all()
        # masks from the physical terrain and the sampled levels
        phi0 = g * z0
        below = phi0 < g * hs
        above = phi0 > phi[0] * (1 + END_TOLERANCE * np.sign(phi[0]))
        under = (~below) & (phi0 < phi[-1] - END_TOLERANCE
                            * np.abs(phi[-1]))
        assert ((s.mask == MASK_BELOW_SURFACE) == below).all()
        assert ((s.mask == MASK_ABOVE_TOP_LEVEL) == above).all()
        assert ((s.mask == MASK_BELOW_BOTTOM_LEVEL) == under).all()
        assert (s.bracket[ok] >= 0).all() and (s.bracket[~ok] == -1).all()
        # placeholders are finite, never samples
        assert np.isfinite(s["pressure"]).all()
        assert (s["pressure"][s.mask == MASK_BELOW_SURFACE] == 0).all()
        if z0 < 0:
            assert ok.any()                   # valid below the datum
            assert (hs[ok] < z0).all()
    assert seen == {MASK_VALID, MASK_BELOW_SURFACE, MASK_ABOVE_TOP_LEVEL,
                    MASK_BELOW_BOTTOM_LEVEL}
    # nothing extrapolated: beyond every column at both ends
    lo, hi = cols.sampled_height_range()
    assert lo == pytest.approx(phi[-1].min() / g)
    assert hi == pytest.approx(phi[0].max() / g)
    with pytest.raises(SurfaceUnavailableError, match="no valid node"):
        interpolate_columns(cols, ("geopotential_height", hi + 10.0))
    with pytest.raises(SurfaceUnavailableError, match="terrain spans"):
        interpolate_columns(cols, ("geopotential_height", hs.min() - 10.0))


def test_pressure_recovered_by_inverting_pressure_surfaces(tmp_path):
    """Pressure and height surfaces are inverse readings of one piecewise
    linear Phi(ln sigma): at a node, the height surface through the Z that
    the p0 isobar has there returns p0, and the same T, u, v."""
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    cols, _ = _columns(root)
    rng = np.random.default_rng(3)
    checked = 0
    for p0 in (85_000.0, 61_234.5, 30_000.0, 7_000.0):
        isobar = interpolate_columns(cols, ("pressure", p0))
        nodes = np.argwhere(isobar.valid)
        for j, i in nodes[rng.choice(len(nodes), 12, replace=False)]:
            z0 = float(isobar["geopotential_height"][j, i])
            s = interpolate_columns(cols, ("geopotential_height", z0))
            assert s.mask[j, i] == MASK_VALID
            assert s["pressure"][j, i] == pytest.approx(p0, rel=1e-11)
            for name in ("temperature", "u", "v"):
                assert s[name][j, i] == pytest.approx(
                    isobar[name][j, i], rel=1e-10, abs=1e-9)
            checked += 1
    assert checked == 48


def test_level_coincidence_and_end_tolerance(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    cols, _ = _columns(root)
    g = cols.gravity
    K = cols.nlev
    j, i = 7, 13
    for k in range(K):
        z0 = cols.geopotential[k, j, i] / g     # g (Phi/g) != Phi in bits
        s = interpolate_columns(cols, ("geopotential_height", z0))
        assert s.mask[j, i] == MASK_VALID
        assert s["temperature"][j, i] == pytest.approx(
            cols.temperature[k, j, i], rel=1e-12)
        assert s["pressure"][j, i] == pytest.approx(
            cols.level_positions[k] * cols.surface_pressure[j, i], rel=1e-12)
        assert s.bracket[j, i] in (max(k - 1, 0), min(k, K - 2))
    # the band: synthetic columns (positive and negative geopotential), a
    # target within END_TOLERANCE of an end level counts as on it
    sig = (0.1, 0.5, 0.9)
    for phi_col, phi_s in (([30_000.0, 10_000.0, 2_000.0], 0.0),
                           ([5_000.0, -3_000.0, -7_000.0], -9_000.0)):
        phi = np.array(phi_col)[:, None]
        temp = np.array([200.0, 240.0, 280.0])[:, None]
        for end, sign_out in ((0, +1.0), (-1, -1.0)):
            edge = phi[end, 0]
            inside = edge + sign_out * 0.5 * END_TOLERANCE * abs(edge)
            outside = edge + sign_out * 2.0 * END_TOLERANCE * abs(edge)
            v, m, b = interpolate_geopotential(sig, phi, {"T": temp}, inside,
                                               np.array([phi_s]))
            assert m[0] == MASK_VALID and v["T"][0] == temp[end, 0]
            assert v["log_sigma"][0] == pytest.approx(math.log(sig[end]))
            v, m, b = interpolate_geopotential(sig, phi, {"T": temp}, outside,
                                               np.array([phi_s]))
            assert m[0] == (MASK_ABOVE_TOP_LEVEL if end == 0
                            else MASK_BELOW_BOTTOM_LEVEL)
        # between terrain and the lowest level, and below the terrain
        _, m, _ = interpolate_geopotential(sig, phi, {"T": temp},
                                           0.5 * (phi[-1, 0] + phi_s),
                                           np.array([phi_s]))
        assert m[0] == MASK_BELOW_BOTTOM_LEVEL
        _, m, _ = interpolate_geopotential(sig, phi, {"T": temp}, phi_s,
                                           np.array([phi_s]))
        assert m[0] == MASK_BELOW_BOTTOM_LEVEL      # on the ground: above it
        _, m, _ = interpolate_geopotential(sig, phi, {"T": temp},
                                           phi_s - 1.0, np.array([phi_s]))
        assert m[0] == MASK_BELOW_SURFACE


def test_invalid_column_geometry_is_refused_never_divided():
    sig = (0.1, 0.5, 0.9)
    temp = np.array([[200.0], [240.0], [280.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no division by zero
        for phi, phi_s, what in (
                ([[3e4], [1e4], [1e4]], -1.0, "zero interval"),
                ([[3e4], [5e4], [1e3]], -1.0, "inverted interval"),
                ([[3e4], [1e4], [2e3]], 5e3, "lowest level under ground"),
                ([[3e4], [np.nan], [2e3]], 0.0, "not finite")):
            with pytest.raises(ValueError, match="invalid column geometry|"
                               "not finite"):
                interpolate_geopotential(sig, np.array(phi), {"T": temp},
                                         2e4, np.array([phi_s]))
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_geopotential((0.5, 0.1), np.array([[1e4], [0.0]]), {},
                                 5e3, np.array([-1.0]))


def test_stored_columns_error_bounded_apart_from_interpolation(tmp_path):
    """Stored (16-bit) columns against exact ones on height surfaces: masks
    agree away from the quantum of each boundary; where both are valid the
    pressure, temperature and wind errors stay within the first-order bound
    of the stored quanta (the weight moves by at most the geopotential
    quantum over the bracket's thickness); the interpolation on the stored
    columns is the same rule (tested elsewhere: level coincidence)."""
    root, _, _ = write_holed_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    lat, lon = _mesh(RES)
    cols = pressure_columns(storage.frame(0), storage.pe_context(), lat, lon)
    q, bounds, p_s = quantize_columns(cols)
    deq = dequantize_columns(q, bounds, p_s, (lat.size, lon.size), cols)
    g = cols.gravity
    quantum = {name: np.array([(hi - lo) / 65535.0 for lo, hi in
                               bounds[name]]) for name in COLUMN_FIELDS}
    # the stored terrain is float32, as on the page
    assert deq.surface_height.dtype == np.float64
    assert (deq.surface_height == cols.surface_height.astype(
        np.float32)).all()
    worst = {"pressure_rel": 0.0, "temperature": 0.0, "wind": 0.0}
    disagree = 0
    for z0 in (-9000.0, -2500.0, 0.0, 1234.5, 8000.0, 20_000.0):
        exact = interpolate_columns(cols, ("geopotential_height", z0))
        stored = interpolate_columns(deq, ("geopotential_height", z0))
        phi0 = g * z0
        qphi = quantum["geopotential"]
        near = ((np.abs(phi0 - cols.geopotential[0]) <= qphi[0])
                | (np.abs(phi0 - cols.geopotential[-1]) <= qphi[-1])
                | (np.abs(phi0 - g * cols.surface_height)
                   <= 1e-6 * g * np.abs(cols.surface_height) + 1e-9))
        assert (exact.mask == stored.mask)[~near].all()
        disagree += int((exact.mask != stored.mask).sum())
        ok = exact.valid & stored.valid
        k = exact.bracket[ok]
        thick = (cols.geopotential[k, *np.nonzero(ok)]
                 - cols.geopotential[k + 1, *np.nonzero(ok)])
        dlns = np.abs(np.log(np.array(cols.level_positions)[k + 1])
                      - np.log(np.array(cols.level_positions)[k]))
        dw = 0.5 * np.maximum(qphi[k], qphi[k + 1]) / thick * 1.01
        bound_p = dw * dlns + 2e-7                     # + float32 p_s
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(stored["pressure"] / exact["pressure"] - 1.0)[ok]
        assert (rel <= bound_p).all()
        tk = cols.temperature[k, *np.nonzero(ok)]
        tk1 = cols.temperature[k + 1, *np.nonzero(ok)]
        bound_t = 0.5 * np.maximum(quantum["temperature"][k],
                                   quantum["temperature"][k + 1]) \
            + dw * np.abs(tk1 - tk) + 1e-5
        dt = np.abs(stored["temperature"] - exact["temperature"])[ok]
        assert (dt <= bound_t).all()
        worst["pressure_rel"] = max(worst["pressure_rel"], rel.max())
        worst["temperature"] = max(worst["temperature"], dt.max())
        worst["wind"] = max(worst["wind"], np.abs(
            np.hypot(stored["u"], stored["v"])
            - np.hypot(exact["u"], exact["v"]))[ok].max())
    # the expected size: half a geopotential quantum over R_d T displaces
    # ln p (isothermal 230 K here); the tall terrain makes the quantum
    # large (0.55 m in Z), still a few 1e-5 in pressure, sub-millikelvin
    half_q = 0.5 * quantum["geopotential"].max()
    assert worst["pressure_rel"] <= 1.05 * half_q / (R_DRY * 230.0) + 2e-7
    assert worst["temperature"] < 1e-3 and worst["wind"] < 1e-3
    print(f"stored vs exact on height surfaces: {worst}, "
          f"{disagree} boundary nodes differ in mask")


def test_pressure_surface_entry_point_takes_heights(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    lat, lon = _mesh(RES)
    a = pressure_surface(storage.frame(0), storage.pe_context(), lat, lon,
                         ("geopotential_height", 9000.0))
    cols = pressure_columns(storage.frame(0), storage.pe_context(), lat, lon)
    b = interpolate_columns(cols, ("geopotential_height", 9000.0))
    assert (a.mask == b.mask).all()
    for name in ("temperature", "pressure", "u", "v"):
        assert (a[name] == b[name]).all()
    assert a.counts()[MASK_VALID] > 0
