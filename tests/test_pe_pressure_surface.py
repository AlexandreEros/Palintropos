"""Fixed-pressure surfaces from saved PE states (issue #13, first slice).

The numerics of ``representation.diagnostics.pe_surfaces`` on their own,
without any page: the log-sigma interpolant, mask codes and brackets on
synthetic columns; isothermal columns over flat and sloping terrain against
the closed-form hydrostatic height (the hydrostatic level positions make it
exact, the nominal ones carry a measured error); coincidence with a native
level; time variation (a warming column lifts the isobar); invalid geometry
and wholly unavailable surfaces; streamline clipping at masked cells.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
from tropoi.representation.diagnostics.pe_surfaces import (
    MASK_ABOVE_TOP_LEVEL, MASK_BELOW_BOTTOM_LEVEL, MASK_BELOW_SURFACE,
    MASK_VALID, PressureSurface, SurfaceUnavailableError,
    interpolate_log_sigma, parse_surface, pressure_surface)
from tropoi.representation.visual.sphere_streamlines import (
    integrate_streamlines)
from tropoi.spatial.sigma_coordinate import SigmaGrid
from test_atmosphere_view import (GRAVITY, L_MAX, LAT, LON, NLEV, R_DRY,
                                  SIGMA, SQ4PI, _band_limited_terrain,
                                  _context, make_frame)


# ---------------------------------------------------------------------------
# The interpolant on synthetic columns
# ---------------------------------------------------------------------------

def test_fields_linear_in_log_sigma_are_recovered_exactly():
    sig = (0.05, 0.2, 0.45, 0.8)
    target = np.array([[0.05, 0.07, 0.2], [0.3, 0.6, 0.8]])
    a, b = 250.0, -30.0
    stack = a + b * np.log(np.asarray(sig))[:, None, None] * np.ones(
        (1, *target.shape))
    values, mask, bracket = interpolate_log_sigma(sig, {"t": stack}, target)
    assert (mask == MASK_VALID).all()
    assert values["t"] == pytest.approx(a + b * np.log(target), abs=1e-12)
    # brackets: upper level index, the last level clamps to the last bracket
    assert bracket.tolist() == [[0, 0, 1], [1, 2, 2]]


def test_mask_codes_distinguish_ground_from_the_sampled_range():
    sig = (0.05, 0.3, 0.6, 0.9)
    target = np.array([0.01, 0.05, 0.5, 0.9, 0.95, 1.0, 1.2])
    stack = np.arange(4.0)[:, None] * np.ones((1, target.size))
    values, mask, bracket = interpolate_log_sigma(sig, {"x": stack}, target)
    assert mask.tolist() == [MASK_ABOVE_TOP_LEVEL, MASK_VALID, MASK_VALID,
                             MASK_VALID, MASK_BELOW_BOTTOM_LEVEL,
                             MASK_BELOW_BOTTOM_LEVEL, MASK_BELOW_SURFACE]
    assert bracket.tolist() == [-1, 0, 1, 2, -1, -1, -1]
    # placeholders are finite and never pretend to be samples
    assert np.isfinite(values["x"]).all()
    assert values["x"][0] == 0.0 and values["x"][4] == 3.0
    assert values["x"][6] == 0.0


@pytest.mark.parametrize("sig, columns, target, message", [
    ((0.5,), {"x": np.ones((1, 2))}, np.ones(2), "at least two"),
    ((0.5, 0.3), {"x": np.ones((2, 2))}, np.ones(2), "strictly increasing"),
    ((0.2, 1.5), {"x": np.ones((2, 2))}, np.ones(2), "strictly increasing"),
    ((0.2, 0.8), {"x": np.ones((3, 2))}, np.ones(2), "shape"),
    ((0.2, 0.8), {"x": np.array([[1.0, np.nan], [1.0, 1.0]])}, np.ones(2),
     "not finite"),
    ((0.2, 0.8), {"x": np.ones((2, 2))}, np.array([0.5, np.inf]),
     "not finite"),
])
def test_invalid_columns_and_targets_are_refused(sig, columns, target,
                                                  message):
    with pytest.raises(ValueError, match=message):
        interpolate_log_sigma(sig, columns, target)


def test_hydrostatic_level_positions_follow_the_simmons_burridge_step():
    grid = SigmaGrid(SIGMA)
    s, h = grid.full_levels, grid.hydrostatic_full_levels
    assert h[0] == pytest.approx(s[0], rel=1e-15)        # alpha_1 = ln 2
    for k in range(1, grid.nlev):
        assert h[k] == pytest.approx(
            grid.interfaces[k + 1] * math.exp(-grid.alpha[k]), rel=1e-15)
        assert grid.interfaces[k] < h[k] < grid.interfaces[k + 1]
        assert h[k] != pytest.approx(s[k], rel=1e-6)   # interior differs
    assert all(np.diff(h) > 0)


def test_surface_specification_is_explicit_about_coordinate_and_units():
    assert parse_surface(("pressure", 50_000.0)).pressure_pa == 50_000.0
    assert parse_surface(("pressure", 300)).label == "3 hPa"
    assert PressureSurface(61.0).label == "61 Pa"
    assert PressureSurface(50_000.0).label == "500 hPa"
    # heights are a coordinate of their own now (test_pe_height_surface)
    assert parse_surface(("geopotential_height", 5000.0)).height_m == 5000.0
    # sigma is a coordinate too now (test_pe_sigma_surface); others are not
    assert parse_surface(("sigma", 0.5)).sigma == 0.5
    with pytest.raises(ValueError, match="unknown surface coordinate"):
        parse_surface(("theta", 300.0))
    with pytest.raises(TypeError, match="'pressure'"):
        parse_surface(50_000.0)
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and > 0"):
            PressureSurface(bad)
    with pytest.raises(TypeError):
        PressureSurface(True)


# ---------------------------------------------------------------------------
# Isothermal columns: closed-form hydrostatic height
# ---------------------------------------------------------------------------

def _isothermal_frame(temperature, ln_ps_lm, *, l_max=L_MAX, nlev=NLEV):
    n = l_max + 1
    frame = np.zeros((3 * nlev + 1, n, n), dtype=np.complex128)
    for k in range(nlev):
        frame[2 * nlev + k, 0, 0] = temperature * SQ4PI
    frame[3 * nlev] = ln_ps_lm
    return frame


def _balanced_ln_ps(terrain_lm, temperature, p_ref, *, gravity=GRAVITY):
    """ln p_s = ln p_ref - g h / (R_d T): linear in h, so exact spectrally."""
    lm = -(gravity / (R_DRY * temperature)) * np.asarray(terrain_lm,
                                                          dtype=complex)
    lm[0, 0] += math.log(p_ref) * SQ4PI
    return lm


@pytest.mark.parametrize("terrain", [False, True])
def test_isothermal_column_height_is_exact_with_hydrostatic_positions(
        terrain):
    T, p_ref = 230.0, 9.0e4
    lm = _band_limited_terrain() if terrain else None
    context = _context(terrain=lm)
    ln_ps = (_balanced_ln_ps(lm, T, p_ref) if terrain else
             np.zeros((L_MAX + 1, L_MAX + 1), complex))
    if not terrain:
        ln_ps[0, 0] = math.log(p_ref) * SQ4PI
    frame = _isothermal_frame(T, ln_ps)
    levels = reconstruct_levels(frame, context, LAT, LON, winds=False,
                                vorticity_divergence=False)
    for p0 in (3.0e4, 5.0e4, 7.0e4):
        s = pressure_surface(frame, context, LAT, LON, ("pressure", p0),
                             winds=False, levels=levels)
        h_s = 0.0 if s.surface_height is None else s.surface_height
        closed = h_s + (R_DRY * T / GRAVITY) * np.log(s.surface_pressure / p0)
        ok = s.valid
        assert ok.any()
        assert s["geopotential_height"][ok] == pytest.approx(closed[ok],
                                                              abs=1e-6)
        assert s["temperature"][ok] == pytest.approx(T, abs=1e-9)
        # where the surface is underground the mask says so, nothing else
        under = s.surface_pressure < p0
        assert ((s.mask == MASK_BELOW_SURFACE) == under).all()


def test_nominal_level_positions_carry_the_measured_isothermal_error():
    T, p_ref = 230.0, 9.0e4
    context = _context(terrain=None)
    ln_ps = np.zeros((L_MAX + 1, L_MAX + 1), complex)
    ln_ps[0, 0] = math.log(p_ref) * SQ4PI
    frame = _isothermal_frame(T, ln_ps)
    levels = reconstruct_levels(frame, context, LAT, LON, winds=False,
                                vorticity_divergence=False)
    grid = SigmaGrid(SIGMA)
    worst = 0.0
    for k in range(1, NLEV):                      # interior levels only
        p0 = grid.full_levels[k] * p_ref
        s = pressure_surface(frame, context, LAT, LON, ("pressure", p0),
                             winds=False, levels=levels, level_sigma="nominal")
        closed = (R_DRY * T / GRAVITY) * math.log(p_ref / p0)
        # the nominal attachment hands back Phi_k at sigma_k p_s: its error
        # is the Simmons-Burridge offset R T (alpha_k - ln(s_{k+1/2}/s_k))/g
        expected = (R_DRY * T / GRAVITY) * (
            grid.alpha[k] - math.log(grid.interfaces[k + 1]
                                     / grid.full_levels[k]))
        err = s["geopotential_height"] - closed
        assert err == pytest.approx(expected, abs=1e-6)
        worst = max(worst, abs(expected))
    assert worst > 10.0          # metres: a real, visible offset
    # the top level has no offset (alpha_1 = ln 2 exactly)
    s = pressure_surface(frame, context, LAT, LON,
                         ("pressure", grid.full_levels[0] * p_ref),
                         winds=False, levels=levels, level_sigma="nominal")
    assert s["geopotential_height"] == pytest.approx(
        (R_DRY * T / GRAVITY) * math.log(1.0 / grid.full_levels[0]),
        abs=1e-6)


# ---------------------------------------------------------------------------
# Native-level coincidence and time variation on a general state
# ---------------------------------------------------------------------------

def test_a_target_on_a_level_position_recovers_that_level():
    rng = np.random.default_rng(3)
    frame = make_frame(rng)
    frame[3 * NLEV] = 0.0
    frame[3 * NLEV, 0, 0] = math.log(1.0e5) * SQ4PI      # uniform p_s
    context = _context(terrain=None)
    levels = reconstruct_levels(frame, context, LAT, LON)
    positions = context.sigma.hydrostatic_full_levels
    for k in range(NLEV):
        s = pressure_surface(frame, context, LAT, LON,
                             ("pressure", positions[k] * 1.0e5),
                             levels=levels)
        assert (s.mask == MASK_VALID).all()
        for name in ("temperature", "geopotential", "u", "v"):
            assert s[name] == pytest.approx(levels[name][k], rel=1e-12,
                                            abs=1e-12)
        assert s["geopotential_height"] == pytest.approx(
            levels["geopotential_height"][k], rel=1e-12)


def test_a_warming_column_lifts_the_isobar_between_saved_times():
    rng = np.random.default_rng(5)
    context = _context(terrain=None)
    early = make_frame(rng, t_offset=0.0)
    late = early.copy()
    for k in range(NLEV):                       # +4 K everywhere, same p_s
        late[2 * NLEV + k, 0, 0] += 4.0 * SQ4PI
    p0 = 5.0e4
    a = pressure_surface(early, context, LAT, LON, ("pressure", p0))
    b = pressure_surface(late, context, LAT, LON, ("pressure", p0))
    ok = a.valid & b.valid
    assert ok.any()
    lift = b["geopotential_height"][ok] - a["geopotential_height"][ok]
    # hydrostatics: dZ = (R_d dT / g) ln(p_s / p0) > 0 wherever p_s > p0,
    # with the same p_s in both states
    assert (lift > 0).all()
    expected = (R_DRY * 4.0 / GRAVITY) * np.log(a.surface_pressure[ok] / p0)
    assert lift == pytest.approx(expected, rel=1e-9)
    assert (a.mask == b.mask).all()
    assert (b["temperature"][ok] - a["temperature"][ok]) == pytest.approx(
        4.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Failures: invalid geometry, nothing to show
# ---------------------------------------------------------------------------

def test_non_monotonic_geopotential_is_refused_not_interpolated():
    rng = np.random.default_rng(9)
    frame = make_frame(rng)
    context = _context(terrain=None)
    frame[2 * NLEV + 1, 0, 0] = -300.0 * SQ4PI          # a negative layer T
    with pytest.raises(ValueError, match="non-monotonic column geometry"):
        pressure_surface(frame, context, LAT, LON, ("pressure", 5.0e4))


def test_a_wholly_unavailable_surface_fails_and_names_the_range():
    rng = np.random.default_rng(1)
    frame = make_frame(rng)
    context = _context(terrain=None)
    with pytest.raises(SurfaceUnavailableError, match="no valid node"):
        pressure_surface(frame, context, LAT, LON, ("pressure", 2.0e5),
                         time_label="day 1")
    with pytest.raises(SurfaceUnavailableError, match="day 1"):
        pressure_surface(frame, context, LAT, LON, ("pressure", 1.0),
                         time_label="day 1")


def test_partial_coverage_gives_holes_not_zeros():
    T, p_ref = 230.0, 1.0e5
    lm = _band_limited_terrain(scale=6000.0)
    context = _context(terrain=lm)
    frame = _isothermal_frame(T, _balanced_ln_ps(lm, T, p_ref))
    s = pressure_surface(frame, context, LAT, LON,
                         ("pressure", 0.93 * p_ref), winds=False)
    counts = s.counts()
    assert counts[MASK_VALID] > 0 and counts[MASK_BELOW_SURFACE] > 0
    assert counts[MASK_BELOW_BOTTOM_LEVEL] > 0       # the ring above ground
    assert counts[MASK_ABOVE_TOP_LEVEL] == 0
    # the three regions nest as the physics says: under ground where
    # p_s < p0; in the gap where p0 / p_s is between the last level and 1
    sig_last = s.level_positions[-1]
    assert ((s.mask == MASK_BELOW_SURFACE) == (s.surface_pressure < 0.93
                                                * p_ref)).all()
    gap = (s.sigma_target > sig_last) & (s.sigma_target <= 1.0)
    assert ((s.mask == MASK_BELOW_BOTTOM_LEVEL) == gap).all()


# ---------------------------------------------------------------------------
# Streamlines never cross a masked gap
# ---------------------------------------------------------------------------

def test_streamlines_stop_at_masked_cells():
    lat = np.linspace(-90.0, 90.0, 37)
    lon = np.linspace(-180.0, 180.0, 73)
    u = np.full((lat.size, lon.size), 20.0)      # a zonal flow everywhere
    v = np.zeros_like(u)
    valid = np.ones_like(u, dtype=bool)
    valid[:, 30:43] = False                       # a masked band of longitudes
    lines = integrate_streamlines(u, v, lat, lon, valid=valid, seed_count=200)
    assert lines
    lon_lo, lon_hi = lon[29], lon[43]            # masked cells span these
    for la, lo in lines:
        inside = (lo > lon_lo) & (lo < lon_hi)
        assert not inside.any(), (lo.min(), lo.max())
    # the same flow without a mask crosses the band
    free = integrate_streamlines(u, v, lat, lon, seed_count=200)
    assert any(((lo > lon_lo) & (lo < lon_hi)).any() for _, lo in free)
    with pytest.raises(ValueError, match="valid has shape"):
        integrate_streamlines(u, v, lat, lon, valid=valid[:, :10])


def test_a_step_longer_than_a_cell_cannot_jump_a_thin_masked_band():
    # 1 deg mesh, one masked node column at 1 degE (cells 0..2 invalid), a
    # zonal flow and a 6 deg step: the RK4 evaluation points of a seed at
    # -0.1 degE lie at -0.1, 2.9, 2.9 and 5.9 degE, all valid, so only a check
    # along the whole segment can stop the line.
    lat = np.linspace(-90.0, 90.0, 181)
    lon = np.linspace(-180.0, 180.0, 361)
    u = np.full((lat.size, lon.size), 20.0)
    v = np.zeros_like(u)
    valid = np.ones_like(u, dtype=bool)
    valid[:, 181] = False                                  # lon = +1 deg
    seeds = (np.zeros(5), np.array([-0.1, -0.4, -3.0, 4.0, 10.0]))
    lines = integrate_streamlines(u, v, lat, lon, valid=valid, seeds=seeds,
                                  step_deg=6.0, max_steps=4)
    for la, lo in lines:
        assert not ((lo > 0.0) & (lo < 2.0)).any(), (lo.min(), lo.max())
        # a line is entirely west or entirely east of the band
        assert (lo <= 0.0).all() or (lo >= 2.0).all()
    assert lines                                           # some survive
