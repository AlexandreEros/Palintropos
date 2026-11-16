"""The default discretization: which grid, which evaluator, sized how.

Three things need establishing, and they are deliberately separate:

1. the **sizing policy** is self-consistent -- what
   :func:`gauss_grid_for_truncation` returns satisfies analysis exactness,
   the 3/2 quadratic rule, and separable applicability, at every
   truncation, stated as literal arithmetic rather than by re-calling the
   production predicates;
2. the **defaults select it** -- every core and ``Planet.generate`` land on
   Gauss lat-lon with the separable evaluator and dimensions from ``lmax``,
   and an explicitly given value still wins;
3. the **numerics are right there** -- on the grid the defaults actually
   produce, a round trip of a degree-``l_max`` field is exact to round-off
   and the separable evaluator is used without falling back.

(1) and (2) are pure CPU; (3) needs CUDA and is guarded.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from tropoi.spatial.truncation import gauss_grid_for_truncation


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:  # pragma: no cover - import guard
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(),
                                   reason="CUDA/CuPy not available")


def _real_field_coeffs(l_max, seed):
    """One real field's coefficients: lower triangle, m = 0 column real.

    Same convention as tests/test_separable_transform.py::_coeffs -- the
    (l_max+1, l_max+1) packed triangle the transforms consume.
    """
    import cupy as cp
    rng = np.random.default_rng(seed)
    n = l_max + 1
    c = np.tril(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    c[:, 0] = c[:, 0].real
    return cp.asarray(c)

#: Truncations the policy is checked over: every small case, the two in
#: production use (21, 42), the Held-Suarez one (43), and a large one.
TRUNCATIONS = list(range(1, 25)) + [28, 42, 43, 63, 85, 170]


# ---------------------------------------------------------------------------
# 1. the sizing policy is self-consistent (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l_max,expected", [
    (1, (2, 4)), (2, (4, 8)), (4, (8, 16)), (5, (8, 16)),
    (8, (14, 28)), (10, (16, 32)), (15, (24, 48)),
    (21, (32, 64)), (42, (64, 128)), (63, (96, 192)),
])
def test_the_sizing_policy_is_pinned(l_max, expected):
    """The values the policy is specified to produce.

    T21 -> 32x64 and T42 -> 64x128 are exactly the resolutions in
    production use, which is the point: the policy was chosen to reproduce
    the repository's established convention, not to replace it.
    """
    assert gauss_grid_for_truncation(l_max) == expected


@pytest.mark.parametrize("l_max", TRUNCATIONS)
def test_the_sizing_policy_permits_exact_analysis(l_max):
    """nlat >= l_max+1 and nlon >= 2 l_max+1 (spatial.grids.latlon_grid).

    Below this a round trip of a degree-l_max field leaks energy across
    modes, and GaussLatLonSphericalHarmonics warns.
    """
    nlat, nlon = gauss_grid_for_truncation(l_max)
    assert nlat >= l_max + 1
    assert nlon >= 2 * l_max + 1


@pytest.mark.parametrize("l_max", TRUNCATIONS)
def test_the_sizing_policy_satisfies_the_three_halves_rule(l_max):
    """nlat >= ceil((3 l_max+1)/2) and nlon >= 3 l_max+1.

    The quadratic-dealiasing sizing LatLonBackend applies to its *fine*
    product grid. Holding the state grid to it too is the policy's whole
    content, so this is the condition that actually binds.
    """
    nlat, nlon = gauss_grid_for_truncation(l_max)
    assert nlat >= -(-(3 * l_max + 1) // 2)
    assert nlon >= 3 * l_max + 1


@pytest.mark.parametrize("l_max", TRUNCATIONS)
def test_the_sizing_policy_keeps_the_separable_evaluator_applicable(l_max):
    """2 l_max < nlon (transforms.separable_gauss_sh).

    Every retained zonal wavenumber needs its own real-FFT bin strictly
    below Nyquist, or the default evaluator falls back to the dense path.
    """
    _, nlon = gauss_grid_for_truncation(l_max)
    assert 2 * l_max < nlon


@pytest.mark.parametrize("l_max", TRUNCATIONS)
def test_the_sizing_policy_is_even_and_square_aspect(l_max):
    nlat, nlon = gauss_grid_for_truncation(l_max)
    assert nlat % 2 == 0, "even nlat: Gauss nodes symmetric, none on the equator"
    assert nlon == 2 * nlat


@pytest.mark.parametrize("l_max", TRUNCATIONS)
def test_the_sizing_policy_clears_the_configuration_floors(l_max):
    """It never produces a pair the configuration layer would reject."""
    from tropoi.run.bve.config import MIN_NLAT, MIN_NLON
    nlat, nlon = gauss_grid_for_truncation(l_max)
    assert nlat >= MIN_NLAT and nlon >= MIN_NLON


def test_the_sizing_policy_rejects_a_degenerate_truncation():
    with pytest.raises(ValueError, match="l_max must be >= 1"):
        gauss_grid_for_truncation(0)


# ---------------------------------------------------------------------------
# 2. the defaults select that grid and that evaluator (CPU)
# ---------------------------------------------------------------------------

def _configs():
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.pe.config import PERunConfig
    from tropoi.run.swe.config import SWERunConfig
    return {"bve": BVERunConfig, "swe": SWERunConfig, "pe": PERunConfig}


@pytest.mark.parametrize("solver", ["bve", "swe", "pe"])
def test_every_core_defaults_to_the_gauss_latlon_grid(solver):
    cfg = _configs()[solver].resolve({})
    assert cfg.grid == "latlon"


@pytest.mark.parametrize("solver", ["bve", "swe", "pe"])
def test_every_core_sizes_its_default_grid_from_its_truncation(solver):
    cfg = _configs()[solver].resolve({})
    assert (cfg.nlat, cfg.nlon) == gauss_grid_for_truncation(cfg.lmax)


@pytest.mark.parametrize("solver", ["bve", "swe", "pe"])
@pytest.mark.parametrize("l_max", [8, 21, 42])
def test_the_derived_dimensions_follow_an_explicit_truncation(solver, l_max):
    cfg = _configs()[solver].resolve({"lmax": l_max})
    assert (cfg.nlat, cfg.nlon) == gauss_grid_for_truncation(l_max)


@pytest.mark.parametrize("solver", ["bve", "swe", "pe"])
def test_explicit_dimensions_override_the_derived_ones(solver):
    """The rule the policy must not break: given dimensions always win."""
    cfg = _configs()[solver].resolve({"lmax": 21, "nlat": 48, "nlon": 96})
    assert (cfg.nlat, cfg.nlon) == (48, 96)
    # and one at a time, so a half-given pair is not silently replaced
    half = _configs()[solver].resolve({"lmax": 21, "nlat": 48})
    assert half.nlat == 48 and half.nlon == gauss_grid_for_truncation(21)[1]


@pytest.mark.parametrize("solver", ["bve", "swe", "pe"])
def test_the_geodesic_grid_is_reachable_but_never_implicit(solver):
    """Retained, and only by asking for it -- by either accepted spelling."""
    cls = _configs()[solver]
    assert cls.resolve({}).grid == "latlon"
    assert cls.resolve({"grid": "geodesic"}).grid == "geodesic"
    assert cls.resolve({"grid": "gauss-latlon"}).grid == "latlon"


def test_the_cli_supplies_no_grid_of_its_own():
    """`--grid` must not become mandatory, and must not carry a default.

    The configuration layer owns the default; argparse passing one would
    mean two sources of truth for the same decision.
    """
    from tropoi.cli.main import build_parser
    parser = build_parser()
    for solver in ("bve", "swe", "pe"):
        args = parser.parse_args(["run", solver])
        assert args.grid is None
        assert args.nlat is None and args.nlon is None


def test_every_bve_preset_lands_on_the_default_grid():
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.bve.presets import PRESETS
    for name, preset in PRESETS.items():
        cfg = BVERunConfig.resolve({}, preset["settings"])
        assert cfg.grid == "latlon", name
        assert (cfg.nlat, cfg.nlon) == gauss_grid_for_truncation(cfg.lmax), name


def test_planet_generate_defaults_to_latlon_and_derived_dimensions():
    # Importing Planet pulls in CuPy. The assertions need only the module,
    # not a device, so this skips on the CPU-only CI rather than requiring
    # CUDA.
    pytest.importorskip("cupy")
    from tropoi.spatial.planet import Planet
    sig = inspect.signature(Planet.generate)
    assert sig.parameters["grid_type"].default == "latlon"
    # None, not a literal pair: the sizing lives in one place.
    assert sig.parameters["nlat"].default is None
    assert sig.parameters["nlon"].default is None


def test_the_default_evaluator_is_separable():
    pytest.importorskip("cupy")   # latlon_grid imports CuPy; no device needed
    from tropoi.spatial.grids.latlon_grid import default_transform_backend
    assert default_transform_backend() == "separable"


# ---------------------------------------------------------------------------
# 3. on the grid the defaults produce, the numerics are right (CUDA)
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("l_max", [8, 10, 21])
def test_a_round_trip_on_the_default_grid_is_exact_to_round_off(l_max):
    """Analysis exactness at the *derived* size, which is new sizing.

    A random degree-l_max coefficient vector, synthesized and re-analyzed
    on the grid the policy chooses, must come back unchanged: that is what
    "the resolution is appropriate for the solver" means for the transform.
    """
    import cupy as cp
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)

    nlat, nlon = gauss_grid_for_truncation(l_max)
    grid = GaussLatLonGridGeometry(nlat, nlon, 6.371e6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # no under-resolution, no fallback
        sh = GaussLatLonSphericalHarmonics(grid, l_max)

    a = _real_field_coeffs(l_max, seed=20261116 + l_max)
    back = sh.transform(sh.inv_transform(a))
    err = float(cp.max(cp.abs(back - a)) / cp.max(cp.abs(a)))
    assert err < 1e-12, f"l_max={l_max} on {nlat}x{nlon}: round trip {err:.2e}"


@requires_cuda
@pytest.mark.parametrize("l_max", [8, 10, 21, 42])
def test_the_default_grid_uses_the_separable_evaluator_without_falling_back(l_max):
    """The default sizing and the default evaluator must agree.

    If the policy ever produced a grid the separable path could not use,
    every run would silently land on the dense path -- which warns, so
    turning the DeprecationWarning into an error is the assertion.
    """
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)

    nlat, nlon = gauss_grid_for_truncation(l_max)
    grid = GaussLatLonGridGeometry(nlat, nlon, 6.371e6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sh = GaussLatLonSphericalHarmonics(grid, l_max)
    assert sh.transform_backend == "separable"


@requires_cuda
def test_a_field_on_the_default_grid_renders_through_the_planet_viewer():
    """The default grid must be *plottable*, not merely correct.

    `PlanetViewer.plot_scalar` takes the flat (n_points,) array the
    transforms produce and, for a structured grid, hands it to `imshow`,
    which needs (nlat, nlon). Nothing caught this when lat-lon became the
    default: the geodesic path interpolates onto a 2-D view grid first, and
    the one existing `plot_scalar` call in the suite sits inside a
    try/except that only prints. Bare `tropoi gen` failed on an invalid
    image shape.
    """
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cupy as cp
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)
    from tropoi.representation.visual.planet_viewer import PlanetViewer

    l_max = 10
    nlat, nlon = gauss_grid_for_truncation(l_max)
    grid = GaussLatLonGridGeometry(nlat, nlon, 6.371e6)
    sh = GaussLatLonSphericalHarmonics(grid, l_max)

    field = sh.inv_transform(_real_field_coeffs(l_max, seed=4))
    assert field.ndim == 1 and field.size == nlat * nlon, (
        "a transform returns a flat lat-major field; this test is only "
        "meaningful while that is true")

    fig, ax = plt.subplots(1, 1)
    try:
        PlanetViewer.plot_scalar(field, grid, title="default grid", ax=ax)
        image = [a for a in ax.get_images()]
        assert image, "no image was drawn"
        assert image[0].get_array().shape == (nlat, nlon)
    finally:
        plt.close(fig)


@requires_cuda
def test_the_default_bve_discretization_builds_and_transforms():
    """End to end: the default config -> a Planet -> an exact round trip.

    Covers the wiring, not the transform: that the grid family, the
    dimensions and the evaluator the configuration layer resolves are the
    ones the model is actually built on.
    """
    import cupy as cp
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.temporal.models import build_planet

    cfg = BVERunConfig.resolve({})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        planet = build_planet(cfg.to_run_config_dict())

    assert planet.grid.nlat == cfg.nlat == 32
    assert planet.grid.nlon == cfg.nlon == 64
    assert planet.sh.transform_backend == "separable"

    a = _real_field_coeffs(cfg.lmax, seed=20261116)
    back = planet.sh.transform(planet.sh.inv_transform(a))
    assert float(cp.max(cp.abs(back - a)) / cp.max(cp.abs(a))) < 1e-12
