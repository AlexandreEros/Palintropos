"""Isca's Monin-Obukhov surface layer and its stability-dependent diffusivity.

Reference agreement is exact, and it does not need a Fortran compiler.
Isca's kernel carries its own unit tests, and each one stores its expected
answer as ``transfer(sum(x), integer(8))`` - the IEEE-754 bit pattern of
the double reinterpreted as an integer. Two consequences are used here:

* a test over a single array decodes straight back to the reference value
  (``CHKSUM_STABLE_MIX`` is the bit pattern of ``0.07454248254045348``);
* for a test that sums several arrays, the *integer* difference between
  our checksum and Isca's is the total error in units in the last place,
  because adjacent doubles have adjacent bit patterns. Zero therefore
  means bit-for-bit, and a small number would mean "agrees to a few ulp".

Fortran's ``integer(8)`` addition wraps, so the sums are reduced modulo
2^64 before comparison. The three checksums below are Isca's published
Intel/Lahey values at commit ``d1321ac3``, in
``src/atmos_param/monin_obukhov/monin_obukhov_kernel.F90``.

The rest is physics and plumbing: the five regimes the scheme has to
survive (stable, unstable, neutral, near-critical, low wind), the
closure's own residual, the neutral-limit identity with the FHZ06 scheme
it is an alternative to, the configuration contract, and on a GPU the
agreement of the vectorized and fused-kernel paths plus a short coupled
run.
"""
from __future__ import annotations

import json
import struct
from fractions import Fraction

import numpy as np
import pytest

from tropoi.physics.boundary_layer import (boundary_layer_depth, diffusivity_profile,
                                           monin_obukhov_diffusivity_profile)
from tropoi.physics.monin_obukhov import (ISCA_SMALL, VON_KARMAN,
                                          MoninObukhovConvergenceError,
                                          MoninObukhovOptions, diffusivities,
                                          solve_zeta, stability_derivative_m,
                                          stability_derivative_t, stable_mix,
                                          surface_drag)
from tropoi.physics.surface import (monin_obukhov_sensible_heat_flux, sensible_heat_flux)
from tropoi.run.radiative.config import PRESETS, RadiativeConfig


def _has_cuda() -> bool:
    try:
        import cupy
        return bool(cupy.is_available())
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs CuPy and a CUDA GPU")

G_MARS, CP_MARS, R_MARS = 3.71, 844.8, 192.0
Z0_MARS = 3.21e-5

_WRAP = 1 << 64


def _bits(x: float) -> int:
    """Fortran ``transfer(real(8), integer(8))``."""
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def _wrap(i: int) -> int:
    i &= _WRAP - 1
    return i - _WRAP if i >= (1 << 63) else i


def _fma(a: float, b: float, c: float) -> np.float64:
    """``a * b + c`` with a single rounding, as the hardware fma does.

    ``math.fma`` arrived in Python 3.13; the product is formed exactly in
    :class:`~fractions.Fraction` instead and rounded once on the way back to
    a double. Used to show that the CUDA kernel's only departure from the
    NumPy path is ``nvcc``'s default contraction of ``x*y + z``.
    """
    return np.float64(float(Fraction(float(a)) * Fraction(float(b))
                            + Fraction(float(c))))


def _fortran_sum(a) -> float:
    """``sum()`` in index order, so the rounding matches the Fortran."""
    total = 0.0
    for v in np.asarray(a).ravel():
        total += float(v)
    return total


# the kernel's own test settings, which are not Isca's run-time defaults
KERNEL_TEST_OPTS = MoninObukhovOptions(rich_crit=10.0, drag_min=1.0e-5,
                                       stable_option=1, zeta_trans=0.5)
KERNEL_TEST_GRAVITY = 9.80

# monin_obukhov_kernel.F90, subroutine test_drag
DRAG_CASE = dict(
    pt=np.array([268.559120403867, 269.799228886728, 277.443023238556,
                 295.79192777341, 293.268717243262]),
    pt0=np.array([273.42369841804, 272.551410044203, 278.638168565727,
                  298.133068766049, 292.898163706587]),
    z=np.array([29.432779269303, 30.0497139076724, 31.6880000418153,
                34.1873479240475, 33.2184943356517]),
    z0=np.array([5.86144925739178e-05, 0.0001, 0.000641655193293549,
                 3.23383768877187e-05, 0.07]),
    zt=np.array([3.69403636275411e-05, 0.0001, 1.01735489109205e-05,
                 7.63933834969505e-05, 0.00947346982656289]),
    zq=np.array([5.72575636226887e-05, 0.0001, 5.72575636226887e-05,
                 5.72575636226887e-05, 5.72575636226887e-05]),
    speed=np.array([2.9693638452068, 2.43308757772094, 5.69418282305367,
                    9.5608693754561, 4.35302260074334]),
)
CHKSUM_DRAG = 4466746452959549648
CHKSUM_STABLE_MIX = 4590035772608644256
CHKSUM_DIFF = -9222066590093362639


def _drag_case(opts=None, **over):
    kw = dict(DRAG_CASE)
    kw.update(over)
    return surface_drag(kw["pt"], kw["pt0"], kw["z"], kw["speed"],
                        gravity=KERNEL_TEST_GRAVITY, roughness_momentum=kw["z0"],
                        roughness_heat=kw["zt"], roughness_moisture=kw["zq"],
                        opts=KERNEL_TEST_OPTS if opts is None else opts)


# ---------------------------------------------------------------------------
# reference agreement with Isca, bit for bit
# ---------------------------------------------------------------------------

def test_the_drag_iteration_reproduces_iscas_published_aggregate_checksum():
    """``test_drag``: the whole nonlinear solve over five columns - unstable,
    stable, three roughness regimes.

    What this establishes, precisely: the checksum is
    ``sum_arrays bits(sum_elements(array))``, so matching it at zero
    constrains the bit pattern of each array's *total*, summed over the five
    arrays. It is an **aggregate** agreement, not elementwise bit identity -
    an error in one element could in principle be cancelled by another
    element of the same array, or by another array's bit pattern.

    Elementwise reference agreement is established separately, against the
    per-column literals Isca publishes in ``test_profile``
    (:func:`test_the_published_per_column_scales_agree_elementwise`), and
    elementwise identity *between this implementation and the fused CUDA
    kernel* by
    :func:`test_the_fused_kernel_is_elementwise_identical_on_the_reference_case`.
    """
    e = _drag_case()
    e.require_convergence()
    total = _wrap(sum(_bits(_fortran_sum(a)) for a in
                      (e.drag_m, e.drag_t, e.drag_q, e.u_star, e.b_star)))
    assert total - CHKSUM_DRAG == 0, f"{total - CHKSUM_DRAG} ulp from Isca"


def test_the_published_per_column_scales_agree_elementwise():
    """**Elementwise** reference agreement, which the aggregate checksum
    cannot give.

    Isca's ``test_profile`` hardcodes ``u_star`` and ``b_star`` as inputs,
    column by column, alongside the same five columns of geometry
    ``test_drag`` uses. Those are ten independent published decimal
    literals, so matching them pins each column separately and closes the
    cancellation loophole in the checksum.

    One subtlety had to be resolved first. Isca's test *program* sets
    ``rich_crit = 10.0`` before calling ``test_drag``, but ``test_profile``'s
    literals were evidently generated with the **module default of 2.0**:
    the four unstable columns do not depend on ``rich_crit`` at all (the
    Businger-Dyer branch has no ``b_stab`` in it), and only the fifth,
    stable column does. At 10.0 that one column is off by 4e-4; at 2.0 every
    column agrees. 2.0 is also the value grey-Mars runs, so this check sits
    at the production setting.

    The residual is a few ulp and is limited by the printed precision of the
    literals (15-16 significant digits), so this is elementwise agreement to
    the reference's own precision, not a demonstration of bit identity.
    """
    opts = MoninObukhovOptions(rich_crit=2.0, drag_min=1.0e-5,
                               stable_option=1, zeta_trans=0.5)
    e = _drag_case(opts=opts)
    e.require_convergence()
    published_u = np.array([0.109462510724615, 0.0932942802513508, 0.223232887323184,
                            0.290918439028557, 0.260087579361467])
    published_b = np.array([0.00690834676781433, 0.00428178089592372,
                            0.00121229800895103, 0.00262353784027441,
                            -0.000570314880866852])
    # elementwise, every column, at the printed precision of the literals
    assert np.abs(e.u_star - published_u).max() / np.abs(published_u).max() < 1e-13
    assert np.abs(e.b_star - published_b).max() / np.abs(published_b).max() < 1e-12
    for i in range(5):
        assert float(e.u_star[i]) == pytest.approx(float(published_u[i]), rel=1e-13)
        assert float(e.b_star[i]) == pytest.approx(float(published_b[i]), rel=1e-11)
    # the fifth column is the stable one, and the only rich_crit-sensitive one
    assert float(e.richardson[4]) > 0.0
    assert bool((e.richardson[:4] < 0.0).all())
    at_ten = _drag_case()                     # the test program's rich_crit
    assert at_ten.u_star[:4] == pytest.approx(e.u_star[:4], rel=0, abs=0)
    assert abs(float(at_ten.u_star[4]) - float(published_u[4])) \
        / float(published_u[4]) > 1e-4


def test_the_stable_mixing_factor_reproduces_iscas_published_checksum():
    """``test_stable_mix``. One array, so the checksum decodes to the exact
    reference value - but that value is the *sum* of the five elements, so
    it is still an aggregate. Three of the five are pinned individually
    below: they are beyond ``rich_crit`` and must be identically zero, which
    leaves the aggregate constraining only the remaining two."""
    rich = np.array([1650.92431853365, 1650.9256285137, 77.7636819036559,
                     1.92806556391324, 0.414767442012442])
    mix = stable_mix(rich, KERNEL_TEST_OPTS)
    assert _bits(_fortran_sum(mix)) == CHKSUM_STABLE_MIX
    # and, decoded, that reference is:
    assert _fortran_sum(mix) == 0.07454248254045348
    # the three columns at or beyond rich_crit = 10 contribute nothing
    assert list(mix[:3]) == [0.0, 0.0, 0.0]


def test_the_diffusivity_reproduces_iscas_published_checksum():
    """``test_diff``: an unstable point, K_m and K_h together.

    Here the arrays are 1x1x1, so each ``sum`` is a single element and there
    is no intra-array cancellation to worry about - the aggregate is over
    exactly two scalars. They differ by a factor of 1.7 (1.77 against 3.03),
    i.e. they sit in different binades, so a compensating pair of errors
    would have to be contrived rather than merely unlucky. Still, formally,
    two unknowns and one equation."""
    k_m, k_h = diffusivities(np.array([19.9982554527751]),
                             np.array([0.129638955971075]),
                             np.array([0.000991799765557209]), KERNEL_TEST_OPTS)
    total = _wrap(_bits(_fortran_sum(k_m)) + _bits(_fortran_sum(k_h)))
    assert total - CHKSUM_DIFF == 0, f"{total - CHKSUM_DIFF} ulp from Isca"


# ---------------------------------------------------------------------------
# the five regimes
# ---------------------------------------------------------------------------

def _mars_column(delta_theta, speed, *, z=10.0, opts=None):
    """One Mars-like column at a given surface-air potential-temperature
    difference (positive = surface warmer = unstable) and wind."""
    ts = np.full(np.shape(delta_theta), 240.0, dtype=float)
    theta = ts - np.asarray(delta_theta, dtype=float)
    return surface_drag(theta, ts, np.full_like(ts, float(z)),
                        np.broadcast_to(np.asarray(speed, dtype=float),
                                        ts.shape).astype(float),
                        gravity=G_MARS, roughness_momentum=Z0_MARS,
                        roughness_heat=Z0_MARS, opts=opts)


def test_neutral_stratification_gives_the_logarithmic_law():
    """Zero potential-temperature difference: the iteration must land on the
    neutral coefficients, which are known in closed form."""
    e = _mars_column(np.zeros(1), 8.0)
    ln = np.log(10.0 / Z0_MARS)
    assert e.richardson == pytest.approx(0.0, abs=0.0)
    assert e.zeta == pytest.approx(0.0, abs=0.0)
    assert e.drag_m == pytest.approx((VON_KARMAN / ln) ** 2, rel=1e-14)
    assert e.drag_t == pytest.approx((VON_KARMAN / ln) ** 2, rel=1e-14)
    assert bool(e.converged.all()) and not bool(e.capped.any())


def test_unstable_enhances_exchange_and_favours_heat_over_momentum():
    e = _mars_column(np.array([1.0, 5.0, 20.0, 60.0]), 8.0)
    neutral = (VON_KARMAN / np.log(10.0 / Z0_MARS)) ** 2
    assert bool((e.richardson < 0).all()) and bool((e.zeta < 0).all())
    assert bool((e.drag_m > neutral).all())
    # phi_h < phi_m when unstable, so heat is exchanged more efficiently
    assert bool((e.drag_t > e.drag_m).all())
    # and the enhancement grows with instability
    assert bool((np.diff(e.drag_m) > 0).all())
    assert bool(e.converged.all())


def test_stable_suppresses_exchange_monotonically():
    e = _mars_column(-np.array([0.5, 2.0, 6.0, 15.0]), 8.0)
    neutral = (VON_KARMAN / np.log(10.0 / Z0_MARS)) ** 2
    assert bool((e.richardson > 0).all()) and bool((e.zeta > 0).all())
    assert bool((e.drag_m < neutral).all())
    assert bool((np.diff(e.drag_m) < 0).all())
    # stable_option 1 uses the same phi for heat and momentum
    assert e.drag_t == pytest.approx(e.drag_m, rel=1e-12)
    # mixing is suppressed but never identically zero, which is the point of
    # the rational stable form
    assert bool((e.drag_m > 0).all())


def test_near_critical_columns_take_drag_min_instead_of_iterating():
    """Isca stops short of ``rich_crit`` at ``0.95 rich_crit`` because the
    Newton step is ill-conditioned there, and assigns ``drag_min``."""
    opts = MoninObukhovOptions()
    # a strong inversion in a light wind drives Ri past 0.95 * 2.0
    e = _mars_column(np.array([-30.0, -60.0]), 0.5, opts=opts)
    assert bool((e.richardson >= 0.95 * opts.rich_crit).all())
    assert bool(e.capped.all())
    assert e.drag_m == pytest.approx(opts.drag_min, rel=0, abs=0)
    assert e.drag_t == pytest.approx(opts.drag_min, rel=0, abs=0)
    assert e.zeta == pytest.approx(0.0, abs=0.0)        # never solved
    # capped is not a failure, so this must not raise
    e.require_convergence()


def test_a_calm_column_is_regularized_by_an_offset_not_a_clamp():
    """Isca divides by ``speed^2 + 1e-4`` rather than clamping, so a column
    at rest looks *neutral* rather than infinitely stable."""
    e = _mars_column(np.array([-5.0, -5.0, -5.0]), np.array([1.0, 1e-3, 0.0]))
    # the calm columns tend to the offset-limited Richardson number, not to
    # something enormous
    assert abs(float(e.richardson[-1])) == pytest.approx(
        G_MARS * 10.0 * 5.0 / 240.0 / ISCA_SMALL, rel=1e-9)
    assert np.isfinite(e.drag_m).all() and np.isfinite(e.u_star).all()
    # at exactly zero wind the friction velocity vanishes but the
    # coefficient stays defined
    assert float(e.u_star[-1]) == 0.0
    assert float(e.drag_m[-1]) > 0.0


def test_a_level_below_its_roughness_length_is_lifted_to_it():
    """``zz = max(z, z0, zt, zq)`` stops ``ln(z/z0)`` going *negative*, which
    would flip the sign of the coefficient. It does not make the level
    valid: a level at or under the roughness height gives ``ln(zz/z0) = 0``
    and the logarithmic law is genuinely singular there, in this code and
    in Isca alike.

    What is asserted here is the useful half - that the lift happens, so a
    level just above its roughness height behaves normally and the answer
    is the one for ``max(z, z0)`` rather than for ``z``.
    """
    rough = 0.07
    just_above = surface_drag(np.array([240.0]), np.array([240.0]),
                              np.array([rough * 1.5]), np.array([5.0]),
                              gravity=G_MARS, roughness_momentum=rough,
                              roughness_heat=rough)
    assert float(just_above.drag_m) > 0.0 and np.isfinite(float(just_above.drag_m))
    assert float(just_above.drag_m) == pytest.approx(
        (VON_KARMAN / np.log(1.5)) ** 2, rel=1e-12)
    # below the roughness height the result is the one for z = z0: the
    # singular limit, not a negative coefficient
    at_and_below = surface_drag(np.array([240.0, 240.0]), np.array([240.0, 240.0]),
                                np.array([rough, rough * 0.1]), np.array([5.0, 5.0]),
                                gravity=G_MARS, roughness_momentum=rough,
                                roughness_heat=rough)
    assert not np.any(at_and_below.drag_m < 0.0)
    assert not np.any(np.isfinite(at_and_below.drag_m) & (at_and_below.drag_m
                                                          < (VON_KARMAN / 1e-3) ** 2))


# ---------------------------------------------------------------------------
# the closure solves its own equation
# ---------------------------------------------------------------------------

def test_the_solve_satisfies_the_richardson_closure_it_inverts():
    """An outside check on the iteration rather than a comparison with
    Isca: the converged ``zeta`` must satisfy ``Ri = zeta f_t / f_m^2``,
    over a wide sweep of regimes.

    It does, to about 4e-7 for a typical column. The exceptions are
    instructive and are asserted rather than excused: Isca stops when the
    *smaller* of the absolute and the relative correction passes 1e-4, so
    for ``|zeta| < 1`` the effective bound on ``zeta`` is absolute and its
    relative accuracy is only ``1e-4 / |zeta|``. Every column whose closure
    residual exceeds 1e-3 is therefore a near-neutral one, and none has
    ``|zeta| > 1``.
    """
    from tropoi.physics.monin_obukhov import (stability_integral_m,
                                              stability_integral_t)
    rng = np.random.default_rng(20261114)
    n = 40000
    ts = rng.uniform(140.0, 300.0, n)
    theta = ts + rng.uniform(-60.0, 60.0, n)
    z = 10.0 ** rng.uniform(-1.0, 2.5, n)
    speed = 10.0 ** rng.uniform(-4.0, 2.0, n)
    z0 = 10.0 ** rng.uniform(-6.0, -1.0, n)
    opts = MoninObukhovOptions()
    e = surface_drag(theta, ts, z, speed, gravity=G_MARS,
                     roughness_momentum=z0, roughness_heat=z0, opts=opts)
    zz = np.maximum(z, z0)
    ln = np.log(zz / z0)
    f_m = stability_integral_m(e.zeta, e.zeta / (zz / z0), ln, opts)
    f_t = stability_integral_t(e.zeta, e.zeta / (zz / z0), ln, opts)
    closure = e.zeta * f_t / (f_m * f_m)
    sel = ~e.capped & e.converged & (np.abs(e.zeta) > 1e-6)
    assert int(sel.sum()) > n // 4, "the sweep should solve most columns"
    residual = np.abs(closure - e.richardson) / np.abs(e.richardson)
    assert float(np.median(residual[sel])) < 1e-5
    assert float(residual[sel].max()) < 3e-2
    loose = sel & (residual > 1e-3)
    assert float(np.abs(e.zeta[loose]).max()) < 1.0, \
        "a loose column must be a near-neutral one, not a strongly stratified one"


def test_the_near_neutral_looseness_does_not_reach_the_exchange_coefficient():
    """The quantity the model actually uses is ``f_m``, and it depends only
    weakly on ``zeta`` near neutral. So where the previous test allows
    ``zeta`` to sit a per cent from the root, the coefficient is still
    right to parts in 1e4.

    Checked against the exact root of the closure, found by bisection
    rather than by Isca's Newton iteration, on the columns where the
    iteration is loosest.
    """
    from tropoi.physics.monin_obukhov import (stability_integral_m,
                                              stability_integral_t)
    opts = MoninObukhovOptions()
    rng = np.random.default_rng(20261114)
    n = 20000
    ts = rng.uniform(140.0, 300.0, n)
    theta = ts + rng.uniform(-60.0, 60.0, n)
    z = 10.0 ** rng.uniform(-1.0, 2.5, n)
    speed = 10.0 ** rng.uniform(-4.0, 2.0, n)
    z0 = 10.0 ** rng.uniform(-6.0, -1.0, n)
    e = surface_drag(theta, ts, z, speed, gravity=G_MARS,
                     roughness_momentum=z0, roughness_heat=z0, opts=opts)
    zz = np.maximum(z, z0)
    ln = np.log(zz / z0)

    def closure_at(zeta, k):
        za, lk = np.array([zeta]), np.array([ln[k]])
        ratio = zz[k] / z0[k]
        f_m = stability_integral_m(za, za / ratio, lk, opts)
        f_t = stability_integral_t(za, za / ratio, lk, opts)
        return float((za * f_t / (f_m * f_m))[0])

    f_m_all = stability_integral_m(e.zeta, e.zeta / (zz / z0), ln, opts)
    f_t_all = stability_integral_t(e.zeta, e.zeta / (zz / z0), ln, opts)
    residual = np.abs(e.zeta * f_t_all / (f_m_all * f_m_all) - e.richardson) \
        / np.abs(e.richardson)
    sel = ~e.capped & e.converged & (np.abs(e.zeta) > 1e-6)
    order = np.argsort(-np.where(sel & (residual > 1e-4), residual, 0.0))[:60]

    worst = 0.0
    checked = 0
    for k in order:
        target = float(e.richardson[k])
        lo = float(e.zeta[k]) * 0.5 - 1e-3
        hi = float(e.zeta[k]) * 2.0 + 1e-3
        f_lo, f_hi = closure_at(lo, k) - target, closure_at(hi, k) - target
        if f_lo * f_hi > 0.0:
            continue
        for _ in range(200):                     # plain bisection, no SciPy
            mid = 0.5 * (lo + hi)
            if (closure_at(mid, k) - target) * f_lo > 0.0:
                lo, f_lo = mid, closure_at(mid, k) - target
            else:
                hi = mid
        root = 0.5 * (lo + hi)
        exact = (VON_KARMAN / float(stability_integral_m(
            np.array([root]), np.array([root]) / (zz[k] / z0[k]),
            np.array([ln[k]]), opts)[0])) ** 2
        worst = max(worst, abs(float(e.drag_m[k]) - exact) / exact)
        checked += 1
    assert checked >= 20, "the sweep should offer loose columns to check"
    assert worst < 5e-3, f"coefficient error {worst:.2e} against the exact root"


def test_the_sweep_converges_everywhere_and_stays_finite():
    rng = np.random.default_rng(5)
    n = 50000
    ts = rng.uniform(140.0, 320.0, n)
    theta = ts + rng.uniform(-80.0, 80.0, n)
    z = 10.0 ** rng.uniform(-1.0, 3.0, n)
    speed = 10.0 ** rng.uniform(-5.0, 2.0, n)
    for opts in (MoninObukhovOptions(), MoninObukhovOptions(stable_option=2),
                 MoninObukhovOptions(rich_crit=10.0)):
        e = surface_drag(theta, ts, z, speed, gravity=G_MARS,
                         roughness_momentum=Z0_MARS, roughness_heat=Z0_MARS,
                         opts=opts)
        e.require_convergence()                         # no column fails
        for name in ("drag_m", "drag_t", "u_star", "b_star", "zeta"):
            assert np.isfinite(getattr(e, name)).all(), name
        assert bool((e.drag_m >= opts.drag_min).all())


def test_a_failed_iteration_is_raised_and_never_quietly_replaced():
    """Starve the iteration and check that the failure is reported, that the
    report names the offending columns, and that nothing substituted a
    neutral or clipped coefficient for them."""
    rng = np.random.default_rng(3)
    n = 400
    ts = rng.uniform(150.0, 300.0, n)
    theta = ts + rng.uniform(-40.0, 40.0, n)
    z = np.full(n, 10.0)
    rich = -z * G_MARS * (ts - theta) / ts / (25.0 + ISCA_SMALL)
    mask = rich < 0.95 * 2.0
    f_m, f_t, f_q, zeta, converged = solve_zeta(
        rich, z, np.full(n, Z0_MARS), np.full(n, Z0_MARS), np.full(n, Z0_MARS),
        mask, MoninObukhovOptions(), max_iter=1)
    assert not bool(converged[mask].all()), "one iteration should not converge"

    from tropoi.physics.monin_obukhov import SurfaceExchange
    broken = SurfaceExchange(drag_m=np.ones(n), drag_t=np.ones(n), drag_q=np.ones(n),
                             u_star=np.ones(n), b_star=np.ones(n), richardson=rich,
                             zeta=zeta, converged=converged, capped=~mask)
    with pytest.raises(MoninObukhovConvergenceError) as excinfo:
        broken.require_convergence()
    assert "did not converge" in str(excinfo.value)
    assert "Richardson" in str(excinfo.value)
    assert excinfo.value.count == int((~converged & mask).sum())


# ---------------------------------------------------------------------------
# relation to the FHZ06 scheme it is an alternative to
# ---------------------------------------------------------------------------

def test_the_neutral_option_reproduces_the_fhz06_k_profile():
    """With the stability functions bypassed, the Isca-structured profile is
    the FHZ06 one, which is why the existing scheme was already consistent
    with Isca in the neutral limit."""
    h = np.array([800.0, 1500.0, 250.0])
    z = np.linspace(1.0, 2000.0, 40)[:, None] * np.ones((1, 3))
    u_star = np.array([0.3, 0.12, 0.6])
    k_m, k_h = monin_obukhov_diffusivity_profile(
        z, u_star, np.zeros(3), h, options=MoninObukhovOptions(neutral=True))
    reference = diffusivity_profile(z, u_star, h)
    assert k_m == pytest.approx(reference, rel=1e-13, abs=1e-13)
    assert k_h == pytest.approx(k_m, rel=0, abs=0)
    # a vanishing buoyancy scale gives the same thing with phi switched on
    k_m0, _ = monin_obukhov_diffusivity_profile(z, u_star, np.zeros(3), h)
    assert k_m0 == pytest.approx(reference, rel=1e-13, abs=1e-13)
    assert float(np.abs(k_m[z >= h]).max()) == 0.0


def test_the_stability_dependent_profile_moves_the_right_way():
    h = np.full(3, 1000.0)
    z = np.linspace(5.0, 900.0, 30)[:, None] * np.ones((1, 3))
    u_star = np.full(3, 0.3)
    neutral = diffusivity_profile(z, u_star, h)
    unstable_m, unstable_h = monin_obukhov_diffusivity_profile(
        z, u_star, np.full(3, 0.01), h)
    stable_m, stable_h = monin_obukhov_diffusivity_profile(
        z, u_star, np.full(3, -0.01), h)
    assert bool((unstable_m > neutral).all()) and bool((stable_m < neutral).all())
    assert bool((unstable_h > unstable_m).all())
    assert stable_h == pytest.approx(stable_m, rel=1e-12)


def test_the_monin_obukhov_flux_matches_fhz06_in_the_neutral_limit():
    """The two schemes drive the flux with different thermodynamic
    differences - dry static energy against surface-referenced potential
    temperature. At neutral stratification both reduce to the same
    logarithmic coefficient, so the fluxes may differ only by that
    second-order thermodynamic term."""
    ts = np.array([240.0])
    z = np.array([10.0])
    # choose the air temperature so the FHZ06 dry-static-energy difference
    # vanishes: cp*Ts = cp*Ta + g*z
    t_air = ts - G_MARS * z / CP_MARS
    rho = np.array([0.015])
    speed = np.array([8.0])
    fhz, c_fhz = sensible_heat_flux(ts, t_air, z, speed, rho, CP_MARS, G_MARS,
                                    roughness_length=Z0_MARS)
    mo, exchange = monin_obukhov_sensible_heat_flux(
        ts, t_air, z, speed, rho, CP_MARS, R_MARS, G_MARS, roughness_length=Z0_MARS)
    ln = np.log(10.0 / Z0_MARS)
    assert float(c_fhz) == pytest.approx((VON_KARMAN / ln) ** 2, rel=1e-12)
    assert float(exchange.drag_t) == pytest.approx((VON_KARMAN / ln) ** 2, rel=1e-8)
    assert float(fhz) == pytest.approx(0.0, abs=1e-12)
    # What is left of the Monin-Obukhov flux is exactly the second order of
    # x = g z / (cp T_a), and it is worth pinning rather than bounding:
    #   T_s - theta_a = (T_a + g z/cp) - T_a e^x = -T_a (x^2/2 + ...)
    x = G_MARS * float(z) / (CP_MARS * float(t_air))
    predicted = (rho * exchange.drag_t * speed * CP_MARS
                 * (float(ts) - float(t_air) * np.exp(x)))
    assert float(mo) == pytest.approx(float(predicted), rel=1e-12)
    assert float(mo) == pytest.approx(
        -float(rho * exchange.drag_t * speed * CP_MARS) * float(t_air) * x * x / 2.0,
        rel=1e-3)
    # which is 1.7e-8 of the flux a one-kelvin difference would drive
    scale = rho * exchange.drag_t * speed * CP_MARS * float(ts)
    assert abs(float(mo)) / float(scale) < 3e-8


def test_the_depth_diagnosis_agrees_with_iscas_potential_temperature_form():
    """Isca's ``pbl_depth`` builds its Richardson number from
    ``T + (g/cp) z`` where this code uses ``cp T + g z``; the ``cp``
    cancels, so the two are the same number."""
    rng = np.random.default_rng(11)
    k, n = 12, 50
    t = rng.uniform(150.0, 280.0, (k, n))
    z = np.cumsum(rng.uniform(50.0, 400.0, (k, n))[::-1], axis=0)[::-1]
    wind_sq = rng.uniform(1.0, 400.0, (k, n))
    dse = CP_MARS * t + G_MARS * z
    svcp = t + (G_MARS / CP_MARS) * z
    h_dse = boundary_layer_depth(z, dse, wind_sq, G_MARS, wind_floor=0.0,
                                 wind_offset=ISCA_SMALL)
    h_svcp = boundary_layer_depth(z, svcp, wind_sq, G_MARS, wind_floor=0.0,
                                  wind_offset=ISCA_SMALL)
    assert h_dse == pytest.approx(h_svcp, rel=1e-12)


def test_the_depth_offset_and_clamp_differ_only_where_the_wind_vanishes():
    k, n = 6, 3
    z = np.array([[3000.0], [2000.0], [1200.0], [600.0], [200.0], [20.0]]) * np.ones((1, n))
    t = np.full((k, n), 230.0)
    dse = CP_MARS * t + G_MARS * z
    # at any finite wind the two differ only by the offset's relative weight,
    # offset / |v|^2, which is 1e-6 here - not zero, just negligible
    strong = np.full((k, n), 100.0)
    assert boundary_layer_depth(z, dse, strong, G_MARS) == pytest.approx(
        boundary_layer_depth(z, dse, strong, G_MARS, wind_floor=0.0,
                             wind_offset=ISCA_SMALL), rel=1e-5)
    calm = np.full((k, n), 1e-14)
    clamped = boundary_layer_depth(z, dse, calm, G_MARS)
    offset = boundary_layer_depth(z, dse, calm, G_MARS, wind_floor=0.0,
                                  wind_offset=ISCA_SMALL)
    # the clamp makes a calm, stably stratified column look critical at the
    # first level above the surface; the offset makes it look neutral
    assert float(clamped[0]) < float(offset[0])


def test_giving_both_wind_regularizations_is_rejected():
    z = np.ones((2, 1))
    with pytest.raises(ValueError, match="not both"):
        boundary_layer_depth(z, z, z, G_MARS, wind_floor=1e-6, wind_offset=ISCA_SMALL)


# ---------------------------------------------------------------------------
# the similarity functions themselves
# ---------------------------------------------------------------------------

def test_the_similarity_functions_are_one_at_neutral_and_continuous():
    opts = MoninObukhovOptions()
    assert float(stability_derivative_m(np.zeros(1), opts)) == 1.0
    assert float(stability_derivative_t(np.zeros(1), opts)) == 1.0
    eps = 1e-12
    for fn in (stability_derivative_m, stability_derivative_t):
        below = float(fn(np.array([-eps]), opts))
        above = float(fn(np.array([eps]), opts))
        assert below == pytest.approx(1.0, abs=1e-10)
        assert above == pytest.approx(1.0, abs=1e-10)


def test_stable_option_two_joins_continuously_at_the_transition():
    opts = MoninObukhovOptions(stable_option=2, zeta_trans=0.5)
    zt = opts.zeta_trans
    for fn in (stability_derivative_m, stability_derivative_t):
        low = float(fn(np.array([zt - 1e-10]), opts))
        high = float(fn(np.array([zt + 1e-10]), opts))
        assert low == pytest.approx(high, rel=1e-8)
        assert low == pytest.approx(1.0 + 5.0 * zt, rel=1e-8)
    # and the two options agree in the weakly stable limit only near zero
    assert float(stability_derivative_m(np.array([1e-6]), opts)) == pytest.approx(
        float(stability_derivative_m(np.array([1e-6]), MoninObukhovOptions())), rel=1e-5)


def test_the_options_reject_values_the_formulation_cannot_use():
    with pytest.raises(ValueError, match="stable_option"):
        MoninObukhovOptions(stable_option=3)
    with pytest.raises(ValueError, match="rich_crit"):
        MoninObukhovOptions(rich_crit=0.0)
    with pytest.raises(ValueError, match="drag_min"):
        MoninObukhovOptions(drag_min=-1.0)
    with pytest.raises(ValueError, match="zeta_trans"):
        MoninObukhovOptions(stable_option=2, zeta_trans=0.0)


def test_the_isca_defaults_are_the_grey_mars_values():
    """grey_mars_test_case.py sets no ``monin_obukhov_nml`` block, so the
    module defaults are what that experiment runs."""
    o = MoninObukhovOptions()
    assert (o.rich_crit, o.drag_min, o.neutral, o.stable_option, o.zeta_trans) == \
        (2.0, 1.0e-5, False, 1, 0.5)
    assert o.b_stab == 0.5


# ---------------------------------------------------------------------------
# the configuration contract
# ---------------------------------------------------------------------------

def test_the_most_preset_selects_the_scheme_on_both_sides_and_keeps_everything_else():
    cfg = PRESETS["dev-mars-gray-flat-bl-most"]()
    bl = PRESETS["dev-mars-gray-flat-bl"]()
    assert cfg.experiment_id == "dev-mars-gray-flat-bl-most-heated-T21L25"
    assert cfg.sensible["scheme"] == "monin_obukhov"
    assert cfg.boundary_layer["scheme"] == "monin_obukhov"
    assert cfg.sensible["monin_obukhov"] == {} == cfg.boundary_layer["monin_obukhov"]
    # the FHZ06 critical Richardson number is gone from both, because its
    # role passes to monin_obukhov.rich_crit
    assert "critical_richardson" not in cfg.sensible
    assert "critical_richardson" not in cfg.boundary_layer
    # everything else is the boundary-layer preset
    assert (cfg.days, cfg.drag, cfg.convection) == (bl.days, bl.drag, bl.convection)
    assert cfg.sponge == bl.sponge and cfg.sensible["implicit"] is True
    assert cfg.with_(sensible=bl.sensible, boundary_layer=bl.boundary_layer,
                     experiment_id=bl.experiment_id) == bl


def test_only_opted_in_presets_serialize_boundary_layer_and_most():
    boundary_layer_presets = {
        "dev-mars-gray-flat-bl",
        "dev-mars-gray-flat-bl-most",
        "dev-thai-ben1",
        "dev-thai-ben2",
        "dev-thai-ben1-ck",
        "dev-thai-ben2-ck",
    }
    for name, factory in PRESETS.items():
        d = factory().to_dict()
        assert ("boundary_layer" in d) == (name in boundary_layer_presets), name
        assert ("scheme" in (d.get("sensible") or {})) == (
            name == "dev-mars-gray-flat-bl-most"), name


def test_the_most_preset_round_trips_through_json_with_its_hash():
    cfg = PRESETS["dev-mars-gray-flat-bl-most"]()
    again = RadiativeConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.sha256() == cfg.sha256()


def test_half_a_scheme_is_a_configuration_error():
    cfg = PRESETS["dev-mars-gray-flat-bl-most"]()
    with pytest.raises(ValueError, match="must be the same for both"):
        cfg.with_(boundary_layer={**cfg.boundary_layer, "scheme": "fhz06"})
    with pytest.raises(ValueError, match="needs both"):
        cfg.with_(boundary_layer=None)
    with pytest.raises(ValueError, match="same monin_obukhov options"):
        cfg.with_(boundary_layer={**cfg.boundary_layer,
                                  "monin_obukhov": {"rich_crit": 10.0}})
    with pytest.raises(ValueError, match="disagree on roughness_length"):
        cfg.with_(boundary_layer={**cfg.boundary_layer, "roughness_length": 1e-3})


def test_the_fhz06_critical_richardson_is_rejected_rather_than_ignored():
    from tropoi.physics.column_model import ColumnPhysics
    cfg = PRESETS["dev-mars-gray-flat-bl-most"]()
    bad = dict(cfg.sensible)
    bad["critical_richardson"] = 1.0
    with pytest.raises(ValueError, match="rich_crit"):
        ColumnPhysics(radiation=None, gravity=G_MARS, cp=CP_MARS,
                      gas_constant=R_MARS, surface=object(), sensible=bad)


def test_an_unknown_scheme_is_rejected():
    from tropoi.physics.column_model import ColumnPhysics
    with pytest.raises(ValueError, match="scheme must be one of"):
        ColumnPhysics(radiation=None, gravity=G_MARS, cp=CP_MARS,
                      gas_constant=R_MARS, surface=object(),
                      sensible={"scheme": "louis"})


def test_the_boundary_layer_signature_records_the_scheme_only_when_chosen():
    """An FHZ06 term's signature is byte for byte what it always was."""
    from tropoi.temporal.tendencies import boundary_layer as mod

    class _Term(mod.BoundaryLayerDiffusion):
        def __init__(self, **kw):           # skip the CuPy model wiring
            self.roughness_length = 3.21e-5
            self.critical_richardson = 1.0
            self.rich_crit_pbl = 1.0
            self.frac_inner = 0.1
            self.gustiness = 0.0
            self.heating = True
            self.scheme = kw.get("scheme", "fhz06")
            self.roughness_heat = 3.21e-5
            self.fused_kernel = kw.get("fused_kernel", False)
            self.monin_obukhov = (MoninObukhovOptions()
                                  if self.scheme == "monin_obukhov" else None)

    assert _Term().signature() == {
        "kind": "boundary_layer_diffusion", "roughness_length": 3.21e-5,
        "critical_richardson": 1.0, "rich_crit_pbl": 1.0, "frac_inner": 0.1,
        "gustiness": 0.0, "heating": True}
    most = _Term(scheme="monin_obukhov").signature()
    assert "critical_richardson" not in most
    assert most["scheme"] == "monin_obukhov"
    assert most["monin_obukhov"]["rich_crit"] == 2.0
    assert "fused_kernel" not in most
    fused = _Term(scheme="monin_obukhov", fused_kernel=True).signature()
    assert fused["fused_kernel"] is True


# ---------------------------------------------------------------------------
# GPU: the two backends and the fused kernel
# ---------------------------------------------------------------------------

def _gpu_sweep(n=60000, seed=99):
    rng = np.random.default_rng(seed)
    ts = rng.uniform(140.0, 320.0, n)
    return dict(theta=ts + rng.uniform(-70.0, 70.0, n), ts=ts,
                z=10.0 ** rng.uniform(-1.0, 2.5, n),
                speed=10.0 ** rng.uniform(-4.0, 2.0, n),
                z0=10.0 ** rng.uniform(-6.0, -1.0, n))


@needs_cuda
@pytest.mark.parametrize("opts", [MoninObukhovOptions(),
                                  MoninObukhovOptions(stable_option=2),
                                  MoninObukhovOptions(rich_crit=10.0)])
def test_the_vectorized_solve_agrees_between_numpy_and_cupy(opts):
    import cupy as cp
    s = _gpu_sweep()
    host = surface_drag(s["theta"], s["ts"], s["z"], s["speed"], gravity=G_MARS,
                        roughness_momentum=s["z0"], roughness_heat=s["z0"], opts=opts)
    device = surface_drag(cp.asarray(s["theta"]), cp.asarray(s["ts"]),
                          cp.asarray(s["z"]), cp.asarray(s["speed"]), gravity=G_MARS,
                          roughness_momentum=cp.asarray(s["z0"]),
                          roughness_heat=cp.asarray(s["z0"]), opts=opts)
    for name in ("drag_m", "drag_t", "u_star", "b_star", "zeta", "richardson"):
        a = getattr(host, name)
        b = cp.asnumpy(getattr(device, name))
        scale = max(float(np.abs(a).max()), 1e-300)
        assert float(np.abs(b - a).max()) < 1e-11 * scale, name
    assert np.array_equal(host.capped, cp.asnumpy(device.capped))


@needs_cuda
@pytest.mark.parametrize("opts", [MoninObukhovOptions(),
                                  MoninObukhovOptions(stable_option=2),
                                  MoninObukhovOptions(rich_crit=10.0)])
def test_the_fused_kernel_agrees_with_the_vectorized_solve(opts):
    """One thread per column must reproduce the vectorized solve, because a
    column's iteration never depends on another's. Not bit for bit: the
    device and host transcendentals differ in the last ulp, and the
    iteration's stopping rule is a relative 1e-4."""
    import cupy as cp
    from tropoi.physics.monin_obukhov_cuda import surface_drag_fused
    s = _gpu_sweep()
    host = surface_drag(s["theta"], s["ts"], s["z"], s["speed"], gravity=G_MARS,
                        roughness_momentum=s["z0"], roughness_heat=s["z0"], opts=opts)
    fused = surface_drag_fused(
        cp.asarray(s["theta"]), cp.asarray(s["ts"]), cp.asarray(s["z"]),
        cp.asarray(s["speed"]), gravity=G_MARS, roughness_momentum=cp.asarray(s["z0"]),
        roughness_heat=cp.asarray(s["z0"]), opts=opts)
    for name in ("drag_m", "drag_t", "drag_q", "u_star", "b_star", "zeta"):
        a = getattr(host, name)
        b = cp.asnumpy(getattr(fused, name))
        scale = max(float(np.abs(a).max()), 1e-300)
        assert float(np.abs(b - a).max()) < 1e-10 * scale, name
    assert np.array_equal(host.capped, cp.asnumpy(fused.capped))
    assert np.array_equal(host.converged, cp.asnumpy(fused.converged))


@needs_cuda
def test_the_fused_kernel_reproduces_iscas_drag_checksum_too():
    """The aggregate reference test, on the device."""
    import cupy as cp
    from tropoi.physics.monin_obukhov_cuda import surface_drag_fused
    e = surface_drag_fused(
        cp.asarray(DRAG_CASE["pt"]), cp.asarray(DRAG_CASE["pt0"]),
        cp.asarray(DRAG_CASE["z"]), cp.asarray(DRAG_CASE["speed"]),
        gravity=KERNEL_TEST_GRAVITY, roughness_momentum=cp.asarray(DRAG_CASE["z0"]),
        roughness_heat=cp.asarray(DRAG_CASE["zt"]),
        roughness_moisture=cp.asarray(DRAG_CASE["zq"]), opts=KERNEL_TEST_OPTS)
    total = _wrap(sum(_bits(_fortran_sum(cp.asnumpy(a))) for a in
                      (e.drag_m, e.drag_t, e.drag_q, e.u_star, e.b_star)))
    assert total - CHKSUM_DRAG == 0, f"{total - CHKSUM_DRAG} ulp from Isca"


@needs_cuda
def test_the_fused_kernel_is_elementwise_identical_on_the_reference_case():
    """Elementwise bit identity, between this implementation and the kernel.

    This is the one place a bit-level *elementwise* claim is warranted: both
    sides are ours, so every element can be compared directly rather than
    through a published aggregate. On Isca's own ``test_drag`` inputs the
    fused kernel reproduces the NumPy path exactly, element for element, in
    every output but the Richardson number.

    The Richardson number differs by up to 2 ulp, and the cause is pinned
    rather than bounded: ``nvcc`` contracts ``s*s + small`` into
    ``fma(s, s, small)`` by default, so the denominator is rounded once on
    the device and twice in NumPy. Asserting *that* is stronger than any ulp
    budget - the kernel must equal the contracted expression exactly, which
    leaves no room for a second source of difference to hide. ``-fmad`` is
    left at its default deliberately; see the kernel module.

    It does **not** generalize: the randomized sweeps show ~1e-13 relative
    differences, which is why the kernel is opt-in. See
    :func:`test_the_fused_kernel_agrees_with_the_vectorized_solve`.
    """
    import cupy as cp
    from tropoi.physics.monin_obukhov_cuda import surface_drag_fused
    host = _drag_case()
    fused = surface_drag_fused(
        cp.asarray(DRAG_CASE["pt"]), cp.asarray(DRAG_CASE["pt0"]),
        cp.asarray(DRAG_CASE["z"]), cp.asarray(DRAG_CASE["speed"]),
        gravity=KERNEL_TEST_GRAVITY, roughness_momentum=cp.asarray(DRAG_CASE["z0"]),
        roughness_heat=cp.asarray(DRAG_CASE["zt"]),
        roughness_moisture=cp.asarray(DRAG_CASE["zq"]), opts=KERNEL_TEST_OPTS)
    for name in ("drag_m", "drag_t", "drag_q", "u_star", "b_star"):
        a = getattr(host, name)
        b = cp.asnumpy(getattr(fused, name))
        assert np.array_equal(a, b), f"{name} is not elementwise identical"
    kernel_rich = cp.asnumpy(fused.richardson)
    host_rich = np.asarray(host.richardson)
    # Rebuild the Richardson number with the denominator contracted into a
    # single-rounded fma, which is what the device computes.
    z = np.asarray(DRAG_CASE["z"], dtype=np.float64)
    speed = np.asarray(DRAG_CASE["speed"], dtype=np.float64)
    pt = np.asarray(DRAG_CASE["pt"], dtype=np.float64)
    pt0 = np.asarray(DRAG_CASE["pt0"], dtype=np.float64)
    delta_b = KERNEL_TEST_GRAVITY * (pt0 - pt) / pt0
    contracted = np.asarray([
        np.float64(-zi * db) / _fma(si, si, ISCA_SMALL)
        for zi, db, si in zip(z, delta_b, speed)])
    assert np.array_equal(kernel_rich, contracted), (
        "the kernel's Richardson number is not the fma-contracted expression, "
        "so something other than fma contraction differs")
    # And that contraction costs no more than a couple of eps.
    assert np.max(np.abs(kernel_rich - host_rich)
                  / np.abs(host_rich)) < 4.0 * np.finfo(np.float64).eps


@needs_cuda
def test_the_fused_kernel_refuses_the_neutral_branch_it_does_not_implement():
    import cupy as cp
    from tropoi.physics.monin_obukhov_cuda import surface_drag_fused
    with pytest.raises(ValueError, match="neutral"):
        surface_drag_fused(cp.asarray([240.0]), cp.asarray([240.0]),
                           cp.asarray([10.0]), cp.asarray([5.0]), gravity=G_MARS,
                           roughness_momentum=Z0_MARS, roughness_heat=Z0_MARS,
                           opts=MoninObukhovOptions(neutral=True))


def test_the_fused_kernel_is_opt_in_and_only_for_this_scheme():
    from tropoi.temporal.tendencies.boundary_layer import BoundaryLayerDiffusion
    with pytest.raises(ValueError, match="fused_kernel applies only"):
        BoundaryLayerDiffusion.__init__(
            object.__new__(BoundaryLayerDiffusion), None, None, gravity=1.0, cp=1.0,
            gas_constant=1.0, dt=1.0, roughness_length=1.0, fused_kernel=True)


# ---------------------------------------------------------------------------
# GPU: a short coupled run
# ---------------------------------------------------------------------------

def _tiny_most(days: int, experiment_id: str, **over) -> RadiativeConfig:
    cfg = PRESETS["dev-mars-gray-flat-bl-most"](days=days).with_(
        l_max=10, retained_truncation=9, nlat=16, nlon=32, experiment_id=experiment_id,
        checkpoint_every_days=1, state_every_days=1)
    return cfg.with_(**over) if over else cfg


@needs_cuda
def test_a_coupled_monin_obukhov_day_runs_and_closes_its_budget(tmp_path):
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny_most(1, "mars-most-test")
    run = RadiativeRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None)
    assert run.boundary_layer is not None
    assert run.boundary_layer.scheme == "monin_obukhov"
    run.run()
    series = json.loads((tmp_path / "run" / "series.json").read_text())
    assert abs(series["energy_residual"][-1]) < 1e-2 * series["absorbed_stellar"][-1]
    assert series["bl_kinetic_sink"][-1] <= 0.0
    assert series["bl_max_depth"][-1] >= 0.0
    slab = cfg.surface["heat_capacity"] * (series["mean_surface_temperature"][-1]
                                           - series["mean_surface_temperature"][0])
    assert slab == pytest.approx(series["surface_net"][-1] * 86400.0, rel=1e-9)


@needs_cuda
def test_a_resumed_monin_obukhov_run_is_identical(tmp_path):
    import cupy as cp
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny_most(2, "mars-most-resume")
    straight = RadiativeRun(cfg, tmp_path / "a", allow_dirty=True, log=lambda s: None)
    straight.run()
    split = RadiativeRun(cfg, tmp_path / "b", allow_dirty=True, log=lambda s: None)
    split.run(until_day=1)
    resumed = RadiativeRun(cfg, tmp_path / "b", allow_dirty=True, log=lambda s: None)
    resumed.run()
    assert (cp.asnumpy(straight.stepper.state) == cp.asnumpy(resumed.stepper.state)).all()
    assert json.loads((tmp_path / "a" / "series.json").read_text()) == \
        json.loads((tmp_path / "b" / "series.json").read_text())


@needs_cuda
def test_the_monin_obukhov_run_differs_from_the_fhz06_run_it_replaces(tmp_path):
    """The point of the scheme: same configuration otherwise, different
    surface layer, so the state must actually move."""
    import cupy as cp
    from tropoi.run.radiative.experiment import RadiativeRun
    most = RadiativeRun(_tiny_most(1, "most-vs-fhz-most"), tmp_path / "most",
                        allow_dirty=True, log=lambda s: None)
    most.run()
    base_cfg = PRESETS["dev-mars-gray-flat-bl"](days=1).with_(
        l_max=10, retained_truncation=9, nlat=16, nlon=32,
        experiment_id="most-vs-fhz-fhz", checkpoint_every_days=1, state_every_days=1)
    fhz = RadiativeRun(base_cfg, tmp_path / "fhz", allow_dirty=True, log=lambda s: None)
    fhz.run()
    a = cp.asnumpy(most.stepper.state)
    b = cp.asnumpy(fhz.stepper.state)
    assert a.shape == b.shape
    assert not np.allclose(a, b), "the two surface layers should not agree"
