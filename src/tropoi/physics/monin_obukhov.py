"""Monin-Obukhov similarity: surface exchange coefficients and stability
dependent vertical diffusivities, in the formulation Isca's grey-Mars
experiment runs.

This is an independent Python reimplementation of the published
Monin-Obukhov surface-layer formulation of the GFDL Flexible Modeling
System as carried by Isca (``ExeClim/Isca``, GPL-3.0) at commit
``d1321ac3``: ``src/atmos_param/monin_obukhov/monin_obukhov.F90`` and its
kernel ``monin_obukhov_kernel.F90``, with the diffusivity structure of
``src/atmos_param/diffusivity/diffusivity.F90`` and the flux assembly of
``src/coupler/surface_flux.F90``. No Fortran source is copied; the
formulation, its numerical safeguards and its defaults were read from
those files and re-expressed here. See ``docs/radiation/MONIN_OBUKHOV.md``
for the provenance record, the verified agreement and the differences that
remain. That record keeps two kinds of agreement apart, and so should any
summary of it: Isca's three published kernel checksums reproduce *exactly*,
but each is a sum of bit patterns of per-array sums, so it constrains
aggregates rather than individual elements; the ten per-column ``u_star``
and ``b_star`` literals of Isca's ``test_profile`` are what establish
*elementwise* agreement, to a few ulp.

What the scheme is
------------------
The surface layer is assumed to obey Monin-Obukhov similarity: the
dimensionless wind and buoyancy gradients are universal functions
``phi_m(zeta)``, ``phi_t(zeta)`` of the stability parameter
``zeta = z / L``, with ``L`` the Obukhov length. Integrating them from the
roughness heights to the lowest model level gives the profile functions
``f_m``, ``f_t``, ``f_q``, and the exchange coefficients follow as

    C_m = (k / f_m)^2,    C_t = (k / f_m)(k / f_t),    C_q = (k / f_m)(k / f_q).

``zeta`` is not known in advance: it depends on the fluxes, which depend on
the coefficients. The closure is a bulk Richardson number ``Ri`` of the
lowest level, and ``zeta`` is recovered from it by a Newton iteration on

    Ri(zeta) = zeta f_t(zeta) / f_m(zeta)^2,

which is :func:`solve_zeta`. This replaces the FHZ06 shortcut
(:func:`tropoi.physics.surface.drag_coefficient`), which uses the neutral
coefficient reduced by an algebraic factor and never iterates.

Stability functions (``stable_option``)
---------------------------------------
Unstable (``zeta < 0``), the Businger-Dyer forms

    phi_t = (1 - 16 zeta)^(-1/2),   phi_m = (1 - 16 zeta)^(-1/4).

Stable (``zeta >= 0``), two options; Isca's default and its grey-Mars
setting is ``1``:

* ``stable_option = 1``: ``phi = 1 + zeta (5 + b zeta) / (1 + zeta)``,
  ``b = 1 / rich_crit`` - a rational form that stays finite and tends to
  ``b zeta`` at large ``zeta``, so mixing never vanishes identically.
* ``stable_option = 2``: linear ``1 + 5 zeta`` below ``zeta_trans``,
  joined continuously to ``lambda + b zeta`` above it.

Both branches of ``phi`` are used twice over: as differential functions in
the Newton derivative and the diffusivity, and in integrated form
(:func:`stability_integral_m`, :func:`stability_integral_t`) in the profile
functions.

Defaults
--------
:class:`MoninObukhovOptions` carries Isca's ``monin_obukhov_nml``
*defaults*, which is what the grey-Mars test case runs: that experiment
sets no ``monin_obukhov_nml`` block at all, so ``rich_crit = 2.0``,
``drag_min = 1e-5``, ``stable_option = 1``, ``zeta_trans = 0.5``,
``neutral = False``. Note that ``rich_crit`` here (2.0, the surface layer)
is a different constant from ``rich_crit_pbl`` (1.0, the depth diagnosis in
:mod:`tropoi.physics.boundary_layer`) and from the FHZ06 critical
Richardson number the existing scheme uses (1.0).

The iteration constants are Isca's module parameters, not namelist
entries: 20 iterations, a relative tolerance of 1e-4, a ``zeta`` floor of
1e-6 below which the point is taken as neutral, a wind-squared offset of
1e-4 m^2/s^2 in the Richardson number, and a 1e-10 m/s floor on ``u_star``
in the diffusivity.

Failure is reported, never hidden
---------------------------------
:func:`solve_zeta` returns a per-column convergence mask and
:func:`surface_drag` carries it on :class:`SurfaceExchange`. Callers that
cannot act on a partial result raise :exc:`MoninObukhovConvergenceError`
through :meth:`SurfaceExchange.require_convergence`; nothing in this module
substitutes a neutral or a clipped coefficient for a column that failed to
converge. (Isca returns a global ``ier`` flag which its callers ignore.)

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = ["MoninObukhovOptions", "MoninObukhovConvergenceError", "SurfaceExchange",
           "ISCA_MAX_ITER", "ISCA_ERROR", "ISCA_ZETA_MIN", "ISCA_SMALL", "ISCA_USTAR_MIN",
           "stability_derivative_m", "stability_derivative_t", "stability_integral_m",
           "stability_integral_t", "solve_zeta", "surface_drag", "diffusivities",
           "stable_mix"]

# Isca monin_obukhov.F90 module parameters (not namelist entries).
ISCA_MAX_ITER = 20
ISCA_ERROR = 1.0e-4
ISCA_ZETA_MIN = 1.0e-6
ISCA_SMALL = 1.0e-4
ISCA_USTAR_MIN = 1.0e-10

VON_KARMAN = 0.4


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


class MoninObukhovConvergenceError(RuntimeError):
    """The Monin-Obukhov ``zeta`` iteration did not converge somewhere.

    Carries the number of offending columns and the extreme inputs among
    them, so a run that stops can be diagnosed without a rerun.
    """

    def __init__(self, count: int, total: int, detail: str = ""):
        self.count = int(count)
        self.total = int(total)
        message = (f"Monin-Obukhov zeta iteration did not converge in "
                   f"{ISCA_MAX_ITER} iterations for {count} of {total} columns")
        if detail:
            message += f"; {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class MoninObukhovOptions:
    """Isca ``monin_obukhov_nml``. The defaults are Isca's own, which is
    what the grey-Mars test case runs (it overrides none of them)."""

    rich_crit: float = 2.0
    drag_min: float = 1.0e-5
    neutral: bool = False
    stable_option: int = 1
    zeta_trans: float = 0.5

    def __post_init__(self):
        if self.stable_option not in (1, 2):
            raise ValueError("stable_option must be 1 or 2")
        if not self.rich_crit > 0.0:
            raise ValueError("rich_crit must be > 0")
        if self.drag_min < 0.0:
            raise ValueError("drag_min must be >= 0")
        if self.stable_option == 2 and not self.zeta_trans > 0.0:
            raise ValueError("zeta_trans must be > 0 for stable_option 2")

    @property
    def b_stab(self) -> float:
        return 1.0 / self.rich_crit

    @property
    def lambda_stab(self) -> float:
        """The ``stable_option = 2`` matching constant at ``zeta_trans``."""
        return 1.0 + (5.0 - self.b_stab) * self.zeta_trans

    def signature(self) -> dict:
        return {"rich_crit": self.rich_crit, "drag_min": self.drag_min,
                "neutral": self.neutral, "stable_option": self.stable_option,
                "zeta_trans": self.zeta_trans}


@dataclass
class SurfaceExchange:
    """One Monin-Obukhov surface-layer solve.

    ``drag_m``, ``drag_t``, ``drag_q`` are the dimensionless exchange
    coefficients C_m, C_t, C_q (Isca's ``cd_m``, ``cd_t``, ``cd_q``, i.e.
    before multiplication by the wind speed); ``u_star`` (m/s) and
    ``b_star`` (m/s^2) the friction velocity and buoyancy scale;
    ``richardson`` the bulk Richardson number the solve closed on and
    ``zeta`` the stability parameter z/L it converged to (so the Obukhov
    length is ``height / zeta``, and ``zeta`` is 0 on a capped column,
    which was never solved);
    ``converged`` a per-column mask, False only where the iteration ran and
    failed; ``capped`` marks the columns at or beyond ``0.95 rich_crit``,
    which are not iterated at all but assigned ``drag_min``. A capped
    column therefore reports ``converged = True`` - there was nothing to
    converge - and is identified by ``capped``.
    """

    drag_m: Any
    drag_t: Any
    drag_q: Any
    u_star: Any
    b_star: Any
    richardson: Any
    zeta: Any             # z / L, the converged stability parameter
    converged: Any
    capped: Any           # mask: Ri >= 0.95 rich_crit, coefficients = drag_min

    def require_convergence(self) -> "SurfaceExchange":
        """Raise :exc:`MoninObukhovConvergenceError` if any column failed.

        The capped (``Ri >= 0.95 rich_crit``) columns are not failures:
        Isca assigns ``drag_min`` there by design, because the Newton
        iteration becomes ill-conditioned as ``Ri`` approaches
        ``rich_crit``.
        """
        xp = _xp(self.drag_m)
        bad = ~self.converged & ~self.capped
        count = int(bad.sum())
        if count:
            ri = self.richardson
            lo = float(xp.min(ri[bad]))
            hi = float(xp.max(ri[bad]))
            raise MoninObukhovConvergenceError(
                count, int(self.converged.size),
                f"bulk Richardson number of the failing columns in [{lo:.6g}, {hi:.6g}]")
        return self


# ---------------------------------------------------------------------------
# similarity functions
# ---------------------------------------------------------------------------
def _split(zeta, xp):
    """Unstable mask plus the one-sided clamps that keep both branches of a
    similarity function finite everywhere, so they can be evaluated on the
    whole array and selected with ``where`` (no boolean indexing, no NaN,
    no warnings, and no host synchronization on a GPU)."""
    unstable = zeta < 0.0
    return unstable, xp.minimum(zeta, 0.0), xp.maximum(zeta, 0.0)


def _phi_stable(zeta_s, opts: MoninObukhovOptions, xp):
    """The stable branch of ``phi``, shared by momentum and heat (Isca's
    ``monin_obukhov_derivative_m`` and ``_t`` are identical when stable)."""
    if opts.stable_option == 1:
        return 1.0 + zeta_s * (5.0 + opts.b_stab * zeta_s) / (1.0 + zeta_s)
    lam = opts.lambda_stab
    weak = 1.0 + 5.0 * zeta_s
    strong = lam + opts.b_stab * zeta_s
    return xp.where(zeta_s < opts.zeta_trans, weak, strong)


def stability_derivative_t(zeta, opts: Optional[MoninObukhovOptions] = None):
    """``phi_t(zeta)``, the differential similarity function for buoyancy
    and tracers (Isca ``monin_obukhov_derivative_t``)."""
    xp = _xp(zeta)
    opts = MoninObukhovOptions() if opts is None else opts
    unstable, zu, zs = _split(zeta, xp)
    return xp.where(unstable, (1.0 - 16.0 * zu) ** (-0.5), _phi_stable(zs, opts, xp))


def stability_derivative_m(zeta, opts: Optional[MoninObukhovOptions] = None):
    """``phi_m(zeta)``, the differential similarity function for momentum
    (Isca ``monin_obukhov_derivative_m``). Unstable it is
    ``(1 - 16 zeta)^(-1/4)``, formed as Isca forms it - the square root of
    the ``-1/2`` power - so the rounding is the same."""
    xp = _xp(zeta)
    opts = MoninObukhovOptions() if opts is None else opts
    unstable, zu, zs = _split(zeta, xp)
    x = (1.0 - 16.0 * zu) ** (-0.5)
    return xp.where(unstable, xp.sqrt(x), _phi_stable(zs, opts, xp))


def stability_integral_m(zeta, zeta_0, ln_z_z0,
                         opts: Optional[MoninObukhovOptions] = None):
    """``f_m``, the integral similarity function for momentum (Isca
    ``monin_obukhov_integral_m``): the neutral ``ln(z/z0)`` plus the
    stability correction between ``zeta_0`` and ``zeta``."""
    xp = _xp(zeta)
    opts = MoninObukhovOptions() if opts is None else opts
    unstable, zu, zs = _split(zeta, xp)
    _, zu0, zs0 = _split(zeta_0, xp)
    # unstable: the Paulson/Businger-Dyer integral
    x = xp.sqrt(xp.sqrt(1.0 - 16.0 * zu))
    x_0 = xp.sqrt(xp.sqrt(1.0 - 16.0 * zu0))
    x1, x1_0 = 1.0 + x, 1.0 + x_0
    num = x1 * x1 * (1.0 + x * x)
    denom = x1_0 * x1_0 * (1.0 + x_0 * x_0)
    y = xp.arctan(x) - xp.arctan(x_0)
    psi_u = ln_z_z0 - xp.log(num / denom) + 2 * y
    if opts.stable_option == 1:
        psi_s = (ln_z_z0 + (5.0 - opts.b_stab) * xp.log((1.0 + zs) / (1.0 + zs0))
                 + opts.b_stab * (zs - zs0))
    else:
        lam, b, zt = opts.lambda_stab, opts.b_stab, opts.zeta_trans
        weak = ln_z_z0 + 5.0 * (zs - zs0)
        bend = (lam - 1.0) * xp.log(xp.maximum(zs, zt) / zt) + b * (zs - zt)
        strong_low = ln_z_z0 + bend + 5.0 * (zt - zs0)
        strong_high = lam * ln_z_z0 + b * (zs - zs0)
        strong = xp.where(zs0 <= zt, strong_low, strong_high)
        psi_s = xp.where(zs <= zt, weak, strong)
    return xp.where(unstable, psi_u, psi_s)


def stability_integral_t(zeta, zeta_t, ln_z_zt,
                         opts: Optional[MoninObukhovOptions] = None):
    """``f_t`` (and ``f_q``, the same function with the moisture roughness),
    the integral similarity function for heat and tracers (Isca
    ``monin_obukhov_integral_tq``)."""
    xp = _xp(zeta)
    opts = MoninObukhovOptions() if opts is None else opts
    unstable, zu, zs = _split(zeta, xp)
    _, zut, zst = _split(zeta_t, xp)
    x = xp.sqrt(1.0 - 16.0 * zu)
    x_t = xp.sqrt(1.0 - 16.0 * zut)
    psi_u = ln_z_zt - 2.0 * xp.log((1.0 + x) / (1.0 + x_t))
    if opts.stable_option == 1:
        psi_s = (ln_z_zt + (5.0 - opts.b_stab) * xp.log((1.0 + zs) / (1.0 + zst))
                 + opts.b_stab * (zs - zst))
    else:
        lam, b, zt = opts.lambda_stab, opts.b_stab, opts.zeta_trans
        weak = ln_z_zt + 5.0 * (zs - zst)
        bend = (lam - 1.0) * xp.log(xp.maximum(zs, zt) / zt) + b * (zs - zt)
        strong_low = ln_z_zt + bend + 5.0 * (zt - zst)
        strong_high = lam * ln_z_zt + b * (zs - zst)
        strong = xp.where(zst <= zt, strong_low, strong_high)
        psi_s = xp.where(zs <= zt, weak, strong)
    return xp.where(unstable, psi_u, psi_s)


# ---------------------------------------------------------------------------
# the nonlinear solve
# ---------------------------------------------------------------------------
def solve_zeta(richardson, z, z0, zt, zq, mask,
               opts: Optional[MoninObukhovOptions] = None, *,
               max_iter: int = ISCA_MAX_ITER, tolerance: float = ISCA_ERROR,
               zeta_min: float = ISCA_ZETA_MIN):
    """Recover the profile functions from the bulk Richardson number.

    Newton iteration on ``Ri(zeta) = zeta f_t / f_m^2`` (Isca
    ``monin_obukhov_solve_zeta``). All arrays are (npts,) and conformable;
    ``mask`` selects the columns to solve. Returns
    ``(f_m, f_t, f_q, zeta, converged)``, with the three profile functions set to
    their neutral values wherever ``|zeta|`` fell below ``zeta_min`` (a
    column that close to neutral is taken as neutral, as there) and left at
    zero outside ``mask``.

    The iteration detail that matters for reproducing Isca: a column drops
    out of the iteration the moment its own correction passes the
    tolerance, and its ``zeta`` is then frozen - the correction is applied
    only to columns still active. So the trajectory of each column is
    independent of the others, which is what makes the fused CUDA kernel in
    :mod:`tropoi.physics.monin_obukhov_cuda` able to reproduce this
    column-wise.

    Isca's own loop ends on a *global* test, ``maxval(corr) <= error``,
    where ``corr`` keeps its last value for columns that dropped out at the
    ``zeta_min`` test rather than by converging. A stale entry above the
    tolerance therefore runs the loop to ``max_iter`` and returns Isca's
    global ``ier = 1`` although every column has in fact converged. That is
    a status artefact only - no column's result changes - and this
    implementation reports convergence per column instead.
    """
    xp = _xp(richardson, z)
    opts = MoninObukhovOptions() if opts is None else opts
    f_m = xp.zeros_like(richardson)
    f_t = xp.zeros_like(richardson)
    f_q = xp.zeros_like(richardson)
    z_z0, z_zt, z_zq = z / z0, z / zt, z / zq
    ln_z_z0, ln_z_zt, ln_z_zq = xp.log(z_z0), xp.log(z_zt), xp.log(z_zq)

    active = mask.copy()
    # initial guess: the neutral-limit inversion of Ri(zeta), stretched by
    # 1 / (1 - Ri / Ri_c) on the stable side so it stays below the critical
    # value (Isca's guess, verbatim in form).
    zeta = xp.where(active, richardson * ln_z_z0 * ln_z_z0 / ln_z_zt, 0.0)
    stable = active & (richardson >= 0.0)
    zeta = xp.where(stable, zeta / (1.0 - richardson / opts.rich_crit), zeta)

    converged = ~mask                 # outside the mask there is nothing to solve

    for _ in range(int(max_iter)):
        # a column within zeta_min of neutral is taken as neutral and left
        near_neutral = active & (xp.abs(zeta) < zeta_min)
        zeta = xp.where(near_neutral, 0.0, zeta)
        f_m = xp.where(near_neutral, ln_z_z0, f_m)
        f_t = xp.where(near_neutral, ln_z_zt, f_t)
        f_q = xp.where(near_neutral, ln_z_zq, f_q)
        converged = converged | near_neutral
        active = active & ~near_neutral

        # evaluate on everything with zeta neutralized where inactive, so
        # no division by zero and no branch-dependent array shapes
        zeta_a = xp.where(active, zeta, 1.0)
        rzeta = 1.0 / zeta_a
        zeta_0 = zeta_a / z_z0
        zeta_t = zeta_a / z_zt
        zeta_q = zeta_a / z_zq

        phi_m = stability_derivative_m(zeta_a, opts)
        phi_m_0 = stability_derivative_m(zeta_0, opts)
        phi_t = stability_derivative_t(zeta_a, opts)
        phi_t_0 = stability_derivative_t(zeta_t, opts)

        fm = stability_integral_m(zeta_a, zeta_0, ln_z_z0, opts)
        ft = stability_integral_t(zeta_a, zeta_t, ln_z_zt, opts)
        fq = stability_integral_t(zeta_a, zeta_q, ln_z_zq, opts)
        f_m = xp.where(active, fm, f_m)
        f_t = xp.where(active, ft, f_t)
        f_q = xp.where(active, fq, f_q)

        # Newton step on Ri(zeta); d_rich is dRi/dzeta through the
        # logarithmic derivatives of f_m and f_t
        df_m = (phi_m - phi_m_0) * rzeta
        df_t = (phi_t - phi_t_0) * rzeta
        rich_1 = zeta_a * ft / (fm * fm)
        d_rich = rich_1 * (rzeta + df_t / ft - 2.0 * df_m / fm)
        correction = (richardson - rich_1) / d_rich
        # Isca's convergence measure: the *smaller* of the absolute and the
        # relative correction, so whichever is easier to satisfy ends the
        # iteration. For |zeta| < 1 that is the absolute one, which means
        # the relative accuracy in zeta is only tolerance / |zeta| - near
        # neutral, zeta can land a per cent or two from the true root. The
        # exchange coefficient is far less sensitive than that (f_m depends
        # weakly on zeta there), which is why the scheme gets away with it;
        # the measured figures are in docs/radiation/MONIN_OBUKHOV.md.
        measure = xp.minimum(xp.abs(correction), xp.abs(correction / zeta_a))
        passed = active & (measure <= tolerance)
        converged = converged | passed
        active = active & ~passed
        if not bool(active.any()):
            break
        zeta = xp.where(active, zeta + correction, zeta)

    return f_m, f_t, f_q, xp.where(mask, zeta, 0.0), converged


def surface_drag(potential_temperature, surface_temperature, height, speed, *,
                 gravity: float, roughness_momentum, roughness_heat,
                 roughness_moisture=None,
                 opts: Optional[MoninObukhovOptions] = None,
                 small: float = ISCA_SMALL) -> SurfaceExchange:
    """Monin-Obukhov surface exchange coefficients (Isca
    ``monin_obukhov_drag_1d``, reached through ``mo_drag``).

    ``potential_temperature`` is the lowest level's potential temperature
    referenced to the surface pressure (Isca's ``thv_atm``; with
    ``use_virtual_temp`` off, as grey-Mars sets it, the virtual correction
    is absent and this is ``T_a (p_s/p_a)^kappa``) and
    ``surface_temperature`` the surface value (``thv_surf``, likewise
    unmodified in a dry atmosphere). ``height`` is the level's height above
    the surface, ``speed`` the wind magnitude including any gustiness.

    ``roughness_moisture`` defaults to ``roughness_heat`` rather than to
    the grey-Mars value of 0 m. A zero moisture roughness sends
    ``ln(z/zq)`` to infinity and ``drag_q`` to zero, which is what Isca
    intends for a case with no moisture flux; this model is dry, so
    ``drag_q`` is never consumed, and the substitution only avoids
    manufacturing infinities. Passing ``roughness_moisture=0.0`` leaves
    every other output unchanged, because ``f_q`` enters nothing else.

    Three safeguards are Isca's and are kept:

    * the Richardson number divides by ``speed^2 + small`` with
      ``small = 1e-4`` m^2/s^2 - an offset, not a clamp, so a calm column
      gives ``Ri -> 0`` smoothly rather than a large stable value;
    * the height entering the profile functions is
      ``max(z, z0, zt, zq)``, so ``ln(z/z0)`` is never negative for a level
      nominally below a roughness length. It can still be exactly zero,
      when the level is at or below the roughness height, and the
      logarithmic law is genuinely singular there - the coefficient
      diverges. Isca shares that; see the limitation noted in
      ``docs/radiation/MONIN_OBUKHOV.md``;
    * columns at or beyond ``0.95 rich_crit`` skip the iteration and take
      ``drag_min`` (Isca backs off from ``rich_crit`` itself because the
      Newton step becomes ill-conditioned there), and every coefficient is
      floored at ``drag_min``.

    Returns a :class:`SurfaceExchange`. Call
    :meth:`SurfaceExchange.require_convergence` to turn a failed iteration
    into an exception.
    """
    xp = _xp(potential_temperature, surface_temperature)
    opts = MoninObukhovOptions() if opts is None else opts
    pt = potential_temperature
    pt0 = surface_temperature
    z = height
    z0 = xp.broadcast_to(xp.asarray(roughness_momentum, dtype=pt.dtype), pt.shape)
    zt = xp.broadcast_to(xp.asarray(roughness_heat, dtype=pt.dtype), pt.shape)
    if roughness_moisture is None:
        zq = zt
    else:
        zq = xp.broadcast_to(xp.asarray(roughness_moisture, dtype=pt.dtype), pt.shape)

    r_crit = 0.95 * opts.rich_crit
    sqrt_drag_min = float(np.sqrt(opts.drag_min)) if opts.drag_min != 0.0 else 0.0

    delta_b = gravity * (pt0 - pt) / pt0
    richardson = -z * delta_b / (speed * speed + small)
    zz = xp.maximum(xp.maximum(z, z0), xp.maximum(zt, zq))

    capped = richardson >= r_crit
    solve = ~capped
    f_m, f_t, f_q, zeta, converged = solve_zeta(richardson, zz, z0, zt, zq, solve, opts)

    # keep the division finite on the capped columns, whose f_* are unset
    safe = xp.where(solve, f_m, 1.0)
    us = xp.maximum(VON_KARMAN / safe, sqrt_drag_min)
    bs = xp.maximum(VON_KARMAN / xp.where(solve, f_t, 1.0), sqrt_drag_min)
    qs = xp.maximum(VON_KARMAN / xp.where(solve, f_q, 1.0), sqrt_drag_min)

    drag_m = xp.where(capped, opts.drag_min, us * us)
    drag_t = xp.where(capped, opts.drag_min, us * bs)
    drag_q = xp.where(capped, opts.drag_min, us * qs)
    u_star = xp.where(capped, sqrt_drag_min, us) * speed
    b_star = xp.where(capped, sqrt_drag_min, bs) * delta_b
    return SurfaceExchange(drag_m=drag_m, drag_t=drag_t, drag_q=drag_q,
                           u_star=u_star, b_star=b_star, richardson=richardson,
                           zeta=zeta, converged=converged, capped=capped)


# ---------------------------------------------------------------------------
# diffusivity
# ---------------------------------------------------------------------------
def diffusivities(z, u_star, b_star, opts: Optional[MoninObukhovOptions] = None, *,
                  von_karman: float = VON_KARMAN, ustar_min: float = ISCA_USTAR_MIN):
    """Monin-Obukhov diffusivities ``(K_m, K_h)`` at heights ``z`` above the
    surface (Isca ``monin_obukhov_diff``, reached through ``mo_diff``).

        zeta = -k b_star z / max(u_star, ustar_min)^2
        K_m  = k max(u_star, ustar_min) z / phi_m(zeta)
        K_h  = k max(u_star, ustar_min) z / phi_h(zeta)

    ``z`` may carry a leading level axis broadcast against ``u_star`` and
    ``b_star`` (shape ``(m, npts)`` against ``(npts,)``). With
    ``opts.neutral`` the stability functions are bypassed and
    ``K_m = K_h = k u_star z``, which is the neutral profile the existing
    FHZ06 scheme uses throughout.

    Note the sign convention: ``b_star > 0`` is an upward buoyancy flux
    (unstable), so ``zeta < 0`` there and the diffusivities are *enhanced*
    relative to neutral.
    """
    xp = _xp(z, u_star)
    opts = MoninObukhovOptions() if opts is None else opts
    uss = xp.maximum(u_star, ustar_min)
    if opts.neutral:
        k_m = von_karman * uss * z
        return k_m, 1.0 * k_m         # a distinct array, so a caller may edit either
    zeta = -von_karman * b_star * z / (uss * uss)
    phi_m = stability_derivative_m(zeta, opts)
    phi_h = stability_derivative_t(zeta, opts)
    base = von_karman * uss * z
    return base / phi_m, base / phi_h


def stable_mix(richardson, opts: Optional[MoninObukhovOptions] = None):
    """The stable mixing factor ``1 / phi^2`` as a function of the bulk
    Richardson number (Isca ``monin_obukhov_stable_mix``).

    Only Isca's free-atmosphere diffusivity uses this, which the grey-Mars
    configuration leaves off (``free_atm_diff = .false.``). It is provided
    because it is part of the referenced formulation and because its
    published checksum is an exact test of the stable branch: ``zeta`` is
    recovered from ``Ri`` in closed form for ``stable_option = 1`` (the
    quadratic ``a zeta^2 + b zeta + c`` below), so no iteration is needed.
    Zero outside ``0 < Ri < rich_crit``.
    """
    xp = _xp(richardson)
    opts = MoninObukhovOptions() if opts is None else opts
    b_stab = opts.b_stab
    if opts.stable_option == 1:
        r_in = xp.where((richardson > 0.0) & (richardson < opts.rich_crit),
                        richardson, 1.0)
        r = 1.0 / r_in
        a = r - b_stab
        b = r - (1.0 + 5.0)
        c = -1.0
        zeta = (-b + xp.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        phi = 1.0 + b_stab * zeta + (5.0 - b_stab) * zeta / (1.0 + zeta)
        mix = 1.0 / (phi * phi)
        return xp.where((richardson > 0.0) & (richardson < opts.rich_crit), mix, 0.0)
    rich_trans = opts.zeta_trans / (1.0 + 5.0 * opts.zeta_trans)
    lam = opts.lambda_stab
    weak = (1.0 - 5.0 * richardson) ** 2
    strong = ((1.0 - b_stab * richardson) / lam) ** 2
    mix = xp.where(richardson <= rich_trans, weak, strong)
    return xp.where((richardson > 0.0) & (richardson < opts.rich_crit), mix, 0.0)
