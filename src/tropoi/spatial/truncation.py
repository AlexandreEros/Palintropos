"""Spectral support contracts shared by every core (import-light).

This module deliberately imports nothing from the package: it is consumed
by the spatial operators and initializers, the temporal cores, the
diagnostics, the configuration layer and the CLI, so it must stay free of
CuPy, Matplotlib, and the heavy package initializers. Keeping it neutral
avoids an operators -> tendencies dependency.

Vocabulary (docs/ARCHITECTURE.md, docs/validation/
preset_support_characterization.md):

* **stored capacity** ``l_max``: the triangular ``0 <= m <= l <= l_max``
  coefficient extent every prognostic array carries;
* **product truncation cut** :func:`product_truncation_cut`: the highest
  degree an analyzed nonlinear product retains (the 2/3 rule). The shared
  contract is only that analyzed nonlinear-product contributions are
  zeroed for ``cut < l <= l_max`` (and ``m > cut``); which OTHER terms act
  on those stored degrees (exact linear spectral operators, forcing,
  hyperdiffusion or viscosity, prescribed topographic terms) is decided by
  each core, not by this module.
* **default state grid** :func:`gauss_grid_for_truncation`: the Gauss
  lat-lon ``(nlat, nlon)`` a core uses for stored capacity ``l_max`` when
  the caller names no dimensions. A sizing *policy*, not a correctness
  contract -- an explicitly given pair always wins, and a configuration
  may legitimately sit below it (Held-Suarez runs ``l_max = 43`` on
  ``64 x 128`` and leans on the finer product grid for dealiasing).
"""
from __future__ import annotations

__all__ = ["product_truncation_cut", "gauss_grid_for_truncation",
           "SWE_SCENARIO_SUPPORT",
           "require_scenario_support", "THERMAL_WAVE_MIN_RETAINED_DEGREE",
           "require_thermal_wave_support"]


def product_truncation_cut(l_max: int) -> int:
    """The 2/3-rule truncation degree for analyzed nonlinear products.

    ``cut = floor(2 * l_max / 3)``: an analyzed product keeps degrees and
    orders ``<= cut`` and zeroes ``cut < l <= l_max`` (and ``m > cut``);
    it says nothing about the non-product terms a core applies there.
    Pinned values: ``1->0, 2->1, 3->2, 4->2, 5->3, 6->4, 10->6, 15->10,
    21->14, 42->28, 63->42``. This is the single production definition;
    every operator, core, and diagnostic band derives from it.
    """
    return (2 * int(l_max)) // 3


def gauss_grid_for_truncation(l_max: int) -> tuple[int, int]:
    """Default Gauss lat-lon ``(nlat, nlon)`` for stored capacity ``l_max``.

    The repository's established sizing convention, made explicit: the
    **3/2 (quadratic) rule** with an even latitude count and
    ``nlon = 2 * nlat``::

        nlat = ceil((3 * l_max + 1) / 2) rounded up to even
        nlon = 2 * nlat

    Pinned values: ``1->(2, 4), 2->(4, 8), 4->(8, 16), 5->(8, 16),
    8->(14, 28), 10->(16, 32), 15->(24, 48), 21->(32, 64), 42->(64, 128),
    63->(96, 192)``. The T21 and T42 pairs are exactly the resolutions in
    production use.

    Three requirements are satisfied by construction, for every
    ``l_max >= 1``:

    * **analysis exactness** (``spatial.grids.latlon_grid``):
      ``nlat >= l_max + 1`` and ``nlon >= 2 * l_max + 1``, so a round trip
      of a degree-``l_max`` field leaks nothing across modes;
    * **quadratic dealiasing** (the 3/2 rule that
      ``spatial.spherical_backend.LatLonBackend`` applies to its *product*
      grid): ``nlat >= ceil((3 * l_max + 1) / 2)`` and
      ``nlon >= 3 * l_max + 1``. Holding the *state* grid to it as well
      means the ``product_quadrature="coarse"`` A/B path is also
      quadratically exact, and the "fine" product grid then needs no
      enlargement at all;
    * **separable applicability**
      (``spatial.transforms.separable_gauss_sh.separable_is_applicable``):
      ``2 * l_max < nlon``, so every retained zonal wavenumber owns a
      real-FFT bin strictly below Nyquist and the default evaluator never
      falls back to the dense path.

    An even ``nlat`` keeps the Gauss-Legendre latitudes symmetric about the
    equator with no node *on* it, and makes ``nlon`` a multiple of four --
    both friendly to the real FFT the separable evaluator runs in
    longitude.

    This is a **policy**, not a validity test. Dimensions the caller gives
    explicitly always override it, and a configuration below it is not
    thereby wrong: it forfeits state-grid quadratic exactness and relies on
    the finer product grid instead, which is what Held-Suarez does at
    ``l_max = 43`` on ``64 x 128``.
    """
    l_max = int(l_max)
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}")
    nlat = -(-(3 * l_max + 1) // 2)   # ceil((3 l_max + 1) / 2)
    nlat += nlat % 2                  # up to even
    return nlat, 2 * nlat


# ---------------------------------------------------------------------------
# Preset support guards. Shared by the CPU configuration layer
# (tropoi.run.swe.config / tropoi.run.pe.config re-export them) and the
# initial-condition factories in tropoi.spatial.initialization, so both
# boundaries enforce one contract.
# ---------------------------------------------------------------------------

#: Preset support contract, measured in
#: docs/validation/preset_support_characterization.md: ``(min_lmax,
#: min_retained_degree)``. ``min_lmax`` is the stored capacity the preset's
#: literal coefficients need; ``min_retained_degree`` (or None) is the
#: degree the preset's advertised behavior needs INSIDE the 2/3 product cut
#: (``product_truncation_cut(lmax) >= value``). Williamson 2's steady state
#: needs the degree-2 curl/kinetic-energy products to cancel its pressure
#: term (lost at lmax=2); gravity_wave's Y_4^2 mode needs storage only,
#: because above the cut its delta/phi pair receives no product term (the
#: exact linear pressure pair, plus any hyperdiffusion, still acts there);
#: Williamson 5's value is the initial-state storage requirement only: the
#: cone/topography representability gate and the benchmark policy are
#: separate and unchanged.
SWE_SCENARIO_SUPPORT = {
    "rest": (1, None),
    "gravity_wave": (4, None),
    "williamson2": (2, 2),
    "williamson5": (2, None),
}


def _min_lmax_retaining(degree: int) -> int:
    lmax = degree
    while product_truncation_cut(lmax) < degree:
        lmax += 1
    return lmax


def require_scenario_support(scenario: str, lmax: int) -> None:
    """Raise ValueError unless ``lmax`` supports ``scenario``.

    Shared by the CPU configuration layer and the CUDA initial-condition
    factory so both boundaries enforce the same contract. Unknown scenarios
    are the caller's concern (reported separately).
    """
    support = SWE_SCENARIO_SUPPORT.get(scenario)
    if support is None:
        return
    min_lmax, min_retained = support
    lmax = int(lmax)
    if lmax < min_lmax:
        raise ValueError(
            f"scenario {scenario!r} needs lmax >= {min_lmax} to store its "
            f"initial coefficients, got lmax={lmax}")
    if min_retained is not None and product_truncation_cut(lmax) < min_retained:
        raise ValueError(
            f"scenario {scenario!r} needs the 2/3 product cut to retain "
            f"degree {min_retained} (product_truncation_cut(lmax) >= "
            f"{min_retained}, i.e. lmax >= {_min_lmax_retaining(min_retained)}); "
            f"lmax={lmax} retains only degrees <= "
            f"{product_truncation_cut(lmax)}. See docs/validation/"
            "preset_support_characterization.md.")


#: thermal_wave places its perturbation on the (2, 2) mode; a NONZERO
#: amplitude additionally needs degree 2 inside the 2/3 product cut, or the
#: temperature perturbation is frozen for all time while only divergence
#: responds (measured at lmax=2, docs/validation/
#: preset_support_characterization.md). Zero amplitude is exact rest and
#: keeps the storage boundary only.
THERMAL_WAVE_MIN_RETAINED_DEGREE = 2


def require_thermal_wave_support(lmax: int, thermal_amplitude: float, *,
                                 scenario: str = "thermal_wave") -> None:
    """Raise ValueError unless ``lmax`` supports thermal_wave at this amplitude.

    Shared by the CPU configuration layer and the CUDA initial-condition
    factory. Storage (``lmax >= 2``) is always required; the retained-degree
    requirement applies only to a nonzero amplitude.
    """
    lmax = int(lmax)
    if lmax < 2:
        raise ValueError(
            f"scenario {scenario!r} needs lmax >= 2 for its degree-2 "
            f"perturbation, got {lmax}")
    if float(thermal_amplitude) != 0.0 and \
            product_truncation_cut(lmax) < THERMAL_WAVE_MIN_RETAINED_DEGREE:
        raise ValueError(
            f"scenario {scenario!r} with a nonzero thermal_amplitude needs "
            "the 2/3 product cut to retain degree 2 (lmax >= 3); "
            f"lmax={lmax} retains only degrees <= "
            f"{product_truncation_cut(lmax)}, which freezes the temperature "
            "perturbation. Use lmax >= 3, or thermal_amplitude=0 for exact "
            "rest. See docs/validation/preset_support_characterization.md.")
