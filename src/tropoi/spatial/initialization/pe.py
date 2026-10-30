"""Initial conditions for the dry primitive-equation core.

Four presets. All are built spectrally on top of
:func:`~tropoi.spatial.states.primitive_equations.isothermal_rest_state`,
so vorticity and divergence are exactly zero and the fields are exactly
band-limited (no grid round trip).

``isothermal_rest``
    Exactly resting, horizontally uniform isothermal atmosphere: zeta = delta
    = 0, T_k = ``temperature`` at every level, p_s = ``surface_pressure``
    everywhere. This is the model's exact-rest state (its tendency is exactly
    zero), used to verify that the runner preserves rest bit-for-bit.

``thermal_wave``
    The resting isothermal state plus a single deterministic degree-2
    temperature perturbation. The perturbation is one real spherical-harmonic
    coefficient placed on the (l, m) = (2, 2) sectoral mode at *every* full
    level (a vertically uniform profile), following the repository's
    real-field coefficient convention (a single real coefficient at
    (l, m>0) synthesizes a valid longitude-varying real field, exactly as the
    shallow-water ``gravity_wave`` preset does). ``thermal_amplitude`` is that
    coefficient's value in kelvin; ~1 K keeps the perturbed temperature
    positive everywhere. Surface pressure stays uniform and the initial winds
    stay zero, so the state is deliberately *unbalanced* — it exists to show
    the PE model launches a smooth, finite response, not a balanced flow.

``orographic_isothermal_rest``
    Analytically balanced resting isothermal atmosphere over the model's
    prescribed surface geopotential Phi_s (zero terrain reduces bitwise to
    ``isothermal_rest``): zeta = delta = 0, T_k = ``temperature`` (T0) at
    every full level, and

        ln(p_s) = ln(p_ref) - Phi_s / (R_d T0)

    with p_ref = ``surface_pressure`` (the surface pressure where Phi_s = 0).
    The relation is applied directly to the spectral coefficients — the
    ln(p_ref) monopole minus ``model.phi_surface_lm / (r_dry * T0)`` — so the
    state derives from the EXACT resolved Phi_s the model integrates with (no
    grid round trip, no independent reconstruction of the terrain). The state
    is horizontally and hydrostatically balanced because grad(Phi_s) +
    R_d T0 grad(ln p_s) = 0 pointwise, so the pressure-gradient terms of the
    momentum tendency cancel analytically.

    The isothermal restriction is a property of this benchmark, not of the
    PE topography support: a resting atmosphere with a horizontally uniform
    NON-isothermal T(sigma) profile over terrain is generally NOT balanced,
    because sigma surfaces cut across pressure surfaces where p_s varies. A
    future exact non-isothermal construction must instead define a
    pressure-coordinate reference profile T_ref(p), hydrostatically derive
    Phi_ref(p), solve Phi_ref(p_s) = Phi_s for the local surface pressure,
    and evaluate T_k = T_ref(sigma_k p_s(lambda, phi)) — a separate
    constructor, deliberately not begun here.

``orographic_thermal_wave``
    ``orographic_isothermal_rest``, built by its unchanged constructor, plus
    the ``thermal_wave`` temperature perturbation: the real coefficient
    ``thermal_amplitude`` (K) on the (l, m) = (2, 2) mode at every full
    level (imaginary part zero, vertical weights all 1). The terrain-aware
    surface pressure and the zero vorticity/divergence of the balanced base
    are left exactly as they are, and zero amplitude reduces bitwise to
    ``orographic_isothermal_rest``. The state is a perturbation OF a
    balanced base, not a balanced state: it starts a small, smooth,
    deterministic evolution over terrain. In the repository's real-field
    convention (f = a_l0 Y_l0 + 2 Re sum_{m>0} a_lm Y_lm, orthonormal
    Y_lm) a real coefficient ``a`` on (2, 2) peaks pointwise at
    ``2 sqrt(15 / (32 pi)) a``, about 0.773 a (:func:`thermal_wave_peak`):
    the coefficient is not the pointwise peak.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tropoi.spatial.states.primitive_equations import (
    PrimitiveEquationsState, isothermal_rest_state)
from tropoi.spatial.truncation import require_thermal_wave_support

if TYPE_CHECKING:  # annotation only: the factories use l_max/nlev/terrain
    from tropoi.temporal.tendencies.primitive_equations import (
        PrimitiveEquationsModel)

#: The thermal-wave perturbation lives on this single real spherical-harmonic
#: mode (degree 2, order 2): a sectoral, longitude-varying, low-degree mode
#: well below any usable l_max.
THERMAL_WAVE_DEGREE = 2
THERMAL_WAVE_ORDER = 2

#: Presets that place the thermal-wave perturbation (degree 2 must lie
#: inside the dealiased product cut for a nonzero amplitude).
THERMAL_WAVE_SCENARIOS = ("thermal_wave", "orographic_thermal_wave")


def thermal_wave_peak(amplitude: float) -> float:
    """Pointwise maximum of the thermal-wave perturbation, in kelvin.

    A real coefficient ``a`` on (2, 2) synthesizes
    ``2 a sqrt(15 / (32 pi)) sin^2(theta) cos(2 lambda)``, whose maximum
    over the sphere is ``2 sqrt(15 / (32 pi)) |a|`` (about 0.7725 |a|).
    """
    return 2.0 * math.sqrt(15.0 / (32.0 * math.pi)) * abs(float(amplitude))


def _isothermal_rest(model: PrimitiveEquationsModel, *, temperature: float,
                     surface_pressure: float,
                     thermal_amplitude: float = 0.0) -> PrimitiveEquationsState:
    del thermal_amplitude  # unused: rest has no perturbation
    return isothermal_rest_state(model.l_max, model.nlev,
                                 temperature=temperature,
                                 surface_pressure=surface_pressure)


def _thermal_wave(model: PrimitiveEquationsModel, *, temperature: float,
                  surface_pressure: float,
                  thermal_amplitude: float) -> PrimitiveEquationsState:
    if model.l_max < THERMAL_WAVE_DEGREE:
        raise ValueError(
            f"thermal_wave needs l_max >= {THERMAL_WAVE_DEGREE}, got "
            f"{model.l_max}")
    state = isothermal_rest_state(model.l_max, model.nlev,
                                  temperature=temperature,
                                  surface_pressure=surface_pressure)
    # One real coefficient on the (2, 2) mode at every full level. The base
    # monopole (T0) and the perturbation are the only nonzero temperature
    # coefficients; zeta/delta/ln_ps are untouched (rest, uniform p_s).
    state.temperature[:, THERMAL_WAVE_DEGREE, THERMAL_WAVE_ORDER] = \
        float(thermal_amplitude)
    return state


def _orographic_isothermal_rest(model: PrimitiveEquationsModel, *,
                                temperature: float, surface_pressure: float,
                                thermal_amplitude: float = 0.0
                                ) -> PrimitiveEquationsState:
    del thermal_amplitude  # unused: rest has no perturbation
    state = isothermal_rest_state(model.l_max, model.nlev,
                                  temperature=temperature,
                                  surface_pressure=surface_pressure)
    # ln(p_s) = ln(p_ref) - Phi_s/(R_d T0), applied directly to the spectral
    # coefficients of the exact Phi_s the model integrates with. The
    # ln(p_ref) monopole is already in place from the rest state.
    state.coeffs[3 * model.nlev] -= model.phi_surface_lm / (
        model.r_dry * float(temperature))
    return state


def _orographic_thermal_wave(model: PrimitiveEquationsModel, *,
                             temperature: float, surface_pressure: float,
                             thermal_amplitude: float
                             ) -> PrimitiveEquationsState:
    if model.l_max < THERMAL_WAVE_DEGREE:
        raise ValueError(
            f"orographic_thermal_wave needs l_max >= {THERMAL_WAVE_DEGREE}, "
            f"got {model.l_max}")
    # The balanced base from the unchanged constructor (ln p_s from the
    # exact resolved Phi_s), then only the (2, 2) temperature coefficients:
    # the isothermal base has none there, so this adds the perturbation and
    # leaves ln p_s, zeta and delta at their base values.
    state = _orographic_isothermal_rest(model, temperature=temperature,
                                        surface_pressure=surface_pressure)
    state.temperature[:, THERMAL_WAVE_DEGREE, THERMAL_WAVE_ORDER] += \
        float(thermal_amplitude)
    return state


PE_INITIAL_CONDITIONS = {
    "isothermal_rest": _isothermal_rest,
    "thermal_wave": _thermal_wave,
    "orographic_isothermal_rest": _orographic_isothermal_rest,
    "orographic_thermal_wave": _orographic_thermal_wave,
}


def make_pe_ic(name: str, model: PrimitiveEquationsModel, *,
               temperature: float, surface_pressure: float,
               thermal_amplitude: float = 0.0) -> PrimitiveEquationsState:
    """Construct a named primitive-equation initial state.

    ``temperature`` (K) and ``surface_pressure`` (Pa) set the resting base
    state; ``thermal_amplitude`` (K) is the degree-2 perturbation coefficient,
    used only by the presets in :data:`THERMAL_WAVE_SCENARIOS`.
    """
    if name not in PE_INITIAL_CONDITIONS:
        raise ValueError(
            f"Unknown pe initial condition: {name}. "
            f"Available: {sorted(PE_INITIAL_CONDITIONS)}")
    if name in THERMAL_WAVE_SCENARIOS:
        # Same contract as the CPU config layer (tropoi.spatial.truncation.
        # require_thermal_wave_support, re-exported by run/pe/config): storage lmax >= 2 always, and
        # degree 2 inside the product cut for a nonzero amplitude. Direct
        # low-level state construction stays available for characterization.
        require_thermal_wave_support(model.l_max, thermal_amplitude)
    return PE_INITIAL_CONDITIONS[name](
        model, temperature=temperature, surface_pressure=surface_pressure,
        thermal_amplitude=thermal_amplitude)
