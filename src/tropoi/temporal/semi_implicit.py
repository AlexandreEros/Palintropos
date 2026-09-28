"""Semi-implicit leapfrog stepper for the dry hydrostatic primitive equations.

Formulation (docs/held_suarez/SEMI_IMPLICIT.md; plan §3): the tendency is
split as ``tendency(X) = N(X) + L X`` where **L** is the fast linear
gravity-wave operator about a resting isothermal reference atmosphere
(temperature ``t_ref``, flat reference surface pressure) and ``N`` is
everything else, evaluated explicitly at the middle time level. One
leapfrog step is

    X^{n+1} = X^{n-1} + 2 dt [ N(X^n) + L (X^{n+1} + X^{n-1}) / 2 ].

Per spherical-harmonic degree ``l`` (L is diagonal in l because its only
horizontal operator is the Laplacian) the implicit system in
``(delta_k, T_k, q)`` — K divergences, K temperatures and one ``q = ln
p_s`` — reduces exactly to one K×K solve for the time-averaged divergence
(plan §3, with the ``1·nu^T`` correction):

    [ I + dt^2 c_l ( G tau + R T_ref 1 nu^T ) ] delta_bar = rhs_l,
    c_l = l (l+1) / a^2.

Nothing in this module hand-derives a matrix entry. **G** (hydrostatic
geopotential, ``Phi = G T``), **tau** (adiabatic conversion, ``T_dot =
-tau delta`` with ``tau delta = -kappa T_ref (omega/p)^lin(delta)``) and
**nu** (layer thickness, ``q_dot = -nu . delta``) are read off the
discrete column operators of :mod:`tropoi.spatial.sigma_coordinate` by
applying them to unit vectors, so L is the exact linearization of the
discrete tendency about isothermal rest on a non-rotating planet
(tested). The reduced solve is verified against a direct dense solve of
the unreduced (2K+1)×(2K+1) coupled system (tested).

Dealiasing: the discrete tendency zeroes every analyzed nonlinear-product
contribution above the product-truncation cut (2/3 rule). Three of the
fast terms — ``R T_ref grad q`` (through the weak-form pressure-gradient
analysis), the adiabatic conversion and the mass divergence — reach the
tendency through that pathway, while ``-lap(G T)`` is an exact diagonal
spectral term. L reproduces this: above the cut only the ``-lap(G T)``
coupling survives (``mask``). Any state band-limited at the cut never
sees the difference; the mask only makes the linearization tests exact
for arbitrary perturbations.

Time filter: the RAW filter (Williams 2009) in the plan's normalisation,
``d = nu_raw (X^{n-1}_f - 2 X^n + X^{n+1})``, ``X^n_f = X^n + alpha d``,
``X^{n+1} -= (1 - alpha) d`` (plan/PROTOCOL: nu_raw = 0.1, alpha = 0.53;
Williams' own ``nu/2`` convention would call this nu = 0.2). ``raw_nu =
0`` disables it exactly.

Hooks (stage S4, :mod:`tropoi.temporal.hooks`): ``explicit_terms`` are
added to ``N(X^n)`` (e.g. Newtonian relaxation); ``dampers`` are
:class:`~tropoi.temporal.hooks.DiagonalDamping` objects applied to
``X^{n+1}`` with the backward-Euler factor ``1 / (1 + 2 dt r)`` after the
SI solve and before the RAW filter (e.g. Rayleigh drag, del^8
hyperdiffusion). With no hooks every operation is the S3 one, bit for bit.

Damping placement (``damping_scheme``, SEMI_IMPLICIT.md §9): the default
``"lagged"`` is the S4 placement above, which evaluates the damping at
n+1 and makes the damped part of the scheme first order in time. The
opt-in ``"centred"`` scheme takes ``-R X`` (R = the summed damper rates,
diagonal per row and degree) trapezoidally over n-1, n+1 inside the
solve, i.e. L is replaced by L - R: for rows outside L (zeta) that is the
per-row factor ``X^{n+1} = X^{n-1} + 2 dt (N - r X^{n-1}) / (1 + dt r)``;
for (delta, T, q) the rates enter the per-degree K×K system,

    [ I + dt R_delta + dt^2 c_l m_l ( G D_T tau + R T_ref d_q 1 nu^T ) ] delta_bar = rhs_l,
    D_T = (I + dt R_T)^{-1},  d_q = 1 / (1 + dt r_q),

assembled from the same column operators and verified against a direct
dense solve of the unreduced damped system (tested, 1e-12). The option
is part of ``state_dict()`` and a mismatch refuses to resume.

Startup: X^1 is produced by RK4 over one interval dt in ``n_sub``
substeps of the COMPLETE right-hand side (tendency + explicit terms +
the dampers' explicit form ``-r X``), where ``n_sub`` is the smallest count
for which the RK4 stability polynomial ``|R_4(z dt / n_sub)| <= 1`` holds
for ``z = -r_l + i omega`` at every degree: ``omega = sqrt(c_l
lambda_k(B))`` the discrete gravity-wave frequencies of L — measured from
the assembled operator, not assumed from sqrt(R T) — and ``r_l`` the
summed damping rates at that degree (zero without hooks, which recovers
the S3 imaginary-axis rule exactly).

Import-light: NumPy only. Coefficient arrays may be NumPy or CuPy; the
small per-degree matrices are mirrored into the array module of the
first coefficient array seen and cached. The layout is the PE state
stack ``(3K+1, l_max+1, l_max+1)`` with rows ``[zeta_1..K, delta_1..K,
T_1..K, ln p_s]`` and axes (degree l, order m).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from tropoi.spatial.sigma_coordinate import (SigmaGrid, hydrostatic_geopotential,
                                             omega_over_p)
from tropoi.spatial.truncation import product_truncation_cut
from tropoi.temporal.hooks import complete_tendency, hook_signatures
from tropoi.temporal.integration import rk4_step_array

__all__ = ["SemiImplicitOperator", "SemiImplicitSolver",
           "SemiImplicitLeapfrogStepper", "rk4_imaginary_axis_gain", "rk4_gain"]

_RK4_IMAGINARY_LIMIT = 2.0 * math.sqrt(2.0)


def _array_module(a):
    """NumPy or CuPy, from the array's type (no CuPy import unless needed)."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: WPS433  (optional backend)
        return cupy
    return np


def rk4_imaginary_axis_gain(y: float) -> float:
    """|R_4(i y)| for the classical RK4 stability polynomial R_4."""
    z = 1j * float(y)
    return abs(1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0)


def rk4_gain(z: complex) -> float:
    """|R_4(z)| for the classical RK4 stability polynomial R_4."""
    z = complex(z)
    return abs(1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0)


class SemiImplicitOperator:
    """The fast linear operator L of the semi-implicit split, per degree.

    Assembled once from the discrete column operators of ``sigma`` (see
    the module docstring). Public arrays (NumPy, float64):

    ``G``    (K, K)  hydrostatic geopotential matrix, ``Phi = G T``
    ``tau``  (K, K)  adiabatic conversion, ``T_dot|_L = -tau delta``
    ``nu``   (K,)    layer thicknesses, ``q_dot|_L = -nu . delta``
    ``B``    (K, K)  ``G tau + R T_ref 1 nu^T`` (squared gravity-wave
                     speeds are its eigenvalues)
    ``c_l``  (n,)    ``l (l+1) / a^2``
    ``mask`` (n,)    1 for ``l <= fast_cut`` else 0 (dealiasing, see above)
    """

    def __init__(self, sigma: SigmaGrid, *, l_max: int, radius: float,
                 r_dry: float, cp_dry: float, t_ref: float,
                 fast_cut: Optional[int] = None):
        if not isinstance(sigma, SigmaGrid):
            raise TypeError(f"sigma must be a SigmaGrid, got {type(sigma)!r}")
        l_max = int(l_max)
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}")
        for name, val, lo in (("radius", radius, 0.0), ("r_dry", r_dry, 0.0),
                              ("t_ref", t_ref, 0.0)):
            if not (math.isfinite(val) and val > lo):
                raise ValueError(f"{name} must be finite and > {lo}, got {val}")
        if not (math.isfinite(cp_dry) and cp_dry > r_dry):
            raise ValueError(f"cp_dry must be finite and > r_dry, got {cp_dry}")

        self.sigma = sigma
        self.K = sigma.nlev
        self.l_max = l_max
        self.n = l_max + 1
        self.radius = float(radius)
        self.r_dry = float(r_dry)
        self.cp_dry = float(cp_dry)
        self.kappa = self.r_dry / self.cp_dry
        self.t_ref = float(t_ref)
        self.fast_cut = (product_truncation_cut(l_max) if fast_cut is None
                         else int(fast_cut))

        K = self.K
        eye = np.eye(K, dtype=np.float64)
        zeros = np.zeros((K, K), dtype=np.float64)
        # Column j of each matrix is the operator's response to the unit
        # vector e_j (the trailing axis of the column operators is "points").
        phi_full, _ = hydrostatic_geopotential(sigma, eye, 0.0, self.r_dry)
        self.G = np.array(phi_full, dtype=np.float64)
        wp = omega_over_p(sigma, eye, zeros)          # A_k = 0
        self.tau = -(self.kappa * self.t_ref) * np.array(wp, dtype=np.float64)
        self.nu = sigma.thickness_array().astype(np.float64)
        ones = np.ones((K, 1), dtype=np.float64)
        self.B = self.G @ self.tau + (self.r_dry * self.t_ref) * (ones @ self.nu[None, :])

        l = np.arange(self.n, dtype=np.float64)
        self.c_l = l * (l + 1.0) / self.radius**2
        self.mask = (np.arange(self.n) <= self.fast_cut).astype(np.float64)

        self._device: dict[str, dict[str, Any]] = {}
        # Validate B now (real, positive eigenvalues = oscillatory gravity
        # waves) rather than only when a stepper asks for the startup count.
        self.gravity_wave_speeds()

    def signature(self) -> dict[str, Any]:
        """Everything that determines L: for checkpoint compatibility checks."""
        return {"sigma_interfaces": list(self.sigma.interfaces), "l_max": self.l_max,
                "radius": self.radius, "r_dry": self.r_dry, "cp_dry": self.cp_dry,
                "t_ref": self.t_ref, "fast_cut": self.fast_cut}

    @classmethod
    def from_model(cls, model, t_ref: float = 300.0,
                   fast_cut: Optional[int] = None) -> "SemiImplicitOperator":
        """Build L for a :class:`PrimitiveEquationsModel` (reads ``sigma``,
        ``l_max``, ``R`` (radius), ``r_dry``, ``cp_dry``). ``fast_cut``
        defaults to the model's ``retained_truncation`` (the degree at which
        its analyzed products are cut), so L stays the exact linearization;
        for a model without that attribute, the 2/3 product cut."""
        if fast_cut is None:
            fast_cut = getattr(model, "retained_truncation", None)
        return cls(model.sigma, l_max=model.l_max, radius=model.R,
                   r_dry=model.r_dry, cp_dry=model.cp_dry, t_ref=t_ref,
                   fast_cut=fast_cut)

    # ------------------------------------------------------------------
    def _dev(self, xp) -> dict[str, Any]:
        key = xp.__name__
        d = self._device.get(key)
        if d is None:
            d = {
                "G": xp.asarray(self.G), "tau": xp.asarray(self.tau),
                "nu": xp.asarray(self.nu),
                "cl": xp.asarray(self.c_l)[None, :, None],
                "clq": xp.asarray(self.c_l * self.mask)[None, :, None],
                "mask3": xp.asarray(self.mask)[None, :, None],
                "mask2": xp.asarray(self.mask)[:, None],
            }
            self._device[key] = d
        return d

    def apply(self, coeffs):
        """L X for a (3K+1, n, n) coefficient stack (same array family)."""
        xp = _array_module(coeffs)
        K = self.K
        if coeffs.shape != (3 * K + 1, self.n, self.n):
            raise ValueError(
                f"coefficients must have shape {(3 * K + 1, self.n, self.n)}, "
                f"got {coeffs.shape}")
        d = self._dev(xp)
        delta = coeffs[K:2 * K]
        T = coeffs[2 * K:3 * K]
        q = coeffs[3 * K]
        out = xp.zeros_like(coeffs)
        out[K:2 * K] = (d["cl"] * xp.einsum("ij,jlm->ilm", d["G"], T)
                        + (self.r_dry * self.t_ref) * (d["clq"] * q[None]))
        out[2 * K:3 * K] = -d["mask3"] * xp.einsum("ij,jlm->ilm", d["tau"], delta)
        out[3 * K] = -d["mask2"] * xp.einsum("j,jlm->lm", d["nu"], delta)
        return out

    def dense_matrix(self, l: int) -> np.ndarray:
        """L restricted to degree l as a (2K+1)×(2K+1) matrix on (delta, T, q)."""
        K = self.K
        c = float(self.c_l[l])
        m = float(self.mask[l])
        A = np.zeros((2 * K + 1, 2 * K + 1), dtype=np.float64)
        A[0:K, K:2 * K] = c * self.G
        A[0:K, 2 * K] = c * m * self.r_dry * self.t_ref
        A[K:2 * K, 0:K] = -m * self.tau
        A[2 * K, 0:K] = -m * self.nu
        return A

    def gravity_wave_speeds(self) -> np.ndarray:
        """sqrt of the eigenvalues of B, descending (m/s): the phase speeds
        of the K discrete vertical gravity-wave modes of the reference
        atmosphere. Raises if B has a non-real or non-positive eigenvalue."""
        ev = np.linalg.eigvals(self.B)
        if np.abs(ev.imag).max() > 1e-9 * np.abs(ev).max() or ev.real.min() <= 0.0:
            raise ValueError(
                "the semi-implicit operator B = G tau + R T_ref 1 nu^T has a "
                f"non-real or non-positive eigenvalue: {ev}")
        return np.sqrt(np.sort(ev.real)[::-1])

    def frequencies(self, l: int) -> np.ndarray:
        """omega_k = sqrt(c_l mask_l lambda_k(B)) for degree l (rad/s)."""
        return math.sqrt(float(self.c_l[l] * self.mask[l])) * self.gravity_wave_speeds()

    def max_frequency(self) -> float:
        return float(max(self.frequencies(l).max() for l in range(self.n)))


class SemiImplicitSolver:
    """The per-degree reduced K×K solve of the SI leapfrog step for one dt.

    ``advance(x_prev, explicit)`` returns ``X^{n+1}`` from ``X^{n-1}`` and
    ``N(X^n)``; ``matrix(l)`` / ``inverse(l)`` expose the assembled system
    and ``cayley_matrix(l)`` the analytic two-step amplification matrix
    ``(I - dt L_l)^{-1} (I + dt L_l)`` of the pure linear scheme (tests).
    """

    def __init__(self, operator: SemiImplicitOperator, dt: float,
                 dampers: Sequence[Any] = ()):
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        self.operator = operator
        self.dt = float(dt)
        op = operator
        K = op.K
        scale = (self.dt * self.dt) * (op.c_l * op.mask)             # (n,)
        self.rates: Optional[np.ndarray] = None
        if dampers:
            # Centred damping (DEVLOG 2026-09-28): the summed rates R enter
            # the implicit operator as L - R. Assembled from the same column
            # operators as B, with the T and q rows' Crank–Nicolson factors
            # folded in: B_l = G D_T,l tau + R T_ref d_q,l 1 nu^T.
            rates = np.zeros((3 * K + 1, op.n))
            for d in dampers:
                if d.rates.shape != rates.shape:
                    raise ValueError(
                        f"damping {d.name!r} rates {d.rates.shape} do not match the "
                        f"stack {rates.shape}")
                rates = rates + d.rates
            self.rates = rates
            f_t = 1.0 / (1.0 + self.dt * rates[2 * K:3 * K])         # (K, n)
            f_q = 1.0 / (1.0 + self.dt * rates[3 * K])                # (n,)
            ones = np.ones((K, 1))
            B_l = (np.einsum("ij,jl,jk->lik", op.G, f_t, op.tau)
                   + (op.r_dry * op.t_ref) * f_q[:, None, None] * (ones @ op.nu[None, :])[None])
            self._M = (np.eye(K)[None, :, :]
                       + self.dt * np.einsum("kl,kj->lkj", rates[K:2 * K], np.eye(K))
                       + scale[:, None, None] * B_l)
            self._f_zeta = 1.0 / (1.0 + self.dt * rates[0:K])
            self._f_t, self._f_q = f_t, f_q
        else:
            self._M = np.eye(K)[None, :, :] + scale[:, None, None] * op.B[None, :, :]
        # LU-based inverse per degree (np.linalg.inv factorizes with LAPACK
        # getrf/getri); applied as one batched product over (l, m).
        self._Minv = np.linalg.inv(self._M)
        self._device: dict[str, Any] = {}

    def matrix(self, l: int) -> np.ndarray:
        return self._M[l]

    def inverse(self, l: int) -> np.ndarray:
        return self._Minv[l]

    def implicit_matrix(self, l: int) -> np.ndarray:
        """The operator treated implicitly at degree l on (delta, T, q):
        L_l, or L_l - R_l with centred damping."""
        A = self.operator.dense_matrix(l)
        if self.rates is not None:
            A = A - np.diag(self.rates[self.operator.K:, l])
        return A

    def cayley_matrix(self, l: int) -> np.ndarray:
        A = self.implicit_matrix(l)
        I = np.eye(A.shape[0])
        return np.linalg.solve(I - self.dt * A, I + self.dt * A)

    def _minv(self, xp):
        key = xp.__name__
        m = self._device.get(key)
        if m is None:
            m = xp.asarray(self._Minv)
            self._device[key] = m
        return m

    def _centred_tables(self, xp):
        key = (xp.__name__, "centred")
        t = self._device.get(key)
        if t is None:
            K = self.operator.K
            t = {"fz": xp.asarray(self._f_zeta)[:, :, None],
                 "rz": xp.asarray(self.rates[0:K])[:, :, None],
                 "ft": xp.asarray(self._f_t)[:, :, None],
                 "rt": xp.asarray(self.rates[2 * K:3 * K])[:, :, None],
                 "fq": xp.asarray(self._f_q)[:, None],
                 "rq": xp.asarray(self.rates[3 * K])[:, None]}
            self._device[key] = t
        return t

    def advance(self, x_prev, explicit):
        """X^{n+1} = X^{n-1} + 2 dt [ N + L (X^{n+1} + X^{n-1}) / 2 ]
        (with centred damping: L - R in place of L)."""
        op = self.operator
        xp = _array_module(x_prev)
        K = op.K
        dt = self.dt
        if x_prev.shape != explicit.shape or x_prev.shape != (3 * K + 1, op.n, op.n):
            raise ValueError(
                f"x_prev {x_prev.shape} and explicit {explicit.shape} must both "
                f"have shape {(3 * K + 1, op.n, op.n)}")
        d = op._dev(xp)
        minv = self._minv(xp)
        if self.rates is not None:
            return self._advance_centred(x_prev, explicit, d, minv, xp)

        d_prev = x_prev[K:2 * K]
        t_prev = x_prev[2 * K:3 * K]
        q_prev = x_prev[3 * K]
        n_d = explicit[K:2 * K]
        n_t = explicit[2 * K:3 * K]
        n_q = explicit[3 * K]

        # T*, q*: the n-1 level advanced by the explicit tendency over dt.
        t_star = t_prev + dt * n_t
        q_star = q_prev + dt * n_q
        rhs = d_prev + dt * (n_d
                             + d["cl"] * xp.einsum("ij,jlm->ilm", d["G"], t_star)
                             + (op.r_dry * op.t_ref) * (d["clq"] * q_star[None]))
        d_bar = xp.einsum("lij,jlm->ilm", minv, rhs)

        out = xp.empty_like(x_prev)
        out[0:K] = x_prev[0:K] + (2.0 * dt) * explicit[0:K]
        out[K:2 * K] = 2.0 * d_bar - d_prev
        out[2 * K:3 * K] = t_prev + (2.0 * dt) * (
            n_t - d["mask3"] * xp.einsum("ij,jlm->ilm", d["tau"], d_bar))
        out[3 * K] = q_prev + (2.0 * dt) * (
            n_q - d["mask2"] * xp.einsum("j,jlm->lm", d["nu"], d_bar))
        return out

    def _advance_centred(self, x_prev, explicit, d, minv, xp):
        """The step with the damping -R X trapezoidal over n-1, n+1:

            T_bar = D_T (T* - dt m tau delta_bar),  q_bar = d_q (q* - dt m nu.delta_bar),
            M_l delta_bar = delta* + dt c_l (G D_T T* + R T_ref m d_q 1 q*),
            X^{n+1} = X^{n-1} + 2 dt (N - m L-coupling - R X^{n-1}) / (1 + dt R)

        for the T, q and zeta rows (with zero rates every factor is exactly
        1.0 and every subtracted term exactly 0.0)."""
        op = self.operator
        K = op.K
        dt = self.dt
        c = self._centred_tables(xp)
        d_prev = x_prev[K:2 * K]
        t_prev = x_prev[2 * K:3 * K]
        q_prev = x_prev[3 * K]
        n_d = explicit[K:2 * K]
        n_t = explicit[2 * K:3 * K]
        n_q = explicit[3 * K]

        t_star = c["ft"] * (t_prev + dt * n_t)
        q_star = c["fq"] * (q_prev + dt * n_q)
        rhs = d_prev + dt * (n_d
                             + d["cl"] * xp.einsum("ij,jlm->ilm", d["G"], t_star)
                             + (op.r_dry * op.t_ref) * (d["clq"] * q_star[None]))
        d_bar = xp.einsum("lij,jlm->ilm", minv, rhs)

        out = xp.empty_like(x_prev)
        out[0:K] = x_prev[0:K] + (2.0 * dt) * c["fz"] * (explicit[0:K] - c["rz"] * x_prev[0:K])
        out[K:2 * K] = 2.0 * d_bar - d_prev
        out[2 * K:3 * K] = t_prev + (2.0 * dt) * c["ft"] * (
            n_t - d["mask3"] * xp.einsum("ij,jlm->ilm", d["tau"], d_bar) - c["rt"] * t_prev)
        out[3 * K] = q_prev + (2.0 * dt) * c["fq"] * (
            n_q - d["mask2"] * xp.einsum("j,jlm->lm", d["nu"], d_bar) - c["rq"] * q_prev)
        return out


class SemiImplicitLeapfrogStepper:
    """Semi-implicit leapfrog + RAW filter behind the ``TimeStepper`` protocol.

    ``tendency(X)`` is the full explicit model tendency (e.g.
    ``PrimitiveEquationsModel.tendency``); ``N(X) = tendency(X) - L X`` is
    formed here. Time levels: ``x_prev`` is the filtered ``X^{n-1}_f``,
    ``state`` is ``X^n`` (unfiltered until the next step). The first
    ``step()`` after ``initialize`` is the RK4 startup (``X^0 -> X^1``);
    every later step is one leapfrog step. ``state_dict()`` carries both
    levels, so a restored stepper never redoes the startup.

    ``explicit_terms`` (callables ``X -> dX/dt`` with ``signature()``, and
    optionally ``max_rate`` for the startup rule) are added to ``N(X^n)``;
    ``dampers`` (:class:`~tropoi.temporal.hooks.DiagonalDamping`) multiply
    ``X^{n+1}`` by ``1 / (1 + 2 dt r)`` after the SI solve, in order, before
    the RAW filter. The startup integrates the complete right-hand side.
    """

    scheme = "si_leapfrog"
    # "lagged": each damper's 1/(1 + 2 dt r) on X^{n+1} after the SI solve
    # (S4, the default); "centred": -R X trapezoidal inside the solve
    # (Crank–Nicolson; second order; DEVLOG 2026-09-28), opt-in.
    DAMPING_SCHEMES = ("lagged", "centred")

    def __init__(self, tendency: Callable, operator: SemiImplicitOperator,
                 dt: float, *, raw_nu: float = 0.1, raw_alpha: float = 0.53,
                 startup_substeps: Optional[int] = None,
                 stage_validator: Optional[Callable] = None,
                 explicit_terms: Sequence[Callable] = (),
                 dampers: Sequence[Any] = (),
                 damping_scheme: str = "lagged"):
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        if not (0.0 <= raw_nu < 1.0):
            raise ValueError(f"raw_nu must be in [0, 1), got {raw_nu}")
        if not (0.0 <= raw_alpha <= 1.0):
            raise ValueError(f"raw_alpha must be in [0, 1], got {raw_alpha}")
        if damping_scheme not in self.DAMPING_SCHEMES:
            raise ValueError(
                f"damping_scheme must be one of {self.DAMPING_SCHEMES}, got {damping_scheme!r}")
        self.tendency = tendency
        self.operator = operator
        self.dt = float(dt)
        self.raw_nu = float(raw_nu)
        self.raw_alpha = float(raw_alpha)
        self.damping_scheme = str(damping_scheme)
        self.stage_validator = stage_validator
        self.explicit_terms = tuple(explicit_terms)
        self.dampers = tuple(dampers)
        self._startup_tendency = complete_tendency(
            tendency, self.explicit_terms, self.dampers)
        self.solver = SemiImplicitSolver(
            operator, self.dt,
            dampers=self.dampers if self.damping_scheme == "centred" else ())
        if startup_substeps is None:
            startup_substeps = self._stable_startup_substeps()
        elif not (isinstance(startup_substeps, int) and startup_substeps >= 1):
            raise ValueError(
                f"startup_substeps must be a positive int, got {startup_substeps!r}")
        self.startup_substeps = int(startup_substeps)
        self._x_prev = None
        self._x_curr = None
        self._t = 0.0
        self._n = 0

    def _stable_startup_substeps(self) -> int:
        """Smallest n_sub with |R_4(i omega_max dt / n_sub)| <= 1 and, with
        hooks, |R_4((-r_l + i omega) dt / n_sub)| <= 1 at every degree."""
        omega_max = self.operator.max_frequency()
        n_sub = max(1, math.ceil(omega_max * self.dt / _RK4_IMAGINARY_LIMIT - 1e-12))
        while rk4_imaginary_axis_gain(omega_max * self.dt / n_sub) > 1.0 + 1e-12:
            n_sub += 1
        if not (self.explicit_terms or self.dampers):
            return n_sub
        op = self.operator
        r_l = np.zeros(op.n)
        for d in self.dampers:
            r_l += d.rates.max(axis=0)
        r_l += sum(float(getattr(t, "max_rate", 0.0)) for t in self.explicit_terms)
        z = [complex(-r_l[l], w) for l in range(op.n)
             for w in (0.0, *op.frequencies(l))]
        while max(rk4_gain(zz * self.dt / n_sub) for zz in z) > 1.0 + 1e-12:
            n_sub += 1
        return n_sub

    # ------------------------------------------------------------------
    def initialize(self, y0, t0: float = 0.0) -> None:
        self._x_prev = None
        self._x_curr = y0.copy()
        self._t = float(t0)
        self._n = 0

    def step(self) -> float:
        if self._x_curr is None:
            raise RuntimeError("SemiImplicitLeapfrogStepper.step() before initialize()")
        if self._x_prev is None:
            self._startup()
        else:
            self._leapfrog()
        self._t += self.dt
        self._n += 1
        return self._t

    def _startup(self) -> None:
        y = self._x_curr
        h = self.dt / self.startup_substeps
        t = self._t
        for _ in range(self.startup_substeps):
            y = rk4_step_array(self._startup_tendency, y, t, h,
                               stage_validator=self.stage_validator)
            t += h
        self._x_prev = self._x_curr
        self._x_curr = y

    def _leapfrog(self) -> None:
        x_prev = self._x_prev
        x_curr = self._x_curr
        explicit = self.tendency(x_curr)
        for term in self.explicit_terms:
            explicit = explicit + term(x_curr)
        explicit = explicit - self.operator.apply(x_curr)
        x_next = self.solver.advance(x_prev, explicit)
        if self.damping_scheme == "lagged":
            for damper in self.dampers:
                x_next = damper.apply_implicit(x_next, 2.0 * self.dt)
        if self.raw_nu != 0.0:
            d = self.raw_nu * (x_prev - 2.0 * x_curr + x_next)
            x_curr_f = x_curr + self.raw_alpha * d
            x_next = x_next - (1.0 - self.raw_alpha) * d
        else:
            x_curr_f = x_curr
        self._x_prev = x_curr_f
        self._x_curr = x_next

    # ------------------------------------------------------------------
    @property
    def state(self):
        return self._x_curr

    @property
    def x_prev(self):
        return self._x_prev

    @property
    def t(self) -> float:
        return self._t

    @property
    def step_count(self) -> int:
        return self._n

    def state_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "dt": self.dt, "t": self._t,
                "step": self._n,
                "raw_nu": self.raw_nu, "raw_alpha": self.raw_alpha,
                "damping_scheme": self.damping_scheme,
                "t_ref": self.operator.t_ref,
                "operator": self.operator.signature(),
                "startup_substeps": self.startup_substeps,
                "hooks": hook_signatures(self.explicit_terms, self.dampers),
                "x_prev": None if self._x_prev is None else self._x_prev.copy(),
                "x_curr": None if self._x_curr is None else self._x_curr.copy()}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("scheme") != self.scheme:
            raise ValueError(
                f"state dict is for scheme {d.get('scheme')!r}, not {self.scheme}")
        for key, mine in (("dt", self.dt), ("raw_nu", self.raw_nu),
                          ("raw_alpha", self.raw_alpha),
                          ("t_ref", self.operator.t_ref)):
            if float(d[key]) != mine:
                raise ValueError(
                    f"state dict {key}={d[key]} differs from stepper {key}={mine}")
        # dicts written before the option existed come from the lagged code
        if d.get("damping_scheme", "lagged") != self.damping_scheme:
            raise ValueError(
                f"state dict damping_scheme={d.get('damping_scheme', 'lagged')!r} differs "
                f"from stepper damping_scheme={self.damping_scheme!r}")
        if int(d["startup_substeps"]) != self.startup_substeps:
            raise ValueError(
                f"state dict startup_substeps={d['startup_substeps']} differs from "
                f"stepper startup_substeps={self.startup_substeps}")
        mine_sig = self.operator.signature()
        if d.get("operator") != mine_sig:
            raise ValueError(
                "state dict was produced with a different semi-implicit operator "
                f"({d.get('operator')}) than this stepper's ({mine_sig})")
        mine_hooks = hook_signatures(self.explicit_terms, self.dampers)
        if d.get("hooks", []) != mine_hooks:
            raise ValueError(
                "state dict was produced with different stepper hooks "
                f"({d.get('hooks', [])}) than this stepper's ({mine_hooks})")
        step = int(d["step"])
        if d["x_curr"] is None:
            raise ValueError("state dict has no current state")
        if step > 0 and d["x_prev"] is None:
            raise ValueError(
                "state dict at step > 0 must carry both leapfrog time levels "
                "(x_prev is None); resuming would silently redo the RK4 startup")
        self._x_curr = d["x_curr"].copy()
        self._x_prev = None if d["x_prev"] is None else d["x_prev"].copy()
        self._t = float(d["t"])
        self._n = step
