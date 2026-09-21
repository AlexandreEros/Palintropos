"""Williamson et al. (1992) test case 5: the benchmark's prescribed world.

Canonical planet, fluid and cone constants, stdlib-only so the CPU
configuration layer (``tropoi.run.swe.config``, which re-exports every
name) and the case-5 initial condition share one bitwise definition.
Moved verbatim from ``run/swe/config.py``.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Williamson et al. (1992) test case 5: canonical constants.
#
# This module stays import-light, so the cone geometry constants are
# duplicated from physics/topography.py (kept in sync by a test). The
# fluid/planet constants are the published case-5 values. Case 5 prescribes
# the case-2 field as the FREE SURFACE, eta = h0 - (C/g) sin^2(lat), and
# the fluid-layer depth as eta - h_s (Williamson et al. 1992, Sect. 2 +
# Sect. 3.5; derivation in notebooks/W5_MRI_SEMANTIC_AUDIT.md). Because the
# model carries the mean thickness in Phi0 = g*H with the prognostic phi
# monopole pinned to zero, the canonical mean depth must absorb the cone's
# spherical mean (global mean of sin^2(lat) is 1/3):
#
#     H = mean(eta - h_s) = h0 - C/(3g) - mean(h_s)
#
# All expressions are the exact float forms shared with the
# initial-condition builder, so config, hash, and model agree bitwise.
# ---------------------------------------------------------------------------
W5_GRAVITY = 9.80616                      # m/s^2
W5_RADIUS_M = 6.37122e6                   # m (perfect sphere)
W5_OMEGA = 7.292e-5                       # s^-1
#: Day length whose 2*pi/(day_hours*3600) round-trips to exactly W5_OMEGA
#: (verified float identity, pinned by tests).
W5_DAY_HOURS = 2.0 * math.pi / W5_OMEGA / 3600.0
W5_U0_MS = 20.0                           # m/s
W5_H0_M = 5960.0                          # m (canonical free-surface h0)
W5_C = W5_RADIUS_M * W5_OMEGA * W5_U0_MS + 0.5 * W5_U0_MS * W5_U0_MS
W5_CONE_HEIGHT_M = 2000.0                 # m
W5_CONE_RADIUS_RAD = math.pi / 9.0        # rad (coordinate-plane distance)
W5_CONE_LAT_DEG = 30.0
W5_CONE_LON_DEG = -90.0


def _w5_cone_mean_height_m() -> float:
    """Exact spherical mean of the analytic Williamson-5 cone (metres).

    hbar = (1/4pi) * integral of hs0*(1 - r/R0)*cos(lat) over the
    coordinate-plane disk r = sqrt(dlon^2 + dlat^2) <= R0 centered at
    (lat_c, lon_c). Substituting lat = lat_c + y, lon = lon_c + x and
    dropping the odd sin(y) part of cos(lat_c + y) (disk and cone are even
    in y) leaves, via the Bessel identity
    integral_0^2pi cos(r sin(a)) da = 2*pi*J0(r):

        hbar = (hs0 * cos(lat_c) / 2) * I
        I    = integral_0^R0 (1 - r/R0) * J0(r) * r dr
             = sum_k (-1)^k R0^(2k+2) / (4^k (k!)^2 (2k+2)(2k+3))

    The alternating series converges superexponentially for R0 = pi/9
    (each term falls by ~300x) and is summed to float64 exhaustion, so the
    constant is bitwise deterministic. Cross-checked by tests against a
    brute-force numerical quadrature of the cone, and consistent with the
    MRI-JMA reference model's logged initial global mean of mass,
    5619.9259 m (STDOUT of Williamson5/N959_1920x960/sh; see
    notebooks/W5_MRI_SEMANTIC_AUDIT.md).
    """
    r0sq = W5_CONE_RADIUS_RAD * W5_CONE_RADIUS_RAD
    total = 0.0
    k = 0
    power = r0sq            # R0^(2k+2)
    factorial = 1.0         # k!
    while True:
        term = power / ((4.0 ** k) * factorial * factorial
                        * (2 * k + 2) * (2 * k + 3))
        new_total = total + (term if k % 2 == 0 else -term)
        if new_total == total:
            break
        total = new_total
        k += 1
        power *= r0sq
        factorial *= k
    return 0.5 * W5_CONE_HEIGHT_M * math.cos(
        math.radians(W5_CONE_LAT_DEG)) * total


#: Exact spherical mean of the analytic cone (~17.427 m): the terrain
#: monopole the canonical mean depth must absorb.
W5_CONE_MEAN_HEIGHT_M = _w5_cone_mean_height_m()
#: Canonical mean fluid depth H = h0 - C/(3g) - mean(h_s): the spherical
#: mean of the canonical layer depth eta - h_s.
W5_MEAN_DEPTH_M = (W5_H0_M - W5_C / (3.0 * W5_GRAVITY)
                   - W5_CONE_MEAN_HEIGHT_M)
#: The benchmark-owned topography token recorded in W5 run identities.
W5_TOPOGRAPHY = "williamson5_cone"
#: Spectral representation policy for the cone (hashed): the analytic cone
#: is analyzed once on the backend's state sampling and kept at the full
#: model truncation (no extra cut); see Topography.williamson5_cone.
W5_PROJECTION_POLICY = "state-grid-analysis-full-truncation"
