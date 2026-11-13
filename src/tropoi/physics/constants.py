"""Fundamental and nominal constants shared by all of ``tropoi.physics``.

One source of truth. The SI defining constants are exact since the 2019
redefinition (CODATA lists them as exact): Planck ``H``, speed of light
``C``, Boltzmann ``K_B``, Avogadro ``N_A``. Derived from them exactly:

* the molar gas constant ``R_UNIVERSAL = K_B * N_A``;
* the Stefan–Boltzmann constant ``SIGMA_SB = 2 pi^5 k^4 / (15 h^3 c^2)``;
* the second radiation constant ``C2_CM_K = h c / k_B`` in cm K (converts a
  wavenumber in cm^-1 to a characteristic temperature).

``G`` is the measured Newtonian constant of gravitation (CODATA 2018;
unchanged in CODATA 2022). ``LUMINOUS_EFFICACY`` = 683 lm/W is the SI
definition of the candela (K_cd at 540e12 Hz); applied to the CIE 1931
y-bar function it gives luminance in cd m^-2 (conventional photopic
photometry).

Nominal astronomical values (conventions, not measurements): IAU 2015
Resolution B3 (solar T_eff, radius, luminosity, total solar irradiance)
and the IAU 2012 Resolution B2 astronomical unit.

Molar masses of gases are data, not constants: they live in
``tropoi/physics/data/species.toml`` and are served by
:mod:`tropoi.physics.species`.
"""
from __future__ import annotations

import math

__all__ = ["AU_M", "C", "C2_CM_K", "DAY_SECONDS", "G", "H", "K_B",
           "LUMINOUS_EFFICACY", "L_SUN_W", "N_A", "NOMINAL_SOLAR_SOURCE",
           "R_SUN_M", "R_UNIVERSAL", "SIGMA_SB", "SOLAR_CONSTANT", "T_SUN_K"]

# -- exact SI defining constants --------------------------------------------
H = 6.62607015e-34            # J s
C = 299792458.0               # m s^-1
K_B = 1.380649e-23            # J K^-1
N_A = 6.02214076e23           # mol^-1

# -- exact derived constants -------------------------------------------------
#: Molar gas constant K_B * N_A, J mol^-1 K^-1 (= 8.31446261815324).
R_UNIVERSAL = K_B * N_A
#: Stefan–Boltzmann constant from the exact SI constants, W m^-2 K^-4.
SIGMA_SB = 2.0 * math.pi ** 5 * K_B ** 4 / (15.0 * H ** 3 * C ** 2)
#: Second radiation constant h c / k_B in cm K.
C2_CM_K = H * C / K_B * 100.0

# -- measured constants -------------------------------------------------------
#: Newtonian constant of gravitation, m^3 kg^-1 s^-2 (CODATA 2018/2022).
G = 6.67430e-11

# -- conventions --------------------------------------------------------------
#: SI definition of the candela: K_cd = 683 lm/W at 540e12 Hz.
LUMINOUS_EFFICACY = 683.0
#: A "day" of bookkeeping, s (always 86400 s, whatever the planet rotates at).
DAY_SECONDS = 86400.0

# -- IAU nominal astronomical values ------------------------------------------
#: Astronomical unit, m (IAU 2012 Resolution B2: 149 597 870 700 m exactly).
AU_M = 149597870700.0
#: Nominal solar radius, m (IAU 2015 Resolution B3).
R_SUN_M = 695700.0e3
#: Nominal solar luminosity, W (IAU 2015 Resolution B3).
L_SUN_W = 3.828e26
#: Nominal solar effective temperature, K (IAU 2015 Resolution B3).
T_SUN_K = 5772.0
#: Nominal total solar irradiance at 1 au, W m^-2 (IAU 2015 Resolution B3).
SOLAR_CONSTANT = 1361.0

NOMINAL_SOLAR_SOURCE = ("IAU 2015 B3 nominal T_eff = 5772 K, R = 695 700 km, "
                        "L = 3.828e26 W; IAU 2012 B2 au")
