"""Physical constants used by the optics package.

SI defining constants (exact since the 2019 SI redefinition; CODATA 2018
lists them as exact): Planck ``H``, speed of light ``C``, Boltzmann ``K_B``,
Avogadro ``N_A``. The molar gas constant is ``K_B * N_A`` (exact).

Atomic weights: IUPAC/CIAAW abridged standard atomic weights (conventional
values, Prohaska et al., Pure Appl. Chem. 94, 573 (2022)), in g/mol.

``LUMINOUS_EFFICACY`` = 683 lm/W: the SI definition of the candela (K_cd at
540e12 Hz); applied to the CIE 1931 y-bar function it gives luminance in
cd m^-2 (the conventional photopic photometry).
"""
from __future__ import annotations

H = 6.62607015e-34            # J s
C = 299792458.0               # m s^-1
K_B = 1.380649e-23            # J K^-1
N_A = 6.02214076e23           # mol^-1
R_MOLAR = K_B * N_A           # J mol^-1 K^-1

ATOMIC_WEIGHT_G_PER_MOL = {"N": 14.007, "O": 15.999, "Ar": 39.95,
                           "C": 12.011}
ATOMIC_WEIGHT_SOURCE = ("IUPAC/CIAAW abridged standard atomic weights "
                        "(Prohaska et al., Pure Appl. Chem. 94, 573, 2022)")

LUMINOUS_EFFICACY = 683.0     # lm W^-1
