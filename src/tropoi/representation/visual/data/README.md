# Sky colour data

[CIE_xyz_1931_2deg.csv](CIE_xyz_1931_2deg.csv) is the CIE 1931 2° standard
observer colour-matching table (CIE 2019, DOI `10.25039/CIE.DS.xvudnb9b`).
It contains wavelength in nanometres and the three XYZ matching functions
from 360 to 830 nm at 1 nm spacing.

[colour.py](../colour.py) loads the packaged CSV and checks its SHA-256
before using it to integrate spectral bins into XYZ values for the
[sky viewer](../../../../../docs/SKY_VIEW.md). Preserve the original file
bytes, including line endings.
