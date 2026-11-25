# Airless-body thermal experiments

`tropoi.run.airless` runs independent conductive columns
(`tropoi.physics.substrate.ConductiveSubstrate`) under the shared orbit and
rotation illumination geometry (`tropoi.physics.insolation`). It runs on the
NumPy or the CuPy backend, and both share one implementation. Presets are
configurations of that machinery; Mercury with the Vasavada, Paige & Wood
(1999) regolith is the primary one. Tier-2 output is streamed to disk in
per-block files and read back with `load_tier2`.

* entry path: [`calls/airless-mercury.mmd`](../current/calls/airless-mercury.mmd)
* acceptance register: [`../../validation/PLANET-ONE-GATES.md`](../../validation/PLANET-ONE-GATES.md)
* status and handoff: [`../../validation/PLANET-ONE-STATUS.md`](../../validation/PLANET-ONE-STATUS.md)
