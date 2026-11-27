# Test fixtures

[airless/ceres-equatorial-column.airless.json](airless/ceres-equatorial-column.airless.json)
is a saved airless configuration document (a `config.json`-style envelope)
for one equatorial column on an idealized Ceres. Its configuration hashes to
`13dfd66a305cadb5932c8c7e001ef9a3e2f90c6b8eb304ce0788969812750c8a`.
[test_airless_transient.py](../test_airless_transient.py) loads it through
`--config` and runs shortened transients from it. It is an input example,
not a validated Ceres model; its `provenance` block says which inputs are
measured, derived, inferred or assumed.

[hs-T21L10-dev4](hs-T21L10-dev4/README.md) is a compact Held–Suarez run
fixture for checking report and plot behavior on a CPU. Its receipt covers
the preserved run files. The [climate fixture test](../test_held_suarez_climate_fixture.py)
checks that inventory.

[substrate_m1_historical_grids.json](substrate_m1_historical_grids.json) holds
the exact interface grids of the original milestone-1 conductive-substrate
gate run. They were computed by the original implementation at commit
`cc33bfb` and verified against that run's frozen results. They include the
original stretch-mode grids, whose final-layer ratios lie outside [1, 1.5],
and they are not regenerated. `tests/audit_substrate_gates.py --grids
historical` uses them to reproduce the frozen results, and
`tests/test_substrate.py` checks them.
The fixture also holds `uniform:N=400`. That grid belongs to the post hoc G1-C
supplementary study, not to a gate run, and the gate harness never requests it.
