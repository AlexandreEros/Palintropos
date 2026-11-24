# Test fixtures

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
