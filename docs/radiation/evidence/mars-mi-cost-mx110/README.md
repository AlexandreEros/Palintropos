# CV-1 cost evidence: NVIDIA GeForce MX110 (2026-11-14)

These are the raw records behind [`MARS_MI_COST.md`](../../MARS_MI_COST.md).
They were written by
[`tools/radiation/mars_cost_profile.py`](../../../../tools/radiation/mars_cost_profile.py)
at `cfebd16` (the preset `dev-mars-gray-flat-bl` of `06f5356`), on Windows 11
with CuPy 13.4.0 and CUDA 11.8. Each record carries its own `environment`
block.

| file | what |
| --- | --- |
| `profile-T{10,15,21}-dt600-run1.json`, `profile-T42-dt600-chunk64-run1.json` | first pass. T42 with benchmark-only GEMM chunking (64 rows) |
| `profile-T{10,15,21}-dt600.json`, `profile-T42-dt600-chunk64.json` | second pass, about 2 h later, adding `instrumented.by_operator`. T42 ran throttled (SM clock 764 MHz, against 1006 MHz in the first pass) |
| `probe-T{10,21,42}-dt*.json` | time-step probes from day 60 of `dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4` (`checkpoint-s000008640.npz`, local) |
| `timescales.json` | slab, radiative and spin-down timescales; the exponential fit of the mean T_s |
| `projection-mx110-run1.json`, `projection-mx110.json` | CV-1 wall-clock arithmetic from the profiles |

**The unchunked T42 attempt.** The first T42 profile, without
`--chunk-rows`, wrote no file. It failed in the RK4 start-up step
(`semi_implicit._startup` → `PrimitiveEquationsModel._tendency_batched`)
with `CUDADriverError: CUDA_ERROR_LAUNCH_TIMEOUT`, the Windows display-driver
watchdog. The error surfaced asynchronously, at a later `cumsum` compile in
`interface_sigma_dot`.

The `scratch_dir` fields point at deleted scratch directories. They are
kept only as provenance.
