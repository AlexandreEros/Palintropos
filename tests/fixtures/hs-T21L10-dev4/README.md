# `hs-T21L10-dev4`: Held–Suarez climate-figure fixture (real GPU run)

A production Held–Suarez run of exactly
`development_config(days=4, retained=True)`:

- T21 retained (stored `l_max` 22), L10 uniform σ, 32 × 64 Gaussian grid;
- dt 1200 s, lagged damping;
- `spinup_days` 0, one state per day, two 2-day statistics blocks.

`tests/test_held_suarez_climate_fixture.py` reduces its saved days 1–4 on the
CPU and compares them with the production statistics. CI needs no GPU.

It was made by `HeldSuarezRun` on an NVIDIA GeForce MX110 (CuPy 13.4.0, CUDA
11.8), Windows, on 2026-11-04. The source was the clean commit `8ac1ca1a`
(`config.json` → `code`). The exact command was:

```bash
python -m tools.held_suarez.climate_fixture --repeat 2
```

Two independent runs gave bit-identical states and statistics. A third,
earlier trial from `26f7f98`, whose solver source was the same, also matched
bit for bit. Each run took about 50 s.

Files:

| file | what |
| --- | --- |
| `config.json`, `events.jsonl`, `series.json`, `progress.json` | the run's own files, unchanged |
| `statistics.npz` | the production block accumulators (`daily_sample` → `OnlineStatistics`) |
| `states/state-d0000{1..4}.npy` | the saved states, the very states the daily samples were taken from |
| `gpu_daily_samples.npz` | each day's production `daily_sample` zonal fields, captured through the run's `sample_hook` (days 0–4) |
| `generation.json` | command, provenance, the repeat check, and the measured GPU-versus-host synthesis difference on every state |
| `SHA256SUMS` | receipt for every file above (not this README) |

`checkpoints/` is left out because it only serves resumption.

Measured on these four states:

- The stored buffer degree 22 is exactly zero. The highest nonzero degree
  and order are both 21, below 64/2.
- The largest |GPU − host| difference in u, v or T over the grid is 1.7e-14
  of max |field|.
- The CPU sections agree with the production daily samples and block means to
  4.5e-13 or better in absolute terms.

The test's tolerances are derived from a stated synthesis budget (1e-12 of
max |field|). They are not fitted. See the test's docstring.

Regenerating needs CUDA and replaces the fixture:

```bash
python -m tools.held_suarez.climate_fixture --repeat 2 --force
```

Other GPUs or library versions may give different last bits. Keep the
committed files unless the production path itself changes.
