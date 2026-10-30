# Published runs

Palintropos runs that are committed on purpose as auditable evidence or
examples. Ordinary runs stay in the gitignored `runs/` directory. A
published run keeps the layout every run has, plus a checksum receipt:

```
docs/runs/<run-id>/
    manifest.json, config.json      configuration and provenance
    *_coeffs.npy                    saved spectral states
    *_snapshot_times.npy            their time axis (s)
    diagnostics/                    per-step diagnostics
    SHA256SUMS                      receipt for the files above, and only those
    assets/                         derived products: figures, sidecars, ...
```

`SHA256SUMS` covers the run's evidence and nothing derived from it. Nothing
under `assets/` is listed there. Each derived product records instead the
SHA-256 of the files it read: the PNG metadata and JSON sidecar of an
overview carry `CoefficientsSHA256` and `DiagnosticsSHA256`. Verify a
receipt from inside the run directory with `sha256sum -c SHA256SUMS`.
`tests/test_published_layout.py` checks every published run on every CI
run. For each run it checks that:

- the receipt lists every primary file and nothing under `assets/`;
- every digest matches the file;
- every figure under `assets/` names the coefficient and diagnostics hashes
  in the receipt;
- the run is listed in the index below;
- Git never converts its line endings.

Other directories hold different material:

- `docs/assets/` holds external or reference material that belongs to no
  single Palintropos run.
- `docs/validation/` holds written validation analyses and figures that
  combine several runs, or a run and an external reference.

## Index

### `20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a`

The canonical Williamson test case 5 run at T63: Gauss–Legendre grid
96 × 192, ℓ ≤ 63, inviscid, 15 days, 4 saved states. It was produced at commit
`668e6c9a` with a clean worktree, on Google Colab, on 2026-07-30. This is the
run compared with MRI-JMA in
[williamson5_mri_2026-07-30.md](../validation/williamson5_mri_2026-07-30.md).

- **Receipt provenance** (verified 2026-09-25): the files are byte-identical
  to the capsule inside `w5-mri-canonical-validation.zip`, whose SHA-256 is
  recorded in
  [CHECKSUMS.txt](../validation/williamson5_mri_2026-07-30/CHECKSUMS.txt).
  `swe_coeffs.npy` also matches the coefficient hash embedded in the
  2026-07-30 figure `assets/overview_2026-07-30.png`, and `manifest.json`
  parses equal to the report's receipt `aeolus_t63_run_manifest.json`.
- **Assets:** `overview.png` and `overview.json` are the README figure and its
  numbers, drawn by the pinned recipe
  [docs/figures/williamson5_t63_overview.py](../figures/williamson5_t63_overview.py).
  Redrawing reproduces `overview.json` byte for byte. The PNG can differ from
  the committed one in a few antialiased pixels (7–19 of 3.8 million were
  measured between processes for the earlier streamline figure). Since
  2026-09-26 the figure colours the free-surface perturbation η′ = H − H̄
  and draws ψ contours, chosen automatically because the divergent wind
  carries at most 0.08 % of the kinetic energy.
- `overview_2026-07-30.png`: the previous README figure, cited by the
  validation report. It was drawn by
  [plot_swe_holistic.py](../validation/williamson_5/plot_swe_holistic.py)
  from this run's coefficients and diagnostics and from `aeolus_w5_t63.npz`,
  the grid package the validation notebook wrote from this same run. Its PNG
  metadata records the SHA-256 of all three.

### `20260926T051829Z_two-vortices_rot24h_r4_l21_dt120h_e5dfc047_50ae9e93`

The README's BVE example: two opposite-signed vortices on a sphere with a
24 h rotation, Gauss–Legendre grid 128 × 256, ℓ ≤ 21, inviscid, 25 days,
6 saved states (`tropoi run bve --day-hours 24 --duration-days 25
--n-snapshots 6 --backend gauss-latlon`). It was produced at commit
`50ae9e93` on 2026-09-26, on an NVIDIA GeForce MX110.

- **Worktree:** the manifest records `dirty: true`, and the uncommitted
  changes were not recorded. They did not affect the evidence: re-running
  the same command from commit `93720a50`, with every file under `src/`
  committed, reproduced
  `vorticity_coeffs.npy`, `vorticity_grid.npy`, `bve_snapshot_times.npy`
  and both files in `diagnostics/` byte for byte (2026-09-27). Only
  `config.json` differed, in its `out` and `run_id` fields.
- **Conservation** (recorded every step): kinetic energy stays within
  6.1 × 10⁻⁴ of its initial value and absolute enstrophy within 1.1 × 10⁻⁴.
- **Assets:** `overview.png` and `overview.json` are the default
  `tropoi plot` overview (`sim.plot(sidecar=True)`), shown in the README.

### `20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a`

An idealized dry-CO₂ perturbation experiment over MOLA terrain, the main run
of [mars_evolving_2026-10-27](../validation/mars_evolving_2026-10-27/README.md).
It starts from the balanced resting atmosphere over real Mars relief plus a
1 K (2, 2) temperature coefficient (`orographic_thermal_wave`). Mars-like
constants; `l_max` 21, Gauss–Legendre grid 34 × 68, 4 σ levels; fixed 60 s
RK4 steps for one rotation period (88642.44 s); 9 saved states. It was
produced at commit `71c5441a` on 2026-10-27, on an NVIDIA GeForce MX110.

- **Terrain:** `terrain/` holds the prepared MOLA terrain the run integrated
  (band limit 14, digest `sha256:afb016d8…c35c31b8`). The capsule is
  self-contained: `open_simulation` and `view()` need neither the MOLA files
  nor a GPU.
- **Worktree:** the manifest records `dirty: true`. The experiment script
  recorded `git status --porcelain` at launch. The only modified tracked files
  were two example scripts, neither imported by the run, and nothing under
  `src/` was modified.
- **Evidence:** the halved-step twin and the resting control are not
  published. Their measured comparison with this run is in the record's
  `results.json`.
- **Assets:** none. The figures and browser captures combine this run with the
  others, so they live in the validation folder.
