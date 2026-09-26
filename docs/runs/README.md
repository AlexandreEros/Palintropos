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
receipt from inside the run directory with `sha256sum -c SHA256SUMS`;
`tests/test_w5_canonical_capsule.py` does so on every CI run.

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
  the committed one in a handful of antialiased pixels: 7–19 of 3.8 million
  were measured between processes. The fields, streamline seeds and computed
  streamline segments are bitwise identical from run to run; the variation
  arises when Matplotlib rasterizes the streamline layer.
- `overview_2026-07-30.png`: the previous README figure, cited by the
  validation report. It was drawn by
  [plot_swe_holistic.py](../validation/williamson_5/plot_swe_holistic.py)
  from this run's coefficients and diagnostics and from `aeolus_w5_t63.npz`,
  the grid package the validation notebook wrote from this same run. Its PNG
  metadata records the SHA-256 of all three.
