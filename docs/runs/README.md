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
    SHA256SUMS                      byte-level evidence receipt
    archive.json                    optional full-cloud-capsule index
    assets/                         derived products: figures, sidecars, ...
```

Receipts identify preserved bytes and may include derived figures. Older fully
committed capsules retain their original primary-file receipts and embedded
figure-input hashes. Checksums prove byte identity, not scientific correctness
or faithful plotting: preserve source capsules and plotting recipes too.

Large capsules have a complete cloud copy and a smaller Git core. Their receipt
covers **every original file**, including states, checkpoints, reports and assets.
`archive.json` records a Drive file ID and size for each entry, the exact
`included_files` in Git, and provenance links. These two metadata files are not
original payload. CI checks every committed byte and requires an archive record
for every omitted file; it does not download external data.

From the repository root (no CUDA needed):

```sh
tropoi capsule verify hs-T42L20-prod-002 --core
tropoi capsule fetch hs-T42L20-prod-002
tropoi capsule verify hs-T42L20-prod-002
```

`fetch` downloads into `runs/<id>/`, verifies before publishing each file, and
reuses matching files after interruption. It refuses differing local files.
A verified file appears under its final name in one atomic step that never
replaces an existing file: a rename on Windows, a hard link on POSIX, or Linux
`renameat2(RENAME_NOREPLACE)` where hard links are unsupported. Before any
download, `fetch` probes each destination filesystem with hidden probe files it
removes again; if neither method works (for example some FUSE or network
mounts, possibly Colab's `/content/drive`), it stops with an error recommending
local storage such as `--path /content/runs/<id>`. Copy the verified capsule
afterwards and check the copy with `verify --path`. Reusing an already complete
capsule probes and writes nothing.

`verify` is offline, needs no write access, and fails for missing or corrupt
files. If the capsule directory does not exist it reports that once, suggesting
`fetch` or `--path`. `--core` explicitly checks the Git subset and reports
absent archive files; this is not a full pass. Use `--path DIR` for another
full-capsule location and `--catalog PATH` for a different local catalogue.
Passing a receipt directory directly to `verify` checks that directory unless
`--path` is supplied.

The catalogue currently requires a repository checkout; it is not in the wheel.
Drive files must be downloadable through the shared link. Private/quota-limited
downloads fail without accepting sign-in pages as evidence. No credentials or
permissions are changed. Private data can be downloaded through your own Drive
access and checked with `verify --path DIR`. Fetch is explicit, never triggered
by plotting. The receipt pins bytes: changes in the cloud fail verification.

For a full local copy, `sha256sum -c /absolute/path/to/SHA256SUMS` also works from
inside that capsule. All published runs are indexed and Git preserves their
line endings byte-for-byte.

Other directories hold different material:

- `docs/assets/` holds external or reference material that belongs to no
  single Palintropos run.
- `docs/validation/` holds written validation analyses and figures that
  combine several runs, or a run and an external reference.

## Index

### `hs-T42L20-prod-002`

Held–Suarez T42L20 production run: the historical passing climate evidence,
including its report and later assets. Full capsule: 141 files, 241,858,901 bytes.
Git core: 18 original files, 2,079,182 bytes, plus receipt and archive descriptor.
Only `states/` and `checkpoints/` are omitted; the receipt covers all files.

- [Full capsule on Drive](https://drive.google.com/drive/folders/1ID1ZhJJP2NoFPzWus4627dzwUBTgbEqQ):
  the owner's November 12 bulk upload, including later figures.
- [Historical Colab output](https://drive.google.com/drive/folders/1dBYGmQ9YsjQjrkh17cUbFpMX0MaSZDmr):
  identified by the owner as the incrementally saved source of the local download.
  This older folder has no `assets/`; it is provenance, not a full mirror of the
  later capsule.
- [Climate validation record](../validation/held_suarez_climate_2026-11-04/).
  Preservation does not independently revalidate the historical scientific claims.
- **Archive read-back verified 2026-11-12:** `capsule fetch` downloaded all
  141 files from the new Drive copy into an empty directory. Every SHA-256
  matched the receipt generated from the local original, including all 120
  states and three checkpoints (241,858,901 bytes total). The historical
  Colab folder was inspected for provenance, not fully downloaded or hashed.

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

## Additional preserved evidence (2026-11-12)

These 24 capsules supplement the HS run 002 entry above. Every full capsule is
in the owner's Drive collection and has been downloaded afresh and verified
against its committed `SHA256SUMS`. `archive.json` provides the file IDs used by
`tropoi capsule fetch <id>` and the exact Git file list. All original bytes are
unchanged. Original dirty-worktree flags and failed/incomplete verdicts remain
part of the evidence; preservation is not a new scientific validation.

The Dinosaur entry is explicitly an **external-reference** capsule. Its pickle
checkpoint is archived as opaque bytes, never loaded or executed by fetch/verify.
Its large `series.npz` and `checkpoint.pkl`, the HS/radiative states/checkpoints,
and four large R5 montage PNGs stay in Drive. Every omission is named in its
descriptor; the older compact RH4 and orientation capsules are committed whole.

The [preservation record](../validation/run_archive_2026-11-12/README.md) records
the selection, exact sizes, source paths, verification, and config-hash matching.
The original R3/R4 analysis bundles live with that validation record rather than
being labelled simulation runs. The rescued W5 explorer is in its source
capsule's `assets/w5_interactive.html`; the HS interactive-pressure page is beside
its [cost measurements](../validation/interactive_pressure_2026-11-08/README.md).

| Capsule | Evidence | Git bytes (including receipts) | Full archive bytes |
| --- | --- | ---: | ---: |
| `hs-T42L20-prod-001` | Held–Suarez failed B2 baseline (lagged damping), compared with passing run 002; retain the failure verdict. [Drive](https://drive.google.com/drive/folders/1wjk0yT3a_HmNCTM3e7VH2vVi0Q1CnJz9) | 1,104,925 | 240,842,357 |
| `hs-reference-dinosaur-T42L20-001` | External Dinosaur T42L20 reference used for the HS climate comparison; not a Palintropos simulation. [Drive](https://drive.google.com/drive/folders/1vvahNZIv_6tamBW0DvH6IYEQBLuFTqWd) | 13,721 | 174,273,168 |
| `20260712T033832Z_two-vortices_norot_r4_l21_dt10h_89985aaa` | Post-R5 non-rotating control; documents the absolute-vorticity construction correction. [Drive](https://drive.google.com/drive/folders/1wf37QDAaCVCepwrcu5xOTt5Gb0TXV4I6) | 2,370,601 | 16,914,842 |
| `20260712T033635Z_two-vortices_rot24h_r4_l21_dt10h_89985aaa` | Post-R5 rotating baseline; documents the absolute-vorticity construction correction. [Drive](https://drive.google.com/drive/folders/1gtwGNJpGo7YCEVm4arZwioRcIttOrJ3Q) | 2,499,316 | 17,669,522 |
| `20260712T032424Z_two-vortices_norot_r4_l21_dt10h_0b6c1351` | Pre-R5 non-rotating control; documents the absolute-vorticity construction correction. [Drive](https://drive.google.com/drive/folders/12MR88QW1NwYNc9lb8zwySVB6K2feuWbI) | 2,393,924 | 16,901,300 |
| `20260712T023305Z_two-vortices_rot24h_r4_l21_dt10h_501bce43` | Pre-R5 rotating baseline; documents the absolute-vorticity construction correction. [Drive](https://drive.google.com/drive/folders/1yxVVAWY6qJPDgmZo14YOR8_u_HdYowVE) | 2,500,341 | 17,674,127 |
| `20260923T225840Z_rh4_rot24h_r4_l21_dt6h_9e5d0eb1` | Additional historical RH4 validation capsule; retained with the complete validation series, without assigning it the five-day comparison claims. [Drive](https://drive.google.com/drive/folders/1VlI4JvsGsnYCvVxBMQuAxnr35IbE3z4B) | 7,452,763 | 7,446,262 |
| `20260717T004614Z_rh4_rot24h_r4_l21_dt6h_bb87c06a` | Additional historical RH4 validation capsule; retained with the complete validation series, without assigning it the five-day comparison claims. [Drive](https://drive.google.com/drive/folders/1TWVeD41WORgiGTykcblYVw8CgH6sYVRn) | 4,560,960 | 4,557,568 |
| `20260717T004524Z_rh4_rot24h_r4_l21_dt6h_bb87c06a` | Additional historical RH4 validation capsule; retained with the complete validation series, without assigning it the five-day comparison claims. [Drive](https://drive.google.com/drive/folders/1IXvhfcPlxJ5AJWZ0gtfQPrwEmdRoYgUq) | 4,560,661 | 4,557,269 |
| `20260713T022542Z_rh4_rot24h_r4_l21_dt6h_4a840226` | One-day geodesic RH4 source of the published backend-comparison figure (not the five-day matched-timestep experiment). [Drive](https://drive.google.com/drive/folders/1_g1OWLCVW2-QcyRoFclxg-vZV2UM6-pE) | 4,561,526 | 4,558,134 |
| `20260713T022748Z_rh4_rot24h_r4_l21_dt6h_4a840226` | One-day Gauss lat–lon RH4 partner of the published backend-comparison figure. [Drive](https://drive.google.com/drive/folders/1_cO4tfpshNd0HfS0LdT2MkPSaYEtsxKS) | 4,392,248 | 4,388,852 |
| `20260712T080551Z_rh4_rot24h_r4_l21_dt6h_0678ac18` | Post-R3 RH4 CLI smoke named in the fine-product-grid characterization. [Drive](https://drive.google.com/drive/folders/143avDu8dpldaSvdBPM462f7r0OCcqfLf) | 4,561,828 | 4,558,442 |
| `20260713T024914Z_two-vortices-equator-90e_norot_r4_l21_dt12h_4a840226` | Non-rotating orientation comparison, explicitly cited by VALIDATION.md and the legacy figure provenance. [Drive](https://drive.google.com/drive/folders/1T9xnzE0M_e6Dz4KqSBIAVBujmAT01rgG) | 3,117,864 | 3,114,379 |
| `20260713T024900Z_two-vortices-equator-0e_norot_r4_l21_dt12h_4a840226` | Non-rotating orientation comparison, explicitly cited by VALIDATION.md and the legacy figure provenance. [Drive](https://drive.google.com/drive/folders/1FJXCauK1m87UaWkPIj5iWV80egPiGviW) | 3,115,205 | 3,111,724 |
| `20260713T024849Z_two-vortices-north-pole_norot_r4_l21_dt12h_4a840226` | Non-rotating orientation comparison, explicitly cited by VALIDATION.md and the legacy figure provenance. [Drive](https://drive.google.com/drive/folders/1g_lzQ9xzBaB5x_cY2-MRAoIIBdZB1j-Q) | 1,550,066 | 1,546,587 |
| `20260713T044053Z_two-vortices-equator-90e_rot24h_r4_l21_dt24h_4a840226` | Rotating orientation comparison, explicitly cited by VALIDATION.md and the legacy figure provenance. [Drive](https://drive.google.com/drive/folders/1huDUL6l-Q0pMDhgK6lw8FiEiHNSznIVY) | 5,276,305 | 5,272,813 |
| `20260713T044027Z_two-vortices-equator-0e_rot24h_r4_l21_dt24h_4a840226` | Rotating orientation comparison, explicitly cited by VALIDATION.md and the legacy figure provenance. [Drive](https://drive.google.com/drive/folders/1x4JNWglgnKwMw98M9n01KE6agASWMmd1) | 5,234,448 | 5,230,960 |
| `20260713T044006Z_two-vortices-north-pole_rot24h_r4_l21_dt24h_4a840226` | Rotating orientation comparison, explicitly cited by VALIDATION.md and the legacy figure provenance. [Drive](https://drive.google.com/drive/folders/1jCNd2x29r4GEkictaeyNdA19KHAsbfaI) | 2,399,116 | 2,395,629 |
| `earth-gray-dry-T21L25-b2b8ea2` | Completed 300-day Earth gray-radiation baseline before dissipative heating. [Drive](https://drive.google.com/drive/folders/1c7uqL8Y5Wegg1JPqqFnmNqW7rpsOuRLp) | 180,061 | 22,872,407 |
| `jupiter-sl09-T21L30-b2b8ea2` | Completed 200-day Jupiter gray-radiation baseline before dissipative heating. [Drive](https://drive.google.com/drive/folders/1R5mbp1WfvmiKuV6tSWScdz5Vu9-sh0C4) | 127,507 | 18,835,779 |
| `earth-gray-dry-heated-T21L25-d38d2a3` | Completed 300-day heated Earth run on MX110; energy-budget correction comparison. [Drive](https://drive.google.com/drive/folders/1ndQqRmVulKjmJyh70aG92nEmABnXNuKI) | 180,809 | 22,874,896 |
| `earth-gray-dry-heated-T21L25-d38d2a3-colab` | Completed heated Earth run on Colab T4; cross-hardware and checkpoint-resume evidence. [Drive](https://drive.google.com/drive/folders/1WYOLLDPrEs5zUmi4okbNsgtru3FiWVbm) | 209,321 | 22,901,845 |
| `earth-gray-dry-T21L25` | Earlier 30-day Earth development run, explicitly published as partial evidence; not the later 300-day run. [Drive](https://drive.google.com/drive/folders/1xZMOPpGXDcACwQOB1n_1YDmcj6mDJ13D) | 21,802 | 4,635,034 |
| `jupiter-sl09-heated-T21L30-d38d2a3-colab` | Completed heated Jupiter run on Colab T4; checkpoint-resume evidence. [Drive](https://drive.google.com/drive/folders/1SMmZ1M7K05E6g5on0eRWyTMhXHPGWMO7) | 131,397 | 18,839,186 |
| `earth-gray-dry-heated-T21L25-d376c67` | R1 of the controlled comparison (2026-11-13): the heated Earth preset as shipped, Colab L4; criteria C1/C4. [Drive](https://drive.google.com/drive/folders/12LVuArfZUs39wlwCDtpFhiJ4ms4HzZrf) | 181,507 | 22,875,657 |
| `earth-gray-dry-T21L25-d376c67` | R2 of the controlled comparison (2026-11-13): the same preset with `--no-frictional-heating`, same L4; criteria C2/C3. [Drive](https://drive.google.com/drive/folders/16P4sZ3PXhg8YBKKP-4QltGd9NBM-6ecC) | 180,661 | 22,873,098 |
