# Historical run preservation — 2026-11-12

This record adds 24 capsules to the already preserved `hs-T42L20-prod-002`.
The selection follows published figures, numerical claims, before/after fixes,
and the complete RH4 validation series. It includes failed and partial runs;
no verdict or original byte has been rewritten.

The new capsules contain **620 files, 645,973,082 bytes** in the
[full Drive collection](https://drive.google.com/drive/folders/1Yjig-jolvU15gvCqTq-19kyjCbJi9iX1). Every one was downloaded
afresh into an initially empty ignored directory and verified against local-source
SHA-256 receipts. The committed subsets and archive metadata total **62,516,715
bytes**. This is independent of the previous 141-file HS run 002 verification.

Two accompanying analysis bundles add **7 files, 124,700 bytes**, also downloaded
and verified in full (127,409 bytes in Git including their archive metadata):

- [R3 product attribution](analysis/r3-product/): final states, summaries and the
  historical upper-band review diff cited by the [R3 record](../r3_characterization.md).
- [R4 timestep convergence](analysis/r4-convergence/): aligned arrays and summary,
  including the corrected order-estimator note. These are analyses, not capsules.

The complete per-capsule inventory, original paths, receipt hashes, omitted file
names and verification counts are in [inventory.json](inventory.json). The
[published-run index](../../runs/README.md#additional-preserved-evidence-2026-11-12)
gives each capsule's role and exact full/core sizes. Fetch a run by its catalogue
ID. Fetch an analysis bundle by passing its receipt directory, for example:

```sh
tropoi capsule fetch earth-gray-dry-heated-T21L25-d38d2a3-colab
tropoi capsule verify earth-gray-dry-heated-T21L25-d38d2a3-colab
tropoi capsule fetch docs/validation/run_archive_2026-11-12/analysis/r3-product
```

## Provenance checks and scientific limits

- All six radiative config objects match the corresponding committed evidence.
  Their raw config hashes differ **only because of CRLF/LF line endings**;
  both original hashes are recorded in [radiative-provenance.json](radiative-provenance.json).
  For the five completed runs, `means.npz` and `series.json` match byte-for-byte.
  Colab event logs also differ only by line endings. The partial run's published
  time series matches byte-for-byte; no absent published mean is invented.
- The July 13 RH4 pair is the one-day figure comparison; it must not be relabelled
  as the five-day matched-timestep result. The other three RH4 runs are retained
  as historical validation variants, not assigned unverified claims.
- The six orientation runs are named in `docs/assets/provenance.json`. Their
  dirty-worktree provenance remains visible; publishing bytes does not recover
  uncommitted source code or prove reproducibility from the named commit alone.
- HS run 001 remains B FAIL. The Dinosaur reference remains an external model,
  not Palintropos output. Its pickle is only hashed/downloaded, never deserialized.
- Existing W5/T63, two-vortex example and Mars capsules are not duplicated.
  Duplicate `.download` and extracted ZIP wrappers, package-test source trees,
  and exploratory/superseded runs without a traced claim remain in the owner's
  bulk archive. W5/T42 canonical reference work has its separate existing archive;
  the obsolete W5 acceptance runs are not substituted for it.

## Rescued interactive evidence

- [HS interactive isobars](../interactive_pressure_2026-11-08/hs_isobars_interactive.html):
  29,375,084 bytes, moved unchanged beside the committed issue #13/PR #24 cost
  and browser measurements. Its source is `hs-T42L20-prod-002`.
- [W5 saved-state explorer](../../runs/20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a/assets/w5_interactive.html):
  3,613,467 bytes, moved unchanged beside its canonical capsule. All four embedded
  source-file hashes match that capsule (coefficients, times, config, manifest).

Both page hashes and exact destinations are recorded in `inventory.json`. This
rescue preserves their bytes and source links; it does not repeat the historical
browser benchmarks or simulation integrations.
