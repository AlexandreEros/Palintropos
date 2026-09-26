# docs/assets

This directory is for **external or reference material that belongs to no
single Palintropos run**: for example a reference solution from another model
that a validation study compares against, or a canonical Held–Suarez dataset
if one is adopted. Other material has its own home:

- A figure or product derived from one Palintropos run is an asset of that
  run: `docs/runs/<run-id>/assets/` for published runs, `runs/<run-id>/assets/`
  for local ones (the default of `tropoi plot`).
- A figure that combines several runs, or a run and an external reference,
  sits beside its validation analysis under `docs/validation/<study>/`.

No external reference data is committed yet. The MRI-JMA reference packages
for Williamson 5 are kept outside Git; their SHA-256 sums are in
[CHECKSUMS.txt](../validation/williamson5_mri_2026-07-30/CHECKSUMS.txt).

## Legacy figures awaiting published source runs

The files below predate this layout. `docs/readme_figures.py` drew them from
local runs made at commit `4a840226` with uncommitted changes
([provenance.json](provenance.json) records `dirty: true` for every one). No
committed run can reproduce them, so they cannot honestly be assigned to a run
and stay here until each is replaced by a figure drawn from a published run.

| File | Used by | Source runs | Replacement |
| --- | --- | --- | --- |
| `two_vortices_evolution.png` | README | one BVE run (`two_vortices`, 10 days) | the overview of a clean, published two-vortex run |
| `two_vortices_rotation_comparison.png` | VALIDATION.md | several BVE runs | a validation-study figure from published runs |
| `two_vortices_rotating_streamlines.png` | VALIDATION.md | several BVE runs | as above |
| `rh4_geodesic_vs_latlon.png` | VALIDATION.md | two RH4 runs (geodesic, Gauss) | as above |
| `rh4_simulation_summary.png` | nothing | one RH4 run | none; obsolete, proposed for deletion |
| `provenance.json` | the figures above | | removed with the last of them |
