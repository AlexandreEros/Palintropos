# Palintropos notebooks

| Notebook | Runtime and use | Evidence |
| --- | --- | --- |
| [Heated radiative runs](radiative_colab_pilot.ipynb) | Colab GPU. Time a warmed-up Earth or Jupiter run, profile it, or resume from a checkpoint. | [Colab run results](../docs/radiation/README.md#the-same-heated-runs-on-a-colab-tesla-t4-2026-11-1112-commit-d38d2a3), [full capsule catalogue](../docs/runs/README.md) |
| [CUDA regression suite](Palintropos-Colab-CUDA-Regression.ipynb) | Colab GPU. Run pytest against a selected revision and download the logs and JUnit results. | [Recorded Linux/CUDA regression](../docs/radiation/TEST_RESULTS.md#full-suite-on-linuxcuda-at-341acfe-2026-11-12-colab-tesla-t4) |
| [Held–Suarez launcher](held_suarez_colab.ipynb) | Colab GPU and Google Drive. Launch or resume the pinned 1200-day T42L20 experiment with a verified Dinosaur reference and a runtime budget gate. | [Protocol and acceptance record](../docs/held_suarez/README.md), [preserved production run](../docs/runs/README.md#hs-t42l20-prod-002) |
| [Williamson 5 reference preparation](mri_w5_reference_preparation.ipynb) | CPU. Prepare and verify the MRI-JMA reference package for T42 and T63. | [Williamson 5 validation report](../docs/validation/williamson5_mri_2026-07-30.md) |
| [Williamson 5 comparison](w5_mri_validation_colab.ipynb) | GPU for verification and integration, CPU for postprocessing. Compare the pinned Palintropos predecessor with the MRI-JMA reference. | [Williamson 5 validation report](../docs/validation/williamson5_mri_2026-07-30.md) |

## Radiative timing and production runs

Select a GPU runtime, choose `CASE = "earth"` or `"jupiter"`, and run the
setup and timing cells (1–4). The notebook checks out `d38d2a3` and uses
the production loop with frictional heating enabled. The pilot stops at a
checkpoint. Keep the same case and run directory to resume with the optional
production cell. The default Google Drive location keeps checkpoints across
Colab sessions. The optional profiling cells use scratch runs. Their
synchronised timings describe component cost, while the uninstrumented pilot
measures throughput. See the [pilot tooling](../tools/colab/README.md).

## CUDA regression

Set `REF` to a branch or commit and `SUITE` to `routine`, `slow` or
`all`. Set `REF` explicitly if the default PR #29 branch has been deleted.
The notebook records the resolved SHA, runs pytest using Colab's CuPy, and
downloads an evidence ZIP containing the environment, log, JUnit XML when
available, and per-test skip reasons. For the historical full run, select
`341acfeb9e9331d3a9a53e0eecceb966177cebf4` and `all`. The documented
result applies to Linux with Python 3.13 and CuPy 14. The
[test record](../docs/radiation/TEST_RESULTS.md) also lists the separate
Windows Python 3.12/CuPy 13.4 runs and their coverage.

## Williamson Test Case 5 — MRI-JMA vs Aeolus

> **Project name:** Aeolus is now Palintropos (distribution `palintropos`,
> import package `tropoi`, command `tropoi`). The historical model name,
> pinned commits, and artifact filenames below are retained for provenance.
> Notebook B is deliberately *not* migrated: it clones the pinned pre-rename
> commit `668e6c9a` and therefore keeps `AEOLUS_REPO_URL`, its
> `planetary_sandbox.*` imports, and `python -m planetary_sandbox.cli.main`.
> Those names describe that commit, not this checkout. GitHub redirects the
> old `AlexandreEros/Aeolus` URL to `AlexandreEros/Palintropos`, so the clone
> still resolves; re-pointing the notebook at a post-rename commit means
> switching it to `tropoi.*` at the same time.

Two small, self-contained Colab notebooks that answer one question:

> Does Aeolus reproduce the MRI-JMA Williamson test-case-5 solution when both
> models are compared on the **same physical fields** and the **same
> initial-value problem**?

**Current status: yes, within the documented envelopes.** The day-zero physical
contract passes at T42 and T63, both 15-day runs completed, and the accepted
result — with figures, conservation diagnostics, comparison tables, and full
provenance — is
[`docs/validation/williamson5_mri_2026-07-30.md`](../docs/validation/williamson5_mri_2026-07-30.md).

The path there matters. **`W5_MRI_SEMANTIC_AUDIT.md`** is the source-cited audit
that establishes what MRI's archived `h` is, what Aeolus's `phi / Phi0 / phi_s`
are, and whether the two initial conditions describe the same problem. Its
finding was that they did **not**: Aeolus prescribed the Williamson case-2 field
as layer *thickness*, while MRI and Williamson prescribe it as the *free
surface*. Notebook B's day-zero gate failed by design and refused the 15-day
runs until the initial condition was corrected — which it was, in commit
`668e6c9a`, *fix(swe): canonicalize Williamson-5 initial state*. Every
pre-correction W5 trajectory is superseded.

### Workflow

```
Notebook A (CPU)                        Notebook B (GPU except postprocess)
mri_w5_reference_preparation.ipynb      w5_mri_validation_colab.ipynb
--------------------------------        -----------------------------------
download MRI archive (sha256)     -->   MODE="verify":
verify archived-field semantics           verify package hashes + schema
   (day-0 == case-2 surface,              verify field contract
    STDOUT mass identity)                 day-zero physical comparison
build physical fields on the              6-h smoke run, then STOP
   t42 (64x128) / t63 (96x192)          MODE="run_t42"/"run_t63"/"run_both":
   Gaussian grids                         15-day runs -- ONLY if the day-zero
freeze: npz per grid + manifest           contract passed (or explicit
   (hashes, units, equations)             override) -- then postprocess
zip the package for handoff             MODE="postprocess":
                                          figures + metrics from saved
                                          outputs (CPU; never re-runs)
```

The reference package moves from A to B through the browser: Notebook A writes
`mri-w5-reference.zip` and downloads it; Notebook B uploads and extracts it. No
Google Drive mount is required.

### Field contract

```
MRI:     free_surface_height = archived h
         layer_depth         = archived h - topography_height
Aeolus:  layer_depth         = (Phi0 + phi) / gravity
         free_surface_height = (Phi0 + phi + phi_s) / gravity
         topography_height   = phi_s / gravity   (band-limited cone)
topography_height (reference) = 2000 * (1 - min(r, pi/9)/(pi/9)),
         r = coordinate-plane distance to (30N, 270E), analytic, static
```

Both notebooks assert these identities; nothing is inferred from numerical
agreement between the two models. Because Aeolus carries a **band-limited** cone
and the reference an **analytic** one, the raw `layer_depth` difference includes
a large static terrain-representation term; removing it returns the free-surface
difference identically. See § 6 of the validation report.

### Usage

1. Run Notebook A once (any runtime; ~320 MB download from
   `climate.mri-jma.go.jp`). Output: `mri-w5-reference/` with two `.npz`
   packages, `manifest.json`, check figures, and `mri-w5-reference.zip`.
2. Open Notebook B on a **GPU runtime**, upload the ZIP when prompted, set
   `MODE = "verify"`, run all cells, and read the day-zero verdict. `verify`
   never starts a 15-day integration.
3. To reproduce the accepted result: set `MODE = "run_both"` (the mode these
   notebooks were last executed with); results and comparisons land in
   `w5-validation/`. `MODE = "postprocess"` re-renders comparisons without ever
   re-integrating; `FORCE_RERUN = False` reuses valid completed outputs.

Notebook B pins Aeolus to commit `668e6c9a5735b2d7200dcc121eb0f7c40450ae08` —
the commit that produced the canonical result — and asserts that the checkout
actually landed on it. Edit `AEOLUS_COMMIT` in the settings cell to validate
another commit. Notebook A never installs or imports Aeolus; Notebook B installs
only genuinely missing dependencies (CuPy on Colab) and never restarts the
kernel itself.

These are the versions that were executed, with their outputs stripped for
version control. They are not byte-identical to the copies committed at
`668e6c9a`, whose pin was still stale; the authoritative record of what ran is
the `manifest.json` inside each run capsule, which records argv, configuration,
environment, GPU, and clean-worktree status independently of any notebook.
