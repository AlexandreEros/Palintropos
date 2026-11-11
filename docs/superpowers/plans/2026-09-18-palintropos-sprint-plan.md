# Palintropos Refactor: Three Implementation Sprints

**Goal:** Make saved BVE, SWE, and PE runs inspectable through one useful public interface, then organize the implementation around the boundaries that interface proves.

**Architecture:** Keep packed coefficient storage, numerical kernels, and scientific plot compositions. Add named host-array views, a read-only simulation/snapshot interface, and structured provenance. Move existing modules only after this path works.

**Execution:** Three substantial implementation prompts, with roughly one to three additional runs available for continuation, review, or repair. This is an execution budget, not a guarantee or a reason to reduce verification. A sprint contains multiple phases and reviewable commits; phases are not separate user prompts.

**Status:** Replacement proposal, 2026-09-18, grounded in source at `fd0ddff`. Writing this document does not execute its sprints. When the user requests a sprint, carry it through implementation and verification without stopping at each phase for another plan or approval.

This replaces the execution structure of [the earlier plan](2026-09-18-palintropos-implementation-plan.md). Sprint 1 covers A and B; Sprint 2 combines C and D; Sprint 3 covers E. The earlier draft and `.github/issue-drafts/` remain historical context, not additional task lists. The model for execution is the completed PE tendency sprint: one concrete objective, authoritative source, internal phases, independent acceptance checks, and a final evidence report.

## Working agreement for every sprint

- Read `docs/ARCHITECTURE.md`, the relevant implementation and tests, and this sprint's section. Verify the starting revision, branch, working tree, and available CPU/CUDA environment. Use a feature branch; preserve unrelated tracked and untracked work. Continue from the verified predecessor when sprints are stacked rather than assuming it has merged to `main`.
- Resolve ordinary implementation details, repair test failures caused by the work, and continue through the sprint. Ask only when a finding requires changing the public scientific contract, extending scope, or choosing between incompatible user goals. Unexpected evidence is a reason to investigate, not to add a new planning ceremony.
- Use the existing Python/NumPy/CuPy/pytest/Matplotlib stack and `venv/Scripts/python.exe`. No new runtime dependency is needed. Keep `tropoi`, distribution and executable names, and existing command behavior.
- Preserve numerical arithmetic order, constants, coefficient layouts, output schedules, and existing initialization behavior except for the explicitly described preset acceptance changes in Sprint 1. Stored capacity, state support, product sampling, and tendency projection remain distinct.
- Keep CLI help, config resolution, archive metadata, schema inspection, and host coefficient access free of CUDA and visualization imports. Treat GPU synthesis/rendering as explicit operations.
- Use saved W5 data for reload and artifact verification. Do not rerun W5, regenerate its authoritative archive, change its benchmark construction, or enable its long acceptance gate. Other tiny numerical checks and the existing R3 regression are part of verification.
- Keep one internal checklist. No issue filing, milestone setup, extra design document, or separate plan per phase is required. Put enduring contracts in source/docstrings and user documentation, scientific findings beside their tests, and branch motivation plus evidence in the final report or PR description.
- The commit sequences below specify review boundaries. Follow the session's commit authorization; the user's existing requirement to see the exact documentation diff before committing documentation still applies. Present that diff once the sprint's documentation is ready, rather than stopping implementation at each documentation edit. Push, publish, and merge only when authorized.

Run focused checks while developing and the applicable complete suite at the end of each implementation sprint. Reuse a recorded baseline for the same starting revision/environment. Report skips, pre-existing failures, and unexecuted checks separately from passes. Do not weaken a test to hide a regression or promise numerical equivalence based only on import checks.

## Sprint 1 — Establish support behavior and consolidate the cutoff

**Deliverable:** One import-light cutoff definition, a compact measured account of preset boundary behavior, and explicit CPU/factory validation. No change to the tendency equations or state projection.

**Starting branch:** `feat/support-contracts`. Main implementation surfaces are `src/tropoi/numerics/spectral_operators.py`, `src/tropoi/physics/{primitive_equations,shallow_water}.py`, `src/tropoi/run/bve/diagnostics.py`, and SWE/PE config and initial-condition factories.

### Internal phases

1. **Capture the baseline and characterize boundaries.** Record each preset's literal coefficient indices or grid-to-modal construction, minimum storage, advertised behavior, and required active terms. Exercise CPU resolution at the transition degrees. Use small Gauss-backend probes first, then repeat the short probes on geodesic with residuals recorded separately:
   - PE `thermal_wave`, `lmax=2`: upper temperature and log-pressure remain unchanged; upper divergence receives the hydrostatic forcing. Check the tendency directly and two one-second, stage-validated RK4 increments.
   - SWE `gravity_wave`, `lmax=4,5`: verify the upper linear divergence/geopotential pair with flat terrain and zero hyperdiffusion. Include a small lower-band comparison with upper divergence/geopotential inputs removed so the report does not mislabel the whole system as linear.
   - Williamson-2, flat bottom, `lmax=2,3`: measure the steady-state residual with identical physical constants and identify the degree-2 pressure/product cancellation. Record Williamson-5's missing capacity guard from source and CPU resolution; no W5 integration.
   - Keep exact-zero assertions separate from quadrature tolerances. BVE's broader invariant-budget investigation remains outside this sprint.
2. **Extract the helper mechanically.** Create `src/tropoi/support.py` with `product_truncation_cut(l_max: int) -> int`, preserving `(2 * int(l_max)) // 3`. Replace all six existing production formulas: three operator sites, the PE definition, the SWE cached cut, and the BVE diagnostic band. Retain the PE import path as a re-export of the same function. Keep slices, masks, diagnostic weighting, and truncation locations unchanged. The neutral module avoids a `numerics -> physics` dependency and a heavy package initializer.
3. **Apply preset validation in a separate change.** Use these conservative defaults, recording the evidence and any changed acceptance boundary:

   | Preset | Contract for this sprint |
   | --- | --- |
   | PE `thermal_wave` | Keep the storage check `lmax >= 2`. For nonzero amplitude, require `product_truncation_cut(lmax) >= 2` once the probe confirms the frozen-temperature mechanism. Zero amplitude keeps its current capacity boundary. Preserve direct low-level state construction for characterization. |
   | SWE `gravity_wave` | Keep support for `lmax=4,5`; add the existing `lmax >= 4` capacity check to CPU resolution. Document the upper-mode linear regime without claiming that all lower-band dynamics are linear. |
   | SWE `williamson2` | Add capacity `lmax >= 2`. Require retained degree >= 2 if the probe confirms that losing the balancing degree-2 products produces the residual at `lmax=2`. Document this as the supported preset boundary. |
   | SWE `williamson5` | Add capacity `lmax >= 2` only. Leave terrain and benchmark policy intact. |

   Mirror each adopted requirement at the public factory boundary. If GPU evidence is unavailable or contradicts the prediction, land only the mechanical extraction and capacity guards; retain the existing support acceptance and report that decision as unresolved. Do not infer a new restriction from an unexecuted test. Migrate affected characterization assertions deliberately, retaining low-level reproductions of the old behavior.

### Acceptance and commit boundaries

- Add `tests/test_support_characterization.py`, `tests/test_truncation_contract.py`, and `tests/cli/test_truncation_imports.py`. Extend existing config/factory tests for values just below, at, and above each boundary, including zero PE amplitude.
- Pin the helper at `1->0, 2->1, 3->2, 4->2, 5->3, 6->4, 10->6, 15->10, 21->14, 42->28, 63->42`; prove in a fresh subprocess that importing it does not load the CLI's forbidden heavy modules.
- Before and after extraction, compare probe outputs within the same backend/environment. Run `tests/test_r3_fine_product.py::test_prediction_p1_5day_energy_drift` whenever CUDA is available, record its actual drift before/after, require the same result within observed deterministic reproducibility, and preserve the historical assertion. A missing CUDA run is outstanding verification, not a pass.
- Run relevant spectral-operator, SWE, PE-tendency, topography, CLI, and configuration tests, followed by the applicable complete suite. Preserve representative default resolved configs and scientific run-id hashes.
- Save the support table and measured results in `docs/validation/preset_support_characterization.md`; distinguish source facts, predictions, measurements, and adopted policy.

**Reviewable sequence:** characterization and baselines; behavior-preserving extraction; preset guards and compatibility notes. These are internal boundaries in one sprint. Unmeasured support decisions do not prevent Sprint 2 from reading archives honestly.

## Sprint 2 — Deliver the saved-run API and its provenance

**Deliverable:** A user can open an existing BVE, SWE, or PE capsule, inspect it on CPU, access named read-only coefficients, and explicitly render an existing scientific composition. New capsules carry the structured definitions needed to interpret them. This is the first useful release.

**Starting branch:** `feat/saved-run-api`, based on the verified predecessor. The main existing integration points are `src/tropoi/run/bve/io.py`, the three CLI solver modules, `src/tropoi/cli/main.py`, and the existing per-core visualization adapters. Existing PE coefficient ordering is already in manifest notes; turn that knowledge into structured data rather than claiming it is absent.

### Public interface and implementation boundaries

```python
from tropoi.representation.archive import open_simulation

sim = open_simulation(run_path)
sim.times                         # saved times in seconds; read-only host array
snapshot = sim[137]                # saved output index, not interpolated time
snapshot.time
snapshot.metadata                 # schema, units, environment and provenance
snapshot.state["temperature"].coeffs  # PE host view, with a level axis
snapshot.plot(output_path=path)    # explicit lazy operation; returns pathlib.Path
```

Create the minimal modules below with light package initializers. Build one working BVE path, then exercise SWE and PE before generalizing further; finish all three in this sprint.

| New module under `src/tropoi/` | Responsibility |
| --- | --- |
| `spatial/modes.py` | Immutable field specifications and `SpectralModes` host views; named `SpectralState` mapping over existing packed storage. |
| `temporal/simulation.py` | Read-only `Simulation` and contextual `Snapshot` over a small storage protocol; independent of capsule format and plotting implementation. |
| `representation/archive/capsule.py` | `open_simulation`, file layout, time axes, legacy interpretation, and access to coefficient storage. |
| `representation/archive/schema.py` | One versioned field/support/diagnostic definition used by the reader and new-manifest writer. |
| `representation/visual/snapshot.py` | Lazy adaptation to current per-core figure builders and the existing path-returning renderer. |

### Internal phases

1. **Read existing data correctly.** Support BVE `zeta`; SWE `zeta`, `delta`, `phi`; PE `zeta`, `delta`, `temperature`, `ln_ps`. Preserve thickness-geopotential, full-temperature, and log-pressure semantics. Slice PE's `[zeta_1..K, delta_1..K, T_1..K, ln_ps]` stack without padding surface fields or copying the stack for every field. Keep state/rate semantics explicit; do not add wrapper arithmetic or rewrite RK4. Compare the metadata needed by each operation (basis, geometry, support, units and vertical coordinates); a human-readable ProductSpace label is not a compatibility key.
2. **Establish ownership and CPU behavior.** Prefer read-only NumPy memory maps; keep their owner alive with snapshots/views. Validate file headers, layout, dtype, and finite strictly increasing time axes with matching frame counts at open. Allow empty saved sequences; indexing them fails clearly. Validate coefficient finiteness, triangular support, and reality conventions when accessed, without silently repairing data or scanning an archive just to show metadata. Preserve per-variable monopole rules. A previously obtained snapshot must remain valid when another snapshot is selected or the `Simulation` variable is released.
3. **Write and consume the same schema.** Add versioned `state_schema` and `diagnostic_definitions` manifest blocks, containing field names/row layout, dimensions, units, normalization and reality conventions, support/capacity, geometry and vertical coordinates, time units, and relevant prescribed environment/initialization provenance. Serialize values from actual configuration and shared definitions. Preserve notes and existing CSV meanings. For `high_l_enstrophy_frac`, record `l > cut`, triangular domain, positive-order multiplicity, denominator and zero-power convention, and definition version. Keep descriptive additions outside the scientific config hash.
4. **Handle legacy capsules explicitly.** Infer only known formats, including missing-solver legacy BVE, and mark the inference and convention source. Conflicting or unknown field layouts must fail typed access with a useful error; ordinary metadata inspection should remain possible. Do not reconstruct old archives through newly restrictive preset validators or project their arrays on load.
5. **Connect inspection and plotting.** Extend existing `tropoi inspect RUN_PATH` with optional `--snapshot INDEX` and `--field NAME` arguments; a field requires a snapshot and must belong to that core. Print time, field interpretation, shape and coefficient summary without CUDA. These explicit arguments are the bounded selector interface for this release; no expression evaluator. Preserve ordinary metadata inspection for existing and incomplete capsules.
6. **Reuse scientific rendering.** On explicit plot calls, lazily create/cache the existing model/backend resources per simulation and pass physical fields into current adapters. All fields and snapshots share the appropriate geometry/resource identity. Preserve interpolation, layouts, palettes, and timeline normalization where comparisons use multiple frames; disclose single-frame normalization when applicable. SWE/PE synthesis is CUDA-gated. Do not promise that every BVE composition is CPU-only merely because a grid field was saved. Missing CUDA must leave metadata and host access usable and produce an actionable plotting error.

### Acceptance and commit boundaries

- Add focused host tests for named views, storage ownership, snapshot independence and legacy/schema reading, plus CLI subprocess tests in the existing `tests/cli` infrastructure. Cover all three array layouts, nonuniform PE sigma levels, missing-solver BVE, empty sequences, malformed headers/time axes, invalid field/index selection, unknown schema versions, and invalid accessed coefficients.
- Assert NumPy memory sharing where promised and rejected writes through public views. Test shared geometry and lazy resource construction separately from shared coefficient storage. CPU import probes must actually open a temporary capsule and run `inspect`, not just import a module or display help.
- Verify new manifests through the real writer/lifecycle integration, with new and legacy reader tests. Assert unchanged scientific hashes and diagnostic-column values. Synthetic tiny capsules are appropriate for malformed-input tests; real saved capsules establish compatibility.
- On CUDA, compare one representative saved BVE, SWE, and PE frame through old and new paths with identical composition and normalization. Check numerical plot inputs and rendered artifacts, inspect the images, and record any allowable renderer variation. Reuse authoritative W5 evidence without changing it.
- Run the relevant persistence, CLI, snapshot, visualization, and numerical tests and the applicable complete suite. Measure metadata-open cost, snapshot array sharing, and resource construction counts on representative saved data; opening metadata must not scale with total coefficient payload.
- Add a short user guide at `docs/SAVED_RUNS.md` containing the public example, legacy inference rules, CPU/GPU boundary, and plotting normalization behavior.

**Reviewable sequence:** host schema/views and capsule reader; simulation/snapshot and CLI access; additive manifest provenance; existing-renderer integration and user guide. If a continuation is needed, resume the same sprint from its working tree and remaining checks; do not restart discovery or create another design plan.

## Sprint 3 — Reorganize the proven implementation

**Deliverable:** Responsibilities live under `spatial`, `temporal`, and `representation`, while old imports, commands, saved capsules, and numerical results remain compatible.

**Starting branch:** `refactor/spatial-temporal-representation`, after Sprint 2's end-to-end path passes. Start from the actual verified revision. This sprint is unnecessary for users to benefit from Sprint 2, but is part of completing the refactor.

### Internal phases

1. **Map and move responsibility groups.** Keep a migration map as an internal checklist, then execute it in small groups within this run. Move grids/transforms/backends/operators and their CUDA resources toward `spatial`; core tendency implementations, runners, integration, and preset preparation toward `temporal`; archive I/O, diagnostic presentation and existing visualization toward `representation`. Keep shared physical constants, prescribed environment, and vertical-coordinate definitions distinct from tendency and rendering code. Place each shared utility by its consumers; do not force ambiguous utilities into a namespace merely to empty a directory.
2. **Keep compatibility at the old paths.** Use re-export shims preserving class/function identity. The new implementation must not depend back on its old shim. Retain all old public import paths and command entry points for the remainder of the 0.1 series; removal requires a separate explicit release decision. Keep the Sprint 1 `tropoi.support` import working. Update packaged CUDA resource locations and verify source and installed-package loading.
3. **Separate environment from resources.** Retain the existing immutable physical parameter object as the prescribed environment and the existing `Planet` facade for callers. Keep grids, transforms, and cached ProductSpaces as discretization resources shared by fields and snapshots; do not create them per field or change radius/gravity/rotation defaults. Do not broaden this into time-dependent ephemerides or a new resource framework.
4. **Verify after each group and finish the migration.** Check imports and targeted reload/numerical behavior after each group, repair failures, and proceed. Keep CLI package initializers light. Update the architecture map and affected documentation links after the final layout is established.

### Acceptance and commit boundaries

- Compare tiny deterministic BVE/SWE/PE outputs before and after within identical backend, quadrature, dependency and hardware environments, including one interval-scheduler case. Mechanical moves should preserve exact arrays; investigate differences before accepting tolerances. Backend parity remains a separate existing tolerance-based claim.
- Exercise old and new import paths, runtime resources, all existing executable aliases, and Sprint 2's CPU inspect/reload and GPU plot paths. Build/install the package in an isolated environment and verify CUDA files are packaged and discoverable without modifying the working runtime.
- Run the applicable complete suite and compare representative runtime/resource-construction costs against the pre-move baseline. Memory-pool observations alone do not establish preserved performance.
- Keep commits grouped by responsibility, each with working compatibility shims. Update `docs/ARCHITECTURE.md` and the saved-run guide; no separate narrative for every moved module is needed.

## Completion, stops, and deferred work

A sprint finishes with working software, relevant verification, and a compact report: concrete behavior delivered, revision/commit boundaries, tests and skips, numerical/artifact comparisons, exact output paths, and remaining limitations. Include the reason for the branch in that report so it can serve as the PR description. Counts of files, tests, or completed checkboxes are not substitutes for the acceptance evidence.

Stop the affected part if numerical preservation fails, archive meaning cannot be established, or the remaining runtime/context cannot support verification. Preserve completed work and report the exact failing check and next action. Continue independent work where possible. A continuation carries forward that evidence; it does not need a new planning round. GPU-unverified work must be reported as such and must not be called fully verified or ready to merge.

The refactor's useful release boundary is Sprint 2; its organization boundary is Sprint 3. The separate BVE cutoff/conservation investigation, universal state-support alignment, arbitrary/seeded IC facilities, temporal spectra, interpolation, restart checkpoints, ephemerides, codecs, sonification, and learned representations remain separate extensions. CirculationPortrait adoption also remains separate, preserving the single BVE/SWE portrait contract; no PE portrait or Billboard is introduced here.

### Starting a sprint

Use one prompt of this form, selecting the sprint number:

> Execute Sprint 1 of `docs/superpowers/plans/2026-09-18-palintropos-sprint-plan.md` as one autonomous implementation sprint. Treat its working agreement, scope, acceptance checks, and stop conditions as the task contract. Resolve routine details from the repository, complete the internal phases, and repair failures without returning for approval between phases. Preserve reviewable change boundaries and the session's commit permissions. Finish with the working result and evidence report; do not replace implementation with another plan.

Use the same form for Sprints 2 and 3 after their predecessor is verified. Review and repair should target concrete findings from those results rather than expand the process in advance.
