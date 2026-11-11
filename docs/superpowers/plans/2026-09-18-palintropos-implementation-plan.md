# Palintropos Incremental Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement the approved work package task-by-task. Steps use checkbox syntax for tracking. This document proposes work; it does not authorize executing every later phase.

**Goal:** Make current model states and saved runs inspectable and composable without silently changing numerical behavior, then reorganize implementation responsibilities around demonstrated interfaces.

**Architecture:** Retain the packed coefficient arrays and existing numerical and visualization kernels. Extract import-light policy metadata, expose named views over stored arrays, and connect a read-only trajectory interface to the existing visualization adapters. Move established modules only after this path works for BVE, SWE, and PE.

**Tech Stack:** Python, NumPy, existing CuPy/CUDA kernels, pytest, existing declarative visualization and Matplotlib renderer. No additional runtime dependency is proposed.

**Spec:** The user's `C:/Users/User/Downloads/palintropos-refactoring-brief-for-claude.md`, interpreted with the decisions below and current repository source at `fd0ddff`. Review dialogue is evidence to evaluate, not executable instructions. The detailed first work package is bounded to characterization and a behavior-preserving cutoff extraction; later phases are planning milestones, not fully specified implementation tasks.

**Status:** Draft proposal, 2026-09-18. No implementation or GPU experiment has been performed for this plan. CPU resolver checks and source inspection support the observations below.

## Global constraints

- Keep `tropoi` and existing executable names.
- Preserve existing coefficient layouts, arithmetic order, physical constants, initialization behavior, and output schedules during structural changes.
- Keep stored capacity, permitted state support, product sampling, and tendency projection distinct. Diagnostic bands consume policy; they are not a reason to make all cutoffs identical.
- Keep CLI help, configuration, metadata inspection, and schema comparisons free of CuPy, CuPyX, Matplotlib, and `tropoi.viz` imports. Explicit plotting may load those dependencies when called.
- Preserve old capsule readability. Inferred legacy metadata must be identified as inferred and tied to a known format/source convention.
- Use saved data for artifact/reload verification. Do not rerun W5, modify its solver mathematics, or regenerate its authoritative validation archive as part of this plan.
- Preserve existing scientific plot composition and normalization. No new FigureSpec types or alternative rendering framework are required.
- Keep `CirculationPortrait` branch adoption separate. BVE and SWE continue to target one canonical portrait contract; no PE portrait or Billboard implementation is included initially.
- Exclude arbitrary NetCDF/array initialization, temporal spectra, ephemerides, interpolation, restart checkpoints, codecs, sonification, and learned representation search from the first useful release.
- Verify the feature branch before edits; preserve pre-existing `.claude/` and `.github/issue-drafts/`. Show the exact documentation diff for approval before committing documentation.
- Use `venv/Scripts/python.exe` for repository checks. CUDA tests must report skips distinctly from passes.

## Decisions established by the review

1. A mapping-style state API and packed-array storage are compatible. PE already exposes differently shaped named slices over one stack. Do not rewrite RK4 around mappings.
2. Keep the semantic distinction between state and rate. A shared array implementation does not require eliminating that distinction or adding generic arithmetic to wrappers.
3. The PE `thermal_wave` resolver accepts `lmax=2`, while its degree-2 temperature perturbation lies above the retained product degree 1. Source predicts frozen upper temperature with a constant upper divergence tendency. Characterize it before describing measured consequences.
4. SWE `gravity_wave` accepts `lmax=4,5` at resolution, with its degree-4 perturbation above the product cutoff. Those modes remain active through exact linear terms. The preset is explicitly described as a linear gravity-wave test. Requiring `lmax>=6` is a proposed support policy, not an automatic consequence of the dispersion test.
5. Upper-band SWE modal evolution can be linear while those inputs still drive nonlinear lower-band products. Do not describe the whole run as necessarily linear.
6. Audit all preset requirements. SWE Williamson-2 and Williamson-5 both write degree 2 without a corresponding minimum-capacity check in config. A universal replacement of every guard with `cut>=degree` is not justified.
7. Extract a pure cutoff helper through an import-light package chain. Existing `tropoi.numerics.__init__` eagerly loads heavy modules; putting a pure file beneath it does not solve the import problem. An initial `tropoi.physics.truncation` module fits the existing empty physics initializer without reorganizing the package.
8. The BVE diagnostic band needs explicit metadata. Its definition can be reconstructed from an exact clean source revision, but the capsule alone is not fully self-describing.
9. PE row ordering is already persisted as prose in manifest notes. Upgrade to structured schema metadata; do not describe this information as wholly absent.
10. Shared-array checks do not prove shared geometry, immutability, or snapshot independence. Verify these separately. Use NumPy checks for host views and CUDA-gated CuPy checks for device views.

## Proposed sequence and release boundaries

| Work package | Deliverable | Exit evidence |
| --- | --- | --- |
| A. Support characterization and pure helper | Per-preset support table, short boundary probes, one import-light cutoff function with compatibility re-export | CPU import gates, documented numerical observations, unchanged existing cutoff behavior |
| B. Explicit preset validation | CPU-side storage checks and separately justified support restrictions | Boundary acceptance/rejection tests, matching direct-factory behavior, default configuration/run-id comparisons |
| C. Read-only saved-run path | `Simulation -> Snapshot -> named SpectralModes -> existing plot` for all three cores | Legacy reload, real CLI inspection in a fresh interpreter, ownership tests, saved-data plot comparison |
| D. Structured provenance | Versioned state schema, normalization, units, time/support and diagnostic definitions | New-manifest and legacy-read tests; unchanged scientific arrays and existing diagnostic columns |
| E. Mechanical organization | Move established responsibilities under spatial/temporal/representation, retaining shims | Import compatibility, numeric equivalence within each fixed environment, CLI/reload/plot checks |
| F. Separate extensions | Seeded and arbitrary ICs, spectral diagnostics, portrait adoption, then other capabilities as independent proposals | Each extension has its own contract and scientific validation |

Packages A and B should remain separate reviewable changes. Package C does not depend on changing SWE support policy; it can proceed once its metadata describes current behavior honestly. Package D may be split: persist support/diagnostic definitions early when A provides them, then add field schemas alongside C. Package E is not a prerequisite for useful saved-run access.

## Work package A: detailed first implementation scope

### Task A1: Record the current support contract and boundary behavior

**Files:**

- Read: `src/tropoi/run/{bve,swe,pe}/initial_conditions.py`
- Read: `src/tropoi/run/{swe,pe}/config.py`
- Read: `src/tropoi/physics/{barotropic,shallow_water,primitive_equations}.py`
- Create: `docs/validation/preset_support_characterization.md`
- Create: `tests/test_support_characterization.py`
- Reuse patterns from: `tests/test_shallow_water.py`, `tests/test_pe_tendency.py`, and `tests/test_pe_initial_conditions.py`

**Interfaces:** Consumes existing config resolvers, preset builders, model tendencies, and `rk4_step_array`. Produces a support table and bounded characterization results, not a new supported-input policy.

- [ ] Record each preset's literal coefficient indices or construction method, minimum storage degree, advertised behavior, and terms needed to realize it. Distinguish zero-amplitude cases, terrain corrections, and BVE grid-to-modal initialization.
- [ ] Execute the following CPU matrix and record the full resolved configuration, not just cutoff arithmetic. These assertions characterize current behavior and are expected to pass before any guard change:

```python
from tropoi.run.swe.config import SWERunConfig
from tropoi.run.pe.config import PERunConfig

for lmax, expected_cut in [(4, 2), (5, 3), (6, 4)]:
    cfg = SWERunConfig.resolve({"scenario": "gravity_wave", "lmax": lmax})
    assert cfg.lmax == lmax
    assert (2 * cfg.lmax) // 3 == expected_cut

for lmax, expected_cut in [(2, 1), (3, 2)]:
    cfg = PERunConfig.resolve({"scenario": "thermal_wave", "lmax": lmax})
    assert cfg.lmax == lmax
    assert (2 * cfg.lmax) // 3 == expected_cut
```

- [ ] Add CUDA-gated tendency checks using fixed, small, flat-bottom states on the Gauss backend first. Use the existing tests' model-construction conventions. Verify the upper-band equations directly; do not use a long frequency run as a substitute.

The SWE assertion body below assumes the test has constructed `model`, its valid preset state `state`, and `cut = (2 * model.l_max) // 3`. Set rotation and hyperdiffusion to zero and use a flat bottom. Run at `lmax=4,5`; degree 4 must exceed `cut`.

```python
import cupy as cp

y = state.coeffs
dy = model.tendency(y)
high = (slice(cut + 1, None), slice(None))
cp.testing.assert_array_equal(dy[0][high], cp.zeros_like(dy[0][high]))
cp.testing.assert_allclose(
    dy[1][high], (-model.lap_eigs[:, None] * y[2])[high], rtol=1e-13, atol=0
)
cp.testing.assert_allclose(
    dy[2][high], (-model.phi0 * y[1])[high], rtol=1e-13, atol=0
)
```

- [ ] For PE at `lmax=2`, construct the existing thermal-wave preset with a small nonzero perturbation. Verify zero upper temperature/log-pressure tendencies and a nonzero upper divergence tendency. Advance two one-second RK4 steps with state validation; compare the unchanged upper temperature and linear divergence increment against the initial tendency. Record this as a short characterization, not evidence of long-run stability or general conservation failure.

```python
from tropoi.run.engine import rk4_step_array

K = model.nlev
y0 = state.coeffs.copy()
d0 = model.tendency(y0)
y1 = rk4_step_array(model.tendency, y0, 0.0, 1.0)
y2 = rk4_step_array(model.tendency, y1, 1.0, 1.0)
cp.testing.assert_array_equal(y2[2*K:3*K, 2:, :], y0[2*K:3*K, 2:, :])
cp.testing.assert_array_equal(y2[3*K, 2:, :], y0[3*K, 2:, :])
assert float(cp.max(cp.abs(d0[K:2*K, 2:, :]))) > 0
cp.testing.assert_allclose(
    y2[K:2*K, 2:, :] - y0[K:2*K, 2:, :],
    2.0 * d0[K:2*K, 2:, :], rtol=1e-12, atol=1e-20
)
```

- [ ] Check the SWE lower-band response with nonzero degree-4 divergence as well as geopotential, comparing against the same state with those upper inputs removed. Report whether retained nonlinear tendencies change; do not infer full-system linearity from the upper equations.
- [ ] Characterize the flat-bottom Williamson-2 steady-state residual at `lmax=2` and `3`, using identical prescribed physical constants. Do not run a W5 integration. Record missing Williamson storage guards from source and CPU resolution separately.
- [ ] Repeat the short boundary probes on the geodesic backend and document backend-specific residuals. Exact algebraic zero claims and quadrature-dependent residuals must have separate assertions.
- [ ] Run `venv/Scripts/python.exe -m pytest tests/test_support_characterization.py -q`. If CUDA is unavailable, report the skipped measurements and retain the support-policy decision as unmeasured; this does not block the pure extraction in A2.

**Exit:** A table distinguishes verified storage requirements, analytically inferred behavior, executed short-probe results, and recommended preset policy. No guard or solver behavior changes are included in this task.

### Task A2: Extract the existing cutoff without changing its meaning

**Files:**

- Create: `src/tropoi/physics/truncation.py`
- Modify: `src/tropoi/physics/primitive_equations.py`
- Modify: `src/tropoi/physics/shallow_water.py`
- Modify: `src/tropoi/numerics/spectral_operators.py`
- Modify: `src/tropoi/run/bve/diagnostics.py`
- Create: `tests/cli/test_truncation_imports.py`
- Create: `tests/test_truncation_contract.py`

**Interfaces:** Produces `product_truncation_cut(l_max: int) -> int` in an import-light module. Retains the existing import from `tropoi.physics.primitive_equations` as a re-export. The helper describes retained product degree only; it does not project states or validate every preset.

- [ ] Add a subprocess import test using the existing CLI probe infrastructure. It must fail before the module exists and pass without importing a forbidden heavy module afterward:

```python
from .conftest import assert_probe_passes

def test_cutoff_is_available_without_cuda_or_visualization():
    assert_probe_passes(
        "from tropoi.physics.truncation import product_truncation_cut",
        "0 if product_truncation_cut(21) == 14 else 1",
    )
```

- [ ] Add boundary-value tests expressing the observed support transitions:

```python
import pytest
from tropoi.physics.truncation import product_truncation_cut

@pytest.mark.parametrize("capacity, retained", [
    (1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (6, 4),
    (10, 6), (15, 10), (21, 14), (42, 28), (63, 42),
])
def test_current_inclusive_product_support(capacity, retained):
    assert product_truncation_cut(capacity) == retained
```

- [ ] Implement only the existing convention, preserving valid integer-input behavior:

```python
def product_truncation_cut(l_max: int) -> int:
    """Inclusive retained degree for the current analyzed-product policy.

    This is not a universal limit on state storage or exact linear terms.
    Callers validate resolution inputs at their existing boundaries.
    """
    return (2 * int(l_max)) // 3
```

- [ ] Replace the PE function definition with an import of the same name. Replace the three product cutoff formulas in `spectral_operators.py`, the SWE `_trunc_cut` formula, and the BVE diagnostic cutoff with calls to this helper. Preserve every existing slice, location of truncation, zero-mode rule, arithmetic order, and diagnostic denominator. Include the operator site using local variable `L`, not just sites spelling `self.l_max`.
- [ ] Check production sources with `rg -n 'truncation_cut|// 3' src/tropoi/physics src/tropoi/numerics/spectral_operators.py src/tropoi/run/bve/diagnostics.py`. Inspect matches; unrelated resolution formulas must remain separate.
- [ ] Run `venv/Scripts/python.exe -m pytest tests/cli tests/test_truncation_contract.py tests/test_pe_run_config.py -q`.
- [ ] Run the relevant existing GPU tests for spectral operators, SWE, PE tendencies, and SWE topography. Compare A1 probe outputs before and after extraction within each identical configuration. Run the existing historical R3 drift test only if needed for the integration gate; do not change its expected band to accommodate this mechanical extraction.
- [ ] Verify default resolved configurations and run-id hashes are unchanged, and run `git diff --check`.

**Exit:** Existing consumers share one import-light definition, with identical numerical and diagnostic behavior. No new general policy class, global mask, package move, or guard restriction is introduced.

## Work package B: decide and enforce preset contracts

Implement this as a separately reviewed fix after A1 supplies the contract table. It is not part of the behavior-preserving extraction.

- Move unavoidable storage-capacity checks into CPU configuration resolution so invalid inputs fail before model construction. Mirror the same requirement at direct IC factory boundaries.
- For PE thermal-wave behavior, decide whether the supported nonzero preset requires degree 2 within thermodynamic tendency support. The recommended normal-preset contract is `product_truncation_cut(lmax) >= 2`; preserve low-level state construction for explicit characterization. Specify zero-amplitude handling rather than accidentally changing it.
- For SWE gravity-wave behavior, choose and document either support for the existing linear-wave use case at `lmax=4,5`, or a stronger preset contract requiring nonlinear feedback (`lmax>=6`). Do not relabel the dispersion test as a nonlinear validation test.
- For Williamson-2, enforce the degree/support needed by its advertised nonlinear steady balance, using the boundary evidence. Treat Williamson-5's capacity, terrain, and benchmark fidelity separately.
- Test just below, at, and above each accepted boundary through both CPU config and direct construction. Preserve representative default configs, scientific hashes, and saved-run reading. An absence of affected in-repository tests is not proof of zero downstream impact.
- Explain newly rejected inputs as an intentional compatibility change. Do not alter old capsules or silently project their coefficients on load.

## Work package C: first useful end-to-end path

The next implementation plan should specify a small set of modules, without moving existing code:

| Proposed module | Responsibility |
| --- | --- |
| `spatial/modes.py` | Import-light immutable modal/field specifications and named host-array views |
| `temporal/simulation.py` | Trajectory and contextual snapshot interfaces over a small storage protocol |
| `representation/archive/capsule.py` | Legacy/new capsule reading, field layout interpretation, times and lazy resource construction |
| `representation/visual/snapshot.py` | Adapt snapshots to existing specs and render on explicit request |

Keep all new package initializers light. Numerical stepping must not depend on the archive or plotting implementation.

Public behavior for the first release:

```python
sim = open_simulation(run_path)
sim.times
snapshot = sim[137]                 # saved output index
snapshot.time
snapshot.state["temperature"]       # PE only; explicit schema elsewhere
snapshot.plot(output_path=path)     # lazy adapter; returns pathlib.Path
```

Use current prognostic identities: BVE `zeta`; SWE `zeta`, `delta`, `phi`; PE `zeta`, `delta`, `temperature`, `ln_ps`. Preserve SWE thickness-geopotential semantics and PE full-temperature/log-pressure meaning. Layered PE fields have a level axis; surface log-pressure does not. Named access must not pad fields or copy a stack per field.

Load host arrays with explicit ownership, preferably read-only memory mapping where supported. Validate headers and time axes cheaply; validate accessed coefficient payloads for finiteness and convention violations without scanning an entire archive merely to inspect metadata. Reading must not initialize CUDA. Derived physical maps may instantiate the existing backend lazily when explicitly requested. An empty saved-state sequence can be inspected but cannot produce a snapshot plot.

Keep schema compatibility operation-specific: basis/normalization, degree-order/reality conventions, geometry, support and vertical coordinates/units as needed. A human-readable ProductSpace label is not an authoritative identity key. Preserve state/rate semantics without implementing wrapper arithmetic.

Reuse existing scientific compositions and the existing path-returning renderer. Build specifications before rendering; render once. Do not replace map interpolation with direct synthesis on a new view grid in this package. Preserve timeline normalization when comparing frames; a single-frame plot must disclose its normalization scope.

Acceptance must include all three array layouts, nonuniform PE sigma levels, missing-solver legacy BVE capsules, malformed headers/times, empty archives, expected views and values, read-only access, snapshot independence, shared geometry/resource identity, actual `inspect` against a temporary capsule in a fresh process, and saved-data render comparison. Test CPU array sharing with NumPy; CUDA view tests require CUDA and are supplemental. Add a constrained selector grammar only for the supported snapshot/state operations; no `eval`, time interpolation, pressure-level interpolation, or arbitrary expressions.

## Work package D: persistence and diagnostic definitions

Write additive structured metadata including schema version, core/field layout, normalization/reality conventions, units, vertical coordinates, time units, prescribed environment, capacity/support policy and available initialization provenance. Preserve existing notes for human readers.

For `high_l_enstrophy_frac`, persist the retained cutoff, band inequality (`l > cut`), valid triangular modal domain, positive-order multiplicity weighting, normalization/denominator and zero-power convention, plus a diagnostic-definition version. This metadata is derived from the same helper/configuration as the measured quantity. Keep historical CSV column meanings fixed; use explicit versioning for any deliberate redefinition.

Legacy readers should infer only known conventions and expose that provenance. Unknown formats should fail clearly or remain metadata-only rather than guessing field meanings. Keep additive descriptive metadata outside the existing scientific config hash unless the actual scientific policy changes. Analysis snapshots remain distinct from restart checkpoints.

## Work package E: moves after the interfaces work

Move numerical kernels toward `spatial`, tendencies/integration toward `temporal`, and archive/diagnostic/visual adapters toward `representation`. Retain re-export shims and command behavior through a declared compatibility period. Apply small moves, with import and numeric/reload checks after each group. Split physical planet descriptions from discretization resources without changing benchmark gravity/radius defaults.

A validation gate compares old/new outputs for the same backend, quadrature, hardware and dependency environment. Backend parity is a separate tolerance-based claim. Use tiny deterministic BVE/SWE/PE cases and one interval-mode scheduler case; preserve authoritative W5 saved evidence without rerunning it. Existing full-suite GPU gates apply at integration milestones, not repeatedly to every documentation or host-only adapter edit. Measure representative runtime and resource construction as well as array sharing; a CuPy pool measurement alone is not a proof of no regression.

## Deferred capabilities and completion boundary

The initial refactor is complete when existing saved runs for all three cores can be interpreted, inspected and plotted through the common public path, with compatible commands and archives and no unplanned numerical changes. Broader support alignment, arbitrary IC preparation, seeded generators, temporal spectra, portrait adoption and other representations each receive a separate bounded plan after this milestone.

The recommended next implementation action is work package A. No issue filing, publication, merge, solver change, or benchmark rerun is implied by this draft.
