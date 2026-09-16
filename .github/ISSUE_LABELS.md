# Issue labels

Palintropos tracks open numerical and scientific problems as GitHub Issues rather
than as a living `KNOWN_LIMITATIONS.md` section. Labels exist to make that
tracker filterable along a few **stable, mostly orthogonal** dimensions — not to
mirror the model's state vector or the package tree.

The label set is deliberately sparse, and
[`setup-issue-labels.ps1`](setup-issue-labels.ps1) is its source of truth.

## Namespaces

| Namespace | Answers | Labels |
|---|---|---|
| `engine:` | Which equation set / dynamical core? | `barotropic-vorticity`, `shallow-water`, `primitive-equations` |
| `backend:` | Which grid + quadrature backend? | `geodesic`, `gauss-latlon` |
| `numerics:` | Which numerical subsystem? | `spectral-transforms`, `nonlinear-products`, `time-integration` |
| `area:` | Which cross-cutting concern? | `validation`, `performance`, `reproducibility`, `visualization`, `cuda`, `cli`, `documentation` |
| `failure-mode:` | How does it fail? | `silent-wrong-result`, `unsupported-claim`, `instability`, `crash` |

GitHub's default labels (`bug`, `enhancement`, `question`, …) are kept and
describe the *kind* of item; they are a separate dimension from the ones above.

A few boundaries worth stating once:

- **`backend:` vs `numerics:`.** `backend:` says *where* a problem occurs: use
  it when the problem is specific to one backend, or when the issue is
  explicitly about comparing them (a parity issue may take both). `numerics:`
  says which mechanism is involved. A generic numerical issue needs no
  `backend:` label.
- **`area:cuda` vs `area:performance`.** `area:cuda` is for behavior that is
  specific to the GPU/toolchain (kernels, CuPy/CUDA versions, driver-TDR limits,
  device memory, the absence of a CPU path). `area:performance` is for cost,
  throughput, and scaling regardless of device.
- **`area:cli` vs `area:reproducibility`.** Flags, presets, config resolution,
  and snapshot scheduling are `area:cli`; what a run capsule records about
  itself — manifests, provenance, determinism, seeds — is
  `area:reproducibility`.
- **`area:visualization` is a subsystem, not a symptom.** A plot that misleads
  is a visualization issue; it does not by itself imply a physics or numerics
  defect.

## How to use them

- **Not every issue needs one label from every namespace.** Most issues carry
  one or two labels. A documentation fix may carry only `area:documentation`.
- **`failure-mode:` is optional and not exclusive.** Use it when *how* the thing
  fails is the interesting part. An issue can be both
  `failure-mode:silent-wrong-result` and `failure-mode:instability`.
- **There is no severity scale.** For scientific software, failure character
  carries more information than a scalar: a plausible-but-wrong field is more
  dangerous than a loud crash in one unusual configuration. The `S1`–`S4`
  ratings in [`docs/KNOWN_RISKS.md`](../docs/KNOWN_RISKS.md) are a dated audit
  artifact and are not reproduced as labels.
- **`failure-mode:unsupported-claim`** is for the case where the implementation
  may well be correct but the *claim* made about it is not backed by evidence —
  as distinct from `failure-mode:silent-wrong-result`, where the output itself
  is wrong.

## Adding labels

Create a label only when it names a **recurring kind of work that is genuinely
useful to filter for**, not because the noun exists in the codebase. In
particular, do not create one label per prognostic variable or diagnostic
(`field:vorticity`, `diagnostic:energy`, …) — the issue title states that far
better than a label can.

Postponed for now, to be revisited once there is a demonstrated need to filter
for them: `numerics:quadrature`, `numerics:vertical-discretization`, and
`area:diagnostics`.

## Labels vs commit prefixes

They are separate vocabularies and should stay that way.

- A **label** describes the problem or work item.
- A **commit prefix** (`feat`, `fix`, `refactor`, `docs`, `test`, `perf`, plus a
  scope) describes one change.

One issue labeled `engine:primitive-equations` + `numerics:time-integration` +
`area:validation` may legitimately produce `feat(pe): …`, `test(pe): …`,
`refactor(integration): …`, and `docs(validation): …`. The taxonomy can suggest
stable commit scopes, but a label never dictates a prefix.
