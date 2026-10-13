# PE dealiasing audit and the retained-truncation option (2026-09-27)

Question: the PE core with stored `l_max = 42` evolves only degrees ≤ 28, although nonlinear
products are evaluated on the Gauss "fine" (3/2-rule) product grid. Is the additional 2/3 cut
necessary, or does it combine an enlarged product quadrature with an unintended reduction of the
retained spectrum?

## Verdict

- **Not necessary for dealiasing (proven).** On the 3/2-rule Gauss grid every quadratic PE term
  is analyzed exactly at every degree ≤ L; the only aliasing terms are the cubic ones through
  V·∇ln p_s, at the 1e−6 level for an HS-like spectrum.
- **Load-bearing for another reason (proven).** `SpectralOperators.sin_theta_d_theta_coeffs` stores
  degrees ≤ l_max only (`C+[l_max] = 0`), so sinθ∂θ of degree-l_max content loses its degree-(l_max+1)
  part. A state with content AT l_max therefore gets an O(ε) spurious ζ/δ tendency on every degree,
  which no cut removes. The 2/3 cut avoided this only by never letting the state reach l_max, at the
  cost of a third of the band.
- **Repair (implemented, opt-in):** store one extra degree and retain `L = l_max − 1`
  (`PrimitiveEquationsModel(retained_truncation=L)`); degree l_max becomes a pure derivative
  buffer, the same spectral layout as Dinosaur (`Grid.T42()`: longitude wavenumbers 43, total
  wavenumbers 44). No operator changes. The default (None) is the historical 2/3 cut, bitwise.

## Trace (scalar and batched paths identical in policy)

| item | value |
|---|---|
| storage | triangular 0 ≤ m ≤ l ≤ l_max, complex orthonormal SH |
| state grid | Gauss nlat × uniform nlon (64 × 128 for T42), exact analysis for nlat ≥ l_max+1, nlon ≥ 2 l_max+1 |
| product grid ("fine") | `max(⌊3 l_max/2⌋+1, nlat) × max(3 l_max+1, nlon)`: 64×128 at l_max 42, **65×130 at l_max 43** |
| weights | exact Gauss × uniform longitude (solid angle) |
| meridional derivative | sinθ∂θ via C±; degree-(l_max+1) part dropped (`C+[l_max] = 0`) |
| vector weak form | analysis extended to l_max+1 (Bourke construction), exact |
| linear terms | −∇²(Φ + E) with Φ from spectral T: exact, never cut |
| cut | once per analyzed product at `retained_truncation` (default ⌊2 l_max/3⌋); ζ/δ l = 0 rows zeroed |
| dissipation | none in the core; ∇⁸ is a separate stepper hook (kept) |

## Quadrature requirements

A spin-0 product of total degree D analyzed against Y_l is exact on N Gauss latitudes when
D + l ≤ 2N − 1 (and nlon > D + l). With the 1/cosφ metric factors, products of the band-limited
wind/gradient components are polynomial (the factors cancel exactly). For retained degree L:
quadratic terms (advection, R T ∇ln p_s, ηζ, ηδ, |V|², and every term when ln p_s is uniform) have
D ≤ 2L → need N ≥ (3L+1)/2 (the 3/2 rule); the cubic terms κT(ω/p), σ̇∂T/∂σ, σ̇∂V/∂σ contain
T·V·∇ln p_s or V·V·∇ln p_s, D ≤ 3L → need N ≥ 2L+1. The dynamics has no quartic or non-polynomial
term (ln p_s enters only through its gradient); the HS Newtonian forcing (exp, the 200 K floor) is
non-polynomial and never exactly analyzed on any grid.

## Evidence (T21 unless stated; reference = the same coefficients on a 3× overresolved Gauss grid)

| check | result |
|---|---|
| current regime (state ≤ cut), analysis at every l ≤ L, cut removed | 3e−13 (cubic terms exact too: 3c + L = 3L) |
| state at l_max, uniform ln p_s (all terms quadratic) | 3e−13 |
| state at l_max, cubic terms, white spectrum | 4.5e−3 (worst block, max-norm) |
| state at l_max, cubic terms, HS-like red spectrum | 1.2e−6 (ζ), 4e−6 (T); T42 L20: 4.9e−7 / 1.6e−6 |
| same on the 2-rule grid (N = 2L+1) | 1e−13 (cubic exactness confirmed) |
| Newtonian forcing | 1e−4 on any grid (non-polynomial) |
| rest + 1e−3 ln p_s single mode, degree 15…20 (above cut, < l_max) | \|ζ̇\| ≤ 6e−24 |
| same at degree 21 = l_max | \|ζ̇\| = 2.6e−10, spread over every degree 2…21 |
| same with the degree-(l_max+1) derivative term restored | 1.4e−24 |
| rest linearization, perturbation to l_max, fast_cut = l_max | ratios ζ, δ 2.00 (O(ε)); with the exact derivative 4.00 |
| scalar vs batched, cut removed | 3.6e−15 |
| above the cut (legacy): δ̇ from ln p_s′ is dropped while −∇²Φ(T) is kept | dormant inconsistency (T, q never reach there) |

Cause of S3's "q′ above the cut gives O(ε) vorticity": the top-degree derivative clipping, not
aliasing, not the weak-form vector operator (its random perturbations contained degree l_max).

Store L+1 / retain L with the unmodified production derivative (`tests/test_pe_retained_truncation.py`):
quadratic terms exact (3.5e−14), cubic HS-like 6.7e−8, the storage-only degree exactly 0, the top
retained degree's rest cancellation 1.2e−23, rest linearization ratios 4.00 in every block with the
SI operator's default mask, scalar vs batched 3.6e−15, terrain accepted up to L.

## Options compared

| option | correctness | evolved resolution | cost (per-level T42 L20 tendency, MX110) |
|---|---|---|---|
| store 42 / retain 28 (legacy default) | exact incl. cubic | T28 | 4.59 s, 363 MiB |
| store 63 / retain 42 | exact incl. cubic | T42 | out of memory locally (> 2 GB; 620 MB per transform matrix); ~4–5× work (estimate) |
| **store 43 / retain 42** (production preset) | quadratic exact, cubic ~1e−6 | T42 | 4.80 s (+4.6 %), 388 MiB (+7 %) |

## What "matches Dinosaur T42" means here

The **retained spectral resolution** (l, m ≤ 42 evolved; one extra stored degree for derivatives)
and the **checked reference parameters** (`REFERENCE_CONFIG.json`; `config_mismatches` = [] for the
production preset). It does **not** mean identical sampling: with stored l_max = 43 the fine rule
gives a **65×130 product grid**, while the state/diagnostic grid stays 64×128 (Dinosaur uses
64×128 for everything). Product sampling, the ~1e−6 cubic aliasing it permits, the transform
implementation and the time integrator remain documented core differences.

## Contracts that follow the retained cut

SI operator mask (`SemiImplicitOperator.from_model` defaults `fast_cut` to the model's
`retained_truncation`), the Newtonian forcing's truncation (`model._trunc_cut`), the terrain
band-limit check, the RK4 startup bound (ω_max now at l = 42), the ∇⁸ reference degree (the
retained truncation), the HS perturbation-degree check, the configuration hash and checkpoint
operator signature, the report's effective truncation and B4 range, and the RAW/advection jet
bound (125.1 m/s at 900 s with ∇⁸ at 42; corrected 2026-09-28 from 105.3, STATUS.md). BVE, SWE, the PE runner/CLI and every published
capsule use the default and are unchanged.
