# Viewer 3 workspace consolidation: mathematical and architectural review and decision register

Branch `feature/viewer3-workspace-consolidation`, draft PR #89, stacked on #88 (`bb3cf33`, frozen).
Prepared 2026-11-28 for owner review. **This document records decisions; it makes none.**

## Status of this document

- The owner approved the *direction* of the consolidation on 2026-11-28:
  - the Explorer as the only flat renderer;
  - configurations preserved across quantity and archive changes;
  - explicit incompatibility with user-controlled adaptation;
  - archived series kept where their sampling or definition differs;
  - physical coordinates kept separate from display transforms.
- That approval is **not** scientific acceptance of the implementation, and it is **not** approval of the numerical rules below.
- An entry is marked **owner-ruled** only where the owner chose an option explicitly. That happened in the four questions of the implementation session: time-mean weighting, the extremum mask policy, the vertical mean, and the density gas constant.
- Every other entry is **provisional**: an implementation choice awaiting a decision.
- Neither this PR nor this register establishes completion of Operation Planet One or readiness for v0.4.0. The release criteria (PR #87) are unchanged and are not restated here.

All measurements below come from `runs/hs-T42L20-prod-002` (Held–Suarez, T42L20, 120 saved states every 10 days, flat terrain), read-only, on the branch head `e0c8376`. The probe scripts were scratch files and are not committed. Each number can be reproduced with a selection given in the text.

---

## 1. Reduction semantics

### 1.1 What a reduction acts on

A selection gives each of longitude λ, latitude φ, vertical and time one role: `x`, `y`, `fixed`, `reduce` or `none`. The provider maps each role to a finite **sample set**. The quantity is evaluated on the tensor product of those sets, one saved state at a time. A reduction acts only on the native samples of its coordinate; it never acts on display-mesh or interpolated values.

| Coordinate | Sample set under `reduce` | Under `x`/`y` | Under `fixed` |
|---|---|---|---|
| Time | saved states `first..last` inclusive | the same states | one saved state |
| Longitude (spectral) | the `nlon` equally spaced native longitudes | 2° display mesh (181 incl. closing meridian), or native longitudes if latitude is reduced | the requested point (exact synthesis), or the nearest native node when snapping (see D9) |
| Latitude (spectral) | the `nlat` Gauss–Legendre nodes | 2° display mesh, or Gauss nodes if longitude is reduced | as for longitude |
| Longitude / latitude (columnar) | archived meridians of a row / archived rows | archived columns | the nearest archived column |
| Vertical | native full levels (representation forced to `native`) | native full levels drawn at σ (or p_k, Z_k for one column at one time, D11) | one surface: native level, σ, p, Z or Z − Zₛ (D10) |

### 1.2 Time mean

For selected saved times t₀ < t₁ < … < t_{N−1} (indices `first..last`, N = last − first + 1):

$$
\overline{f}^{\,t} = \frac{\sum_{i} w_i\, f(t_i)}{\sum_i w_i},\qquad
w_i = \tfrac12 (t_{i+1}-t_{i-1})\ (0<i<N-1),\quad w_0 = t_1-t_0,\quad w_{N-1} = t_{N-1}-t_{N-2}.
$$

This is `selection.time_weights` plus `_TimeReduction`.

- **Regular cadence.** If every wᵢ is exactly equal, which includes any N ≤ 2, the code takes the plain path Σf/N. That path is bit-identical to the pre-consolidation accumulator, and it is what `test_area_weighted_mean_unchanged` and the HS tests rely on.
- **N = 1.** The mean is f(t₀) exactly, since 0 + f, divided by 1, is exact. The payload reports `cadence: "single sample"`.
- **Irregular cadence.** The cell weights apply, and the payload lists them (`weights_seconds`) with `weighting: "cell weights (irregular cadence)"`. The airless columnar archive is the only irregular case in the test corpus (`test_viewer_four_coordinates`, cell-weighted expectation).
- **The interval actually represented.** Mirrored end cells mean sample t₀ stands for [t₀ − (t₁−t₀)/2, t₀ + (t₁−t₀)/2]. So the weighted mean is the rectangle-rule mean over
  $$[\,t_0 - \tfrac12(t_1-t_0),\; t_{N-1} + \tfrac12(t_{N-1}-t_{N-2})\,],$$
  which extends **half a cell beyond both selected endpoints**. It is not the trapezoidal mean over [t₀, t_{N−1}], which would halve the end weights. The UI reports the span as "first … last saved time", so it understates the represented interval by one cell in total. **Open item D1b.**
- **Silent fallback.** If the time axis has no recorded seconds, or fewer values than the selected indices, `evaluate` uses equal weights and reports "equal (regular cadence)". That label would then be false. No archive in the corpus triggers this. **Open item D1c.**
- **Exactness of the regularity test.** "Regular" means exact floating-point equality of the cell weights (`np.all(w == w[0])`). Saved times with rounding jitter get cell weights that are numerically almost equal, so the result changes negligibly, but the label says "irregular".

### 1.3 Horizontal means

- **Longitude (spectral archives).**
  $$\overline{f}^{\lambda} = \frac1{n_\lambda}\sum_{k=0}^{n_\lambda-1} f(\lambda_k).$$
  This is exact for every zonal wavenumber |m| < n_λ, and so exact for the prognostic fields of a band-limited state.
- **Latitude (spectral archives).** With Gauss–Legendre nodes μ_j = sin φ_j and weights W_j, where Σ W_j = 2:
  $$\langle f\rangle^{\varphi} = \frac{\sum_j W_j f(\varphi_j)}{\sum_j W_j} \approx \frac12\int_{-1}^{1} f\, d\mu .$$
  This is exact for polynomials in μ of degree ≤ 2n_φ − 1.
- **Both together** give the area mean
  $$\frac{1}{4\pi}\iint f \cos\varphi\, d\varphi\, d\lambda.$$
  The engine forms it as the latitude mean of the longitude mean. That equals the two-dimensional quadrature, because the weights are separable (product weights W_j/n_λ).
- **Exactness is lost for nonlinear quantities.** |V|, ρ = p/(RT), and anything formed after an extremum are not band-limited at the state truncation. For them the quadrature is an approximation, not an identity. Its error has not been measured. **Open item D6b.**
- **Columnar archives.**
  - Rows are weighted by the recorded column area weights.
  - The longitude mean is the plain mean of a row, since cells in a row have equal area.
  - If the recorded weights vary along a row, the latitude mean is refused, because a separable rule is undefined.
  - An archive with no recorded weights offers no horizontal reduction at all.

### 1.4 Vertical reductions

- **Mean:** refused. The reason is stated as "a vertical mean needs a mass (Δp) weighting rule, which this release does not define". **Owner-ruled (D3).**
- **Min / Max:** the extremum over the native full levels of each column, unweighted. The representation is forced to `native` whatever the user had shown. The result is the extremum of the level samples, not of the continuous profile and not of a constant-p or constant-Z family.

### 1.5 Minimum and maximum

Over a coordinate with samples x₁…x_n:

$$
\min{}_x f = \min\{\,f(x_i) : f(x_i)\ \text{valid}\,\},\qquad \text{masked iff every } f(x_i) \text{ is masked.}
$$

- No weights are used.
- Time extrema are accumulated with `np.fmin`/`np.fmax`, which ignore NaN unless both operands are NaN. Spatial extrema use the fill-with-±∞ form.
- **Owner-ruled (D2):** valid samples only.

### 1.6 Masks

**Where masks come from.**
- A fixed p, Z or Z − Zₛ surface outside the sampled column, above the top full level or below the lowest. These are never extrapolated.
- If no column in a state has a valid node, the request fails as a whole. Measured: p = 1000 hPa on HS raises `SurfaceUnavailableError`, which states the valid range: pₛ spans 951.2–1021.6 hPa and the lowest full level reaches at most 995.99 hPa.
- Future terrain-intersecting surfaces will add more masks.

**Rules.**
- **Mean:** any masked sample makes the mean masked. No mean is formed over a subset, so its normalisation never changes silently.
- **Min/Max:** valid samples only (1.5). The payload adds `masked_excluded`.

**Consequence: the two operators treat partial domains asymmetrically.** Measured at p = 970 hPa, state 119:

| Selection | Result |
|---|---|
| Lon × lat map | 3 827 of 16 471 display cells masked |
| Zonal **mean** × latitude | masked at **31 of 64** Gauss latitudes |
| Zonal **max** × latitude | masked at **0** latitudes; `masked_excluded = 1937` |

At those 31 latitudes the "zonal maximum" is the maximum over only the longitudes where 970 hPa lies above the lowest level. It is a different domain at each latitude. The payload and the notes state the count. No figure marks *which* rows are partial-domain results. **Open item D2c.**

**`masked_excluded` is counted inconsistently. Open item D2b.**
- A spatial reduction counts raw masked samples, summed over the saved states read.
- A time extremum counts NaN cells of the already spatially reduced values, so after a spatial mean it counts masked *means*, not masked samples.
- The number is therefore not one well-defined quantity when spatial and time reductions combine.

### 1.7 Order of composition

The engine applies reductions in one fixed order: longitude, latitude, vertical (within each saved state), then time. Time comes last because states are streamed one at a time, so memory stays bounded.

**When order does not matter (exact in exact arithmetic):**
- **mean∘mean:** the weights are a product measure. Under the all-or-nothing mask rule both orders give the same mask, so the means commute up to floating-point rounding.
- **min∘min and max∘max:** commute; extrema over the union of the sample sets are associative.
- **Fixed and axis roles:** commute with everything, because they select samples and do not combine them.

**When order matters:**
- **mean and extremum:** for any f,
  $$\max_t\,\overline{f}^{\lambda}(t) \;\le\; \overline{\max_t f}^{\,\lambda},$$
  with equality only when the maximising time is the same at every longitude. The same holds for min with ≥.
- **min and max:**
  $$\max_x \min_y f \le \min_y \max_x f.$$
- **Measured:** σ = 0.9, states 100–119. The engine's "longitude mean, time max" × latitude peaks at 303.52 K. The reverse order (time max, then zonal mean), formed by hand from the engine's time-max map on the 2° display mesh, peaks at 304.18 K. The difference is 0.66 K. The two use different latitude sets (Gauss nodes against 2° mesh), so the number only illustrates the inequality. The inequality itself is exact.
- **Not expressible today:** "zonal mean of the time maximum" and other time-first compositions. Time-first evaluation would also fit in bounded memory: accumulate the time statistic per (level, latitude, longitude) cell, then reduce spatially, one state's grid at a time. The fixed order is a design choice, not a memory necessity. **Open item D14.**
- **Derived-after-reduction quantities follow their own definition.** ψ is formed from the time mean of [pₛ v] (`post`), so ψ offers only means, and its capability text says why.

### 1.8 Does the interface communicate the operation unambiguously?

**Mostly, with one verified defect.**

What works:
- The description string lists every reduction in engine order, for example `Latitude · longitude mean · time max`.
- When operators differ, a note states the order and that it matters.
- Each reduced coordinate's rule text is shown, including any `masked_excluded` count.
- Time reductions show count, span, cadence and weighting.
- Level positions (p, Z) under a reduction are labelled "(mean)".

**Defect F-1 (verified, introduced in `0f58a55`):**
- Under a time **Min/Max** profile, the profile table's T, u and v columns are labelled "(time min)" / "(time max)" but hold **time means**.
- Measured: profile at (10°E, 30°S), states 60–63. The table's T column equals the time-mean profile exactly, and differs from the plotted minimum by up to **5.32 K**.
- Cause: in `viewer_atmosphere._vertical_plan`, the T/u/v companions are not flagged `statistic="quantity"`, so they take the mean.
- The existing test checks only the header text.
- The plotted curve and the lowest-level marker are correct.
- **Fixed in `ee745b3`** (approved by the owner on 2026-11-28): T, u and v now take the quantity's statistic. p and Z stay level-position means. The test now asserts the tabulated T equals the plotted minimum, that u is the minimum over the saved states, and that p is unchanged from the mean profile.

Smaller ambiguities:
- The span text understates the time interval (D1b).
- Partial-domain extrema are not visibly marked (D2c).
- When snapping is on, the "requested" and "used" points differ silently in the figure title; the coordinate text states them (D9).

---

## 2. Derived quantities

### 2.1 Why density availability depends on other level fields, and why that is not a sound rule

Density ρ = p/(R T) is offered when all of these hold:
- the archive records `physics.r_dry` or `physics.gas_constant`;
- it stores T on levels and ln pₛ;
- it stores **no level field other than ζ, δ, T** (`archive.viewer_manifest.DRY_PROGNOSTICS`).

The last condition was meant as a proxy. With moisture or a variable composition, the mixture gas constant is R_m = R_d(1 + (R_v/R_d − 1) q + …), so one recorded R would be wrong.

The owner is right that the proxy is unsound. It fails in both directions:
- **False refusal.** An unfamiliar level field, for example a passive dye, an age-of-air tracer or a stored diagnostic, says nothing about the equation of state.
- **False acceptance.** A moist run that stored humidity outside the level fields, or not at all, would pass the check, yet one R would be invalid.

It is a quantity-specific heuristic standing in for a physical assumption the archive never states. It should be replaced by the contract below. Until then it stays as implemented, and it does not affect any archive in the corpus: all HS, Mars and airless archives store only the dry prognostics.

### 2.2 Proposed contract (for review; not implemented)

A derived quantity is declared once, in the quantity registry beside its presentation, as data:

| Member | Meaning | Density example |
|---|---|---|
| `expression` | the definition, evaluated pointwise | ρ = p / (R T) |
| `dependencies.fields` | stored or derived fields, with the coordinates each must share | `pressure` (derived), `temperature` (stored) on the same samples |
| `dependencies.constants` | recorded physics constants, with units | `physics.gas_constant` [J kg⁻¹ K⁻¹] |
| `assumptions` | named physical statements the archive must **assert**, not merely fail to contradict | `equation_of_state = ideal_gas`; `gas_constant_applies_to = whole_mixture` |
| `units` | result units, checked against the dependency units | kg m⁻³ |
| `evaluation` | where the expression is evaluated: native samples, then interpolated, or inputs interpolated first (2.3) | inputs interpolated to the surface, then evaluated |
| `reduction` | `pointwise_then_reduce` (default) or `reduce_then_derive` with the reduced inputs named | pointwise_then_reduce |
| `available_when` | generated: every dependency present, every assumption asserted, units consistent | — |
| `refusal` | generated: names the first missing dependency or unasserted assumption | "this archive does not assert that its gas constant applies to the whole mixture" |

**Where assertions come from.**
- In order of preference:
  1. an explicit archive metadata block written by the run;
  2. for legacy archives, a rule keyed on the solver's **declared state schema**, not on the set of fields found. A dry PE solver has no composition tracer by construction, so its schema implies a fixed composition.
- An archive with neither is refused, with the reason.
- This keeps the "recorded R only" ruling (D5): the contract decides *whether* the recorded R applies, never substitutes another value.

**Generality.** The same declaration covers every existing derived field:
- pressure p = σ pₛ;
- geopotential (hydrostatic, needs g and R);
- Z and Z − Zₛ;
- |V|;
- ψ, which is `reduce_then_derive` over the longitude mean and time mean of [pₛ v].

No quantity would need special-case code for availability.

**Decision requested (D5b):** adopt the contract and the legacy rule, or keep the current guard until a moist archive exists.

### 2.3 Deriving before or after a reduction, and before or after an interpolation

**Reductions.**
- The engine evaluates density per saved state at each native sample, then reduces. The time mean is the mean of ρ, not ρ of the means:
  $$\overline{\rho} = \frac1R\,\overline{\left(\frac{p}{T}\right)} \;\neq\; \frac{\overline{p}}{R\,\overline{T}}.$$
- To second order,
  $$\overline{\rho} \approx \frac{\overline p}{R\overline T}\left(1 + \frac{\overline{T'^2}}{\overline T^{2}} - \frac{\overline{p'T'}}{\overline p\,\overline T}\right).$$
- **Measured:** native level 10, (0°E, 45°N), all 120 states.

  | Quantity | Value |
  |---|---|
  | Mean ρ | 0.7316434 kg m⁻³ |
  | ρ(p̄, T̄) | 0.7311247 kg m⁻³ |
  | Relative difference | 7.1 × 10⁻⁴ |

  The engine's time-mean density equals the hand-formed mean of ρ to the last digit.
- Extrema are not derivable from reduced inputs at all. In general min ρ ≠ min p/(R max T), because the extremes need not occur at the same sample.

**Interpolation to a fixed surface.**
- The engine interpolates the **inputs** to the surface and then evaluates the expression.
  - On a constant-p surface it uses ρ = p₀/(R T̂), where T̂ is T interpolated linearly in ln σ.
  - On σ or Z surfaces it uses the interpolated p̂ and T̂.
- Interpolating ρ itself would give a different value, because ρ is not linear in ln σ when T is.
- This choice is currently implicit; the contract's `evaluation` member would make it explicit. **Open item D10b.**

The current behaviour, pointwise and then reduced, is the physically conventional one for statistics of a field. Nothing in the engine computes a derived quantity from reduced inputs, except ψ, whose definition requires it.

---

## 3. Decision register

Status values:
- **owner-ruled:** chosen explicitly by the owner, with the date.
- **direction approved:** approved in principle on 2026-11-28; details still open.
- **provisional:** implemented, not approved.
- **defect:** verified wrong behaviour.

| ID | Topic | Status |
|---|---|---|
| D1 | Time-mean weighting | owner-ruled |
| D1b | Represented time interval | provisional |
| D1c | Missing time values fall back silently | provisional |
| D2 | Extremum mask policy | owner-ruled |
| D2b | `masked_excluded` definition | provisional |
| D2c | Partial-domain extrema | provisional |
| D3 | Vertical mean | owner-ruled (refused) |
| D4 | Vertical extremum on native levels | needs confirmation |
| D5 | Density gas constant | owner-ruled |
| D5b | Density availability contract | provisional |
| D6 | Latitude weighting | provisional |
| D6b | Quadrature accuracy for nonlinear quantities | provisional |
| D7 | Longitude mean | provisional |
| D8 | Horizontal sample sets for axes | provisional |
| D9 | Horizontal snapping | provisional |
| D10 | Fixed p and Z surfaces | provisional |
| D10b | Derived quantities on surfaces | provisional |
| D11 | Per-sample coordinates and reduced positions | provisional |
| D12 | Display transforms | direction approved |
| D13 | Profile-table statistics | defect F-1, fixed in `ee745b3` |
| D14 | Reduction order | provisional |
| D15 | Archived series | direction approved |

### D1. Time-mean weighting: owner-ruled (cell weights, 2026-11-28)
- **Current behaviour:** §1.2.
- **Justification:**
  - The weights are the measure each sample represents, so the result does not depend on the cadence.
  - A regular cadence gives exactly the plain 1/N mean.
- **Alternatives:**
  - Sample mean 1/N always: biased towards densely saved periods.
  - Trapezoid on [t_first, t_last]: the integral mean over exactly the selected span, but a regular cadence would no longer reproduce 1/N, which breaks bit-identity with existing results.
- **Evidence:** `test_time_cell_weights`, `test_irregular_time_mean_is_cell_weighted`, `test_selected_interval_only`, the airless cell-weighted expectation, and the HS equal-weight path in `test_spatially_averaged_profile`.

### D1b. Represented time interval: provisional
- **Current behaviour:** the mean represents [t_first − ½Δ_first, t_last + ½Δ_last]. The UI says "first … last".
- **Alternatives:**
  - (a) State the represented interval in the payload and UI.
  - (b) Clip the end cells to the selected span. This equals the trapezoid and breaks 1/N.
- **Recommendation:** (a). It changes text, not numbers.
- **Evidence:** analysis only.

### D1c. Missing time values fall back silently: provisional
- **Current behaviour:** without recorded seconds, the mean uses equal weights and is labelled "regular cadence".
- **Recommendation:** label it "equal weights (sample times not recorded)", or refuse.
- **Evidence:** code reading. No archive in the corpus triggers it.

### D2. Extremum mask policy: owner-ruled (valid samples only, 2026-11-28)
- **Current behaviour:** §1.5–1.6.
- **Evidence:**
  - `test_time_extrema_exclude_masked_samples`
  - `test_time_extremum_is_masked_where_every_sample_is`
  - `test_time_mean_stays_masked_if_any_sample_is`
  - `test_spatial_extrema_and_order`
  - the 970 hPa measurement in §1.6

### D2b. Definition of `masked_excluded`: provisional
- **Current behaviour:** raw masked samples for spatial reductions; masked reduced cells for time (§1.6).
- **Alternatives:**
  - (a) Count raw (state, level, lat, lon) samples for every coordinate.
  - (b) Report, per output cell, the fraction of its domain that was valid.
- **Recommendation:** (a) for the number. Consider (b) as a companion field if D2c is pursued.

### D2c. Partial-domain extrema: provisional
- **Current behaviour:** a row of a zonal-max section can come from a subset of longitudes, with nothing visible in the figure (31 of 64 latitudes at 970 hPa).
- **Alternatives:**
  - (a) Accept as ruled.
  - (b) Return a per-cell valid-fraction companion and hatch the partial cells.
  - (c) Require a minimum valid fraction.
- **Recommendation:** (b). It keeps the ruling and makes it visible.

### D3. Vertical mean: owner-ruled (refused, 2026-11-28)
- **Background for a future decision:**
  - A mass-weighted column mean is Σ_k f_k Δp_k / Σ_k Δp_k = Σ_k f_k Δσ_k per column, because Δp_k = pₛ Δσ_k and Σ Δσ = 1.
  - A *global* mass-weighted mean then weights columns by pₛ.
  - The archived `mean_T` is the area mean of Σ Δσ_k T_k. That is a σ-weighted column mean, area-weighted without the pₛ factor. It therefore differs from the global mass-weighted mean by the pₛ–T correlation. It was verified equal to the engine's area-mean profile weighted by Δσ (`test_archived_mean_T_is_a_sigma_weighted_vertical_mean`).
- **Recommendation when reopened:** define the vertical mean with Δp weights and state the column-versus-global distinction explicitly.

### D4. Vertical extremum on native levels: needs confirmation
- **Current behaviour:** allowed, unweighted, over native full levels.
- The implementation session treated it as part of the D3 answer. **The owner should confirm it was intended as a ruling.**
- **Consequence:** the extremum is over the level samples. A local extremum between levels, such as a jet core between two σ levels, is underestimated.
- **Evidence:**
  - `test_vertical_extrema_and_refused_vertical_mean`
  - `test_maximum_wind_speed_over_an_interval_and_the_vertical`: the combined max equals the max over 20 per-level maps to 1e−13.
  - `test_archived_extrema_equal_the_general_reduction_at_saved_states`: `t_max`/`t_min` reproduced to 1e−9.

### D5. Density gas constant: owner-ruled (recorded R only, 2026-11-28)
- **Current behaviour:** §2.1.
- The ruling covers *which constant* is used. The tracer guard was an implementation addition and is covered by D5b.
- **Evidence:** `test_viewer_density.py`.

### D5b. Density availability: provisional
- **Recommendation:** replace the guard with the contract in §2.2. Decide the legacy rule: infer from the state schema, or refuse.

### D6. Latitude weighting: provisional
- **Current behaviour:** Gauss–Legendre weights on spectral archives; recorded area weights on columnar archives; refused otherwise.
- **Justification:** exact quadrature of band-limited fields, and consistent with the existing zonal-mean and archived `mean_ps` computations.
- **Alternatives:**
  - cos φ_j at the nodes: not exact, and inconsistent with the archived series;
  - band areas of the display mesh: wrong sample set.
- **Recommendation:** keep.
- **Evidence:**
  - `test_area_weighted_mean_unchanged`
  - `test_archived_mean_ps_equals_the_area_mean_at_saved_states`
  - the zonal-mean equivalence gate inherited from #88

### D6b. Quadrature accuracy for nonlinear quantities: provisional
- **Current behaviour:** the same rule applies to |V|, ρ and extrema-derived fields. It is no longer exact for them.
- **Recommendation:** measure the error against a doubled grid on HS before stating a tolerance.

### D7. Longitude mean: provisional
- **Current behaviour:** plain mean over native longitudes, or over a columnar row.
- **Justification:** exact for |m| < n_λ.
- **Recommendation:** keep.
- **Evidence:** the inherited zonal-mean gate.

### D8. Horizontal sample sets for axes: provisional
- **Current behaviour:**
  - With no horizontal reduction, the axes use the 2° display mesh with exact synthesis.
  - With one horizontal coordinate reduced, the other axis uses native nodes (the zonal-mean rule).
- **Consequence:** a time-max *map* and a zonal-max *line* use different longitude sets. Extrema over them can differ.
- **Alternatives:** native nodes for every axis, which gives blockier maps but one sample set everywhere.
- **Recommendation:** keep, and state the sample set in the figure caption. It already appears in the coordinate text.

### D9. Horizontal snapping: provisional
- **Current behaviour:**
  - **On by default** on Gaussian archives: a fixed λ and φ move independently to the nearest native node.
  - The payload records the requested and used values.
  - Columnar archives always use the nearest archived column.
- **Measured:**

  | Setting | Latitude used | Lowest-level T, state 119 |
  |---|---|---|
  | Snap on, request 45° | 46.0447° | 267.008 K |
  | Snap off, exact synthesis at 45° | 45° | 267.536 K |
  | Difference | | 0.53 K |

- **Justification for the current default:** values at a node equal grid-point diagnostics.
- **Argument against it:** spectral archives can be synthesised exactly at any point, so snapping changes the answer without a numerical reason.
- **Alternatives:**
  - (a) Default off for spectral archives, kept for columnar.
  - (b) Keep it on, and show "requested → used" in the figure title.
- **Recommendation:** (a).

### D10. Fixed p and Z surfaces: provisional
- **Current behaviour:**
  - Values are interpolated linearly in ln σ between bracketing native full levels at their hydrostatic positions.
  - Z surfaces are found by inverting Φ, piecewise linear in ln σ, with target Φ₀ = g(h + Zₛ) for Z − Zₛ.
  - Anything outside the sampled column is masked. There is no extrapolation.
  - A surface with no valid node anywhere is refused with its range.
- **Alternatives:**
  - extrapolation below the lowest level (standard-atmosphere or lapse-rate rules): fills masks but invents data;
  - interpolation linear in p, or cubic.
- **Recommendation:** keep no-extrapolation. It is consistent with "never fabricate". Its masks feed D2/D2c.
- **Evidence:** the inherited #24 / #88 pressure-surface tests, and the 1000 / 970 hPa measurements above.

### D10b. Derived quantities on surfaces: provisional
- **Current behaviour:** inputs are interpolated, then the expression is evaluated (§2.3).
- **Recommendation:** declare this through the contract's `evaluation` member.

### D11. Per-sample coordinates and reduced positions: provisional
- **Current behaviour:**
  - p_k and Z_k can be an axis only for one column at one time (`PER_SAMPLE`).
  - Under reductions, rows are native σ levels, and their p/Z companions are the **means** of the level positions, labelled "(mean)". The payload note says these are not constant-p/Z surfaces.
- **Alternatives:** remap each sample to a common p or Z grid, then reduce. That needs D10 interpolation and inherits D2 masks, but gives true isobaric statistics.
- **Recommendation:** keep for this release, and offer remapped statistics as a separate, explicitly labelled representation later.

### D12. Display transforms: direction approved (2026-11-28)
- **Current behaviour:**
  - The physical coordinate, interpolation space, display scale (log or linear) and orientation are independent members of an axis.
  - σ and p default to log scale, increasing downward.
  - Linear σ is drawn increasing downward. Verified in Chrome, screenshot 05.
- **Open:** none beyond the defaults themselves.

### D13. Profile-table statistics: defect F-1, fixed in `ee745b3`
- See §1.8.
- **Fix:** the T, u and v companions are flagged `statistic="quantity"`, and `test_minimum_profile_labels_its_statistic_and_keeps_position_means` now asserts the values, not only the header.

### D14. Reduction order: provisional
- **Current behaviour:** fixed order: longitude → latitude → vertical → time.
- **Alternatives:** a user-selectable order, or a time-first option. Both fit in bounded memory (§1.7).
- **Recommendation:** keep the fixed order for now. Decide later whether time-first compositions such as "zonal mean of the time maximum" are wanted. If so, expose the order as an explicit, displayed member of the selection.

### D15. Archived series: direction approved (2026-11-28)
- `t_max`, `t_min` and `mean_ps` match the engine at saved states.
- `mean_T` is a distinct σ-weighted diagnostic.
- All of them are daily, while saved states are 10 days apart.
- All are kept.
- **Evidence:** the audit tests in `test_viewer_reductions.py`.

---

## 4. Verification status

### 4.1 Evidence preserved

Nothing from #88 was removed or weakened. That includes the equivalence gate, the pressure-surface tests and the zonal-mean accumulator gate. New evidence:
- `tests/test_viewer_reductions.py`: synthetic irregular-cadence provider (weights, masks, order, companions); HS archive checks; series audits.
- `tests/test_viewer_density.py`
- Browser tests updated for the single-renderer page: `test_viewer_explorer_browser.py`, `test_viewer_local_browser.py`
- Chrome screenshots 01–06 in this directory.

### 4.2 Results at `e0c8376`

| Environment | Result |
|---|---|
| Windows Python 3.12.12 (viewer suite) | pass |
| Windows Chrome via CDP (browser suite) | 8 / 8 pass; airless browser test deselected |
| WSL Python (viewer suite, incl. airless native library) | 112 passed, 5 skipped (Ben-2 archives not present in WSL) |
| Architecture snapshot (WSL `arch312`) | regenerated; `--check` up to date (regenerated again at `ee745b3` for the F-1 edit) |
| Real Chrome, manual | the flows of screenshots 01–06, plus search, Min profile, PNG export |

### 4.3 Not verified, and why

- **Airless browser path: never run.**
  - Windows cannot load the airless native library, which needs WSL.
  - WSL has no Chromium.
  - No single host has both. The airless Python tests pass in WSL; the airless *page* has not been exercised anywhere.
- **Transient "Failed to fetch".**
  - Seen once in real Chrome during a burst of eight rapid configuration changes against the local server. It did not reproduce on retry.
  - Not diagnosed. Possible causes include request cancellation on rapid re-selection, or the single-threaded server refusing a connection under load. Neither has been tested.
- **Phone width in real Chrome:** the maximised window would not resize. Only the 390 px CDP test covers it.
- **F-1:** found during this review and fixed in `ee745b3`. The browser was not re-checked after the fix; only the Python suites were.
- **Earlier WSL runs:** the WSL worktree had silently stayed at `a2abc33`, because leftover snapshot files blocked its checkout. The Python source at `a2abc33` and `e0c8376` is byte-identical, which the architecture content hashes confirm, so those results stand for `e0c8376`. The run at `ee745b3` was on the correct commit: 112 passed, 5 skipped.
- **D6b quadrature accuracy for nonlinear quantities:** not measured.
- **Environment limits:**
  - Windows and WSL worktrees are separate checkouts of the same commit; the WSL tree fetches from the main clone.
  - The 8777 server reads assets once at start.
  - GPU paths are not exercised; everything runs under `TROPOI_EXECUTION=cpu-verification`.
  - The Ben-2 acceptance tests skip in WSL because the archives are absent there.

### 4.4 Draft PR content

PR #89 against `feature/viewer3-four-coordinate-engine`.
- Before this document: 5 commits and 39 files. All of them are:
  - viewer modules under `src/tropoi/representation/visual/` (Python and assets);
  - `archive/viewer_manifest.py`;
  - tests;
  - the regenerated `docs/architecture/current/` (14 files);
  - six evidence screenshots.
- No other package, run archive or release file is touched.
- This register adds one Markdown file. The architecture generator graphs only Python, so the snapshot stays current.
- #88 (`bb3cf33`) is unchanged, locally and on origin.

### 4.5 Remaining verification work (proposed, not started)

1. Re-check the time-Min profile table in real Chrome after the F-1 fix.
2. Find a host with both the native library and a browser for the airless page. Options: a WSL Chromium install, or loading the airless data through a Windows server backed by WSL-computed results.
3. Reproduce or rule out the transient fetch failure with a scripted burst test.
4. Test phone width in real Chrome on a resizable window or a device.
5. Measure D6b quadrature errors.
6. Re-run the full suites after any decision from this register is implemented.

## 5. Scope

This PR consolidates the viewer workspace. It does not:
- change the solver, physics or run archives;
- establish completion of Operation Planet One;
- establish readiness for v0.4.0;
- change any release criterion.

The next implementation phase will not begin until the owner has reviewed this register.
