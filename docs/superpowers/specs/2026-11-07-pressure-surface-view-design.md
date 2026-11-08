# Fixed-pressure surfaces across saved times — first slice of #13

Date: 2026-11-07. Status: implementation record for the first slice of
[issue #13](https://github.com/AlexandreEros/Palintropos/issues/13). The
issue stays open: geopotential-height surfaces, target-coordinate vorticity
and divergence, lazy decoding and any GPU path are later slices.

## Intent

Let `Simulation.view()` show one **isobaric** surface of a saved
primitive-equation run at every saved time, as the issue describes: one
derived 2-D slice per saved state, no native levels embedded, the same
offline page, the same playback. The solver is untouched; only saved-state
reconstruction and the page change.

## Goals of this slice

- `sim.view(path, surface=("pressure", 300.0), times="all")`: explicit
  coordinate name and SI unit (Pa). `levels=` keeps working unchanged.
- Numerics independent of the viewer: a reconstruction module that returns
  arrays and masks, tested on analytic columns without any HTML.
- Explicit interpolation rules, units, valid brackets and mask codes.
- Below-ground, above-top-level and below-bottom-level samples are masked,
  not zeroed, and the mask travels through triangulation, flat map, colour
  ranges, hover and streamlines.
- `times="all"` never subsamples; the payload estimate runs before any
  reconstruction; frames are processed one saved state at a time.
- Target-surface vorticity and divergence are **not exported**; the page
  says why.
- Measured costs of reconstruction versus interpolation on a representative
  archive.

## Non-goals of this slice

Geopotential-height surfaces (the API refuses them by name), any
derivative on the target surface, temporal interpolation, lazy or compressed
decoding, CUDA, changes to the native-level export.

## Interpolation rules (decided)

| item | rule |
| --- | --- |
| vertical coordinate | `ln sigma`, with the target `sigma* = p0 / p_s` at each node and saved time; `p_s = exp(ln p_s)` of the saved state. Node coordinates `ln sigma_k` are the full levels, identical at every column, so the bracket search depends on `p_s` only through `sigma*`. |
| level pressure | `p_k = sigma_k p_s`, the model's own pure-sigma convention and the pressure the native-level page shows. Simmons–Burridge `Phi_k` is attached to `sigma_k`, as `pe_levels` does; the isothermal error of that attachment is measured (see the validation record), not hidden. |
| interpolant | linear in `ln sigma` between the two bracketing full levels, for temperature, geopotential, `u` and `v`. `Z = Phi / g` after interpolation. |
| valid bracket | `sigma_1 <= sigma* <= sigma_K` (first to last full level). A target equal to a full level reproduces that level's values to round-off. |
| masks (uint8) | 0 valid; 1 below the surface (`p0 > p_s`); 2 above the top full level (`sigma* < sigma_1`); 3 between the bottom full level and the surface (`sigma_K < sigma* <= 1`). Codes 2 and 3 are "outside the sampled full-level range", distinct from 1. No extrapolation anywhere. |
| invalid geometry | a non-finite reconstruction, or a column whose `Phi_k` does not decrease downward, raises; nothing is guessed. |
| wholly unavailable | a saved time with no valid node raises `SurfaceUnavailableError` naming the time, the `p_s` range and the pressure range the full levels cover. |

## Payload

`atmosphere.version` 2 with a `surface` block (`kind`, `value`, `units`,
`label`) and a `mask` block (one byte per node per time). Fields:
temperature (K), geopotential height (m), wind speed (m/s). Pressure is not a
field (it is constant on the surface). Per-slice 16-bit quantization over
the valid nodes only; colour ranges exclude masked nodes. Streamlines are
clipped to cells whose four nodes are valid (`integrate_streamlines(valid=)`).

## Page

Same template. With a `surface` block the level slider is replaced by the
surface label, triangles touching a masked node are dropped per frame, the
flat map greys masked nodes, hover names the mask reason, and the notes say
the fields are interpolated display quantities, not solver state.

## Addendum (2026-11-08): interactive pressure, the second slice

Decided and built on the same branch (PR #24):

- **API.** `view(surface=("pressure", p0), interactive_pressure=True, times=…)`;
  `surface` is the initial isobar. `levels`/`level` refused, streamlines off.
- **Export.** Native columns per saved state (T, Φ, u, v on every full level,
  16-bit per level slice with its own range; surface pressure float32), the
  seam column stored equal to the first; shared level positions, mask
  labels, end tolerance, conservative colour ranges from the columns. One
  reconstruction per state, dropped before the next. Payload `version` 3,
  `columns` block organised by saved time; versions 1 and 2 untouched.
- **Numerics.** `pe_surfaces.pressure_columns` + `interpolate_columns`
  (= `pressure_surface`) is the reference; the page's `computeSurface`
  mirrors `interpolate_log_sigma` (σ* = p₀/pₛ, ln σ linear, end tolerance
  1e−9, masks 1/2/3, no extrapolation). `atmosphere.quantize_columns` /
  `dequantize_columns` reproduce the page's arithmetic for the error tests.
- **Precision.** Half a 16-bit quantum per level slice (Mars: 0.02 mK, 0.03
  mm/s, 0.12 m; HS T42L20: 0.4 mK, 1 mm/s, 0.01 m); float32 pₛ moves a node
  on an end level across the 1e−9 band, nothing else.
- **Page.** Log slider over the sampled range + typed value with unit;
  synchronous recompute through one `applyPressure`, rapid input coalesced
  per frame (newest wins); two-state LRU decode; empty surface with
  explanation; hover checks all four contributors; `setPressure`,
  `surfaceArrays`, `timing`, `decodedTimes`, `memory` in the page API.
- **Budget.** `estimate_column_payload_bytes` before reconstruction,
  `decoded_column_bytes` reported apart; strict `times="all"` with column
  advice (HS: 433 MB at 2° refused; 8 states fit).
- **Measured.** Mars 3.64 MB / 1.6 s / 0.9 s to interactive / 6 ms per
  pressure update; HS 8 states 29.3 MB / 1.05 s / 15 ms per update / 140 ms
  per time switch (decode-bound). Record: `docs/validation/interactive_pressure_2026-11-08/`.
- **Not done.** Browser streamlines, target-surface ζ/δ, Z surfaces,
  compression, GPU.
