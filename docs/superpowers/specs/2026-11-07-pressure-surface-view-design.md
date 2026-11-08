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
