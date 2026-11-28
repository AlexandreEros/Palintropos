# Explorer horizontal-wind representations: evidence

Recorded 2026-11-28 on branch `feature/explorer-wind-representations`. Each PNG is the
Explorer's own plot canvas (the pixels its *Save PNG* exports), drawn in headless
Chrome by the real viewer page through its own controls:

```powershell
python -m tools.viewer.wind_evidence docs/validation/viewer3/wind-representations
```

The script writes `cases.json` alongside the images. For each case it records the archive,
the URL, the payload's coordinates, the wind settings, what was drawn (mode, ψ interval,
line, closed-loop, arrowhead and quiver counts, quiver scale) and the payload's
kinetic-energy split. No page exceptions occurred in any case. PNG output is not
bit-reproducible across machines; the numbers in `cases.json` are.

## Cases

| # | Archive | Saved time | Surface | Settings | Drawn |
| --- | --- | --- | --- | --- | --- |
| 01 | W5 T63 (`20260730T011700Z_williamson5_…_668e6c9a`), free-surface height | day 15 (index 3) | single layer | Lines: automatic, limit 1 % | ψ contours, interval 2 × 10⁷ m² s⁻¹ (automatic), 11 lines, all 11 closed across the seam; divergent share 0.0713 % |
| 02 | W5, as 01 | day 15 | single layer | Lines: ψ contours; arrows 12°, ∝ speed, coloured by speed | as 01, plus 284 arrows; key arrow 20 m s⁻¹ |
| 03 | W5, as 01 | day 15 | single layer | Lines: full-wind streamlines, density 1 | 239 evenly spaced streamlines |
| 04 | W5, as 01 | day 15 | single layer | Arrows only, 10°, equal length | 412 arrows, direction only (max 38.6 m s⁻¹) |
| 05 | Held–Suarez T42L20 (`hs-T42L20-prod-002`), air temperature | day 1200 (index 119) | native level index 4 (σ ≈ 0.225) | Lines: automatic, limit 1 % | divergent share 1.02 % > 1 %: full-wind streamlines, 313 lines |
| 06 | Held–Suarez, as 05 | day 1200 | native level index 4 | Lines: ψ contours (forced) | 21 ψ contours, all closed, interval 2 × 10⁷; the red key line states that 1.02 % of the kinetic energy (above the limit) is NOT drawn |
| 07 | Held–Suarez, as 05 | time mean of days 1010–1200 (20 saved states, equal weights) | native level index 4 | Lines: automatic, limit 1 %; arrows 10°, ∝ speed | time-mean divergent share 0.195 %: ψ contours of the time-mean rotational wind, 13 lines, all closed; 412 arrows |
| 08 | Held–Suarez, as 05 | day 1200 | pressure 500 hPa | Lines: automatic; arrows 10° | ψ unavailable on a pressure surface (reason in the key): full-wind streamlines, 303 lines; 412 arrows, key arrow 50 m s⁻¹ |

## Interactive check in the owner's Chrome

The same page was opened in the owner's desktop Chrome through the Claude in Chrome
extension. A direct navigation to the loopback viewer gets 403 from the viewer's
cross-site guard, as designed; a same-origin `location.assign` then loads it. The page
was driven through its controls, and its layout record matched the headless run:

| Case | Drawn in desktop Chrome |
| --- | --- |
| W5 day 15, automatic, arrows 12° | ψ contours, interval 2 × 10⁷, 11 lines, all closed; 284 arrows, key arrow 20 m s⁻¹ |
| Held–Suarez day 1200, level index 4, automatic | full-wind streamlines, 313 lines; divergent share 0.010193044691565258 |
| the same, ψ contours forced | 21 ψ contours, all closed, interval 2 × 10⁷ |

The extension's screenshot call timed out on that window (the renderer did not answer
`Page.captureScreenshot`), so no desktop-Chrome image is included. The PNGs above are
from headless Chrome.

## Comparison with the README's W5 figure

The figure (`docs/runs/…668e6c9a/assets/overview.png`, sidecar `overview.json`) and the
Explorer were checked against each other in `tests/test_viewer_wind_helmholtz.py`:

- **Divergent share.** Day 10's share, 7.954538109950725 × 10⁻⁴, equals bit for bit the
  figure's recorded maximum over its four maps (`vector_choice.max_divergent_kinetic_energy_fraction`).
  The figure prints this as "≤ 0.08 %".
- **rms speeds.** Each map's rotational and divergent rms speeds agree with the figure's to
  a relative 10⁻¹².
- **Interval and spacing.** The ψ interval is 2 × 10⁷ m² s⁻¹, the same rule as the figure
  (about 20 steps across the range, the smallest of 1, 2, 2.5, 5 × 10ᵏ at least
  range/20). The spacing reads "4.5° apart at 40 m s⁻¹", as in the figure.
- **"max" value.** The figure's "(max 42.4)" is taken over its four maps. On day 15 alone
  the Explorer reports the rotational wind's maximum, 40.5 m s⁻¹; on day 10 it is
  42.5 m s⁻¹.
