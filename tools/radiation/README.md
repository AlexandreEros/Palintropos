# Radiative experiment analysis

- [summarize_run.py](summarize_run.py) reduces a saved run into daily series,
  climate means, a figure, and a summary under the radiation evidence tree.
- [adiabatic_energy.py](adiabatic_energy.py) measures energy drift with and
  without hyperdiffusion. [energy_attribution.py](energy_attribution.py)
  examines the contributions to the energy budget.
  [time_stepping_residual.py](time_stepping_residual.py) splits what is left
  to the time integration (RAW filter, step size, startup) over three days.
- [regimes.py](regimes.py) measures idealized radiative regimes.
- For the flat-Mars experiments (`docs/radiation/MARS_MI_PLAN.md`):
  [mars_tides.py](mars_tides.py) fits the diurnal tides (M5).
  [judge_mars_pilot.py](judge_mars_pilot.py) applies M1–M6.
  [mars_state_diagnostics.py](mars_state_diagnostics.py) reads the saved
  states for where the winds peak, the surface-exchange stiffness and the
  ln p_s mass terms.
  [mars_cost_profile.py](mars_cost_profile.py) measures what the T42 CV-1
  run would cost (`docs/radiation/MARS_MI_COST.md`): s/step and its
  breakdown by operator, short time-step stability probes from a
  transferred state, and the wall-clock projection arithmetic. It changes no
  production behaviour.

The [evidence index](../../docs/radiation/evidence/README.md) links the
saved products, and the [radiation report](../../docs/radiation/README.md)
documents the experiment settings and interpretation.

## Real-gas opacity tables

- [opacity/fetch_socrates.py](opacity/fetch_socrates.py) downloads the
  ROCKE-3D SOCRATES spectral files with pinned SHA-256 into the ignored
  `opacity_cache/`. [opacity/socrates.py](opacity/socrates.py) reads their
  look-up tables, and [opacity/import_socrates.py](opacity/import_socrates.py)
  premixes them for a fixed composition into a `KTable` with a provenance
  record.
- [thai_column.py](thai_column.py) saves the interface fluxes and layer
  heating of a preset's columns, checks the heating against the flux
  convergence and compares with a reference table.
  [thai_dev_evidence.py](thai_dev_evidence.py) records a short coupled run
  and compares an interrupted run with an uninterrupted control.
  [thai_vertical_coherence.py](thai_vertical_coherence.py) reads the saved
  states (CPU) for the layer means, the alternating temperature component
  and the adjacent-layer vorticity and divergence correlations (issue #47).

See [docs/radiation/OPACITY.md](../../docs/radiation/OPACITY.md).
