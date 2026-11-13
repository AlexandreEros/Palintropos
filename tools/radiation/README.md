# Radiative experiment analysis

- [summarize_run.py](summarize_run.py) reduces a saved run into daily series,
  climate means, a figure, and a summary under the radiation evidence tree.
- [adiabatic_energy.py](adiabatic_energy.py) measures energy drift with and
  without hyperdiffusion. [energy_attribution.py](energy_attribution.py)
  examines the contributions to the energy budget.
  [time_stepping_residual.py](time_stepping_residual.py) splits what is left
  to the time integration (RAW filter, step size, startup) over three days.
- [regimes.py](regimes.py) measures idealized radiative regimes.

The [evidence index](../../docs/radiation/evidence/README.md) links the
saved products, and the [radiation report](../../docs/radiation/README.md)
documents the experiment settings and interpretation.
