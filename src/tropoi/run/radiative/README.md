# Radiative–dynamical experiments

`python -m tropoi.run.radiative {run,column,show-config}` drives the dry PE
core with column physics as a lagged stepper term. `config.py` holds the
hashed configuration and two presets (`jupiter-schneider-liu`,
`earth-gray-dry`); `model.py` builds grid, physics, stepper and initial
state; `experiment.py` runs with checkpoints, exact resume and a daily
energy record (`series.json`: `energy_rate`, `column_energy_tendency`,
`energy_residual`).

Both presets use gray radiation and state in their docstrings what is and is
not matched to the paper they follow. Neither has been run to equilibrium;
see the [status page](../../../../docs/radiation/README.md).
