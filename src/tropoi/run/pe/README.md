# Dry PE command runner

`config.py` validates PE options, `execution.py` manages command execution
and `runner.py` advances the configured state with fixed-step RK4 and
stage validation. `model.py` re-exports the temporal model builder.

[GUIDE.md](GUIDE.md) covers scenarios, terrain, fixed timestep, snapshots,
diagnostics and CLI examples. The [PE formulation](../../temporal/tendencies/PRIMITIVE_EQUATIONS.md)
defines the numerical assumptions. The sibling [Held–Suarez driver](../held_suarez/README.md)
provides semi-implicit forcing/checkpointing; those features are not options
of this fixed-step command. [Saved-run views](../../representation/visual/README.md)
render its persisted output. Terrain rest tests and short perturbation
experiments do not establish general climate accuracy.
