# Held–Suarez experiment driver

`python -m tropoi.run.held_suarez --help` exposes the experiment workflow.
`config.py` defines frozen numerical parameters and hashes; `model.py`
builds the dry PE model/hooks; `experiment.py` runs fixed-step semi-implicit
integration; `checkpoint.py` saves/verifies both leapfrog levels and restart
metadata; `statistics.py` accumulates block statistics; `report.py`
applies the declared verdict tiers on CPU.

The cross-cutting [experiment record](../../../../docs/held_suarez/README.md)
holds the protocol, reference configuration, criteria and completed results.
[Semi-implicit algebra](../../temporal/SEMI_IMPLICIT.md) and
[retained truncation](../../temporal/tendencies/RETAINED_TRUNCATION.md)
document the scheme. [Climate figures](../../representation/visual/HELD_SUAREZ_CLIMATE.md)
read saved states without rerunning the experiment.

Run 002 completed 1200 days and passed A/B/C under its frozen configuration;
run 001's B2 failure remains in the record. This dry relaxation benchmark
does not establish moist, radiative or general planetary-climate capability.
Checkpoint resume is separate from the read-only Simulation API.
