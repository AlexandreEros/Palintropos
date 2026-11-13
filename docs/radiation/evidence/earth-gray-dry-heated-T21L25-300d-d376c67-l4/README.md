# R1: heated dry gray Earth preset on a Colab L4

Run R1 of the controlled comparison in the [radiation report](../../README.md)
("Frictional heating becomes the preset default"). It is
`--preset earth-gray-dry` as shipped from `d376c67` (configuration `748cec4b…`),
300 days at T21L25 on an NVIDIA L4 (CuPy 14.0.1), run without resuming in
37 minutes. Its criteria were committed before it started.

See the [summary](summary.json) (per-state sinks included) and the
[climate figure](climate.png). The [preserved run capsule](../../../runs/earth-gray-dry-heated-T21L25-d376c67/)
holds the receipt for all 66 original files and the Drive index of the
states and checkpoints.
