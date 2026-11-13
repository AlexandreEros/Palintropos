# Radiative validation evidence

The [radiation report](../README.md) gives the methods, interpretation, and
limits. The [test record](../TEST_RESULTS.md) covers automated regressions.
For the full archived inputs and outputs, see the [run catalogue](../../runs/README.md)
and the [preservation record](../../validation/run_archive_2026-11-12/README.md).

- [adiabatic_energy.json](adiabatic_energy.json), [energy_attribution.json](energy_attribution.json),
  and [regimes.json](regimes.json) hold the budget and regime measurements.
- [Earth without frictional heating](earth-gray-dry-T21L25-300d/README.md)
  and [Jupiter without frictional heating](jupiter-sl09-T21L30-200d/README.md)
  are the completed baseline integrations. The
  [partial Earth run](earth-gray-dry-T21L25-partial/README.md) records an earlier stop.
- [Heated Earth on MX110](earth-gray-dry-heated-T21L25-300d/README.md),
  [heated Earth on Colab](earth-gray-dry-heated-T21L25-300d-colab/README.md),
  and [heated Jupiter on Colab](jupiter-sl09-heated-T21L30-200d-colab/README.md)
  preserve summaries for the frictional-heating runs.
- [time_stepping_residual.json](time_stepping_residual.json) splits the adiabatic
  time-stepping residual.
- [R1, heated preset on an L4](earth-gray-dry-heated-T21L25-300d-d376c67-l4/README.md)
  and [R2, unheated on the same L4](earth-gray-dry-T21L25-300d-d376c67-l4/README.md)
  are the controlled pair of 2026-11-13, judged by criteria fixed before they ran.
- [Flat-Mars pilot H](dev-mars-gray-flat-heated-T21L25-30d-b3aaef8-l4/README.md) and
  [pilot U](dev-mars-gray-flat-T21L25-30d-b3aaef8-l4/README.md): the 30-day gray
  development experiment toward Isca M-I, judged by M1-M6 (not an M-I reproduction).
- [Sponge rerun Hs](dev-mars-gray-flat-sponge-heated-T21L25-30d-e38890e-l4/README.md) and
  [Us](dev-mars-gray-flat-sponge-T21L25-30d-e38890e-l4/README.md): the same pilot with
  M-I's top sponge, judged by M1-M6 against predictions fixed before the code.
- [Implicit-exchange run Hi](dev-mars-gray-flat-sponge-implicit-heated-T21L25-60d-63d05a2-t4/README.md)
  and [Ui](dev-mars-gray-flat-sponge-implicit-T21L25-60d-63d05a2-t4/README.md): the sponge
  rerun with a backward-Euler surface exchange, 60 days on a T4. N1 and M2-M4 met; M1, M5
  and M6 not met (M1 and M6 against the prediction).
