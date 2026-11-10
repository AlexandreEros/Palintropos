# Architecture comparison data

The JSON files here preserve measured before/after import-graph metrics from
the completed package-ownership cleanup. They are dated structural comparison
inputs, not current architecture, scientific validation or unresolved code
review findings. The original reform base was
`5c8d7e8539be07e3e6c1f808380510f18d84a965`; the reconciliation included
`6a03708`. Each JSON carries its own provenance.

[Current structure](../current/README.md) is regenerated from today's source.
The [handwritten architecture](../../ARCHITECTURE.md) states responsibilities
and dependency rules. Export tooling includes this directory as comparison
context; the data is retained for that reproducible export. Graph counts are
descriptive measurements, not quality scores or proof of runtime acyclicity.
