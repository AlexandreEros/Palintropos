# Architecture metrics: pre-Sprint-3

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 66, `function+module`: 1, `function+type_checking`: 2, `module`: 140, `type_checking`: 5.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 73 | 73 |
| Directed edges \|E\| | 214 | 141 |
| Density | 0.040715 | 0.026826 |
| Weakly connected components | 2 | 5 |
| WCC sizes | 72, 1 | 66, 4, 1, 1, 1 |
| Strongly connected components | 64 | 73 |
| Nontrivial SCCs | 2 | 0 |
| Nontrivial SCC sizes | 7, 4 | - |
| Nodes in cycles | 11 | 0 |
| Fraction of nodes in cycles | 0.150685 | 0 |
| Sources (in-degree 0) | 2 | 20 |
| Sinks (out-degree 0) | 15 | 22 |
| Reciprocal pairs | 2 | 0 |
| Reciprocity | 0.018692 | 0 |
| In-degree min / median / max | 0 / 2 / 11 | 0 / 1 / 9 |
| In-degree mean | 2.931507 | 1.931507 |
| In-degree quartiles | 1, 2, 4 | 0, 1, 3 |
| In-degree zeros | 2 | 20 |
| Out-degree min / median / max | 0 / 2 / 11 | 0 / 1 / 11 |
| Out-degree mean | 2.931507 | 1.931507 |
| Out-degree quartiles | 1, 2, 5 | 0, 1, 3 |
| Out-degree zeros | 15 | 22 |
| Condensation nodes | 64 | 73 |
| Condensation edges | 170 | 141 |
| Condensation sources / sinks | 3 / 15 | 20 / 22 |
| Condensation longest path (edges) | 11 | 9 |

In-degree histogram (all imports) {degree: modules}: 0: 2, 1: 25, 2: 15, 3: 8, 4: 7, 5: 6, 6: 2, 7: 4, 8: 2, 10: 1, 11: 1

Out-degree histogram (all imports) {degree: modules}: 0: 15, 1: 14, 2: 15, 3: 7, 4: 3, 5: 4, 6: 5, 7: 4, 8: 2, 10: 1, 11: 3

**Top 10 fan-in (all imports)** (ties at cutoff: 2)

| Node | Degree |
|---|---|
| `tropoi.planet` | 11 |
| `tropoi.viz.normalization` | 10 |
| `tropoi.viz.renderers` | 8 |
| `tropoi.viz.specs` | 8 |
| `tropoi.numerics` | 7 |
| `tropoi.representation.archive.schema` | 7 |
| `tropoi.support` | 7 |
| `tropoi.viz.fields` | 7 |
| `tropoi.run.bve.config` | 6 |
| `tropoi.run.engine` | 6 |

**Top 10 fan-out (all imports)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 11 |
| `tropoi.cli.pe` | 11 |
| `tropoi.numerics` | 11 |
| `tropoi.cli.swe` | 10 |
| `tropoi.cli.bve` | 8 |
| `tropoi.run.bve.runner` | 8 |
| `tropoi.planet.planet` | 7 |
| `tropoi.representation.visual.snapshot` | 7 |
| `tropoi.run.swe.visualization` | 7 |
| `tropoi.viz` | 7 |

**Nontrivial strongly connected components (all imports)**

1. (7 modules) `tropoi.cli.bve`, `tropoi.cli.main`, `tropoi.cli.pe`, `tropoi.cli.swe`, `tropoi.representation.archive`, `tropoi.representation.archive.capsule`, `tropoi.representation.visual.snapshot`
2. (4 modules) `tropoi.numerics`, `tropoi.numerics.optimized_geodesic_sh`, `tropoi.numerics.spectral_operators`, `tropoi.numerics.spherical_backend`

**Reciprocal pairs (all imports)**

* `tropoi.cli.bve` <-> `tropoi.cli.main`
* `tropoi.numerics` <-> `tropoi.numerics.optimized_geodesic_sh`

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 95 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 15 |
| Directed edges \|E\| | 43 |
| Density | 0.204762 |
| Weakly connected components | 1 |
| WCC sizes | 15 |
| Strongly connected components | 12 |
| Nontrivial SCCs | 2 |
| Nontrivial SCC sizes | 3, 2 |
| Nodes in cycles | 5 |
| Fraction of nodes in cycles | 0.333333 |
| Sources (in-degree 0) | 0 |
| Sinks (out-degree 0) | 4 |
| Reciprocal pairs | 1 |
| Reciprocity | 0.046512 |
| In-degree min / median / max | 1 / 3 / 6 |
| In-degree mean | 2.866667 |
| In-degree quartiles | 1.5, 3, 4 |
| In-degree zeros | 0 |
| Out-degree min / median / max | 0 / 2 / 9 |
| Out-degree mean | 2.866667 |
| Out-degree quartiles | 0.5, 2, 5 |
| Out-degree zeros | 4 |
| Condensation nodes | 12 |
| Condensation edges | 29 |
| Condensation sources / sinks | 1 / 4 |
| Condensation longest path (edges) | 7 |

**Top 10 fan-in (packages)** (ties at cutoff: 3)

| Node | Degree |
|---|---|
| `tropoi` | 6 |
| `tropoi.run.bve` | 5 |
| `tropoi.viz` | 5 |
| `tropoi.physics` | 4 |
| `tropoi.planet` | 4 |
| `tropoi.run` | 3 |
| `tropoi.run.swe` | 3 |
| `tropoi.spatial` | 3 |
| `tropoi.numerics` | 2 |
| `tropoi.numerics.cuda` | 2 |

**Top 10 fan-out (packages)** (ties at cutoff: 2)

| Node | Degree |
|---|---|
| `tropoi.cli` | 9 |
| `tropoi.run.pe` | 6 |
| `tropoi.representation.visual` | 5 |
| `tropoi.run.bve` | 5 |
| `tropoi.run.swe` | 5 |
| `tropoi.representation.archive` | 4 |
| `tropoi.viz` | 3 |
| `tropoi.numerics` | 2 |
| `tropoi.physics` | 2 |
| `tropoi.planet` | 1 |

**Nontrivial SCCs (packages)**

1. `tropoi.cli`, `tropoi.representation.archive`, `tropoi.representation.visual`
2. `tropoi.run.bve`, `tropoi.viz`

## Class graph

* Classes: 81
* `inherits` edges between analysed classes: 9
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x3, `ValueError` x7, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x2

## Call graph

* Nodes (functions, methods, classes): 749
* Edges by kind: `call` 440, `constructor` 29, `instantiate` 183, `override` 6, `self` 146, `super` 2
* Call sites by resolution outcome: builtin 1340, external 959, internal 1067, unresolved 1379

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 25 | 26 | 6 | 42 |
| `calls/cli-inspect.mmd` | `tropoi.cli.main._cmd_inspect` | 4 | 26 | 31 | 1 | 36 |
| `calls/cli-run-bve.mmd` | `tropoi.cli.main._cmd_run_bve` | 4 | 25 | 28 | 6 | 140 |
| `calls/cli-run-pe.mmd` | `tropoi.cli.main._cmd_run_pe` | 4 | 44 | 50 | 11 | 146 |
| `calls/cli-run-swe.mmd` | `tropoi.cli.main._cmd_run_swe` | 4 | 47 | 54 | 14 | 156 |

## Definitions

| Metric | Definition |
|---|---|
| `nodes` | \|V\|, the number of nodes (`G.number_of_nodes()`). |
| `edges` | \|E\|, the number of directed edges; parallel import statements between the same ordered pair count once (`G.number_of_edges()`). |
| `self_loops` | Edges u->u (`nx.number_of_selfloops`). Always 0 for the import graph, which drops a module importing itself. |
| `density` | \|E\| / (\|V\|(\|V\|-1)), the fraction of possible ordered pairs that are edges (`nx.density` for a DiGraph). |
| `weakly_connected_components` | Number of maximal node sets connected when edge direction is ignored (`nx.weakly_connected_components`); `sizes` lists their sizes, descending. |
| `strongly_connected_components` | Number of maximal node sets in which every node reaches every other along directed edges (`nx.strongly_connected_components`), singletons included. |
| `nontrivial_sccs` | SCCs with at least 2 nodes, or one node with a self-loop. Each is a set of mutually reachable nodes, i.e. a dependency cycle or union of cycles. Listed with sizes and members. |
| `nodes_in_cycles` | Nodes that belong to a nontrivial SCC (equivalently, lie on at least one directed cycle); `fraction` = count / \|V\|. |
| `in_degree / out_degree` | Per-node number of incoming / outgoing edges (fan-in / fan-out). Summary: min, max, mean (= \|E\|/\|V\| for both), median (`statistics.median`), quartiles (`statistics.quantiles(n=4, method='inclusive')`), count of zeros, and the full histogram {degree: node count}. |
| `top_fan_in / top_fan_out` | The 10 nodes with the highest in- / out-degree, ties broken by name; `ties_at_cutoff` gives how many nodes share the last listed value. |
| `sources / sinks` | Nodes with in-degree 0 / out-degree 0. |
| `reciprocal_pairs` | Unordered pairs {u, v} with both u->v and v->u. |
| `reciprocity` | Fraction of edges whose reverse edge also exists (`nx.overall_reciprocity`) = 2 x reciprocal_pairs / \|E\|. |
| `condensation` | The DAG obtained by contracting each SCC to one node (`nx.condensation`): its node and edge counts, source and sink counts, and `longest_path_edges`, the number of edges on a longest directed path in that DAG (`nx.dag_longest_path_length`). |
| `intra_package_module_edges` | Package graph only: module-level edges whose two modules map to the same package. They are counted here instead of becoming package self-loops. |
