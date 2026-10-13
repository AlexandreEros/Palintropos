# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 126, `function+module`: 2, `function+type_checking`: 2, `module`: 249, `type_checking`: 7.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 150 | 150 |
| Directed edges \|E\| | 386 | 251 |
| Density | 0.017271 | 0.01123 |
| Weakly connected components | 3 | 8 |
| WCC sizes | 99, 50, 1 | 91, 50, 4, 1, 1, 1, 1, 1 |
| Strongly connected components | 140 | 150 |
| Nontrivial SCCs | 1 | 0 |
| Nontrivial SCC sizes | 11 | - |
| Nodes in cycles | 11 | 0 |
| Fraction of nodes in cycles | 0.073333 | 0 |
| Sources (in-degree 0) | 56 | 80 |
| Sinks (out-degree 0) | 27 | 34 |
| Reciprocal pairs | 1 | 0 |
| Reciprocity | 0.005181 | 0 |
| In-degree min / median / max | 0 / 1 / 49 | 0 / 0 / 49 |
| In-degree mean | 2.573333 | 1.673333 |
| In-degree quartiles | 0, 1, 3 | 0, 0, 2 |
| In-degree zeros | 56 | 80 |
| Out-degree min / median / max | 0 / 1 / 18 | 0 / 1 / 11 |
| Out-degree mean | 2.573333 | 1.673333 |
| Out-degree quartiles | 1, 1, 3 | 1, 1, 2 |
| Out-degree zeros | 27 | 34 |
| Condensation nodes | 140 | 150 |
| Condensation edges | 319 | 251 |
| Condensation sources / sinks | 57 / 27 | 80 / 34 |
| Condensation longest path (edges) | 11 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 56, 1: 23, 2: 25, 3: 9, 4: 10, 5: 3, 6: 7, 7: 5, 8: 4, 9: 4, 11: 1, 12: 1, 14: 1, 49: 1

Out-degree histogram (all imports) {degree: modules}: 0: 27, 1: 63, 2: 17, 3: 6, 4: 7, 5: 6, 6: 5, 7: 4, 8: 4, 9: 1, 10: 5, 11: 2, 12: 2, 18: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi._compat` | 49 |
| `tropoi.spatial.planet` | 14 |
| `tropoi.spatial.truncation` | 12 |
| `tropoi.representation.visual.normalization` | 11 |
| `tropoi.representation.archive.schema` | 9 |
| `tropoi.representation.visual.renderers` | 9 |
| `tropoi.representation.visual.specs` | 9 |
| `tropoi.temporal.integration` | 9 |
| `tropoi.representation.visual.fields` | 8 |
| `tropoi.run.bve.config` | 8 |

**Top 10 fan-out (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 18 |
| `tropoi.cli.pe` | 12 |
| `tropoi.cli.swe` | 12 |
| `tropoi.numerics` | 11 |
| `tropoi.run.custom.execute` | 11 |
| `tropoi.cli.bve` | 10 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.run.custom.presets` | 10 |
| `tropoi.spatial.planet` | 10 |

**Nontrivial strongly connected components (all imports)**

1. (11 modules) `tropoi.cli.bve`, `tropoi.cli.main`, `tropoi.cli.pe`, `tropoi.cli.swe`, `tropoi.representation.archive`, `tropoi.representation.archive.capsule`, `tropoi.representation.visual.snapshot`, `tropoi.run.custom`, `tropoi.run.custom.execute`, `tropoi.run.custom.prepare`, `tropoi.run.custom.presets`

**Reciprocal pairs (all imports)**

* `tropoi.cli.bve` <-> `tropoi.cli.main`

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 105 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 26 |
| Directed edges \|E\| | 95 |
| Density | 0.146154 |
| Weakly connected components | 1 |
| WCC sizes | 26 |
| Strongly connected components | 17 |
| Nontrivial SCCs | 3 |
| Nontrivial SCC sizes | 7, 3, 2 |
| Nodes in cycles | 12 |
| Fraction of nodes in cycles | 0.461538 |
| Sources (in-degree 0) | 7 |
| Sinks (out-degree 0) | 3 |
| Reciprocal pairs | 5 |
| Reciprocity | 0.105263 |
| In-degree min / median / max | 0 / 3 / 15 |
| In-degree mean | 3.653846 |
| In-degree quartiles | 0.25, 3, 5.75 |
| In-degree zeros | 7 |
| Out-degree min / median / max | 0 / 3 / 11 |
| Out-degree mean | 3.653846 |
| Out-degree quartiles | 1, 3, 4.75 |
| Out-degree zeros | 3 |
| Condensation nodes | 17 |
| Condensation edges | 34 |
| Condensation sources / sinks | 7 / 3 |
| Condensation longest path (edges) | 7 |

**Top 10 fan-in (packages)** (ties at cutoff: 3)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 15 |
| `tropoi` | 9 |
| `tropoi.temporal.tendencies` | 9 |
| `tropoi.temporal` | 7 |
| `tropoi.representation.visual` | 6 |
| `tropoi.spatial.states` | 6 |
| `tropoi.spatial.transforms` | 6 |
| `tropoi.run.bve` | 5 |
| `tropoi.spatial.grids` | 5 |
| `tropoi.spatial.terrain` | 5 |

**Top 10 fan-out (packages)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.cli` | 11 |
| `tropoi.run.custom` | 10 |
| `tropoi.representation.visual` | 9 |
| `tropoi.run.pe` | 8 |
| `tropoi.run.bve` | 7 |
| `tropoi.run.swe` | 7 |
| `tropoi.numerics` | 5 |
| `tropoi.representation.archive` | 4 |
| `tropoi.run.held_suarez` | 4 |
| `tropoi.spatial` | 4 |

**Nontrivial SCCs (packages)**

1. `tropoi.cli`, `tropoi.representation.archive`, `tropoi.representation.visual`, `tropoi.run.bve`, `tropoi.run.custom`, `tropoi.run.pe`, `tropoi.run.swe`
2. `tropoi.spatial`, `tropoi.spatial.operators`, `tropoi.spatial.terrain`
3. `tropoi.spatial.grids`, `tropoi.spatial.transforms`

## Class graph

* Classes: 138
* `inherits` edges between analysed classes: 9
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x5, `ValueError` x12, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 1250
* Edges by kind: `call` 838, `constructor` 40, `instantiate` 338, `override` 6, `self` 278, `super` 2
* Call sites by resolution outcome: builtin 2221, external 1567, internal 2006, unresolved 2454

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 27 | 28 | 6 | 44 |
| `calls/cli-inspect.mmd` | `tropoi.cli.main._cmd_inspect` | 4 | 26 | 31 | 1 | 36 |
| `calls/cli-run-bve.mmd` | `tropoi.cli.main._cmd_run_bve` | 4 | 45 | 60 | 13 | 207 |
| `calls/cli-run-pe.mmd` | `tropoi.cli.main._cmd_run_pe` | 4 | 44 | 50 | 12 | 147 |
| `calls/cli-run-swe.mmd` | `tropoi.cli.main._cmd_run_swe` | 4 | 82 | 119 | 15 | 209 |

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
