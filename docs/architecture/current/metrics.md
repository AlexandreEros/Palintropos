# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 144, `function+module`: 3, `function+type_checking`: 1, `module`: 196, `type_checking`: 6.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 108 | 108 |
| Directed edges \|E\| | 350 | 199 |
| Density | 0.030287 | 0.01722 |
| Weakly connected components | 1 | 5 |
| WCC sizes | 108 | 98, 7, 1, 1, 1 |
| Strongly connected components | 108 | 108 |
| Nontrivial SCCs | 0 | 0 |
| Nontrivial SCC sizes | - | - |
| Nodes in cycles | 0 | 0 |
| Fraction of nodes in cycles | 0 | 0 |
| Sources (in-degree 0) | 3 | 30 |
| Sinks (out-degree 0) | 29 | 39 |
| Reciprocal pairs | 0 | 0 |
| Reciprocity | 0 | 0 |
| In-degree min / median / max | 0 / 2 / 13 | 0 / 1 / 12 |
| In-degree mean | 3.240741 | 1.842593 |
| In-degree quartiles | 1, 2, 4 | 0, 1, 2 |
| In-degree zeros | 3 | 30 |
| Out-degree min / median / max | 0 / 2 / 19 | 0 / 1 / 9 |
| Out-degree mean | 3.240741 | 1.842593 |
| Out-degree quartiles | 0, 2, 5 | 0, 1, 3 |
| Out-degree zeros | 29 | 39 |
| Condensation nodes | 108 | 108 |
| Condensation edges | 350 | 199 |
| Condensation sources / sinks | 3 / 29 | 30 / 39 |
| Condensation longest path (edges) | 12 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 3, 1: 27, 2: 35, 3: 8, 4: 10, 5: 6, 6: 5, 7: 1, 8: 6, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1

Out-degree histogram (all imports) {degree: modules}: 0: 29, 1: 14, 2: 17, 3: 8, 4: 9, 5: 8, 6: 4, 7: 3, 8: 5, 9: 5, 10: 4, 13: 1, 19: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 6)

| Node | Degree |
|---|---|
| `tropoi.spatial.truncation` | 13 |
| `tropoi.spatial.planet` | 12 |
| `tropoi.spatial.specification` | 11 |
| `tropoi.representation.archive.schema` | 10 |
| `tropoi.representation.visual.normalization` | 10 |
| `tropoi.spatial.sigma_coordinate` | 9 |
| `tropoi.temporal.integration` | 9 |
| `tropoi.representation.archive.initial_state` | 8 |
| `tropoi.representation.visual.renderers` | 8 |
| `tropoi.representation.visual.specs` | 8 |

**Top 10 fan-out (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 19 |
| `tropoi.run.custom.execute` | 13 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.run.custom.presets` | 10 |
| `tropoi.spatial.planet` | 10 |
| `tropoi.representation.archive.capsule` | 9 |
| `tropoi.representation.visual.snapshot` | 9 |
| `tropoi.run.bve.runner` | 9 |
| `tropoi.run.custom.prepare` | 9 |

**Nontrivial strongly connected components (all imports)**

None.

**Reciprocal pairs (all imports)**

None.

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 115 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 20 |
| Directed edges \|E\| | 92 |
| Density | 0.242105 |
| Weakly connected components | 1 |
| WCC sizes | 20 |
| Strongly connected components | 13 |
| Nontrivial SCCs | 3 |
| Nontrivial SCC sizes | 6, 2, 2 |
| Nodes in cycles | 10 |
| Fraction of nodes in cycles | 0.5 |
| Sources (in-degree 0) | 2 |
| Sinks (out-degree 0) | 2 |
| Reciprocal pairs | 7 |
| Reciprocity | 0.152174 |
| In-degree min / median / max | 0 / 4 / 13 |
| In-degree mean | 4.6 |
| In-degree quartiles | 2, 4, 6 |
| In-degree zeros | 2 |
| Out-degree min / median / max | 0 / 3.5 / 11 |
| Out-degree mean | 4.6 |
| Out-degree quartiles | 1, 3.5, 9 |
| Out-degree zeros | 2 |
| Condensation nodes | 13 |
| Condensation edges | 39 |
| Condensation sources / sinks | 2 / 2 |
| Condensation longest path (edges) | 9 |

**Top 10 fan-in (packages)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 13 |
| `tropoi.temporal.tendencies` | 8 |
| `tropoi.representation.archive` | 7 |
| `tropoi.temporal` | 7 |
| `tropoi.representation.visual` | 6 |
| `tropoi.run.bve` | 6 |
| `tropoi.spatial.states` | 6 |
| `tropoi.spatial.terrain` | 6 |
| `tropoi.spatial.transforms` | 6 |
| `tropoi.representation.diagnostics` | 4 |

**Top 10 fan-out (packages)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.representation.visual` | 11 |
| `tropoi.run.custom` | 11 |
| `tropoi.run.pe` | 11 |
| `tropoi.run.swe` | 10 |
| `tropoi.cli` | 9 |
| `tropoi.run.bve` | 9 |
| `tropoi.representation.archive` | 4 |
| `tropoi.run.held_suarez` | 4 |
| `tropoi.spatial` | 4 |
| `tropoi.temporal.tendencies` | 4 |

**Nontrivial SCCs (packages)**

1. `tropoi.representation.archive`, `tropoi.representation.visual`, `tropoi.run`, `tropoi.run.bve`, `tropoi.run.pe`, `tropoi.run.swe`
2. `tropoi.spatial`, `tropoi.spatial.operators`
3. `tropoi.spatial.grids`, `tropoi.spatial.transforms`

## Class graph

* Classes: 152
* `inherits` edges between analysed classes: 10
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x6, `UserWarning` x1, `ValueError` x16, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 1360
* Edges by kind: `call` 959, `constructor` 41, `instantiate` 381, `override` 6, `self` 273, `super` 2
* Call sites by resolution outcome: builtin 2615, external 1809, internal 2263, unresolved 2697

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 27 | 28 | 6 | 44 |
| `calls/cli-inspect.mmd` | `tropoi.cli.main._cmd_inspect` | 4 | 32 | 42 | 2 | 43 |
| `calls/cli-run-bve.mmd` | `tropoi.cli.main._cmd_run_bve` | 4 | 45 | 60 | 13 | 207 |
| `calls/cli-run-pe.mmd` | `tropoi.cli.main._cmd_run_pe` | 4 | 56 | 65 | 19 | 169 |
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
