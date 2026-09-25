# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 82, `function+module`: 2, `function+type_checking`: 2, `module`: 210, `type_checking`: 7.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 132 | 132 |
| Directed edges \|E\| | 303 | 212 |
| Density | 0.017523 | 0.01226 |
| Weakly connected components | 3 | 7 |
| WCC sizes | 81, 50, 1 | 74, 50, 4, 1, 1, 1, 1 |
| Strongly connected components | 126 | 132 |
| Nontrivial SCCs | 1 | 0 |
| Nontrivial SCC sizes | 7 | - |
| Nodes in cycles | 7 | 0 |
| Fraction of nodes in cycles | 0.05303 | 0 |
| Sources (in-degree 0) | 55 | 74 |
| Sinks (out-degree 0) | 23 | 30 |
| Reciprocal pairs | 1 | 0 |
| Reciprocity | 0.006601 | 0 |
| In-degree min / median / max | 0 / 1 / 49 | 0 / 0 / 49 |
| In-degree mean | 2.295455 | 1.606061 |
| In-degree quartiles | 0, 1, 3 | 0, 0, 2 |
| In-degree zeros | 55 | 74 |
| Out-degree min / median / max | 0 / 1 / 12 | 0 / 1 / 11 |
| Out-degree mean | 2.295455 | 1.606061 |
| Out-degree quartiles | 1, 1, 2.25 | 1, 1, 2 |
| Out-degree zeros | 23 | 30 |
| Condensation nodes | 126 | 132 |
| Condensation edges | 273 | 212 |
| Condensation sources / sinks | 56 / 23 | 74 / 30 |
| Condensation longest path (edges) | 11 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 55, 1: 23, 2: 14, 3: 12, 4: 9, 5: 3, 6: 6, 7: 3, 8: 1, 9: 3, 11: 1, 13: 1, 49: 1

Out-degree histogram (all imports) {degree: modules}: 0: 23, 1: 61, 2: 15, 3: 5, 4: 5, 5: 4, 6: 5, 7: 4, 8: 2, 9: 2, 10: 2, 11: 3, 12: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 3)

| Node | Degree |
|---|---|
| `tropoi._compat` | 49 |
| `tropoi.spatial.planet` | 13 |
| `tropoi.representation.visual.normalization` | 11 |
| `tropoi.representation.visual.renderers` | 9 |
| `tropoi.representation.visual.specs` | 9 |
| `tropoi.spatial.truncation` | 9 |
| `tropoi.representation.visual.fields` | 8 |
| `tropoi.representation.archive.schema` | 7 |
| `tropoi.spatial.grids.geodesic_grid` | 7 |
| `tropoi.spatial.grids.grid_base` | 7 |

**Top 10 fan-out (all imports)** (ties at cutoff: 2)

| Node | Degree |
|---|---|
| `tropoi.cli.pe` | 12 |
| `tropoi.cli.main` | 11 |
| `tropoi.cli.swe` | 11 |
| `tropoi.numerics` | 11 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.spatial.planet` | 10 |
| `tropoi.cli.bve` | 9 |
| `tropoi.run.bve.runner` | 9 |
| `tropoi.representation.visual.evaluate` | 8 |
| `tropoi.representation.visual.planet_viewer` | 8 |

**Nontrivial strongly connected components (all imports)**

1. (7 modules) `tropoi.cli.bve`, `tropoi.cli.main`, `tropoi.cli.pe`, `tropoi.cli.swe`, `tropoi.representation.archive`, `tropoi.representation.archive.capsule`, `tropoi.representation.visual.snapshot`

**Reciprocal pairs (all imports)**

* `tropoi.cli.bve` <-> `tropoi.cli.main`

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 85 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 24 |
| Directed edges \|E\| | 77 |
| Density | 0.139493 |
| Weakly connected components | 1 |
| WCC sizes | 24 |
| Strongly connected components | 16 |
| Nontrivial SCCs | 3 |
| Nontrivial SCC sizes | 6, 3, 2 |
| Nodes in cycles | 11 |
| Fraction of nodes in cycles | 0.458333 |
| Sources (in-degree 0) | 6 |
| Sinks (out-degree 0) | 3 |
| Reciprocal pairs | 3 |
| Reciprocity | 0.077922 |
| In-degree min / median / max | 0 / 3 / 13 |
| In-degree mean | 3.208333 |
| In-degree quartiles | 0.75, 3, 4.25 |
| In-degree zeros | 6 |
| Out-degree min / median / max | 0 / 2.5 / 10 |
| Out-degree mean | 3.208333 |
| Out-degree quartiles | 1, 2.5, 4.25 |
| Out-degree zeros | 3 |
| Condensation nodes | 16 |
| Condensation edges | 29 |
| Condensation sources / sinks | 6 / 3 |
| Condensation longest path (edges) | 6 |

**Top 10 fan-in (packages)** (ties at cutoff: 6)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 13 |
| `tropoi` | 9 |
| `tropoi.temporal.tendencies` | 7 |
| `tropoi.representation.visual` | 6 |
| `tropoi.spatial.grids` | 5 |
| `tropoi.spatial.transforms` | 5 |
| `tropoi.representation.diagnostics` | 4 |
| `tropoi.run.bve` | 4 |
| `tropoi.spatial.operators` | 4 |
| `tropoi.spatial.states` | 4 |

**Top 10 fan-out (packages)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.cli` | 10 |
| `tropoi.run.pe` | 8 |
| `tropoi.representation.visual` | 7 |
| `tropoi.run.bve` | 7 |
| `tropoi.run.swe` | 7 |
| `tropoi.numerics` | 5 |
| `tropoi.representation.archive` | 4 |
| `tropoi.spatial` | 4 |
| `tropoi.planet` | 3 |
| `tropoi.spatial.initialization` | 3 |

**Nontrivial SCCs (packages)**

1. `tropoi.cli`, `tropoi.representation.archive`, `tropoi.representation.visual`, `tropoi.run.bve`, `tropoi.run.pe`, `tropoi.run.swe`
2. `tropoi.spatial`, `tropoi.spatial.operators`, `tropoi.spatial.terrain`
3. `tropoi.spatial.grids`, `tropoi.spatial.transforms`

## Class graph

* Classes: 104
* `inherits` edges between analysed classes: 9
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x3, `ValueError` x9, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x2

## Call graph

* Nodes (functions, methods, classes): 885
* Edges by kind: `call` 530, `constructor` 30, `instantiate` 233, `override` 6, `self` 205, `super` 2
* Call sites by resolution outcome: builtin 1585, external 1124, internal 1323, unresolved 1600

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 27 | 28 | 6 | 44 |
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
