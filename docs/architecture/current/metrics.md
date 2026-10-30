# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 147, `function+module`: 3, `function+type_checking`: 2, `module`: 262, `type_checking`: 5.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 158 | 158 |
| Directed edges \|E\| | 419 | 265 |
| Density | 0.016891 | 0.010683 |
| Weakly connected components | 3 | 7 |
| WCC sizes | 109, 48, 1 | 102, 48, 4, 1, 1, 1, 1 |
| Strongly connected components | 144 | 158 |
| Nontrivial SCCs | 3 | 0 |
| Nontrivial SCC sizes | 13, 2, 2 | - |
| Nodes in cycles | 17 | 0 |
| Fraction of nodes in cycles | 0.107595 | 0 |
| Sources (in-degree 0) | 54 | 81 |
| Sinks (out-degree 0) | 28 | 37 |
| Reciprocal pairs | 4 | 0 |
| Reciprocity | 0.019093 | 0 |
| In-degree min / median / max | 0 / 2 / 47 | 0 / 0 / 47 |
| In-degree mean | 2.651899 | 1.677215 |
| In-degree quartiles | 0, 2, 3 | 0, 0, 2 |
| In-degree zeros | 54 | 81 |
| Out-degree min / median / max | 0 / 1 / 19 | 0 / 1 / 11 |
| Out-degree mean | 2.651899 | 1.677215 |
| Out-degree quartiles | 1, 1, 4 | 1, 1, 2 |
| Out-degree zeros | 28 | 37 |
| Condensation nodes | 144 | 158 |
| Condensation edges | 331 | 265 |
| Condensation sources / sinks | 55 / 29 | 81 / 37 |
| Condensation longest path (edges) | 10 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 54, 1: 21, 2: 35, 3: 10, 4: 8, 5: 8, 6: 4, 7: 4, 8: 3, 9: 5, 10: 2, 11: 1, 13: 1, 14: 1, 47: 1

Out-degree histogram (all imports) {degree: modules}: 0: 28, 1: 64, 2: 18, 3: 7, 4: 11, 5: 6, 6: 3, 7: 3, 8: 3, 9: 5, 10: 5, 11: 2, 12: 1, 13: 1, 19: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi._compat` | 47 |
| `tropoi.spatial.planet` | 14 |
| `tropoi.spatial.truncation` | 13 |
| `tropoi.representation.visual.normalization` | 11 |
| `tropoi.representation.archive.schema` | 10 |
| `tropoi.spatial.specification` | 10 |
| `tropoi.representation.visual.renderers` | 9 |
| `tropoi.representation.visual.specs` | 9 |
| `tropoi.spatial.environment` | 9 |
| `tropoi.spatial.sigma_coordinate` | 9 |

**Top 10 fan-out (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 19 |
| `tropoi.cli.pe` | 13 |
| `tropoi.cli.swe` | 12 |
| `tropoi.numerics` | 11 |
| `tropoi.run.custom.execute` | 11 |
| `tropoi.cli.bve` | 10 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.run.custom.presets` | 10 |
| `tropoi.spatial.planet` | 10 |

**Nontrivial strongly connected components (all imports)**

1. (13 modules) `tropoi.cli.bve`, `tropoi.cli.main`, `tropoi.cli.pe`, `tropoi.cli.swe`, `tropoi.cli.terrain`, `tropoi.representation.archive`, `tropoi.representation.archive.capsule`, `tropoi.representation.visual.snapshot`, `tropoi.run.custom`, `tropoi.run.custom.execute`, `tropoi.run.custom.planet`, `tropoi.run.custom.prepare`, `tropoi.run.custom.presets`
2. (2 modules) `tropoi.representation.archive.held_suarez`, `tropoi.representation.visual.atmosphere`
3. (2 modules) `tropoi.spatial.terrain.height_function`, `tropoi.spatial.terrain.raster`

**Reciprocal pairs (all imports)**

* `tropoi.cli.bve` <-> `tropoi.cli.main`
* `tropoi.representation.archive.held_suarez` <-> `tropoi.representation.visual.atmosphere`
* `tropoi.run.custom.planet` <-> `tropoi.run.custom.prepare`
* `tropoi.spatial.terrain.height_function` <-> `tropoi.spatial.terrain.raster`

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 116 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 26 |
| Directed edges \|E\| | 97 |
| Density | 0.149231 |
| Weakly connected components | 1 |
| WCC sizes | 26 |
| Strongly connected components | 18 |
| Nontrivial SCCs | 3 |
| Nontrivial SCC sizes | 7, 2, 2 |
| Nodes in cycles | 11 |
| Fraction of nodes in cycles | 0.423077 |
| Sources (in-degree 0) | 7 |
| Sinks (out-degree 0) | 3 |
| Reciprocal pairs | 5 |
| Reciprocity | 0.103093 |
| In-degree min / median / max | 0 / 3 / 15 |
| In-degree mean | 3.730769 |
| In-degree quartiles | 0.25, 3, 5.75 |
| In-degree zeros | 7 |
| Out-degree min / median / max | 0 / 3 / 11 |
| Out-degree mean | 3.730769 |
| Out-degree quartiles | 1, 3, 4.75 |
| Out-degree zeros | 3 |
| Condensation nodes | 18 |
| Condensation edges | 40 |
| Condensation sources / sinks | 7 / 3 |
| Condensation longest path (edges) | 8 |

**Top 10 fan-in (packages)** (ties at cutoff: 3)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 15 |
| `tropoi` | 9 |
| `tropoi.temporal.tendencies` | 9 |
| `tropoi.representation.visual` | 7 |
| `tropoi.spatial.transforms` | 7 |
| `tropoi.temporal` | 7 |
| `tropoi.spatial.states` | 6 |
| `tropoi.run.bve` | 5 |
| `tropoi.spatial.grids` | 5 |
| `tropoi.spatial.terrain` | 5 |

**Top 10 fan-out (packages)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.cli` | 11 |
| `tropoi.run.custom` | 11 |
| `tropoi.representation.visual` | 9 |
| `tropoi.run.pe` | 9 |
| `tropoi.run.bve` | 7 |
| `tropoi.run.swe` | 7 |
| `tropoi.numerics` | 5 |
| `tropoi.representation.archive` | 4 |
| `tropoi.run.held_suarez` | 4 |
| `tropoi.spatial` | 4 |

**Nontrivial SCCs (packages)**

1. `tropoi.cli`, `tropoi.representation.archive`, `tropoi.representation.visual`, `tropoi.run.bve`, `tropoi.run.custom`, `tropoi.run.pe`, `tropoi.run.swe`
2. `tropoi.spatial`, `tropoi.spatial.operators`
3. `tropoi.spatial.grids`, `tropoi.spatial.transforms`

## Class graph

* Classes: 155
* `inherits` edges between analysed classes: 12
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x6, `UserWarning` x1, `ValueError` x16, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 1404
* Edges by kind: `call` 998, `constructor` 41, `instantiate` 392, `override` 6, `self` 277, `super` 2
* Call sites by resolution outcome: builtin 2739, external 1864, internal 2385, unresolved 2805

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
