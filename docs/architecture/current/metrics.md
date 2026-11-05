# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 145, `function+module`: 3, `function+type_checking`: 1, `module`: 207, `type_checking`: 6.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 115 | 115 |
| Directed edges \|E\| | 362 | 210 |
| Density | 0.027613 | 0.016018 |
| Weakly connected components | 1 | 5 |
| WCC sizes | 115 | 104, 8, 1, 1, 1 |
| Strongly connected components | 115 | 115 |
| Nontrivial SCCs | 0 | 0 |
| Nontrivial SCC sizes | - | - |
| Nodes in cycles | 0 | 0 |
| Fraction of nodes in cycles | 0 | 0 |
| Sources (in-degree 0) | 3 | 31 |
| Sinks (out-degree 0) | 30 | 39 |
| Reciprocal pairs | 0 | 0 |
| Reciprocity | 0 | 0 |
| In-degree min / median / max | 0 / 2 / 14 | 0 / 1 / 12 |
| In-degree mean | 3.147826 | 1.826087 |
| In-degree quartiles | 1, 2, 4 | 0, 1, 2 |
| In-degree zeros | 3 | 31 |
| Out-degree min / median / max | 0 / 2 / 20 | 0 / 1 / 9 |
| Out-degree mean | 3.147826 | 1.826087 |
| Out-degree quartiles | 0, 2, 5 | 0, 1, 3 |
| Out-degree zeros | 30 | 39 |
| Condensation nodes | 115 | 115 |
| Condensation edges | 362 | 210 |
| Condensation sources / sinks | 3 / 30 | 31 / 39 |
| Condensation longest path (edges) | 14 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 3, 1: 30, 2: 34, 3: 12, 4: 11, 5: 6, 6: 6, 7: 2, 8: 4, 9: 2, 10: 4, 14: 1

Out-degree histogram (all imports) {degree: modules}: 0: 30, 1: 18, 2: 19, 3: 7, 4: 10, 5: 7, 6: 4, 7: 4, 8: 5, 9: 5, 10: 4, 13: 1, 20: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.spatial.truncation` | 14 |
| `tropoi.representation.archive.schema` | 10 |
| `tropoi.representation.visual.normalization` | 10 |
| `tropoi.spatial.planet` | 10 |
| `tropoi.spatial.specification` | 10 |
| `tropoi.spatial.sigma_coordinate` | 9 |
| `tropoi.temporal.integration` | 9 |
| `tropoi.representation.archive.initial_state` | 8 |
| `tropoi.representation.visual.renderers` | 8 |
| `tropoi.representation.visual.specs` | 8 |

**Top 10 fan-out (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 20 |
| `tropoi.run.custom.execute` | 13 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.run.custom.presets` | 10 |
| `tropoi.spatial.planet` | 10 |
| `tropoi.representation.archive.capsule` | 9 |
| `tropoi.run.bve.runner` | 9 |
| `tropoi.run.custom` | 9 |
| `tropoi.run.custom.prepare` | 9 |

**Nontrivial strongly connected components (all imports)**

None.

**Reciprocal pairs (all imports)**

None.

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 119 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 21 |
| Directed edges \|E\| | 91 |
| Density | 0.216667 |
| Weakly connected components | 1 |
| WCC sizes | 21 |
| Strongly connected components | 17 |
| Nontrivial SCCs | 4 |
| Nontrivial SCC sizes | 2, 2, 2, 2 |
| Nodes in cycles | 8 |
| Fraction of nodes in cycles | 0.380952 |
| Sources (in-degree 0) | 2 |
| Sinks (out-degree 0) | 3 |
| Reciprocal pairs | 4 |
| Reciprocity | 0.087912 |
| In-degree min / median / max | 0 / 4 / 13 |
| In-degree mean | 4.333333 |
| In-degree quartiles | 2, 4, 6 |
| In-degree zeros | 2 |
| Out-degree min / median / max | 0 / 3 / 11 |
| Out-degree mean | 4.333333 |
| Out-degree quartiles | 1, 3, 9 |
| Out-degree zeros | 3 |
| Condensation nodes | 17 |
| Condensation edges | 63 |
| Condensation sources / sinks | 2 / 3 |
| Condensation longest path (edges) | 11 |

**Top 10 fan-in (packages)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 13 |
| `tropoi.temporal.tendencies` | 9 |
| `tropoi.temporal` | 8 |
| `tropoi.representation.archive` | 7 |
| `tropoi.representation.visual` | 6 |
| `tropoi.spatial.states` | 6 |
| `tropoi.spatial.transforms` | 6 |
| `tropoi.spatial.terrain` | 5 |
| `tropoi.representation.diagnostics` | 4 |
| `tropoi.run` | 4 |

**Top 10 fan-out (packages)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.run.custom` | 11 |
| `tropoi.run.bve` | 10 |
| `tropoi.run.pe` | 10 |
| `tropoi.cli` | 9 |
| `tropoi.representation.visual` | 9 |
| `tropoi.run.swe` | 9 |
| `tropoi.representation.archive` | 4 |
| `tropoi.run.held_suarez` | 4 |
| `tropoi.spatial` | 4 |
| `tropoi.temporal.tendencies` | 4 |

**Nontrivial SCCs (packages)**

1. `tropoi.representation.archive`, `tropoi.representation.visual`
2. `tropoi.spatial`, `tropoi.spatial.operators`
3. `tropoi.spatial.grids`, `tropoi.spatial.transforms`
4. `tropoi.temporal`, `tropoi.temporal.tendencies`

## Declared layer graph

The only part of this report that compares the code with an expectation. The layers below are declared in DEFAULT_LAYERS in tools/architecture/layers.py (top = most dependent), not inferred from the code: every package is assigned to the first layer whose prefix its name matches. All import edges between packages of different layers count, `TYPE_CHECKING`-only ones included; edges inside one layer are counted on the layer instead of becoming self-loops.

| Layer | Prefixes | Packages |
|---|---|---|
| `cli` | `tropoi.cli` | `tropoi.cli` |
| `run` | `tropoi.run` | `tropoi.run`, `tropoi.run.bve`, `tropoi.run.custom`, `tropoi.run.held_suarez`, `tropoi.run.pe`, `tropoi.run.swe` |
| `representation` | `tropoi.representation` | `tropoi.representation`, `tropoi.representation.archive`, `tropoi.representation.diagnostics`, `tropoi.representation.visual` |
| `core` | `tropoi.spatial`, `tropoi.temporal` | `tropoi.spatial`, `tropoi.spatial.grids`, `tropoi.spatial.initialization`, `tropoi.spatial.operators`, `tropoi.spatial.states`, `tropoi.spatial.terrain`, `tropoi.spatial.transforms`, `tropoi.spatial.transforms.cuda`, `tropoi.temporal`, `tropoi.temporal.tendencies` |

Packages matched by no declared layer: none.

| Metric | layers |
|---|---|
| Nodes \|V\| | 4 |
| Directed edges \|E\| | 6 |
| Density | 0.5 |
| Weakly connected components | 1 |
| WCC sizes | 4 |
| Strongly connected components | 4 |
| Nontrivial SCCs | 0 |
| Nontrivial SCC sizes | - |
| Nodes in cycles | 0 |
| Fraction of nodes in cycles | 0 |
| Sources (in-degree 0) | 1 |
| Sinks (out-degree 0) | 1 |
| Reciprocal pairs | 0 |
| Reciprocity | 0 |
| In-degree min / median / max | 0 / 1.5 / 3 |
| In-degree mean | 1.5 |
| In-degree quartiles | 0.75, 1.5, 2.25 |
| In-degree zeros | 1 |
| Out-degree min / median / max | 0 / 1.5 / 3 |
| Out-degree mean | 1.5 |
| Out-degree quartiles | 0.75, 1.5, 2.25 |
| Out-degree zeros | 1 |
| Condensation nodes | 4 |
| Condensation edges | 6 |
| Condensation sources / sinks | 1 / 1 |
| Condensation longest path (edges) | 3 |

**Edges against the declared order (layer violations)**

None: every import edge, `TYPE_CHECKING`-only ones included, points downward or stays inside a layer, so the layer quotient graph is a DAG.

## Class graph

* Classes: 162
* Classes in at least one `inherits` edge (drawn in `classes.mmd`): 18; with neither an analysed base nor an analysed subclass: 144
* `inherits` edges between analysed classes: 12
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x6, `UserWarning` x1, `ValueError` x17, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 1447
* Edges by kind: `call` 1023, `constructor` 42, `instantiate` 411, `override` 6, `self` 281, `super` 2
* Call sites by resolution outcome: builtin 2874, external 1923, internal 2456, unresolved 2952

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
| `declared_by / declared / packages_by_layer / unassigned_packages` | Layer graph only: where the declaration came from (`DEFAULT_LAYERS` or `--layer`), the layers it gives (top first) with their package prefixes, the packages each one matched, and the packages no prefix matched. A declaration, not a measurement. |
| `upward_package_edges / modules_in_upward_edges` | Layer graph only: the package edges running from a lower declared layer to a higher one (the layer violations), each flagged `type_checking_only` when all its imports are under `if TYPE_CHECKING:`, and the total number of module-level import edges merged into them. Empty means every import edge points downward or stays inside a layer, i.e. the layer quotient graph is a DAG and the layers can be levelized. |
