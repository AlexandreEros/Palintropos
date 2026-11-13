# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 167, `function+module`: 5, `function+type_checking`: 1, `module`: 262, `type_checking`: 6.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 144 | 144 |
| Directed edges \|E\| | 441 | 267 |
| Density | 0.021416 | 0.012966 |
| Weakly connected components | 1 | 7 |
| WCC sizes | 144 | 131, 8, 1, 1, 1, 1, 1 |
| Strongly connected components | 144 | 144 |
| Nontrivial SCCs | 0 | 0 |
| Nontrivial SCC sizes | - | - |
| Nodes in cycles | 0 | 0 |
| Fraction of nodes in cycles | 0 | 0 |
| Sources (in-degree 0) | 6 | 38 |
| Sinks (out-degree 0) | 40 | 51 |
| Reciprocal pairs | 0 | 0 |
| Reciprocity | 0 | 0 |
| In-degree min / median / max | 0 / 2 / 15 | 0 / 1 / 12 |
| In-degree mean | 3.0625 | 1.854167 |
| In-degree quartiles | 1, 2, 4 | 0, 1, 2 |
| In-degree zeros | 6 | 38 |
| Out-degree min / median / max | 0 / 2 / 21 | 0 / 1 / 9 |
| Out-degree mean | 3.0625 | 1.854167 |
| Out-degree quartiles | 0, 2, 5 | 0, 1, 3 |
| Out-degree zeros | 40 | 51 |
| Condensation nodes | 144 | 144 |
| Condensation edges | 441 | 267 |
| Condensation sources / sinks | 6 / 40 | 38 / 51 |
| Condensation longest path (edges) | 14 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 6, 1: 39, 2: 40, 3: 17, 4: 14, 5: 5, 6: 6, 7: 4, 8: 4, 9: 2, 10: 5, 11: 1, 15: 1

Out-degree histogram (all imports) {degree: modules}: 0: 40, 1: 24, 2: 22, 3: 12, 4: 7, 5: 10, 6: 5, 7: 6, 8: 5, 9: 3, 10: 7, 13: 1, 16: 1, 21: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.spatial.truncation` | 15 |
| `tropoi.spatial.planet` | 11 |
| `tropoi.representation.archive.schema` | 10 |
| `tropoi.representation.visual.normalization` | 10 |
| `tropoi.spatial.sigma_coordinate` | 10 |
| `tropoi.spatial.specification` | 10 |
| `tropoi.spatial.states.primitive_equations` | 10 |
| `tropoi.spatial.laplacian` | 9 |
| `tropoi.temporal.integration` | 9 |
| `tropoi.representation.archive.initial_state` | 8 |

**Top 10 fan-out (all imports)** (ties at cutoff: 7)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 21 |
| `tropoi.run.radiative.model` | 16 |
| `tropoi.run.custom.execute` | 13 |
| `tropoi.representation.archive.capsule` | 10 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.representation.visual.sky` | 10 |
| `tropoi.run.custom.prepare` | 10 |
| `tropoi.run.custom.presets` | 10 |
| `tropoi.spatial.planet` | 10 |

**Nontrivial strongly connected components (all imports)**

None.

**Reciprocal pairs (all imports)**

None.

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 155 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 24 |
| Directed edges \|E\| | 101 |
| Density | 0.182971 |
| Weakly connected components | 1 |
| WCC sizes | 24 |
| Strongly connected components | 20 |
| Nontrivial SCCs | 4 |
| Nontrivial SCC sizes | 2, 2, 2, 2 |
| Nodes in cycles | 8 |
| Fraction of nodes in cycles | 0.333333 |
| Sources (in-degree 0) | 2 |
| Sinks (out-degree 0) | 4 |
| Reciprocal pairs | 4 |
| Reciprocity | 0.079208 |
| In-degree min / median / max | 0 / 4 / 14 |
| In-degree mean | 4.208333 |
| In-degree quartiles | 2, 4, 6 |
| In-degree zeros | 2 |
| Out-degree min / median / max | 0 / 3.5 / 11 |
| Out-degree mean | 4.208333 |
| Out-degree quartiles | 1, 3.5, 7.5 |
| Out-degree zeros | 4 |
| Condensation nodes | 20 |
| Condensation edges | 71 |
| Condensation sources / sinks | 2 / 4 |
| Condensation longest path (edges) | 11 |

**Top 10 fan-in (packages)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 14 |
| `tropoi.temporal.tendencies` | 10 |
| `tropoi.temporal` | 9 |
| `tropoi.spatial.states` | 8 |
| `tropoi.representation.archive` | 7 |
| `tropoi.representation.visual` | 6 |
| `tropoi.spatial.transforms` | 6 |
| `tropoi.spatial.terrain` | 5 |
| `tropoi.representation.diagnostics` | 4 |
| `tropoi.run` | 4 |

**Top 10 fan-out (packages)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.run.custom` | 11 |
| `tropoi.representation.visual` | 10 |
| `tropoi.run.bve` | 10 |
| `tropoi.run.pe` | 10 |
| `tropoi.cli` | 9 |
| `tropoi.run.swe` | 9 |
| `tropoi.run.radiative` | 7 |
| `tropoi.representation.archive` | 4 |
| `tropoi.run.held_suarez` | 4 |
| `tropoi.spatial` | 4 |

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
| `run` | `tropoi.run` | `tropoi.run`, `tropoi.run.bve`, `tropoi.run.custom`, `tropoi.run.held_suarez`, `tropoi.run.pe`, `tropoi.run.radiative`, `tropoi.run.swe` |
| `representation` | `tropoi.representation` | `tropoi.representation`, `tropoi.representation.archive`, `tropoi.representation.diagnostics`, `tropoi.representation.visual` |
| `temporal` | `tropoi.temporal` | `tropoi.temporal`, `tropoi.temporal.tendencies` |
| `spatial` | `tropoi.spatial` | `tropoi.spatial`, `tropoi.spatial.grids`, `tropoi.spatial.initialization`, `tropoi.spatial.operators`, `tropoi.spatial.states`, `tropoi.spatial.terrain`, `tropoi.spatial.transforms`, `tropoi.spatial.transforms.cuda` |
| `physics` | `tropoi.physics` | `tropoi.physics`, `tropoi.physics.radiation` |

Packages matched by no declared layer: none.

| Metric | layers |
|---|---|
| Nodes \|V\| | 6 |
| Directed edges \|E\| | 12 |
| Density | 0.4 |
| Weakly connected components | 1 |
| WCC sizes | 6 |
| Strongly connected components | 5 |
| Nontrivial SCCs | 1 |
| Nontrivial SCC sizes | 2 |
| Nodes in cycles | 2 |
| Fraction of nodes in cycles | 0.333333 |
| Sources (in-degree 0) | 1 |
| Sinks (out-degree 0) | 1 |
| Reciprocal pairs | 1 |
| Reciprocity | 0.166667 |
| In-degree min / median / max | 0 / 2 / 4 |
| In-degree mean | 2 |
| In-degree quartiles | 1.25, 2, 2.75 |
| In-degree zeros | 1 |
| Out-degree min / median / max | 0 / 2 / 4 |
| Out-degree mean | 2 |
| Out-degree quartiles | 1, 2, 3 |
| Out-degree zeros | 1 |
| Condensation nodes | 5 |
| Condensation edges | 8 |
| Condensation sources / sinks | 1 / 2 |
| Condensation longest path (edges) | 3 |

**Edges against the declared order (layer violations)**

| From | To | Crosses | Module edges | `TYPE_CHECKING` only |
|---|---|---|---|---|
| `tropoi.spatial.initialization` | `tropoi.temporal.tendencies` | spatial -> temporal | 2 | yes |

Module-level import edges involved: 2.

## Class graph

* Classes: 211
* Classes in at least one `inherits` edge (drawn in `class-inheritance.mmd`): 21; with neither an analysed base nor an analysed subclass: 190
* `inherits` edges between analysed classes: 14
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x6, `UserWarning` x1, `ValueError` x20, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `str` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 1892
* Edges by kind: `call` 1290, `constructor` 62, `instantiate` 492, `override` 6, `self` 328, `super` 4
* Call sites by resolution outcome: builtin 3674, external 2448, internal 2931, unresolved 3680

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/cli-capsule-command.mmd` | `tropoi.cli.capsule._command` | 4 | 14 | 21 | 0 | 14 |
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 27 | 28 | 6 | 45 |
| `calls/cli-inspect.mmd` | `tropoi.cli.main._cmd_inspect` | 4 | 32 | 42 | 2 | 43 |
| `calls/cli-list-presets.mmd` | `tropoi.cli.main._cmd_list_presets` | 4 | 1 | 0 | 0 | 1 |
| `calls/cli-list-scenarios.mmd` | `tropoi.cli.main._cmd_list_scenarios` | 4 | 1 | 0 | 0 | 1 |
| `calls/cli-plot.mmd` | `tropoi.cli.main._cmd_plot` | 4 | 57 | 76 | 4 | 71 |
| `calls/cli-recompile.mmd` | `tropoi.cli.main._cmd_recompile` | 4 | 6 | 5 | 0 | 6 |
| `calls/cli-run-bve.mmd` | `tropoi.cli.main._cmd_run_bve` | 4 | 45 | 60 | 13 | 208 |
| `calls/cli-run-pe.mmd` | `tropoi.cli.main._cmd_run_pe` | 4 | 56 | 65 | 19 | 170 |
| `calls/cli-run-swe.mmd` | `tropoi.cli.main._cmd_run_swe` | 4 | 82 | 119 | 15 | 210 |
| `calls/cli-terrain-run-prepare.mmd` | `tropoi.cli.terrain.run_prepare` | 4 | 45 | 62 | 4 | 89 |
| `calls/custom-initial-states.mmd` | `tropoi.run.custom.execute.run_initial_state`, `tropoi.run.custom.presets.initial_state_from_preset`, `tropoi.run.custom.prepare.prepare_bve_state`, `tropoi.run.custom.prepare.prepare_swe_state` | 4 | 133 | 209 | 12 | 244 |
| `calls/sky-hypothetical.mmd` | `tropoi.representation.visual.sky.write_sky_page`, `tropoi.representation.visual.sky.standalone_page` | 4 | 15 | 17 | 1 | 16 |
| `calls/sky-numpy-reference.mmd` | `tropoi.representation.visual.sky.radiance_table`, `tropoi.representation.visual.sky.star_path_column`, `tropoi.representation.visual.sky.optical_depth_diagnostic`, `tropoi.physics.column.thin_atmosphere_mass_kg`, `tropoi.physics.column.thin_atmosphere_surface_pressure_pa`, `tropoi.physics.rayleigh.composition_cross_section` | 4 | 11 | 8 | 0 | 11 |
| `calls/sky-saved-column.mmd` | `tropoi.temporal.simulation.Simulation.sky`, `tropoi.representation.archive.capsule.CapsuleStorage.sky`, `tropoi.representation.archive.held_suarez.HeldSuarezStorage.sky` | 4 | 32 | 36 | 2 | 34 |
| `calls/terrain-raster.mmd` | `tropoi.spatial.terrain.raster.read_elevation`, `tropoi.spatial.terrain.raster.encode_grey16` | 4 | 19 | 32 | 0 | 19 |

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
