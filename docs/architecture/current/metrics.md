# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 184, `function+module`: 7, `function+type_checking`: 1, `module`: 295, `type_checking`: 6.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 157 | 157 |
| Directed edges \|E\| | 493 | 302 |
| Density | 0.020129 | 0.012331 |
| Weakly connected components | 1 | 7 |
| WCC sizes | 157 | 144, 8, 1, 1, 1, 1, 1 |
| Strongly connected components | 157 | 157 |
| Nontrivial SCCs | 0 | 0 |
| Nontrivial SCC sizes | - | - |
| Nodes in cycles | 0 | 0 |
| Fraction of nodes in cycles | 0 | 0 |
| Sources (in-degree 0) | 5 | 41 |
| Sinks (out-degree 0) | 40 | 53 |
| Reciprocal pairs | 0 | 0 |
| Reciprocity | 0 | 0 |
| In-degree min / median / max | 0 / 2 / 17 | 0 / 1 / 14 |
| In-degree mean | 3.140127 | 1.923567 |
| In-degree quartiles | 1, 2, 4 | 0, 1, 2 |
| In-degree zeros | 5 | 41 |
| Out-degree min / median / max | 0 / 2 / 23 | 0 / 1 / 9 |
| Out-degree mean | 3.140127 | 1.923567 |
| Out-degree quartiles | 0, 2, 5 | 0, 1, 3 |
| Out-degree zeros | 40 | 53 |
| Condensation nodes | 157 | 157 |
| Condensation edges | 493 | 302 |
| Condensation sources / sinks | 5 / 40 | 41 / 53 |
| Condensation longest path (edges) | 14 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 5, 1: 44, 2: 41, 3: 19, 4: 14, 5: 10, 6: 6, 7: 4, 8: 5, 9: 2, 10: 3, 11: 1, 12: 2, 17: 1

Out-degree histogram (all imports) {degree: modules}: 0: 40, 1: 29, 2: 22, 3: 14, 4: 10, 5: 11, 6: 5, 7: 7, 8: 5, 9: 3, 10: 6, 11: 2, 13: 1, 19: 1, 23: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.spatial.truncation` | 17 |
| `tropoi.representation.archive.schema` | 12 |
| `tropoi.spatial.sigma_coordinate` | 12 |
| `tropoi.spatial.planet` | 11 |
| `tropoi.representation.visual.normalization` | 10 |
| `tropoi.spatial.specification` | 10 |
| `tropoi.spatial.states.primitive_equations` | 10 |
| `tropoi.spatial.laplacian` | 9 |
| `tropoi.temporal.integration` | 9 |
| `tropoi.representation.archive.initial_state` | 8 |

**Top 10 fan-out (all imports)** (ties at cutoff: 6)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 23 |
| `tropoi.run.radiative.model` | 19 |
| `tropoi.run.custom.execute` | 13 |
| `tropoi.representation.archive.capsule` | 11 |
| `tropoi.spatial.planet` | 11 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.representation.visual.sky` | 10 |
| `tropoi.representation.visual.viewer_fields` | 10 |
| `tropoi.run.custom.prepare` | 10 |

**Nontrivial strongly connected components (all imports)**

None.

**Reciprocal pairs (all imports)**

None.

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 175 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 24 |
| Directed edges \|E\| | 102 |
| Density | 0.184783 |
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
| Reciprocity | 0.078431 |
| In-degree min / median / max | 0 / 4 / 14 |
| In-degree mean | 4.25 |
| In-degree quartiles | 2, 4, 6 |
| In-degree zeros | 2 |
| Out-degree min / median / max | 0 / 3.5 / 11 |
| Out-degree mean | 4.25 |
| Out-degree quartiles | 1, 3.5, 7.5 |
| Out-degree zeros | 4 |
| Condensation nodes | 20 |
| Condensation edges | 72 |
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

**Top 10 fan-out (packages)** (ties at cutoff: 4)

| Node | Degree |
|---|---|
| `tropoi.run.custom` | 11 |
| `tropoi.representation.visual` | 10 |
| `tropoi.run.bve` | 10 |
| `tropoi.run.pe` | 10 |
| `tropoi.cli` | 9 |
| `tropoi.run.swe` | 9 |
| `tropoi.run.radiative` | 7 |
| `tropoi.temporal.tendencies` | 5 |
| `tropoi.representation.archive` | 4 |
| `tropoi.run.held_suarez` | 4 |

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
| Directed edges \|E\| | 13 |
| Density | 0.433333 |
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
| Reciprocity | 0.153846 |
| In-degree min / median / max | 0 / 2.5 / 4 |
| In-degree mean | 2.166667 |
| In-degree quartiles | 1.25, 2.5, 3 |
| In-degree zeros | 1 |
| Out-degree min / median / max | 0 / 2.5 / 4 |
| Out-degree mean | 2.166667 |
| Out-degree quartiles | 1.25, 2.5, 3 |
| Out-degree zeros | 1 |
| Condensation nodes | 5 |
| Condensation edges | 9 |
| Condensation sources / sinks | 1 / 1 |
| Condensation longest path (edges) | 4 |

**Edges against the declared order (layer violations)**

| From | To | Crosses | Module edges | `TYPE_CHECKING` only |
|---|---|---|---|---|
| `tropoi.spatial.initialization` | `tropoi.temporal.tendencies` | spatial -> temporal | 2 | yes |

Module-level import edges involved: 2.

## Class graph

* Classes: 226
* Classes in at least one `inherits` edge (drawn in `class-inheritance.mmd`): 21; with neither an analysed base nor an analysed subclass: 205
* `inherits` edges between analysed classes: 14
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `RuntimeError` x7, `UserWarning` x1, `ValueError` x22, `abc.ABC` x2, `collections.abc.Mapping` x1, `enum.Enum` x2, `http.server.BaseHTTPRequestHandler` x1, `http.server.ThreadingHTTPServer` x1, `str` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 2037
* Edges by kind: `call` 1423, `constructor` 71, `instantiate` 531, `override` 6, `self` 362, `super` 4
* Call sites by resolution outcome: builtin 4062, external 2664, internal 3183, unresolved 4203

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/cli-capsule-command.mmd` | `tropoi.cli.capsule._command` | 4 | 14 | 21 | 0 | 14 |
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 28 | 29 | 6 | 53 |
| `calls/cli-inspect.mmd` | `tropoi.cli.main._cmd_inspect` | 4 | 45 | 61 | 3 | 63 |
| `calls/cli-list-presets.mmd` | `tropoi.cli.main._cmd_list_presets` | 4 | 1 | 0 | 0 | 1 |
| `calls/cli-list-scenarios.mmd` | `tropoi.cli.main._cmd_list_scenarios` | 4 | 1 | 0 | 0 | 1 |
| `calls/cli-plot.mmd` | `tropoi.cli.main._cmd_plot` | 4 | 50 | 59 | 3 | 80 |
| `calls/cli-recompile.mmd` | `tropoi.cli.main._cmd_recompile` | 4 | 6 | 5 | 0 | 6 |
| `calls/cli-run-bve.mmd` | `tropoi.cli.main._cmd_run_bve` | 4 | 45 | 60 | 13 | 218 |
| `calls/cli-run-pe.mmd` | `tropoi.cli.main._cmd_run_pe` | 4 | 57 | 66 | 19 | 180 |
| `calls/cli-run-swe.mmd` | `tropoi.cli.main._cmd_run_swe` | 4 | 83 | 120 | 15 | 220 |
| `calls/cli-terrain-run-prepare.mmd` | `tropoi.cli.terrain.run_prepare` | 4 | 46 | 63 | 4 | 98 |
| `calls/cli-view.mmd` | `tropoi.cli.main._cmd_view` | 4 | 9 | 8 | 2 | 64 |
| `calls/custom-initial-states.mmd` | `tropoi.run.custom.execute.run_initial_state`, `tropoi.run.custom.presets.initial_state_from_preset`, `tropoi.run.custom.prepare.prepare_bve_state`, `tropoi.run.custom.prepare.prepare_swe_state` | 4 | 135 | 212 | 12 | 254 |
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
