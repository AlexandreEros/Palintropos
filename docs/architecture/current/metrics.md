# Architecture metrics: current

Descriptive graph measurements. No value is ranked as better or worse. Definitions are at the end; machine-readable values are in `metrics.json`.

## Module import graph

Nodes are analysed modules; a directed edge A -> B means A has at least one static import statement that resolves to module B. Two edge sets are measured:

* **all imports**: every static import statement, wherever it appears;
* **import-time**: only edges with at least one import statement at module scope (executed when the importing module is imported). Edges made only of function-scope or `if TYPE_CHECKING:` imports are dropped.

Edge counts by scope set: `function`: 195, `function+module`: 9, `function+type_checking`: 1, `module`: 315, `type_checking`: 6.

| Metric | all imports | import-time |
|---|---|---|
| Nodes \|V\| | 170 | 170 |
| Directed edges \|E\| | 526 | 324 |
| Density | 0.018308 | 0.011277 |
| Weakly connected components | 1 | 8 |
| WCC sizes | 170 | 155, 8, 2, 1, 1, 1, 1, 1 |
| Strongly connected components | 170 | 170 |
| Nontrivial SCCs | 0 | 0 |
| Nontrivial SCC sizes | - | - |
| Nodes in cycles | 0 | 0 |
| Fraction of nodes in cycles | 0 | 0 |
| Sources (in-degree 0) | 7 | 45 |
| Sinks (out-degree 0) | 45 | 59 |
| Reciprocal pairs | 0 | 0 |
| Reciprocity | 0 | 0 |
| In-degree min / median / max | 0 / 2 / 17 | 0 / 1 / 14 |
| In-degree mean | 3.094118 | 1.905882 |
| In-degree quartiles | 1, 2, 4 | 0, 1, 2 |
| In-degree zeros | 7 | 45 |
| Out-degree min / median / max | 0 / 2 / 23 | 0 / 1 / 9 |
| Out-degree mean | 3.094118 | 1.905882 |
| Out-degree quartiles | 0, 2, 5 | 0, 1, 3 |
| Out-degree zeros | 45 | 59 |
| Condensation nodes | 170 | 170 |
| Condensation edges | 526 | 324 |
| Condensation sources / sinks | 7 / 45 | 45 / 59 |
| Condensation longest path (edges) | 14 | 7 |

In-degree histogram (all imports) {degree: modules}: 0: 7, 1: 49, 2: 42, 3: 20, 4: 17, 5: 10, 6: 7, 7: 3, 8: 5, 9: 2, 10: 3, 11: 1, 12: 3, 17: 1

Out-degree histogram (all imports) {degree: modules}: 0: 45, 1: 28, 2: 27, 3: 13, 4: 13, 5: 12, 6: 4, 7: 8, 8: 6, 9: 3, 10: 5, 11: 3, 13: 1, 19: 1, 23: 1

**Top 10 fan-in (all imports)** (ties at cutoff: 2)

| Node | Degree |
|---|---|
| `tropoi.spatial.truncation` | 17 |
| `tropoi.physics.constants` | 12 |
| `tropoi.representation.archive.schema` | 12 |
| `tropoi.spatial.sigma_coordinate` | 12 |
| `tropoi.spatial.planet` | 11 |
| `tropoi.representation.visual.normalization` | 10 |
| `tropoi.spatial.specification` | 10 |
| `tropoi.spatial.states.primitive_equations` | 10 |
| `tropoi.spatial.laplacian` | 9 |
| `tropoi.temporal.integration` | 9 |

**Top 10 fan-out (all imports)** (ties at cutoff: 5)

| Node | Degree |
|---|---|
| `tropoi.cli.main` | 23 |
| `tropoi.run.radiative.model` | 19 |
| `tropoi.run.custom.execute` | 13 |
| `tropoi.representation.archive.capsule` | 11 |
| `tropoi.representation.visual.viewer_fields` | 11 |
| `tropoi.spatial.planet` | 11 |
| `tropoi.representation.visual.compose` | 10 |
| `tropoi.representation.visual.evaluate` | 10 |
| `tropoi.representation.visual.sky` | 10 |
| `tropoi.run.custom.prepare` | 10 |

**Nontrivial strongly connected components (all imports)**

None.

**Reciprocal pairs (all imports)**

None.

## Package-aggregated import graph

Every module is mapped to its containing package (a package's `__init__` to the package itself) and parallel edges merged; edges inside one package are not self-loops but are counted separately: 185 module edges stay inside their package.

| Metric | packages |
|---|---|
| Nodes \|V\| | 27 |
| Directed edges \|E\| | 114 |
| Density | 0.162393 |
| Weakly connected components | 1 |
| WCC sizes | 27 |
| Strongly connected components | 23 |
| Nontrivial SCCs | 4 |
| Nontrivial SCC sizes | 2, 2, 2, 2 |
| Nodes in cycles | 8 |
| Fraction of nodes in cycles | 0.296296 |
| Sources (in-degree 0) | 3 |
| Sinks (out-degree 0) | 6 |
| Reciprocal pairs | 4 |
| Reciprocity | 0.070175 |
| In-degree min / median / max | 0 / 4 / 15 |
| In-degree mean | 4.222222 |
| In-degree quartiles | 1.5, 4, 6 |
| In-degree zeros | 3 |
| Out-degree min / median / max | 0 / 4 / 11 |
| Out-degree mean | 4.222222 |
| Out-degree quartiles | 1, 4, 7 |
| Out-degree zeros | 6 |
| Condensation nodes | 23 |
| Condensation edges | 81 |
| Condensation sources / sinks | 3 / 6 |
| Condensation longest path (edges) | 11 |

**Top 10 fan-in (packages)** (ties at cutoff: 3)

| Node | Degree |
|---|---|
| `tropoi.spatial` | 15 |
| `tropoi.temporal` | 10 |
| `tropoi.temporal.tendencies` | 10 |
| `tropoi.spatial.grids` | 8 |
| `tropoi.spatial.states` | 8 |
| `tropoi.representation.archive` | 7 |
| `tropoi.representation.visual` | 6 |
| `tropoi.spatial.transforms` | 6 |
| `tropoi.physics` | 5 |
| `tropoi.run` | 5 |

**Top 10 fan-out (packages)** (ties at cutoff: 3)

| Node | Degree |
|---|---|
| `tropoi.run.custom` | 11 |
| `tropoi.representation.visual` | 10 |
| `tropoi.run.bve` | 10 |
| `tropoi.run.pe` | 10 |
| `tropoi.cli` | 9 |
| `tropoi.run.swe` | 9 |
| `tropoi.run.radiative` | 7 |
| `tropoi.temporal` | 7 |
| `tropoi.run.airless` | 6 |
| `tropoi.representation.archive` | 5 |

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
| `run` | `tropoi.run` | `tropoi.run`, `tropoi.run.airless`, `tropoi.run.bve`, `tropoi.run.custom`, `tropoi.run.held_suarez`, `tropoi.run.pe`, `tropoi.run.radiative`, `tropoi.run.swe` |
| `representation` | `tropoi.representation` | `tropoi.representation`, `tropoi.representation.archive`, `tropoi.representation.diagnostics`, `tropoi.representation.visual` |
| `temporal` | `tropoi.temporal` | `tropoi.temporal`, `tropoi.temporal.native`, `tropoi.temporal.tendencies` |
| `spatial` | `tropoi.spatial` | `tropoi.spatial`, `tropoi.spatial.grids`, `tropoi.spatial.initialization`, `tropoi.spatial.operators`, `tropoi.spatial.states`, `tropoi.spatial.terrain`, `tropoi.spatial.transforms`, `tropoi.spatial.transforms.cuda` |
| `physics` | `tropoi.physics` | `tropoi.physics`, `tropoi.physics.radiation` |
| `numerics` | `tropoi.numerics` | `tropoi.numerics` |

Packages matched by no declared layer: none.

| Metric | layers |
|---|---|
| Nodes \|V\| | 7 |
| Directed edges \|E\| | 14 |
| Density | 0.333333 |
| Weakly connected components | 1 |
| WCC sizes | 7 |
| Strongly connected components | 6 |
| Nontrivial SCCs | 1 |
| Nontrivial SCC sizes | 2 |
| Nodes in cycles | 2 |
| Fraction of nodes in cycles | 0.285714 |
| Sources (in-degree 0) | 1 |
| Sinks (out-degree 0) | 2 |
| Reciprocal pairs | 1 |
| Reciprocity | 0.142857 |
| In-degree min / median / max | 0 / 2 / 4 |
| In-degree mean | 2 |
| In-degree quartiles | 1, 2, 3 |
| In-degree zeros | 1 |
| Out-degree min / median / max | 0 / 3 / 4 |
| Out-degree mean | 2 |
| Out-degree quartiles | 0.5, 3, 3 |
| Out-degree zeros | 2 |
| Condensation nodes | 6 |
| Condensation edges | 10 |
| Condensation sources / sinks | 1 / 2 |
| Condensation longest path (edges) | 4 |

**Edges against the declared order (layer violations)**

| From | To | Crosses | Module edges | `TYPE_CHECKING` only |
|---|---|---|---|---|
| `tropoi.spatial.initialization` | `tropoi.temporal.tendencies` | spatial -> temporal | 2 | yes |

Module-level import edges involved: 2.

## Class graph

* Classes: 246
* Classes in at least one `inherits` edge (drawn in `class-inheritance.mmd`): 21; with neither an analysed base nor an analysed subclass: 225
* `inherits` edges between analysed classes: 14
* `nests` edges (class defined in a class body): 0
* Bases outside the analysed package: `ArithmeticError` x1, `RuntimeError` x9, `UserWarning` x1, `ValueError` x22, `abc.ABC` x2, `collections.abc.Mapping` x1, `ctypes.Structure` x3, `enum.Enum` x2, `http.server.BaseHTTPRequestHandler` x1, `http.server.ThreadingHTTPServer` x1, `str` x2, `typing.NamedTuple` x2, `typing.Protocol` x3

## Call graph

* Nodes (functions, methods, classes): 2224
* Edges by kind: `call` 1543, `constructor` 74, `instantiate` 573, `override` 6, `self` 405, `super` 4
* Call sites by resolution outcome: builtin 4513, external 2918, internal 3434, unresolved 4680

| Scoped graph | Roots | Depth | Nodes | Edges | Truncated nodes | Reachable (unbounded) |
|---|---|---|---|---|---|---|
| `calls/airless-mercury.mmd` | `tropoi.run.airless.config.mercury_vpw99`, `tropoi.run.airless.config.mercury_hayne_analogue`, `tropoi.run.airless.experiment.load_tier2` | 4 | 12 | 11 | 0 | 12 |
| `calls/cli-capsule-command.mmd` | `tropoi.cli.capsule._command` | 4 | 14 | 21 | 0 | 14 |
| `calls/cli-gen.mmd` | `tropoi.cli.main._cmd_gen` | 4 | 28 | 29 | 7 | 56 |
| `calls/cli-inspect.mmd` | `tropoi.cli.main._cmd_inspect` | 4 | 47 | 63 | 4 | 66 |
| `calls/cli-list-presets.mmd` | `tropoi.cli.main._cmd_list_presets` | 4 | 1 | 0 | 0 | 1 |
| `calls/cli-list-scenarios.mmd` | `tropoi.cli.main._cmd_list_scenarios` | 4 | 1 | 0 | 0 | 1 |
| `calls/cli-plot.mmd` | `tropoi.cli.main._cmd_plot` | 4 | 50 | 59 | 3 | 80 |
| `calls/cli-recompile.mmd` | `tropoi.cli.main._cmd_recompile` | 4 | 6 | 5 | 0 | 6 |
| `calls/cli-run-bve.mmd` | `tropoi.cli.main._cmd_run_bve` | 4 | 45 | 60 | 13 | 221 |
| `calls/cli-run-pe.mmd` | `tropoi.cli.main._cmd_run_pe` | 4 | 57 | 66 | 19 | 183 |
| `calls/cli-run-swe.mmd` | `tropoi.cli.main._cmd_run_swe` | 4 | 83 | 120 | 15 | 223 |
| `calls/cli-terrain-run-prepare.mmd` | `tropoi.cli.terrain.run_prepare` | 4 | 46 | 63 | 4 | 101 |
| `calls/cli-view.mmd` | `tropoi.cli.main._cmd_view` | 4 | 9 | 8 | 2 | 76 |
| `calls/custom-initial-states.mmd` | `tropoi.run.custom.execute.run_initial_state`, `tropoi.run.custom.presets.initial_state_from_preset`, `tropoi.run.custom.prepare.prepare_bve_state`, `tropoi.run.custom.prepare.prepare_swe_state` | 4 | 135 | 212 | 12 | 257 |
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
