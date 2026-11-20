"""Viewer manifest v1 discovery. Local directory only, no listener or writes."""
import json
from pathlib import Path

from tropoi.representation.archive.capsule import open_storage
from tropoi.representation.archive.local_resource import resource
from tropoi.representation.archive.radiative import FORMAT, RadiativeStorage
from tropoi.representation.archive.schema import COEFFICIENT_FILES, TIME_FILES, SchemaError, StateSchema
from tropoi.representation.archive.viewer_diagnostics import diagnostics
from tropoi.representation.visual.quantities import viewer_presentation


def _derived_fields(fields, physics, geometry, resources):
    """Advertise only recipes whose scientific inputs can be resolved."""
    from tropoi.representation.visual.quantities import quantity

    def add(name, inputs, **extra):
        q = quantity(name)
        fields[name] = dict(kind=q.kind, recipe=name + "/1", inputs=inputs,
                            definition=q.definition, units=q.units, **extra)

    def add_viewer_field(name, inputs, units, **extra):
        fields[name] = dict(kind="derived", recipe=name + "/1", inputs=inputs, units=units,
                            definition=viewer_presentation(name, {})["definition"], **extra)

    if "zeta" in fields and (geometry.get("reference_radius_m") or geometry.get("dynamical_radius_m")):
        inputs = ["zeta"] + (["delta"] if "delta" in fields else [])
        add("wind", inputs, levels=fields["zeta"]["levels"], basis="east, north")
        add("wind_speed", ["wind"], levels=fields["zeta"]["levels"])
        for component in ("u", "v"):
            add_viewer_field(component, ["wind"], "m s^-1", levels=fields["zeta"]["levels"])
    terrain = physics.get("topography")
    if terrain == "williamson5_cone" or "terrain" in resources:
        add("terrain", ["resources.terrain" if "terrain" in resources else "physics.topography"],
            cadence="static")
    if "phi" in fields and physics.get("gravity") and physics.get("mean_depth_m"):
        add("layer_depth", ["phi", "physics.gravity", "physics.mean_depth_m"])
        if "terrain" in fields or terrain == "flat":
            add("free_surface_height", ["layer_depth"] + (["terrain"] if "terrain" in fields else []))
            add_viewer_field("free_surface_geopotential", ["free_surface_height", "physics.gravity"], "m^2 s^-2")
    if "ln_ps" in fields and fields.get("temperature", {}).get("levels"):
        add_viewer_field("pressure", ["ln_ps", "axes.vertical"], "Pa", levels=True)
        gas_constant = "r_dry" if physics.get("r_dry") else "gas_constant"
        if (physics.get("gravity") and physics.get(gas_constant)
                and (terrain in (None, "flat") or "terrain" in fields)):
            inputs = ["temperature", "pressure", "physics.gravity", "physics." + gas_constant]
            if "terrain" in fields:
                inputs.append("terrain")
            add_viewer_field("geopotential", inputs, "m^2 s^-2", levels=True)
            add_viewer_field("geopotential_height", ["geopotential", "physics.gravity"], "m", levels=True)


def _operations(fields, vertical, physics, diagnostics):
    """Resolve scientific selections from quantities and coordinates."""
    names = set(fields)
    ops = {}
    if names:
        ops["map"] = sorted(names)
        ops["globe"] = sorted(names)
    if "wind" in names:
        ops["wind"] = ["wind"]
    levels = [k for k, f in fields.items() if f.get("levels")]
    # Sigma and pressure surfaces interpolate in columns that carry Z = Φ/g,
    # so they need the run's gravity. Native levels do not.
    if vertical and levels and physics.get("gravity"):
        ops.update(column=levels, sigma=levels)
        if "ln_ps" in names:
            reconstructed = [key for key in levels if key in
                             {"temperature", "u", "v", "wind", "wind_speed",
                              "geopotential_height", "geopotential", "pressure"}]
            ops["pressure"] = reconstructed
            if physics.get("gravity") and (physics.get("r_dry") or physics.get("gas_constant")):
                ops["geopotential_height"] = reconstructed
                # Columns, sections and native-grid zonal reductions need the
                # full column and a pure-sigma vertical from 0 to 1.
                interfaces = vertical.get("interfaces") or []
                if (interfaces and interfaces[0] == 0 and interfaces[-1] == 1
                        and {"temperature", "zeta", "delta"} <= names):
                    ops["structure"] = ["column", "meridional_section", "zonal_section",
                                        "zonal_mean", "mass_streamfunction"]
    for kind in ("series", "zonal_section"):
        keys = [key for key, product in diagnostics.items() if product["kind"] == kind]
        if keys:
            ops[kind] = keys
    return ops


def viewer_manifest(directory):
    """Describe one explicit archive. All resource references are archive-relative."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected a capsule directory: {root}")
    # Do not follow latest_run pointers from the selected filesystem boundary.
    config = resource(root, "config.json")
    manifest = resource(root, "manifest.json")
    if not (config.is_file() or manifest.is_file()):
        raise ValueError(f"No capsule metadata in {root}")
    for name in (*COEFFICIENT_FILES.values(), *TIME_FILES.values(), "states", "statistics.npz", "series.json", "diagnostics/timeseries.csv"):
        resource(root, name)
    for path in (root / "states").glob("state-d*.npy"):
        resource(root, "states/" + path.name)
    source = dict(directory=root.name, reader="viewer-manifest/1",
                  receipt="SHA256SUMS" if resource(root, "SHA256SUMS").is_file() else None)
    output = dict(schema="palintropos.viewer/1.0", source=source, resources={}, fields={},
                  axes={}, geometry={}, physics={}, diagnostics={}, time_axes={}, operations={},
                  unavailable={})
    if config.is_file() and not manifest.is_file():
        document = json.loads(config.read_text(encoding="utf-8"))
        version = document.get("config", {}).get("schema") if isinstance(document.get("config"), dict) else None
        if version and version.startswith("palintropos.radiative.config/") and version != FORMAT:
            source.update(format=version, git=document.get("code"))
            output["unavailable"]["snapshots"] = f"Unsupported radiative schema {version}"
            return output
    # An unreadable state convention should not require even a known coefficient filename.
    if manifest.is_file():
        header = json.loads(manifest.read_text(encoding="utf-8"))
        if header.get("state_schema") is not None:
            try:
                declared = StateSchema.from_manifest_dict(header["state_schema"])
            except SchemaError as err:
                source.update(run_id=header.get("run_id"), git=header.get("git"), format="canonical")
                output["unavailable"]["snapshots"] = str(err)
                return output
            resource(root, declared.coefficient_file)
            if declared.time_file:
                resource(root, declared.time_file)
    storage = open_storage(root)
    meta = storage.metadata
    source.update(run_id=meta.get("run_id") or root.name, git=meta.get("git"),
                  config_sha256=meta.get("config_sha256"), solver=storage.solver,
                  format=meta.get("format", "canonical"), provenance=meta["provenance"])
    schema = meta["schema"]
    if not isinstance(schema, dict):
        output["unavailable"]["snapshots"] = schema
        return output
    vertical = schema["vertical"]
    output.update(geometry=schema["geometry"], physics=schema["environment"],
                  axes=dict(support=schema["support"], vertical=vertical,
                            storage=schema["storage_axes"], normalization=schema["normalization"]))
    output["time_axes"]["snapshots"] = dict(values=storage.times.tolist(), units="s",
                                            provenance=meta["provenance"]["time_axis"])
    geometry = output["geometry"]
    if (geometry.get("planet") == "from_earth_like" and geometry.get("radius_earth_units")
            and geometry.get("day_hours") and not geometry.get("dynamical_radius_m")):
        from tropoi.spatial.environment import PlanetaryParameters
        planet = PlanetaryParameters.from_earth_like(radius_earth_units=geometry["radius_earth_units"],
                                                     day_hours=geometry["day_hours"])
        geometry["dynamical_radius_m"] = float(planet.radius)
    if isinstance(storage, RadiativeStorage):
        source["physics_datasets"] = meta["physics_datasets"]
        output["physics"] = dict(output["physics"], enabled=meta["physics_enabled"])
    products, axes = diagnostics(root, vertical, storage.run_config)
    output["diagnostics"] = products
    output["time_axes"].update(axes)
    if meta.get("diagnostic_definitions"):
        output["diagnostic_definitions"] = meta["diagnostic_definitions"]
    if len(storage.times):
        index = dict(layout=schema["layout"], frame_shape=list(storage.schema.frame_shape))
        if isinstance(storage, RadiativeStorage):
            index["dtype"] = storage.dtypes["state"]
            index["paths"] = [storage.states[d] for d in storage.days]
            if storage.surfaces:
                days = sorted(storage.surfaces)
                output["resources"]["slab"] = dict(dtype=storage.dtypes["surface"], paths=[storage.surfaces[d] for d in days],
                                                   frame_shape=[storage.run_config["nlat"] * storage.run_config["nlon"]])
                output["time_axes"]["slab"] = dict(values=[d*86400 for d in days], units="s",
                                                   source="surface filenames")
                output["fields"]["surface_temperature"] = dict(resource="slab", units="K",
                    dimensions=["latitude", "longitude"], time_axis="slab", description="slab surface temperature")
        elif hasattr(storage, "state_path"):
            index["paths"] = [storage.state_path(i).relative_to(root).as_posix() for i in range(len(storage.times))]
        else:
            index.update(path=schema["coefficient_file"], dtype=str(storage.coefficients.dtype))
        output["resources"]["state"] = index
        for field in schema["fields"]:
            output["fields"][field["name"]] = dict(field, resource="state", time_axis="snapshots")
    else:
        output["unavailable"]["snapshots"] = "No saved atmospheric states in this archive"
    if storage.run_config.get("retained_truncation") is not None:
        cut = storage.run_config["retained_truncation"]
        output["axes"]["support"] = dict(output["axes"]["support"], product_truncation_cut=cut,
                                         retained_truncation=cut)
    if storage.run_config.get("terrain_digest"):
        terrain = "terrain/topography_elevation_lm.npy"
        if resource(root, terrain).is_file() and resource(root, "terrain/terrain.json").is_file():
            output["resources"]["terrain"] = dict(path=terrain, metadata="terrain/terrain.json",
                                                  digest=storage.run_config["terrain_digest"])
        else:
            output["unavailable"]["terrain"] = "Prepared terrain is missing"
    if len(storage.times):
        _derived_fields(output["fields"], output["physics"], output["geometry"], output["resources"])
    if "terrain" in output["fields"]:
        for prefix in ("w5_cone", "mountain"):
            lat = output["physics"].get(prefix + "_lat_deg")
            lon = output["physics"].get(prefix + "_lon_deg")
            if lat is not None and lon is not None:
                output["geometry"]["initial_center"] = dict(latitude=lat, longitude=lon)
                break
    for key, field in output["fields"].items():
        field["presentation"] = viewer_presentation(key, field)
    for key, product in products.items():
        product["presentation"] = viewer_presentation(key, product, recorded=True)
    output["operations"] = _operations(output["fields"], vertical, output["physics"], products)
    if "terrain" in output["unavailable"]:
        output["operations"].pop("geopotential_height", None)
    if (any(field.get("levels") for field in output["fields"].values())
            and not output["physics"].get("gravity")):
        output["unavailable"]["vertical_surfaces"] = (
            "The run records no gravity, so geopotential height, sigma and pressure surfaces "
            "and vertical structure are undefined; native levels remain available")
    structure = output["operations"].get("structure")
    if structure and output["geometry"].get("grid") != "latlon":
        # Zonal reductions use Gauss–Legendre latitudes and equally spaced longitudes.
        output["operations"]["structure"] = [op for op in structure if op not in
                                             ("zonal_mean", "mass_streamfunction")]
        output["unavailable"]["zonal_mean"] = "The state grid has no Gaussian latitude–longitude quadrature"
    if isinstance(storage, RadiativeStorage) and storage.run_config.get("surface") is not None:
        missing = sorted(set(storage.states) - storage.surfaces.keys())
        if missing:
            output["unavailable"]["slab_days"] = missing
    return output
