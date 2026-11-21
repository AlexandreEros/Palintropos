"""Read daily PE and slab snapshots in radiative experiment format 1."""
from dataclasses import replace
from pathlib import Path
import re

import numpy as np

from tropoi.representation.archive.local_resource import resource
from tropoi.representation.archive.pe_context import PEColumnContext
from tropoi.representation.archive.schema import state_schema_for
from tropoi.spatial.sigma_coordinate import SigmaGrid

FORMAT = "palintropos.radiative.config/1"


def _sigma(v):
    kind = v["kind"]
    if kind == "interfaces":
        return SigmaGrid(tuple(v["sigma"]))
    if kind == "uniform":
        return SigmaGrid.uniform(v["nlev"])
    if kind == "frierson":
        return SigmaGrid.frierson(v["nlev"])
    if kind == "stretched":
        return SigmaGrid.stretched(v["nlev"], v["sigma_top"], v.get("linear_scale"))
    raise ValueError(f"Unsupported radiative vertical coordinate {kind!r}")


class RadiativeStorage:
    solver = "pe"
    format = FORMAT

    def __init__(self, root, document, *, mmap=True):
        self.run_dir = Path(root)
        self._mmap = mmap
        c = self.run_config = document["config"]
        if c["schema"] != FORMAT:
            raise ValueError(f"Unsupported radiative schema {c['schema']}")
        sigma = _sigma(c["vertical"])
        geometry = dict(grid="latlon", nlat=c["nlat"], nlon=c["nlon"],
                        dynamical_radius_m=c["radius"], latitude="Gaussian north to south",
                        longitude="360*j/nlon degrees east", slab_order="latitude, longitude")
        self.context = PEColumnContext(c["experiment_id"], FORMAT, c["l_max"],
                                       tuple(sigma.interfaces), c["gas_constant"], c["cp"],
                                       c["radius"], c["gravity"], None, "flat",
                                       {"source": "radiative config.json"})
        schema = state_schema_for("pe", dict(lmax=c["l_max"], grid="latlon",
                                  sigma_interfaces=sigma.interfaces),
                                  provenance={"source": FORMAT})
        self.schema = replace(schema, geometry=geometry, environment=dict(c),
                              coefficient_file="states/state-d*.npy", time_file=None)
        self.field_specs = self.schema.fields
        self.level_values = self.schema.level_values
        self.dtypes = {}
        self.states = self._index("state", self.schema.frame_shape, "c")
        self.surfaces = self._index("surface", (c["nlat"] * c["nlon"],), "f")
        if self.surfaces.keys() - self.states.keys():
            raise ValueError("Slab snapshot without a matching atmospheric day")
        self.days = tuple(sorted(self.states))
        self.times = np.array(self.days, dtype=float) * 86400
        self.times.flags.writeable = False
        self.metadata = dict(run_id=c["experiment_id"], format=FORMAT, solver="pe",
                             git=document.get("code"), run_config=c,
                             config_sha256=document.get("config_sha256"),
                             physics_enabled=document.get("physics"),
                             physics_datasets=document.get("physics_datasets"),
                             schema=self.schema.to_manifest_dict(), snapshot_count=len(self.days),
                             provenance={"time_axis": {"source": "state filenames",
                                                       "rule": "day * 86400 s"}})

    def _index(self, prefix, shape, dtype_kind):
        result = {}
        folder = resource(self.run_dir, "states")
        for entry in sorted(folder.glob(f"{prefix}-d*.npy")):
            match = re.fullmatch(fr"{prefix}-d([0-9]+)\.npy", entry.name)
            if not match:
                raise ValueError(f"Invalid saved day in {entry.name}")
            day = int(match[1])
            name = f"states/{entry.name}"
            array = np.load(resource(self.run_dir, name), mmap_mode="r", allow_pickle=False)
            if day in result or array.shape != shape or array.dtype.kind != dtype_kind:
                raise ValueError(f"Invalid or duplicate snapshot {name}: {array.shape}, {array.dtype}")
            dtype = self.dtypes.setdefault(prefix, str(array.dtype))
            if str(array.dtype) != dtype:
                raise ValueError(f"Inconsistent precision in {name}")
            result[day] = name
        return result

    def _read(self, name):
        array = np.load(resource(self.run_dir, name),
                        mmap_mode="r" if self._mmap else None, allow_pickle=False)
        array.flags.writeable = False
        return array.view()

    def frame(self, index):
        return self._read(self.states[self.days[index]])

    def surface(self, index):
        day = self.days[index]
        if day not in self.surfaces:
            raise ValueError(f"No slab temperature saved for day {day}")
        return self._read(self.surfaces[day]).reshape(self.run_config["nlat"], self.run_config["nlon"])

    def pe_context(self, *, allow_cuda=False):
        return self.context

    def quantities(self):
        # Simulation.quantities describes its plotting API. Discovery has its own manifest.
        return []

    def plot_simulation(self, *args, **kwargs):
        raise ValueError("Radiative snapshot plotting is unavailable")

    plot_snapshot = plot_simulation
