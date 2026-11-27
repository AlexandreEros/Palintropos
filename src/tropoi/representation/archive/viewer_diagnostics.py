"""Inventory recorded diagnostics independently of instantaneous snapshots."""
import csv
import json

import numpy as np

from tropoi.representation.archive.local_resource import resource


_UNITS = {
    **dict.fromkeys(("mean_T", "t_min", "t_max", "mean_surface_temperature", "T"), "K"),
    **dict.fromkeys(("max_abs_u", "max_wind_bottom", "u", "v"), "m s^-1"),
    **dict.fromkeys(("mean_ps", "energy_mean_surface_pressure"), "Pa"),
    **dict.fromkeys(("toa_net_down", "olr", "absorbed_stellar", "bottom_net_down", "sensible",
                     "interior", "column_energy_tendency", "surface_net", "energy_rate",
                     "energy_residual"), "W m^-2"),
    **dict.fromkeys(("energy_total", "energy_surface_potential", "energy_kinetic",
                     "energy_enthalpy"), "J m^-2"),
    **dict.fromkeys(("ke", "usus", "usvs", "u_sq"), "m^2 s^-2"),
    "vsTs": "K m s^-1", "TsTs": "K^2", "T_sq": "K^2", "comp_mode": "1",
    "mean_lnps": "ln(Pa)",
}


def diagnostics(root, vertical, cfg):
    products, axes = {}, {}
    path = resource(root, "diagnostics/timeseries.csv")
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            columns = next(csv.reader(stream), [])
        products["timeseries"] = dict(path="diagnostics/timeseries.csv", columns=columns,
                                      time_axis="steps", kind="series")
        axes["steps"] = dict(path="diagnostics/timeseries.csv", column="time_s", units="s")
    path = resource(root, "series.json")
    if path.is_file():
        series = json.loads(path.read_text(encoding="utf-8"))
        days = np.asarray(series["day"], dtype=float)
        if days.ndim != 1 or not np.isfinite(days).all() or np.any(np.diff(days) <= 0):
            raise ValueError("Invalid daily diagnostic axis")
        axes["daily"] = dict(values=(days * 86400).tolist(), units="s", source="series.json:day")
        for key, values in series.items():
            if key == "day":
                continue
            # Energy changes require a preceding sample and start on day one.
            offset = len(days) - len(values)
            if offset and not (offset == 1 and key in ("energy_rate", "energy_residual")):
                raise ValueError(f"Ambiguous diagnostic time axis for {key}")
            products[key] = dict(kind="series", path="series.json", member=key,
                                 time_axis="daily", start_index=offset, units=_UNITS.get(key))
    for name in ("means.npz", "statistics.npz"):
        path = resource(root, name)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            # Read only coordinate/count members. Scientific members are indexed by name.
            if vertical:
                for key, axis in (("sigma", "full_levels"), ("sigma_interfaces", "interfaces")):
                    if key in archive:
                        actual = archive[key]
                        expected = vertical[axis]
                        if actual.shape != (len(expected),) or not np.allclose(
                                actual, expected, rtol=1e-12, atol=1e-14):
                            raise ValueError(f"{name}:{key} disagrees with configured vertical grid")
            if "lat" in archive:
                lat = archive["lat"]
                from tropoi.spatial.grids.quadrature import gauss_latitudes
                expected_lat = np.rad2deg(gauss_latitudes(cfg["nlat"]))
                if lat.shape != expected_lat.shape or not np.allclose(lat, expected_lat, atol=1e-8):
                    raise ValueError(f"{name}:lat disagrees with configured Gaussian grid")
            if name == "means.npz":
                count = int(archive["count"])
                if count < 0:
                    raise ValueError("Invalid aggregate sample count")
                start = cfg["days"] - cfg["mean_last_days"] if cfg["mean_last_days"] else cfg["days"] // 2
                members = [k for k in ("u", "v", "T", "usvs", "vsTs") if k in archive and count]
                info = dict(count=count, configured_window_days=[start, cfg["days"]],
                            interval="open-left, closed-right", weighting="equal daily samples")
            else:
                counts = archive["counts"].tolist()
                members = [k for k in archive.files if k.startswith("block_") and k != "block_ke_spectrum"]
                info = dict(block_counts=counts)
            for member in members:
                products[f"zonal.{member}"] = dict(kind="zonal_section", path=name,
                    member=member, axes=(["block"] if name == "statistics.npz" else []) + ["latitude", "level"],
                    units=_UNITS.get(member.removeprefix("block_")), coordinates=["lat", "sigma"], **info)
    return products, axes
