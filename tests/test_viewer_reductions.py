"""Reduction operators of the four-coordinate engine: Mean, Minimum, Maximum.

Engine tests use a synthetic provider (any cadence, any mask), so the rules
are checked exactly; archive tests check the operators on the Held–Suarez
T42L20 archive against fixed-time evaluations, and audit the run's archived
statistical series against the general reductions.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.visual import selection as model
from tropoi.representation.visual.selection import SelectionError, evaluate, parse_selection, time_weights
from tropoi.representation.visual.viewer_server import ViewerSession, provider_for

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "runs/hs-T42L20-prod-002"


# ----------------------------------------------------------------------
# Engine rules on a synthetic provider
# ----------------------------------------------------------------------

TIMES = [0.0, 10.0, 30.0, 40.0, 100.0]            # irregular cadence
NV, NLAT, NLON = 3, 4, 5


def _field(t):
    k, j, i = np.meshgrid(np.arange(NV), np.arange(NLAT), np.arange(NLON), indexing="ij")
    return np.sin(0.1 * t + k) + 0.3 * j - 0.2 * i


def _mask(t, values):
    out = values.copy()
    if t == 30.0:
        out[0, 0, 0] = np.nan                     # one masked sample at one time
    out[2, 3, 4] = np.nan                         # masked at every time
    return out


class Synthetic:
    """The provider protocol over an analytic field; NaN marks masked samples."""

    def __init__(self, masked=False):
        self.masked = masked
        caps = dict(roles=["x", "y", "fixed", "reduce"], reduce=model.reductions("rule"))
        self.manifest = dict(
            time_axes=dict(t=dict(values=TIMES)),
            explorer=dict(quantities=dict(q=dict(
                presentation=dict(name="Q", units="1"), time_axis="t",
                vertical=dict(domain="atmosphere", levels=[]),
                coordinates=dict(longitude=caps, latitude=caps, time=caps, vertical=dict(
                    roles=["x", "y", "fixed", "reduce"], fixed=["native"], axis=["native"],
                    reduce=model.reductions(), reduce_unavailable=dict(mean="no weights")))))))

    def plan(self, sel):
        axes = dict(longitude=model.axis("longitude", "", "lon", "", range(NLON)),
                    latitude=model.axis("latitude", "", "lat", "", range(NLAT)),
                    vertical=model.axis("vertical", "native", "level", "", range(NV)))
        w = np.array([1.0, 2.0, 3.0, 4.0])
        return dict(presentation=dict(name="Q"), coordinates={c: dict(role=sel[c]["role"]) for c in model.COORDINATES},
                    axes=axes, companions=dict(pos=dict(label="p", units="Pa")),
                    latitude=dict(weights=w if sel["latitude"]["role"] == "reduce" else None))

    def samples(self, sel, plan, indices):
        for i in indices:
            t = TIMES[i]
            values = _field(t)
            if self.masked:
                values = _mask(t, values)
            pos = np.full((NV, NLAT, NLON), 100.0 * t)
            if sel["vertical"]["role"] == "fixed":
                k = int(sel["vertical"]["value"])
                values, pos = values[k:k + 1], pos[k:k + 1]
            yield t, values, dict(pos=pos)


def run(provider, **sel):
    sel.setdefault("quantity", "q")
    return evaluate(provider, parse_selection(provider.manifest, json.dumps(sel)))


def cube(masked=False):
    return np.stack([(_mask(t, _field(t)) if masked else _field(t)) for t in TIMES])     # (t, v, lat, lon)


def lonlat(**rest):
    return dict(longitude=dict(role="x"), latitude=dict(role="y"), **rest)


def test_time_cell_weights():
    assert time_weights([0.0]) is None and time_weights([0.0, 7.0]) is None
    assert time_weights([0.0, 10.0, 20.0, 30.0]) is None                  # regular: plain 1/N mean
    np.testing.assert_array_equal(time_weights(TIMES), [10.0, 15.0, 15.0, 35.0, 60.0])


def test_irregular_time_mean_is_cell_weighted():
    d = run(Synthetic(), **lonlat(vertical=dict(role="fixed", representation="native", value=1),
                                  time=dict(role="reduce", op="mean", first=0, last=4)))
    w = np.array([10.0, 15.0, 15.0, 35.0, 60.0])
    expected = (cube()[:, 1] * w[:, None, None]).sum(axis=0) / w.sum()
    np.testing.assert_allclose(np.array(d["values"]), expected, rtol=0, atol=1e-14)
    t = d["coordinates"]["time"]
    assert t["cadence"] == "irregular" and t["weighting"].startswith("cell weights")
    assert t["weights_seconds"] == list(w)


def test_selected_interval_only():
    """A sub-interval is weighted by its own samples, not by the archive's."""
    d = run(Synthetic(), **lonlat(vertical=dict(role="fixed", representation="native", value=0),
                                  time=dict(role="reduce", op="mean", first=1, last=3)))
    w = np.array([20.0, 15.0, 10.0])                                       # t = 10, 30, 40
    expected = (cube()[1:4, 0] * w[:, None, None]).sum(axis=0) / w.sum()
    np.testing.assert_allclose(np.array(d["values"]), expected, rtol=0, atol=1e-14)
    assert d["coordinates"]["time"]["first_seconds"] == 10.0 and d["coordinates"]["time"]["last_seconds"] == 40.0


@pytest.mark.parametrize("op", ["min", "max"])
def test_time_extrema_exclude_masked_samples(op):
    d = run(Synthetic(masked=True), **lonlat(vertical=dict(role="fixed", representation="native", value=0),
                                             time=dict(role="reduce", op=op, first=0, last=4)))
    data = cube(masked=True)[:, 0]
    expected = (np.nanmin if op == "min" else np.nanmax)(data, axis=0)
    values = np.array(d["values"], dtype=float)
    np.testing.assert_array_equal(values, expected)
    assert np.isfinite(values[0, 0])                           # masked at one time only: still a value
    t = d["coordinates"]["time"]
    assert t["op"] == op and t["masked_excluded"] == 1 and "excluded" in t["rule"]


def test_time_extremum_is_masked_where_every_sample_is():
    d = run(Synthetic(masked=True), **lonlat(vertical=dict(role="fixed", representation="native", value=2),
                                             time=dict(role="reduce", op="max", first=0, last=4)))
    assert d["values"][3][4] is None


def test_time_mean_stays_masked_if_any_sample_is():
    d = run(Synthetic(masked=True), **lonlat(vertical=dict(role="fixed", representation="native", value=0),
                                             time=dict(role="reduce", op="mean", first=0, last=4)))
    assert d["values"][0][0] is None and d["values"][1][1] is not None


def test_spatial_extrema_and_order():
    """Longitude max, then latitude min, then time mean: the documented order, stated in the payload."""
    d = run(Synthetic(), longitude=dict(role="reduce", op="max"), latitude=dict(role="reduce", op="min"),
            vertical=dict(role="y", representation="native"), time=dict(role="reduce", op="mean", first=0, last=4))
    w = np.array([10.0, 15.0, 15.0, 35.0, 60.0])
    per_time = cube().max(axis=3).min(axis=2)                                   # (t, v)
    expected = (per_time * w[:, None]).sum(axis=0) / w.sum()
    np.testing.assert_allclose(d["values"], expected, rtol=0, atol=1e-14)
    assert d["reduction_order"] == ["longitude", "latitude", "time"]
    assert any("order" in n for n in d["notes"])
    # The companion locates the rows: its mean, not its max or min.
    pos = (np.array([100.0 * t for t in TIMES]) * w).sum() / w.sum()
    np.testing.assert_allclose(d["companions"]["pos"]["values"], [pos] * NV, rtol=1e-15)


def test_vertical_extrema_and_refused_vertical_mean():
    d = run(Synthetic(), **lonlat(vertical=dict(role="reduce", op="max"), time=dict(role="fixed", index=2)))
    np.testing.assert_array_equal(np.array(d["values"]), cube()[2].max(axis=0))
    with pytest.raises(SelectionError, match="'mean' over vertical is not available: no weights"):
        run(Synthetic(), **lonlat(vertical=dict(role="reduce", op="mean"), time=dict(role="fixed", index=2)))


def test_area_weighted_mean_unchanged():
    d = run(Synthetic(), longitude=dict(role="reduce", op="mean"), latitude=dict(role="reduce", op="mean"),
            vertical=dict(role="y", representation="native"), time=dict(role="fixed", index=0))
    w = np.array([1.0, 2.0, 3.0, 4.0])
    expected = (cube()[0].mean(axis=2) * w).sum(axis=1) / w.sum()
    np.testing.assert_array_equal(d["values"], expected)


# ----------------------------------------------------------------------
# Held–Suarez archive
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def hs():
    if not HS.is_dir():
        pytest.skip(f"Missing archive: {HS}")
    session = ViewerSession([HS])
    return session, provider_for(session.roots[0], session.manifests[0])


def select(session, **sel):
    data, _ = session.select(0, json.dumps(sel))
    return json.loads(data)


SIGMA_SURFACE = dict(role="fixed", representation="sigma", value=0.5)


@pytest.mark.parametrize("op,fn", [("max", np.fmax), ("min", np.fmin)])
def test_time_extremum_map_equals_fixed_time_fields(hs, op, fn):
    session, _ = hs
    base = dict(quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"), vertical=SIGMA_SURFACE)
    d = select(session, **base, time=dict(role="reduce", op=op, first=20, last=22))
    fields = [np.array(select(session, **base, time=dict(role="fixed", index=i))["values"]) for i in (20, 21, 22)]
    np.testing.assert_array_equal(np.array(d["values"]), fn(fn(fields[0], fields[1]), fields[2]))
    assert d["description"] == f"Longitude × latitude · vertical fixed · time {op}"


def test_minimum_temperature_profile_over_an_interval(hs):
    session, _ = hs
    base = dict(quantity="temperature", longitude=dict(role="fixed", value=0.0),
                latitude=dict(role="fixed", value=45.0), vertical=dict(role="y", representation="sigma"))
    d = select(session, **base, time=dict(role="reduce", op="min", first=100, last=103))
    profiles = np.array([select(session, **base, time=dict(role="fixed", index=i))["values"] for i in range(100, 104)])
    np.testing.assert_array_equal(d["values"], profiles.min(axis=0))


def test_maximum_wind_speed_over_an_interval_and_the_vertical(hs):
    """Max |V| over 3 saved states and every native level: one lon × lat field."""
    session, provider = hs
    d = select(session, quantity="wind_speed", longitude=dict(role="x"), latitude=dict(role="y"),
               vertical=dict(role="reduce", op="max"), time=dict(role="reduce", op="max", first=50, last=52))
    per_level = [np.array(select(session, quantity="wind_speed", longitude=dict(role="x"), latitude=dict(role="y"),
                                 vertical=dict(role="fixed", representation="native", value=k),
                                 time=dict(role="reduce", op="max", first=50, last=52))["values"])
                 for k in range(20)]
    np.testing.assert_allclose(np.array(d["values"]), np.max(per_level, axis=0), rtol=1e-13, atol=0)
    assert d["coordinates"]["vertical"]["op"] == "max"


def test_spatially_averaged_profile(hs):
    session, _ = hs
    d = select(session, quantity="temperature", longitude=dict(role="reduce", op="mean"),
               latitude=dict(role="reduce", op="mean"), vertical=dict(role="y", representation="sigma"),
               time=dict(role="reduce", op="mean", first=110, last=119))
    assert len(d["values"]) == 20 and 180 < min(d["values"]) < max(d["values"]) < 300
    assert d["coordinates"]["time"]["weighting"] == "equal (regular cadence)"


def test_refusals_name_their_reason(hs):
    session, _ = hs
    with pytest.raises(SelectionError, match="mass"):
        select(session, quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"),
               vertical=dict(role="reduce", op="mean"), time=dict(role="fixed", index=0))
    with pytest.raises(SelectionError, match=r"time mean of \[pₛ v\]"):
        select(session, quantity="mass_streamfunction", longitude=dict(role="reduce", op="mean"),
               latitude=dict(role="x"), vertical=dict(role="y", representation="sigma"),
               time=dict(role="reduce", op="max", first=0, last=3))


# ----------------------------------------------------------------------
# Audit of the archived statistical series (series.json, daily cadence)
# ----------------------------------------------------------------------

def _series(name):
    data = json.loads((HS / "series.json").read_text(encoding="utf-8"))
    return dict(zip(data["day"], data[name]))


@pytest.mark.parametrize("name,op", [("t_max", "max"), ("t_min", "min")])
def test_archived_extrema_equal_the_general_reduction_at_saved_states(hs, name, op):
    """series:t_max/t_min (daily, T over the native state grid at every level) equal the engine's
    longitude, latitude and vertical extremum at the saved states, to rounding. The archived
    series is kept: it is daily, the saved states every 10 days."""
    session, provider = hs
    series = _series(name)
    for index in (0, 60, 119):
        day = provider.manifest["time_axes"]["snapshots"]["values"][index] / 86400
        d = select(session, quantity="temperature", longitude=dict(role="reduce", op=op),
                   latitude=dict(role="reduce", op=op), vertical=dict(role="reduce", op=op),
                   time=dict(role="fixed", index=index))
        assert abs(d["values"] - series[day]) <= 1e-9 * abs(series[day])


def test_archived_mean_T_is_a_sigma_weighted_vertical_mean(hs):
    """series:mean_T = area mean of Σ Δσ_k T_k. The engine's area-mean profile reproduces it with
    Δσ weights applied outside the engine; the engine itself offers no vertical mean (refused, by
    decision), so the archived series is a distinct diagnostic and is kept."""
    session, provider = hs
    series = _series("mean_T")
    interfaces = np.array(provider.manifest["axes"]["vertical"]["interfaces"])
    dsig = np.diff(interfaces)
    for index in (0, 119):
        day = provider.manifest["time_axes"]["snapshots"]["values"][index] / 86400
        d = select(session, quantity="temperature", longitude=dict(role="reduce", op="mean"),
                   latitude=dict(role="reduce", op="mean"), vertical=dict(role="y", representation="sigma"),
                   time=dict(role="fixed", index=index))
        assert abs(float(np.dot(dsig, d["values"])) - series[day]) <= 1e-9 * series[day]


def test_minimum_profile_labels_its_statistic_and_keeps_position_means(hs):
    """Under a minimum the quantity's table column and marker say so; p and Z stay level-position means."""
    session, _ = hs
    d = select(session, quantity="temperature", longitude=dict(role="fixed", value=10.0),
               latitude=dict(role="fixed", value=-30.0), vertical=dict(role="y", representation="sigma"),
               time=dict(role="reduce", op="min", first=60, last=63))
    assert d["table"]["columns"][5] == "T (K) (time min)" and d["table"]["columns"][2] == "p (Pa) (mean)"
    assert d["markers"][0]["label"].startswith("Lowest full level air T (time min) = ")
    assert d["markers"][0]["value"] == d["values"][-1]


def test_value_companion_takes_the_quantity_statistic():
    """A companion flagged statistic="quantity" (a skin, a slab Tₛ) is reduced like the quantity."""
    class Skin(Synthetic):
        def plan(self, sel):
            out = super().plan(sel)
            out["companions"]["skin"] = dict(label="skin", units="K", surface=True, statistic="quantity")
            return out

        def samples(self, sel, plan, indices):
            for (t, values, companions), i in zip(super().samples(sel, plan, indices), indices):
                yield t, values, dict(companions, skin=np.full((1, NLAT, NLON), float(i)))

    d = run(Skin(), longitude=dict(role="reduce", op="mean"), latitude=dict(role="reduce", op="mean"),
            vertical=dict(role="y", representation="native"), time=dict(role="reduce", op="max", first=0, last=4))
    assert d["companions"]["skin"]["values"] == 4.0                                # max of 0..4, not the mean
    pos = (np.array([100.0 * t for t in TIMES])).mean()                            # positions: the plain mean
    assert d["companions"]["pos"]["values"] == pytest.approx([pos] * NV, rel=1e-15)
