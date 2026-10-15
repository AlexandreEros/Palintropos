"""Saved resolved initial states and the custom-run path, on the CPU.

Covers the persistence contract (exact coefficient round trips, strict
versioned JSON, no pickles, verification on load, fresh-process loading
with CuPy blocked), coefficient-level validation, the content identity, the
CPU configuration layer, and the run lifecycle with the CUDA integration
stubbed out (capsule contents, run ids, inspection, failure handling).
The GPU counterparts live in tests/test_custom_runs.py.
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tropoi.run.custom import (CustomRunConfig, DiscretizationSpec,
                               FluidLayerSpec, InitialState,
                               InitialStateError, PlanetSpec, TopographySpec,
                               load_initial_state)
from tropoi.representation.archive.initial_state import (METADATA_FILE,
                                                         STATE_FILE,
                                                         TOPOGRAPHY_FILE,
                                                         array_digest)

L = 6


def _triangle(rng, *, scale=1.0, zero_mean=True):
    a = np.zeros((L + 1, L + 1), dtype=np.complex128)
    for l in range(L + 1):
        for m in range(l + 1):
            re, im = rng.standard_normal(2) * scale
            a[l, m] = complex(re, 0.0 if m == 0 else im)
    if zero_mean:
        a[0, 0] = 0.0
    return a


def _bve_state(rng=None, **overrides):
    rng = rng or np.random.default_rng(1)
    coeffs = _triangle(rng, scale=1e-5, zero_mean=False)
    kwargs = dict(solver="bve", coeffs=coeffs,
                  planet=PlanetSpec.ideal_sphere(3.0e6, 36000.0),
                  discretization=DiscretizationSpec.gauss_latlon(L, 12, 24),
                  label="vortices")
    kwargs.update(overrides)
    return InitialState(**kwargs)


def _swe_state(rng=None, *, terrain=True, **overrides):
    rng = rng or np.random.default_rng(2)
    coeffs = np.stack([_triangle(rng, scale=1e-6), _triangle(rng, scale=1e-7),
                       _triangle(rng, scale=50.0)])
    topo = None
    spec = TopographySpec.flat()
    if terrain:
        topo = _triangle(rng, scale=100.0, zero_mean=False)
        spec = TopographySpec.mountain(height_m=2000, lat_deg=30, lon_deg=90,
                                       width_deg=20)
    kwargs = dict(solver="swe", coeffs=coeffs,
                  planet=PlanetSpec.oblate(5.0e6, 50000.0),
                  discretization=DiscretizationSpec.gauss_latlon(L, 12, 24),
                  fluid=FluidLayerSpec(3.71, 4000.0), topography=spec,
                  topography_lm=topo, label="jet")
    kwargs.update(overrides)
    return InitialState(**kwargs)


# ---------------------------------------------------------------------------
# Persistence: exact round trips, strict format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [_bve_state, _swe_state,
                                  lambda: _swe_state(terrain=False)])
def test_save_load_round_trip_is_exact(tmp_path, make):
    state = make()
    path = state.save(tmp_path / "state")
    loaded = load_initial_state(path)
    assert loaded.coeffs.dtype == np.complex128
    # Bitwise, including signed zeros.
    assert loaded.coeffs.tobytes() == state.coeffs.tobytes()
    if state.topography_lm is None:
        assert loaded.topography_lm is None
    else:
        assert loaded.topography_lm.tobytes() == state.topography_lm.tobytes()
    assert loaded.digest == state.digest
    assert loaded.planet == state.planet
    assert loaded.discretization == state.discretization
    assert loaded.fluid == state.fluid and loaded.topography == state.topography
    assert loaded.label == state.label
    assert not loaded.coeffs.flags.writeable


def test_round_trip_preserves_extreme_and_signed_values(tmp_path):
    coeffs = np.zeros((L + 1, L + 1), dtype=np.complex128)
    coeffs[1, 0] = -0.0
    coeffs[2, 1] = complex(5e-324, -1.7976931348623157e308)
    coeffs[3, 3] = complex(np.nextafter(1.0, 2.0), -0.0)
    state = _bve_state(coeffs=coeffs)
    loaded = load_initial_state(state.save(tmp_path / "s"))
    assert loaded.coeffs.tobytes() == coeffs.tobytes()
    assert math.copysign(1.0, loaded.coeffs[1, 0].real) == -1.0


def test_saved_document_is_strict_versioned_json_with_separate_blocks(tmp_path):
    path = _swe_state().save(tmp_path / "s")
    text = (path / METADATA_FILE).read_text(encoding="utf-8")
    document = json.loads(text, parse_constant=lambda c: pytest.fail(c))
    assert document["format"] == "tropoi.initial_state"
    assert document["version"] == 1
    for block in ("planet", "discretization", "fluid", "topography", "state"):
        assert document[block] is not None, block
    assert document["state"]["state_schema"]["fields"][2]["name"] == "phi"
    assert "perturbation thickness geopotential" in \
        document["state"]["state_schema"]["fields"][2]["description"]
    assert sorted(p.name for p in path.iterdir()) == sorted(
        [METADATA_FILE, STATE_FILE, TOPOGRAPHY_FILE])
    # Plain .npy arrays that load without pickle support.
    arr = np.load(path / STATE_FILE, allow_pickle=False)
    assert arr.dtype == np.complex128 and arr.shape == (3, L + 1, L + 1)


def test_save_refuses_to_clobber(tmp_path):
    state = _bve_state()
    path = state.save(tmp_path / "s")
    with pytest.raises(FileExistsError):
        state.save(path)
    other = _bve_state(label="renamed")
    other.save(path, overwrite=True)
    assert load_initial_state(path).label == "renamed"
    (tmp_path / "plain").mkdir()
    with pytest.raises(InitialStateError, match="not a saved initial state"):
        state.save(tmp_path / "plain", overwrite=True)


# ---------------------------------------------------------------------------
# Overwrite is recoverable: failure injection
# ---------------------------------------------------------------------------

def _fail_renames(monkeypatch, should_fail):
    """Make ``os.replace`` raise for the (src, dst) pairs ``should_fail`` picks."""
    import os

    real = os.replace
    calls = []

    def replace(src, dst):
        src, dst = pathlib.Path(src), pathlib.Path(dst)
        calls.append((src.name, dst.name))
        if should_fail(src, dst):
            raise OSError(f"injected rename failure {src.name} -> {dst.name}")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    return calls


def _siblings(path):
    return sorted(p.name for p in path.parent.iterdir())


def _old_and_new(tmp_path):
    old = _swe_state(label="previous")
    path = old.save(tmp_path / "s")
    coeffs = np.array(old.coeffs)
    coeffs[2, 3, 1] *= 2.0
    return old, _swe_state(coeffs=coeffs, label="replacement"), path


def test_failed_install_during_overwrite_keeps_the_previous_state(
        tmp_path, monkeypatch):
    old, new, path = _old_and_new(tmp_path)
    calls = _fail_renames(monkeypatch, lambda src, dst:
                          ".tmp-" in src.name and dst == path)
    with pytest.raises(OSError, match="injected"):
        new.save(path, overwrite=True)
    monkeypatch.undo()
    assert any(".tmp-" in src and dst == "s" for src, dst in calls)
    restored = load_initial_state(path)
    assert restored.digest == old.digest and restored.label == "previous"
    assert _siblings(path) == ["s"]              # no staging/backup litter


def test_failed_restore_deletes_nothing_and_says_where_both_copies_are(
        tmp_path, monkeypatch):
    old, new, path = _old_and_new(tmp_path)
    _fail_renames(monkeypatch, lambda src, dst: dst == path
                  and (".tmp-" in src.name or ".bak-" in src.name))
    with pytest.raises(InitialStateError, match="nothing was deleted") as info:
        new.save(path, overwrite=True)
    monkeypatch.undo()
    assert not path.exists()
    names = _siblings(path)
    (backup,) = [n for n in names if ".bak-" in n]
    (staging,) = [n for n in names if ".tmp-" in n]
    assert backup in str(info.value) and staging in str(info.value)
    assert load_initial_state(tmp_path / backup).digest == old.digest
    assert load_initial_state(tmp_path / staging).digest == new.digest


def test_failure_setting_the_previous_state_aside_leaves_it_in_place(
        tmp_path, monkeypatch):
    old, new, path = _old_and_new(tmp_path)
    _fail_renames(monkeypatch, lambda src, dst: src == path)
    with pytest.raises(OSError, match="injected"):
        new.save(path, overwrite=True)
    monkeypatch.undo()
    assert load_initial_state(path).digest == old.digest
    assert _siblings(path) == ["s"]


def test_failure_writing_the_replacement_leaves_the_previous_state(
        tmp_path, monkeypatch):
    import tropoi.representation.archive.initial_state as module

    old, new, path = _old_and_new(tmp_path)

    def broken_write(target, array):
        raise OSError("injected: disk full")

    monkeypatch.setattr(module, "_write_npy", broken_write)
    with pytest.raises(OSError, match="disk full"):
        new.save(path, overwrite=True)
    assert load_initial_state(path).digest == old.digest
    assert _siblings(path) == ["s"]


def test_undeletable_previous_state_is_reported_not_fatal(tmp_path,
                                                          monkeypatch):
    import shutil
    import tropoi.representation.archive.initial_state as module

    old, new, path = _old_and_new(tmp_path)
    real_rmtree = shutil.rmtree

    def rmtree(target, *args, **kwargs):
        if ".bak-" in pathlib.Path(target).name:
            raise OSError("injected: permission denied")
        return real_rmtree(target, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", rmtree)
    with pytest.warns(UserWarning, match="could not delete the previous"):
        new.save(path, overwrite=True)
    monkeypatch.undo()
    assert load_initial_state(path).digest == new.digest
    (backup,) = [n for n in _siblings(path) if ".bak-" in n]
    assert load_initial_state(tmp_path / backup).digest == old.digest


def test_successful_overwrite_leaves_no_litter(tmp_path):
    old, new, path = _old_and_new(tmp_path)
    new.save(path, overwrite=True)
    assert load_initial_state(path).digest == new.digest != old.digest
    assert _siblings(path) == ["s"]


def _rewrite(path, mutate):
    meta = path / METADATA_FILE
    document = json.loads(meta.read_text(encoding="utf-8"))
    mutate(document)
    meta.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(version=2), "version"),
    (lambda d: d.update(format="other"), "format"),
    (lambda d: d.update(solver="pe"), "solver"),
    (lambda d: d.update(runtime={"device": 0}), "unknown key"),
    (lambda d: d.update(digest="sha256:" + "0" * 64), "digest"),
    (lambda d: d["state"].update(coefficient_file="../escape.npy"),
     "coefficient_file"),
    (lambda d: d["state"].update(sha256="sha256:" + "0" * 64), "SHA-256"),
    (lambda d: d["fluid"].update(gravity_m_s2=9.81), "state_schema|digest"),
    (lambda d: d["fluid"].update(phi_definition="free surface"), "convention"),
    (lambda d: d["planet"].update(radius_convention="geoid"), "convention"),
    (lambda d: d["planet"].update(reference_radius_m=5.1e6), "derived"),
    (lambda d: d["state"]["state_schema"]["fields"][2].update(units="m"),
     "state_schema"),
    (lambda d: d["topography"].update(sha256="sha256:" + "1" * 64), "SHA-256"),
    (lambda d: d["state"].update(layout="(m, l)"), "implements"),
    (lambda d: d["topography"]["field"].update(units="km"), "implements"),
    (lambda d: d["state"].update(device="cuda:0"), "unknown key"),
])
def test_load_verifies_instead_of_trusting(tmp_path, mutate, match):
    path = _swe_state().save(tmp_path / "s")
    _rewrite(path, mutate)
    with pytest.raises(InitialStateError, match=match):
        load_initial_state(path)


def test_load_rejects_modified_arrays_and_pickles(tmp_path):
    state = _bve_state()
    path = state.save(tmp_path / "s")
    tampered = np.array(state.coeffs)
    tampered[2, 1] += 1e-20
    np.save(path / STATE_FILE, tampered, allow_pickle=False)
    with pytest.raises(InitialStateError, match="SHA-256"):
        load_initial_state(path)
    np.save(path / STATE_FILE, np.array([{"a": 1}], dtype=object),
            allow_pickle=True)
    with pytest.raises(InitialStateError, match="not a readable"):
        load_initial_state(path)


def test_load_rejects_non_json_constants(tmp_path):
    path = _bve_state().save(tmp_path / "s")
    meta = path / METADATA_FILE
    meta.write_text(meta.read_text(encoding="utf-8").replace(
        '"mass_kg": 5.972e+24', '"mass_kg": NaN', 1), encoding="utf-8")
    with pytest.raises(InitialStateError, match="non-JSON constant"):
        load_initial_state(path)


def test_fresh_process_loads_and_validates_without_cupy(tmp_path):
    from cli.conftest import HEAVY_MODULES

    state = _swe_state()
    path = state.save(tmp_path / "s")
    code = (
        "import sys\n"
        "sys.modules['cupy'] = None\n"            # importing CuPy would fail
        "from tropoi.run.custom import load_initial_state\n"
        f"s = load_initial_state({str(path)!r})\n"
        "s.validate()\n"
        "s.state['phi'].validate()\n"
        "assert s.topography_modes.validate() is not None\n"
        f"assert s.digest == {state.digest!r}\n"
        "summary = s.summary()\n"
        "assert summary['fluid']['gravity_m_s2'] == 3.71\n"
        f"banned = [m for m in {HEAVY_MODULES!r} if sys.modules.get(m)]\n"
        "assert not banned, banned\n"
        "assert not [m for m in sys.modules if m.startswith(("
        "'tropoi.spatial.grids', 'tropoi.spatial.transforms', "
        "'tropoi.spatial.states', 'tropoi.temporal.tendencies'))]\n"
        "import numpy as np\n"
        "print(s.coeffs.tobytes().hex())\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == state.coeffs.tobytes().hex()


# ---------------------------------------------------------------------------
# Coefficient-level validation
# ---------------------------------------------------------------------------

def test_construction_validates_coefficient_conventions():
    good = _swe_state()

    def broken(edit):
        coeffs = np.array(good.coeffs)
        edit(coeffs)
        return coeffs

    cases = [
        (broken(lambda c: c.__setitem__((0, 3, 2), np.nan)), "non-finite"),
        (broken(lambda c: c.__setitem__((1, 1, 3), 1e-9)), "padding"),
        (broken(lambda c: c.__setitem__((2, 3, 0), 1 + 1e-12j)), "Im"),
        (broken(lambda c: c.__setitem__((2, 0, 0), 1.0)), "monopole"),
        (broken(lambda c: c.__setitem__((0, 0, 0), 1e-3)), "monopole"),
    ]
    for coeffs, match in cases:
        with pytest.raises(InitialStateError, match=match):
            _swe_state(coeffs=coeffs)


def test_bve_circulation_monopole_is_conserved_not_constrained():
    coeffs = np.zeros((L + 1, L + 1), dtype=np.complex128)
    coeffs[0, 0] = 1e-6                   # a conserved, dynamically inert mean
    state = _bve_state(coeffs=coeffs)
    assert state.state["zeta"].spec.monopole == "conserved"


@pytest.mark.parametrize("overrides,match", [
    ({"coeffs": np.zeros((L + 1, L + 1), dtype=np.complex64)}, "complex128"),
    ({"coeffs": np.zeros((L + 1, L + 1))}, "complex128"),
    ({"coeffs": np.zeros((L, L), dtype=np.complex128)}, "shape"),
    ({"coeffs": [[0j]]}, "numpy.ndarray"),
    ({"discretization": DiscretizationSpec.gauss_latlon(L + 1, 12, 24)},
     "shape"),
    ({"fluid": FluidLayerSpec(9.8, 10.0)}, "no fluid"),
    ({"provenance": {"x": math.inf}}, "strict-JSON"),
    ({"solver": "pe"}, "solver"),
    ({"planet": {"radius": 1.0}}, "PlanetSpec"),
])
def test_bve_state_rejects_invalid_inputs(overrides, match):
    with pytest.raises(InitialStateError, match=match):
        _bve_state(**overrides)


def test_swe_state_requires_its_physical_context():
    with pytest.raises(InitialStateError, match="gravity is never derived"):
        _swe_state(fluid=None)
    with pytest.raises(InitialStateError, match="resolved"):
        _swe_state(topography_lm=None)
    with pytest.raises(InitialStateError, match="flat topography"):
        _swe_state(terrain=False, topography_lm=np.ones(
            (L + 1, L + 1), dtype=np.complex128))
    with pytest.raises(InitialStateError, match="all-zero"):
        _swe_state(topography_lm=np.zeros((L + 1, L + 1),
                                          dtype=np.complex128))
    bad = np.zeros((L + 1, L + 1), dtype=np.complex128)
    bad[2, 4] = 1.0
    with pytest.raises(InitialStateError, match="padding"):
        _swe_state(topography_lm=bad)
    # Defaults: a flat bottom.
    assert _swe_state(terrain=False, topography=None).topography.is_flat


def test_stored_arrays_are_copies_and_read_only():
    coeffs = _bve_state().coeffs.copy()
    state = _bve_state(coeffs=coeffs)
    coeffs[1, 0] = 123.0
    assert state.coeffs[1, 0] != 123.0
    with pytest.raises(ValueError):
        state.coeffs[1, 0] = 1.0


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_digest_is_content_identity_not_name():
    base = _swe_state()
    assert _swe_state(label="other name", provenance={"note": "x"}).digest \
        == base.digest
    # The resolution record is not part of the scientific content.
    assert _swe_state(discretization=DiscretizationSpec.geodesic(L, 3)).digest \
        == base.digest

    nudged = np.array(base.coeffs)
    nudged[2, 4, 2] = np.nextafter(nudged[2, 4, 2].real, np.inf) \
        + 1j * nudged[2, 4, 2].imag
    variants = [
        _swe_state(coeffs=nudged),
        _swe_state(fluid=FluidLayerSpec(3.71, 4000.000001)),
        _swe_state(fluid=FluidLayerSpec(3.7100001, 4000.0)),
        _swe_state(planet=PlanetSpec.ideal_sphere(5.0e6, 50000.0)),
        _swe_state(planet=PlanetSpec.oblate(5.0e6, 50001.0)),
        _swe_state(topography=TopographySpec.mountain(
            height_m=2001, lat_deg=30, lon_deg=90, width_deg=20)),
        _swe_state(topography_lm=base.topography_lm * 1.0000001),
    ]
    digests = {v.digest for v in variants} | {base.digest}
    assert len(digests) == len(variants) + 1


def test_array_digest_covers_shape_and_values():
    a = np.zeros((2, 3), dtype=np.complex128)
    assert array_digest(a) != array_digest(a.reshape(3, 2))
    b = a.copy()
    b[0, 0] = -0.0
    assert array_digest(a) != array_digest(b)


# ---------------------------------------------------------------------------
# Configuration layer (CPU)
# ---------------------------------------------------------------------------

def test_custom_run_config_resolves_on_cpu():
    state = _swe_state()
    cfg = CustomRunConfig.resolve(state, duration_days=0.5)
    assert cfg.snapshot_mode == "count" and cfg.n_snapshots == 5
    assert cfg.snapshot_times_seconds() == [0.0, 10800.0, 21600.0, 32400.0,
                                            43200.0]
    config = cfg.to_run_config_dict()
    assert config["solver"] == "swe" and config["scenario"] == "custom"
    assert config["gravity"] == 3.71 and config["mean_depth_m"] == 4000.0
    assert config["initial_state_digest"] == state.digest
    assert config["planet"] == state.planet.to_dict()
    assert config["topography"] == "mountain"
    assert config["topography_digest"] == array_digest(state.topography_lm)
    assert "radius_earth_units" not in config
    assert config["day_hours"] == 50000.0 / 3600.0
    assert "initial_state_label" not in cfg.scientific_config_dict()

    other_grid = DiscretizationSpec.geodesic(L, 3)
    assert CustomRunConfig.resolve(state, duration_days=1,
                                   discretization=other_grid
                                   ).to_run_config_dict()["grid"] == "geodesic"
    bve = CustomRunConfig.resolve(_bve_state(), duration_days=1,
                                  snapshot_interval_s=3600.0, viscosity=1e5)
    assert bve.snapshot_mode == "interval" and bve.n_snapshots is None
    assert bve.to_run_config_dict()["viscosity"] == 1e5


@pytest.mark.parametrize("kwargs,match", [
    ({"duration_days": 0.0}, "duration_days"),
    ({"duration_days": math.nan}, "duration_days"),
    ({"duration_days": 1, "n_snapshots": 3, "snapshot_interval_s": 60.0},
     "mutually exclusive"),
    ({"duration_days": 1, "n_snapshots": -1}, "n_snapshots"),
    ({"duration_days": 1, "viscosity": 1.0}, "BVE only"),
    ({"duration_days": 1, "discretization":
      DiscretizationSpec.gauss_latlon(L + 2, 12, 24)}, "never truncated"),
    ({"duration_days": 1, "discretization":
      DiscretizationSpec(lmax=L, product_quadrature="coarse")}, "fine"),
    ({"duration_days": 1, "plots": ["movie"]}, "unknown plot"),
    ({"duration_days": 1, "n_snapshots": 0, "plots": ["summary"]},
     "stored state"),
])
def test_custom_run_config_rejects_user_errors_before_cuda(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CustomRunConfig.resolve(_swe_state(), **kwargs)


# ---------------------------------------------------------------------------
# Lifecycle with the CUDA integration stubbed
# ---------------------------------------------------------------------------

def _stub_integration(monkeypatch, *, fail=False):
    import tropoi.run.custom.execute as execute
    from tropoi.representation.archive.schema import (COEFFICIENT_FILES,
                                                      TIME_FILES)

    def fake(cfg, run_dir, run_config):
        if fail:
            raise RuntimeError("thickness collapsed (stub)")
        times = np.asarray(cfg.snapshot_times_seconds(), dtype=np.float64)
        frames = np.stack([cfg.initial_state.coeffs] * len(times))
        np.save(run_dir.path / COEFFICIENT_FILES[cfg.solver], frames)
        np.save(run_dir.path / TIME_FILES[cfg.solver], times)

    monkeypatch.setattr(execute, "_integrate", fake)


def test_custom_run_capsule_records_state_identity(tmp_path, monkeypatch):
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import INITIAL_STATE_DIR, run_initial_state

    _stub_integration(monkeypatch)
    state = _swe_state()
    saved = state.save(tmp_path / "jet")
    run = run_initial_state(saved, duration_days=0.25, n_snapshots=3,
                            out=str(tmp_path / "runs"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    config = manifest["run_config"]
    assert config["initial_state_digest"] == state.digest
    assert "_custom_" in run.name or run.name.split("_")[1] == "custom"
    copy = load_initial_state(run / INITIAL_STATE_DIR)
    assert copy.digest == state.digest
    assert copy.coeffs.tobytes() == state.coeffs.tobytes()
    schema = manifest["state_schema"]
    assert schema["geometry"]["planet"] == "oblate_volumetric_mean"
    assert schema["geometry"]["dynamical_radius_m"] == \
        state.planet.dynamical_radius_m
    assert schema["environment"]["initial_state_digest"] == state.digest
    assert "initial_state" in manifest["notes"]
    assert (tmp_path / "runs" / "latest_run.txt").read_text().strip() == \
        run.name

    sim = open_simulation(run)
    assert sim.solver == "swe" and len(sim) == 3
    assert sim[0].state["phi"].coeffs.tobytes() == \
        state.coeffs[2].tobytes()


def test_run_identity_follows_content_not_label(tmp_path, monkeypatch):
    from tropoi.run.custom import run_initial_state

    _stub_integration(monkeypatch)

    def run_hash(state, sub):
        run = run_initial_state(state, duration_days=0.1, n_snapshots=2,
                                out=str(tmp_path / sub))
        return run.name.split("_")[6]      # ts_scn_rot_r_l_snap_HASH_commit

    base = _bve_state()
    renamed = _bve_state(label="same content, different name")
    coeffs = np.array(base.coeffs)
    coeffs[3, 1] *= 1.5
    changed = _bve_state(coeffs=coeffs)        # same label, new content
    assert run_hash(base, "a") == run_hash(renamed, "b")
    assert run_hash(base, "c") != run_hash(changed, "d")


def test_interval_mode_custom_bve_run_ids_still_carry_the_hash(tmp_path,
                                                               monkeypatch):
    from tropoi.run.bve.io import make_run_id

    cfg = CustomRunConfig.resolve(_bve_state(), duration_days=0.25,
                                  snapshot_interval_s=7200.0)
    config = cfg.to_run_config_dict()
    run_id = make_run_id(config, commit="abc")
    assert run_id.split("_")[-2] != "dt2h" and len(run_id.split("_")) == 8


def test_failed_custom_run_is_marked_and_keeps_its_state(tmp_path,
                                                         monkeypatch):
    from tropoi.run.custom import INITIAL_STATE_DIR, run_initial_state

    _stub_integration(monkeypatch, fail=True)
    state = _swe_state()
    with pytest.raises(RuntimeError, match="stub"):
        run_initial_state(state, duration_days=0.1,
                          out=str(tmp_path / "runs"))
    (run,) = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert load_initial_state(run / INITIAL_STATE_DIR).digest == state.digest
    assert not (tmp_path / "runs" / "latest_run.txt").exists()


def test_inspect_shows_the_initial_state(tmp_path, monkeypatch, capsys):
    from tropoi.cli.main import main
    from tropoi.run.custom import run_initial_state

    _stub_integration(monkeypatch)
    state = _bve_state()
    run = run_initial_state(state, duration_days=0.1, n_snapshots=2,
                            out=str(tmp_path / "runs"))
    capsys.readouterr()
    assert main(["inspect", str(run), "--snapshot", "-1"]) == 0
    out = capsys.readouterr().out
    assert state.digest in out and "vortices" in out
    assert "ideal_sphere" in out


def test_preset_run_ids_are_unchanged_by_the_custom_keys():
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.bve.io import _config_hash, make_run_id
    from tropoi.run.swe.config import SWERunConfig

    # Preset configurations carry none of the custom keys, so their
    # scientific subset (and hash) is exactly what it was.
    for cfg in (BVERunConfig.resolve({}), SWERunConfig.resolve({}),
                BVERunConfig.resolve({}, snapshot_default="interval")):
        config = cfg.to_run_config_dict()
        assert not {"planet", "initial_state_digest",
                    "initial_state_label"} & set(config)
        assert _config_hash(config) == _config_hash(
            dict(config, initial_state_label="ignored"))
    legacy = BVERunConfig.resolve({}, snapshot_default="interval")
    assert make_run_id(legacy.to_run_config_dict(), commit="c").split(
        "_")[-2] == "dt6h"
