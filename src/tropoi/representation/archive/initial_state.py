"""Resolved spectral initial states: validation, identity, save and load.

An :class:`InitialState` is a complete, already-resolved BVE or SWE initial
condition: the packed prognostic coefficient frame a run starts from, plus
every specification needed to interpret it (planet, discretization, and for
the shallow-water core the fluid layer and the resolved bottom topography).
It is a host (NumPy) object: building, validating, saving, loading and
inspecting one never imports CuPy or initializes CUDA.

On-disk format (``format = "tropoi.initial_state"``, ``version = 1``)
--------------------------------------------------------------------
A directory holding

* ``initial_state.json`` -- the versioned document: separate blocks for the
  prescribed physics (``planet``; SWE ``fluid`` and ``topography``), the
  discretization the state was resolved for (``discretization``), the
  stored state (``state``: file, dtype, shape, SHA-256, and the
  ``state_schema`` block that run manifests use), a ``digest`` (the
  scientific identity, below), and descriptive ``label`` / ``provenance``.
  Strict JSON: no NaN/Infinity, no code, nothing evaluated on load.
* ``state_coeffs.npy`` -- the packed frame, ``complex128``: ``(L+1, L+1)``
  for the BVE (``zeta``), ``(3, L+1, L+1)`` for the SWE (``[zeta, delta,
  phi]``), in the repository's coefficient layout and normalization
  (``spatial.modes``).
* ``topography_elevation_lm.npy`` -- SWE over non-flat terrain only: the
  RESOLVED surface-elevation coefficients (m), ``complex128 (L+1, L+1)``.
  A run uses these exact coefficients; it never re-projects the terrain.

Arrays are written and read with ``allow_pickle=False``; loading verifies
the dtype, shape and SHA-256 of every array against the document, re-derives
the ``state_schema`` block and refuses a document that disagrees with it,
and re-checks the coefficient conventions. Live GPU resources and Python
callables are never part of the format.

Identity
--------
:attr:`InitialState.digest` is ``"sha256:<hex>"`` over a canonical JSON
description of the solver, the planet, the fluid layer, the topography
specification, and the exact bytes (dtype, shape, values) of the state and
terrain coefficients. It excludes the label, the provenance and the
discretization record, so it identifies the scientific content only: two
states share a digest exactly when their prescribed physics and stored
coefficients are identical, whatever they are called.

Coefficient-level validation (always, on construction and on load) is the
``spatial.modes`` convention check: finite values, zero ``m > l`` padding,
real ``a_l0``, and the field monopole rules of the archive schema (SWE
``zeta``, ``delta`` and ``phi`` are zero-mean to the cores' relative
tolerance; the BVE ``zeta`` monopole is the conserved circulation and is
not constrained here). Physical validation that needs the model's
transforms (SWE positive thickness on the state and product samplings) is
done by the model at execution or preparation time, not here.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import pathlib
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from tropoi.spatial.modes import (COEFFICIENT_LAYOUT,
                                  COEFFICIENT_NORMALIZATION, FieldSpec,
                                  SpectralConventionError, SpectralModes,
                                  SpectralState)
from tropoi.spatial.specification import (DiscretizationSpec, FluidLayerSpec,
                                          PlanetSpec, SpecificationError,
                                          TopographySpec)
from tropoi.representation.archive.schema import (SchemaError, StateSchema,
                                                  state_schema_for)

__all__ = [
    "INITIAL_STATE_FORMAT",
    "INITIAL_STATE_FORMAT_VERSION",
    "INITIAL_STATE_SOLVERS",
    "InitialState",
    "InitialStateError",
    "METADATA_FILE",
    "STATE_FILE",
    "TOPOGRAPHY_FILE",
    "array_digest",
    "load_initial_state",
    "save_initial_state",
]

INITIAL_STATE_FORMAT = "tropoi.initial_state"
INITIAL_STATE_FORMAT_VERSION = 1
#: Solvers whose initial states this format describes (PE is out of scope).
INITIAL_STATE_SOLVERS = ("bve", "swe")

METADATA_FILE = "initial_state.json"
STATE_FILE = "state_coeffs.npy"
TOPOGRAPHY_FILE = "topography_elevation_lm.npy"

#: The scenario token custom runs record (the run id's scenario slot).
CUSTOM_SCENARIO = "custom"

_IDENTITY_FORMAT = "tropoi.initial_state.identity"
_DTYPE = np.dtype("<c16")

#: Field description of the resolved terrain coefficients (a FieldSpec so
#: the same convention checks apply; its monopole is the mean elevation).
TOPOGRAPHY_FIELD = FieldSpec(
    "surface_elevation", "m",
    "resolved bottom-topography surface elevation h_s (metres); the "
    "surface geopotential is phi_s = gravity * h_s",
    rows=None, levels=False, monopole="horizontal-mean")

_DOCUMENT_KEYS = {"format", "version", "solver", "label", "digest", "planet",
                  "discretization", "fluid", "topography", "state",
                  "provenance", "created_utc", "software"}


class InitialStateError(ValueError):
    """An initial state is invalid, or a saved one cannot be read as such."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError) as err:
        raise InitialStateError(
            f"value is not strict-JSON serializable: {err}") from err


def array_digest(array: np.ndarray) -> str:
    """``sha256:<hex>`` of an array's little-endian complex128 bytes + shape."""
    arr = np.ascontiguousarray(np.asarray(array).astype(_DTYPE, copy=False))
    h = hashlib.sha256()
    h.update(_canonical_json({"dtype": arr.dtype.str,
                              "shape": list(arr.shape)}).encode("utf-8"))
    h.update(arr.tobytes())
    return "sha256:" + h.hexdigest()


def _host_complex_array(name: str, value: Any, shape: tuple[int, ...]
                        ) -> np.ndarray:
    module = type(value).__module__ or ""
    if module.startswith("cupy"):
        raise InitialStateError(
            f"{name} must be a host NumPy array; convert device arrays "
            "explicitly with cupy.asnumpy(...)")
    if not isinstance(value, np.ndarray):
        raise InitialStateError(
            f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.complex128:
        raise InitialStateError(
            f"{name} must have dtype complex128 (the stored coefficient "
            f"dtype), got {value.dtype}; convert explicitly")
    if tuple(value.shape) != shape:
        raise InitialStateError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}")
    frozen = np.array(value, dtype=np.complex128, order="C", copy=True)
    frozen.flags.writeable = False
    return frozen


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class InitialState:
    """A resolved BVE or SWE spectral initial state with its specifications.

    Parameters
    ----------
    solver : {"bve", "swe"}
    coeffs : numpy.ndarray
        The packed complex128 frame: ``(L+1, L+1)`` (BVE ``zeta``, s^-1) or
        ``(3, L+1, L+1)`` (SWE ``[zeta, delta, phi]``; s^-1, s^-1,
        m^2 s^-2). Copied and frozen.
    planet : PlanetSpec
    discretization : DiscretizationSpec
        The discretization the state was resolved for; ``lmax`` must equal
        the coefficients' ``L``. A run may use another grid with the same
        ``lmax`` (the coefficients are grid-independent).
    fluid : FluidLayerSpec
        SWE only: explicit gravity and mean thickness H. ``phi`` is the
        perturbation thickness geopotential: total thickness
        ``h = H + phi / g``.
    topography : TopographySpec
        SWE only; default flat.
    topography_lm : numpy.ndarray or None
        SWE over non-flat terrain: the resolved elevation coefficients (m)
        the model must use, complex128 ``(L+1, L+1)``. ``None`` for flat.
    label : str
        Descriptive name; never part of the identity.
    provenance : mapping
        Descriptive, strict-JSON-serializable record of how the state was
        made (preset export, preparation report, ...); never part of the
        identity.
    """

    solver: str
    coeffs: np.ndarray
    planet: PlanetSpec
    discretization: DiscretizationSpec
    fluid: FluidLayerSpec | None = None
    topography: TopographySpec | None = None
    topography_lm: np.ndarray | None = None
    label: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.solver not in INITIAL_STATE_SOLVERS:
            raise InitialStateError(
                f"solver must be one of {INITIAL_STATE_SOLVERS}, got "
                f"{self.solver!r}")
        for name, value, cls in (("planet", self.planet, PlanetSpec),
                                 ("discretization", self.discretization,
                                  DiscretizationSpec)):
            if not isinstance(value, cls):
                raise InitialStateError(
                    f"{name} must be a {cls.__name__}, got "
                    f"{type(value).__name__}")
        lmax = self.discretization.lmax
        n = lmax + 1
        frame = (n, n) if self.solver == "bve" else (3, n, n)
        object.__setattr__(self, "coeffs",
                           _host_complex_array("coeffs", self.coeffs, frame))

        if self.solver == "bve":
            if self.fluid is not None or self.topography is not None \
                    or self.topography_lm is not None:
                raise InitialStateError(
                    "a BVE initial state takes no fluid layer or topography")
        else:
            if not isinstance(self.fluid, FluidLayerSpec):
                raise InitialStateError(
                    "an SWE initial state needs fluid=FluidLayerSpec(gravity, "
                    "mean depth); gravity is never derived from mass and "
                    "radius")
            topography = self.topography or TopographySpec.flat()
            if not isinstance(topography, TopographySpec):
                raise InitialStateError(
                    f"topography must be a TopographySpec, got "
                    f"{type(topography).__name__}")
            object.__setattr__(self, "topography", topography)
            if topography.is_flat:
                if self.topography_lm is not None:
                    raise InitialStateError(
                        "flat topography takes no resolved coefficients")
            else:
                if self.topography_lm is None:
                    raise InitialStateError(
                        f"{topography.preset!r} topography needs its resolved "
                        "elevation coefficients (topography_lm); the run uses "
                        "them exactly instead of re-projecting")
                lm = _host_complex_array("topography_lm", self.topography_lm,
                                         (n, n))
                if not np.any(lm):
                    raise InitialStateError(
                        f"{topography.preset!r} topography has all-zero "
                        "coefficients; use TopographySpec.flat()")
                object.__setattr__(self, "topography_lm", lm)

        if not isinstance(self.label, str):
            raise InitialStateError("label must be a string")
        provenance = dict(self.provenance or {})
        _canonical_json(provenance)        # strict JSON, or raise
        object.__setattr__(self, "provenance", copy.deepcopy(provenance))
        self.validate()

    # -- shape and schema ----------------------------------------------------

    @property
    def l_max(self) -> int:
        return self.discretization.lmax

    def run_config_entries(self) -> dict:
        """Run-configuration keys describing this state's physics.

        The subset of a custom run's configuration that the archive schema
        consumes (fields, geometry, environment); the run layer adds the
        execution settings.
        """
        entries: dict = {
            "solver": self.solver,
            "scenario": CUSTOM_SCENARIO,
            **self.discretization.run_config_entries(),
            "day_hours": self.planet.day_hours,
            "planet": self.planet.to_dict(),
            "initial_state_format": INITIAL_STATE_FORMAT_VERSION,
            "initial_state_digest": self.digest,
            "initial_state_label": self.label,
        }
        if self.solver == "swe":
            entries["gravity"] = self.fluid.gravity_m_s2
            entries["mean_depth_m"] = self.fluid.mean_depth_m
            entries.update(self.topography.run_config_entries())
            if self.topography_lm is not None:
                entries["topography_digest"] = array_digest(self.topography_lm)
        return entries

    @property
    def schema(self) -> StateSchema:
        """The archive ``StateSchema`` of the stored frame (fields, units)."""
        return state_schema_for(self.solver, self.run_config_entries(),
                                provenance={"source": "initial_state"})

    @property
    def state(self) -> SpectralState:
        """Read-only named views (``state["zeta"].coeffs`` ...)."""
        return SpectralState(self.coeffs, self.schema.fields)

    @property
    def topography_modes(self) -> SpectralModes | None:
        if self.topography_lm is None:
            return None
        return SpectralModes(self.topography_lm, TOPOGRAPHY_FIELD)

    def validate(self) -> "InitialState":
        """Coefficient-level convention checks (CPU); raise, never repair."""
        try:
            view = SpectralState(self.coeffs, self.schema.fields)
            for name in view:
                view[name].validate()
            if self.topography_lm is not None:
                SpectralModes(self.topography_lm, TOPOGRAPHY_FIELD).validate()
        except (SpectralConventionError, SchemaError) as err:
            raise InitialStateError(
                f"initial state violates the coefficient conventions: {err}"
            ) from err
        return self

    # -- identity ------------------------------------------------------------

    def identity(self) -> dict:
        """The canonical description the digest is computed from."""
        identity: dict = {
            "format": _IDENTITY_FORMAT,
            "version": INITIAL_STATE_FORMAT_VERSION,
            "solver": self.solver,
            "planet": self.planet.to_dict(),
            "state": array_digest(self.coeffs),
        }
        if self.solver == "swe":
            identity["fluid"] = self.fluid.to_dict()
            identity["topography"] = {
                "spec": self.topography.to_dict(),
                "elevation_lm": (None if self.topography_lm is None
                                 else array_digest(self.topography_lm)),
            }
        return identity

    @property
    def digest(self) -> str:
        """``sha256:<hex>`` scientific identity (see the module docstring)."""
        blob = _canonical_json(self.identity()).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    # -- description ---------------------------------------------------------

    def summary(self) -> dict:
        """Host-only description: specifications, digest, per-field numbers."""
        view = self.state
        out = {
            "solver": self.solver,
            "label": self.label,
            "digest": self.digest,
            "l_max": self.l_max,
            "planet": self.planet.describe(),
            "discretization": self.discretization.describe(),
            "fields": {name: {"units": view[name].units,
                              **view[name].summary()} for name in view},
        }
        if self.solver == "swe":
            out["fluid"] = {"gravity_m_s2": self.fluid.gravity_m_s2,
                            "mean_depth_m": self.fluid.mean_depth_m,
                            "phi0_m2_s2": self.fluid.phi0}
            out["topography"] = self.topography.to_dict()
        return out

    def __repr__(self) -> str:
        return (f"InitialState(solver={self.solver!r}, l_max={self.l_max}, "
                f"label={self.label!r}, digest={self.digest[:19]}...)")

    # -- persistence ---------------------------------------------------------

    def to_document(self) -> dict:
        """The ``initial_state.json`` document (strict JSON)."""
        topography = None
        if self.solver == "swe":
            topography = {
                "spec": self.topography.to_dict(),
                "coefficient_file": (None if self.topography_lm is None
                                     else TOPOGRAPHY_FILE),
                "sha256": (None if self.topography_lm is None
                           else array_digest(self.topography_lm)),
                "field": TOPOGRAPHY_FIELD.to_dict(),
            }
        return {
            "format": INITIAL_STATE_FORMAT,
            "version": INITIAL_STATE_FORMAT_VERSION,
            "solver": self.solver,
            "label": self.label,
            "digest": self.digest,
            "planet": self.planet.to_dict(),
            "discretization": self.discretization.to_dict(),
            "fluid": None if self.fluid is None else self.fluid.to_dict(),
            "topography": topography,
            "state": {
                "coefficient_file": STATE_FILE,
                "dtype": "complex128",
                "shape": list(self.coeffs.shape),
                "sha256": array_digest(self.coeffs),
                "normalization": COEFFICIENT_NORMALIZATION,
                "layout": COEFFICIENT_LAYOUT,
                "state_schema": self.schema.to_manifest_dict(),
            },
            "provenance": copy.deepcopy(dict(self.provenance)),
        }

    def save(self, path, *, overwrite: bool = False) -> pathlib.Path:
        """Write this state to the directory ``path`` (see module docstring)."""
        return save_initial_state(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path) -> "InitialState":
        return load_initial_state(path)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _software() -> dict:
    try:
        from importlib.metadata import version
        package_version = version("palintropos")
    except Exception:  # noqa: BLE001 - descriptive only
        package_version = None
    return {"package": "palintropos", "version": package_version}


def _write_npy(path: pathlib.Path, array: np.ndarray) -> None:
    with open(path, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array, dtype=_DTYPE),
                allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())


def save_initial_state(state: InitialState, path, *,
                       overwrite: bool = False) -> pathlib.Path:
    """Write ``state`` to the directory ``path``; return the path.

    The directory is assembled beside its destination and moved into place,
    so an interrupted save never leaves a partial state under ``path``. An
    existing ``path`` is replaced only with ``overwrite=True`` and only if it
    is itself a saved initial state.
    """
    if not isinstance(state, InitialState):
        raise InitialStateError(
            f"expected an InitialState, got {type(state).__name__}")
    target = pathlib.Path(path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True to replace it")
        if not (target.is_dir() and (target / METADATA_FILE).is_file()):
            raise InitialStateError(
                f"refusing to overwrite {target}: it is not a saved initial "
                "state directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    document = state.to_document()
    document["created_utc"] = datetime.now(timezone.utc).isoformat()
    document["software"] = _software()
    text = json.dumps(document, indent=2, allow_nan=False)

    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_npy(staging / STATE_FILE, state.coeffs)
        if state.topography_lm is not None:
            _write_npy(staging / TOPOGRAPHY_FILE, state.topography_lm)
        with open(staging / METADATA_FILE, "w", encoding="utf-8",
                  newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        with contextlib.suppress(OSError):
            shutil.rmtree(staging)
        raise
    return target


def _require_mapping(value: Any, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise InitialStateError(
            f"{label} must be an object, got {type(value).__name__}")
    return value


def _load_array(directory: pathlib.Path, block: Mapping, expected_file: str,
                label: str) -> np.ndarray:
    name = block.get("coefficient_file")
    if name != expected_file:
        raise InitialStateError(
            f"{label} coefficient_file must be {expected_file!r}, got {name!r}")
    path = directory / expected_file
    if not path.is_file():
        raise InitialStateError(f"{label} coefficients are missing: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as err:
        raise InitialStateError(
            f"{label} coefficients are not a readable .npy array: {path}: "
            f"{err}") from err
    if not isinstance(array, np.ndarray) or array.dtype != _DTYPE:
        raise InitialStateError(
            f"{label} coefficients must be little-endian complex128, got "
            f"{getattr(array, 'dtype', type(array).__name__)}")
    if "shape" in block and list(array.shape) != list(block["shape"]):
        raise InitialStateError(
            f"{label} coefficients have shape {list(array.shape)}, the "
            f"document records {block['shape']}")
    recorded = block.get("sha256")
    if recorded != array_digest(array):
        raise InitialStateError(
            f"{label} coefficients do not match the recorded SHA-256 "
            f"({recorded!r}); the file was modified or replaced")
    return array.astype(np.complex128, copy=False)


def _check_block(block: Mapping, label: str, keys: Mapping) -> None:
    """Known keys only; fixed convention entries must be the expected ones."""
    unknown = set(block) - set(keys)
    if unknown:
        raise InitialStateError(f"{label} has unknown key(s) {sorted(unknown)}")
    for key, expected in keys.items():
        if expected is not None and key in block and block[key] != expected:
            raise InitialStateError(
                f"{label} declares {key} {block[key]!r}; this reader "
                f"implements {expected!r} only")


def _reject_constant(token: str):
    raise InitialStateError(
        f"{METADATA_FILE} contains the non-JSON constant {token}; the format "
        "is strict JSON")


def load_initial_state(path) -> InitialState:
    """Read a saved initial state (host only; never imports CuPy).

    Every recorded property is verified rather than trusted: the format and
    version, each specification block, array dtypes/shapes/hashes, the
    ``state_schema`` block (re-derived and compared), the coefficient
    conventions, and finally the identity digest.
    """
    directory = pathlib.Path(path)
    meta = directory / METADATA_FILE
    if not meta.is_file():
        raise InitialStateError(
            f"{directory} is not a saved initial state (no {METADATA_FILE})")
    try:
        document = json.loads(meta.read_text(encoding="utf-8"),
                              parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise InitialStateError(f"malformed {meta}: {err}") from err
    document = _require_mapping(document, METADATA_FILE)
    if document.get("format") != INITIAL_STATE_FORMAT:
        raise InitialStateError(
            f"{meta} declares format {document.get('format')!r}, expected "
            f"{INITIAL_STATE_FORMAT!r}")
    if document.get("version") != INITIAL_STATE_FORMAT_VERSION:
        raise InitialStateError(
            f"{meta} has format version {document.get('version')!r}; this "
            f"reader supports version {INITIAL_STATE_FORMAT_VERSION}")
    unknown = set(document) - _DOCUMENT_KEYS
    if unknown:
        raise InitialStateError(
            f"{meta} has unknown key(s) {sorted(unknown)}")
    solver = document.get("solver")
    if solver not in INITIAL_STATE_SOLVERS:
        raise InitialStateError(f"{meta} names unknown solver {solver!r}")

    try:
        planet = PlanetSpec.from_dict(document.get("planet"))
        discretization = DiscretizationSpec.from_dict(
            document.get("discretization"))
        fluid = (None if document.get("fluid") is None
                 else FluidLayerSpec.from_dict(document["fluid"]))
    except SpecificationError as err:
        raise InitialStateError(f"{meta}: {err}") from err

    state_block = _require_mapping(document.get("state"), "state block")
    _check_block(state_block, "state block", {
        "coefficient_file": None, "dtype": "complex128", "shape": None,
        "sha256": None, "normalization": COEFFICIENT_NORMALIZATION,
        "layout": COEFFICIENT_LAYOUT, "state_schema": None})
    coeffs = _load_array(directory, state_block, STATE_FILE, "state")

    topography = None
    topography_lm = None
    topo_block = document.get("topography")
    if topo_block is not None:
        topo_block = _require_mapping(topo_block, "topography block")
        _check_block(topo_block, "topography block", {
            "spec": None, "coefficient_file": None, "sha256": None,
            "field": TOPOGRAPHY_FIELD.to_dict()})
        try:
            topography = TopographySpec.from_dict(topo_block.get("spec"))
        except SpecificationError as err:
            raise InitialStateError(f"{meta}: {err}") from err
        if topo_block.get("coefficient_file") is not None:
            topography_lm = _load_array(directory, topo_block,
                                        TOPOGRAPHY_FILE, "topography")

    label = document.get("label", "")
    provenance = document.get("provenance") or {}
    state = InitialState(solver=solver, coeffs=coeffs, planet=planet,
                         discretization=discretization, fluid=fluid,
                         topography=topography, topography_lm=topography_lm,
                         label=label if isinstance(label, str) else "",
                         provenance=_require_mapping(provenance,
                                                     "provenance"))

    recorded_schema = state_block.get("state_schema")
    try:
        StateSchema.from_manifest_dict(recorded_schema)
    except SchemaError as err:
        raise InitialStateError(
            f"{meta}: unreadable state_schema block: {err}") from err
    expected_schema = state.schema.to_manifest_dict()
    if json.loads(_canonical_json(recorded_schema)) != json.loads(
            _canonical_json(expected_schema)):
        raise InitialStateError(
            f"{meta}: the recorded state_schema block does not match the "
            "schema derived from the recorded specifications")
    if document.get("digest") != state.digest:
        raise InitialStateError(
            f"{meta}: recorded digest {document.get('digest')!r} does not "
            f"match the content digest {state.digest!r}")
    return state

