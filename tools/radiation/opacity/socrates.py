"""Reader for SOCRATES spectral files with look-up k-tables (text format).

A SOCRATES configuration is two text files: the spectral file proper
(``sp_lw_17_dsa_mars``: numbered ``*BLOCK`` sections) and its extended file
(``sp_lw_17_dsa_mars_k``: the pressure-temperature look-up tables). This
module reads only what a gas-absorption import needs:

* block 0  - absorber and generalised-continuum indexing;
* block 1  - band limits (wavelength, m);
* block 2  - (stellar files) the fraction of the stellar flux in each band;
* block 3  - (stellar files, subtype 1) Rayleigh mass scattering
  coefficients per band and gas, m^2 kg^-1;
* block 5  - per (band, gas): number of k-terms and their weights (the
  reference k of block 5 is NOT used; the look-up table replaces it), and
  the self-broadening index of each gas;
* block 18 / 19 - per (band, continuum): k-terms, weights and the overlap
  treatment (0 = random overlap with the gases; n > 0 = perfectly
  correlated with the k-terms of gas number n);
* ``_k``: ``k(T, p[, gas fraction])`` per (band, gas, k-term), in m^2 per kg
  of the absorber, and ``k(T)`` per (band, continuum, k-term), in m^5 kg^-2.

Layout of the ``_k`` k-table, established from the files themselves (line
counts and the header): for each (band, gas) the terms are written in
order; each term holds ``n_gas_fraction`` (1 unless the gas is
self-broadened) consecutive blocks of ``n_p`` records, each record the
``n_T`` values of one pressure. :func:`gas_fraction_order_evidence` measures
which of the two possible nestings of pressure and gas fraction is present
instead of assuming it.

Nothing here interprets the physics; :mod:`import_socrates` does. NumPy
only; no SOCRATES code is used or needed.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SpectralFile", "read_spectral_file", "gas_fraction_order_evidence"]

_NUM = re.compile(r"^[\s0-9.eEdD+\-]+$")


@dataclass
class SpectralFile:
    path: str
    absorbers: dict[int, str]                       # index -> name
    continua: dict[int, tuple[int, int]]            # index -> (gas 1 index, gas 2 index)
    band_wavelength: np.ndarray                     # (n_band, 2) m, lower/upper
    band_gases: dict[int, list[int]]                # band -> gas indices active
    self_broadening: dict[int, int]                 # gas index -> 0 or sb index
    gas_weights: dict[tuple[int, int], np.ndarray]  # (band, gas) -> weights
    cont_weights: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    cont_overlap: dict[tuple[int, int], int] = field(default_factory=dict)
    pressure: np.ndarray = None                     # (n_p,) Pa
    temperature: np.ndarray = None                  # (n_p, n_T) K (per pressure)
    gas_fraction: np.ndarray = None                 # (n_gf,) or None
    gas_k: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    #   (band, gas) -> (n_term, n_block, n_p, n_T) with n_block = n_gf or 1
    cont_temperature: np.ndarray = None             # (n_Tc,) K
    stellar_fraction: np.ndarray = None             # (n_band,) or None
    rayleigh: dict = field(default_factory=dict)    # gas index -> (n_band,) m^2/kg
    cont_k: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    #   (band, continuum) -> (n_term, n_Tc)

    @property
    def n_band(self) -> int:
        return self.band_wavelength.shape[0]

    def band_edges_cm(self) -> np.ndarray:
        """Band limits as wavenumbers (cm^-1), per band (low, high)."""
        lam = self.band_wavelength
        return np.stack([0.01 / lam[:, 1], 0.01 / lam[:, 0]], axis=1)


def _blocks(text: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    current, key = None, None
    for line in text.splitlines():
        m = re.match(r"\*BLOCK: TYPE =\s*(\d+)", line)
        if m:
            key, current = int(m.group(1)), []
            continue
        if line.startswith("*END"):
            if key is not None:
                out[key] = current
            key, current = None, None
            continue
        if current is not None:
            current.append(line)
    return out


def _floats(line: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in line.split()]


def _read_block5(lines, n_gas):
    it = iter(lines)
    self_b: dict[int, int] = {}
    weights: dict[tuple[int, int], np.ndarray] = {}
    for line in it:
        if line.startswith("(Zero means"):
            vals = [int(x) for x in next(it).split()]
            if len(vals) != n_gas:
                raise ValueError("self-broadening record does not list every absorber")
            self_b = {i + 1: v for i, v in enumerate(vals)}
        if "k-terms, weights and scaling parameters" in line:
            break
    rest = list(it)
    i = 0
    while i < len(rest):
        head = rest[i].split()
        if len(head) != 5:
            raise ValueError(f"block 5: unexpected record {rest[i]!r}")
        band, gas, n_k, _scale_type, scale_fnc = (int(x) for x in head)
        if scale_fnc != 9:
            raise ValueError(f"block 5: band {band} gas {gas} uses scaling function "
                             f"{scale_fnc}, not the look-up table (9); not supported")
        i += 2                                   # header, then "n_p n_T"
        w = [_floats(rest[i + j])[1] for j in range(n_k)]
        weights[(band, gas)] = np.asarray(w)
        i += n_k
    return self_b, weights


def _read_block19(lines):
    weights, overlap = {}, {}
    start = next(i for i, ln in enumerate(lines) if "followed by, k-terms" in ln) + 1
    rest = lines[start:]
    i = 0
    while i < len(rest):
        band, cont, n_k, ovl = (int(x) for x in rest[i].split())
        w = [_floats(rest[i + 1 + j])[1] for j in range(n_k)]
        weights[(band, cont)] = np.asarray(w)
        overlap[(band, cont)] = ovl
        i += 1 + n_k
    return weights, overlap


def _read_block3(lines, n_band):
    """Rayleigh coefficients per gas (subtype 1); {} for other layouts."""
    try:
        at = next(i for i, ln in enumerate(lines) if ln.startswith("Indexing numbers"))
    except StopIteration:
        return {}
    gases = [int(x) for x in lines[at + 1].split()]
    values: list[list[float]] = []
    for line in lines[at + 2:]:
        if line.startswith("Band ="):
            values.append([])
        elif _NUM.match(line) and line.strip() and values:
            values[-1].extend(_floats(line))
    if len(values) != n_band or any(len(v) != len(gases) for v in values):
        raise ValueError("block 3: unexpected Rayleigh record layout")
    table = np.asarray(values)
    return {g: table[:, i].copy() for i, g in enumerate(gases)}


def _numbers(lines, start, count):
    """``count`` floats starting at line ``start``; returns (array, next line)."""
    vals: list[float] = []
    i = start
    while len(vals) < count:
        vals.extend(_floats(lines[i]))
        i += 1
    if len(vals) != count:
        raise ValueError(f"expected {count} numbers ending at line {i}, got {len(vals)}")
    return np.asarray(vals), i


def _read_k_file(sp: SpectralFile, text: str) -> None:
    lines = text.splitlines()
    i = 0
    n = len(lines)
    section = None
    n_p = n_t = n_tc = 0
    while i < n:
        line = lines[i]
        if line.startswith("*BLOCK:"):
            section = line.split(":", 1)[1].strip()
            i += 1
            continue
        if section == "k-table":
            m = re.match(r"Lookup table:\s*(\d+) pressures,\s*(\d+) temperatures", line)
            if m:
                n_p, n_t = int(m.group(1)), int(m.group(2))
                arr, i = _numbers(lines, i + 1, n_p * (n_t + 1))
                arr = arr.reshape(n_p, n_t + 1)
                sp.pressure, sp.temperature = arr[:, 0].copy(), arr[:, 1:].copy()
                continue
            m = re.match(r"Lookup table:\s*(\d+) gas fractions", line)
            if m:
                sp.gas_fraction, i = _numbers(lines, i + 1, int(m.group(1)))
                continue
            m = re.match(r"Band:\s*(\d+), gas:\s*(\d+), k-terms:\s*(\d+)", line)
            if m:
                band, gas, n_k = (int(x) for x in m.groups())
                n_block = (sp.gas_fraction.size
                           if sp.self_broadening.get(gas, 0) and sp.gas_fraction is not None
                           else 1)
                arr, i = _numbers(lines, i + 1, n_k * n_block * n_p * n_t)
                sp.gas_k[(band, gas)] = arr.reshape(n_k, n_block, n_p, n_t)
                continue
        elif section == "continuum k-table":
            m = re.match(r"Lookup table:\s*(\d+) temperatures", line)
            if m:
                n_tc = int(m.group(1))
                sp.cont_temperature, i = _numbers(lines, i + 1, n_tc)
                continue
            m = re.match(r"Band:\s*(\d+), continuum:\s*(\d+), k-terms:\s*(\d+)", line)
            if m:
                band, cont, n_k = (int(x) for x in m.groups())
                arr, i = _numbers(lines, i + 1, n_k * n_tc)
                sp.cont_k[(band, cont)] = arr.reshape(n_k, n_tc)
                continue
        i += 1


def read_spectral_file(path) -> SpectralFile:
    """Read ``path`` and its ``_k`` companion."""
    path = pathlib.Path(path)
    blocks = _blocks(path.read_text(encoding="ascii"))
    b0 = blocks[0]
    absorbers: dict[int, str] = {}
    continua: dict[int, tuple[int, int]] = {}
    mode = None
    for line in b0:
        if line.startswith("Index       Absorber"):
            mode = "gas"
            continue
        if line.startswith("Index     Gas 1"):
            mode = "cont"
            continue
        if line.startswith("Listing of continuum"):
            mode = None
            continue
        parts = line.split()
        if mode == "gas" and len(parts) >= 3 and parts[0].isdigit():
            absorbers[int(parts[0])] = " ".join(parts[2:])
        elif mode == "cont" and len(parts) >= 3 and parts[0].isdigit():
            ints = [int(p) for p in parts if p.isdigit()]
            continua[ints[0]] = (ints[1], ints[2])
    lam = np.asarray([_floats(ln)[1:3] for ln in blocks[1]
                      if _NUM.match(ln) and len(ln.split()) == 3])
    band_gases: dict[int, list[int]] = {}
    b4 = [ln for ln in blocks[4] if _NUM.match(ln) and ln.strip()]
    i = 0
    while i < len(b4):
        band, n_abs = (int(x) for x in b4[i].split())
        band_gases[band] = [int(x) for x in b4[i + 1].split()] if n_abs else []
        i += 2 if n_abs else 1
    self_b, gas_w = _read_block5(blocks[5], len(absorbers))
    sp = SpectralFile(str(path), absorbers, continua, lam, band_gases, self_b, gas_w)
    if 19 in blocks:
        sp.cont_weights, sp.cont_overlap = _read_block19(blocks[19])
    if 2 in blocks:
        rows = [_floats(ln) for ln in blocks[2] if _NUM.match(ln) and len(ln.split()) == 2]
        sp.stellar_fraction = np.asarray([r[1] for r in rows])
        if sp.stellar_fraction.size != lam.shape[0]:
            raise ValueError(f"{path}: block 2 does not list every band")
    if 3 in blocks:
        sp.rayleigh = _read_block3(blocks[3], lam.shape[0])
    _read_k_file(sp, pathlib.Path(str(path) + "_k").read_text(encoding="ascii"))
    for key, w in gas_w.items():
        if key not in sp.gas_k or sp.gas_k[key].shape[0] != w.size:
            raise ValueError(f"{path}: band/gas {key} has {w.size} weights but no matching "
                             "look-up table")
    for key, w in sp.cont_weights.items():
        if key not in sp.cont_k or sp.cont_k[key].shape[0] != w.size:
            raise ValueError(f"{path}: band/continuum {key} has no matching look-up table")
    if lam.shape[0] != len(band_gases):
        raise ValueError(f"{path}: {lam.shape[0]} band limits, {len(band_gases)} gas records")
    return sp


def gas_fraction_order_evidence(sp: SpectralFile, gas: int) -> dict:
    """Decide how pressure and gas fraction are nested for a self-broadened
    gas, from the data.

    The file holds ``n_gf * n_p`` records per k-term. Either the records
    are [gas fraction][pressure] (what the reader assumes) or
    [pressure][gas fraction]. In the first reading k varies smoothly along
    pressure inside each block; in the second, consecutive records
    alternate between gas fractions, so the "pressure" axis of the first
    reading would zig-zag. The roughness (mean absolute second difference
    of ln k along the assumed pressure axis) of both readings is returned;
    the smaller one is the layout actually present.
    """
    n_gf = sp.gas_fraction.size
    rough = {"fraction_outer": 0.0, "pressure_outer": 0.0}
    count = 0
    for (band, g), k in sp.gas_k.items():
        if g != gas or k.shape[1] != n_gf:
            continue
        n_k, _, n_p, n_t = k.shape
        flat = np.log(np.maximum(k.reshape(n_k, n_gf * n_p, n_t), 1e-300))
        a = flat.reshape(n_k, n_gf, n_p, n_t)                       # [gf][p]
        b = flat.reshape(n_k, n_p, n_gf, n_t).transpose(0, 2, 1, 3)  # [p][gf]
        rough["fraction_outer"] += float(np.abs(np.diff(a, 2, axis=2)).mean())
        rough["pressure_outer"] += float(np.abs(np.diff(b, 2, axis=2)).mean())
        count += 1
    return {"bands": count, **{k: v / max(count, 1) for k, v in rough.items()}}
