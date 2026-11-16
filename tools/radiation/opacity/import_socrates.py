"""Import a SOCRATES look-up spectral file as a premixed Palintropos k-table.

    python -I tools/radiation/opacity/import_socrates.py --case thai_ben2
    python -I tools/radiation/opacity/import_socrates.py --case thai_ben1
    python -I tools/radiation/opacity/import_socrates.py \
        --spectral-file <path> --composition <preset> --name <table name>

Reads a spectral file and its ``_k`` companion (:mod:`socrates`), keeps the
absorbers and generalised continua whose gases are present in the given
fixed composition, and writes one :class:`~tropoi.physics.radiation.kdist.
KTable` per kg of that mixture:

    k_mix(p, T) = sum_gas q_gas k_gas(p, T; x_gas)
                + sum_cont q_1 q_2 rho(p, T) k_cont(T),   rho = p / (R_mix T)

with ``q`` mass fractions, ``x`` mole fractions, ``k_gas`` in m^2 per kg of
the gas and ``k_cont`` in m^5 kg^-2 (SOCRATES' generalised-continuum
convention: mass mixing ratios of the two gases times the air density).

How terms combine inside a band is what the spectral file itself
prescribes, not a choice made here:

* a continuum flagged as perfectly correlated with a gas (overlap = that
  gas's index) is added term by term to that gas's k-terms;
* everything else overlaps randomly: the points of the band are the
  products of the terms, weights multiplied and k added.

This is NOT a premix on a common wavenumber grid - a k-term file no longer
holds the spectrum, so that cannot be done from it. Where a band has a
single gas term or a single continuum term the product is exact; the
bands where the random-overlap assumption actually acts are listed in the
table's metadata.

The table's metadata records the source files and their SHA-256, units and
mass basis, coverage, and what the source documents about broadening, far
wings and continua (``SOURCES`` below; statements the documentation does
not make are recorded as not stated). The table's own identity is
``KTable.content_sha256()``, printed at the end and pinned by the presets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import socrates  # noqa: E402
from tropoi.physics.composition import COMPOSITIONS, Composition  # noqa: E402
from tropoi.physics.radiation.kdist import KTable  # noqa: E402

BASE_URL = "https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/spectral_files"
DOC_URL = "https://simplex.giss.nasa.gov/gcm/ROCKE-3D/Spectralfiles.html"
#: SHA-256 of the documentation page as read on 2026-11-16.
DOC_SHA256 = "ff20b814dd80761e9ebf3e20cb74d914027d5c368a0bf026001d47e1a9a983a6"

SOCRATES_SPECIES = {"Water Vapour": "H2O", "Carbon Dioxide": "CO2", "Methane": "CH4",
                    "Nitrogen": "N2", "Hydrogen": "H2", "Oxygen": "O2", "Ozone": "O3",
                    "Nitrous Oxide": "N2O", "Carbon Monoxide": "CO", "Argon": "Ar"}

_COMMON = {
    "line_data": "HITRAN 2012 (Rothman et al. 2013, JQSRT 130, 4)",
    "generator": "SOCRATES spectral files prepared for ROCKE-3D by D. S. Amundsen "
                 "(github.com/DavidSAmundsen/socrates_tools)",
    "documented_ranges": "opacity look-up tables 100-400 K; Planck table 1-500 K; "
                         "long-wave accuracy decreases above about 310 K",
}
_MARS = {
    **_COMMON,
    "directory": "sp_lw_dsa_mars",
    "documented_validity": "CO2-dominated atmospheres with > 90 % CO2, surface pressures up "
                           "to 10 bar",
    "broadening": "CO2 lines self-broadened (look-up axis of CO2 fraction 0.9 and 1.0); "
                  "H2O broadened by air (H2O is absent from the dry cases)",
    "far_wings": "CO2 sub-Lorentzian line shapes (Perrin & Hartmann 1989, JQSRT 42, 311; "
                 "Wordsworth et al. 2010, Icarus 210, 992). The truncation distance is not "
                 "stated in the documentation",
    "continua": "CO2-CO2 collision-induced absorption (Gruszka & Borysow 1998, Mol. Phys. "
                "93, 1007; Baranov et al. 2004, J. Mol. Spectrosc. 228, 432; Wordsworth et "
                "al. 2010). H2O continuum MT_CKD 3.0 (unused: no water)",
    "max_mole_fraction": {},
}
_AR10BAR = {
    **_COMMON,
    "directory": "sp_lw_dsa_ar10bar",
    "documented_validity": "atmospheres dominated by a mixture of N2 and H2 with up to 1 % "
                           "CO2 and CH4, surface pressures up to 10 bar",
    "broadening": "CO2 lines broadened by air (HITRAN air widths; the file's single "
                  "self-broadening entry is CO2 fraction 0), used here for an N2 background: "
                  "N2 broadening is approximated by air broadening",
    "far_wings": "not stated in the documentation for this file (the CO2 sub-Lorentzian "
                 "line shape is listed only for the CO2-rich files)",
    "continua": "N2-N2 collision-induced absorption from HITRAN (Richard et al. 2012, JQSRT "
                "113, 1276). N2-CO2 CIA is NOT included (the documentation: data not found "
                "in the literature). N2-CH4, N2-H2, H2-CH4, H2-H2, H2-CO2 CIA and the H2O "
                "continuum are in the file but unused (those gases are absent)",
    "max_mole_fraction": {"CO2": 0.01, "CH4": 0.01},
}
#: What the ROCKE-3D documentation states about each supported spectral file.
SOURCES = {
    "sp_lw_17_dsa_mars": {**_MARS, "stream": "longwave",
                          "documented_accuracy": "about 1 W m^-2 for temperatures below "
                                                 "about 310 K, against sp_lw_350_dsa_mars"},
    "sp_lw_350_dsa_mars": {**_MARS, "stream": "longwave",
                           "documented_accuracy": "the high-resolution reference of "
                                                  "sp_lw_17_dsa_mars"},
    "sp_lw_15_dsa_ar10bar": {**_AR10BAR, "stream": "longwave",
                             "documented_accuracy": "about 1 W m^-2 for temperatures below "
                                                    "about 310 K, against "
                                                    "sp_lw_350_dsa_ar10bar"},
    "sp_lw_350_dsa_ar10bar": {**_AR10BAR, "stream": "longwave",
                              "documented_accuracy": "the high-resolution reference of "
                                                     "sp_lw_15_dsa_ar10bar"},
}
_STAR = ("TRAPPIST-1 band fluxes as normalised in block 2 of the spectral file (the "
         "documentation does not name the stellar spectrum behind them)")
_SW_NOTE = {"stream": "shortwave", "star": _STAR,
            "documented_accuracy": "below about 5 W m^-2 for all stars, against the "
                                   "280-band reference"}
SOURCES.update({
    "sp_sw_42_dsa_mars_trappist1": {
        **_MARS, **_SW_NOTE, "directory": "sp_sw_dsa_mars",
        "continua": _MARS["continua"] + ". The file's generalised Rayleigh scattering "
                    "(per-gas coefficients) is recorded in the metadata and not applied"},
    "sp_sw_43_dsa_ar10bar_trappist1": {
        **_AR10BAR, **_SW_NOTE, "directory": "sp_sw_dsa_ar10bar",
        "continua": "the documentation lists N2-CH4, N2-N2, N2-H2, H2-CH4, H2-H2 and "
                    "H2-CO2 CIA for these files; whichever involve only the gases present "
                    "are kept (see the per-band components). The file's generalised "
                    "Rayleigh scattering is recorded in the metadata and not applied"},
})

#: The THAI dry cases: composition preset and spectral file.
CASES = {
    "thai_ben2_shortwave": {"composition": "thai_ben2",
                            "spectral_file": "sp_sw_dsa_mars/sp_sw_42_dsa_mars_trappist1"},
    "thai_ben1_shortwave": {
        "composition": "thai_ben1",
        "spectral_file": "sp_sw_dsa_ar10bar/sp_sw_43_dsa_ar10bar_trappist1"},
    "thai_ben2": {"composition": "thai_ben2",
                  "spectral_file": "sp_lw_dsa_mars/sp_lw_17_dsa_mars"},
    "thai_ben2_reference": {"composition": "thai_ben2",
                            "spectral_file": "sp_lw_dsa_mars/sp_lw_350_dsa_mars"},
    "thai_ben1": {"composition": "thai_ben1",
                  "spectral_file": "sp_lw_dsa_ar10bar/sp_lw_15_dsa_ar10bar"},
    "thai_ben1_reference": {"composition": "thai_ben1",
                            "spectral_file": "sp_lw_dsa_ar10bar/sp_lw_350_dsa_ar10bar"},
}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _interp_temperature(values: np.ndarray, t_from: np.ndarray, t_to: np.ndarray):
    """Linear in T along the last axis; ``t_to`` must lie inside ``t_from``."""
    if t_to[0] < t_from[0] or t_to[-1] > t_from[-1]:
        raise ValueError("continuum table does not cover the gas table's temperatures")
    return np.stack([np.interp(t_to, t_from, row) for row in values])


def premix(sp: socrates.SpectralFile, composition: Composition, *, name: str,
           source: dict) -> KTable:
    """The premixed table of ``composition`` from the spectral file ``sp``."""
    x = dict(composition.mole_fractions)
    q = dict(composition.mass_fractions)
    species_of = {}
    for idx, label in sp.absorbers.items():
        if label not in SOCRATES_SPECIES:
            raise ValueError(f"unknown SOCRATES absorber {label!r}")
        species_of[idx] = SOCRATES_SPECIES[label]
    for s, limit in source.get("max_mole_fraction", {}).items():
        if x.get(s, 0.0) > limit:
            raise ValueError(f"{s} mole fraction {x[s]} exceeds the documented validity "
                             f"({limit}) of this spectral file")
    if not np.all(sp.temperature == sp.temperature[0]):
        raise ValueError("the look-up temperatures differ between pressures; not supported")
    pressure, temperature = sp.pressure, sp.temperature[0]
    rho = pressure[:, None] / (composition.gas_constant * temperature[None, :])
    edges = sp.band_edges_cm()

    band_terms, report, fraction_used = [], [], {}
    for b in range(1, sp.n_band + 1):
        gas_terms: dict[int, list] = {}
        for g in sp.band_gases[b]:
            s = species_of[g]
            if q.get(s, 0.0) <= 0.0:
                continue
            kt = sp.gas_k[(b, g)]                               # (n_k, n_block, n_p, n_T)
            if kt.shape[1] > 1:
                gf = sp.gas_fraction
                if not (gf[0] <= x[s] <= gf[-1]):
                    raise ValueError(f"{s} mole fraction {x[s]} is outside the "
                                     f"self-broadening axis {gf.tolist()}")
                i = min(int(np.searchsorted(gf, x[s], side="right")) - 1, gf.size - 2)
                w = (x[s] - gf[i]) / (gf[i + 1] - gf[i])
                k = (1.0 - w) * kt[:, i] + w * kt[:, i + 1]
                fraction_used[s] = {"axis": gf.tolist(), "used": x[s]}
            else:
                k = kt[:, 0]
                if sp.self_broadening.get(g, 0) and sp.gas_fraction is not None:
                    fraction_used[s] = {"axis": sp.gas_fraction.tolist(),
                                        "used": float(sp.gas_fraction[0])}
            gas_terms[g] = [q[s] * k, sp.gas_weights[(b, g)].copy(), s]
        random_terms = []
        for (bb, c), w in sp.cont_weights.items():
            if bb != b:
                continue
            s1, s2 = (species_of[i] for i in sp.continua[c])
            amount = q.get(s1, 0.0) * q.get(s2, 0.0)
            if amount <= 0.0:
                continue
            kc = _interp_temperature(sp.cont_k[(b, c)], sp.cont_temperature, temperature)
            k = amount * kc[:, None, :] * rho[None]             # (n_k, n_p, n_T)
            overlap = sp.cont_overlap[(b, c)]
            label = f"{s1}-{s2}"
            if overlap == 0:
                random_terms.append([k, w.copy(), label])
            elif overlap in gas_terms and gas_terms[overlap][0].shape[0] == k.shape[0]:
                gas_terms[overlap][0] = gas_terms[overlap][0] + k
                gas_terms[overlap][2] += f"+{label}(correlated)"
            else:
                raise ValueError(f"band {b}: continuum {label} is correlated with gas "
                                 f"{overlap}, which is absent or has other k-terms")
        comps = list(gas_terms.values()) + random_terms
        k_band = np.zeros((1,) + rho.shape)
        w_band = np.ones(1)
        for k, w, _ in comps:
            k_band = (k_band[:, None] + k[None]).reshape((-1,) + rho.shape)
            w_band = (w_band[:, None] * w[None]).ravel()
        total = float(w_band.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"band {b}: weights sum to {total}")
        band_terms.append((k_band, w_band / total))
        multi = [c[2] for c in comps if c[1].size > 1]
        report.append({"band": b, "wavenumber_cm": [float(edges[b - 1, 0]),
                                                    float(edges[b - 1, 1])],
                       "components": [f"{c[2]}:{c[1].size}" for c in comps],
                       "points": int(w_band.size),
                       "random_overlap_acts": len(multi) > 1})

    # Stellar files list their bands by increasing wavelength; a KTable's
    # bands increase in wavenumber. ``report`` keeps the file's band numbers.
    order = np.argsort(edges[:, 0])
    edges = edges[order]
    band_terms = [band_terms[i] for i in order]
    report = [report[i] for i in order]
    if not np.allclose(edges[1:, 0], edges[:-1, 1], rtol=1e-9):
        raise ValueError("bands are not contiguous in wavenumber")
    stellar = {}
    if source["stream"] == "shortwave":
        if sp.stellar_fraction is None:
            raise ValueError("a stellar spectral file needs the band fluxes of block 2")
        fraction = sp.stellar_fraction[order]
        rayleigh = np.zeros(sp.n_band)
        for gas, coefficient in sp.rayleigh.items():
            rayleigh += q.get(species_of[gas], 0.0) * coefficient[order]
        stellar = {
            "star": source["star"],
            "band_fractions": [float(v) for v in fraction],
            "band_fractions_sum": float(fraction.sum()),
            "rayleigh_m2_per_kg_of_mixture": [float(v) for v in rayleigh],
            "rayleigh_note": "per band, summed over the gases present with their mass "
                             "fractions; NOT applied by the provider (absorption only)",
        }
    n_g = max(w.size for _, w in band_terms)
    weights = np.zeros((sp.n_band, n_g))
    k = np.zeros(rho.shape + (sp.n_band, n_g))
    for b, (kb, wb) in enumerate(band_terms):
        weights[b, :wb.size] = wb
        k[:, :, b, :wb.size] = np.moveaxis(kb, 0, -1)
    band_edges = np.concatenate([edges[:, 0], edges[-1:, 1]])

    path = pathlib.Path(sp.path)
    files = []
    for f in (path, pathlib.Path(str(path) + "_k")):
        files.append({"name": f.name, "url": f"{BASE_URL}/{source['directory']}/{f.name}",
                      "bytes": f.stat().st_size, "sha256": _sha256(f)})
    metadata = {
        "schema": "palintropos.ktable.metadata/1",
        "stream": source["stream"],
        "mass_basis": {"kind": "premixed", "per": "kg of the mixture",
                       "composition": {s: v for s, v in composition.mole_fractions},
                       "composition_label": composition.label,
                       "mass_fractions": q,
                       "molar_mass_kg_per_mol": composition.molar_mass_kg_per_mol,
                       "gas_constant": composition.gas_constant},
        "units": {"k": "m^2 kg^-1 (of mixture)", "pressure": "Pa", "temperature": "K",
                  "band_edges": "cm^-1"},
        "coverage": {"pressure_pa": [float(pressure[0]), float(pressure[-1])],
                     "n_pressure": int(pressure.size),
                     "temperature_k": [float(temperature[0]), float(temperature[-1])],
                     "n_temperature": int(temperature.size),
                     "wavenumber_cm": [float(band_edges[0]), float(band_edges[-1])],
                     "n_band": int(sp.n_band),
                     "n_points": int(sum(r["points"] for r in report)),
                     "documented_validity": source["documented_validity"],
                     "documented_ranges": source["documented_ranges"],
                     "documented_accuracy": source["documented_accuracy"]},
        "source": {"kind": "SOCRATES look-up spectral file", "files": files,
                   "generator": source["generator"],
                   "documentation": {"url": DOC_URL, "sha256": DOC_SHA256,
                                     "read": "2026-11-16"}},
        "assumptions": {
            "line_data": source["line_data"],
            "broadening": source["broadening"],
            "self_broadening_axis": fraction_used,
            "far_wings": source["far_wings"],
            "continua": source["continua"],
            "continuum_density": "k_cont q1 q2 rho with rho = p / (R_mix T) of the stated "
                                 "mixture; continuum look-up (10 K steps) interpolated "
                                 "linearly in T to the gas table's temperatures. Where the "
                                 "source data stop (CO2-CO2 below 200 K) the file itself "
                                 "holds the edge value",
            "overlap": "as flagged in the spectral file: random overlap (products of "
                       "k-terms) between a gas and a continuum, exact where either has one "
                       "term. Not a common-wavenumber-grid premix",
            "random_overlap_bands": [r["band"] for r in report if r["random_overlap_acts"]],
            "thermal_source": "band Planck fractions are computed by the provider "
                              "(planck_band_fraction); SOCRATES' own Planck table is not "
                              "used. Nothing is emitted outside the bands"
            if source["stream"] == "longwave" else "not applicable (stellar stream)",
            "absent_gases_dropped": sorted(
                s for s in set(species_of.values()) if q.get(s, 0.0) <= 0.0),
        },
        "bands": report,
    }
    if stellar:
        metadata["stellar"] = stellar
    provenance = (f"{path.name} (ROCKE-3D SOCRATES) premixed for {composition.label}; "
                  f"source sha256 {files[0]['sha256'][:16]}/{files[1]['sha256'][:16]}")
    return KTable(name, pressure, temperature, band_edges, weights, k, provenance, metadata)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--spectral-file", help="path under --cache, e.g. "
                                            "sp_lw_dsa_mars/sp_lw_17_dsa_mars")
    ap.add_argument("--composition", help="a tropoi composition preset")
    ap.add_argument("--name", help="species label of the table (default: the composition)")
    ap.add_argument("--cache", default="opacity_cache/socrates/rocke3d")
    ap.add_argument("--out-dir", default="opacity_cache/tables")
    ap.add_argument("--record-dir", default=None,
                    help="also write the metadata record (JSON) here")
    args = ap.parse_args(argv)
    if args.case:
        case = CASES[args.case]
        spectral, comp = case["spectral_file"], case["composition"]
    elif args.spectral_file and args.composition:
        spectral, comp = args.spectral_file, args.composition
    else:
        ap.error("give --case, or --spectral-file and --composition")
    path = pathlib.Path(args.cache) / spectral
    if path.name not in SOURCES:
        raise SystemExit(f"{path.name}: no documented source record (add it to SOURCES)")
    composition = COMPOSITIONS[comp]
    sp = socrates.read_spectral_file(path)
    for gas, flag in sp.self_broadening.items():
        if flag and sp.gas_fraction is not None and sp.gas_fraction.size > 1:
            ev = socrates.gas_fraction_order_evidence(sp, gas)
            if not ev["fraction_outer"] < 0.2 * ev["pressure_outer"]:
                raise SystemExit(f"cannot establish the gas-fraction layout: {ev}")
    table = premix(sp, composition, name=args.name or composition.label,
                   source=SOURCES[path.name])
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{composition.label}-{path.name}".replace("_", "-")
    out = out_dir / f"{stem}.npz"
    table.save(out)
    sha = table.content_sha256()
    record = {"file": out.name, "sha256": sha, "species": table.species,
              "provenance": table.provenance, "metadata": table.metadata,
              "command": "python -I tools/radiation/opacity/import_socrates.py "
                         + " ".join(a for a in (argv or sys.argv[1:]))}
    targets = [out.with_suffix(".json")]
    if args.record_dir:
        pathlib.Path(args.record_dir).mkdir(parents=True, exist_ok=True)
        targets.append(pathlib.Path(args.record_dir) / f"{stem}.json")
    for target in targets:
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    for r in table.metadata["bands"]:
        print(f"band {r['band']:3d} {r['wavenumber_cm'][0]:8.1f}-{r['wavenumber_cm'][1]:8.1f}"
              f"  points {r['points']:4d}  {' x '.join(r['components']) or 'transparent'}"
              f"{'   [random overlap]' if r['random_overlap_acts'] else ''}")
    print(f"{out}  points {table.metadata['coverage']['n_points']}  "
          f"k shape {table.k.shape}")
    print(json.dumps({"file": out.name, "sha256": sha}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
