# Consolidation evidence (machine-readable)

T4 measurements of the substrate consolidation. The audit and the earlier
evidence (baseline, kernel census, prototypes) are in the decision record,
PR #74 (`docs/validation/CONSOLIDATION-AUDIT.md` there).

| File | What it records |
| --- | --- |
| `t4-native-gate-7a47d2b.json` | The migration gate of the native C++ step (`tools/colab/substrate_gate.py` at `7a47d2b`) on a Colab Tesla T4: the representative state regenerated bit for bit (`ef413855…`); 200 production steps with 0 state, report or Newton-count mismatches against the CuPy reference; the CPU library within the G-GPU bound; timings of the complete production runner, native against reference |
