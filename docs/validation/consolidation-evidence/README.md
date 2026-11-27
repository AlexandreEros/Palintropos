# Consolidation evidence (machine-readable)

T4 measurements of the substrate consolidation. The audit and the earlier
evidence (baseline, kernel census, prototypes) are in the decision record,
PR #74 (`docs/validation/CONSOLIDATION-AUDIT.md` there).

| File | What it records |
| --- | --- |
| `t4-native-gate-7a47d2b.json` | The migration gate of the native C++ step (`tools/colab/substrate_gate.py` at `7a47d2b`) on a Colab Tesla T4: the representative state regenerated bit for bit (`ef413855…`); 200 production steps with 0 state, report or Newton-count mismatches against the CuPy reference; the CPU library within the G-GPU bound; timings of the complete production runner, native against reference |
| `t4-native-gate-83320e0.json` | The same gate after stage E1 (each Newton residual evaluated once): state still bitwise (`ef413855…`), 200 steps with 0 mismatches, `run_block` **22.4 ms/step** (P2 met), memory pool 115 MB, cold compile 3.1 s |
| `t4-census-between-outputs-0c00121.json` | Nsight census (P3 protocol, sampling only at the segment end, i.e. the steps between outputs): 134 kernel launches per step, 3.0 synchronizations plus 3.0 device-to-host copies per step, 19.8 ms of kernel time of which 18.8 ms is the fused substrate kernel. P3 is **not met**; see PR E |

A census at the production cadence on the same session reported zero synchronizations, which cannot be right (its CSV parse failed), so it is not recorded.
