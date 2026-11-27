# Consolidation evidence (machine-readable)

T4 measurements of the substrate consolidation. The audit and the earlier
evidence (baseline, kernel census, prototypes) are in the decision record,
PR #74 (`docs/validation/CONSOLIDATION-AUDIT.md` there).

| File | What it records |
| --- | --- |
| `t4-native-gate-7a47d2b.json` | The migration gate of the native C++ step (`tools/colab/substrate_gate.py` at `7a47d2b`) on a Colab Tesla T4: the representative state regenerated bit for bit (`ef413855…`); 200 production steps with 0 state, report or Newton-count mismatches against the CuPy reference; the CPU library within the G-GPU bound; timings of the complete production runner, native against reference |
| `t4-native-gate-83320e0.json` | The same gate after stage E1 (each Newton residual evaluated once): state still bitwise (`ef413855…`), 200 steps with 0 mismatches, `run_block` **22.4 ms/step** (P2 met), memory pool 115 MB, cold compile 3.1 s |
| `t4-native-gate-a07b0b5.json` | Gate after caching the per-substrate preparation: state bitwise (`ef413855…`), 200 production steps with 0 mismatches, **terminator batches (VPW99 and constant properties, 16200 columns, 200 steps each) with 0 mismatches**, `run_block` **21.0 ms/step**, memory pool and compile as before |
| `t4-ceres-fixture-d81d547.json` | The Ceres finite-transient fixture end to end through `AirlessRun` on the GPU (8640 steps), native against reference: 122 arrays and values, **0 differing** |
| `t4-census-d81d547.json` | Nsight census (P3 protocol): per step as the difference of 24- and 96-step traces (setup and teardown cancel). Between outputs: **122 kernel launches** and **2 synchronizations**, each a device-to-host copy (validation, and the iteration/failure summary). Output events: +1.0 launch and +0.21 synchronizations per step at the production cadence. Launches by NVTX phase (nested ranges; approximately): forcing 14, step validation and preparation ~13, native kernel 3, report 33, extents 18, runner accumulation ~33 |

An earlier census at `0c00121` counted each device-to-host copy a second time as a synchronization, and divided the whole `run_block` call, setup and teardown included, by the step count ("3.0 + 3.0 per step"). It was superseded by `t4-census-d81d547.json` and removed. A census at the production cadence in that session had failed to parse and read as zero; such failures now raise.
