# Held–Suarez report: hs-T42L20-prod-001

- config_sha256: `4cd2eeae9b89ba095609f2675f1951a0e33f1025ee9e67e0b52bc1546e155802`
- git_commit: `206ef25f6503f4e8f7230bae00eb6c79fb639341`
- effective truncation (retained degrees): T42

- reference own tier B: PASS (failed []; ['B4 not checkable (no KE spectrum stored)', 'B5 by construction (single unmodified run, meta.json script_sha256)', 'B2 at endpoints only: rel. change 3.226e-04'])

| tier | verdict |
|---|---|
| A | **INCOMPLETE** |
| B | **FAIL** |
| C | **PASS** |

## Tier A: INCOMPLETE

| criterion | verdict | detail | values |
|---|---|---|---|
| A1 | INCOMPLETE | no full-suite result / tests diff evidence |  |
| A2 | INCOMPLETE | no new-test results |  |
| A3 | INCOMPLETE | T21 L10 2-day SI smoke, forced stop + resume at day 1, bit-identical | bit_identical=None |
| A4 | INCOMPLETE | branch pushed; notebook SOLVER_COMMIT is a pushed ancestor of HEAD | pushed_ancestor=None; solver_commit=None |

## Tier B: FAIL

| criterion | verdict | detail | values |
|---|---|---|---|
| B1 | PASS | reaches day 1200 without abort/NaN; every resume hash-verified | aborted=0; days_completed=1200; days_total=1200; resumes=0; unverified_resumes=0 |
| B2 | FAIL | max_d \|<p_s>(d) - <p_s>(0)\| / <p_s>(0) <= 1e-3 (Gaussian-weighted mean of exp(ln p_s)) | final_rel_change=0.00154779; max_rel_change=0.00154779; threshold=0.001 |
| B3 | PASS | linear trend across block means: KE <= 10 % of mean, <T> <= 0.5 K | T_block_means=[246.251, 246.203, 246.182, 246.122, 246.176]; T_trend_K=0.115076; ke_block_means=[126.775, 128.693, 129.408, 131.801, 127.527]; ke_trend=2.3061; ke_trend_frac=0.0178988 |
| B4 | PASS | log-log KE slope over (28, 42] < 0 and KE(42) < KE(28); L = retained (effective) truncation | L=42; ke_L=0.00290305; ke_l0=0.181245; l0=28; slope=-10.5526 |
| B5 | PASS | one configuration hash and one clean commit across every resume | config_sha256=[4cd2eeae9b89ba095609f2675f1951a0e33f1025ee9e67e0b52bc1546e155802]; event_log=True; git_commits=[206ef25f6503f4e8f7230bae00eb6c79fb639341]; git_dirty=False |

## Tier C: PASS

| criterion | verdict | detail | values |
|---|---|---|---|
| C1 | PASS | zonal-mean u: pattern corr >= 0.98, mass-weighted RMSE <= 2.0 m/s | corr=0.993649; corr_verdict=PASS; rmse=1.07279; rmse_verdict=PASS; sampling_rms=1.09008 |
| C2 | PASS | jet max: magnitude +-10 %, latitude +-4 deg, level +-1 (per hemisphere) | NH={noise: [0.660802, 2.23243, 0], ours: [32.966, 43.2542, 4], reference: [32.8234, 46.0447, 4], verdicts: [PASS, PASS, PASS]}; SH={noise: [0.568946, 2.95324, 0], ours: [32.7458, 43.2542, 4], reference: [32.6319, 40.4636, 4], verdicts: [PASS, PASS, PASS]} |
| C3 | PASS | zonal-mean T mass-weighted RMSE <= 1.5 K | rmse=0.327975; sampling_rms=0.27015 |
| C4 | PASS | [T'^2], [v'T']: peak +-20 %, peak lat +-5 deg, pattern corr >= 0.9 | TsTs={NH: {ours: [41.3165, 40.4636], reference: [41.8264, 40.4636], verdicts: [PASS, PASS]}, SH: {ours: [39.6292, 37.6731], reference: [40.2324, 37.6731], verdicts: [PASS, PASS]}, corr: 0.992773, corr_verdict: PASS}; vsTs={NH: {ours: [21.5824, 37.6731], reference: [21.4617, 37.6731], verdicts: [PASS, PASS]}, SH: {ours: [21.1427, 37.6731], reference: [21.4397, 34.8825], verdicts: [PASS, PASS]}, corr: 0.998005, corr_verdict: PASS} |
| C5 | PASS | \|NH - SH\| jet: FAIL only if > 10 % and > 3 sigma_block | latitude={mean: 0.558109, per_block: [0, 0, 2.79055, 0, 0], sigma_block: 1.24797, verdict: PASS}; magnitude={mean: 0.459172, per_block: [0.935047, 0.738224, 0.405611, 0.0414135, 0.175566], sigma_block: 0.374883, verdict: PASS} |
| C6 | INFO | paper figures 1-3 (no verdict weight) | jet_consistent=True; jet_lat=43.2542; jet_ms=32.966; jet_sigma=0.225; surface_easterlies=True; tropical_surface_u_mean=-5.555; vT_peak_lat=37.6731; vT_peak_sigma=0.875 |
