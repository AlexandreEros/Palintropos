# Held–Suarez report: hs-T42L20-prod-002

- config_sha256: `d1fa645db12038f21725bb40fa1f89e036b115ea4b752401a6e24e33f97d381c`
- git_commit: `a52ea20550c1c73bddf3b25c53ee371501f05aa0`
- effective truncation (retained degrees): T42

- reference own tier B: PASS (failed []; ['B4 not checkable (no KE spectrum stored)', 'B5 by construction (single unmodified run, meta.json script_sha256)', 'B2 at endpoints only: rel. change 3.226e-04'])

| tier | verdict |
|---|---|
| A | **PASS** |
| B | **PASS** |
| C | **PASS** |

## Tier A: PASS

| criterion | verdict | detail | values |
|---|---|---|---|
| A1 | PASS | pre-existing tests pass, tests/ diff is additions only, new skips have reasons | additions_only=True; errors=0; failed=0; new_skips=[]; passed=1132; skipped=5 |
| A2 | PASS | every new S1-S6 test passes | failed=[]; n_tests=112 |
| A3 | PASS | T21 L10 2-day SI smoke, forced stop + resume at day 1, bit-identical | bit_identical=True |
| A4 | PASS | branch pushed; notebook SOLVER_COMMIT is a pushed ancestor of HEAD | pushed_ancestor=True; solver_commit=a52ea20550c1c73bddf3b25c53ee371501f05aa0 |

## Tier B: PASS

| criterion | verdict | detail | values |
|---|---|---|---|
| B1 | PASS | reaches day 1200 without abort/NaN; every resume hash-verified | aborted=0; days_completed=1200; days_total=1200; resumes=0; unverified_resumes=0 |
| B2 | PASS | max_d \|<p_s>(d) - <p_s>(0)\| / <p_s>(0) <= 1e-3 (Gaussian-weighted mean of exp(ln p_s)) | final_rel_change=2.01603e-06; max_rel_change=2.01603e-06; threshold=0.001 |
| B3 | PASS | linear trend across block means: KE <= 10 % of mean, <T> <= 0.5 K | T_block_means=[246.301, 246.23, 246.249, 246.259, 246.243]; T_trend_K=0.0429071; ke_block_means=[129.829, 130.469, 129.228, 130.433, 129.258]; ke_trend=0.589375; ke_trend_frac=0.00453913 |
| B4 | PASS | log-log KE slope over (28, 42] < 0 and KE(42) < KE(28); L = retained (effective) truncation | L=42; ke_L=0.00306639; ke_l0=0.182818; l0=28; slope=-10.4529 |
| B5 | PASS | one configuration hash and one clean commit across every resume | config_sha256=[d1fa645db12038f21725bb40fa1f89e036b115ea4b752401a6e24e33f97d381c]; event_log=True; git_commits=[a52ea20550c1c73bddf3b25c53ee371501f05aa0]; git_dirty=False |

## Tier C: PASS

| criterion | verdict | detail | values |
|---|---|---|---|
| C1 | PASS | zonal-mean u: pattern corr >= 0.98, mass-weighted RMSE <= 2.0 m/s | corr=0.995529; corr_verdict=PASS; rmse=0.901799; rmse_verdict=PASS; sampling_rms=1.04839 |
| C2 | PASS | jet max: magnitude +-10 %, latitude +-4 deg, level +-1 (per hemisphere) | NH={noise: [0.801876, 2.8458, 0], ours: [32.5611, 43.2542, 4], reference: [32.8234, 46.0447, 4], verdicts: [PASS, PASS, PASS]}; SH={noise: [0.549842, 2.95324, 0], ours: [32.8041, 43.2542, 4], reference: [32.6319, 40.4636, 4], verdicts: [PASS, PASS, PASS]} |
| C3 | PASS | zonal-mean T mass-weighted RMSE <= 1.5 K | rmse=0.293059; sampling_rms=0.259685 |
| C4 | PASS | [T'^2], [v'T']: peak +-20 %, peak lat +-5 deg, pattern corr >= 0.9 | TsTs={NH: {ours: [39.9566, 40.4636], reference: [41.8264, 40.4636], verdicts: [PASS, PASS]}, SH: {ours: [41.7156, 40.4636], reference: [40.2324, 37.6731], verdicts: [PASS, PASS]}, corr: 0.994151, corr_verdict: PASS}; vsTs={NH: {ours: [21.385, 37.6731], reference: [21.4617, 37.6731], verdicts: [PASS, PASS]}, SH: {ours: [21.5335, 37.6731], reference: [21.4397, 34.8825], verdicts: [PASS, PASS]}, corr: 0.998815, corr_verdict: PASS} |
| C5 | PASS | \|NH - SH\| jet: FAIL only if > 10 % and > 3 sigma_block | latitude={mean: 0.558106, per_block: [2.79055, 0, 2.79053, -5.58108, 2.79053], sigma_block: 3.63842, verdict: PASS}; magnitude={mean: -0.162775, per_block: [0.917242, -1.53634, 0.118982, -0.142702, -0.171056], sigma_block: 0.885011, verdict: PASS} |
| C6 | INFO | paper figures 1-3 (no verdict weight) | jet_consistent=True; jet_lat=43.2542; jet_ms=32.5611; jet_sigma=0.225; surface_easterlies=True; tropical_surface_u_mean=-5.64719; vT_peak_lat=37.6731; vT_peak_sigma=0.875 |
