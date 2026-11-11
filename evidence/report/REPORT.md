# GPU Workload Lab - results report

Generated 2026-11-11T05:25:38.878Z from raw artifacts at commit `unknown`. Every number below is read from files in `artifacts/`; nothing here is a target presented as a result.

## Acceptance status

| ID | Status | Criterion | Details |
|---|---|---|---|
| AC01 | **SKIP** | Deploy from documented commands with pinned dependencies and cached weights | approved skip: ASU Sol has no container runtime. Native deployment instead: deploy log (artifacts/deploy/native-up-*.log), pip env lock sha256 in preflight.json |
| AC02 | **PASS** | GPU profile proves CUDA execution and fails when GPU access is withheld; CPU profile works independently | {'gpu_preflight_ready': True, 'f05_no_silent_fallback': True, 'cpu_profile_independent': True} |
| AC03 | **PASS** | Fixed-fixture correctness passes frozen tolerances; invalid inputs fail clearly | {'checks_passed': True, 'tolerances_frozen': True, 'api_contract_passed': True} |
| AC04 | **PASS** | 18-cell, 54-trial baseline and a 30-minute soak; unsupported cells and failures flagged | {'runs': '54/54', 'cells': '18/18', 'soak_complete_30min': True, 'flagged_missing': 0} |
| AC05 | **PASS** | 5 fault scenarios x 3 trials with complete timelines and cause-specific recovery outcomes | validated+clean trials per scenario: {'F01': 3, 'F02': 3, 'F03': 3, 'F04': 3, 'F05': 3}; recorded: {'F01': 3, 'F02': 3, 'F03': 3, 'F04': 4, 'F05': 3} |
| AC06 | **PASS** | Held service/exporter faults alert locally within 45 s; transient crashes logged within 5 s | alert<=45s: 9/9, crash<=5s: 3/3, worst notification 24.259s |
| AC07 | **PASS** | Validated recovery within 60 s (cached restart); cold healthy start within 120 s | {'f01_recovery_within_60s': True, 'cold_start_within_120s': True} |
| AC08 | **PASS** | Overload keeps pending work bounded with explicit rejection/timeout counts and no unbounded memory growth | {'queue_bounded': True, 'explicit_rejections_or_timeouts': True, 'memory_not_growing': True, 'recovered_and_valid': True} |
| AC09 | **NOT_RUN** | A second person reproduces the demo and diagnoses a saved incident from the evidence | needs artifacts/review/reviewer_signoff.json (see docs/reviewer_checklist.md) |

## Environment

- Mode `gpu`, preflight run `preflight-gpu-20261111T015621Z`, executed in `native-process`
- OS `Rocky Linux 8.10 (Green Obsidian)`, CPU `AMD EPYC 7413 24-Core Processor` (48 logical), RAM 503.5 GiB
- GPU NVIDIA A100-SXM4-80GB (GPU-c5bef930-c1a4-4894-0d30-0c3a6dd28f11, driver 615.71.09)
- torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, CUDA runtime `12.8`, cuDNN `91900`
- Image `native-env:dfaa2e2267ba4585934bafc7`
- Time sync: `{'ntp_synchronized': 'yes'}`

## Baseline benchmark

Experiment `baseline-v1`: 54/54 runs, 18/18 cells. Percentiles: linear interpolation between closest ranks (numpy.percentile, method='linear'). p99 is weak evidence when a run has fewer than 1000 successful requests.

| Cell | img/s (mean ± sd) | p50 / p95 / p99 ms (mean) | success | timeout | reject | attempts | note |
|---|---|---|---|---|---|---|---|
| cpu-b1-c1 | 76.7 ± 1.8 (cv 0.024) | 11.9 / 13.4 / 19.9 | 1.000 | 0.000 | 0.000 | 13800 |  |
| cpu-b1-c4 | 75.5 ± 6.2 (cv 0.082) | 47.6 / 64.6 / 171.3 | 1.000 | 0.000 | 0.000 | 13604 |  |
| cpu-b1-c8 | 74.7 ± 1.8 (cv 0.024) | 96.8 / 120.5 / 457.5 | 1.000 | 0.000 | 0.000 | 13464 |  |
| cpu-b32-c1 | 119.2 ± 3.6 (cv 0.030) | 255.9 / 282.0 / 595.3 | 1.000 | 0.000 | 0.000 | 672 | p99 weak |
| cpu-b32-c4 | 117.4 ± 2.5 (cv 0.021) | 1034.6 / 1432.8 / 1719.1 | 1.000 | 0.000 | 0.000 | 671 | p99 weak |
| cpu-b32-c8 | 115.6 ± 2.7 (cv 0.023) | 2069.6 / 2493.4 / 3228.4 | 1.000 | 0.000 | 0.000 | 672 | p99 weak |
| cpu-b8-c1 | 131.7 ± 1.6 (cv 0.012) | 58.2 / 66.9 / 82.0 | 1.000 | 0.000 | 0.000 | 2964 | p99 weak |
| cpu-b8-c4 | 126.8 ± 7.0 (cv 0.055) | 233.0 / 279.0 / 860.7 | 1.000 | 0.000 | 0.000 | 2876 | p99 weak |
| cpu-b8-c8 | 130.0 ± 1.1 (cv 0.008) | 466.3 / 620.6 / 1020.0 | 1.000 | 0.000 | 0.000 | 2947 | p99 weak |
| gpu-b1-c1 | 286.5 ± 1.6 (cv 0.006) | 3.4 / 3.5 / 8.0 | 1.000 | 0.000 | 0.000 | 51573 |  |
| gpu-b1-c4 | 308.0 ± 5.7 (cv 0.018) | 11.9 / 26.9 / 44.9 | 1.000 | 0.000 | 0.000 | 55450 |  |
| gpu-b1-c8 | 330.1 ± 4.7 (cv 0.014) | 22.9 / 43.3 / 78.1 | 1.000 | 0.000 | 0.000 | 59425 |  |
| gpu-b32-c1 | 2027.0 ± 45.0 (cv 0.022) | 15.3 / 16.0 / 20.8 | 1.000 | 0.000 | 0.000 | 11403 |  |
| gpu-b32-c4 | 1793.2 ± 41.8 (cv 0.023) | 62.2 / 80.5 / 363.2 | 1.000 | 0.000 | 0.000 | 10095 |  |
| gpu-b32-c8 | 1778.1 ± 47.1 (cv 0.027) | 126.7 / 221.4 / 611.0 | 1.000 | 0.000 | 0.000 | 10021 |  |
| gpu-b8-c1 | 1461.4 ± 18.6 (cv 0.013) | 5.2 / 5.5 / 7.8 | 1.000 | 0.000 | 0.000 | 32882 |  |
| gpu-b8-c4 | 1559.0 ± 39.0 (cv 0.025) | 19.7 / 30.3 / 41.3 | 1.000 | 0.000 | 0.000 | 35080 |  |
| gpu-b8-c8 | 1594.0 ± 71.2 (cv 0.045) | 37.7 / 55.8 / 139.5 | 1.000 | 0.000 | 0.000 | 35883 |  |

**GPU vs CPU (same batch / concurrency):**

| batch | conc | img/s change | p95 change |
|---|---|---|---|
| 1 | 1 | 273.8% (improvement) | -74.2% (improvement) |
| 1 | 4 | 307.8% (improvement) | -58.4% (improvement) |
| 1 | 8 | 341.9% (improvement) | -64.1% (improvement) |
| 8 | 1 | 1009.8% (improvement) | -91.7% (improvement) |
| 8 | 4 | 1129.0% (improvement) | -89.1% (improvement) |
| 8 | 8 | 1126.3% (improvement) | -91.0% (improvement) |
| 32 | 1 | 1601.2% (improvement) | -94.3% (improvement) |
| 32 | 4 | 1427.1% (improvement) | -94.4% (improvement) |
| 32 | 8 | 1437.7% (improvement) | -91.1% (improvement) |

## Soak

Run `soak-cuda-20261111T020643Z`: complete=True, attempts=286497, success rate=1.0000, drift={'status': 'ok', 'images_per_s': {'first_quarter': 1377.9428571428573, 'last_quarter': 1195.1619047619047, 'change_pct': -13.264770119709175}, 'p95_ms': {'first_quarter': 34.03756428571428, 'last_quarter': 36.93577142857142, 'change_pct': 8.514731308413655}}

## Fault drills

| Scenario | Class | Trials | Final status | Detect s (median) | Notify s (median) | Recover s (median) | Correction→recovery s (median) |
|---|---|---|---|---|---|---|---|
| F01 workload crash | actual | 3 | {'RECOVERED_VALIDATED': 3} | 0.0 | 9.0 | 10.1 | n/a |
| F02 exporter outage | actual | 3 | {'RECOVERED_VALIDATED': 3} | 3.5 | 19.1 | 33.6 | 1.4 |
| F03 missing weights | actual | 3 | {'RECOVERED_VALIDATED': 3} | 3.8 | 22.9 | 58.2 | 39.4 |
| F04 latency regression | injected_latency | 4 | {'BASELINE_ALERT_ACTIVE': 1, 'RECOVERED_VALIDATED': 3} | 0.0 | 47.0 | 47.2 | 0.1 |
| F05 GPU inaccessible | actual | 3 | {'RECOVERED_VALIDATED': 3} | 3.5 | 23.0 | 52.3 | 30.2 |

## Overload

- queue_bounded: PASS {'max_observed': 32.0, 'capacity': 32}
- explicit_rejections_or_timeouts: PASS {'rejected': 143789, 'timeout': 230}
- memory_not_growing: PASS {'slope_mb_per_min': -2.678953239253846, 'limit': 10.0, 'note': ''}
- recovered_and_valid: PASS {'ready': True, 'quick_validate': {'passed': True, 'reason': '', 'n': 16, 'backend': 'cuda', 'model_version': 'resnet18-IMAGENET1K_V1-f37072fd47e8', 'seconds': 0.022382571827620268}}

## Resume claim inputs (filled only from evidence)

- [N] workload configurations evaluated: 18
- [N] controlled fault scenarios (record count): 16
- [observed detection time] (median, s): 3.166
- [observed validated recovery time] (median, s): 10.125
- [measured performance change]: see GPU-vs-CPU comparison table; quote with batch/concurrency and run-to-run variability

## Limits of these results

- Single node, single GPU; no multi-node or production-scale claim.
- Faults are controlled application/configuration faults (F04 is a lab-only injected delay); no firmware, driver or physical hardware faults.
- Synthetic input images validate numerical equivalence and performance, not classification accuracy, unless `manifests/inputs.json` says `synthetic: false`.
- Targets are proposed acceptance gates; see the table above for what the evidence actually shows.
