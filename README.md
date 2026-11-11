# GPU Workload Bring-up, Validation and Observability Lab

A reproducible single-node platform that checks its environment, serves ResNet18 inference on CPU or CUDA,
measures performance, exposes telemetry, and demonstrates fault diagnosis and bounded recovery. It implements
*GPU Workload Lab - Project Specification v1.0* and was evaluated on one NVIDIA A100 on ASU's Sol cluster.

**Status: 7 of 9 acceptance criteria pass, 1 is skipped with a documented reason, 1 is not done.**
Every number in this file comes from `evidence/` (joined in [`evidence/report/REPORT.md`](evidence/report/REPORT.md)). The
spec's numeric limits were treated as targets, not assumed results.

| ID | Criterion (short) | Status | Evidence |
|---|---|---|---|
| AC01 | Deploy from documented commands, pinned, image digest | **SKIP** | Sol has no container runtime. Native deployment with env lock instead (see [Deviations](#deviations-and-honest-caveats)). |
| AC02 | CUDA proven, no silent CPU fallback, CPU profile independent | PASS | PF01-PF06 pass (`gpu_ready=True`); F05 returned 503 in 3/3 trials; CPU preflight + API checks pass |
| AC03 | Correctness within frozen tolerances | PASS | 256 inputs, batch 1/8/32, max abs error 1.5e-5 vs tolerance 1e-4 |
| AC04 | 18 cells x 3 trials baseline + 30-min soak | PASS | 54/54 runs, 18/18 cells, soak complete, nothing missing |
| AC05 | 5 fault scenarios x 3 trials | PASS | 15 validated drills (+1 rejected record) |
| AC06 | Alert <= 45 s for held faults, crash logged <= 5 s | PASS | worst notification 24.3 s (9/9); crash events logged 0.01-0.03 s after the event (3/3) |
| AC07 | Validated recovery <= 60 s, cold start <= 120 s | PASS | in-place crash recovery 5.8-10.3 s; cold start 4.1-4.6 s |
| AC08 | Bounded overload, no memory growth | PASS | queue peaked at its limit (32), 143,789 explicit 429s, RSS slope -2.7 MB/min |
| AC09 | A second person reproduces the demo | **NOT RUN** | needs an independent reviewer (`docs/reviewer_checklist.md`) |

Also not done: the five-minute demo recording and Grafana screenshots (Grafana was not run on Sol; each drill saved the
underlying dashboard data in `panel_data.json` instead).

## What was built

```
 loadgen ──► inference service (1 model worker, queue <= 32, 10 s deadline) ──/metrics──► Prometheus ──► Alertmanager ──► local JSONL receiver
                 ▲  CPU or CUDA, never silent fallback                                        ▲   ▲
 host controller (Docker events / supervisor events, /readyz probes, bounded restarts) ───────┘   └── GPU exporter, node exporter
```

* **Inference service** (`app/`): FastAPI, checksum-verified pinned ResNet18, `/livez` `/readyz` `/infer` `/metrics`,
  bounded labels, JSON logs. `backend=cuda` runs a real CUDA operation at load and otherwise stays not-ready.
* **Preflight** (`preflight/`): PF01-PF06 with PASS/FAIL/SKIP/ERROR, capability discovery (supported / unsupported / unavailable, never zero).
* **Validation** (`validation/`): CPU-vs-GPU logits, batch-vs-single consistency, bad-weights cases, API contract; tolerances are *candidate* until frozen.
* **Benchmarks** (`loadgen/`, `benchmarks/`): closed-loop generator recording every attempt, 18-cell matrix, soak, overload, start-up timing, frozen latency baseline.
* **Recovery and faults** (`ops/`, `faults/`): host controller (3 failed probes at 2 s, then at most 1 restart/60 s and 3/5 min, then explicit `unresolved`),
  config faults routed to an operator instead of restarted, five scripted drills, alert rules tested with promtool.
* **Two deployments:** Docker Compose (`deploy/`) and a Docker-free native mode for Sol (`sol/`). The Sol run is the one with results.
* **Report** (`report/`): joins evidence into the acceptance table above; refuses to mark PASS without evidence.

## Results

Environment: ASU Sol, AMD EPYC 7413 (8 cores allocated per job), Rocky Linux 8.10, **NVIDIA A100-SXM4-80GB**, driver 615.71.09,
torch 2.11.0+cu128, CUDA runtime 12.8, FP32 with TF32 disabled. Jobs ran on shared nodes; different jobs landed on
different physical A100s of the same model.

### Correctness (256 fixed inputs, CPU FP32 reference)

| Check | Result |
|---|---|
| Max abs logit error, batch 1 / 8 / 32 | 1.48e-5 / 1.38e-5 / 1.53e-5 (frozen tolerance: atol 1e-4, rtol 1e-3) |
| Batch(32) vs individual | 8.2e-6 |
| Top-1 agreement | 100%, no flips |
| Bad weights (missing / wrong checksum / corrupt) | each fails with its own reason code |
| Invalid input, oversized batch, bad JSON | 422 |

The inputs are **synthetic** images: this validates numerical equivalence and performance, not classification accuracy.

### Throughput and latency (3 trials per cell, 60 s measured, 100% success in every cell)

| Backend, batch | Images/s at concurrency 1 / 4 / 8 | p95 latency ms at concurrency 1 / 4 / 8 |
|---|---|---|
| CPU, 1 | 76.7 / 75.5 / 74.7 | 13.4 / 64.6 / 120.5 |
| CPU, 8 | 131.7 / 126.8 / 130.0 | 66.9 / 279.0 / 620.6 |
| CPU, 32 | 119.2 / 117.4 / 115.6 | 282.0 / 1433 / 2493 |
| GPU, 1 | 286.5 / 308.0 / 330.1 | 3.5 / 26.9 / 43.3 |
| GPU, 8 | 1461 / 1559 / 1594 | 5.5 / 30.3 / 55.8 |
| GPU, 32 | 2027 / 1793 / 1778 | 16.0 / 80.5 / 221.4 |

GPU/CPU throughput ratio at concurrency 4: **4.1x** (batch 1), **12.3x** (batch 8), **15.3x** (batch 32). Run-to-run variation was small
(coefficient of variation 0.006-0.082). p99 is flagged as weak evidence for the CPU batch 8 and 32 cells (fewer than 1,000 successful
requests per run). With one model worker, extra client concurrency adds queueing, not throughput.

### Soak, overload, start-up

* **Soak (30 min, GPU, batch 8, concurrency 4):** 286,497 requests, 100% success. Throughput fell **13.3%** between the first and last quarter
  (1,378 to 1,195 images/s) and p95 rose 8.5% (34.0 to 36.9 ms). GPU memory flat at 637 MiB, temperature 43-49 C, service RSS flat (1.46 to 1.48 GB). The
  cause of the drift was not investigated.
* **Overload (64 clients, batch 32, 90 s):** pending work never exceeded 32; 143,789 requests rejected with 429 and 230 timed out (all counted);
  RSS slope -2.7 MB/min; the service stayed ready and passed validation afterwards.
* **Start-up:** cold start 4.1-4.6 s, cached restart 3.9-8.2 s (weights on disk, no downloads).

### Fault drills (3 validated trials each, run under the same healthy baseline load)

| Scenario | First observed | Alert received locally | Recovery | Notes |
|---|---|---|---|---|
| F01 crash (SIGKILL) | 0.0-0.2 s | 8.9-12.1 s | 5.8-10.3 s fault-to-validated | requests succeeded only 6-18% of the window while down; **see known issue 1** |
| F02 exporter outage (32 s) | 3.2-3.5 s | 18.8-19.2 s | 1.4-1.8 s after restoring | inference stayed 100% successful; controller restarted nothing |
| F03 missing weights | 3.5-6.7 s | 18.5-24.3 s | 35-43 s after correction | `weights_missing` reason; inference rejected (503); controller marked `needs_operator`, 0 restarts |
| F04 latency regression (injected 100 ms delay) | immediate | 43-51 s | 0.1-0.2 s after removing the delay | queue peaked at 3 of 32; alert needs the 60 s rate window to fill |
| F05 GPU inaccessible | 3.0-3.8 s | 18.0-23.3 s | 29-33 s after correction | `cuda_unavailable`; **no CPU fallback in 3/3**; 0 controller restarts |

F04 is a real but lab-injected delay inside the worker (labelled `injected_latency`); the others are real process, exporter or configuration failures.

## Deviations and honest caveats

**Known issues in this run**

1. **F01: two recovery owners acted in trials 2 and 3.** After the crash, the process supervisor restarted the service within 0.6 s, which is faster
   than the controller's 2 s probe interval, so the controller never saw it go down. While the new process was still loading, three probes failed and the
   controller restarted it a second time about 5 s later (1 controller restart in each of trials 2 and 3, 0 in trial 1). This breaks the design rule
   that only one owner recovers an exited process. Recovery still met the 60 s target, but those two trials took 10.1-10.3 s instead of 5.8 s. The fix is small
   (open a start-up grace window whenever a workload's start time changes), and F01 should then be re-run. Not fixed yet.
2. **Soak throughput drift of -13%** over 30 minutes (above). Unexplained.
3. **F04 trial 1 was rejected** by the harness: a latency alert left over from the previous drill was still active. Cause: starting the fault-case process for F03
   briefly slowed the healthy service enough to exceed twice its baseline p95. The harness now waits for the alert to clear (trial 4 waited 16.1 s) and the first record is kept as `BASELINE_ALERT_ACTIVE`.
4. **Redeploying a service under load is slow:** 29-43 s for F03/F05 against about 4 s on an idle stack. The harness is not inflating this (the service's own log reports 38.4 s
   for the same event). Suspected CPU contention from the baseline load; not tested.

**Deviations from the specification** (also in `docs/deviations.md`)

* **No Docker.** Sol has none, so AC01 is a recorded skip (`artifacts/review/skips.json`) and there is no image digest. Evidence used instead: deploy log,
  environment-lock hash in `preflight.json`, pinned tool versions. The Compose files (`deploy/`), the Dockerfile and the DCGM Exporter path **were written and syntax-checked but never run** against a real Docker daemon. Treat that deployment as untested.
* **GPU telemetry from `nvidia-smi`, not DCGM Exporter,** under DCGM metric names. XID errors are not available. PF06 (`dcgmi diag -r 1`) did run, restricted to the job's own GPU.
* **PF02 compares** the framework's GPU UUID with `nvidia-smi` instead of host-versus-container.
* **F05** hides the device (`CUDA_VISIBLE_DEVICES=""`) instead of removing a container device mapping.
* **Shared nodes.** Other users' load affects CPU numbers and was recorded, not controlled.
* **No git commit identity:** the evidence records `commit: unknown` because the jobs ran from an rsync copy. Two files changed after the final copy (`sol/lab.py`: a
  guard against two jobs sharing one checkout; `sol/lab.sbatch`: an environment self-check). Neither affects the measurements.
* **History is preserved:** earlier preflight records show `gpu_ready=False` because of a bug in my PF06 check (a passing diagnostic was judged by words in stderr,
  and it ran against GPUs the job did not own). It was fixed; the preflight of record (01:56 UTC) passes.
* GPU integration tests (`make test-gpu`) and the container integration tests were not run; the 134 unit/logic tests pass on CPU (10 hardware-dependent tests skip without a GPU or stack).

## How to reproduce

**On ASU Sol (what produced these results)** - full details in [`docs/sol.md`](docs/sol.md)
```bash
./sol/setup_env.sh                                  # mamba env + verified Prometheus/Alertmanager binaries
F="-A <account> -p general -q <qos> -G a100:1"
./sol/submit.sh bringup $F                          # weights, inputs, CPU reference, preflight, tolerance calibration
python -m sol.lab stage freeze-tolerances           # after reading the calibration errors
./sol/submit.sh validate $F ; ./sol/submit.sh benchmark $F ; ./sol/submit.sh soak $F ; ./sol/submit.sh faults $F
./sol/submit.sh cpu -A <account> -p general -q <qos> -G 0
python -m sol.lab stage report                      # artifacts/report/REPORT.md
```
Never run two jobs in one checkout at once; chain them with `sbatch --dependency=afterany:<jobid>`.

**On a GPU host with Docker** (untested, see above): `make init weights fixtures reference up-gpu controller preflight validate`, then `benchmark`, `fault-test`, `report`. See [`docs/setup_gpu_host.md`](docs/setup_gpu_host.md).

**CPU only, any machine:** `make init venv-torch`, fetch weights/inputs/reference, `make test` and `make rules-test`.

## Repository map

| Path | Purpose |
|---|---|
| `app/` | inference service, single model worker, bounded admission, metrics, adapter interface |
| `preflight/`, `validation/` | PF01-PF06; correctness, tolerances, API contract checks |
| `loadgen/`, `benchmarks/` | request generator, matrix, soak, overload, aggregation, baseline freezing |
| `ops/`, `faults/` | controller, alert receiver, GPU exporter, drills F01-F05 and harness |
| `sol/` | Docker-free mode: supervisor, config rendering, tool download, Slurm scripts |
| `deploy/`, `monitoring/` | Compose stack, image pins, alert rules + promtool tests, Alertmanager, Grafana dashboards |
| `report/` | acceptance report and evidence packaging |
| `manifests/` | model, input and preprocessing manifests with SHA256; tested-environment manifest |
| `evidence/` | committed results (small files); `evidence/README.md` explains the layout |
| `docs/` | architecture, runbooks (RB-001..006), Sol guide, deviations, limitations, demo guide, reviewer checklist |

## Evidence

`evidence/` holds byte-identical copies of the small result files. The raw request logs, service samples and logs are in the release archive
`raw-artifacts.tgz` (SHA256 in `evidence/MANIFEST.json`); `evidence/report/evidence_index.json` lists a SHA256 for every raw file so the archive can be verified.
The evidence contains the cluster username and node names in file paths.

## Limits of these results

Single node, single GPU, one model worker; no multi-node or production-scale claim. Faults are controlled application, exporter and configuration faults
(plus one injected delay); no firmware, driver or hardware faults. Inputs are synthetic. Performance numbers apply to this service and protocol on shared Sol nodes.
