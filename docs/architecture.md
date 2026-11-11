# Architecture

## Failure domains

| Domain | Components | Survives a workload crash? |
|---|---|---|
| Request path | `inference-gpu` / `inference-cpu` container, load generator | – (this is what fails) |
| Monitoring path | Prometheus, Alertmanager, Grafana, node-exporter, dcgm-exporter, alert-receiver | Yes: own containers, named volumes, no `depends_on` on the workload |
| Recovery | `ops/controller.py` on the host | Yes: state in `artifacts/controller/state.json` |
| Evidence | `./artifacts` on the host | Yes: bind-mounted, never inside the workload container |

## Request lifecycle (`POST /infer`)

1. Parse + validate (`422`): 1–32 IDs, each in the manifest allowlist (no client path reaches the filesystem).
2. Readiness gate (`503 not_ready`, with `reason_code`).
3. Bounded admission (`429` when 32 requests are already pending).
4. Single model-owning worker thread runs the batch. Inputs are preprocessed once at load (identical for CPU/CUDA),
   so latency measures inference. CUDA model time uses CUDA events with an explicit `synchronize`; CPU uses `perf_counter`.
5. Deadline `10 s` (`504`). A job not yet started is skipped; a job already running on the GPU cannot be cancelled,
   completes, is discarded and logged as a late completion.
6. Response: predictions, **backend derived from the output tensor's device**, model version, request ID, model/queue ms.

`/readyz` is `200` only after config validation → checksum-verified weights → input checksum → requested backend
(real CUDA op) → weights loaded → smoke inference on the requested backend. It also turns `503` if the worker has been on one job
longer than `LAB_STUCK_AFTER_S` (`worker_stuck`).

## Metrics (bounded labels)

Labels: `workload`, `backend`, `precision`, `status`, `batch_size ∈ {1, 8, 32, other}`. Request IDs and run IDs appear only in logs and
result files. Series are pre-created at zero so `rate()` / `increase()` work from the first scrape.

## Recovery controller

```
observe(probe) ─ container not running ─────────────► CONTAINER_DOWN   (Docker restart policy owns it)
              ─ ready ──────────────────────────────► HEALTHY
              ─ not ready, config reason code ──────► NEEDS_OPERATOR  (once; runbook; never restarted)
              ─ not ready, still in startup grace ──► STARTING / RECOVERING
              ─ not ready ×3 consecutive (2 s) ─────► restart if ≥60 s since last and <3 in 300 s
                                                      else cooldown / UNRESOLVED (explicit, runbook RB-006)
```
Docker events (`die`, `start`, …) are written to `faults.jsonl`. A restart is a `start` after a `die`;
the persistent counter feeds `lab_workload_restarts_total`, so `RestartObserved` catches restarts shorter than any alert hold time.

## Alerts

| Alert | Condition | For |
|---|---|---|
| ServiceUnavailable | `up{job="inference"}==0 or inference_ready==0` | 10 s |
| TelemetryUnavailable | exporter `up==0` (GPU health is *unknown*, not idle) | 10 s |
| RestartObserved | `increase(lab_workload_restarts_total[1m])>0` | 0 |
| WorkloadErrorRate | (error+timeout)/(success+error+timeout) > 5 % over 60 s, ≥20 requests; 429/422 excluded | 10 s |
| LatencyRegression | p95 > 2 × frozen matching baseline (backend, batch), ≥100 completions in 60 s; absent baseline or low traffic ⇒ no alert | 15 s |

Extras: `ControllerHeartbeatStale`, `ControllerTargetDown`, `RecoveryUnresolved`, `OperatorActionRequired`.
Alertmanager `group_wait: 5s` routes to a loopback webhook that appends JSONL. Nothing leaves the host.
Rule behaviour (including absent series and low traffic) is covered by `monitoring/tests/alerts_test.yml` (`make rules-test`).

## Timing conventions

Durations inside one process use monotonic clocks. Cross-process incident times are UTC wall clock on one host.
The load generator runs on the same host (loopback); record its placement and NTP state (`preflight.json`).
