# RB-001 Workload crashed, restarted, or unresponsive

**Alerts:** `ServiceUnavailable`, `RestartObserved`, `WorkloadErrorRate` · **Drill:** F01

## Triage (2 minutes)
1. `curl -s 127.0.0.1:8000/readyz` – note `reason_code` (`loading`, `worker_stuck`, `smoke_inference_failed`, …).
2. `docker inspect lab-inference-gpu --format '{{json .State}}'` – `Running`, `ExitCode`, `OOMKilled`, `RestartCount`.
3. `tail artifacts/controller/faults.jsonl` – `docker_event die` (exit code, signal), `state_change`, `restart_action`.
4. `docker logs --tail 100 lab-inference-gpu` or `artifacts/logs/inference-gpu.jsonl` – last `execution failure` / `late completion`.

## Decide
| Observation | Meaning | Action |
|---|---|---|
| container exited, Docker restarting (`on-failure:3`) | Docker owns recovery | Wait for `/readyz` 200; confirm `RestartObserved` and a validated revalidation (`python -m validation.api_checks`) |
| restarts exhausted (container `Exited`) | Docker gave up | Find the cause in logs first; then `docker start lab-inference-gpu` |
| container running, `/readyz` 503 `worker_stuck` or no response | controller owns recovery: restarts after 3 failed probes (≤1/60 s, ≤3/5 min) | Let it act; if state becomes `unresolved` go to RB-006 |
| `reason_code` is a config code (`weights_*`, `cuda_*`) | restarts will not help | RB-002 / RB-003 |
| `OOMKilled: true` | host/GPU memory | reduce concurrency / batch; check `process_resident_memory_bytes` and VRAM panels |

## Verify recovery
`/readyz` 200, quick validation passes on fixed inputs (top-1 matches the CPU reference on the expected backend), error rate back < 5 %, incident noted in `artifacts/faults/incidents.jsonl` if it was a drill.
