# RB-005 Latency regression

**Alert:** `LatencyRegression` (p95 > 2× frozen baseline, ≥100 completions/60 s) · **Drill:** F04

1. Is the injected-delay metric non-zero? Panel “Injected fault delay” / `inference_fault_injection_delay_seconds` → lab fault; remove `artifacts/faults/inject/<profile>/delay_ms`.
2. Queue depth vs capacity (panel “Queue depth”). Rising queue with flat model time ⇒ load or contention; rising model time ⇒ the model/GPU is slower.
3. GPU panels: utilization, clocks (`DCGM_FI_DEV_SM_CLOCK`), temperature/power (if supported), other processes: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
4. Host panels: CPU busy, memory available, disk I/O. A CPU-profile neighbour or the load generator on the same host can contend.
5. Check the load profile matches the baseline (batch size, concurrency 4). The baseline only applies to the same backend + batch size.
6. Restart only if the service is also unhealthy (the controller decides this); a slow-but-ready service is never restarted automatically.
7. Overload is bounded: pending ≤ 32, then `429`; deadline `504`. Confirm `status="rejected_capacity"` / `timeout` counters instead of unbounded queue growth.
