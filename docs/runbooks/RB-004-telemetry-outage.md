# RB-004 Telemetry unavailable

**Alert:** `TelemetryUnavailable` (also `ControllerTargetDown`, `ControllerHeartbeatStale`) · **Drill:** F02

GPU health is **unknown**, not idle. Inference may be fine; do not restart it.

1. Which exporter: the alert label `job` (`dcgm-exporter`, `node-exporter`, `lab-controller`).
2. `docker ps -a --filter name=lab-` and `docker logs lab-dcgm-exporter`; `curl -s 127.0.0.1:9400/metrics | head`.
3. DCGM version pair unsupported for the driver/GPU → fix `DCGM_EXPORTER_IMAGE` (`<exporter>-<DCGM>-<os>`).
4. `docker start lab-dcgm-exporter` (or `make up-gpu`). Confirm `up{job="dcgm-exporter"} == 1`; GPU panels repopulate (gaps stay as gaps).
5. Controller: `make controller`; heartbeat age panel must be < 15 s.
6. Check `artifacts/preflight/*/capabilities.json` – a field listed *unsupported* is expected to be absent permanently and is not an outage.
