"""Generate the three provisioned Grafana dashboards as checked-in JSON.

    python monitoring/grafana/build_dashboards.py

Every panel description states: unit, source, sampling interval and its
missing-data limitation (spec requirement). Panels use `noValue` text so a
missing series reads "unavailable", never 0.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}
SCRAPE = "5 s scrape / 5 s rule interval"

_next_id = [0]


def _id() -> int:
    _next_id[0] += 1
    return _next_id[0]


def describe(text: str, unit: str, source: str, missing: str, interval: str = SCRAPE) -> str:
    return f"{text}\n\nUnit: {unit}\nSource: {source}\nInterval: {interval}\nMissing data: {missing}"


def target(expr: str, legend: str = "", ref: str = "A", instant: bool = False) -> dict:
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "interval": "5s"}
    if instant:
        t.update(instant=True, range=False, format="table")
    return t


def panel(kind: str, title: str, targets: list[dict], x: int, y: int, w: int, h: int, *, unit: str = "short",
          unit_name: str, source: str, missing: str, text: str = "", mappings=None, no_value: str = "unavailable",
          thresholds=None, extra_options=None, min_=None, max_=None) -> dict:
    defaults = {"unit": unit, "noValue": no_value, "custom": {}}
    if mappings:
        defaults["mappings"] = mappings
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
    if min_ is not None:
        defaults["min"] = min_
    if max_ is not None:
        defaults["max"] = max_
    return {
        "id": _id(), "type": kind, "title": title, "datasource": DS,
        "description": describe(text or title, unit_name, source, missing),
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": extra_options or {},
    }


def ts(title, exprs, x, y, w=12, h=8, **kw):
    ts_targets = [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(exprs)]
    return panel("timeseries", title, ts_targets, x, y, w, h, **kw)


def stat(title, expr, x, y, w=6, h=4, legend="{{instance}}", **kw):
    return panel("stat", title, [target(expr, legend)], x, y, w, h,
                 extra_options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "colorMode": "background"}, **kw)


def textpanel(title, md, x, y, w, h):
    return {"id": _id(), "type": "text", "title": title, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {"mode": "markdown", "content": md},
            "description": describe(title, "n/a", "static text", "n/a (does not query data)", "static")}


READY_MAP = [{"type": "value", "options": {"1": {"text": "READY", "color": "green"}, "0": {"text": "NOT READY", "color": "red"}}}]
UP_MAP = [{"type": "value", "options": {"1": {"text": "UP", "color": "green"}, "0": {"text": "DOWN", "color": "red"}}}]
GPU_MISSING = ("If the field is unsupported on this GPU or dcgm-exporter is down the panel is empty ('unavailable'). "
               "It is never plotted as 0. See artifacts/*/capabilities.json and the TelemetryUnavailable alert.")


def dashboard(uid: str, title: str, panels: list[dict], annotations=None) -> dict:
    return {
        "uid": uid, "title": title, "tags": ["gpu-workload-lab"], "schemaVersion": 39, "version": 1,
        "editable": True, "refresh": "5s", "time": {"from": "now-30m", "to": "now"},
        "timezone": "utc", "panels": panels,
        "annotations": {"list": annotations or []},
        "templating": {"list": []},
    }


def readiness_dashboard() -> dict:
    _next_id[0] = 0
    p = [
        stat("Inference readiness", 'inference_ready', 0, 0, unit_name="boolean (1 = ready)", source="inference service /metrics (inference_ready)",
             missing="No data means the service is not being scraped: shown as 'UNKNOWN', not as ready.", mappings=READY_MAP, no_value="UNKNOWN",
             text="Whether /readyz would return 200: config, weights, requested backend and smoke inference passed."),
        stat("Scrape targets", 'up', 6, 0, legend="{{job}}", unit_name="boolean (1 = scrape ok)", source="Prometheus up",
             missing="A target that is not configured for the active profile does not appear.", mappings=UP_MAP, no_value="UNKNOWN",
             text="Monitoring health: can Prometheus reach each exporter."),
        stat("Cold-start time to ready", 'inference_startup_seconds', 12, 0, unit="s", unit_name="seconds", source="inference service (inference_startup_seconds)",
             missing="Empty until the service has become ready once.", no_value="not ready yet", text="Process start to first ready (AC07 cold target 120 s)."),
        stat("Workload restarts (persistent)", 'lab_workload_restarts_total', 18, 0, legend="{{workload}}", unit_name="count", source="host controller (lab_workload_restarts_total)",
             missing="Empty if the controller is not running. Counter survives workload and controller restarts.", no_value="controller not reporting"),
        panel("table", "Loaded model and backend", [target('inference_model_info', "", "A", instant=True)], 0, 4, 12, 6, unit_name="n/a", source="inference service (inference_model_info labels)",
              missing="Empty until a model is loaded.", text="Loaded model version, backend and precision.",
              extra_options={"showHeader": True}),
        panel("table", "GPU identity (from DCGM labels)", [target('DCGM_FI_DEV_GPU_UTIL', "", "A", instant=True)], 12, 4, 12, 6, unit_name="n/a",
              source="dcgm-exporter (labels modelName, UUID, gpu)", missing=GPU_MISSING, text="GPU model and UUID as reported by DCGM (compare with preflight.json inventory)."),
        stat("Host disk free (/)", 'node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}', 0, 10, unit="bytes", unit_name="bytes",
             source="node-exporter", missing="Empty if node-exporter is down (TelemetryUnavailable)."),
        stat("Controller heartbeat age", 'time() - lab_controller_heartbeat_timestamp_seconds', 6, 10, unit="s", unit_name="seconds",
             source="host controller", missing="Empty if the controller is not scraped.", no_value="controller not reporting",
             thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 15}], legend="controller"),
        stat("Recovery state", 'lab_controller_state == 1', 12, 10, legend="{{workload}}: {{state}}", unit_name="state label", source="host controller",
             missing="Only workloads whose container exists are listed.", no_value="controller not reporting"),
        textpanel("Inventory and capability evidence",
                  "Static inventory is recorded by `make preflight` in `artifacts/preflight/<run>/preflight.json` (hardware, driver, toolkit, image digest, "
                  "model/input hashes) and `capabilities.json` (supported / unsupported / unavailable GPU telemetry fields). "
                  "This dashboard shows the live view; the files are the evidence.", 18, 10, 6, 4),
    ]
    return dashboard("lab-readiness", "Lab 1 - System readiness and inventory", p)


def performance_dashboard() -> dict:
    _next_id[0] = 0
    app_missing = "Gaps mean the inference service was not scraped; no traffic yields empty latency panels (insufficient data, not 0 ms)."
    p = [
        ts("Throughput (images/s)", [('sum by (backend) (rate(inference_images_total[30s]))', "{{backend}}")], 0, 0, unit="ops", unit_name="images per second",
           source="inference service (inference_images_total)", missing=app_missing, text="Completed images, counted from actual completed batch sizes."),
        ts("Requests/s by outcome", [('sum by (backend, status) (rate(inference_requests_total[30s]))', "{{backend}} {{status}}")], 12, 0, unit="reqps", unit_name="requests per second",
           source="inference service (inference_requests_total)", missing=app_missing, text="success, error, timeout, rejected_capacity (429), invalid (422), not_ready (503)."),
        ts("Server-side latency p50 / p95 / p99", [
            ('histogram_quantile(0.50, sum by (le, backend) (rate(inference_request_duration_seconds_bucket{status="success"}[1m])))', "p50 {{backend}}"),
            ('histogram_quantile(0.95, sum by (le, backend) (rate(inference_request_duration_seconds_bucket{status="success"}[1m])))', "p95 {{backend}}"),
            ('histogram_quantile(0.99, sum by (le, backend) (rate(inference_request_duration_seconds_bucket{status="success"}[1m])))', "p99 {{backend}}")],
           0, 8, unit="s", unit_name="seconds", source="inference service histogram (includes queueing)",
           missing=app_missing + " p99 is weak evidence below ~1000 samples. Client-side end-to-end numbers live in requests.csv.",
           text="Bucket-interpolated quantiles over a 1 min window."),
        ts("Model-only time p95 by batch size", [('histogram_quantile(0.95, sum by (le, backend, batch_size) (rate(inference_model_duration_seconds_bucket[1m])))', "{{backend}} b={{batch_size}}")],
           12, 8, unit="s", unit_name="seconds", source="inference service (CUDA events on cuda, perf_counter on cpu)", missing=app_missing),
        ts("Queue depth, in-flight, capacity", [('inference_queue_depth', "queue {{backend}}"), ('inference_in_flight_requests', "in-flight {{backend}}"), ('inference_queue_capacity', "capacity {{backend}}")],
           0, 16, unit="short", unit_name="requests", source="inference service gauges", missing=app_missing, text="Pending work is bounded by the capacity line."),
        ts("Error / timeout / rejection share", [
            ('sum by (backend) (increase(inference_requests_total{status=~"error|timeout"}[1m])) / sum by (backend) (increase(inference_requests_total{status=~"success|error|timeout"}[1m]))', "exec error+timeout {{backend}}"),
            ('sum by (backend) (increase(inference_requests_total{status="rejected_capacity"}[1m])) / sum by (backend) (increase(inference_requests_total[1m]))', "capacity rejected {{backend}}")],
           12, 16, unit="percentunit", unit_name="fraction of requests (1 min)", source="inference service counters",
           missing="Empty when there was no traffic in the window (division undefined). The WorkloadErrorRate alert requires >=20 requests."),
        ts("GPU utilization", [('DCGM_FI_DEV_GPU_UTIL', "GPU {{gpu}}")], 0, 24, unit="percent", unit_name="percent (0-100)", source="dcgm-exporter DCGM_FI_DEV_GPU_UTIL", missing=GPU_MISSING, max_=100, min_=0),
        ts("GPU memory (VRAM)", [('DCGM_FI_DEV_FB_USED', "used GPU {{gpu}}"), ('DCGM_FI_DEV_FB_FREE', "free GPU {{gpu}}")], 12, 24, unit="mebibytes", unit_name="MiB", source="dcgm-exporter DCGM_FI_DEV_FB_USED / FB_FREE", missing=GPU_MISSING),
        ts("GPU temperature and power", [('DCGM_FI_DEV_GPU_TEMP', "temp C GPU {{gpu}}"), ('DCGM_FI_DEV_POWER_USAGE', "power W GPU {{gpu}}")], 0, 32, unit="short", unit_name="degrees C and watts (per series legend)",
           source="dcgm-exporter", missing=GPU_MISSING + " Many GPUs do not expose power or temperature through DCGM; absence is shown, not hidden."),
        ts("GPU errors (XID, ECC)", [('DCGM_FI_DEV_XID_ERRORS', "XID GPU {{gpu}}"), ('DCGM_FI_DEV_ECC_DBE_VOL_TOTAL', "ECC DBE GPU {{gpu}}")], 12, 32, unit="short", unit_name="error code / count",
           source="dcgm-exporter", missing=GPU_MISSING + " ECC fields are unsupported on many consumer GPUs; an empty panel means unsupported, not 'no errors'."),
        ts("Host CPU busy", [('100 - avg(rate(node_cpu_seconds_total{mode="idle"}[30s])) * 100', "cpu busy")], 0, 40, unit="percent", unit_name="percent", source="node-exporter", missing="Empty if node-exporter is down.", min_=0, max_=100),
        ts("Host memory available and service RSS", [('node_memory_MemAvailable_bytes', "available RAM"), ('process_resident_memory_bytes{job="inference"}', "service RSS {{instance}}")], 12, 40, unit="bytes", unit_name="bytes",
           source="node-exporter; inference service process collector (Linux only)", missing="Service RSS is empty on non-Linux hosts."),
        ts("Host disk I/O and free space", [('rate(node_disk_io_time_seconds_total[30s])', "io busy {{device}}"), ('node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}', "free bytes /")], 0, 48, unit="short", unit_name="fraction busy / bytes",
           source="node-exporter", missing="Empty if node-exporter is down."),
        ts("Host network throughput", [('sum(rate(node_network_receive_bytes_total{device!="lo"}[30s]))', "rx"), ('sum(rate(node_network_transmit_bytes_total{device!="lo"}[30s]))', "tx")], 12, 48, unit="Bps", unit_name="bytes per second",
           source="node-exporter", missing="Loopback traffic (load generator) is excluded, so local benchmarks read near 0."),
    ]
    return dashboard("lab-performance", "Lab 2 - Workload performance and resources", p)


def incidents_dashboard() -> dict:
    _next_id[0] = 0
    p = [
        panel("state-timeline", "Alerts (pending / firing)", [target('ALERTS{alertname!="Watchdog"}', "{{alertname}} {{alertstate}} {{instance}}")], 0, 0, 24, 7, unit_name="state", source="Prometheus ALERTS series",
              missing="Empty means no alert is pending or firing. Notification receipt times are in artifacts/alerts/alerts.jsonl.", text="Alert lifecycle."),
        ts("Workload restarts (persistent counter) and container state", [('lab_workload_restarts_total', "restarts {{workload}}"), ('lab_workload_container_running', "running {{workload}}")], 0, 7, unit="short", unit_name="count / boolean",
           source="host controller", missing="Empty if the controller is down; ControllerHeartbeatStale fires after 15 s.", text="Restarts remain visible even if shorter than any alert hold period."),
        panel("state-timeline", "Recovery state", [target('lab_controller_state == 1', "{{workload}}: {{state}}")], 12, 7, 12, 8, unit_name="state", source="host controller",
              missing="Containers that do not exist are not listed.", text="healthy, starting, suspect, recovering, container_down (Docker owns it), needs_operator, unresolved."),
        ts("Consecutive failed readiness probes", [('lab_controller_probe_failures_consecutive', "{{workload}}")], 0, 15, unit="short", unit_name="probes (2 s apart)", source="host controller",
           missing="Empty if the controller is down. Restart threshold is 3 consecutive failures."),
        ts("Controller restart actions", [('lab_controller_restart_actions_total', "{{workload}} {{result}}")], 12, 15, unit="short", unit_name="count", source="host controller",
           missing="Only restarts performed by the controller; Docker-policy restarts appear in the restart counter above.", text="Bounded: 1 per 60 s, 3 per 5 min."),
        ts("Readiness and scrape health", [('inference_ready', "ready {{instance}}"), ('up', "up {{job}}")], 0, 23, unit="short", unit_name="boolean", source="inference service; Prometheus up",
           missing="A series that stops reporting is a gap (unknown), distinguishing 'service down' from 'monitoring down' via up."),
        ts("Scrape freshness", [('time() - timestamp(up)', "age {{job}}")], 12, 23, unit="s", unit_name="seconds since last sample", source="Prometheus", missing="Large values mean stale telemetry."),
        ts("Injected fault delay (lab-only)", [('inference_fault_injection_delay_seconds', "{{instance}}")], 0, 31, unit="s", unit_name="seconds", source="inference service",
           missing="Empty if the service is down. Non-zero means a lab-injected latency fault is active (a real delay, clearly labelled).", text="Distinguishes injected from organic latency."),
        ts("Latency vs frozen baseline", [
            ('histogram_quantile(0.95, sum by (le, backend, batch_size) (rate(inference_request_duration_seconds_bucket{status="success"}[60s])))', "p95 {{backend}} b={{batch_size}}"),
            ('2 * lab_baseline_p95_seconds', "2x baseline {{backend}} b={{batch_size}}")],
           12, 31, unit="s", unit_name="seconds", source="inference histogram; frozen baseline rules", missing="No baseline line until `make freeze-baseline` has been run; then LatencyRegression can fire."),
        textpanel("How to read an incident", "Incident records are in `artifacts/faults/incidents.jsonl`, each with injected time, first observation, alert firing, "
                  "notification receipt, corrective action, first successful readiness and validation completion. Controller events: `artifacts/controller/faults.jsonl`.", 0, 39, 24, 3),
    ]
    ann = [
        {"name": "Restarts", "datasource": DS, "enable": True, "iconColor": "red",
         "expr": 'increase(lab_workload_restarts_total[10s]) > 0', "titleFormat": "restart", "textFormat": "{{workload}}", "step": "5s"},
        {"name": "Alerts firing", "datasource": DS, "enable": True, "iconColor": "orange",
         "expr": 'ALERTS{alertstate="firing"} == 1', "titleFormat": "{{alertname}}", "textFormat": "{{instance}}", "step": "5s"},
    ]
    return dashboard("lab-incidents", "Lab 3 - Incidents and recovery", p, ann)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in (("1-readiness.json", readiness_dashboard()), ("2-performance.json", performance_dashboard()), ("3-incidents.json", incidents_dashboard())):
        (OUT / name).write_text(json.dumps(d, indent=2) + "\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
