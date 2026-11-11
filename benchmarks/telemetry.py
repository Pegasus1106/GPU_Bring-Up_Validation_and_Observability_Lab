"""Pull resource telemetry for a run window from Prometheus.

Rules from the spec: a metric that does not exist is reported as *unavailable*
(with the reason), never as zero. A reachable Prometheus with no samples for a
field means the field is unsupported or its exporter was down.
"""
from __future__ import annotations

import statistics

import requests

# name -> (promql, unit, source)
QUERIES = {
    "host_cpu_busy_percent": ('100 - avg(rate(node_cpu_seconds_total{mode="idle"}[15s])) * 100', "%", "node-exporter"),
    "host_mem_available_bytes": ("node_memory_MemAvailable_bytes", "bytes", "node-exporter"),
    "service_rss_bytes": ('process_resident_memory_bytes{job="inference"}', "bytes", "inference service"),
    "gpu_utilization_percent": ("avg(DCGM_FI_DEV_GPU_UTIL)", "%", "dcgm-exporter"),
    "gpu_memory_used_mib": ("sum(DCGM_FI_DEV_FB_USED)", "MiB", "dcgm-exporter"),
    "gpu_power_watts": ("sum(DCGM_FI_DEV_POWER_USAGE)", "W", "dcgm-exporter"),
    "gpu_temperature_celsius": ("max(DCGM_FI_DEV_GPU_TEMP)", "C", "dcgm-exporter"),
}


def collect(prom_url: str, start_ts: float, end_ts: float, step_s: int = 5, timeout: float = 5.0) -> dict:
    out: dict = {"prometheus": prom_url, "window": {"start": start_ts, "end": end_ts, "step_s": step_s}, "fields": {}}
    for name, (q, unit, src) in QUERIES.items():
        entry = {"unit": unit, "source": src, "query": q}
        try:
            r = requests.get(
                f"{prom_url}/api/v1/query_range",
                params={"query": q, "start": start_ts, "end": end_ts, "step": step_s},
                timeout=timeout,
            )
            r.raise_for_status()
            result = r.json()["data"]["result"]
            vals = [float(v) for s in result for _, v in s["values"] if v not in ("NaN", "+Inf", "-Inf")]
            if not vals:
                entry.update(available=False, reason="no samples in window (field unsupported, exporter down, or not scraped)")
            else:
                entry.update(available=True, samples=len(vals), mean=statistics.fmean(vals), max=max(vals), min=min(vals))
        except Exception as e:  # noqa: BLE001
            entry.update(available=False, reason=f"prometheus query failed: {type(e).__name__}: {e}")
        out["fields"][name] = entry
    return out


def instant(prom_url: str, query: str, timeout: float = 5.0) -> float | None:
    """Single instant value or None if absent (never 0 for 'missing')."""
    try:
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=timeout)
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else None
    except Exception:  # noqa: BLE001
        return None
