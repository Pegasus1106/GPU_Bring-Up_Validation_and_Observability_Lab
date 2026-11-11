"""Background sampler of a service's /metrics (queue depth, RSS, outcome counters).

Used by the soak and overload tests to produce the queue and memory traces.
Values missing from the exposition are written as empty cells (unavailable),
not zero.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families

from common.util import iso

FIELDS = ["t_utc", "t_mono", "ready", "queue_depth", "in_flight", "rss_bytes", "worker_busy_s",
          "success_total", "timeout_total", "rejected_total", "error_total"]


def parse_metrics(text: str) -> dict:
    vals: dict[str, float | None] = {k: None for k in FIELDS[2:]}
    sums = {"success": 0.0, "timeout": 0.0, "rejected_capacity": 0.0, "error": 0.0}
    seen_status = False
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            n = s.name
            if n == "inference_ready":
                vals["ready"] = s.value
            elif n == "inference_queue_depth":
                vals["queue_depth"] = s.value
            elif n == "inference_in_flight_requests":
                vals["in_flight"] = s.value
            elif n == "process_resident_memory_bytes":
                vals["rss_bytes"] = s.value
            elif n == "inference_worker_busy_seconds":
                vals["worker_busy_s"] = s.value
            elif n == "inference_requests_total":
                seen_status = True
                st = s.labels.get("status")
                if st in sums:
                    sums[st] += s.value
    if seen_status:
        vals["success_total"], vals["timeout_total"] = sums["success"], sums["timeout"]
        vals["rejected_total"], vals["error_total"] = sums["rejected_capacity"], sums["error"]
    return vals


class MetricsSampler:
    def __init__(self, url: str, path: Path, interval_s: float = 1.0):
        self.url, self.path, self.interval = url, Path(path), interval_s
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="metrics-sampler")
        self.rows: list[dict] = []

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t.start()
        return self

    def stop(self) -> list[dict]:
        self._stop.set()
        self._t.join(timeout=10)
        return self.rows

    def _run(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            while not self._stop.is_set():
                row = {"t_utc": iso(), "t_mono": time.monotonic()}
                try:
                    row.update(parse_metrics(requests.get(f"{self.url}/metrics", timeout=3).text))
                except Exception:  # noqa: BLE001 - a sample gap is itself evidence
                    row.update({k: None for k in FIELDS[2:]})
                self.rows.append(row)
                w.writerow({k: ("" if v is None else v) for k, v in row.items()})
                f.flush()
                self._stop.wait(self.interval)
