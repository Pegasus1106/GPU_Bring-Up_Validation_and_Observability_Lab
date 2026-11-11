"""Prometheus metrics with bounded label sets.

Allowed labels: workload, backend, precision, status, batch_size (bucketed).
Request IDs and run IDs are *never* labels; they belong in logs and result files.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector

from app.config import LABEL_BATCH_SIZES

STATUSES = ("success", "error", "timeout", "rejected_capacity", "invalid", "not_ready")

# Dense between 50 ms and 1 s: LatencyRegression compares a bucket-interpolated p95 with a frozen
# client-side baseline, so coarse buckets there would give false positives.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0, 15.0)
MODEL_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def batch_label(n: int) -> str:
    return str(n) if n in LABEL_BATCH_SIZES else "other"


class Metrics:
    def __init__(self, workload: str, backend: str, precision: str):
        self.registry = CollectorRegistry()
        self.workload, self.backend, self.precision = workload, backend, precision
        base = ["workload", "backend", "precision"]
        r = self.registry
        try:
            ProcessCollector(registry=r)
        except Exception:
            pass

        self.attempts = Counter(
            "inference_attempts_total", "Requests that reached POST /infer", base + ["batch_size"], registry=r
        )
        self.requests = Counter(
            "inference_requests_total",
            "Request outcomes (status: success|error|timeout|rejected_capacity|invalid|not_ready)",
            base + ["status", "batch_size"],
            registry=r,
        )
        self.duration = Histogram(
            "inference_request_duration_seconds",
            "Server-side request duration from arrival to response, including queueing",
            base + ["status", "batch_size"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.model_duration = Histogram(
            "inference_model_duration_seconds",
            "Model-only forward time (CUDA events on cuda; perf_counter on cpu)",
            base + ["batch_size"],
            buckets=MODEL_BUCKETS,
            registry=r,
        )
        self.queue_wait = Histogram(
            "inference_queue_wait_seconds",
            "Time an admitted request waited before the worker started it",
            base + ["batch_size"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.images = Counter(
            "inference_images_total", "Images in successfully completed requests", base, registry=r
        )
        self.queue_depth = Gauge("inference_queue_depth", "Admitted requests waiting for the worker", base, registry=r)
        self.queue_capacity = Gauge("inference_queue_capacity", "Maximum admitted pending requests", base, registry=r)
        self.in_flight = Gauge(
            "inference_in_flight_requests", "Admitted requests not yet answered", base, registry=r
        )
        self.ready = Gauge("inference_ready", "1 when /readyz would return 200", base, registry=r)
        self.model_loaded = Gauge("inference_model_loaded", "1 when weights are loaded on the backend", base, registry=r)
        self.model_info = Gauge(
            "inference_model_info", "Loaded model identity (value is always 1)", base + ["model_version"], registry=r
        )
        self.startup_seconds = Gauge(
            "inference_startup_seconds", "Seconds from process start to first ready", base, registry=r
        )
        self.start_time = Gauge(
            "inference_process_start_time_seconds", "Unix time the service process started", base, registry=r
        )
        self.fault_delay = Gauge(
            "inference_fault_injection_delay_seconds",
            "Lab-only injected per-job delay (0 when no fault is active)",
            base,
            registry=r,
        )
        self.worker_busy = Gauge(
            "inference_worker_busy_seconds", "Seconds the worker has been on its current job (0 when idle)", base, registry=r
        )

    @property
    def base(self) -> dict:
        return {"workload": self.workload, "backend": self.backend, "precision": self.precision}

    def b(self, gauge):
        return gauge.labels(**self.base)

    def record(self, status: str, n: int, seconds: float) -> None:
        bl = batch_label(n)
        self.requests.labels(**self.base, status=status, batch_size=bl).inc()
        self.duration.labels(**self.base, status=status, batch_size=bl).observe(seconds)

    def init_series(self) -> None:
        """Create zero-valued series so rate()/increase() work from the first scrape."""
        for bl in [*map(str, LABEL_BATCH_SIZES), "other"]:
            self.attempts.labels(**self.base, batch_size=bl)
            for st in STATUSES:
                self.requests.labels(**self.base, status=st, batch_size=bl)
                self.duration.labels(**self.base, status=st, batch_size=bl)
            self.model_duration.labels(**self.base, batch_size=bl)
            self.queue_wait.labels(**self.base, batch_size=bl)
        self.b(self.images)
