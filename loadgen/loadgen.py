"""Closed-loop load generator that records *every* attempt to requests.csv.

Each of `concurrency` client threads sends the next request as soon as the
previous one finishes. Every attempt is written, whatever its outcome:
success | rejected (429) | timeout (504) | invalid (422) | not_ready (503) |
error (other 5xx / bad body) | client_timeout | connection_error.

Durations use time.perf_counter (monotonic); timestamps are UTC for correlation.
Warm-up attempts are recorded with phase=warmup and excluded from aggregates.
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from common.util import REPO_ROOT, iso, new_run_id, utc_now

FIELDS = [
    "run_id", "request_id", "phase", "concurrency", "t_start_utc", "t_end_utc",
    "backend_requested", "backend", "batch_size", "outcome", "http_status",
    "e2e_ms", "model_ms", "queue_ms", "error_category",
]
N_INPUTS = 256


@dataclass
class LoadConfig:
    url: str
    backend: str  # expected backend ("cpu" | "cuda"); a mismatch is recorded as an error
    batch_size: int = 8
    concurrency: int = 4
    warmup_requests: int = 20
    duration_s: Optional[float] = 60.0  # None => run until stop()
    timeout_s: float = 12.0  # client timeout, slightly above the 10 s server deadline
    run_id: str = ""
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "artifacts/requests")
    out_name: str = "requests.csv"
    id_pool: int = N_INPUTS

    def __post_init__(self):
        if not self.run_id:
            self.run_id = new_run_id("load")
        self.out_dir = Path(self.out_dir)


def classify(status: int | None, exc: str | None, body_backend: str | None, expect: str) -> tuple[str, str]:
    """-> (outcome, error_category)"""
    if exc == "timeout":
        return "client_timeout", "client_timeout"
    if exc:
        return "connection_error", exc
    if status == 200:
        if body_backend != expect:
            return "error", f"backend_mismatch:{body_backend}"
        return "success", ""
    return {
        429: ("rejected", "capacity"),
        504: ("timeout", "deadline_exceeded"),
        422: ("invalid", "invalid_input"),
        503: ("not_ready", "not_ready"),
    }.get(status, ("error", f"http_{status}"))


class LoadRunner:
    def __init__(self, cfg: LoadConfig):
        self.cfg = cfg
        self.path = cfg.out_dir / cfg.out_name
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self._io_lock = threading.Lock()
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.measure_start_mono: float | None = None
        self.measure_start_utc: str | None = None
        self.measure_end_mono: float | None = None
        self.counts: dict[str, int] = {}
        self.t_first_failure_utc: str | None = None

    # -- control ------------------------------------------------------------
    def start(self) -> "LoadRunner":
        for k in range(self.cfg.concurrency):
            t = threading.Thread(target=self._client, args=(k,), daemon=True, name=f"client-{k}")
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()

    def wait(self) -> None:
        for t in self._threads:
            t.join()
        with self._io_lock:
            self._f.flush()
            self._f.close()

    def run(self) -> dict:
        self.start()
        self.wait()
        return self.summary()

    def summary(self) -> dict:
        return {
            "run_id": self.cfg.run_id,
            "requests_csv": str(self.path),
            "measure_start_utc": self.measure_start_utc,
            "measure_seconds": (self.measure_end_mono - self.measure_start_mono) if self.measure_start_mono and self.measure_end_mono else None,
            "counts": dict(self.counts),
        }

    # -- internals ------------------------------------------------------------
    def _next(self) -> tuple[int, str]:
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            phase = "warmup" if seq < self.cfg.warmup_requests else "measure"
            if phase == "measure" and self.measure_start_mono is None:
                self.measure_start_mono = time.perf_counter()
                self.measure_start_utc = iso()
        return seq, phase

    def _client(self, k: int) -> None:
        cfg = self.cfg
        sess = requests.Session()
        while not self._stop.is_set():
            if cfg.duration_s is not None and self.measure_start_mono is not None:
                if time.perf_counter() - self.measure_start_mono >= cfg.duration_s:
                    break
            seq, phase = self._next()
            ids = [f"img_{(seq * cfg.batch_size + j) % cfg.id_pool:03d}" for j in range(cfg.batch_size)]
            rid = f"{cfg.run_id[-30:]}-{seq:07d}".replace(":", "")
            t_utc = iso(utc_now())
            t0 = time.perf_counter()
            status = exc = body_backend = None
            model_ms = queue_ms = None
            try:
                r = sess.post(f"{cfg.url}/infer", json={"ids": ids}, headers={"X-Request-ID": rid}, timeout=cfg.timeout_s)
                status = r.status_code
                if status == 200:
                    b = r.json()
                    body_backend, model_ms, queue_ms = b.get("backend"), b.get("model_duration_ms"), b.get("queue_ms")
            except requests.Timeout:
                exc = "timeout"
            except requests.RequestException as e:
                exc = type(e).__name__
            except ValueError:
                status = 200
                exc = None
                body_backend = "unparseable"
            e2e_ms = (time.perf_counter() - t0) * 1000.0
            t_end = iso(utc_now())
            outcome, cat = classify(status, exc, body_backend, cfg.backend)
            row = {
                "run_id": cfg.run_id, "request_id": rid, "phase": phase, "concurrency": cfg.concurrency,
                "t_start_utc": t_utc, "t_end_utc": t_end, "backend_requested": cfg.backend,
                "backend": body_backend or "", "batch_size": cfg.batch_size, "outcome": outcome,
                "http_status": status if status is not None else "", "e2e_ms": f"{e2e_ms:.3f}",
                "model_ms": "" if model_ms is None else f"{model_ms:.3f}",
                "queue_ms": "" if queue_ms is None else f"{queue_ms:.3f}", "error_category": cat,
            }
            with self._io_lock:
                self._w.writerow(row)
                if phase == "measure":
                    self.counts[outcome] = self.counts.get(outcome, 0) + 1
                    self.measure_end_mono = time.perf_counter()
                if outcome != "success" and self.t_first_failure_utc is None:
                    self.t_first_failure_utc = t_utc
                if seq % 50 == 0:
                    self._f.flush()
            if outcome in ("connection_error", "not_ready"):
                time.sleep(0.2)  # avoid a tight error loop while the service is down
            elif outcome == "rejected":
                time.sleep(0.005)  # 429s are nearly free; keep pressure on without a CPU-bound spin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/requests")
    a = ap.parse_args(argv)
    cfg = LoadConfig(a.url, a.backend, a.batch_size, a.concurrency, a.warmup, a.duration, run_id=a.run_id, out_dir=a.out_dir)
    s = LoadRunner(cfg).run()
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
