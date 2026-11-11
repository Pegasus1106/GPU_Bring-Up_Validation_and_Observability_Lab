"""Single model-owning worker with bounded admission.

* Exactly one worker thread owns the model (and the GPU context).
* At most `max_pending` admitted requests wait in the queue; the next one is
  rejected immediately (HTTP 429). Nothing is ever enqueued without bound.
* A request that exceeds its deadline is answered with 504 by the HTTP layer.
  If the worker has not started it yet it is skipped; if the GPU work is already
  running it cannot be cancelled, so it completes and its result is discarded
  (counted as a late completion in logs).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.config import Settings
from app.errors import LabError
from app.metrics import Metrics, batch_label
from app.model import InferResult, ModelAdapter

log = logging.getLogger("lab.engine")


class CapacityError(Exception):
    pass


@dataclass
class Job:
    request_id: str
    ids: list[str]
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    enqueued_mono: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)


@dataclass
class JobOutput:
    result: InferResult
    queue_ms: float


def _resolve(fut: asyncio.Future, value=None, exc: BaseException | None = None) -> None:
    if fut.done():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(value)


class Engine:
    def __init__(self, settings: Settings, adapter: ModelAdapter, metrics: Metrics):
        self.s = settings
        self.adapter = adapter
        self.m = metrics
        self._q: queue.Queue[Job] = queue.Queue(maxsize=settings.max_pending)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._state = "loading"  # loading | ready | failed
        self._reason_code: str | None = "loading"
        self._reason: str = "model is loading"
        self._busy_since: float | None = None
        self._thread = threading.Thread(target=self._run, name="model-worker", daemon=True)
        self._process_start = time.monotonic()
        self._stop = threading.Event()
        self.ready_after_s: float | None = None
        self._wire_gauges()

    # -- observability ------------------------------------------------------
    def _wire_gauges(self) -> None:
        m = self.m
        m.b(m.queue_capacity).set(self.s.max_pending)
        m.b(m.queue_depth).set_function(lambda: self._q.qsize())
        m.b(m.in_flight).set_function(lambda: self._in_flight)
        m.b(m.ready).set_function(lambda: 1 if self.readiness()[0] else 0)
        m.b(m.model_loaded).set_function(lambda: 1 if self._state == "ready" else 0)
        m.b(m.start_time).set(time.time())
        m.b(m.worker_busy).set_function(self._busy_seconds)
        m.b(m.fault_delay).set_function(lambda: self._fault_delay_s())

    def _busy_seconds(self) -> float:
        bs = self._busy_since
        return 0.0 if bs is None else time.monotonic() - bs

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            problems = self.s.validate()
            if problems:
                raise LabError("config_invalid", "; ".join(problems))
            self.adapter.load()
            self._smoke()
        except LabError as e:
            self._fail(e.reason_code, e.message)
            return
        except Exception as e:  # unexpected load failure: treat as not ready, keep process alive
            log.exception("model load failed")
            self._fail("unknown", f"{type(e).__name__}: {e}")
            return
        self.ready_after_s = time.monotonic() - self._process_start
        self.m.b(self.m.startup_seconds).set(self.ready_after_s)
        self.m.model_info.labels(**self.m.base, model_version=self.adapter.version).set(1)
        with self._lock:
            self._state, self._reason_code, self._reason = "ready", None, ""
        log.info("ready", extra={"startup_s": round(self.ready_after_s, 3), **self.adapter.info()})
        self._loop()

    def _fail(self, code: str, message: str) -> None:
        with self._lock:
            self._state, self._reason_code, self._reason = "failed", code, message
        log.error("not ready", extra={"reason_code": code, "reason": message})

    def _smoke(self) -> None:
        """Smoke inference required before /readyz returns 200."""
        ids = self.adapter.input_ids[:1]
        if not ids:
            raise LabError("inputs_missing", "no allowlisted inputs available for smoke inference")
        try:
            res = self.adapter.infer(ids)
        except Exception as e:
            raise LabError("smoke_inference_failed", f"smoke inference raised {type(e).__name__}: {e}") from e
        import numpy as np

        if res.logits.shape[0] != 1 or not np.isfinite(res.logits).all():
            raise LabError("smoke_inference_failed", "smoke inference produced invalid logits")
        if res.backend_actual != ("cuda" if self.s.backend == "cuda" else "cpu"):
            raise LabError(
                "smoke_inference_failed",
                f"requested backend {self.s.backend} but executed on {res.backend_actual}",
            )

    # -- readiness ----------------------------------------------------------
    def readiness(self) -> tuple[bool, str | None, str]:
        with self._lock:
            state, code, reason = self._state, self._reason_code, self._reason
        if state != "ready":
            return False, code, reason
        if not self._thread.is_alive():
            return False, "worker_dead", "model worker thread has exited"
        bs = self._busy_since
        if bs is not None and time.monotonic() - bs > self.s.stuck_after_s:
            return False, "worker_stuck", f"worker busy on one job for >{self.s.stuck_after_s:.0f}s"
        return True, None, ""

    @property
    def state(self) -> str:
        return self._state

    # -- admission ----------------------------------------------------------
    def submit(self, request_id: str, ids: list[str], loop: asyncio.AbstractEventLoop) -> Job:
        job = Job(request_id=request_id, ids=ids, loop=loop, future=loop.create_future())
        try:
            self._q.put_nowait(job)
        except queue.Full:
            raise CapacityError() from None
        with self._lock:
            self._in_flight += 1
        return job

    def release(self, job: Job) -> None:
        """Called by the HTTP layer exactly once per admitted job when it has answered."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    # -- worker loop --------------------------------------------------------
    def _fault_delay_s(self) -> float:
        if not (self.s.enable_fault_injection and self.s.fault_dir):
            return 0.0
        try:
            txt = (Path(self.s.fault_dir) / "delay_ms").read_text().strip()
            return max(0.0, min(float(txt), 5000.0)) / 1000.0  # hard cap: bounded fault
        except (OSError, ValueError):
            return 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.cancelled.is_set() or job.future.done():
                continue  # client already got a 504; do not spend GPU time on it
            started = time.monotonic()
            queue_ms = (started - job.enqueued_mono) * 1000.0
            self.m.queue_wait.labels(**self.m.base, batch_size=batch_label(len(job.ids))).observe(queue_ms / 1000.0)
            self._busy_since = started
            try:
                delay = self._fault_delay_s()
                if delay:
                    time.sleep(delay)
                res = self.adapter.infer(job.ids)
                self.m.model_duration.labels(**self.m.base, batch_size=batch_label(len(job.ids))).observe(
                    res.model_ms / 1000.0
                )
                if job.cancelled.is_set():
                    log.warning("late completion discarded", extra={"request_id": job.request_id})
                else:
                    job.loop.call_soon_threadsafe(_resolve, job.future, JobOutput(res, queue_ms), None)
            except Exception as e:
                log.exception("execution failure", extra={"request_id": job.request_id})
                job.loop.call_soon_threadsafe(_resolve, job.future, None, e)
            finally:
                self._busy_since = None
