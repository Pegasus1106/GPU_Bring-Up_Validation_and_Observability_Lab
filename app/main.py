"""FastAPI inference service.

GET  /livez   process responsive (no GPU work)
GET  /readyz  200 only after config, weights, requested backend and a smoke inference
POST /infer   1..32 allowlisted input IDs
GET  /metrics Prometheus exposition
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import MAX_BATCH, Settings
from app.engine import CapacityError, Engine, JobOutput
from app.logging_setup import setup_logging
from app.metrics import Metrics, batch_label
from app.model import ModelAdapter, ResNet18Adapter

log = logging.getLogger("lab.api")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,64}$")


def _err(status: int, code: str, message: str, request_id: str, **extra) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id, **extra}}
    return JSONResponse(body, status_code=status, headers={"X-Request-ID": request_id})


def create_app(settings: Settings | None = None, adapter: ModelAdapter | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    setup_logging(s.log_file)
    metrics = Metrics(s.workload, s.backend, s.precision)
    metrics.init_series()
    engine = Engine(s, adapter or ResNet18Adapter(s), metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="GPU Workload Lab inference service", lifespan=lifespan)
    app.state.engine = engine
    app.state.metrics = metrics
    app.state.settings = s

    @app.get("/livez")
    async def livez():
        return {"status": "alive"}

    @app.get("/readyz")
    async def readyz():
        ready, code, reason = engine.readiness()
        if ready:
            return {"ready": True, "state": engine.state, "startup_s": engine.ready_after_s, **engine.adapter.info()}
        return JSONResponse(
            {"ready": False, "state": engine.state, "reason_code": code, "reason": reason, "backend": s.backend},
            status_code=503,
        )

    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/infer")
    async def infer(request: Request):
        t0 = time.monotonic()
        hdr = request.headers.get("x-request-id", "")
        request_id = hdr if REQUEST_ID_RE.match(hdr) else uuid.uuid4().hex
        m = metrics

        # 1) parse + validate (422). Invalid requests still count as attempts.
        ids, problem, raw = None, None, None
        try:
            body = await request.json()
            raw = body.get("ids") if isinstance(body, dict) else None
            if not isinstance(raw, list) or not raw:
                problem = "'ids' must be a non-empty list"
            elif len(raw) > MAX_BATCH:
                problem = f"batch of {len(raw)} exceeds the maximum of {MAX_BATCH}"
            elif not all(isinstance(i, str) for i in raw):
                problem = "'ids' must contain strings"
            else:
                allowed = set(engine.adapter.input_ids)
                unknown = [i for i in raw if i not in allowed] if allowed else []
                if unknown:
                    problem = f"unknown input id(s): {unknown[:5]}"
                else:
                    ids = raw
        except Exception:
            problem = "request body is not valid JSON"
        n = len(ids) if ids else (len(raw) if isinstance(raw, list) else 0)
        m.attempts.labels(**m.base, batch_size=batch_label(n)).inc()
        if problem:
            m.record("invalid", n, time.monotonic() - t0)
            log.info("invalid", extra={"request_id": request_id, "status": "invalid", "detail": problem})
            return _err(422, "invalid_input", problem, request_id)

        # 2) readiness gate (503) - a not-ready service must not run inference
        ready, code, reason = engine.readiness()
        if not ready:
            m.record("not_ready", n, time.monotonic() - t0)
            log.info("not_ready", extra={"request_id": request_id, "status": "not_ready", "reason_code": code})
            return _err(503, "not_ready", reason or "service not ready", request_id, reason_code=code)

        # 3) bounded admission (429)
        loop = asyncio.get_running_loop()
        try:
            job = engine.submit(request_id, ids, loop)
        except CapacityError:
            m.record("rejected_capacity", n, time.monotonic() - t0)
            log.info("rejected", extra={"request_id": request_id, "status": "rejected_capacity", "batch": n})
            return _err(429, "capacity", f"{s.max_pending} requests already pending", request_id)

        # 4) wait for the worker, bounded by the deadline (504)
        try:
            out: JobOutput = await asyncio.wait_for(job.future, timeout=s.request_deadline_s)
        except asyncio.TimeoutError:
            job.cancelled.set()
            engine.release(job)
            m.record("timeout", n, time.monotonic() - t0)
            log.warning("timeout", extra={"request_id": request_id, "status": "timeout", "batch": n})
            return _err(504, "deadline_exceeded", f"deadline of {s.request_deadline_s}s exceeded", request_id)
        except Exception as e:
            engine.release(job)
            m.record("error", n, time.monotonic() - t0)
            return _err(500, "execution_failure", f"{type(e).__name__}: {e}", request_id)
        engine.release(job)

        res = out.result
        preds = engine.adapter.postprocess(res.logits, top_k=5)
        elapsed = time.monotonic() - t0
        m.record("success", n, elapsed)
        m.b(m.images).inc(n)
        log.info(
            "ok",
            extra={
                "request_id": request_id,
                "status": "success",
                "backend": res.backend_actual,
                "batch": n,
                "duration_ms": round(elapsed * 1000, 3),
                "model_ms": round(res.model_ms, 3),
                "queue_ms": round(out.queue_ms, 3),
            },
        )
        return JSONResponse(
            {
                "request_id": request_id,
                "backend": res.backend_actual,
                "model_version": engine.adapter.version,
                "batch_size": n,
                "model_duration_ms": res.model_ms,
                "queue_ms": out.queue_ms,
                "predictions": [{"id": i, "top5": p} for i, p in zip(ids, preds)],
            },
            headers={"X-Request-ID": request_id},
        )

    return app


def main() -> None:  # pragma: no cover
    import os

    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("LAB_HOST", "127.0.0.1"),
        port=int(os.environ.get("LAB_PORT", "8000")),
        workers=1,  # one process => one model-owning worker
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
