"""API contract tests against the real FastAPI app with a SIMULATED model adapter."""
import dataclasses
import re
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import LabError
from app.main import create_app
from tests.fakes import FakeAdapter, IDS


def make(adapter=None, **kw):
    s = dataclasses.replace(Settings(), backend=kw.pop("backend", "cpu"), **kw)
    return create_app(s, adapter or FakeAdapter(backend=s.backend)), s


def wait_ready(client, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get("/readyz")
        if r.status_code == 200:
            return r
        time.sleep(0.05)
    raise AssertionError(f"not ready: {client.get('/readyz').json()}")


def test_livez_is_cheap_and_ok_even_when_not_ready():
    app, _ = make(FakeAdapter(fail_load=LabError("weights_missing", "nope")))
    with TestClient(app) as c:
        assert c.get("/livez").status_code == 200


def test_readyz_503_with_specific_reason_when_load_fails():
    app, _ = make(FakeAdapter(fail_load=LabError("weights_missing", "no weights at /x")))
    with TestClient(app) as c:
        time.sleep(0.3)
        r = c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["reason_code"] == "weights_missing"
        assert "no weights" in r.json()["reason"]
        # inference is rejected, with the same reason
        r = c.post("/infer", json={"ids": ["img_000"]})
        assert r.status_code == 503 and r.json()["error"]["reason_code"] == "weights_missing"


def test_ready_and_infer_success_reports_real_backend_and_request_id():
    app, _ = make()
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/infer", json={"ids": ["img_001", "img_002"]}, headers={"X-Request-ID": "abc-123"})
        b = r.json()
        assert r.status_code == 200
        assert b["backend"] == "cpu" and b["request_id"] == "abc-123" and b["model_version"] == "fake-1"
        assert [p["id"] for p in b["predictions"]] == ["img_001", "img_002"]
        assert r.headers["x-request-id"] == "abc-123"


def test_backend_reported_is_what_actually_executed():
    # requested cuda but the (simulated) executor says cpu => smoke inference must refuse readiness
    app, _ = make(FakeAdapter(backend="cuda", report_backend="cpu"), backend="cuda")
    with TestClient(app) as c:
        time.sleep(0.3)
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["reason_code"] == "smoke_inference_failed"


@pytest.mark.parametrize("body,fragment", [
    ({"ids": []}, "non-empty"),
    ({"ids": ["nope"]}, "unknown input"),
    ({"ids": ["../../etc/passwd"]}, "unknown input"),
    ({"ids": [f"img_{i:03d}" for i in range(33)]}, "exceeds"),
    ({"ids": [1, 2]}, "strings"),
    ({"wrong": 1}, "non-empty"),
])
def test_invalid_input_is_422(body, fragment):
    app, _ = make()
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/infer", json=body)
        assert r.status_code == 422
        assert fragment in r.json()["error"]["message"]


def test_malformed_json_is_422():
    app, _ = make()
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/infer", content="{bad", headers={"content-type": "application/json"})
        assert r.status_code == 422


def test_max_batch_32_accepted():
    app, _ = make()
    with TestClient(app) as c:
        wait_ready(c)
        assert c.post("/infer", json={"ids": IDS[:32]}).status_code == 200


def test_capacity_rejection_429_and_queue_is_bounded():
    ad = FakeAdapter()
    app, s = make(ad, max_pending=3, request_deadline_s=30)
    with TestClient(app) as c:
        wait_ready(c)
        ad.gate = __import__("threading").Event()  # block the worker
        import concurrent.futures as cf

        def one(_):
            return c.post("/infer", json={"ids": ["img_000"]}).status_code

        with cf.ThreadPoolExecutor(12) as ex:
            futs = [ex.submit(one, i) for i in range(12)]
            time.sleep(1.0)
            depth = app.state.engine._q.qsize()
            assert depth <= 3  # never exceeds admission bound
            ad.gate.set()
            codes = [f.result() for f in futs]
        assert 429 in codes and 200 in codes
        assert set(codes) <= {200, 429}
        text = c.get("/metrics").text
        assert re.search(r'status="rejected_capacity"[^}]*\} [1-9]', text) or re.search(r'inference_requests_total\{[^}]*rejected_capacity[^}]*\} [1-9]', text)


def test_deadline_gives_504_and_is_visible_in_metrics():
    ad = FakeAdapter()
    app, _ = make(ad, request_deadline_s=0.3)
    with TestClient(app) as c:
        wait_ready(c)
        ad.delay_s = 1.0
        r = c.post("/infer", json={"ids": ["img_000"]})
        assert r.status_code == 504 and r.json()["error"]["code"] == "deadline_exceeded"
        assert re.search(r'inference_requests_total\{[^}]*status="timeout"[^}]*\} 1\.0', c.get("/metrics").text)


def test_execution_failure_is_structured_500():
    ad = FakeAdapter(fail_infer=True)
    app, _ = make(ad)
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/infer", json={"ids": ["img_000"]})
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "execution_failure"


def test_metric_labels_are_bounded_and_have_no_ids():
    app, _ = make()
    with TestClient(app) as c:
        wait_ready(c)
        for n in (1, 2, 5, 8, 17, 32):
            c.post("/infer", json={"ids": IDS[:n]}, headers={"X-Request-ID": f"rid-{n}"})
        text = c.get("/metrics").text
        assert "rid-" not in text and "request_id" not in text
        sizes = set(re.findall(r'batch_size="([^"]+)"', text))
        assert sizes <= {"1", "8", "32", "other"}


def test_unready_service_not_ready_gauge_is_zero():
    app, _ = make(FakeAdapter(fail_load=LabError("cuda_unavailable", "no gpu")), backend="cuda")
    with TestClient(app) as c:
        time.sleep(0.3)
        assert re.search(r"inference_ready\{[^}]*\} 0\.0", c.get("/metrics").text)


def test_cuda_backend_without_cuda_fails_explicitly_no_cpu_fallback(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this test needs a host without it")
    import json

    from common.util import REPO_ROOT

    mm = REPO_ROOT / "manifests/model.json"
    wp = REPO_ROOT / "manifests/weights/resnet18-f37072fd.pth"
    if not (mm.exists() and wp.exists()):
        pytest.skip("weights not fetched (make weights)")
    from app.model import ResNet18Adapter

    s = dataclasses.replace(Settings.from_env(), backend="cuda")
    app = create_app(s, ResNet18Adapter(s))
    with TestClient(app) as c:
        time.sleep(1.5)
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["reason_code"] == "cuda_unavailable"
        assert c.post("/infer", json={"ids": ["img_000"]}).status_code == 503
