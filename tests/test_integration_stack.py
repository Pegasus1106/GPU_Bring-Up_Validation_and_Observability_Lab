"""Container-stack integration tests (need the Compose stack up). Skipped otherwise.

    LAB_TEST_URL=http://127.0.0.1:8001 LAB_TEST_BACKEND=cpu pytest -m integration
"""
import os
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.integration
URL = os.environ.get("LAB_TEST_URL")
BACKEND = os.environ.get("LAB_TEST_BACKEND", "cpu")
FAULT_URL = os.environ.get("LAB_TEST_FAULT_URL", "http://127.0.0.1:8010")

needs_stack = pytest.mark.skipif(not URL, reason="set LAB_TEST_URL to a running service")


@needs_stack
def test_health_endpoints_and_metrics_exposition():
    assert requests.get(f"{URL}/livez", timeout=3).status_code == 200
    assert requests.get(f"{URL}/readyz", timeout=3).status_code == 200
    m = requests.get(f"{URL}/metrics", timeout=3).text
    for name in ("inference_requests_total", "inference_queue_depth", "inference_ready", "inference_request_duration_seconds_bucket"):
        assert name in m


@needs_stack
def test_real_backend_is_reported():
    r = requests.post(f"{URL}/infer", json={"ids": ["img_000"]}, timeout=15)
    assert r.status_code == 200 and r.json()["backend"] == BACKEND


@needs_stack
def test_queue_rejection_under_overload():
    import concurrent.futures as cf

    cap = int(__import__("re").search(r"inference_queue_capacity\{[^}]*\} (\d+)", requests.get(f"{URL}/metrics").text).group(1))

    def go(_):
        try:
            return requests.post(f"{URL}/infer", json={"ids": [f"img_{i:03d}" for i in range(32)]}, timeout=20).status_code
        except requests.RequestException:
            return -1

    with cf.ThreadPoolExecutor(cap * 3) as ex:
        codes = list(ex.map(go, range(cap * 3)))
    assert 429 in codes and set(codes) <= {200, 429, 504}


@pytest.mark.skipif(not os.environ.get("LAB_TEST_FAULT"), reason="start the fault case with an invalid model path and set LAB_TEST_FAULT=1")
def test_wrong_model_path_gives_specific_reason():
    r = requests.get(f"{FAULT_URL}/readyz", timeout=3)
    assert r.status_code == 503 and r.json()["reason_code"] in ("weights_missing", "model_manifest_missing")
    assert requests.post(f"{FAULT_URL}/infer", json={"ids": ["img_000"]}, timeout=3).status_code == 503


@pytest.mark.skipif(not os.environ.get("LAB_TEST_STACK"), reason="needs the monitoring stack and a completed drill")
def test_evidence_persists_outside_the_workload_container():
    root = Path(__file__).resolve().parents[1] / "artifacts"
    assert (root / "controller/faults.jsonl").exists() and (root / "alerts/alerts.jsonl").exists()
