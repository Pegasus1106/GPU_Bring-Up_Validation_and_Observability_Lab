"""GPU integration tests: real hardware evidence. Skipped (never faked) when no CUDA device is present.

    make test-gpu   ==   pytest -m gpu   (inside the GPU tools container: `docker compose ... run tools-gpu pytest -m gpu`)
"""
import os

import pytest
import requests

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device: GPU acceptance needs real hardware")]


def test_cuda_tensor_op_runs_on_device():
    from app.model import check_cuda_usable

    info = check_cuda_usable(torch)
    assert info["device_name"] and info["total_memory_bytes"] > 0


def test_correctness_vs_cpu_reference_all_batch_sizes(tmp_path):
    from validation.correctness import run

    report, _ = run("cuda", calibrate=False, out_dir=tmp_path)
    failed = [c for c in report["cases"] if c["status"] != "PASS"]
    assert not failed, failed


def test_model_time_comes_from_cuda_events_and_is_below_wall_time():
    import dataclasses
    import time

    from app.config import Settings
    from app.model import ResNet18Adapter

    ad = ResNet18Adapter(dataclasses.replace(Settings.from_env(), backend="cuda"))
    ad.load()
    ids = ad.input_ids[:8]
    ad.infer(ids)  # warm-up
    t0 = time.perf_counter()
    r = ad.infer(ids)
    wall_ms = (time.perf_counter() - t0) * 1000
    assert r.backend_actual == "cuda" and 0 < r.model_ms <= wall_ms * 1.05


def test_tf32_is_disabled_for_fp32_baseline():
    from app.model import configure_torch
    from app.config import Settings

    configure_torch(Settings())
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False


@pytest.mark.skipif(not os.environ.get("LAB_DCGM_URL"), reason="set LAB_DCGM_URL=http://127.0.0.1:9400 with the gpu profile up")
def test_dcgm_exporter_exposes_core_fields_or_reports_them_unsupported():
    txt = requests.get(os.environ["LAB_DCGM_URL"] + "/metrics", timeout=5).text
    assert "DCGM_FI_DEV_GPU_UTIL" in txt and "DCGM_FI_DEV_FB_USED" in txt  # core fields every supported GPU should export
