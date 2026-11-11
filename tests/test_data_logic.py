"""Manifest parsing, tolerance logic, aggregation, overload evaluation, sampler parsing."""
import csv
import json

import numpy as np
import pytest

from app.errors import LabError
from app.inputs import load_manifest, verify_inputs
from benchmarks import aggregate as agg
from benchmarks.overload import evaluate, rss_slope_mb_per_min
from benchmarks.sampler import parse_metrics
from common.util import sha256_file
from validation.reference import compare, top1_disagreement_allowed


# ---------------------------------------------------------------- input manifest
def make_inputs(tmp_path, n=3):
    items = []
    for i in range(n):
        f = tmp_path / f"img_{i:03d}.png"
        f.write_bytes(bytes([i]) * 50)
        items.append({"id": f"img_{i:03d}", "file": f.name, "sha256": sha256_file(f)})
    (tmp_path / "pre.json").write_text(json.dumps({"resize": 8, "center_crop": 8, "mean": [0, 0, 0], "std": [1, 1, 1]}))
    (tmp_path / "inputs.json").write_text(json.dumps({"items": items, "preprocessing_file": "pre.json", "synthetic": True}))
    return tmp_path


def test_manifest_parses_and_verifies(tmp_path):
    d = make_inputs(tmp_path)
    m = load_manifest(str(d / "inputs.json"), str(d))
    assert m.ids == ["img_000", "img_001", "img_002"] and m.synthetic
    verify_inputs(m)


def test_manifest_detects_tampered_and_missing_files(tmp_path):
    d = make_inputs(tmp_path)
    (d / "img_001.png").write_bytes(b"tampered")
    m = load_manifest(str(d / "inputs.json"), str(d))
    with pytest.raises(LabError) as e:
        verify_inputs(m)
    assert e.value.reason_code == "inputs_checksum_mismatch"
    (d / "img_002.png").unlink()
    with pytest.raises(LabError) as e:
        verify_inputs(m)
    assert e.value.reason_code == "inputs_missing"


def test_manifest_rejects_unsafe_ids(tmp_path):
    d = make_inputs(tmp_path)
    j = json.loads((d / "inputs.json").read_text())
    j["items"][0]["id"] = "../etc/passwd"
    (d / "inputs.json").write_text(json.dumps(j))
    with pytest.raises(LabError) as e:
        load_manifest(str(d / "inputs.json"), str(d))
    assert e.value.reason_code == "config_invalid"


def test_missing_manifest_is_specific():
    with pytest.raises(LabError) as e:
        load_manifest("/nonexistent/inputs.json", "/nonexistent")
    assert e.value.reason_code == "inputs_missing"


# ---------------------------------------------------------------- tolerances
def test_compare_uses_atol_plus_rtol_times_reference():
    ref = np.array([10.0, 0.0, -5.0], dtype=np.float32)
    assert compare(ref + np.array([9e-3, 0.9e-4, 0]), ref, 1e-4, 1e-3)["within_tolerance"]  # 9e-3 <= 1e-4+1e-3*10
    assert not compare(ref + np.array([0, 2e-4, 0]), ref, 1e-4, 1e-3)["within_tolerance"]  # 0 reference => atol only
    r = compare(ref + 0.001, ref, 1e-4, 1e-3)
    assert r["max_abs_error"] == pytest.approx(0.001, rel=1e-3) and r["normalized_error"] == pytest.approx(0.001 / 10, rel=1e-3)


def test_top1_flip_only_excused_inside_tolerance_band():
    assert top1_disagreement_allowed(margin=1e-4, top1_logit=5.0, atol=1e-4, rtol=1e-3)
    assert not top1_disagreement_allowed(margin=0.5, top1_logit=5.0, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------- aggregation
def row(outcome="success", bs=8, e2e=10.0, phase="measure", t0="2026-01-01T00:00:00.000Z", t1="2026-01-01T00:00:01.000Z", model="5", queue="1"):
    return {"phase": phase, "outcome": outcome, "batch_size": str(bs), "e2e_ms": str(e2e), "model_ms": model if outcome == "success" else "",
            "queue_ms": queue if outcome == "success" else "", "t_start_utc": t0, "t_end_utc": t1}


def test_percentiles_linear_interpolation():
    assert agg.pct([1, 2, 3, 4], 50) == 2.5
    assert agg.pct([], 95) is None


def test_throughput_uses_actual_completed_batch_sizes_and_failures_stay_in_denominator():
    rows = [row(bs=8)] * 6 + [row(bs=32)] * 2 + [row("rejected", bs=32)] * 2 + [row("timeout", bs=32)] * 2 + [row("connection_error")] * 1 + [row(phase="warmup")] * 5
    a = agg.aggregate_rows(rows, measure_seconds=10.0)
    assert a["attempts"] == 13 and a["successes"] == 8
    assert a["images_per_s"] == pytest.approx((6 * 8 + 2 * 32) / 10.0)  # only completed batches
    assert a["requests_per_s"] == pytest.approx(0.8)
    assert a["success_rate"] == pytest.approx(8 / 13)
    assert a["rejection_rate"] == pytest.approx(2 / 13) and a["timeout_rate"] == pytest.approx(2 / 13) and a["error_rate"] == pytest.approx(1 / 13)
    assert a["p99_weak_evidence"] is True  # < 1000 successes


def test_p99_not_weak_with_1000_successes():
    assert agg.aggregate_rows([row()] * 1000, 10.0)["p99_weak_evidence"] is False


def test_empty_run_does_not_divide_by_zero():
    a = agg.aggregate_rows([], None)
    assert a["attempts"] == 0 and a["images_per_s"] is None and a["success_rate"] is None


def test_variability_and_compare_flag_regressions():
    v = agg.variability([100.0, 110.0, 90.0])
    assert v["mean"] == 100.0 and v["cv"] == pytest.approx(0.1)
    assert agg.compare(100, 120, True, "x")["verdict"] == "improvement"
    assert agg.compare(100, 80, True, "x")["verdict"] == "regression"
    assert agg.compare(100, 120, False, "latency")["verdict"] == "regression"  # higher latency is worse
    assert agg.compare(None, 1, True, "x")["verdict"] == "insufficient_data"


def test_windows_split_by_minute():
    rows = [row(t0="2026-01-01T00:00:10.000Z", t1="2026-01-01T00:00:11.000Z"), row(t0="2026-01-01T00:01:20.000Z", t1="2026-01-01T00:01:21.000Z")]
    w = agg.windows(rows, 60.0)
    assert [x["window"] for x in w] == [0, 1]


def test_read_requests_roundtrip(tmp_path):
    p = tmp_path / "r.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row().keys()))
        w.writeheader()
        w.writerow(row())
    assert agg.aggregate_file(p, 1.0)["successes"] == 1


# ---------------------------------------------------------------- overload evaluation
def samples(depths, rss):
    return [{"t_mono": float(i), "queue_depth": d, "rss_bytes": r} for i, (d, r) in enumerate(zip(depths, rss))]


def test_overload_pass_when_bounded_and_flat_memory():
    rows = samples([0, 10, 32, 32, 20] * 20, [500e6 + (i % 3) * 1e5 for i in range(100)])
    c = evaluate(rows, 32, {"rejected": 100, "timeout": 2})
    assert all(v["pass"] for v in c.values())


def test_overload_fails_when_queue_exceeds_capacity_or_memory_grows():
    rows = samples([0, 40], [500e6, 500e6])
    assert not evaluate(rows, 32, {"rejected": 1})["queue_bounded"]["pass"]
    growing = samples([1] * 100, [500e6 + i * 5e6 for i in range(100)])  # 5 MB/s => 300 MB/min
    assert rss_slope_mb_per_min(growing) > 100
    assert not evaluate(growing, 32, {"rejected": 1})["memory_not_growing"]["pass"]


def test_overload_memory_unavailable_is_not_a_pass():
    rows = samples([1, 2, 3], [None, None, None])
    c = evaluate(rows, 32, {"rejected": 1})["memory_not_growing"]
    assert c["pass"] is False and "unavailable" in c["note"]


def test_no_rejections_or_timeouts_is_not_overload_evidence():
    assert not evaluate(samples([1, 2], [1e6, 1e6]), 32, {"success": 50})["explicit_rejections_or_timeouts"]["pass"]


# ---------------------------------------------------------------- sampler
def test_sampler_missing_fields_are_none_not_zero():
    text = '# TYPE inference_ready gauge\ninference_ready{backend="cpu"} 1.0\ninference_queue_depth{backend="cpu"} 3.0\n'
    v = parse_metrics(text)
    assert v["ready"] == 1.0 and v["queue_depth"] == 3.0
    assert v["rss_bytes"] is None and v["success_total"] is None
