"""Load generator records every attempt (SIMULATED model behind a real HTTP server); report evidence rules."""
import dataclasses
import json
import socket
import threading
import time

import pytest
import uvicorn

from app.config import Settings
from app.main import create_app
from benchmarks import aggregate as agg
from common.util import append_jsonl, write_json
from loadgen.loadgen import LoadConfig, LoadRunner, classify
from report.generate_report import evaluate, incident_stats
from tests.fakes import FakeAdapter


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    started = []

    def start(adapter, **kw):
        port = free_port()
        s = dataclasses.replace(Settings(), backend="cpu", **kw)
        srv = uvicorn.Server(uvicorn.Config(create_app(s, adapter), host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=srv.run, daemon=True)
        t.start()
        for _ in range(100):
            if srv.started:
                break
            time.sleep(0.05)
        started.append(srv)
        return f"http://127.0.0.1:{port}"

    yield start
    for srv in started:
        srv.should_exit = True


def test_classify_outcomes():
    assert classify(200, None, "cpu", "cpu") == ("success", "")
    assert classify(200, None, "cpu", "cuda")[0] == "error"  # backend mismatch is an error, not a success
    assert classify(429, None, None, "cpu")[0] == "rejected"
    assert classify(504, None, None, "cpu")[0] == "timeout"
    assert classify(None, "timeout", None, "cpu")[0] == "client_timeout"
    assert classify(None, "ConnectionError", None, "cpu")[0] == "connection_error"
    assert classify(503, None, None, "cpu")[0] == "not_ready"


def test_every_attempt_is_recorded_including_rejections_and_timeouts(server, tmp_path):
    ad = FakeAdapter(delay_s=0.15)
    url = server(ad, max_pending=2, request_deadline_s=0.4)
    time.sleep(0.5)
    cfg = LoadConfig(url, "cpu", batch_size=1, concurrency=8, warmup_requests=3, duration_s=3, run_id="t", out_dir=tmp_path)
    runner = LoadRunner(cfg)
    s = runner.run()
    rows = agg.read_requests(tmp_path / "requests.csv")
    outcomes = {r["outcome"] for r in rows}
    assert "success" in outcomes and "rejected" in outcomes  # capacity bound is visible to the client
    assert all(r["run_id"] == "t" and r["request_id"] for r in rows)
    assert len({r["request_id"] for r in rows}) == len(rows)  # unique ids
    assert sum(1 for r in rows if r["phase"] == "warmup") == 3
    assert sum(s["counts"].values()) == sum(1 for r in rows if r["phase"] == "measure")  # nothing dropped
    a = agg.aggregate_rows(rows, s["measure_seconds"])
    assert a["attempts"] == len([r for r in rows if r["phase"] == "measure"])
    assert a["rejection_rate"] > 0


def test_loadgen_records_connection_errors_when_service_is_down(tmp_path):
    cfg = LoadConfig(f"http://127.0.0.1:{free_port()}", "cpu", batch_size=1, concurrency=1, warmup_requests=0, duration_s=1, run_id="d", out_dir=tmp_path)
    LoadRunner(cfg).run()
    rows = agg.read_requests(tmp_path / "requests.csv")
    assert rows and {r["outcome"] for r in rows} == {"connection_error"}


# ---------------------------------------------------------------- report: no evidence => NOT_RUN (never PASS)
def test_report_without_evidence_never_claims_pass(tmp_path):
    acc = evaluate(tmp_path, None)
    assert set(acc) == {f"AC0{i}" for i in range(1, 10)}
    assert {v["status"] for v in acc.values()} == {"NOT_RUN"}


def inc(sid, name, status="RECOVERED_VALIDATED", **kw):
    base = {"scenario": sid, "scenario_name": name, "trial": 1, "final_status": status, "fault_class": "actual", "cleanup": {"ok": True},
            "metrics": {"fault_to_first_observation_s": 1.0, "fault_to_notification_s": 20.0, "fault_to_recovery_s": 30.0}, "restart_count": {}}
    base.update(kw)
    return base


def test_ac05_needs_three_validated_trials_of_every_scenario(tmp_path):
    f = tmp_path / "faults/incidents.jsonl"
    for sid in ("F01", "F02", "F03", "F04"):
        for _ in range(3):
            append_jsonl(f, inc(sid, sid))
    for _ in range(2):
        append_jsonl(f, inc("F05", "F05"))
    assert evaluate(tmp_path, None)["AC05"]["status"] == "FAIL"
    append_jsonl(f, inc("F05", "F05"))
    assert evaluate(tmp_path, None)["AC05"]["status"] == "PASS"
    append_jsonl(f, inc("F05", "F05", status="FAILURE_PHASE_ONLY"))  # extra failure-only record doesn't break/inflate


def test_ac06_fails_when_alert_receipt_exceeds_45s(tmp_path):
    f = tmp_path / "faults/incidents.jsonl"
    append_jsonl(f, inc("F02", "x", ac06_alert_within_45s=True, metrics={"fault_to_notification_s": 30.0}))
    append_jsonl(f, inc("F02", "x", ac06_alert_within_45s=False, metrics={"fault_to_notification_s": 51.0}))
    append_jsonl(f, inc("F01", "c", ac06_crash_logged_within_5s=True))
    assert evaluate(tmp_path, None)["AC06"]["status"] == "FAIL"


def test_ac03_not_passed_while_tolerances_are_candidate(tmp_path):
    write_json(tmp_path / "validation/validation-cuda-20260101T000000Z.json", {"checks_passed": True, "tolerances_frozen": False, "passed": False})
    write_json(tmp_path / "validation/api-cuda-20260101T000000Z.json", {"passed": True})
    r = evaluate(tmp_path, None)["AC03"]
    assert r["status"] == "FAIL" and "candidate" in r["details"]


def test_ac09_requires_signed_off_review(tmp_path):
    assert evaluate(tmp_path, None)["AC09"]["status"] == "NOT_RUN"
    write_json(tmp_path / "review/reviewer_signoff.json", {"reviewer": "A. Person", "reproduced_demo": True, "diagnosed_incident": True})
    assert evaluate(tmp_path, None)["AC09"]["status"] == "PASS"


def test_incident_stats_group_by_scenario():
    st = incident_stats([inc("F01", "crash"), inc("F01", "crash", status="UNRECOVERED"), inc("F02", "exp")])
    assert st["F01"]["trials"] == 2 and st["F01"]["final_status"] == {"RECOVERED_VALIDATED": 1, "UNRECOVERED": 1}
    assert st["F01"]["fault_to_notification_s"]["median"] == 20.0


# ---------------------------------------------------------------- evidence packaging
def test_package_evidence_commits_small_files_and_archives_everything(tmp_path):
    import tarfile

    from report import package_evidence as pe

    art = tmp_path / "artifacts"
    (art / "report").mkdir(parents=True)
    (art / "report/REPORT.md").write_text("r")
    (art / "faults/F01/trial-1-x").mkdir(parents=True)
    (art / "faults/F01/trial-1-x/timeline.json").write_text("{}")
    (art / "faults/F01/trial-1-x/requests.csv").write_text("big,raw\n" * 10)  # raw: archive only
    (art / "benchmark/e/runs/gpu-b1-c1-t1").mkdir(parents=True)
    (art / "benchmark/e/runs/gpu-b1-c1-t1/meta.json").write_text("{}")
    (art / "benchmark/e/runs/gpu-b1-c1-t1/requests.csv").write_text("x")
    (art / "native/prom-data/blk").mkdir(parents=True)
    (art / "native/prom-data/blk/chunk").write_text("tsdb")
    out, arc = tmp_path / "evidence", tmp_path / "raw.tgz"
    assert pe.main(["--artifacts", str(art), "--out", str(out), "--archive", str(arc)]) == 0
    names = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert "report/REPORT.md" in names and "faults/F01/trial-1-x/timeline.json" in names and "benchmark/e/runs/gpu-b1-c1-t1/meta.json" in names
    assert not any(n.endswith("requests.csv") for n in names)  # raw request logs are never committed
    assert "README.md" in names and "MANIFEST.json" in names
    with tarfile.open(arc) as t:
        members = t.getnames()
    assert any(m.endswith("requests.csv") for m in members) and not any("prom-data" in m for m in members)


def test_package_evidence_skips_oversized_files(tmp_path, monkeypatch):
    from report import package_evidence as pe

    art = tmp_path / "artifacts"
    (art / "soak/s").mkdir(parents=True)
    (art / "soak/s/soak.json").write_text("x" * 50)
    monkeypatch.setattr(pe, "MAX_COMMIT_BYTES", 10)
    keep, too_big = pe.collect(art)
    assert keep == [] and len(too_big) == 1
