"""Preflight status logic, alert-decision helpers, incident metrics, receiver, baseline freezing."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from benchmarks.freeze_baseline import build_rules
from common.util import REPO_ROOT, epoch_to_iso
from faults.harness import delta_s, derive_metrics, find_alert_records, first_symptom_row
from ops.alert_receiver import flatten
from preflight import checks as C

def _find_promtool():
    found = shutil.which("promtool")
    if found:
        return found
    cands = sorted((REPO_ROOT / ".tools").glob("*/promtool"))
    return str(cands[0]) if cands else None


PROMTOOL = _find_promtool()


# ---------------------------------------------------------------- preflight (SIMULATED command runner)
def runner(table):
    def run(cmd, timeout=30.0):
        for key, res in table.items():
            if " ".join(cmd).startswith(key):
                return res
        return C.CmdResult(cmd, None, error=f"{cmd[0]}: command not found")
    return run


def ok(out=""):
    return C.CmdResult([], 0, out)


def test_smi_value_classification():
    assert C.classify_smi_value("[Not Supported]") == "unsupported"
    assert C.classify_smi_value("[N/A]") == "unavailable"
    assert C.classify_smi_value("62") == "supported"


def test_capability_discovery_keeps_three_states_separate():
    def run(cmd, timeout=30.0):
        arg = cmd[1]
        if "index" in arg:
            return ok("0\n")
        if "temperature.gpu" in arg:
            return ok("55\n")
        if "power.draw" in arg:
            return ok("[N/A]\n")
        if "ecc" in arg:
            return ok("[Not Supported]\n")
        return ok("1\n")
    caps = C.discover_smi_capabilities(run)
    assert caps["temperature.gpu"]["status"] == "supported"
    assert caps["power.draw"]["status"] == "unavailable"
    assert caps["ecc.mode.current"]["status"] == "unsupported"


def test_dcgm_capabilities_unsupported_vs_exporter_down():
    counters = "# c\nDCGM_FI_DEV_GPU_UTIL, gauge, x\nDCGM_FI_DEV_POWER_USAGE, gauge, y\n"
    caps = C.discover_dcgm_capabilities(counters, "DCGM_FI_DEV_GPU_UTIL{gpu=\"0\"} 3\n")
    assert caps["DCGM_FI_DEV_GPU_UTIL"]["status"] == "supported" and caps["DCGM_FI_DEV_POWER_USAGE"]["status"] == "unsupported"
    down = C.discover_dcgm_capabilities(counters, None, "ConnectionError")
    assert {v["status"] for v in down.values()} == {"unavailable"}


def test_pf02_requires_matching_identity():
    host = [{"uuid": "GPU-AAA", "name": "x"}]
    good = runner({"docker run": ok("GPU-AAA, x\n")})
    assert C.pf02("gpu", good, host, "img").status == C.PASS
    other = runner({"docker run": ok("GPU-BBB, x\n")})
    assert C.pf02("gpu", other, host, "img").status == C.FAIL
    none = runner({"docker run": C.CmdResult([], 125, "", "could not select device driver")})
    assert C.pf02("gpu", none, host, "img").status == C.FAIL
    assert C.pf02("gpu", good, [], "img").status == C.FAIL
    assert C.pf02("cpu", good, host, "img").status == C.SKIP


def test_pf03_fail_error_skip():
    assert C.pf03("gpu", {"ok": True}).status == C.PASS
    assert C.pf03("gpu", {"ok": False, "reason": "no device"}).status == C.FAIL
    assert C.pf03("gpu", None).status == C.ERROR
    assert C.pf03("cpu", None).status == C.SKIP


def test_pf06_unsupported_diagnostic_is_skip_not_pass():
    c = C.pf06("gpu", runner({}))  # dcgmi absent
    assert c.status == C.SKIP and "dcgmi" in c.reason


def test_required_failure_blocks_gpu_readiness_and_skip_does_not_satisfy_gpu_mode():
    checks = [C.Check("PF01", "a", C.PASS), C.Check("PF02", "b", C.PASS), C.Check("PF03", "c", C.FAIL), C.Check("PF05", "d", C.PASS)]
    s = C.summarize(checks, "gpu")
    assert s["gpu_ready"] is False and "PF03" in s["blocking"]
    checks = [C.Check("PF01", "a", C.PASS), C.Check("PF02", "b", C.SKIP), C.Check("PF03", "c", C.PASS), C.Check("PF05", "d", C.PASS)]
    assert C.summarize(checks, "gpu")["gpu_ready"] is False  # skipped visibility check is not acceptable evidence
    checks = [C.Check("PF01", "a", C.PASS), C.Check("PF02", "b", C.PASS), C.Check("PF03", "c", C.PASS), C.Check("PF05", "d", C.PASS), C.Check("PF06", "e", C.SKIP)]
    assert C.summarize(checks, "gpu")["gpu_ready"] is True  # optional diagnostic SKIP does not block


def test_pf04_reports_specific_failing_path():
    c = C.pf04({"checks": {"model_weights": {"ok": False, "reason": "sha256 mismatch"}, "disk_space": {"ok": True}}})
    assert c.status == C.FAIL and "model_weights" in c.reason


# ---------------------------------------------------------------- incident timing helpers
def test_delta_and_derived_metrics_separate_detection_from_recovery():
    t = {"injected_utc": "2026-01-01T00:00:00.000Z", "first_observation_utc": "2026-01-01T00:00:01.500Z",
         "alert_firing_utc": "2026-01-01T00:00:20.000Z", "notification_received_utc": "2026-01-01T00:00:25.000Z",
         "corrective_action_utc": "2026-01-01T00:00:40.000Z", "validation_complete_utc": "2026-01-01T00:00:55.000Z"}
    m = derive_metrics(t, operator_corrected=True)
    assert m["fault_to_first_observation_s"] == 1.5 and m["fault_to_notification_s"] == 25.0
    assert m["fault_to_recovery_s"] == 55.0 and m["correction_to_recovery_s"] == 15.0
    assert "correction_to_recovery_s" not in derive_metrics(t, operator_corrected=False)
    assert delta_s(None, "x") is None


def test_alert_record_selection_filters_name_status_time_instance():
    recs = [
        {"alertname": "ServiceUnavailable", "status": "firing", "received_utc": "2026-01-01T00:00:30.000Z", "labels": {"instance": "127.0.0.1:8010"}},
        {"alertname": "ServiceUnavailable", "status": "firing", "received_utc": "2026-01-01T00:00:31.000Z", "labels": {"instance": "127.0.0.1:8000"}},
        {"alertname": "ServiceUnavailable", "status": "resolved", "received_utc": "2026-01-01T00:01:00.000Z", "labels": {"instance": "127.0.0.1:8010"}},
        {"alertname": "ServiceUnavailable", "status": "firing", "received_utc": "2025-12-31T00:00:00.000Z", "labels": {"instance": "127.0.0.1:8010"}},
    ]
    since = 1767225600.0  # 2026-01-01T00:00:00Z
    only_fault = find_alert_records(recs, "ServiceUnavailable", since, "firing", lambda i: i.endswith(":8010"))
    assert [r["received_utc"] for r in only_fault] == ["2026-01-01T00:00:30.000Z"]
    assert len(find_alert_records(recs, "ServiceUnavailable", since, "resolved")) == 1


def test_first_symptom_row_uses_earliest_matching_row_after_injection():
    rows = [
        {"phase": "measure", "outcome": "success", "t_end_utc": "2026-01-01T00:00:05.000Z"},
        {"phase": "measure", "outcome": "connection_error", "t_end_utc": "2026-01-01T00:00:12.000Z"},
        {"phase": "measure", "outcome": "connection_error", "t_end_utc": "2026-01-01T00:00:11.000Z"},
    ]
    t0 = 1767225600.0 + 10
    assert first_symptom_row(rows, t0, lambda r: r["outcome"] != "success") == "2026-01-01T00:00:11.000Z"


# ---------------------------------------------------------------- local alert receiver
def test_receiver_flattens_alertmanager_payload():
    payload = {"receiver": "local-jsonl", "groupKey": "g", "alerts": [
        {"status": "firing", "labels": {"alertname": "TelemetryUnavailable", "job": "dcgm-exporter"}, "annotations": {"summary": "s"},
         "startsAt": "2026-01-01T00:00:00Z", "endsAt": "0001-01-01T00:00:00Z"}]}
    r = flatten(payload, "2026-01-01T00:00:06.000Z")
    assert r[0]["alertname"] == "TelemetryUnavailable" and r[0]["received_utc"].endswith("06.000Z") and r[0]["status"] == "firing"


# ---------------------------------------------------------------- baseline freezing
SUMMARY = {"experiment_id": "x", "cells": {
    "gpu-b8-c4": {"p95_ms": {"mean": 12.5, "n": 3}},
    "gpu-b8-c1": {"p95_ms": {"mean": 5.0, "n": 3}},
    "cpu-b32-c4": {"p95_ms": {"mean": 800.0, "n": 3}},
    "cpu-b1-c4": {"p95_ms": {"mean": None, "n": 0}},
}}


def test_freeze_baseline_only_matching_concurrency_and_complete_cells():
    text, frozen = build_rules(SUMMARY, concurrency=4)
    assert set(frozen) == {"gpu-b8-c4", "cpu-b32-c4"}
    doc = yaml.safe_load(text)
    rules = doc["groups"][0]["rules"]
    labs = {(r["labels"]["backend"], r["labels"]["batch_size"]) for r in rules}
    assert labs == {("cuda", "8"), ("cpu", "32")}
    assert any("vector(0.012500)" in r["expr"] for r in rules)


def test_freeze_baseline_empty_is_valid_rule_file():
    text, frozen = build_rules({"experiment_id": "x", "cells": {}}, 4)
    assert frozen == {} and yaml.safe_load(text) == {"groups": []}


@pytest.mark.skipif(PROMTOOL is None, reason="promtool not installed")
def test_generated_baseline_passes_promtool(tmp_path):
    text, _ = build_rules(SUMMARY, 4)
    f = tmp_path / "baseline.yml"
    f.write_text(text)
    r = subprocess.run([PROMTOOL, "check", "rules", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- alert rules (promtool)
@pytest.mark.skipif(PROMTOOL is None, reason="promtool not installed")
def test_alert_rule_unit_tests_pass_under_promtool():
    r = subprocess.run([PROMTOOL, "test", "rules", str(REPO_ROOT / "monitoring/tests/alerts_test.yml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(PROMTOOL is None, reason="promtool not installed")
def test_prometheus_config_is_valid(tmp_path):
    # copy config + a target file so file_sd paths resolve for `check config`
    cfg = yaml.safe_load((REPO_ROOT / "monitoring/prometheus.yml").read_text())
    (tmp_path / "rules").mkdir()
    shutil.copy(REPO_ROOT / "monitoring/rules/alerts.yml", tmp_path / "rules/alerts.yml")
    cfg["rule_files"] = [str(tmp_path / "rules/*.yml")]
    for j in cfg["scrape_configs"]:
        for fsd in j.get("file_sd_configs", []):
            fsd["files"] = [str(tmp_path / "t.yml")]
    (tmp_path / "t.yml").write_text("[]")
    (tmp_path / "prom.yml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([PROMTOOL, "check", "config", str(tmp_path / "prom.yml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_alertmanager_routes_to_local_webhook_only_with_short_group_wait():
    am = yaml.safe_load((REPO_ROOT / "monitoring/alertmanager/alertmanager.yml").read_text())
    assert am["route"]["group_wait"] == "5s"
    urls = [w["url"] for r in am["receivers"] for w in r.get("webhook_configs", [])]
    assert urls == ["http://127.0.0.1:9094/alerts"]
    assert all(set(r) <= {"name", "webhook_configs"} for r in am["receivers"])  # no external integrations


def test_scrape_and_rule_intervals_are_5s():
    cfg = yaml.safe_load((REPO_ROOT / "monitoring/prometheus.yml").read_text())
    assert cfg["global"]["scrape_interval"] == "5s" and cfg["global"]["evaluation_interval"] == "5s"
    rules = yaml.safe_load((REPO_ROOT / "monitoring/rules/alerts.yml").read_text())
    assert all(g["interval"] == "5s" for g in rules["groups"])


# ---------------------------------------------------------------- compose invariants
def test_compose_binds_loopback_only_and_no_docker_socket_exposed():
    text = (REPO_ROOT / "deploy/docker-compose.yml").read_text()
    doc = yaml.safe_load(text)
    for name, svc in doc["services"].items():
        for p in svc.get("ports", []):
            assert str(p).startswith("127.0.0.1:"), f"{name} publishes {p} on a non-loopback address"
        assert not any("docker.sock" in str(v) for v in svc.get("volumes", [])), f"{name} mounts the Docker socket"
    assert doc["services"]["inference-gpu"]["restart"] == "on-failure:3"
    assert doc["services"]["inference-gpu"]["environment"]["LAB_BACKEND"] == "cuda"
    # monitoring services must not depend on the workload (they survive its restarts)
    for m in ("prometheus", "alertmanager", "grafana", "node-exporter"):
        assert "depends_on" not in doc["services"][m] and "profiles" not in doc["services"][m]


def test_dashboards_are_three_views_with_documented_panels():
    d = REPO_ROOT / "monitoring/grafana/dashboards"
    files = sorted(d.glob("*.json"))
    assert len(files) == 3
    for f in files:
        for p in json.loads(f.read_text())["panels"]:
            if p["type"] == "text":
                continue
            for key in ("Unit:", "Source:", "Interval:", "Missing data:"):
                assert key in p["description"], f"{f.name}: panel {p['title']!r} lacks {key}"
            assert p["fieldConfig"]["defaults"].get("noValue"), f"{f.name}: {p['title']!r} has no noValue text (missing must not read as 0)"


# ---------------------------------------------------------------- PF06 (output taken from a real Sol run)
OURS = "GPU-6b419034-6a17-774d-dfcd-6e0ef98903ff"
OTHER = "GPU-12867839-714d-b247-640e-73ed4c94faf2"
DISCOVERY = f"""4 GPUs found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA A100-SXM4-80GB                                          |
|        | PCI Bus ID: 00000000:07:00.0                                         |
|        | Device UUID: {OTHER}                  |
+--------+----------------------------------------------------------------------+
| 2      | Name: NVIDIA A100-SXM4-80GB                                          |
|        | PCI Bus ID: 00000000:0F:00.0                                         |
|        | Device UUID: {OURS}                  |
+--------+----------------------------------------------------------------------+
"""
DIAG_PASS = """Successfully ran diagnostic for group.
+---------------------------+------------------------------------------------+
| Diagnostic                | Result                                         |
+===========================+================================================+
|-----  Metadata  ----------+------------------------------------------------|
| DCGM Version              | 4.7.0                                          |
| Driver Version Detected   | 615.71.09                                      |
|-----  Deployment  --------+------------------------------------------------|
| software                  | Pass                                           |
|                           | GPU2: Pass                                     |
+---------------------------+------------------------------------------------+
"""


def pf06_runner(diag_out=DIAG_PASS, diag_rc=0, diag_err="", discovery=DISCOVERY, discovery_rc=0, seen=None):
    def run(cmd, timeout=30.0):
        line = " ".join(cmd)
        if seen is not None:
            seen.append(line)
        if line.startswith("dcgmi --version"):
            return ok("DCGM 4.7.0")
        if line.startswith("nvidia-smi --query-gpu=utilization.gpu"):
            return ok("0\n")
        if line.startswith("nvidia-smi --query-gpu=uuid"):
            return ok(OURS + "\n")
        if line.startswith("dcgmi discovery"):
            return C.CmdResult(cmd, discovery_rc, discovery, "")
        if line.startswith("dcgmi diag"):
            return C.CmdResult(cmd, diag_rc, diag_out, diag_err)
        return C.CmdResult(cmd, None, error="not found")
    return run


def test_pf06_passes_when_the_diag_table_says_pass_even_if_stderr_mentions_failure():
    # regression: a warning containing the word "fail" on stderr used to turn a passing diagnostic into FAIL
    c = C.pf06("gpu", pf06_runner(diag_err="warning: failed to open optional plugin libfoo.so\n"))
    assert c.status == C.PASS and c.evidence["verdicts"]["Pass"] >= 2


def test_pf06_only_diagnoses_the_gpu_this_job_owns():
    seen = []
    c = C.pf06("gpu", pf06_runner(seen=seen))
    diag = [x for x in seen if x.startswith("dcgmi diag")][0]
    assert diag.endswith("-i 2")  # our UUID maps to DCGM id 2; the other three node GPUs are never touched
    assert c.evidence["dcgm_gpu_ids"] == [2]


def test_pf06_real_failure_row_is_a_fail():
    bad = DIAG_PASS.replace("software                  | Pass ", "software                  | Fail ")
    c = C.pf06("gpu", pf06_runner(diag_out=bad))
    assert c.status == C.FAIL and "failing" in c.reason


def test_pf06_skips_instead_of_running_on_unknown_gpus():
    no_match = DISCOVERY.replace(OURS, "GPU-00000000-0000-0000-0000-000000000000")
    seen = []
    c = C.pf06("gpu", pf06_runner(discovery=no_match, seen=seen))
    assert c.status == C.SKIP and not any(x.startswith("dcgmi diag") for x in seen)


def test_pf06_skips_when_dcgm_host_engine_is_unreachable():
    c = C.pf06("gpu", pf06_runner(discovery="Error: unable to establish a connection to the specified host", discovery_rc=254))
    assert c.status == C.SKIP and "host engine" in c.reason


def test_pf06_unparseable_output_is_error_not_pass():
    c = C.pf06("gpu", pf06_runner(diag_out="something unexpected\n"))
    assert c.status == C.ERROR


def test_dcgm_id_mapping_handles_uuid_on_the_id_row():
    one_row = f"| 3 | {OURS} | A100 |\n| 1 | {OTHER} | A100 |\n"
    assert C.dcgm_ids_for_uuids(one_row, [OURS]) == [3]


# ---------------------------------------------------------------- drills wait for a quiet baseline instead of aborting
def test_wait_alert_quiet_waits_for_a_leftover_alert_to_clear(monkeypatch):
    from faults import harness

    state = {"t": 0.0}
    monkeypatch.setattr(harness.time, "time", lambda: state["t"])
    monkeypatch.setattr(harness.time, "sleep", lambda d: state.__setitem__("t", state["t"] + d))
    # firing for the first 30 s (leftover from the previous drill), then clear
    monkeypatch.setattr(harness, "prom_alerts", lambda url: [{"labels": {"alertname": "LatencyRegression"}, "state": "firing"}] if state["t"] < 30 else [])
    s = type("S", (), {"expected_alert": "LatencyRegression", "alert_instance_pred": None})()
    quiet, waited = harness.wait_alert_quiet(s, harness.Ctx(quiet_timeout_s=120, quiet_for_s=15))
    assert quiet and 44 <= waited <= 50  # 30 s leftover + 15 s required quiet


def test_wait_alert_quiet_gives_up_on_a_persistent_alert(monkeypatch):
    from faults import harness

    state = {"t": 0.0}
    monkeypatch.setattr(harness.time, "time", lambda: state["t"])
    monkeypatch.setattr(harness.time, "sleep", lambda d: state.__setitem__("t", state["t"] + d))
    monkeypatch.setattr(harness, "prom_alerts", lambda url: [{"labels": {"alertname": "LatencyRegression"}, "state": "pending"}])
    s = type("S", (), {"expected_alert": "LatencyRegression", "alert_instance_pred": None})()
    quiet, waited = harness.wait_alert_quiet(s, harness.Ctx(quiet_timeout_s=60, quiet_for_s=15))
    assert not quiet and waited >= 60
