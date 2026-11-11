"""Controller: event->counter logic, persistence, probing and bounded actions with a SIMULATED Docker."""
import json
from pathlib import Path

import requests

from common.util import read_jsonl
from ops.controller import Controller, ControllerConfig, Workload, apply_event_to_counters, event_time_iso
from prometheus_client import generate_latest

from ops.recovery import PolicyConfig, Probe


class FakeDocker:
    def __init__(self):
        self.running = True
        self.restarts = []

    def inspect_state(self, name):
        return {"Running": self.running, "Restarting": False}

    def restart(self, name, stop_timeout_s=10):
        self.restarts.append(name)
        return True, "ok"


class Resp:
    def __init__(self, code, body=None):
        self.status_code = code
        self._b = body or {}

    def json(self):
        return self._b


def cfg(tmp_path, **pol):
    return ControllerConfig(
        workloads=[Workload("inference-gpu", "lab-inference-gpu", "http://x")],
        policy=PolicyConfig(**pol), metrics_port=0,
        state_file=tmp_path / "state.json", events_file=tmp_path / "faults.jsonl",
    )


def evt(action, name="lab-inference-gpu", exit_code="137"):
    return {"Type": "container", "Action": action, "timeNano": 1_700_000_000_000_000_000,
            "Actor": {"Attributes": {"name": name, "exitCode": exit_code}}}


def test_restart_counted_only_for_start_after_die():
    st = {}
    assert not apply_event_to_counters(st, "create")
    assert not apply_event_to_counters(st, "start")  # first deploy
    assert not apply_event_to_counters(st, "die")
    assert apply_event_to_counters(st, "start")
    assert st["restarts_total"] == 1
    assert not apply_event_to_counters(st, "start")  # duplicate start does not double count


def test_event_time_is_converted_to_utc_iso():
    assert event_time_iso(evt("die")).endswith("Z")


def test_docker_events_are_recorded_and_counter_persists_across_controller_restarts(tmp_path):
    c = Controller(cfg(tmp_path), docker=FakeDocker())
    for a in ("start", "die", "start"):
        c.handle_event(evt(a))
    kinds = read_jsonl(tmp_path / "faults.jsonl")
    assert [k["action"] for k in kinds] == ["start", "die", "start"]
    assert kinds[1]["exit_code"] == "137" and kinds[2]["restart_counted"] is True
    # a new controller instance (e.g. host reboot of the controller) reloads the counter
    c2 = Controller(cfg(tmp_path), docker=FakeDocker())
    assert c2.state["workloads"]["inference-gpu"]["restarts_total"] == 1
    assert 'lab_workload_restarts_total{workload="inference-gpu"} 1.0' in generate_latest(c2.registry).decode()


def test_events_for_other_containers_are_ignored(tmp_path):
    c = Controller(cfg(tmp_path), docker=FakeDocker())
    c.handle_event(evt("die", name="somebody-else"))
    assert read_jsonl(tmp_path / "faults.jsonl") == []


def test_metrics_include_heartbeat_and_zero_valued_restart_series(tmp_path):
    c = Controller(cfg(tmp_path), docker=FakeDocker(), http_get=lambda *a, **k: Resp(200))
    c.tick()
    text = generate_latest(c.registry).decode()
    assert "lab_controller_heartbeat_timestamp_seconds" in text
    assert 'lab_workload_restarts_total{workload="inference-gpu"} 0.0' in text  # exists before first restart


def test_alive_but_unhealthy_service_is_restarted_after_three_probes_and_bounded(tmp_path):
    d = FakeDocker()
    now = [0.0]
    c = Controller(cfg(tmp_path), docker=d, http_get=lambda *a, **k: Resp(503, {"reason_code": "worker_stuck"}), clock=lambda: now[0])
    c.policies["inference-gpu"].observe(-1, Probe(True, True, True))
    for k in range(3):
        now[0] = k * 2.0
        c.tick()
    assert d.restarts == ["lab-inference-gpu"]
    recs = read_jsonl(tmp_path / "faults.jsonl")
    ra = [r for r in recs if r["kind"] == "restart_action"]
    assert len(ra) == 1 and ra[0]["owner"] == "controller" and ra[0]["ok"] is True
    assert json.loads((tmp_path / "state.json").read_text())["workloads"]["inference-gpu"]["controller_restart_actions"] == 1
    # immediately more failures within the post-restart grace window: no second restart
    for k in range(3, 12):
        now[0] = k * 2.0
        c.tick()
    assert len(d.restarts) == 1


def test_config_error_is_reported_to_operator_not_restarted(tmp_path):
    d = FakeDocker()
    now = [0.0]
    c = Controller(cfg(tmp_path), docker=d, http_get=lambda *a, **k: Resp(503, {"reason_code": "weights_missing"}), clock=lambda: now[0])
    for k in range(20):
        now[0] = k * 2.0
        c.tick()
    assert d.restarts == []
    kinds = [r["kind"] for r in read_jsonl(tmp_path / "faults.jsonl")]
    assert kinds.count("operator_required") == 1
    op = [r for r in read_jsonl(tmp_path / "faults.jsonl") if r["kind"] == "operator_required"][0]
    assert "RB-002" in op["runbook"]


def test_container_not_running_is_left_to_docker(tmp_path):
    d = FakeDocker()
    d.running = False
    now = [0.0]
    c = Controller(cfg(tmp_path), docker=d, http_get=lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()), clock=lambda: now[0])
    for k in range(30):
        now[0] = k * 2.0
        c.tick()
    assert d.restarts == []


def test_missing_optional_container_is_skipped(tmp_path):
    d = FakeDocker()
    d.inspect_state = lambda n: None
    c = Controller(cfg(tmp_path), docker=d)
    c.tick()  # must not raise
