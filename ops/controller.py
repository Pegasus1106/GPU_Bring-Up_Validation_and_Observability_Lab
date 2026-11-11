"""Host recovery controller.

Runs on the host (not in a container), outside the workload's failure domain:
  * streams Docker container events (or, in native mode on Sol, the supervisor's equivalent
    event stream) and writes them to faults.jsonl
  * probes each workload's /readyz every 2 s
  * applies the bounded recovery policy (ops/recovery.py)
  * keeps restart counters in a state file so they survive controller and
    workload restarts, and exposes them (plus a heartbeat) on a loopback
    Prometheus endpoint, which is what the RestartObserved alert reads.

    python -m ops.controller --config ops/controller.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests
import yaml
from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from common import endpoints
from common.util import REPO_ROOT, append_jsonl, iso, read_json, write_json
from ops.docker_cli import DockerCLI, iter_events
from ops.recovery import Decision, PolicyConfig, Probe, RecoveryPolicy, State

log = logging.getLogger("lab.controller")

RECORDED_ACTIONS = {"create", "start", "die", "kill", "stop", "restart", "oom", "destroy", "pause", "unpause"}


@dataclass
class Workload:
    name: str
    container: str
    url: str
    optional: bool = True


@dataclass
class ControllerConfig:
    workloads: list[Workload]
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    metrics_addr: str = "127.0.0.1"
    metrics_port: int = 9105
    state_file: Path = REPO_ROOT / "artifacts/controller/state.json"
    events_file: Path = REPO_ROOT / "artifacts/controller/faults.jsonl"
    probe_timeout_s: float = 1.5
    runtime: str = "docker"  # "native" => ops.native_runtime (Slurm/Sol, no Docker)

    @staticmethod
    def load(path: Path) -> "ControllerConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return ControllerConfig(
            workloads=[Workload(**w) for w in raw["workloads"]],
            policy=PolicyConfig(**raw.get("policy", {})),
            metrics_addr=raw.get("metrics_addr", "127.0.0.1"),
            metrics_port=int(raw.get("metrics_port", 9105)),
            state_file=REPO_ROOT / raw.get("state_file", "artifacts/controller/state.json"),
            events_file=REPO_ROOT / raw.get("events_file", "artifacts/controller/faults.jsonl"),
            probe_timeout_s=float(raw.get("probe_timeout_s", 1.5)),
            runtime=raw.get("runtime", "docker"),
        )


def event_time_iso(evt: dict) -> str | None:
    ns = evt.get("timeNano")
    if ns:
        return iso(datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc))
    if evt.get("time"):
        return iso(datetime.fromtimestamp(int(evt["time"]), tz=timezone.utc))
    return None


def apply_event_to_counters(wstate: dict, action: str) -> bool:
    """Update persistent counters for one container event. Returns True if a
    restart was counted. A restart = `start` that follows a `die`."""
    if action == "die":
        wstate["awaiting_restart"] = True
        return False
    if action == "start":
        if wstate.get("awaiting_restart"):
            wstate["awaiting_restart"] = False
            wstate["restarts_total"] = wstate.get("restarts_total", 0) + 1
            return True
        return False
    return False


class Controller:
    def __init__(self, cfg: ControllerConfig, docker: DockerCLI | None = None,
                 http_get: Callable[..., requests.Response] | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.docker = docker or DockerCLI()
        self._get = http_get or requests.get
        self._clock = clock
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self.policies = {w.name: RecoveryPolicy(cfg.policy) for w in cfg.workloads}
        self.by_container = {w.container: w for w in cfg.workloads}
        self.state = self._load_state()
        self._init_metrics()

    # -- persistent state -------------------------------------------------------------
    def _load_state(self) -> dict:
        p = self.cfg.state_file
        st = read_json(p) if p.exists() else {}
        st.setdefault("workloads", {})
        for w in self.cfg.workloads:
            st["workloads"].setdefault(w.name, {"restarts_total": 0, "controller_restart_actions": 0, "awaiting_restart": False})
        return st

    def _save_state(self) -> None:
        with self._state_lock:
            self.state["updated_utc"] = iso()
            write_json(self.cfg.state_file, self.state)

    # -- metrics ------------------------------------------------------------------------
    def _init_metrics(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.m_heartbeat = Gauge("lab_controller_heartbeat_timestamp_seconds", "Unix time of the last controller loop", registry=r)
        self.m_restarts = Counter("lab_workload_restarts_total", "Workload container restarts observed (persistent across controller restarts)", ["workload"], registry=r)
        self.m_actions = Counter("lab_controller_restart_actions_total", "Restarts performed by the controller", ["workload", "result"], registry=r)
        self.m_state = Gauge("lab_controller_state", "1 for the current recovery state of the workload", ["workload", "state"], registry=r)
        self.m_consec = Gauge("lab_controller_probe_failures_consecutive", "Consecutive failed readiness probes", ["workload"], registry=r)
        self.m_running = Gauge("lab_workload_container_running", "1 if the container is running (absent if it does not exist)", ["workload"], registry=r)
        self.m_ready = Gauge("lab_workload_ready_probe", "Last controller readiness probe result", ["workload"], registry=r)
        self.m_unres = Gauge("lab_controller_unresolved", "1 when automatic recovery is exhausted", ["workload"], registry=r)
        for w in self.cfg.workloads:
            prev = self.state["workloads"][w.name]
            self.m_restarts.labels(w.name).inc(prev.get("restarts_total", 0))
            for res in ("ok", "failed"):
                self.m_actions.labels(w.name, res)
            self._set_state_metric(w.name, State.STARTING)

    def _set_state_metric(self, name: str, st: State) -> None:
        for s in State:
            self.m_state.labels(name, s.value).set(1 if s == st else 0)
        self.m_unres.labels(name).set(1 if st == State.UNRESOLVED else 0)

    # -- evidence --------------------------------------------------------------------------
    def record(self, kind: str, **fields) -> None:
        append_jsonl(self.cfg.events_file, {"ts_utc": iso(), "kind": kind, **fields})

    # -- docker events (own thread) -------------------------------------------------------
    def handle_event(self, evt: dict) -> None:
        actor = evt.get("Actor", {})
        attrs = actor.get("Attributes", {})
        name = attrs.get("name", "")
        w = self.by_container.get(name)
        action = str(evt.get("Action") or evt.get("status") or "").split(":")[0].strip()
        if w is None or action not in RECORDED_ACTIONS:
            return
        with self._state_lock:
            counted = apply_event_to_counters(self.state["workloads"][w.name], action)
        if counted:
            self.m_restarts.labels(w.name).inc()
        self.record(
            "docker_event", workload=w.name, container=name, action=action,
            event_time_utc=event_time_iso(evt), exit_code=attrs.get("exitCode"), signal=attrs.get("signal"),
            restart_counted=counted, restarts_total=self.state["workloads"][w.name]["restarts_total"],
        )
        self._save_state()

    def _events_thread(self) -> None:
        names = [w.container for w in self.cfg.workloads]
        while not self._stop.is_set():
            try:
                proc = self.docker.events(names)
            except (FileNotFoundError, OSError) as e:
                log.warning("docker events unavailable: %s", e)
                self._stop.wait(5)
                continue
            try:
                for evt in iter_events(proc):
                    if self._stop.is_set():
                        break
                    self.handle_event(evt)
            finally:
                proc.terminate()
            self._stop.wait(2)  # reconnect

    # -- probing ------------------------------------------------------------------------------
    def probe(self, w: Workload, running: bool) -> Probe:
        if not running:
            return Probe(False, False, False)
        try:
            r = self._get(f"{w.url}/readyz", timeout=self.cfg.probe_timeout_s)
        except requests.RequestException:
            return Probe(True, False, False)
        reason = None
        if r.status_code != 200:
            try:
                reason = r.json().get("reason_code")
            except ValueError:
                reason = None
        return Probe(True, True, r.status_code == 200, reason)

    def tick(self) -> None:
        now = self._clock()
        self.m_heartbeat.set(time.time())
        for w in self.cfg.workloads:
            cs = self.docker.inspect_state(w.container)
            if cs is None:
                if not w.optional:
                    self.record("container_missing", workload=w.name, container=w.container)
                continue
            running = bool(cs.get("Running")) and not cs.get("Restarting")
            self.m_running.labels(w.name).set(1 if running else 0)
            pol = self.policies[w.name]
            pr = self.probe(w, running)
            self.m_ready.labels(w.name).set(1 if pr.ready else 0)
            d = pol.observe(now, pr)
            self.m_consec.labels(w.name).set(pol.consecutive_failures)
            self._set_state_metric(w.name, pol.state)
            self._act(w, pol, d, pr, now)

    def _act(self, w: Workload, pol: RecoveryPolicy, d: Decision, pr: Probe, now: float) -> None:
        base = dict(workload=w.name, container=w.container, state=d.state.value, reason_code=pr.reason_code,
                    consecutive_failures=pol.consecutive_failures)
        if d.changed:
            self.record("state_change", detail=d.reason, **base)
        if d.action == "restart":
            ok, out = self.docker.restart(w.container)
            pol.restarted(self._clock())
            self.m_actions.labels(w.name, "ok" if ok else "failed").inc()
            with self._state_lock:
                self.state["workloads"][w.name]["controller_restart_actions"] += 1
            self._save_state()
            self.record("restart_action", owner="controller", ok=ok, output=out[:300], detail=d.reason,
                        restarts_in_window=pol.restarts_in_window(self._clock()), **base)
        elif d.action == "hold_operator":
            self.record("operator_required", detail=d.reason, runbook=d.runbook, **base)
        elif d.action == "unresolved":
            self.record("unresolved", detail=d.reason, runbook=d.runbook, **base)
        elif d.action == "recovered":
            self.record("recovered", detail=d.reason, **base)

    # -- run ------------------------------------------------------------------------------
    def run_forever(self) -> None:
        start_http_server(self.cfg.metrics_port, addr=self.cfg.metrics_addr, registry=self.registry)
        threading.Thread(target=self._events_thread, daemon=True, name="docker-events").start()
        log.info("controller up: metrics on %s:%s, %d workloads", self.cfg.metrics_addr, self.cfg.metrics_port, len(self.cfg.workloads))
        self.record("controller_start", workloads=[w.name for w in self.cfg.workloads], policy=self.cfg.policy.__dict__)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the controller must not die on one bad tick
                log.exception("tick failed")
            self._stop.wait(max(0.0, self.cfg.policy.probe_interval_s - (time.monotonic() - t0)))

    def stop(self) -> None:
        self._stop.set()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "ops/controller.yaml")
    a = ap.parse_args(argv)
    cfg = ControllerConfig.load(a.config)
    runtime = None
    if cfg.runtime == "native" or endpoints.native():
        from ops.native_runtime import NativeRuntime

        runtime = NativeRuntime()
    c = Controller(cfg, docker=runtime)
    signal.signal(signal.SIGTERM, lambda *_: c.stop())
    signal.signal(signal.SIGINT, lambda *_: c.stop())
    c.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
