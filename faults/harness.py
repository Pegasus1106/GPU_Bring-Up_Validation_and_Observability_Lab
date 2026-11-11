"""Fault drill harness: runs one scenario trial under healthy baseline load and
produces an incident record plus evidence files.

Every scenario is an explicit lab script (faults/f0X_*.py) declaring:
  precondition()   what must be true before injecting
  inject()         the controlled fault (start timestamp is recorded by the harness)
  expected         the symptoms and alert it should produce
  stop()           the stop / correction action (also timestamped)
  cleanup()        return the lab to a healthy baseline, then verified

Simulated vs actual: every incident record carries `fault_class`
("actual" = a real process/container/config failure; "injected_latency" = a real,
lab-only delay inside the worker; never a faked metric).

Timestamps: UTC wall clock (one host, so processes correlate); durations are
differences of those UTC instants on the same machine.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from benchmarks import aggregate as agg
from common import endpoints
from faults import runtime
from faults.runtime import compose, docker  # noqa: F401  (re-exported for callers)
from common.util import (REPO_ROOT, append_jsonl, epoch_to_iso, git_commit, iso, new_run_id, parse_iso, read_json,
                         read_jsonl, write_json)
from loadgen.loadgen import LoadConfig, LoadRunner
from validation.api_checks import quick_validate

ART = REPO_ROOT / "artifacts"
INCIDENTS = ART / "faults/incidents.jsonl"
ALERTS_FILE = ART / "alerts/alerts.jsonl"
CONTROLLER_EVENTS = ART / "controller/faults.jsonl"
CONTROLLER_STATE = ART / "controller/state.json"


# ----------------------------------------------------------------------------- context
@dataclass
class Ctx:
    profile: str = "gpu"  # gpu | cpu - which baseline workload is under load
    prom_url: str = field(default_factory=lambda: endpoints.url("prometheus"))
    controller_metrics: str = field(default_factory=lambda: endpoints.url("controller") + "/metrics")
    fault_url: str = field(default_factory=lambda: endpoints.url("fault"))
    batch_size: int = 8
    concurrency: int = 4
    baseline_s: float = 20.0  # steady healthy load before injection
    recovery_timeout_s: float = 180.0
    alert_timeout_s: float = 120.0
    quiet_timeout_s: float = 180.0  # max wait for the expected alert to clear on the healthy baseline before a drill
    quiet_for_s: float = 15.0  # ... and it must then stay clear this long
    exporter_container: str = field(default_factory=lambda: runtime.get().exporter())
    exporter_compose_service: str = "dcgm-exporter"
    failure_phase_only: bool = False
    delay_ms: Optional[int] = None
    trial_dir: Path = ART / "faults"

    @property
    def backend(self) -> str:
        return "cuda" if self.profile == "gpu" else "cpu"

    @property
    def url(self) -> str:
        return endpoints.url("gpu" if self.profile == "gpu" else "cpu")

    @property
    def container(self) -> str:
        return runtime.get().workload(self.profile)

    @property
    def service(self) -> str:
        return f"inference-{self.profile}"

    @property
    def inject_dir(self) -> Path:
        return ART / "faults/inject" / self.profile

    @property
    def fault_inject_dir(self) -> Path:
        return ART / "faults/inject/fault"


# ----------------------------------------------------------------------------- probes
def http_ready(url: str, timeout: float = 1.5) -> tuple[bool, dict]:
    try:
        r = requests.get(f"{url}/readyz", timeout=timeout)
        try:
            body = r.json()
        except ValueError:
            body = {}
        return r.status_code == 200, {"status": r.status_code, **body}
    except requests.RequestException as e:
        return False, {"status": None, "error": type(e).__name__}


def prom_query(prom_url: str, q: str) -> list[dict]:
    try:
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": q}, timeout=3)
        return r.json()["data"]["result"]
    except Exception:  # noqa: BLE001
        return []


def prom_alerts(prom_url: str) -> list[dict]:
    try:
        return requests.get(f"{prom_url}/api/v1/alerts", timeout=3).json()["data"]["alerts"]
    except Exception:  # noqa: BLE001
        return []


def container_inspect(name: str) -> dict:
    return runtime.get().inspect(name)


def controller_counters() -> dict:
    return (read_json(CONTROLLER_STATE)["workloads"] if CONTROLLER_STATE.exists() else {})


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def find_alert_records(records: list[dict], alertname: str, since_epoch: float, status: str,
                       instance_pred: Callable[[str], bool] | None = None) -> list[dict]:
    out = []
    for r in records:
        if r.get("alertname") != alertname or r.get("status") != status:
            continue
        # an alert that started firing before the fault was injected is not caused by it
        if status == "firing" and r.get("starts_at") and parse_iso(r["starts_at"]).timestamp() < since_epoch - 1.0:
            continue
        if parse_iso(r["received_utc"]).timestamp() < since_epoch:
            continue
        inst = (r.get("labels") or {}).get("instance", "")
        if instance_pred and not instance_pred(inst):
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["received_utc"])


def delta_s(later_iso: Optional[str], earlier_iso: Optional[str]) -> Optional[float]:
    if not later_iso or not earlier_iso:
        return None
    return round(parse_iso(later_iso).timestamp() - parse_iso(earlier_iso).timestamp(), 3)


def derive_metrics(t: dict, operator_corrected: bool) -> dict:
    """Fault-to-detection and fault-to-recovery are reported separately; operator-corrected
    faults also report correction-to-recovery."""
    m = {
        "fault_to_first_observation_s": delta_s(t.get("first_observation_utc"), t["injected_utc"]),
        "fault_to_alert_pending_s": delta_s(t.get("alert_pending_utc"), t["injected_utc"]),
        "fault_to_alert_firing_s": delta_s(t.get("alert_firing_utc"), t["injected_utc"]),
        "fault_to_notification_s": delta_s(t.get("notification_received_utc"), t["injected_utc"]),
        "fault_to_recovery_s": delta_s(t.get("validation_complete_utc"), t["injected_utc"]),
    }
    if operator_corrected:
        m["correction_to_recovery_s"] = delta_s(t.get("validation_complete_utc"), t.get("corrective_action_utc"))
    return m


def first_symptom_row(rows: list[dict], since_epoch: float, is_symptom: Callable[[dict], bool]) -> Optional[str]:
    best = None
    for r in rows:
        if r["phase"] != "measure" and r["phase"] != "warmup":
            continue
        ts = parse_iso(r["t_end_utc"]).timestamp()
        if ts >= since_epoch and is_symptom(r):
            if best is None or r["t_end_utc"] < best:
                best = r["t_end_utc"]
    return best


# ----------------------------------------------------------------------------- scenario base
class Scenario:
    id = "F00"
    name = "base"
    fault_class = "actual"  # actual | injected_latency
    expected_symptoms = ""
    expected_alert: Optional[str] = None
    alert_instance_pred: Optional[Callable[[str], bool]] = None
    held_fault = False  # True => AC06 45 s alert receipt applies
    receipt_before_stop = True  # False for self-recovering faults (Docker restarts the process)
    symptom_required = True  # a trial where no symptom was ever observed did not exercise the fault
    operator_corrected = False
    recovery_owner = ""
    hold_s = 0.0  # minimum time the fault stays active before stop()
    runbook = ""
    recovery_url_is_fault_case = False
    controller_must_not_restart: Optional[str] = None  # workload name that must see 0 controller restarts

    def precondition(self, c: Ctx) -> tuple[bool, str]:
        ok, info = http_ready(c.url)
        return (True, "") if ok else (False, f"baseline service {c.url} not ready: {info}")

    def inject(self, c: Ctx) -> dict:
        raise NotImplementedError

    def symptom_observed(self, c: Ctx) -> bool:
        return False

    def row_is_symptom(self, row: dict) -> bool:
        return row["outcome"] != "success"

    def stop(self, c: Ctx) -> dict:
        return {}

    def recovery_url(self, c: Ctx) -> str:
        return c.fault_url if self.recovery_url_is_fault_case else c.url

    def recovery_backend(self, c: Ctx) -> str:
        return c.backend if not self.recovery_url_is_fault_case else getattr(self, "fault_backend_after", c.backend)

    def recovery_extra(self, c: Ctx) -> bool:
        """Additional recovery condition beyond readiness (e.g. exporter scraped again)."""
        return True

    def extra_checks(self, c: Ctx, t: dict) -> dict:
        return {}

    def cleanup(self, c: Ctx) -> None:
        pass

    def verify_cleanup(self, c: Ctx) -> tuple[bool, str]:
        ok, info = http_ready(c.url)
        return (ok, "" if ok else f"baseline not ready after cleanup: {info}")


# ----------------------------------------------------------------------------- trial runner
class Observer(threading.Thread):
    """Polls symptom + Prometheus alert state at ~4 Hz / 1 Hz and records first-seen times."""

    def __init__(self, scenario: Scenario, c: Ctx, injected_epoch: float):
        super().__init__(daemon=True, name="observer")
        self.s, self.c, self.t0 = scenario, c, injected_epoch
        self.stop_evt = threading.Event()
        self.first_observation: Optional[str] = None
        self.alert_pending: Optional[str] = None
        self.alert_firing: Optional[str] = None
        self.max_queue_depth: Optional[float] = None
        self.queue_capacity: Optional[float] = None

    def run(self) -> None:
        last_alert_poll = 0.0
        while not self.stop_evt.is_set():
            now = time.time()
            if self.first_observation is None:
                try:
                    if self.s.symptom_observed(self.c):
                        self.first_observation = epoch_to_iso(now)
                except Exception:  # noqa: BLE001
                    pass
            if self.s.expected_alert and now - last_alert_poll >= 1.0:
                last_alert_poll = now
                for a in prom_alerts(self.c.prom_url):
                    if a["labels"].get("alertname") != self.s.expected_alert:
                        continue
                    inst = a["labels"].get("instance", "")
                    if self.s.alert_instance_pred and not self.s.alert_instance_pred(inst):
                        continue
                    if a["state"] == "pending" and self.alert_pending is None:
                        self.alert_pending = epoch_to_iso(now)
                    if a["state"] == "firing" and self.alert_firing is None:
                        self.alert_firing = epoch_to_iso(now)
                        if self.alert_pending is None:
                            self.alert_pending = self.alert_firing
                try:
                    r = requests.get(f"{self.c.url}/metrics", timeout=1.5).text
                    import re

                    d = re.search(r"inference_queue_depth\{[^}]*\} ([0-9.]+)", r)
                    cap = re.search(r"inference_queue_capacity\{[^}]*\} ([0-9.]+)", r)
                    if d:
                        v = float(d.group(1))
                        self.max_queue_depth = v if self.max_queue_depth is None else max(self.max_queue_depth, v)
                    if cap:
                        self.queue_capacity = float(cap.group(1))
                except Exception:  # noqa: BLE001
                    pass
            self.stop_evt.wait(0.25)


def active_alerts(prom_url: str, alertname: str, instance_pred: Callable[[str], bool] | None = None) -> list[dict]:
    return [a for a in prom_alerts(prom_url) if a["labels"].get("alertname") == alertname
            and (not instance_pred or instance_pred(a["labels"].get("instance", "")))]


def wait_alert_quiet(s: Scenario, c: "Ctx", poll_s: float = 2.0) -> tuple[bool, float]:
    """True once the expected alert has been absent (not pending, not firing) for `quiet_for_s` seconds."""
    start = time.time()
    clear_since: Optional[float] = None
    while time.time() - start < c.quiet_timeout_s:
        if active_alerts(c.prom_url, s.expected_alert, s.alert_instance_pred):
            clear_since = None
        else:
            clear_since = clear_since or time.time()
            if time.time() - clear_since >= c.quiet_for_s:
                return True, time.time() - start
        time.sleep(poll_s)
    return False, time.time() - start


def wait_for_receipt(s: Scenario, since_epoch: float, timeout: float) -> Optional[dict]:
    end = time.time() + timeout
    while time.time() < end:
        recs = find_alert_records(read_jsonl(ALERTS_FILE), s.expected_alert, since_epoch, "firing", s.alert_instance_pred)
        if recs:
            return recs[0]
        time.sleep(0.5)
    return None


def wait_ready_validated(s: Scenario, c: Ctx, timeout: float, restart_marker: Optional[str]) -> dict:
    url, backend = s.recovery_url(c), s.recovery_backend(c)
    end = time.time() + timeout
    out: dict = {"first_ready_utc": None, "validation_complete_utc": None, "validation": None}
    while time.time() < end:
        ok, info = http_ready(url)
        restarted = True
        if restart_marker is not None:
            restarted = container_inspect(c.container).get("StartedAt") not in (restart_marker, None)
        if ok and restarted and s.recovery_extra(c):
            out["first_ready_utc"] = epoch_to_iso(time.time())
            v = quick_validate(url, backend)
            out["validation"] = v
            out["validation_complete_utc"] = epoch_to_iso(time.time())
            return out
        time.sleep(0.25)
    return out


def run_trial(s: Scenario, trial: int, c: Ctx) -> dict:
    run_id = new_run_id(f"{s.id}-t{trial}")
    tdir = c.trial_dir / s.id / f"trial-{trial}-{run_id[-17:]}"
    tdir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "scenario": s.id, "scenario_name": s.name, "trial": trial, "run_id": run_id, "commit": git_commit(),
        "profile": c.profile, "fault_class": s.fault_class, "expected_symptoms": s.expected_symptoms,
        "expected_alert": s.expected_alert, "action_owner": s.recovery_owner, "evidence_dir": str(tdir.relative_to(REPO_ROOT)) if tdir.is_relative_to(REPO_ROOT) else str(tdir),
        "load": {"batch_size": c.batch_size, "concurrency": c.concurrency},
        "final_status": "ERROR", "errors": [],
    }
    t: dict[str, Any] = {}
    runner: Optional[LoadRunner] = None
    observer: Optional[Observer] = None
    marker = None
    try:
        ok, why = s.precondition(c)
        if not ok:
            rec.update(final_status="PRECONDITION_FAILED", errors=[why])
            return _finish(rec, t, s, tdir)

        counters_before = controller_counters()
        marker = container_inspect(c.container).get("StartedAt")
        runner = LoadRunner(LoadConfig(c.url, c.backend, c.batch_size, c.concurrency, 20, None, run_id=run_id, out_dir=tdir, out_name="requests.csv")).start()
        time.sleep(c.baseline_s)  # healthy baseline load before the fault

        if s.expected_alert:
            # A drill is only valid if its alert is NOT active on the healthy baseline. The previous drill can leave a
            # real, transient alert behind (e.g. starting a fault-case process briefly slows the healthy service and the
            # 60 s rate window needs time to clear), so wait for quiet; abort only if it never clears.
            quiet, waited = wait_alert_quiet(s, c)
            rec["baseline_quiet_wait_s"] = round(waited, 1)
            if not quiet:
                rec.update(final_status="BASELINE_ALERT_ACTIVE", errors=[f"{s.expected_alert} stayed active on the healthy baseline for {c.quiet_timeout_s:.0f}s "
                                                                            "(persistent false positive); fix the rule/baseline, then re-run"])
                return _finish(rec, t, s, tdir)
        injected = time.time()
        t["injected_utc"] = epoch_to_iso(injected)
        observer = Observer(s, c, injected)
        observer.start()
        rec["inject_detail"] = s.inject(c)

        # hold the fault at least hold_s and (for alerting scenarios) until the alert is received
        hold_until = injected + s.hold_s
        receipt = None
        if s.expected_alert and s.receipt_before_stop:
            # the fault must persist until the alert has been received locally
            receipt = wait_for_receipt(s, injected, max(c.alert_timeout_s, 60))
        while time.time() < hold_until:
            time.sleep(0.25)

        t["corrective_action_utc"] = epoch_to_iso(time.time())
        rec["stop_detail"] = s.stop(c)
        # recovery is timed immediately after the stop action; it must not wait for alert bookkeeping
        if getattr(s, "skip_recovery", False):
            res = {"first_ready_utc": None, "validation_complete_utc": None, "validation": None}
        else:
            res = wait_ready_validated(s, c, c.recovery_timeout_s, marker if s.id == "F01" else None)
        t.update(first_ready_utc=res["first_ready_utc"], validation_complete_utc=res["validation_complete_utc"])
        rec["validation"] = res["validation"]
        if s.expected_alert and receipt is None:  # self-recovering faults (F01): alert arrives around/after recovery
            receipt = wait_for_receipt(s, injected, 30)
        if receipt:
            t["notification_received_utc"] = receipt["received_utc"]
            t["alert_firing_utc"] = receipt.get("starts_at")

        # alert resolution (informational)
        if s.expected_alert and receipt:
            end = time.time() + 90
            while time.time() < end:
                rr = find_alert_records(read_jsonl(ALERTS_FILE), s.expected_alert, injected, "resolved", s.alert_instance_pred)
                if rr:
                    t["alert_resolved_utc"] = rr[0]["received_utc"]
                    break
                time.sleep(1)
        observer.stop_evt.set()
        observer.join(timeout=3)
        t["alert_pending_utc"] = observer.alert_pending
        if t.get("alert_firing_utc") is None:
            t["alert_firing_utc"] = observer.alert_firing

        runner.stop()
        runner.wait()
        rows = agg.read_requests(runner.path)
        sym = first_symptom_row(rows, injected, s.row_is_symptom)
        cand = [x for x in (observer.first_observation, sym) if x]
        t["first_observation_utc"] = min(cand) if cand else None
        rec["first_observation_sources"] = {"poll": observer.first_observation, "loadgen": sym}

        # request outcomes during the fault window
        end_win = parse_iso(t["validation_complete_utc"]).timestamp() if t.get("validation_complete_utc") else time.time()
        win = [r for r in rows if injected <= parse_iso(r["t_start_utc"]).timestamp() <= end_win]
        oc: dict[str, int] = {}
        for r in win:
            oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
        rec["requests_during_fault"] = {"attempts": len(win), "outcomes": oc,
                                        "success_fraction": (oc.get("success", 0) / len(win)) if win else None}
        rec["queue"] = {"max_depth": observer.max_queue_depth, "capacity": observer.queue_capacity,
                        "bounded": (observer.max_queue_depth is not None and observer.queue_capacity is not None and observer.max_queue_depth <= observer.queue_capacity)}

        # restart accounting from the persistent controller counters
        counters_after = controller_counters()
        rec["restart_count"] = {
            w: {"restarts_observed": counters_after.get(w, {}).get("restarts_total", 0) - counters_before.get(w, {}).get("restarts_total", 0),
                "controller_restart_actions": counters_after.get(w, {}).get("controller_restart_actions", 0) - counters_before.get(w, {}).get("controller_restart_actions", 0)}
            for w in counters_after
        }
        if s.controller_must_not_restart:
            ra = rec["restart_count"].get(s.controller_must_not_restart, {}).get("controller_restart_actions", 0)
            rec["controller_restart_check"] = {"workload": s.controller_must_not_restart, "controller_restart_actions": ra, "pass": ra == 0}
        rec["extra"] = s.extra_checks(c, t)

        # controller + alert evidence slices
        ev = [e for e in read_jsonl(CONTROLLER_EVENTS) if e.get("ts_utc", "") >= t["injected_utc"]]
        al = [a for a in read_jsonl(ALERTS_FILE) if a.get("received_utc", "") >= t["injected_utc"]]
        for name, data in (("controller_events.jsonl", ev), ("alerts.jsonl", al)):
            for d in data:
                append_jsonl(tdir / name, d)
        rec["controller_event_kinds"] = sorted({f"{e['kind']}:{e.get('action', '')}".rstrip(":") for e in ev})
        if s.id == "F01":
            die = [e for e in ev if e.get("kind") == "docker_event" and e.get("action") == "die"]
            if die and die[0].get("event_time_utc"):
                rec["crash_event_capture_lag_s"] = delta_s(die[0]["ts_utc"], die[0]["event_time_utc"])
                t["crash_event_logged_utc"] = die[0]["ts_utc"]

        capture_panels(c, tdir, t)
        rec["recovered"] = bool(t.get("validation_complete_utc") and (rec.get("validation") or {}).get("passed"))
        rec["symptom_observed"] = bool(t.get("first_observation_utc"))
        if s.symptom_required and not rec["symptom_observed"]:
            rec["final_status"] = "NO_SYMPTOM_OBSERVED"  # injection had no visible effect: not a valid drill
            rec["errors"].append("no symptom observed after injection; check the injection step")
        elif getattr(s, "skip_recovery", False):
            rec["final_status"] = "FAILURE_PHASE_ONLY"  # failure recorded; correction intentionally not performed
        else:
            rec["final_status"] = "RECOVERED_VALIDATED" if rec["recovered"] else ("RECOVERED_NOT_VALIDATED" if t.get("first_ready_utc") else "UNRECOVERED")
    except Exception as e:  # noqa: BLE001 - a broken drill must still clean up and be recorded
        rec["errors"].append(f"{type(e).__name__}: {e}")
        rec["final_status"] = "ERROR"
    finally:
        if observer:
            observer.stop_evt.set()
        if runner:
            runner.stop()
            try:
                runner.wait()
            except Exception:  # noqa: BLE001
                pass
        try:
            s.cleanup(c)
            ok, why = s.verify_cleanup(c)
            rec["cleanup"] = {"ok": ok, "detail": why}
        except Exception as e:  # noqa: BLE001
            rec["cleanup"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    return _finish(rec, t, s, tdir)


def _finish(rec: dict, t: dict, s: Scenario, tdir: Path) -> dict:
    rec["times"] = t
    if "injected_utc" in t:
        rec["metrics"] = derive_metrics(t, s.operator_corrected)
        m = rec["metrics"]
        if s.held_fault:
            rec["ac06_alert_within_45s"] = (m["fault_to_notification_s"] is not None and m["fault_to_notification_s"] <= 45.0)
        if s.id == "F01":
            lag = rec.get("crash_event_capture_lag_s")
            rec["ac06_crash_logged_within_5s"] = lag is not None and lag <= 5.0
        rr = m["correction_to_recovery_s"] if s.operator_corrected else m["fault_to_recovery_s"]
        rec["ac07_recovery_s"] = rr
        rec["ac07_within_60s"] = rr is not None and rr <= 60.0
    write_json(tdir / "timeline.json", rec)
    append_jsonl(INCIDENTS, rec)
    return rec


def capture_panels(c: Ctx, tdir: Path, t: dict) -> None:
    """Save the Prometheus data behind the incident dashboard for the incident window, plus
    Grafana links for that exact window (screenshots can be taken from them; Grafana image
    rendering needs a renderer plugin that is not part of the default stack)."""
    start = parse_iso(t["injected_utc"]).timestamp() - 30
    end = time.time() + 5
    queries = {
        "inference_ready": "inference_ready", "alerts": 'ALERTS{alertname!="Watchdog"}',
        "restarts": "lab_workload_restarts_total", "controller_state": "lab_controller_state == 1",
        "queue_depth": "inference_queue_depth",
        "p95_s": 'histogram_quantile(0.95, sum by (le, backend) (rate(inference_request_duration_seconds_bucket{status="success"}[30s])))',
        "gpu_util": "DCGM_FI_DEV_GPU_UTIL", "up": "up",
    }
    data = {}
    for k, q in queries.items():
        try:
            r = requests.get(f"{c.prom_url}/api/v1/query_range", params={"query": q, "start": start, "end": end, "step": 2}, timeout=10)
            data[k] = r.json().get("data", {}).get("result", [])
        except Exception as e:  # noqa: BLE001
            data[k] = {"unavailable": f"{type(e).__name__}: {e}"}
    write_json(tdir / "panel_data.json", {"window": {"start": epoch_to_iso(start), "end": epoch_to_iso(end)}, "series": data})
    ms = lambda x: int(x * 1000)  # noqa: E731
    write_json(tdir / "grafana_links.json", {d: f"http://127.0.0.1:{endpoints.port('grafana')}/d/{d}?from={ms(start)}&to={ms(end)}" for d in ("lab-readiness", "lab-performance", "lab-incidents")})


# ----------------------------------------------------------------------------- CLI
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--first-trial", type=int, default=1, help="number of the first trial (use 4 to add a re-run after trials 1-3)")
    ap.add_argument("--profile", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--baseline-s", type=float, default=20.0)
    ap.add_argument("--delay-ms", type=int, default=None)
    ap.add_argument("--failure-phase-only", action="store_true", help="F05 on a host without a GPU: record the failure, skip the GPU correction")
    ap.add_argument("--exporter-container", default=None, help="exporter unit to stop in F02 (default: the runtime's GPU exporter)")
    ap.add_argument("--exporter-service", default="dcgm-exporter")


def ctx_from_args(a: argparse.Namespace) -> Ctx:
    kw = {"exporter_container": a.exporter_container} if a.exporter_container else {}
    return Ctx(profile=a.profile, batch_size=a.batch_size, concurrency=a.concurrency, baseline_s=a.baseline_s, delay_ms=a.delay_ms,
               failure_phase_only=a.failure_phase_only, exporter_compose_service=a.exporter_service, **kw)


def run_scenarios(scenarios: list[Scenario], trials: int, c: Ctx, first_trial: int = 1) -> int:
    """first_trial lets a single drill be re-run later as trial N+1 without colliding with earlier records."""
    bad = 0
    for s in scenarios:
        for k in range(first_trial, first_trial + trials):
            print(f"== {s.id} {s.name} trial {k} ({k - first_trial + 1}/{trials})")
            r = run_trial(s, k, c)
            m = r.get("metrics", {})
            print(f"   {r['final_status']}  detect={m.get('fault_to_first_observation_s')}s notify={m.get('fault_to_notification_s')}s "
                  f"recover={r.get('ac07_recovery_s')}s cleanup_ok={r.get('cleanup', {}).get('ok')} {r.get('errors') or ''}")
            if r["final_status"] != "RECOVERED_VALIDATED" or not r.get("cleanup", {}).get("ok"):
                bad += 1
            time.sleep(10)  # settle between trials so alerts resolve
    return 1 if bad else 0


def scenario_main(factory: Callable[[], Scenario]) -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    a = ap.parse_args()
    return run_scenarios([factory()], a.trials, ctx_from_args(a), a.first_trial)
