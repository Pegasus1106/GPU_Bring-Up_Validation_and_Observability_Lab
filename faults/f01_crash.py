"""F01 Workload crash.

precondition : baseline workload ready, host controller reachable
inject       : SIGKILL the lab workload's main process (`docker kill -s KILL`, or `kill -9` of the supervised
               process in native/Sol mode), nothing else
expected     : Docker `die` event logged by the controller within 5 s; failed requests; Docker's restart
               policy (on-failure:3) restarts the container; readiness is revalidated by a fixed-input check;
               RestartObserved alert from the persistent counter
stop         : none (Docker owns recovery; the controller must not also restart)
cleanup      : wait until the workload is ready and reset restart-policy retry state if exhausted
"""
from __future__ import annotations

import time

import requests

from faults import runtime
from faults.harness import Ctx, Scenario, container_inspect, http_ready, scenario_main


class F01Crash(Scenario):
    id = "F01"
    name = "workload crash"
    expected_symptoms = "die event, failed/connection-error requests, restart, readiness revalidated"
    expected_alert = "RestartObserved"
    recovery_owner = "docker restart policy (on-failure:3)"
    receipt_before_stop = False
    fault_class = "actual"
    runbook = "docs/runbooks/RB-001-workload-unresponsive.md"

    def precondition(self, c: Ctx):
        ok, why = super().precondition(c)
        if not ok:
            return ok, why
        try:
            requests.get(c.controller_metrics, timeout=2).raise_for_status()
        except requests.RequestException as e:
            return False, f"host controller metrics not reachable at {c.controller_metrics}: {e}"
        st = container_inspect(c.container)
        if not st.get("Running"):
            return False, f"container {c.container} not running"
        return True, ""

    def inject(self, c: Ctx) -> dict:
        rt = runtime.get()
        before = container_inspect(c.container)
        r = rt.kill9(c.container)
        what = f"docker kill -s KILL {c.container}" if rt.mode == "docker" else f"kill -9 (supervised process {c.container})"
        return {"cmd": what, "rc": r.returncode, "stderr": r.stderr.strip()[:200], "started_at_before": before.get("StartedAt")}

    def symptom_observed(self, c: Ctx) -> bool:
        ok, _ = http_ready(c.url, timeout=0.5)
        return not ok

    def cleanup(self, c: Ctx) -> None:
        end = time.time() + 120
        while time.time() < end:
            st = container_inspect(c.container)
            if st.get("Running") and http_ready(c.url)[0]:
                return
            if not st.get("Running") and not st.get("Restarting"):
                # restart policy retries exhausted: bring the container back explicitly (documented)
                runtime.get().start(c.container)
            time.sleep(2)


def build() -> Scenario:
    return F01Crash()


if __name__ == "__main__":
    raise SystemExit(scenario_main(build))
