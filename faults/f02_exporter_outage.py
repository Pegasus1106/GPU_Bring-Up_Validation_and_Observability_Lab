"""F02 Exporter outage.

precondition : workload ready; the GPU exporter is up and scraped (up{job="dcgm-exporter"} == 1)
inject       : stop the lab GPU exporter container (`docker stop`), held for >= 30 s
expected     : TelemetryUnavailable fires and is received locally within 45 s; GPU series disappear
               (unknown, NOT zero); inference keeps serving; controller performs NO workload restart
stop         : fault harness starts the exporter again (owner: fault harness)
cleanup      : exporter running and scraped again
"""
from __future__ import annotations

import time

from faults import runtime
from faults.harness import Ctx, Scenario, http_ready, prom_query, scenario_main


class F02ExporterOutage(Scenario):
    id = "F02"
    name = "exporter outage"
    expected_symptoms = "exporter target down; GPU health unknown (series absent); inference unaffected"
    expected_alert = "TelemetryUnavailable"
    held_fault = True
    hold_s = 32.0
    recovery_owner = "fault harness (restores exporter); controller must not restart inference"
    operator_corrected = True
    fault_class = "actual"
    runbook = "docs/runbooks/RB-004-telemetry-outage.md"
    controller_must_not_restart = None  # set per-profile in precondition

    def precondition(self, c: Ctx):
        ok, why = super().precondition(c)
        if not ok:
            return ok, why
        up = prom_query(c.prom_url, 'up{job="dcgm-exporter"}')
        if not up or up[0]["value"][1] != "1":
            return False, "GPU exporter is not up/scraped by Prometheus (is the gpu profile running?)"
        self.controller_must_not_restart = c.service
        return True, ""

    def inject(self, c: Ctx) -> dict:
        r = runtime.get().stop(c.exporter_container)
        return {"cmd": f"stop {c.exporter_container}", "rc": r.returncode, "stderr": r.stderr.strip()[:200]}

    def symptom_observed(self, c: Ctx) -> bool:
        up = prom_query(c.prom_url, 'up{job="dcgm-exporter"}')
        return bool(up) and up[0]["value"][1] == "0"

    def row_is_symptom(self, row: dict) -> bool:
        return False  # inference is expected to continue; the symptom is on the monitoring path

    def stop(self, c: Ctx) -> dict:
        r = runtime.get().start(c.exporter_container)
        return {"cmd": f"start {c.exporter_container}", "rc": r.returncode}

    def recovery_extra(self, c: Ctx) -> bool:
        up = prom_query(c.prom_url, 'up{job="dcgm-exporter"}')
        return bool(up) and up[0]["value"][1] == "1"

    def extra_checks(self, c: Ctx, t: dict) -> dict:
        return {"note": "GPU series must be absent (not zero) while the exporter is down: see panel_data.json 'gpu_util' gap"}

    def cleanup(self, c: Ctx) -> None:
        runtime.get().start(c.exporter_container)
        end = time.time() + 60
        while time.time() < end and not self.recovery_extra(c):
            time.sleep(1)

    def verify_cleanup(self, c: Ctx):
        ok = self.recovery_extra(c) and http_ready(c.url)[0]
        return ok, "" if ok else "exporter not scraped again or workload not ready"


def build() -> Scenario:
    return F02ExporterOutage()


if __name__ == "__main__":
    raise SystemExit(scenario_main(build))
