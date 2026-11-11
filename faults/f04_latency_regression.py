"""F04 Latency regression (lab-only injected delay inside the model worker).

precondition : baseline workload ready AND a frozen matching baseline exists for (backend, batch size)
               (`make freeze-baseline`); otherwise LatencyRegression has nothing to compare against
inject       : write a per-job delay (ms) to artifacts/faults/inject/<profile>/delay_ms. The worker honours it
               only when LAB_ENABLE_FAULT_INJECTION=1, caps it at 5 s, and exports it as
               inference_fault_injection_delay_seconds. It is a real delay, labelled as injected.
expected     : latency and queue depth rise; pending work stays bounded (<= capacity); LatencyRegression fires
               and is received; the controller does NOT restart (service stays ready)
stop         : remove the delay file (the injected cause)
cleanup      : delay file removed; metric back to 0
"""
from __future__ import annotations

import math
import time

import requests

from common.util import REPO_ROOT, read_json
from faults.harness import Ctx, Scenario, http_ready, scenario_main

BASELINE_JSON = REPO_ROOT / "monitoring/rules/baseline.json"


def baseline_p95_ms(backend_label: str, batch_size: int) -> float | None:
    if not BASELINE_JSON.exists():
        return None
    for cid, sec in read_json(BASELINE_JSON).get("p95_seconds", {}).items():
        b, bs, _ = cid.split("-")
        if {"cpu": "cpu", "gpu": "cuda"}[b] == backend_label and int(bs[1:]) == batch_size:
            return sec * 1000.0
    return None


class F04LatencyRegression(Scenario):
    id = "F04"
    name = "latency regression"
    fault_class = "injected_latency"
    expected_symptoms = "p95 > 2x frozen baseline; queue depth up but bounded; no restart"
    expected_alert = "LatencyRegression"
    held_fault = False
    hold_s = 10.0
    operator_corrected = True
    recovery_owner = "fault harness removes the injected cause; controller restarts only if unhealthy (it is not)"
    runbook = "docs/runbooks/RB-005-latency-regression.md"
    controller_must_not_restart = None

    def precondition(self, c: Ctx):
        ok, why = super().precondition(c)
        if not ok:
            return ok, why
        b = baseline_p95_ms(c.backend, c.batch_size)
        if b is None:
            return False, f"no frozen baseline for {c.backend}/batch {c.batch_size}: run `make freeze-baseline` after the benchmark"
        self.base_ms = b
        # delay sized so closed-loop p95 clearly exceeds 2x baseline while still giving >=100 completions/min
        self.delay_ms = c.delay_ms or int(min(500, max(100, math.ceil(b / 2))))
        self.controller_must_not_restart = c.service
        return True, ""

    def inject(self, c: Ctx) -> dict:
        c.inject_dir.mkdir(parents=True, exist_ok=True)
        (c.inject_dir / "delay_ms").write_text(str(self.delay_ms))
        return {"delay_ms": self.delay_ms, "baseline_p95_ms": self.base_ms, "file": str(c.inject_dir / "delay_ms")}

    def symptom_observed(self, c: Ctx) -> bool:
        try:
            t = requests.get(f"{c.url}/metrics", timeout=1.5).text
        except requests.RequestException:
            return False
        for line in t.splitlines():
            if line.startswith("inference_fault_injection_delay_seconds"):
                return float(line.rsplit(" ", 1)[1]) > 0
        return False

    def row_is_symptom(self, row: dict) -> bool:
        return row["outcome"] != "success" or float(row["e2e_ms"]) > 2 * self.base_ms

    def stop(self, c: Ctx) -> dict:
        (c.inject_dir / "delay_ms").unlink(missing_ok=True)
        return {"removed": str(c.inject_dir / "delay_ms")}

    def cleanup(self, c: Ctx) -> None:
        (c.inject_dir / "delay_ms").unlink(missing_ok=True)
        time.sleep(1)

    def verify_cleanup(self, c: Ctx):
        ok, _ = http_ready(c.url)
        gone = not (c.inject_dir / "delay_ms").exists() and not self.symptom_observed(c)
        return (ok and gone), "" if (ok and gone) else f"ready={ok} delay_cleared={gone}"


def build() -> Scenario:
    return F04LatencyRegression()


if __name__ == "__main__":
    raise SystemExit(scenario_main(build))
