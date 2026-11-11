"""F03 Missing weights.

precondition : baseline workload ready (it keeps serving the healthy baseline load)
inject       : start a SEPARATE fault-case container (inference-fault) with an invalid model path
expected     : /readyz returns 503 with reason_code=weights_missing; /infer is rejected (503); the controller
               marks it needs_operator and does NOT restart it; ServiceUnavailable fires and is received
stop         : operator correction: redeploy the fault case with the valid model path (controlled redeploy)
cleanup      : remove the fault-case container and its Prometheus target
"""
from __future__ import annotations

import time

import requests

from faults.fault_case import fault_case_down, fault_case_up
from faults.harness import Ctx, Scenario, http_ready, scenario_main


class F03MissingWeights(Scenario):
    id = "F03"
    name = "missing weights"
    expected_symptoms = "readyz 503 weights_missing; inference rejected; operator_required; no restart loop"
    expected_alert = "ServiceUnavailable"
    alert_instance_pred = staticmethod(lambda inst: inst.endswith(":8010"))
    held_fault = True
    hold_s = 5.0
    operator_corrected = True
    recovery_owner = "operator (harness): corrected configuration + controlled redeploy"
    recovery_url_is_fault_case = True
    fault_backend_after = "cpu"
    controller_must_not_restart = "inference-fault"
    fault_class = "actual"
    runbook = "docs/runbooks/RB-002-missing-weights.md"

    def inject(self, c: Ctx) -> dict:
        r = fault_case_up(backend="cpu", model_path="/opt/lab/manifests/weights/DOES-NOT-EXIST.pth")
        return {"cmd": "compose up inference-fault (FAULT_MODEL_PATH invalid)", "rc": r.returncode, "stderr": r.stderr.strip()[-300:]}

    def symptom_observed(self, c: Ctx) -> bool:
        ok, info = http_ready(c.fault_url, timeout=0.5)
        return (not ok) and info.get("reason_code") == "weights_missing"

    def row_is_symptom(self, row: dict) -> bool:
        return False  # baseline load targets the healthy service; the symptom is on the fault case

    def stop(self, c: Ctx) -> dict:
        # evidence before correcting: inference must be rejected, with a specific reason
        try:
            r = requests.post(f"{c.fault_url}/infer", json={"ids": ["img_000"]}, timeout=3)
            rejected = {"status": r.status_code, "body": r.json()}
        except Exception as e:  # noqa: BLE001
            rejected = {"error": str(e)}
        _, info = http_ready(c.fault_url)
        self.last_symptom = {"readyz": info, "infer_while_broken": rejected}
        r = fault_case_up(backend="cpu", model_path="/opt/lab/manifests/weights/resnet18-f37072fd.pth")
        return {"cmd": "compose up --force-recreate inference-fault (valid path)", "rc": r.returncode, **self.last_symptom}

    def extra_checks(self, c: Ctx, t: dict) -> dict:
        s = getattr(self, "last_symptom", {})
        inf = s.get("infer_while_broken", {})
        return {"reason_code_seen": s.get("readyz", {}).get("reason_code"), "infer_rejected": inf.get("status") == 503}

    def cleanup(self, c: Ctx) -> None:
        fault_case_down()
        time.sleep(1)

    def verify_cleanup(self, c: Ctx):
        gone = not http_ready(c.fault_url, timeout=1)[1].get("status")
        ok, info = http_ready(c.url)
        return (gone and ok), "" if (gone and ok) else f"fault case still answering={not gone}; baseline ready={ok}"


def build() -> Scenario:
    return F03MissingWeights()


if __name__ == "__main__":
    raise SystemExit(scenario_main(build))
