"""F05 GPU inaccessible.

precondition : baseline workload ready; (full run) the host has a working GPU to restore access to
inject       : start a SEPARATE fault-case container requesting backend=cuda WITHOUT a GPU device mapping
expected     : explicit CUDA check fails: /readyz 503 reason_code=cuda_unavailable; /infer is rejected and is
               NEVER served on CPU (no silent fallback); controller marks needs_operator, no restart loop;
               failure evidence retained
stop         : operator correction: restore the device mapping (faults/overrides/gpu-device.yml) and redeploy.
               No driver or hardware repair is attempted.
cleanup      : remove the fault-case container and its Prometheus target

`--failure-phase-only` records the failure on a host without a GPU and skips the correction; such a
record is labelled FAILURE_PHASE_ONLY and does not count as a recovered drill.
"""
from __future__ import annotations

import shutil
import subprocess
import time

import requests

from faults.fault_case import fault_case_down, fault_case_up
from faults.harness import Ctx, Scenario, http_ready, scenario_main

VALID = "/opt/lab/manifests/weights/resnet18-f37072fd.pth"


class F05GpuInaccessible(Scenario):
    id = "F05"
    name = "GPU inaccessible"
    expected_symptoms = "readyz 503 cuda_unavailable; no CPU fallback; operator_required; no restart loop"
    expected_alert = "ServiceUnavailable"
    alert_instance_pred = staticmethod(lambda inst: inst.endswith(":8010"))
    held_fault = True
    hold_s = 5.0
    operator_corrected = True
    recovery_owner = "operator (harness): restore device mapping + redeploy; no driver/hardware repair"
    recovery_url_is_fault_case = True
    fault_backend_after = "cuda"
    controller_must_not_restart = "inference-fault"
    fault_class = "actual"
    runbook = "docs/runbooks/RB-003-gpu-inaccessible.md"
    skip_recovery = False

    def precondition(self, c: Ctx):
        ok, why = super().precondition(c)
        if not ok:
            return ok, why
        if not c.failure_phase_only and (shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0):
            return False, "no NVIDIA GPU on this host: the correction step cannot be validated (use --failure-phase-only to record the failure only)"
        self.skip_recovery = c.failure_phase_only
        return True, ""

    def inject(self, c: Ctx) -> dict:
        r = fault_case_up(backend="cuda", model_path=VALID, with_gpu=False)
        return {"cmd": "compose up inference-fault (backend=cuda, no GPU device mapping)", "rc": r.returncode, "stderr": r.stderr.strip()[-300:]}

    def symptom_observed(self, c: Ctx) -> bool:
        ok, info = http_ready(c.fault_url, timeout=0.5)
        return (not ok) and info.get("reason_code") in ("cuda_unavailable", "cuda_check_failed")

    def row_is_symptom(self, row: dict) -> bool:
        return False

    def stop(self, c: Ctx) -> dict:
        try:
            r = requests.post(f"{c.fault_url}/infer", json={"ids": ["img_000"]}, timeout=3)
            body = r.json()
            infer = {"status": r.status_code, "backend_reported": body.get("backend"), "body": body}
        except Exception as e:  # noqa: BLE001
            infer = {"error": str(e)}
        _, info = http_ready(c.fault_url)
        self.last_symptom = {"readyz": info, "infer_while_broken": infer}
        if c.failure_phase_only:
            return {"cmd": "correction skipped (--failure-phase-only)", **self.last_symptom}
        r = fault_case_up(backend="cuda", model_path=VALID, with_gpu=True)
        return {"cmd": "compose -f gpu-device.yml up --force-recreate inference-fault", "rc": r.returncode, **self.last_symptom}

    def extra_checks(self, c: Ctx, t: dict) -> dict:
        s = getattr(self, "last_symptom", {})
        inf = s.get("infer_while_broken", {})
        return {
            "reason_code_seen": s.get("readyz", {}).get("reason_code"),
            "infer_status_while_broken": inf.get("status"),
            "silent_cpu_fallback": inf.get("status") == 200,  # must be False
            "no_silent_fallback_pass": inf.get("status") not in (200, None),
        }

    def cleanup(self, c: Ctx) -> None:
        fault_case_down()
        time.sleep(1)

    def verify_cleanup(self, c: Ctx):
        gone = not http_ready(c.fault_url, timeout=1)[1].get("status")
        ok, _ = http_ready(c.url)
        return (gone and ok), "" if (gone and ok) else f"fault case still answering={not gone}; baseline ready={ok}"


def build() -> Scenario:
    return F05GpuInaccessible()


if __name__ == "__main__":
    raise SystemExit(scenario_main(build))
