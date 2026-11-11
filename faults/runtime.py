"""Runtime backends for the fault harness: Docker (Compose deployment) or native processes (Sol).

Scenarios only call this interface, so F01-F05 are identical in both modes:

    workload(profile)   name of the workload unit        exporter()   name of the GPU exporter unit
    inspect(name)       {"Running", "StartedAt", ...}    kill9(name)  SIGKILL the unit's main process
    stop/start(name)    manual stop / start              restart(name)
    fault_case_up(backend, model_path, with_gpu) / fault_case_down()   the separate F03/F05 case
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

from common import endpoints
from common.util import REPO_ROOT


@dataclass
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


# ---------------------------------------------------------------------------- docker helpers
def compose(args: list[str], extra_files: list[str] | None = None, env: dict | None = None, timeout: float = 300) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", "deploy/docker-compose.yml"]
    for f in extra_files or []:
        cmd += ["-f", f]
    cmd += ["--env-file", "deploy/versions.env"]
    for opt in (".env", "deploy/image-digests.env"):
        if (REPO_ROOT / opt).exists():
            cmd += ["--env-file", opt]
    e = {**os.environ, "LAB_UID": str(os.getuid()), "LAB_GID": str(os.getgid()), **(env or {})}
    return subprocess.run(cmd + args, cwd=REPO_ROOT, env=e, capture_output=True, text=True, timeout=timeout)


def docker(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _set_targets(mode: str) -> None:
    subprocess.run([sys.executable, str(REPO_ROOT / "deploy/set_targets.py"), mode], cwd=REPO_ROOT, check=False)


GPU_OVERRIDE = "faults/overrides/gpu-device.yml"


class DockerBackend:
    mode = "docker"

    def workload(self, profile: str) -> str:
        return f"lab-inference-{profile}"

    def exporter(self) -> str:
        return "lab-dcgm-exporter"

    def inspect(self, name: str) -> dict:
        r = docker(["inspect", "--format", "{{json .State}}", name], 10)
        if r.returncode != 0:
            return {}
        try:
            return json.loads(r.stdout)
        except ValueError:
            return {}

    def kill9(self, name: str) -> Result:
        r = docker(["kill", "-s", "KILL", name])
        return Result(r.returncode, r.stdout, r.stderr)

    def stop(self, name: str) -> Result:
        r = docker(["stop", name], 60)
        return Result(r.returncode, r.stdout, r.stderr)

    def start(self, name: str) -> Result:
        r = docker(["start", name], 60)
        return Result(r.returncode, r.stdout, r.stderr)

    def fault_case_up(self, backend: str, model_path: str, with_gpu: bool = False) -> Result:
        (REPO_ROOT / "artifacts/faults/inject/fault").mkdir(parents=True, exist_ok=True)
        _set_targets("add-fault")
        r = compose(["--profile", "fault", "up", "-d", "--force-recreate", "--no-build", "inference-fault"],
                    extra_files=[GPU_OVERRIDE] if with_gpu else None,
                    env={"FAULT_BACKEND": backend, "FAULT_MODEL_PATH": model_path})
        return Result(r.returncode, r.stdout, r.stderr)

    def fault_case_down(self) -> Result:
        r = compose(["--profile", "fault", "rm", "-sf", "inference-fault"])
        _set_targets("remove-fault")
        return Result(r.returncode, r.stdout, r.stderr)


class NativeBackend:
    """Sol / any host without Docker: units are supervised processes (sol/supervisor.py)."""

    mode = "native"

    def __init__(self):
        from ops.native_runtime import NativeRuntime

        self.rt = NativeRuntime()

    def workload(self, profile: str) -> str:
        return f"inference-{profile}"

    def exporter(self) -> str:
        return "gpu-exporter"

    def inspect(self, name: str) -> dict:
        s = self.rt.inspect_state(name) or {}
        if s:
            s["StartedAt"] = s.get("StartedAt")
        return s

    def kill9(self, name: str) -> Result:
        r = self.rt.kill9(name)
        return Result(0 if r.get("ok") else 1, json.dumps(r), r.get("reason", ""))

    def stop(self, name: str) -> Result:
        return Result(0 if self.rt.stop(name) else 1)

    def start(self, name: str) -> Result:
        return Result(0 if self.rt.start(name) else 1)

    @staticmethod
    def _local(path: str) -> str:
        return path.replace("/opt/lab/manifests", str(REPO_ROOT / "manifests"))

    def fault_case_up(self, backend: str, model_path: str, with_gpu: bool = False) -> Result:
        (REPO_ROOT / "artifacts/faults/inject/fault").mkdir(parents=True, exist_ok=True)
        _set_targets("add-fault")
        env = {"LAB_BACKEND": backend, "LAB_MODEL_PATH": self._local(model_path)}
        if backend == "cuda" and not with_gpu:
            # Native analogue of "no GPU device mapping": the process is not allowed to see any device.
            env["CUDA_VISIBLE_DEVICES"] = ""
        # with_gpu: the key is omitted, so the process inherits the job's own GPU allocation.
        ok = self.rt.recreate("inference-fault", env=env)
        return Result(0 if ok else 1, "", "" if ok else "supervisor did not start inference-fault")

    def fault_case_down(self) -> Result:
        ok = self.rt.remove("inference-fault")
        _set_targets("remove-fault")
        return Result(0 if ok else 1)


def get():
    return NativeBackend() if endpoints.native() else DockerBackend()
