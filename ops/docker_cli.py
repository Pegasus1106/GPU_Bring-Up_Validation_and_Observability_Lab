"""Thin wrapper over the docker CLI. The controller runs on the host and talks to
the local Docker socket only; no Docker API is exposed on the network."""
from __future__ import annotations

import json
import subprocess
from typing import Iterator


class DockerCLI:
    def __init__(self, binary: str = "docker"):
        self.bin = binary

    def _run(self, args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run([self.bin, *args], capture_output=True, text=True, timeout=timeout)

    def inspect_state(self, container: str) -> dict | None:
        """Container state dict, or None if the container does not exist."""
        try:
            r = self._run(["inspect", "--format", "{{json .State}}", container], timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if r.returncode != 0:
            return None
        try:
            return json.loads(r.stdout)
        except ValueError:
            return None

    def restart(self, container: str, stop_timeout_s: int = 10) -> tuple[bool, str]:
        try:
            r = self._run(["restart", "-t", str(stop_timeout_s), container], timeout=stop_timeout_s + 30)
        except subprocess.TimeoutExpired:
            return False, "docker restart timed out"
        return r.returncode == 0, (r.stdout + r.stderr).strip()

    def events(self, containers: list[str]) -> subprocess.Popen:
        args = [self.bin, "events", "--format", "{{json .}}", "--filter", "type=container"]
        for c in containers:
            args += ["--filter", f"container={c}"]
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)


def iter_events(proc: subprocess.Popen) -> Iterator[dict]:
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue
