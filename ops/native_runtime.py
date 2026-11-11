"""Process runtime used instead of Docker (ASU Sol: no Docker daemon).

`sol/supervisor.py` owns the processes (the analogue of the Docker daemon and its
`restart: on-failure:3` policy). Everything else talks to it through plain files under the
native directory, so there is no network control surface:

    state/<name>.json      current state, written by the supervisor
    events.jsonl           Docker-event-shaped lifecycle events (start, die, kill, stop, ...)
    control/<id>.json      commands: start | stop | restart | recreate | shutdown

This class exposes the same interface as `ops.docker_cli.DockerCLI` (inspect_state, restart,
events) so the host controller is unchanged, plus the extra verbs the fault harness needs.
"""
from __future__ import annotations

import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Iterator

from common import endpoints
from common.util import epoch_to_iso


def pid_alive(pid: int | None) -> bool:
    """True if the process exists and is not a zombie."""
    if not pid:
        return False
    try:  # if it is our own child and has exited, reap it (a zombie still answers kill(pid, 0))
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass  # not our child: fall through
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # Linux: a zombie that is someone else's child
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


class TailProc:
    """Minimal stand-in for the Popen returned by `docker events`: `.stdout` yields lines
    appended to events.jsonl after creation; `.terminate()` stops the tail."""

    def __init__(self, path: Path, poll_s: float = 0.1):
        self.path, self.poll_s = path, poll_s
        self._stop = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._offset = self.path.stat().st_size  # only events from now on, like `docker events`
        self.stdout = self._lines()

    def _lines(self) -> Iterator[str]:
        buf = ""
        while not self._stop:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    f.seek(self._offset)
                    chunk = f.read()
                    self._offset = f.tell()
            except OSError:
                chunk = ""
            if chunk:
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield line
            else:
                time.sleep(self.poll_s)

    def terminate(self) -> None:
        self._stop = True


class NativeRuntime:
    def __init__(self, base: Path | None = None):
        self.base = Path(base) if base else endpoints.native_dir()
        self.state_dir = self.base / "state"
        self.control_dir = self.base / "control"
        self.events_file = self.base / "events.jsonl"

    # ------------------------------------------------------------------ inspection
    def read_state(self, name: str) -> dict | None:
        p = self.state_dir / f"{name}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (ValueError, OSError):
            return None

    def inspect_state(self, name: str) -> dict | None:
        """DockerCLI-compatible: {"Running", "Restarting", "StartedAt", "ExitCode"} or None."""
        s = self.read_state(name)
        if s is None:
            return None
        running = bool(s.get("running")) and pid_alive(s.get("pid"))
        return {
            "Running": running,
            "Restarting": bool(s.get("restarting")),
            "StartedAt": s.get("started_at"),
            "StartedEpoch": s.get("started_epoch"),
            "ExitCode": s.get("exit_code"),
            "Pid": s.get("pid"),
            "Desired": s.get("desired"),
        }

    # ------------------------------------------------------------------ commands
    def command(self, cmd: str, name: str = "", env: dict | None = None, wait_s: float = 0.0) -> str:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        cid = uuid.uuid4().hex[:12]
        body = {"id": cid, "cmd": cmd, "name": name, "env": env or {}, "issued": time.time()}
        tmp = self.control_dir / f".{cid}.tmp"
        tmp.write_text(json.dumps(body))
        os.replace(tmp, self.control_dir / f"{int(time.time()*1000)}-{cid}.json")
        if wait_s:
            end = time.time() + wait_s
            while time.time() < end:
                s = self.read_state(name) if name else None
                if s and s.get("last_cmd_id") == cid:
                    break
                time.sleep(0.1)
        return cid

    def _wait(self, pred, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.1)
        return False

    def restart(self, name: str, stop_timeout_s: int = 10) -> tuple[bool, str]:
        before = (self.read_state(name) or {}).get("started_epoch") or 0
        self.command("restart", name)
        ok = self._wait(lambda: ((self.read_state(name) or {}).get("started_epoch") or 0) > before
                        and (self.inspect_state(name) or {}).get("Running"), stop_timeout_s + 30)
        return ok, "restarted" if ok else "restart not confirmed by supervisor"

    def stop(self, name: str, timeout: float = 30.0) -> bool:
        self.command("stop", name)
        return self._wait(lambda: not (self.inspect_state(name) or {}).get("Running"), timeout)

    def start(self, name: str, env: dict | None = None, timeout: float = 30.0) -> bool:
        self.command("start", name, env=env)
        return self._wait(lambda: (self.inspect_state(name) or {}).get("Running"), timeout)

    def recreate(self, name: str, env: dict | None = None, timeout: float = 30.0) -> bool:
        """Stop (if running) and start again with environment overrides (None values unset a variable)."""
        before = (self.read_state(name) or {}).get("started_epoch") or 0
        self.command("recreate", name, env=env)
        return self._wait(lambda: ((self.read_state(name) or {}).get("started_epoch") or 0) > before
                          and (self.inspect_state(name) or {}).get("Running"), timeout)

    def remove(self, name: str, timeout: float = 30.0) -> bool:
        self.command("stop", name)
        return self._wait(lambda: not (self.inspect_state(name) or {}).get("Running"), timeout)

    def kill9(self, name: str) -> dict:
        """SIGKILL only the workload process (the F01 injection). The supervisor sees a
        signal exit and applies its on-failure policy, exactly like Docker would."""
        s = self.inspect_state(name) or {}
        pid = s.get("Pid")
        if not pid or not s.get("Running"):
            return {"ok": False, "reason": "process not running"}
        os.kill(pid, signal.SIGKILL)
        return {"ok": True, "pid": pid}

    # ------------------------------------------------------------------ events
    def events(self, names: list[str]) -> TailProc:
        return TailProc(self.events_file)

    def emit(self, action: str, name: str, **attrs) -> None:  # used by the supervisor
        evt = {
            "Type": "container", "Action": action, "time": int(time.time()), "timeNano": time.time_ns(),
            "Actor": {"Attributes": {"name": name, **{k: str(v) for k, v in attrs.items() if v is not None}}},
        }
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt) + "\n")

    @staticmethod
    def iso_from_epoch(ts: float | None) -> str | None:
        return epoch_to_iso(ts) if ts else None
