"""User-space process supervisor: the Docker-free stand-in for the Docker daemon.

Runs inside the Slurm job (as your own user, no privileges). It starts the services listed in
`<native dir>/run/services.json`, writes their state and Docker-shaped lifecycle events, and applies
the Docker-equivalent restart policy so the recovery ownership split stays the same as in the
Compose deployment:

    restart "on-failure:N"  restart after a non-zero exit/signal, at most N consecutive times
                            (the count resets if the process ran >= 10 s, like Docker)
    restart "always"        monitoring services
    restart "no"            never

Commands arrive as files in control/ (see ops/native_runtime.py): start, stop, restart, recreate, shutdown.
A *stop* is manual and is never auto-restarted (matches `docker stop`). A *restart* by the host
controller does not consume the on-failure budget.

    python -m sol.supervisor --native-dir artifacts/native
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from common.util import epoch_to_iso, iso
from ops.native_runtime import NativeRuntime

STABLE_UPTIME_S = 10.0  # Docker resets the on-failure counter after 10 s of uptime
STOP_GRACE_S = 10.0


def parse_policy(p: str) -> tuple[str, int]:
    if p.startswith("on-failure"):
        return "on-failure", int(p.split(":")[1]) if ":" in p else 0
    return p, 0


@dataclass
class Managed:
    name: str
    cmd: list[str]
    env: dict
    restart: str = "no"
    cwd: str = "."
    autostart: bool = True
    proc: subprocess.Popen | None = None
    desired: str = "stopped"  # running | stopped
    started_mono: float = 0.0
    started_epoch: float = 0.0
    exit_code: int | None = None
    fail_count: int = 0
    next_start: float | None = None
    gave_up: bool = False
    last_cmd_id: str | None = None
    overrides: dict = field(default_factory=dict)  # env overrides applied by `recreate`/`start`

    @property
    def restarting(self) -> bool:
        return self.proc is None and self.desired == "running" and self.next_start is not None and not self.gave_up


class Supervisor:
    def __init__(self, base: Path, poll_s: float = 0.2):
        self.base = Path(base)
        self.rt = NativeRuntime(self.base)
        self.poll_s = poll_s
        self.log_dir = self.base / "logs"
        for d in (self.rt.state_dir, self.rt.control_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        spec = json.loads((self.base / "run/services.json").read_text())
        self.cwd = spec.get("cwd", ".")
        self.services: dict[str, Managed] = {
            n: Managed(name=n, cmd=s["cmd"], env=s.get("env", {}), restart=s.get("restart", "no"),
                       cwd=s.get("cwd", self.cwd), autostart=s.get("autostart", True))
            for n, s in spec["services"].items()
        }
        self._shutdown = False

    # ------------------------------------------------------------------ state
    def write_state(self, m: Managed) -> None:
        running = m.proc is not None and m.proc.poll() is None
        body = {
            "name": m.name, "running": running, "pid": m.proc.pid if running else None,
            "started_epoch": m.started_epoch or None, "started_at": epoch_to_iso(m.started_epoch) if m.started_epoch else None,
            "exit_code": m.exit_code, "restarting": m.restarting, "desired": m.desired,
            "gave_up": m.gave_up, "fail_count": m.fail_count, "last_cmd_id": m.last_cmd_id, "updated": iso(),
        }
        tmp = self.rt.state_dir / f".{m.name}.tmp"
        tmp.write_text(json.dumps(body))
        os.replace(tmp, self.rt.state_dir / f"{m.name}.json")

    # ------------------------------------------------------------------ process control
    def _env(self, m: Managed) -> dict:
        env = {**os.environ, **{k: str(v) for k, v in m.env.items()}}
        for k, v in m.overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)
        return env

    def _start(self, m: Managed) -> None:
        logf = open(self.log_dir / f"{m.name}.log", "ab")
        try:
            m.proc = subprocess.Popen(m.cmd, cwd=m.cwd, env=self._env(m), stdout=logf, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        except OSError as e:
            logf.write(f"[supervisor] failed to start: {e}\n".encode())
            m.proc, m.exit_code = None, 127
            self.rt.emit("die", m.name, exitCode=127)
            self._after_exit(m, 127, 0.0)
            return
        finally:
            logf.close()
        m.started_mono, m.started_epoch = time.monotonic(), time.time()
        m.next_start = None
        self.rt.emit("start", m.name)
        self.write_state(m)

    def _signal_group(self, m: Managed, sig: int) -> None:
        if m.proc and m.proc.poll() is None:
            try:
                os.killpg(m.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def _stop(self, m: Managed, grace: float = STOP_GRACE_S) -> None:
        if not (m.proc and m.proc.poll() is None):
            return
        self.rt.emit("kill", m.name, signal=int(signal.SIGTERM))
        self._signal_group(m, signal.SIGTERM)
        end = time.monotonic() + grace
        while time.monotonic() < end and m.proc.poll() is None:
            time.sleep(0.05)
        if m.proc.poll() is None:
            self._signal_group(m, signal.SIGKILL)
            m.proc.wait()
        self._reap(m, manual=True)

    def _reap(self, m: Managed, manual: bool = False) -> None:
        rc = m.proc.returncode
        uptime = time.monotonic() - m.started_mono
        code = 128 + (-rc) if rc < 0 else rc  # Docker reports 128+signal
        m.proc, m.exit_code = None, code
        self.rt.emit("die", m.name, exitCode=code, signal=(-rc if rc < 0 else None))
        if manual:
            self.rt.emit("stop", m.name)
            self.write_state(m)
        else:
            self._after_exit(m, code, uptime)

    def _after_exit(self, m: Managed, code: int, uptime: float) -> None:
        kind, limit = parse_policy(m.restart)
        if m.desired != "running":
            self.write_state(m)
            return
        delay = None
        if kind == "always":
            delay = 1.0
        elif kind == "on-failure" and code != 0:
            if uptime >= STABLE_UPTIME_S:
                m.fail_count = 0
            m.fail_count += 1
            if limit and m.fail_count > limit:
                m.gave_up = True  # retries exhausted, exactly like Docker: stays down until an operator acts
            else:
                delay = min(0.5 * 2 ** (m.fail_count - 1), 5.0)
        if delay is not None:
            m.next_start = time.monotonic() + delay
        self.write_state(m)

    # ------------------------------------------------------------------ commands
    def _handle(self, c: dict) -> None:
        name, cmd = c.get("name", ""), c["cmd"]
        if cmd == "shutdown":
            self._shutdown = True
            return
        m = self.services.get(name)
        if m is None:
            return
        if cmd == "stop":
            m.desired = "stopped"
            self._stop(m)
        elif cmd == "start":
            m.desired, m.gave_up, m.fail_count = "running", False, 0
            if c.get("env"):
                m.overrides = c["env"]
            if not (m.proc and m.proc.poll() is None):
                self._start(m)
        elif cmd in ("restart", "recreate"):
            m.desired, m.gave_up = "running", False
            if cmd == "recreate":
                m.fail_count = 0
                m.overrides = c.get("env") or {}
            self._stop(m)
            self._start(m)
        m.last_cmd_id = c["id"]
        self.write_state(m)

    def _poll_control(self) -> None:
        for f in sorted(self.rt.control_dir.glob("*.json")):
            try:
                c = json.loads(f.read_text())
            except (ValueError, OSError):
                f.unlink(missing_ok=True)
                continue
            f.unlink(missing_ok=True)
            self._handle(c)

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_shutdown", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_shutdown", True))
        for m in self.services.values():
            m.desired = "running" if m.autostart else "stopped"
            self.write_state(m)
        for m in self.services.values():
            if m.autostart:
                self._start(m)
        while not self._shutdown:
            self._poll_control()
            now = time.monotonic()
            for m in self.services.values():
                if m.proc is not None and m.proc.poll() is not None:
                    self._reap(m)
                elif m.proc is None and m.desired == "running" and not m.gave_up and m.next_start is not None and now >= m.next_start:
                    self._start(m)
            time.sleep(self.poll_s)
        for m in reversed(list(self.services.values())):
            m.desired = "stopped"
            self._stop(m, grace=5.0)
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    return Supervisor(a.native_dir).run()


if __name__ == "__main__":
    sys.exit(main())
