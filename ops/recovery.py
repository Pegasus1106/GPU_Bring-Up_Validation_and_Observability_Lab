"""Bounded recovery policy as a pure state machine (no I/O, fully unit-tested).

Ownership split (spec; avoids two recovery owners):
  * Docker's restart policy owns *exited* processes (restart: on-failure:3).
  * This policy owns a workload whose container is running but persistently
    not ready.
  * Configuration/hardware problems (missing weights, no GPU access, unsupported
    hardware) are never restarted: they move to NEEDS_OPERATOR with a runbook.

Restart rule: >= `failures_required` consecutive failed probes (default 3 at
2 s intervals), then at most 1 restart per `min_restart_interval_s` (60 s) and
at most `max_restarts_per_window` (3) per `window_s` (300 s). When the budget is
exhausted the state becomes UNRESOLVED (explicit, with a runbook reference).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from app.errors import is_config_reason

RUNBOOKS = {
    "weights_missing": "docs/runbooks/RB-002-missing-weights.md",
    "weights_checksum_mismatch": "docs/runbooks/RB-002-missing-weights.md",
    "weights_corrupt": "docs/runbooks/RB-002-missing-weights.md",
    "model_manifest_missing": "docs/runbooks/RB-002-missing-weights.md",
    "inputs_missing": "docs/runbooks/RB-002-missing-weights.md",
    "inputs_checksum_mismatch": "docs/runbooks/RB-002-missing-weights.md",
    "cuda_unavailable": "docs/runbooks/RB-003-gpu-inaccessible.md",
    "cuda_check_failed": "docs/runbooks/RB-003-gpu-inaccessible.md",
    "unsupported_hardware": "docs/runbooks/RB-003-gpu-inaccessible.md",
    "config_invalid": "docs/runbooks/RB-002-missing-weights.md",
    "unresolved": "docs/runbooks/RB-006-recovery-exhausted.md",
}


class State(str, Enum):
    HEALTHY = "healthy"
    STARTING = "starting"  # container running, within startup grace
    SUSPECT = "suspect"  # failing probes, below the restart threshold
    RECOVERING = "recovering"  # controller restarted it; waiting for readiness
    CONTAINER_DOWN = "container_down"  # not running: Docker's restart policy owns this
    NEEDS_OPERATOR = "needs_operator"  # config/hardware problem: no restarts
    UNRESOLVED = "unresolved"  # automatic recovery exhausted


@dataclass
class PolicyConfig:
    probe_interval_s: float = 2.0
    failures_required: int = 3
    min_restart_interval_s: float = 60.0
    max_restarts_per_window: int = 3
    window_s: float = 300.0
    startup_grace_s: float = 120.0  # cold healthy startup target (AC07)
    # Recovery target for a cached workload is 60 s. Keeping the post-restart grace at
    # 60 s (not 120 s) also keeps budget exhaustion reachable: three restarts fit in 300 s.
    post_restart_grace_s: float = 60.0


@dataclass
class Probe:
    container_running: bool
    live: bool
    ready: bool
    reason_code: str | None = None


@dataclass
class Decision:
    action: str  # none | restart | hold_operator | unresolved | recovered | cooldown
    state: State
    reason: str = ""
    runbook: str | None = None
    changed: bool = False  # state changed on this observation


@dataclass
class RecoveryPolicy:
    cfg: PolicyConfig = field(default_factory=PolicyConfig)
    state: State = State.STARTING
    consecutive_failures: int = 0
    grace_until: float | None = None
    last_restart: float | None = None
    restarts: deque = field(default_factory=deque)  # monotonic times of controller restarts
    _seen_running: bool = False

    # -- helpers ----------------------------------------------------------------
    def _set(self, new: State) -> bool:
        changed = new != self.state
        self.state = new
        return changed

    def restarts_in_window(self, now: float) -> int:
        while self.restarts and now - self.restarts[0] > self.cfg.window_s:
            self.restarts.popleft()
        return len(self.restarts)

    def restarted(self, now: float) -> None:
        """Caller invokes this after it performed the restart action."""
        self.last_restart = now
        self.restarts.append(now)
        self.consecutive_failures = 0
        self.grace_until = now + self.cfg.post_restart_grace_s
        self.state = State.RECOVERING

    # -- main entry point ------------------------------------------------------------
    def observe(self, now: float, p: Probe) -> Decision:
        cfg = self.cfg
        if not p.container_running:
            self._seen_running = False
            self.consecutive_failures = 0
            ch = self._set(State.CONTAINER_DOWN)
            return Decision("none", self.state, "workload not running; the runtime restart policy (Docker, or the Sol supervisor) owns this", changed=ch)

        if not self._seen_running:  # container (re)started: open a startup grace window
            self._seen_running = True
            if self.state not in (State.RECOVERING,):
                self.grace_until = now + cfg.startup_grace_s
                if self.state != State.HEALTHY:
                    self._set(State.STARTING)
            self.consecutive_failures = 0

        if p.ready:
            prev = self.state
            self.consecutive_failures = 0
            self.grace_until = None
            ch = self._set(State.HEALTHY)
            if ch and prev in (State.RECOVERING, State.UNRESOLVED, State.NEEDS_OPERATOR, State.SUSPECT):
                return Decision("recovered", self.state, f"ready again after {prev.value}", changed=True)
            return Decision("none", self.state, changed=ch)

        # ---- not ready ----
        rc = p.reason_code
        if p.live and is_config_reason(rc):
            ch = self._set(State.NEEDS_OPERATOR)
            self.consecutive_failures = 0
            return Decision(
                "hold_operator" if ch else "none", self.state,
                f"{rc}: configuration or hardware access must be corrected; restarts would not help",
                runbook=RUNBOOKS.get(rc or "", None), changed=ch,
            )

        in_grace = self.grace_until is not None and now < self.grace_until
        if in_grace and (rc in ("loading", None) or not p.live):
            # still booting (or Docker is mid-restart): do not count
            if self.state not in (State.RECOVERING, State.STARTING):
                self._set(State.STARTING)
            return Decision("none", self.state, "within startup grace")

        if self.state == State.UNRESOLVED:
            return Decision("none", self.state, "unresolved: waiting for operator", runbook=RUNBOOKS["unresolved"])

        self.consecutive_failures += 1
        ch = self._set(State.SUSPECT)
        if self.consecutive_failures < cfg.failures_required:
            return Decision("none", self.state, f"probe failure {self.consecutive_failures}/{cfg.failures_required}", changed=ch)

        # threshold reached: apply the restart budget
        if self.restarts_in_window(now) >= cfg.max_restarts_per_window:
            ch = self._set(State.UNRESOLVED)
            return Decision(
                "unresolved", self.state,
                f"{cfg.max_restarts_per_window} automatic restarts in {cfg.window_s:.0f}s did not restore readiness",
                runbook=RUNBOOKS["unresolved"], changed=ch,
            )
        if self.last_restart is not None and now - self.last_restart < cfg.min_restart_interval_s:
            wait = cfg.min_restart_interval_s - (now - self.last_restart)
            return Decision("cooldown", self.state, f"restart cooldown, {wait:.0f}s remaining", changed=ch)
        return Decision("restart", self.state, f"{self.consecutive_failures} consecutive failed probes ({rc or 'no response'})", changed=ch)
