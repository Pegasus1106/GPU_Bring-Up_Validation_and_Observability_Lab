"""Recovery policy / incident state transitions and limits (pure logic)."""
from ops.recovery import PolicyConfig, Probe, RecoveryPolicy, State

HEALTHY = Probe(True, True, True)
BROKEN = Probe(True, True, False, "worker_stuck")


def make(**kw):
    p = RecoveryPolicy(PolicyConfig(**kw))
    p.observe(0.0, HEALTHY)  # container seen running, becomes healthy
    return p


def drive(p, t0, n, probe, step=2.0):
    ds = []
    for k in range(n):
        ds.append(p.observe(t0 + k * step, probe))
    return ds


def test_healthy_service_never_acts():
    p = make()
    assert all(d.action == "none" for d in drive(p, 2.0, 20, HEALTHY))
    assert p.state == State.HEALTHY


def test_needs_three_consecutive_failures_then_restart():
    p = make()
    d = drive(p, 2.0, 3, BROKEN)
    assert [x.action for x in d] == ["none", "none", "restart"]


def test_two_failures_then_recovery_resets_counter():
    p = make()
    drive(p, 2.0, 2, BROKEN)
    p.observe(6.0, HEALTHY)
    assert p.consecutive_failures == 0
    assert [x.action for x in drive(p, 8.0, 2, BROKEN)] == ["none", "none"]


def test_cooldown_then_restart_after_60s():
    p = make()
    drive(p, 2.0, 3, BROKEN)
    p.restarted(8.0)
    p.grace_until = 9.0
    drive(p, 10.0, 3, BROKEN)  # failures accumulate, cooldown
    d = p.observe(67.0, BROKEN)  # 59 s after restart
    assert d.action == "cooldown"
    d = p.observe(68.5, BROKEN)  # >60 s
    assert d.action == "restart"


def test_three_restarts_per_five_minutes_then_unresolved():
    p = make()
    t = 2.0
    actions = []
    for _ in range(200):
        d = p.observe(t, BROKEN)
        actions.append(d.action)
        if d.action == "restart":
            p.restarted(t)
            p.grace_until = t  # no grace: isolate the budget rule
        t += 2.0
        if d.action == "unresolved":
            break
    assert actions.count("restart") == 3
    assert actions[-1] == "unresolved"
    assert p.state == State.UNRESOLVED
    # unresolved is explicit, carries a runbook, and is sticky (no further restarts)
    d = p.observe(t + 400, BROKEN)
    assert d.action == "none" and d.runbook and "RB-006" in d.runbook


def test_unresolved_clears_when_service_becomes_ready():
    p = make(max_restarts_per_window=1)
    t = 2.0
    for _ in range(10):
        d = p.observe(t, BROKEN)
        if d.action == "restart":
            p.restarted(t)
            p.grace_until = t
        t += 2.0
    assert p.state == State.UNRESOLVED
    d = p.observe(t, HEALTHY)
    assert d.action == "recovered" and p.state == State.HEALTHY


def test_config_errors_are_never_restarted():
    for code in ("weights_missing", "cuda_unavailable", "unsupported_hardware", "weights_checksum_mismatch"):
        p = make()
        ds = drive(p, 2.0, 30, Probe(True, True, False, code))
        assert "restart" not in [d.action for d in ds]
        assert ds[0].action == "hold_operator" and ds[0].runbook
        assert p.state == State.NEEDS_OPERATOR
        assert [d.action for d in ds[1:]] == ["none"] * 29  # reported once


def test_container_down_is_owned_by_docker_policy():
    p = make()
    ds = drive(p, 2.0, 10, Probe(False, False, False))
    assert all(d.action == "none" for d in ds) and p.state == State.CONTAINER_DOWN
    assert p.consecutive_failures == 0


def test_startup_grace_does_not_count_loading():
    p = RecoveryPolicy(PolicyConfig(startup_grace_s=120))
    ds = drive(p, 0.0, 50, Probe(True, True, False, "loading"))  # 100 s of loading
    assert all(d.action == "none" for d in ds)
    # after grace, a service that is still not ready is counted
    ds = drive(p, 122.0, 3, Probe(True, True, False, "loading"))
    assert ds[-1].action == "restart"


def test_docker_restart_reopens_grace_and_does_not_consume_budget():
    p = make()
    drive(p, 2.0, 2, Probe(False, False, False))  # docker restarting it
    ds = drive(p, 6.0, 10, Probe(True, False, False))  # process up, not answering yet
    assert all(d.action == "none" for d in ds)
    assert p.restarts_in_window(100.0) == 0


def test_recovered_decision_after_controller_restart_when_ready():
    p = make()
    drive(p, 2.0, 3, BROKEN)
    p.restarted(8.0)
    d = p.observe(20.0, HEALTHY)
    assert d.action == "recovered" and p.state == State.HEALTHY
