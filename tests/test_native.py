"""Native (Docker-free, ASU Sol) mode: endpoints, supervisor restart policy, runtime events,
rendered configs, GPU exporter, tool download verification. Uses real subprocesses with tiny scripts;
no GPU or Docker involved (SIMULATED hardware where noted)."""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from common import endpoints
from common.util import REPO_ROOT, read_jsonl
from ops import gpu_exporter
from ops.controller import Controller, ControllerConfig, Workload
from ops.native_runtime import NativeRuntime, TailProc, pid_alive
from ops.recovery import PolicyConfig
from sol import lab, render, tools

PY = sys.executable


def _find_promtool():
    found = shutil.which("promtool")
    if found:
        return found
    cands = sorted((REPO_ROOT / ".tools").glob("*/promtool"))
    return str(cands[0]) if cands else None


PROMTOOL = _find_promtool()


# ------------------------------------------------------------------ endpoints
def test_ports_file_is_only_used_in_native_mode(tmp_path, monkeypatch):
    pf = tmp_path / "ports.json"
    pf.write_text(json.dumps({"gpu": 18000, "prometheus": 19090}))
    monkeypatch.setenv("LAB_PORTS_FILE", str(pf))
    monkeypatch.delenv("LAB_NATIVE", raising=False)
    assert endpoints.port("gpu") == 8000  # Docker mode keeps fixed ports
    monkeypatch.setenv("LAB_NATIVE", "1")
    assert endpoints.port("gpu") == 18000 and endpoints.url("prometheus") == "http://127.0.0.1:19090"
    assert endpoints.port("cpu") == 8001  # unspecified names keep their default


def test_allocate_ports_avoids_occupied_defaults_and_is_unique():
    holder = socket.socket()
    try:
        holder.bind(("127.0.0.1", 9094))  # simulate another user's job holding the default receiver port
    except OSError:
        pass  # already occupied by something else: same situation
    try:
        p = render.allocate_ports()
    finally:
        holder.close()
    assert set(p) == set(endpoints.DEFAULT_PORTS)
    assert len(set(p.values())) == len(p)
    assert p["receiver"] != 9094


# ------------------------------------------------------------------ GPU exporter (nvidia-smi source)
SMI_ROW = "0, GPU-abc123, NVIDIA A100-SXM4-80GB, 37, 12, 1024, 79000, 41, 62.5, 1410, 1593, 0, 0"


def test_exporter_emits_dcgm_compatible_names_and_labels():
    t = gpu_exporter.render_metrics(SMI_ROW, "node1")
    assert 'DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-abc123",modelName="NVIDIA A100-SXM4-80GB",Hostname="node1"} 37' in t
    for m in ("DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_FB_FREE", "DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_SM_CLOCK"):
        assert m in t


def test_exporter_omits_unsupported_fields_instead_of_exporting_zero():
    row = "0, GPU-abc, A100, 10, [N/A], 100, 200, 40, [Not Supported], 1410, 1593, [N/A], [Not Supported]"
    t = gpu_exporter.render_metrics(row)
    assert "DCGM_FI_DEV_GPU_UTIL" in t
    for absent in ("DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL"):
        assert absent not in t
    assert gpu_exporter.render_metrics("") == ""  # nvidia-smi failure => no GPU series at all


def test_exporter_reports_its_source_and_scrape_status_when_smi_is_missing():
    col = gpu_exporter.Collector(smi="definitely-not-nvidia-smi")
    t = col.collect()
    assert "gpu_exporter_scrape_ok 0" in t and 'source="nvidia-smi"' in t and "DCGM_FI_DEV_" not in t


# ------------------------------------------------------------------ supervisor (real subprocesses)
def write_services(base: Path, services: dict):
    (base / "run").mkdir(parents=True, exist_ok=True)
    (base / "run/services.json").write_text(json.dumps({"cwd": str(REPO_ROOT), "services": services}))


def svc(code: str, restart="no", env=None, autostart=True):
    return {"cmd": [PY, "-c", code], "restart": restart, "env": env or {}, "autostart": autostart}


@pytest.fixture
def supervised(tmp_path):
    procs = []

    def start(services):
        write_services(tmp_path, services)
        p = subprocess.Popen([PY, "-m", "sol.supervisor", "--native-dir", str(tmp_path)], cwd=REPO_ROOT,
                             env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        return NativeRuntime(tmp_path)

    yield start
    for p in procs:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def wait(pred, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def events(rt, name, action=None):
    return [e for e in read_jsonl(rt.events_file) if e["Actor"]["Attributes"]["name"] == name and (action is None or e["Action"] == action)]


SLEEPER = "import time; time.sleep(120)"


def test_on_failure_policy_retries_at_most_n_times_then_stays_down(supervised):
    rt = supervised({"crasher": svc("import sys; sys.exit(3)", restart="on-failure:2")})
    assert wait(lambda: (rt.read_state("crasher") or {}).get("gave_up"), 20)
    assert len(events(rt, "crasher", "start")) == 3  # first start + 2 retries
    dies = events(rt, "crasher", "die")
    assert len(dies) == 3 and {d["Actor"]["Attributes"]["exitCode"] for d in dies} == {"3"}
    time.sleep(1.5)
    assert len(events(rt, "crasher", "start")) == 3  # no further attempts
    assert rt.inspect_state("crasher")["Running"] is False


def test_clean_exit_is_not_restarted_under_on_failure(supervised):
    rt = supervised({"ok": svc("pass", restart="on-failure:3")})
    assert wait(lambda: events(rt, "ok", "die"), 10)
    time.sleep(1.5)
    assert len(events(rt, "ok", "start")) == 1


def test_sigkill_is_reported_like_docker_137_and_restarted(supervised):
    rt = supervised({"w": svc(SLEEPER, restart="on-failure:3")})
    assert wait(lambda: (rt.inspect_state("w") or {}).get("Running"))
    first = rt.read_state("w")["started_epoch"]
    assert rt.kill9("w")["ok"]
    assert wait(lambda: events(rt, "w", "die"))
    die = events(rt, "w", "die")[0]["Actor"]["Attributes"]
    assert die["exitCode"] == "137" and die["signal"] == "9"
    assert wait(lambda: (rt.read_state("w") or {}).get("started_epoch", 0) > first and rt.inspect_state("w")["Running"], 15)
    assert len(events(rt, "w", "start")) == 2


def test_manual_stop_is_never_auto_restarted_and_start_brings_it_back(supervised):
    rt = supervised({"exp": svc(SLEEPER, restart="always")})
    assert wait(lambda: (rt.inspect_state("exp") or {}).get("Running"))
    assert rt.stop("exp")
    time.sleep(2.5)  # 'always' would have restarted within 1 s if stop were not manual
    assert not rt.inspect_state("exp")["Running"]
    assert rt.start("exp")


def test_controller_restart_does_not_consume_the_failure_budget(supervised):
    rt = supervised({"w": svc(SLEEPER, restart="on-failure:1")})
    assert wait(lambda: (rt.inspect_state("w") or {}).get("Running"))
    for _ in range(3):
        ok, _ = rt.restart("w")
        assert ok
    assert rt.read_state("w")["fail_count"] == 0 and not rt.read_state("w")["gave_up"]


def test_recreate_applies_env_overrides_and_none_unsets(supervised, tmp_path):
    out = tmp_path / "env.txt"
    code = f"import os,time; open({str(out)!r},'w').write(repr((os.environ.get('A'), os.environ.get('B'), os.environ.get('CUDA_VISIBLE_DEVICES')))); time.sleep(60)"
    rt = supervised({"f": svc(code, env={"A": "base", "B": "keep"}, autostart=False)})
    assert rt.recreate("f", env={"A": "override", "CUDA_VISIBLE_DEVICES": ""})
    assert wait(lambda: out.exists() and out.read_text())
    assert eval(out.read_text()) == ("override", "keep", "")
    out.unlink()
    assert rt.recreate("f", env={"B": None})  # unset B; A returns to its base value
    assert wait(lambda: out.exists() and out.read_text())
    assert eval(out.read_text()) == ("base", None, os.environ.get("CUDA_VISIBLE_DEVICES"))


def test_shutdown_stops_every_service(supervised):
    rt = supervised({"a": svc(SLEEPER), "b": svc(SLEEPER, restart="always")})
    assert wait(lambda: all((rt.inspect_state(n) or {}).get("Running") for n in "ab"))
    pids = [rt.inspect_state(n)["Pid"] for n in "ab"]
    rt.command("shutdown")
    assert wait(lambda: not any(pid_alive(p) for p in pids), 20)


# ------------------------------------------------------------------ controller reads native events
def test_controller_counts_a_restart_from_native_events(tmp_path):
    rt = NativeRuntime(tmp_path)
    cfg = ControllerConfig(workloads=[Workload("inference-gpu", "inference-gpu", "http://x")], policy=PolicyConfig(), metrics_port=0,
                           state_file=tmp_path / "s.json", events_file=tmp_path / "f.jsonl", runtime="native")
    c = Controller(cfg, docker=rt)
    proc = rt.events(["inference-gpu"])
    rt.emit("start", "inference-gpu")
    rt.emit("die", "inference-gpu", exitCode=137, signal=9)
    rt.emit("start", "inference-gpu")
    got = []
    for line in proc.stdout:
        got.append(json.loads(line))
        if len(got) == 3:
            break
    proc.terminate()
    for e in got:
        c.handle_event(e)
    assert c.state["workloads"]["inference-gpu"]["restarts_total"] == 1
    rec = [r for r in read_jsonl(tmp_path / "f.jsonl") if r["action"] == "die"][0]
    assert rec["exit_code"] == "137" and rec["signal"] == "9"


def test_tailproc_only_sees_events_after_creation(tmp_path):
    rt = NativeRuntime(tmp_path)
    rt.emit("die", "old")
    proc = TailProc(rt.events_file)
    rt.emit("start", "new")
    first = json.loads(next(proc.stdout))
    proc.terminate()
    assert first["Actor"]["Attributes"]["name"] == "new"


# ------------------------------------------------------------------ rendered configuration
@pytest.fixture
def rendered(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TOOLS_ROOT", tmp_path / "tools")
    d = tools.tools_dir()
    d.mkdir(parents=True)
    for n in ("prometheus", "alertmanager", "node_exporter"):
        (d / n).write_text("#!/bin/sh\n")
    ports = {k: 20000 + i for i, k in enumerate(endpoints.DEFAULT_PORTS)}
    opt = render.Options(profiles=["gpu", "cpu"], cpu_threads=4, run_id="42")
    render.render_all(tmp_path / "native", ports, opt)
    return tmp_path / "native", ports


def test_rendered_services_match_the_compose_contract(rendered):
    base, ports = rendered
    s = json.loads((base / "run/services.json").read_text())["services"]
    assert s["inference-gpu"]["restart"] == "on-failure:3" and s["inference-gpu"]["env"]["LAB_BACKEND"] == "cuda"
    assert s["inference-cpu"]["env"]["LAB_BACKEND"] == "cpu" and s["inference-cpu"]["env"]["LAB_CPU_THREADS"] == "4"
    assert s["inference-fault"]["autostart"] is False
    assert s["inference-gpu"]["env"]["LAB_PORT"] == str(ports["gpu"]) and s["inference-gpu"]["env"]["LAB_HOST"] == "127.0.0.1"
    assert s["inference-gpu"]["env"]["LAB_MAX_PENDING"] == "32" and s["inference-gpu"]["env"]["LAB_REQUEST_DEADLINE_S"] == "10"
    assert s["inference-gpu"]["env"]["LAB_FAULT_DIR"].endswith("artifacts/faults/inject/gpu")
    assert {"alert-receiver", "alertmanager", "prometheus", "gpu-exporter", "node-exporter", "controller"} <= set(s)
    assert "grafana" not in s  # optional
    # every listener is loopback
    for name, v in s.items():
        for arg in v["cmd"]:
            if "listen-address" in arg and not arg.startswith("--cluster"):  # --cluster.listen-address= disables gossip
                assert "127.0.0.1:" in arg
    assert "127.0.0.1" in s["gpu-exporter"]["cmd"] and "127.0.0.1" in s["alert-receiver"]["cmd"]


def test_rendered_prometheus_uses_runtime_ports_and_drops_absent_exporters(rendered, tmp_path):
    base, ports = rendered
    cfg = yaml.safe_load((base / "run/prometheus.yml").read_text())
    assert cfg["global"]["scrape_interval"] == "5s" and cfg["global"]["evaluation_interval"] == "5s"
    assert cfg["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"] == [f"127.0.0.1:{ports['alertmanager']}"]
    jobs = {j["job_name"]: j for j in cfg["scrape_configs"]}
    assert jobs["lab-controller"]["static_configs"][0]["targets"] == [f"127.0.0.1:{ports['controller']}"]
    assert "dcgm-exporter" in jobs and "node-exporter" in jobs
    assert cfg["rule_files"] == [str(REPO_ROOT / "monitoring/rules/*.yml")]
    # CPU-only run without node exporter: those scrape jobs must not exist (else they would alert as down)
    opt = render.Options(profiles=["cpu"], node_exporter=False)
    cfg2 = render.render_prometheus(base, ports, opt)
    names = {j["job_name"] for j in cfg2["scrape_configs"]}
    assert "dcgm-exporter" not in names and "node-exporter" not in names


@pytest.mark.skipif(PROMTOOL is None, reason="promtool not installed")
def test_rendered_prometheus_config_is_valid(rendered):
    base, _ = rendered
    for n in ("inference.yml", "dcgm.yml"):
        (base / "run/targets" / n).write_text("[]\n")
    r = subprocess.run([PROMTOOL, "check", "config", str(base / "run/prometheus.yml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_rendered_alertmanager_and_controller_follow_ports(rendered):
    base, ports = rendered
    am = yaml.safe_load((base / "run/alertmanager.yml").read_text())
    assert am["route"]["group_wait"] == "5s"
    assert [w["url"] for r in am["receivers"] for w in r["webhook_configs"]] == [f"http://127.0.0.1:{ports['receiver']}/alerts"]
    ctl = ControllerConfig.load(base / "run/controller.yaml")
    assert ctl.runtime == "native" and ctl.metrics_port == ports["controller"]
    assert {w.name: w.url for w in ctl.workloads} == {
        "inference-gpu": f"http://127.0.0.1:{ports['gpu']}", "inference-cpu": f"http://127.0.0.1:{ports['cpu']}",
        "inference-fault": f"http://127.0.0.1:{ports['fault']}"}
    assert all(w.container == w.name for w in ctl.workloads)  # units are addressed by service name
    assert json.loads((base / "ports.json").read_text()) == ports


def test_child_env_points_every_tool_at_the_native_stack(tmp_path):
    e = render.child_env(tmp_path)
    assert e["LAB_NATIVE"] == "1" and e["LAB_PORTS_FILE"].endswith("ports.json") and e["LAB_TARGETS_DIR"].endswith("run/targets")


# ------------------------------------------------------------------ tool downloads are verified
def test_tool_specs_cover_x86_and_grace_hopper_arm(monkeypatch):
    for arch, label in (("amd64", "linux-amd64"), ("arm64", "linux-arm64")):
        monkeypatch.setattr(tools, "os_arch", lambda a=arch: ("linux", a))
        for name in ("prometheus", "alertmanager", "node_exporter", "grafana"):
            sp = tools.spec(name)
            assert label in sp["fname"] and sp["url"].endswith(sp["fname"]) and sp["version"]
        assert tools.spec("prometheus")["sums"].endswith("sha256sums.txt")


def test_tool_versions_come_from_versions_env():
    v = tools.versions()
    env = (REPO_ROOT / "deploy/versions.env").read_text()
    for name, ver in v.items():
        assert ver in env


def make_fake_tgz(path: Path, files=("prometheus", "promtool")):
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as t:
        for f in files:
            data = b"#!/bin/sh\n"
            ti = tarfile.TarInfo(f"prometheus-9.9.9.linux-amd64/{f}")
            ti.size, ti.mode = len(data), 0o755
            t.addfile(ti, io.BytesIO(data))


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TOOLS_ROOT", tmp_path / "t")
    monkeypatch.setattr(tools, "os_arch", lambda: ("linux", "amd64"))
    ref = tmp_path / "ref.tgz"
    make_fake_tgz(ref)  # gzip embeds a timestamp, so build once and serve the same bytes every time
    monkeypatch.setattr(tools.urllib.request, "urlretrieve", lambda url, dest: shutil.copyfile(ref, dest))
    return tmp_path


def test_download_with_matching_checksum_installs(fake_download, monkeypatch):
    ref = fake_download / "ref.tgz"
    monkeypatch.setattr(tools, "expected_sha", lambda sp: tools.sha256(ref))
    p = tools.fetch("prometheus")
    assert p.exists() and os.access(p, os.X_OK)


def test_download_with_wrong_checksum_is_rejected_and_nothing_installed(fake_download, monkeypatch):
    monkeypatch.setattr(tools, "expected_sha", lambda sp: "0" * 64)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        tools.fetch("prometheus")
    assert tools.binary("prometheus") is None


def test_download_without_a_published_checksum_is_refused(fake_download, monkeypatch):
    monkeypatch.setattr(tools, "expected_sha", lambda sp: None)
    with pytest.raises(SystemExit, match="refusing"):
        tools.fetch("prometheus")


# ------------------------------------------------------------------ preflight native logic
def test_pf02_native_matches_framework_device_to_host_uuid():
    from preflight import checks as C

    host = [{"uuid": "GPU-aaa", "name": "NVIDIA A100"}]
    assert C.pf02_native("gpu", host, {"ok": True, "device_uuid": "GPU-aaa"}).status == C.PASS
    assert C.pf02_native("gpu", host, {"ok": True, "device_uuid": "GPU-zzz"}).status == C.FAIL
    assert C.pf02_native("gpu", host, {"ok": False}).status == C.FAIL
    assert C.pf02_native("gpu", [], {"ok": True}).status == C.FAIL
    assert C.pf02_native("gpu", host, {"ok": True, "device_name": "NVIDIA A100"}).status == C.PASS  # name fallback is noted
    assert C.pf02_native("cpu", host, None).status == C.SKIP


def test_pf01_native_does_not_demand_a_container_toolkit_or_image_digest():
    from preflight import checks as C

    inv = {"os": "Linux", "cpu_logical": 8, "ram_bytes": 1, "gpus": [{"uuid": "GPU-a"}], "nvidia_ctk_version": "n/a (native mode: no container runtime)"}
    info = {"torch": "2.x", "torchvision": "0.x", "cuda_runtime": "12.8"}
    assert C.pf01(inv, info, {"image_id": "native-env:abc"}, "deadbeef", "gpu", native=True).status == C.PASS
    assert C.pf01(inv, info, {"image_id": ""}, "deadbeef", "gpu", native=True).status == C.FAIL  # still needs an environment id


# ------------------------------------------------------------------ job scripts and stages
def test_every_stage_resolves_to_commands(monkeypatch):
    monkeypatch.setenv("LAB_NATIVE", "1")
    for st in lab.STAGE_PROFILES:
        if st in ("prep", "freeze-baseline"):
            continue
        cmds = lab.stage_commands(st)
        assert cmds and all(isinstance(c, list) and c for c in cmds)
    assert lab.STAGE_PROFILES["benchmark"] == ["gpu", "cpu"]  # the CPU-vs-GPU matrix needs both services


def test_submit_groups_only_use_known_stages():
    text = (REPO_ROOT / "sol/submit.sh").read_text()
    import re

    for stages in re.findall(r'stages="([^"]+)"', text):
        assert set(stages.split()) <= set(lab.STAGE_PROFILES), stages


def test_sbatch_header_follows_asu_docs():
    h = (REPO_ROOT / "sol/lab.sbatch").read_text()
    for needle in ("#SBATCH -p htc", "#SBATCH -q public", "#SBATCH -G a100:1", "#SBATCH --export=NONE", "module load mamba/latest", "python -m sol.lab run"):
        assert needle in h


def test_native_fault_case_withholds_gpu_by_hiding_devices(monkeypatch):
    from faults import runtime

    seen = {}

    class FakeRT:
        def recreate(self, name, env=None, timeout=30.0):
            seen["env"] = env
            return True

    monkeypatch.setattr(runtime, "_set_targets", lambda mode: None)
    b = runtime.NativeBackend.__new__(runtime.NativeBackend)
    b.rt = FakeRT()
    b.fault_case_up("cuda", "/opt/lab/manifests/weights/x.pth", with_gpu=False)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""  # no device visible: explicit CUDA check must fail
    assert seen["env"]["LAB_MODEL_PATH"] == str(REPO_ROOT / "manifests/weights/x.pth")
    b.fault_case_up("cuda", "/opt/lab/manifests/weights/x.pth", with_gpu=True)
    assert "CUDA_VISIBLE_DEVICES" not in seen["env"]  # correction: inherit the job's own GPU allocation


def test_run_executes_prep_before_the_stack_starts_and_always_stops_it(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("LAB_NATIVE_DIR", str(tmp_path))
    monkeypatch.setattr(lab, "run_stage", lambda name: calls.append(("stage", name)) or 0)
    monkeypatch.setattr(lab, "cmd_up", lambda a: calls.append(("up", a.profiles)) or 0)
    monkeypatch.setattr(lab, "cmd_down", lambda a=None: calls.append(("down",)) or 0)
    ns = type("A", (), dict(stages=["preflight", "prep", "calibrate"], grafana=False, no_node_exporter=False, cpu_threads=0,
                            wait_s=1, continue_on_error=False))()
    assert lab.cmd_run(ns) == 0
    assert calls == [("stage", "prep"), ("up", "gpu"), ("stage", "preflight"), ("stage", "calibrate"), ("down",)]


def test_run_stops_the_stack_even_when_a_stage_raises(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("LAB_NATIVE_DIR", str(tmp_path))

    def boom(name):
        raise RuntimeError("stage crashed")

    monkeypatch.setattr(lab, "run_stage", boom)
    monkeypatch.setattr(lab, "cmd_up", lambda a: 0)
    monkeypatch.setattr(lab, "cmd_down", lambda a=None: calls.append("down") or 0)
    ns = type("A", (), dict(stages=["validate"], grafana=False, no_node_exporter=False, cpu_threads=0, wait_s=1, continue_on_error=False))()
    with pytest.raises(RuntimeError):
        lab.cmd_run(ns)
    assert calls == ["down"]


def test_run_still_runs_preflight_when_the_gpu_service_never_becomes_ready(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setenv("LAB_NATIVE_DIR", str(tmp_path))
    monkeypatch.setattr(lab, "run_stage", lambda name: ran.append(name) or 1)
    monkeypatch.setattr(lab, "cmd_up", lambda a: 1)  # readiness timeout
    monkeypatch.setattr(lab, "cmd_down", lambda a=None: 0)
    ns = type("A", (), dict(stages=["preflight"], grafana=False, no_node_exporter=False, cpu_threads=0, wait_s=1, continue_on_error=False))()
    assert lab.cmd_run(ns) == 1 and ran == ["preflight"]  # failing evidence is recorded, not skipped


def test_cpu_check_stage_needs_only_the_cpu_service(monkeypatch):
    monkeypatch.setenv("LAB_NATIVE", "1")
    assert lab.STAGE_PROFILES["cpu-check"] == ["cpu"]
    cmds = lab.stage_commands("cpu-check")
    assert any("--mode" in c and "cpu" in c for c in cmds) and any("api_checks" in " ".join(c) for c in cmds)
    assert (REPO_ROOT / "sol/submit.sh").read_text().count("cpu-check") >= 1


def test_first_trial_numbers_a_rerun_after_existing_records(monkeypatch):
    from faults import harness

    seen = []
    monkeypatch.setattr(harness, "run_trial", lambda s, k, c: seen.append(k) or {"final_status": "RECOVERED_VALIDATED", "cleanup": {"ok": True}})
    monkeypatch.setattr(harness.time, "sleep", lambda *_: None)
    s = type("S", (), {"id": "F04", "name": "x"})()
    harness.run_scenarios([s], trials=1, c=harness.Ctx(), first_trial=4)
    assert seen == [4]


def test_a_stack_owned_by_another_running_job_on_another_node_is_refused(monkeypatch, tmp_path):
    from common.util import write_json

    write_json(tmp_path / "supervisor.json", {"host": "some-other-node", "pid": 1, "job_id": "999207"})
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("R", (), {"stdout": "RUNNING\n"})()

    monkeypatch.setattr(lab.subprocess, "run", fake_run)
    assert lab.foreign_stack(tmp_path)["job_id"] == "999207" and calls[0][:2] == ["squeue", "-h"]
    ns = type("A", (), dict(profiles="cpu", grafana=False, no_node_exporter=False, cpu_threads=0, wait_s=1))()
    monkeypatch.setenv("LAB_NATIVE_DIR", str(tmp_path))
    assert lab.cmd_up(ns) == 3  # refused before touching any state


def test_a_finished_foreign_job_or_the_same_node_is_not_a_conflict(monkeypatch, tmp_path):
    from common.util import write_json

    monkeypatch.setattr(lab.subprocess, "run", lambda cmd, **kw: type("R", (), {"stdout": ""})())  # squeue: job gone
    write_json(tmp_path / "supervisor.json", {"host": "other", "pid": 1, "job_id": "1"})
    assert lab.foreign_stack(tmp_path) is None
    write_json(tmp_path / "supervisor.json", {"host": os.uname().nodename, "pid": 1, "job_id": "1"})
    assert lab.foreign_stack(tmp_path) is None  # same node: the pid check handles it
    monkeypatch.setattr(lab.subprocess, "run", lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError()))
    write_json(tmp_path / "supervisor.json", {"host": "other", "pid": 1, "job_id": "1"})
    assert lab.foreign_stack(tmp_path) is None  # no squeue (not on a cluster): do not block
