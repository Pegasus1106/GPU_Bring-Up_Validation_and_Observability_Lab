"""Render the native (Docker-free) stack: free ports, tool configs, and the service table.

Everything is derived from the same sources as the Compose deployment (monitoring/prometheus.yml,
alertmanager.yml, rules, dashboards, ops/controller.yaml) so both modes stay consistent.
All listeners bind 127.0.0.1 on ports chosen at start-up, because Sol compute nodes are shared.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common import endpoints
from common.util import REPO_ROOT, iso, read_json, write_json
from sol import tools

ALL_PORT_NAMES = list(endpoints.DEFAULT_PORTS)


@dataclass
class Options:
    profiles: list[str] = field(default_factory=lambda: ["gpu"])  # gpu | cpu
    grafana: bool = False
    node_exporter: bool = True
    cpu_threads: int = 0
    run_id: str = ""


def port_free(p: int) -> bool:
    with socket.socket() as s:  # no SO_REUSEADDR: any existing binder (even not yet listening) counts as "in use"
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


def allocate_ports() -> dict[str, int]:
    """Prefer the default port; fall back to a free ephemeral one (shared nodes often have collisions)."""
    out: dict[str, int] = {}
    taken: set[int] = set()
    for name, default in endpoints.DEFAULT_PORTS.items():
        p = default if default not in taken and port_free(default) else 0
        if p == 0:
            while True:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    p = s.getsockname()[1]
                if p not in taken:
                    break
        out[name] = p
        taken.add(p)
    return out


def _model_path() -> str:
    meta = REPO_ROOT / "manifests/model.json"
    fname = read_json(meta)["file"] if meta.exists() else "resnet18-f37072fd.pth"
    return str(REPO_ROOT / "manifests/weights" / fname)


def inference_env(name: str, profile: str, backend: str, port: int, cpu_threads: int) -> dict:
    return {
        "LAB_HOST": "127.0.0.1", "LAB_PORT": str(port), "LAB_BACKEND": backend, "LAB_WORKLOAD": "resnet18",
        "LAB_PRECISION": "fp32", "LAB_MODEL_PATH": _model_path(),
        "LAB_MODEL_MANIFEST": str(REPO_ROOT / "manifests/model.json"),
        "LAB_INPUT_DIR": str(REPO_ROOT / "manifests/inputs"), "LAB_INPUT_MANIFEST": str(REPO_ROOT / "manifests/inputs.json"),
        "LAB_MAX_PENDING": "32", "LAB_REQUEST_DEADLINE_S": "10", "LAB_CPU_THREADS": str(cpu_threads),
        "LAB_CHANNELS_LAST": os.environ.get("LAB_CHANNELS_LAST", "0"),
        "LAB_ENABLE_FAULT_INJECTION": "1", "LAB_FAULT_DIR": str(REPO_ROOT / "artifacts/faults/inject" / profile),
        "LAB_LOG_FILE": str(REPO_ROOT / f"artifacts/logs/{name}.jsonl"),
    }


def build_services(base: Path, ports: dict, opt: Options) -> dict:
    py = sys.executable
    run = base / "run"
    svc: dict[str, dict] = {}

    def add(name, cmd, restart="always", env=None, autostart=True):
        svc[name] = {"cmd": [str(c) for c in cmd], "restart": restart, "env": env or {}, "autostart": autostart}

    add("alert-receiver", [py, "-m", "ops.alert_receiver", "--addr", "127.0.0.1", "--port", ports["receiver"],
                           "--out", REPO_ROOT / "artifacts/alerts/alerts.jsonl"])
    am = tools.binary("alertmanager")
    add("alertmanager", [am, f"--config.file={run/'alertmanager.yml'}", f"--storage.path={base/'am-data'}",
                         f"--web.listen-address=127.0.0.1:{ports['alertmanager']}", "--cluster.listen-address="])
    prom = tools.binary("prometheus")
    add("prometheus", [prom, f"--config.file={run/'prometheus.yml'}", f"--storage.tsdb.path={base/'prom-data'/(opt.run_id or 'local')}",
                       "--storage.tsdb.retention.time=30d", f"--web.listen-address=127.0.0.1:{ports['prometheus']}", "--web.enable-lifecycle"])
    ne = tools.binary("node_exporter")
    if opt.node_exporter and ne:
        add("node-exporter", [ne, f"--web.listen-address=127.0.0.1:{ports['node_exporter']}"])
    if "gpu" in opt.profiles:
        add("gpu-exporter", [py, "-m", "ops.gpu_exporter", "--addr", "127.0.0.1", "--port", ports["gpu_exporter"]])
    gf = tools.binary("grafana")
    if opt.grafana and gf:
        gdir = tools.tools_dir() / "grafana"
        pw_file = base / "grafana-admin-password"
        if not pw_file.exists():
            pw_file.write_text(os.urandom(12).hex())
            pw_file.chmod(0o600)
        cmd = [gf, "server", f"--homepath={gdir}"] if gf.name == "grafana" else [gf, f"--homepath={gdir}"]
        add("grafana", cmd, env={
            "GF_SERVER_HTTP_ADDR": "127.0.0.1", "GF_SERVER_HTTP_PORT": str(ports["grafana"]),
            "GF_SECURITY_ADMIN_PASSWORD": pw_file.read_text().strip(),
            "GF_PATHS_PROVISIONING": str(run / "grafana/provisioning"), "GF_PATHS_DATA": str(base / "grafana-data"),
            "GF_PATHS_LOGS": str(base / "logs/grafana"), "GF_PATHS_PLUGINS": str(base / "grafana-data/plugins"),
            "GF_ANALYTICS_REPORTING_ENABLED": "false", "GF_ANALYTICS_CHECK_FOR_UPDATES": "false", "GF_USERS_ALLOW_SIGN_UP": "false"})
    add("controller", [py, "-m", "ops.controller", "--config", run / "controller.yaml"])

    if "gpu" in opt.profiles:
        add("inference-gpu", [py, "-m", "app.main"], restart="on-failure:3",
            env=inference_env("inference-gpu", "gpu", "cuda", ports["gpu"], opt.cpu_threads))
    if "cpu" in opt.profiles:
        add("inference-cpu", [py, "-m", "app.main"], restart="on-failure:3",
            env=inference_env("inference-cpu", "cpu", "cpu", ports["cpu"], opt.cpu_threads))
    # Separate fault case (F03 / F05): not started automatically, configured by `recreate` overrides.
    add("inference-fault", [py, "-m", "app.main"], restart="no",
        env=inference_env("inference-fault", "fault", "cpu", ports["fault"], opt.cpu_threads), autostart=False)
    return svc


def render_prometheus(base: Path, ports: dict, opt: Options) -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "monitoring/prometheus.yml").read_text())
    run = base / "run"
    cfg["rule_files"] = [str(REPO_ROOT / "monitoring/rules/*.yml")]
    cfg["alerting"]["alertmanagers"] = [{"static_configs": [{"targets": [f"127.0.0.1:{ports['alertmanager']}"]}]}]
    jobs = []
    for j in cfg["scrape_configs"]:
        name = j["job_name"]
        if name == "inference":
            j["file_sd_configs"] = [{"files": [str(run / "targets/inference.yml")], "refresh_interval": "5s"}]
        elif name == "dcgm-exporter":
            if "gpu" not in opt.profiles:
                continue
            j["file_sd_configs"] = [{"files": [str(run / "targets/dcgm.yml")], "refresh_interval": "5s"}]
        elif name == "node-exporter":
            if not (opt.node_exporter and tools.binary("node_exporter")):
                continue
            j["static_configs"] = [{"targets": [f"127.0.0.1:{ports['node_exporter']}"]}]
        elif name == "lab-controller":
            j["static_configs"] = [{"targets": [f"127.0.0.1:{ports['controller']}"]}]
        elif name == "alertmanager":
            j["static_configs"] = [{"targets": [f"127.0.0.1:{ports['alertmanager']}"]}]
        elif name == "prometheus":
            j["static_configs"] = [{"targets": [f"127.0.0.1:{ports['prometheus']}"]}]
        jobs.append(j)
    cfg["scrape_configs"] = jobs
    return cfg


def render_all(base: Path, ports: dict, opt: Options) -> dict:
    base = Path(base)
    run = base / "run"
    for d in ("run/targets", "state", "control", "logs", "am-data", "prom-data"):
        (base / d).mkdir(parents=True, exist_ok=True)
    for d in ("alerts", "controller", "logs", "faults/inject/gpu", "faults/inject/cpu", "faults/inject/fault", "deploy"):
        (REPO_ROOT / "artifacts" / d).mkdir(parents=True, exist_ok=True)

    write_json(base / "ports.json", ports)
    (run / "prometheus.yml").write_text(yaml.safe_dump(render_prometheus(base, ports, opt), sort_keys=False))

    am = yaml.safe_load((REPO_ROOT / "monitoring/alertmanager/alertmanager.yml").read_text())
    for r in am["receivers"]:
        for w in r.get("webhook_configs", []):
            w["url"] = f"http://127.0.0.1:{ports['receiver']}/alerts"
    (run / "alertmanager.yml").write_text(yaml.safe_dump(am, sort_keys=False))

    ctl = yaml.safe_load((REPO_ROOT / "ops/controller.yaml").read_text())
    ctl["runtime"] = "native"
    ctl["metrics_port"] = ports["controller"]
    ctl["workloads"] = [{"name": n, "container": n, "url": f"http://127.0.0.1:{ports[k]}", "optional": True}
                        for n, k, on in (("inference-gpu", "gpu", "gpu" in opt.profiles), ("inference-cpu", "cpu", "cpu" in opt.profiles),
                                         ("inference-fault", "fault", True)) if on]
    (run / "controller.yaml").write_text(yaml.safe_dump(ctl, sort_keys=False))

    if opt.grafana:
        prov = run / "grafana/provisioning"
        (prov / "datasources").mkdir(parents=True, exist_ok=True)
        (prov / "dashboards").mkdir(parents=True, exist_ok=True)
        ds = yaml.safe_load((REPO_ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml").read_text())
        ds["datasources"][0]["url"] = f"http://127.0.0.1:{ports['prometheus']}"
        (prov / "datasources/prometheus.yml").write_text(yaml.safe_dump(ds))
        dp = yaml.safe_load((REPO_ROOT / "monitoring/grafana/provisioning/dashboards/dashboards.yml").read_text())
        dp["providers"][0]["options"]["path"] = str(REPO_ROOT / "monitoring/grafana/dashboards")
        (prov / "dashboards/dashboards.yml").write_text(yaml.safe_dump(dp))

    services = build_services(base, ports, opt)
    write_json(run / "services.json", {"cwd": str(REPO_ROOT), "created_utc": iso(), "options": opt.__dict__, "services": services})
    # scrape target files for the active profiles (set_targets honours LAB_TARGETS_DIR / LAB_PORTS_FILE)
    return services


def child_env(base: Path) -> dict:
    """Environment every lab tool needs to find the native stack."""
    e = {"LAB_NATIVE": "1", "LAB_NATIVE_DIR": str(base), "LAB_PORTS_FILE": str(base / "ports.json"),
         "LAB_TARGETS_DIR": str(base / "run/targets"), "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return e
