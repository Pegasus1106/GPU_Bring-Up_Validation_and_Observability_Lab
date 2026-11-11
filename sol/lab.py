"""Docker-free lab driver for ASU Sol (and any Linux/macOS host): start the whole stack as user
processes, run lab stages against it, tear it down.

    python -m sol.lab setup                       # login node: download + verify monitoring binaries
    python -m sol.lab run preflight validate      # inside a GPU job: up -> stages -> down (always)
    python -m sol.lab up --profiles gpu,cpu       # manual control (interactive job)
    python -m sol.lab stage benchmark             # run one stage against a running stack
    python -m sol.lab status | down | env

Stages: prep preflight calibrate freeze-tolerances validate cpu-check benchmark freeze-baseline soak overload
        startup faults report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from common import endpoints
from common.util import REPO_ROOT, git_commit, iso, read_json, write_json
from ops.native_runtime import NativeRuntime, pid_alive
from sol import render, tools

PY = sys.executable

# stage -> workload profiles that must be up (empty: no stack needed)
STAGE_PROFILES: dict[str, list[str]] = {
    "prep": [], "calibrate": [], "freeze-tolerances": [], "report": [],
    "preflight": ["gpu"], "validate": ["gpu"], "freeze-baseline": ["gpu"], "soak": ["gpu"], "overload": ["gpu"],
    "startup": ["gpu"], "faults": ["gpu"], "benchmark": ["gpu", "cpu"],
    "cpu-check": ["cpu"],  # CPU profile works independently of any GPU (AC02)
}


# ------------------------------------------------------------------------------ environment
def base_dir() -> Path:
    return endpoints.native_dir()


def apply_env(base: Path | None = None) -> dict:
    base = base or base_dir()
    env = render.child_env(base)
    for k, v in env.items():
        os.environ[k] = v
    return env


def supervisor_pid(base: Path) -> int | None:
    p = base / "supervisor.pid"
    try:
        pid = int(p.read_text())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def foreign_stack(base: Path) -> dict | None:
    """A stack recorded by a job on ANOTHER node that is still pending/running. PIDs are node-local, so the
    pid check alone cannot see it; the checkout (and artifacts/native) is shared over the network filesystem."""
    try:
        rec = read_json(base / "supervisor.json")
    except (OSError, ValueError):
        return None
    if not rec or rec.get("host") == os.uname().nodename or not rec.get("job_id"):
        return None
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(rec["job_id"]), "-o", "%T"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return rec if r.stdout.strip() in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING") else None


def default_cpu_threads() -> int:
    n = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 4)
    return max(1, min(8, n))


# ------------------------------------------------------------------------------ commands
def cmd_setup(a) -> int:
    for d in ("artifacts/native", "artifacts/logs", "artifacts/deploy"):
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)
    rc = tools.main(["prometheus", "alertmanager", "node_exporter"] + (["grafana"] if a.grafana else []))
    print("\nbinaries in", tools.tools_dir())
    print("next: submit the prep job (weights, fixtures, reference) -> see docs/sol.md")
    return rc


def http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def wait_until(pred, timeout: float, what: str, interval: float = 0.5) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    print(f"  timeout waiting for {what}", file=sys.stderr)
    return False


def cmd_up(a) -> int:
    base = base_dir()
    if supervisor_pid(base):
        print("a native stack is already running here (python -m sol.lab down to stop it)")
        return 1
    other = foreign_stack(base)
    if other:
        print(f"refusing to start: Slurm job {other['job_id']} on {other['host']} is still using {base}.\n"
              "Two stacks in one checkout would overwrite each other's ports and state. Wait for it, or submit with\n"
              f"  sbatch --dependency=afterany:{other['job_id']} ...")
        return 3
    for need in ("prometheus", "alertmanager"):
        if not tools.binary(need):
            print(f"missing {need}: run `python -m sol.lab setup` on a login node first")
            return 2
    profiles = [p for p in a.profiles.split(",") if p]
    ports = render.allocate_ports()
    run_id = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%dT%H%M%S")
    opt = render.Options(profiles=profiles, grafana=a.grafana, node_exporter=not a.no_node_exporter,
                         cpu_threads=a.cpu_threads or default_cpu_threads(), run_id=run_id)
    # fresh runtime state (the previous run's events are kept for reference)
    for sub in ("state", "control"):
        for f in (base / sub).glob("*"):
            f.unlink()
    ev = base / "events.jsonl"
    if ev.exists():
        ev.rename(base / f"events-{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
    env = apply_env(base)
    render.render_all(base, ports, opt)
    subprocess.run([PY, str(REPO_ROOT / "deploy/set_targets.py"), "both" if len(profiles) > 1 else (profiles[0] if profiles else "cpu")],
                   env={**os.environ, **env}, check=False)
    if not profiles or "gpu" not in profiles:
        (base / "run/targets/dcgm.yml").write_text("[]\n")

    logf = open(base / "logs/supervisor.log", "ab")
    proc = subprocess.Popen([PY, "-m", "sol.supervisor", "--native-dir", str(base)], cwd=REPO_ROOT, env={**os.environ, **env},
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    (base / "supervisor.pid").write_text(str(proc.pid))
    write_json(base / "supervisor.json", {"host": os.uname().nodename, "pid": proc.pid, "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": iso()})
    print(f"supervisor pid {proc.pid}; ports: " + ", ".join(f"{k}={v}" for k, v in ports.items()))

    ok = wait_until(lambda: http_ok(f"http://127.0.0.1:{ports['prometheus']}/-/ready"), 60, "prometheus")
    ok &= wait_until(lambda: http_ok(f"http://127.0.0.1:{ports['alertmanager']}/-/ready"), 60, "alertmanager")
    ok &= wait_until(lambda: http_ok(f"http://127.0.0.1:{ports['receiver']}/"), 30, "alert receiver")
    for p in profiles:
        url = f"http://127.0.0.1:{ports[p]}/readyz"
        got = wait_until(lambda: http_ok(url), a.wait_s, f"inference-{p} /readyz (cold start)")
        if not got:
            try:
                print("  /readyz:", requests.get(url, timeout=2).text[:300])
            except requests.RequestException as e:
                print("  not reachable:", e)
        ok &= got
    write_deploy_log(base, ports, opt, ok)
    print("stack up" if ok else "stack started with problems: see artifacts/native/logs/*.log")
    if a.grafana:
        print(f"grafana: http://127.0.0.1:{ports['grafana']} (admin / password in {base/'grafana-admin-password'})")
    return 0 if ok else 1


def write_deploy_log(base: Path, ports: dict, opt: render.Options, ok: bool) -> None:
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if shutil_which("nvidia-smi") else None
    freeze = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    import hashlib

    lines = [
        f"# native deploy log {iso()}", f"commit: {git_commit()}", f"host: {os.uname().nodename}",
        f"slurm: job={os.environ.get('SLURM_JOB_ID')} partition={os.environ.get('SLURM_JOB_PARTITION')} gpus={os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"python: {PY}", f"env-lock-sha256: {hashlib.sha256(freeze.encode()).hexdigest()}",
        f"tools: {json.dumps({n: str(tools.binary(n)) for n in ('prometheus', 'alertmanager', 'node_exporter', 'grafana')})}",
        f"tool-versions: {json.dumps(tools.versions())}", f"profiles: {opt.profiles}", f"ports: {json.dumps(ports)}",
        f"gpu: {gpu.stdout.strip() if gpu else 'nvidia-smi not found'}", f"result: {'ready' if ok else 'PROBLEMS'}",
    ]
    out = REPO_ROOT / f"artifacts/deploy/native-up-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def shutil_which(x: str):
    import shutil

    return shutil.which(x)


def cmd_down(a=None) -> int:
    base = base_dir()
    pid = supervisor_pid(base)
    if not pid:
        print("no native stack running")
        return 0
    os.kill(pid, signal.SIGTERM)
    ok = wait_until(lambda: not pid_alive(pid), 60, "supervisor to stop", 0.5)
    if not ok:
        os.kill(pid, signal.SIGKILL)
    (base / "supervisor.pid").unlink(missing_ok=True)
    (base / "supervisor.json").unlink(missing_ok=True)
    print("stack stopped")
    return 0


def cmd_status(a=None) -> int:
    base = base_dir()
    apply_env(base)
    pid = supervisor_pid(base)
    print(f"supervisor: {'pid ' + str(pid) if pid else 'NOT running'}   dir: {base}")
    if not pid:
        return 1
    rt = NativeRuntime(base)
    ports = endpoints.ports()
    for f in sorted(rt.state_dir.glob("*.json")):
        s = read_json(f)
        st = rt.inspect_state(s["name"]) or {}
        extra = ""
        if s["name"].startswith("inference"):
            key = {"inference-gpu": "gpu", "inference-cpu": "cpu", "inference-fault": "fault"}[s["name"]]
            try:
                r = requests.get(f"http://127.0.0.1:{ports[key]}/readyz", timeout=1.5)
                extra = f"readyz={r.status_code}" + ("" if r.status_code == 200 else f" {r.json().get('reason_code')}")
            except requests.RequestException:
                extra = "readyz=unreachable"
        print(f"  {s['name']:16} running={st.get('Running')!s:5} restarts_used={s['fail_count']} gave_up={s['gave_up']} {extra}")
    return 0


def cmd_env(a=None) -> int:
    for k, v in render.child_env(base_dir()).items():
        print(f"export {k}='{v}'")
    return 0


# ------------------------------------------------------------------------------ stages
def latest_summary() -> Path:
    best, score = None, -1
    for p in glob.glob(str(REPO_ROOT / "artifacts/benchmark/*/summary.json")):
        s = read_json(p)
        if s.get("experiment_matrix", {}).get("duration_s", 0) >= 60 and s.get("completed_runs", 0) >= score:
            best, score = Path(p), s.get("completed_runs", 0)
    if not best:
        raise SystemExit("no benchmark summary with duration_s >= 60 found; run the benchmark stage first")
    return best


def prep() -> int:
    ok = True
    if not ((REPO_ROOT / "manifests/model.json").exists() and list((REPO_ROOT / "manifests/weights").glob("*.pth"))):
        ok &= subprocess.run([PY, "-m", "manifests.fetch_weights"], cwd=REPO_ROOT).returncode == 0
    if not (REPO_ROOT / "manifests/inputs.json").exists():
        ok &= subprocess.run([PY, "-m", "manifests.build_inputs"], cwd=REPO_ROOT).returncode == 0
    if not (REPO_ROOT / "manifests/reference/reference.json").exists():
        ok &= subprocess.run([PY, "-m", "manifests.make_reference"], cwd=REPO_ROOT, env={**os.environ, "LAB_BACKEND": "cpu"}).returncode == 0
    return 0 if ok else 1


def reload_prometheus() -> None:
    try:
        requests.post(endpoints.url("prometheus") + "/-/reload", timeout=5).raise_for_status()
        print("prometheus reloaded")
    except requests.RequestException as e:
        print(f"prometheus reload failed ({e}); restart the stack to load the new baseline")


def stage_commands(name: str) -> list[list[str]]:
    gpu, cpu = endpoints.url("gpu"), endpoints.url("cpu")
    return {
        "preflight": [[PY, "-m", "preflight.run_preflight", "--mode", "gpu", "--native"]],
        "cpu-check": [[PY, "-m", "preflight.run_preflight", "--mode", "cpu", "--native"],
                      [PY, "-m", "validation.correctness", "--backend", "cpu"],
                      [PY, "-m", "validation.api_checks", "--url", cpu, "--expect-backend", "cpu"]],
        "calibrate": [[PY, "-m", "validation.correctness", "--backend", "cuda", "--calibrate"]],
        "freeze-tolerances": [[PY, "-m", "validation.correctness", "--freeze"]],
        "validate": [[PY, "-m", "validation.correctness", "--backend", "cuda"],
                     [PY, "-m", "validation.api_checks", "--url", gpu, "--expect-backend", "cuda"]],
        "benchmark": [[PY, "-m", "benchmarks.run_matrix", "--config", "benchmarks/matrix.yaml", "--resume",
                       "--experiment-id", os.environ.get("EXPERIMENT", "baseline-v1")]],
        "soak": [[PY, "-m", "benchmarks.soak", "--backend", "cuda"]],
        "overload": [[PY, "-m", "benchmarks.overload", "--backend", "cuda"]],
        "startup": [[PY, "-m", "benchmarks.measure_startup", "--profile", "gpu"]],
        "faults": [[PY, "-m", "faults.run_faults", "--profile", "gpu"] + (os.environ.get("FAULT_ARGS", "").split() if os.environ.get("FAULT_ARGS") else [])],
        "report": [[PY, "-m", "report.generate_report"]],
    }[name]


def run_stage(name: str) -> int:
    print(f"\n=== stage {name} ({iso()}) ===", flush=True)
    if name == "prep":
        return prep()
    if name == "freeze-baseline":
        rc = subprocess.run([PY, "-m", "benchmarks.freeze_baseline", "--summary", str(latest_summary()), "--concurrency", "4"], cwd=REPO_ROOT).returncode
        reload_prometheus()
        return rc
    rc = 0
    for cmd in stage_commands(name):
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy()).returncode
        rc = rc or r
    return rc


def cmd_stage(a) -> int:
    apply_env()
    return run_stage(a.name)


def cmd_run(a) -> int:
    bad = [s for s in a.stages if s not in STAGE_PROFILES]
    if bad:
        print(f"unknown stage(s): {bad}; valid: {sorted(STAGE_PROFILES)}")
        return 2
    base = base_dir()
    apply_env(base)
    profiles: list[str] = []
    for s in a.stages:
        for p in STAGE_PROFILES[s]:
            if p not in profiles:
                profiles.append(p)
    if "benchmark" in a.stages and "gpu" not in profiles:
        profiles.append("gpu")

    def _term(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _term)  # Slurm sends SIGTERM at the time limit: still shut down cleanly
    results: dict[str, int] = {}
    started = False
    rc = 0
    stages = list(a.stages)
    # Preflight must still run (and record FAIL evidence) when the GPU service cannot become ready.
    tolerate_up_failure = a.continue_on_error or "preflight" in stages
    try:
        if "prep" in stages:  # weights/fixtures/reference must exist before any service tries to load them
            stages.remove("prep")
            results["prep"] = run_stage("prep")
            if results["prep"] != 0:
                print("prep failed; stopping")
                return 1
        if profiles:
            up_args = argparse.Namespace(profiles=",".join(profiles), grafana=a.grafana, no_node_exporter=a.no_node_exporter,
                                         cpu_threads=a.cpu_threads, wait_s=a.wait_s)
            started = True
            if cmd_up(up_args) != 0 and not tolerate_up_failure:
                return 1
        for s in stages:
            results[s] = run_stage(s)
            if results[s] != 0 and not a.continue_on_error and s not in ("preflight",):
                print(f"stage {s} failed (rc={results[s]}); stopping (use --continue-on-error to go on)")
                break
        rc = max(results.values(), default=0)
    finally:
        if started:
            cmd_down()
        write_json(base / f"last-run-{os.environ.get('SLURM_JOB_ID', 'local')}.json",
                   {"finished_utc": iso(), "commit": git_commit(), "stages": results, "profiles": profiles})
        print("\nstage results:", results)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("setup")
    s.add_argument("--grafana", action="store_true", help="also fetch Grafana (optional)")
    s.set_defaults(fn=cmd_setup)
    s = sp.add_parser("up")
    s.add_argument("--profiles", default="gpu", help="comma list of: gpu, cpu")
    s.add_argument("--grafana", action="store_true")
    s.add_argument("--no-node-exporter", action="store_true")
    s.add_argument("--cpu-threads", type=int, default=0)
    s.add_argument("--wait-s", type=float, default=240.0)
    s.set_defaults(fn=cmd_up)
    sp.add_parser("down").set_defaults(fn=cmd_down)
    sp.add_parser("status").set_defaults(fn=cmd_status)
    sp.add_parser("env").set_defaults(fn=cmd_env)
    s = sp.add_parser("stage")
    s.add_argument("name", choices=sorted(STAGE_PROFILES))
    s.set_defaults(fn=cmd_stage)
    s = sp.add_parser("run")
    s.add_argument("stages", nargs="+")
    s.add_argument("--grafana", action="store_true")
    s.add_argument("--no-node-exporter", action="store_true")
    s.add_argument("--cpu-threads", type=int, default=0)
    s.add_argument("--wait-s", type=float, default=240.0)
    s.add_argument("--continue-on-error", action="store_true")
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
