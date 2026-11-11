"""Preflight check logic (PF01-PF06). Command execution is injected so every
branch is unit-testable without a GPU.

Statuses: PASS | FAIL | SKIP | ERROR
  * FAIL  - the check ran and the condition is not met
  * ERROR - the check itself could not run (tool crashed, unparsable output)
  * SKIP  - not applicable / unsupported, always with a reason. An unsupported
            optional diagnostic is SKIP, never PASS.
A required check that is FAIL or ERROR blocks GPU readiness.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional

PASS, FAIL, SKIP, ERROR = "PASS", "FAIL", "SKIP", "ERROR"
STATUSES = (PASS, FAIL, SKIP, ERROR)


@dataclass
class CmdResult:
    cmd: list[str]
    rc: Optional[int]  # None => could not execute
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.rc == 0


Runner = Callable[[list[str], float], CmdResult]


def real_runner(cmd: list[str], timeout: float = 30.0) -> CmdResult:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return CmdResult(cmd, r.returncode, r.stdout, r.stderr)
    except FileNotFoundError:
        return CmdResult(cmd, None, error=f"{cmd[0]}: command not found")
    except subprocess.TimeoutExpired:
        return CmdResult(cmd, None, error=f"timed out after {timeout}s")
    except OSError as e:
        return CmdResult(cmd, None, error=str(e))


@dataclass
class Check:
    id: str
    name: str
    status: str
    reason: str = ""
    required: bool = True
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status, "required": self.required,
                "reason": self.reason, "evidence": self.evidence}


# ------------------------------------------------------------------ parsing helpers
def parse_gpu_csv(text: str, columns: list[str]) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(columns):
            rows.append(dict(zip(columns, parts[: len(columns)])))
    return rows


def classify_smi_value(value: str) -> str:
    """nvidia-smi query values -> supported | unsupported | unavailable."""
    v = value.strip()
    if v.startswith("[Not Supported]") or v == "Not Supported":
        return "unsupported"
    if v in ("[N/A]", "N/A", "", "[Insufficient Permissions]", "[Unknown Error]") or v.startswith("[Insuff"):
        return "unavailable"
    return "supported"


SMI_FIELDS = [
    "temperature.gpu", "power.draw", "power.limit", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "clocks.sm", "clocks.mem", "fan.speed",
    "ecc.mode.current", "ecc.errors.corrected.volatile.total", "ecc.errors.uncorrected.volatile.total",
    "pcie.link.gen.current",
]

DCGM_REQUESTED_PREFIX = "DCGM_FI_DEV_"


def parse_prom_metric_names(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        if line and not line.startswith("#"):
            names.add(line.split("{")[0].split(" ")[0])
    return names


def requested_dcgm_fields(counters_csv: str) -> list[str]:
    out = []
    for line in counters_csv.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line.startswith(DCGM_REQUESTED_PREFIX):
            out.append(line.split(",")[0].strip())
    return out


# ------------------------------------------------------------------ PF01 inventory
def host_inventory(run: Runner, native: bool = False) -> dict:
    inv: dict = {
        "os": platform.platform(), "machine": platform.machine(), "kernel": platform.release(),
        "python": platform.python_version(),
    }
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="):
                inv["os_pretty"] = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    try:
        import psutil

        inv["cpu_logical"] = psutil.cpu_count()
        inv["cpu_physical"] = psutil.cpu_count(logical=False)
        inv["ram_bytes"] = psutil.virtual_memory().total
        inv["disk_free_bytes_cwd"] = shutil.disk_usage(".").free
    except Exception:  # noqa: BLE001
        pass
    lscpu = run(["lscpu"], 5)
    if lscpu.ok:
        for line in lscpu.stdout.splitlines():
            if line.startswith("Model name:"):
                inv["cpu_model"] = line.split(":", 1)[1].strip()
    else:
        inv.setdefault("cpu_model", platform.processor() or "unknown")

    smi = run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id,vbios_version", "--format=csv,noheader"], 15)
    if smi.ok:
        inv["gpus"] = parse_gpu_csv(smi.stdout, ["name", "uuid", "driver_version", "memory_total", "pci_bus_id", "vbios_version"])
    else:
        inv["gpus"] = []
        inv["gpu_query_error"] = smi.error or smi.stderr.strip()[:200]
    if native:
        for k in ("docker_version", "docker_runtimes", "nvidia_ctk_version", "compose_version"):
            inv[k] = "n/a (native mode: no container runtime)"
        inv["slurm"] = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_CPUS_PER_TASK",
                                                       "SLURM_MEM_PER_NODE", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE")
                        if os.environ.get(k)}
    else:
        for key, cmd in (
            ("docker_version", ["docker", "version", "--format", "{{.Server.Version}}"]),
            ("docker_runtimes", ["docker", "info", "--format", "{{json .Runtimes}}"]),
            ("nvidia_ctk_version", ["nvidia-ctk", "--version"]),
            ("compose_version", ["docker", "compose", "version", "--short"]),
        ):
            r = run(cmd, 10)
            inv[key] = r.stdout.strip().splitlines()[0] if r.ok and r.stdout.strip() else (r.error or r.stderr.strip()[:120] or "unavailable")
    ts = run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], 5)
    inv["time_sync"] = {"ntp_synchronized": ts.stdout.strip()} if ts.ok else {"status": "unknown", "reason": ts.error or "timedatectl unavailable"}
    return inv


def pf01(inv: dict, container_info: dict, image: dict, commit: str, mode: str, native: bool = False) -> Check:
    missing = []
    for k in ("os", "cpu_logical", "ram_bytes"):
        if not inv.get(k):
            missing.append(k)
    if mode == "gpu":
        if not inv.get("gpus"):
            missing.append("gpu (name/uuid/driver)")
        if not native and (not inv.get("nvidia_ctk_version") or "not found" in str(inv.get("nvidia_ctk_version"))):
            missing.append("container toolkit version")
        if not container_info.get("cuda_runtime"):
            missing.append("CUDA runtime")
    for k in ("torch", "torchvision"):
        if not container_info.get(k):
            missing.append(k)
    if not image.get("image_id"):
        missing.append("environment lock id (native)" if native else "image digest/id")
    ev = {"commit": commit, "image": image}
    if missing:
        return Check("PF01", "hardware/software inventory recorded", FAIL, f"missing inventory fields: {missing}", evidence=ev)
    return Check("PF01", "hardware/software inventory recorded", PASS, evidence=ev)


# ------------------------------------------------------------------ PF02 visibility
def pf02(mode: str, run: Runner, host_gpus: list[dict], image_ref: str, via_container: bool = True) -> Check:
    name = "GPU visible on host and in container, matching identity"
    if mode != "gpu":
        return Check("PF02", name, SKIP, "cpu profile: GPU visibility not applicable", required=False)
    if not host_gpus:
        return Check("PF02", name, FAIL, "no GPU visible on the host (nvidia-smi found none)")
    if not via_container:
        return Check("PF02", name, SKIP, "container check skipped (--no-container); host-only evidence does not satisfy AC02", required=True,
                     evidence={"host_uuids": [g["uuid"] for g in host_gpus]})
    r = run(["docker", "run", "--rm", "--gpus", "all", image_ref, "nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], 120)
    if r.rc is None:
        return Check("PF02", name, ERROR, f"could not run container GPU check: {r.error}")
    if not r.ok:
        return Check("PF02", name, FAIL, f"container cannot see the GPU: {(r.stderr or r.stdout).strip()[:300]}",
                     evidence={"cmd": r.cmd, "rc": r.rc})
    cont = parse_gpu_csv(r.stdout, ["uuid", "name"])
    h, c = {g["uuid"] for g in host_gpus}, {g["uuid"] for g in cont}
    ev = {"host_uuids": sorted(h), "container_uuids": sorted(c), "cmd": r.cmd}
    if not c:
        return Check("PF02", name, FAIL, "container nvidia-smi returned no devices", evidence=ev)
    if not c <= h:
        return Check("PF02", name, FAIL, f"container UUIDs {sorted(c - h)} not present on host", evidence=ev)
    return Check("PF02", name, PASS, evidence=ev)


def pf02_native(mode: str, host_gpus: list[dict], cuda: dict | None) -> Check:
    """Native (no container) analogue of PF02: the GPU the job process uses (torch) must be one of
    the devices the host driver reports, matched by UUID. Weaker than a host-vs-container check;
    recorded as such."""
    name = "GPU visible to the job and to the framework, matching identity (native; no container)"
    if mode != "gpu":
        return Check("PF02", name, SKIP, "cpu profile: GPU visibility not applicable", required=False)
    if not host_gpus:
        return Check("PF02", name, FAIL, "no GPU visible to nvidia-smi in this job")
    if not cuda or not cuda.get("ok"):
        return Check("PF02", name, FAIL, "the framework cannot use the GPU (see PF03)", evidence={"host_uuids": [g["uuid"] for g in host_gpus]})
    host = {g["uuid"] for g in host_gpus}
    tu = cuda.get("device_uuid")
    ev = {"host_uuids": sorted(host), "torch_device_uuid": tu, "torch_device_name": cuda.get("device_name"),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "scope": "native-process (no container boundary to cross)"}
    if tu:
        return Check("PF02", name, PASS if tu in host else FAIL, "" if tu in host else f"framework device {tu} not reported by nvidia-smi", evidence=ev)
    names = {g["name"] for g in host_gpus}
    ok = cuda.get("device_name") in names
    return Check("PF02", name, PASS if ok else FAIL, "matched by device name only (torch did not expose a UUID)" if ok else "device name mismatch", evidence=ev)


# ------------------------------------------------------------------ PF03..PF05 (container side result -> checks)
def pf03(mode: str, cuda: dict | None) -> Check:
    name = "CUDA tensor operation executed in the workload container"
    if mode != "gpu":
        return Check("PF03", name, SKIP, "cpu profile: CUDA execution not applicable", required=False)
    if not cuda:
        return Check("PF03", name, ERROR, "no container result available")
    if cuda.get("ok"):
        return Check("PF03", name, PASS, evidence=cuda)
    return Check("PF03", name, FAIL, cuda.get("reason", "CUDA operation failed"), evidence=cuda)


def pf04(files: dict | None) -> Check:
    name = "model checksum, input manifest, disk space, mounted-path permissions"
    if not files:
        return Check("PF04", name, ERROR, "no container result available")
    problems = [k for k, v in files.get("checks", {}).items() if not v.get("ok")]
    if problems:
        return Check("PF04", name, FAIL, "; ".join(f"{k}: {files['checks'][k].get('reason', '')}" for k in problems), evidence=files)
    return Check("PF04", name, PASS, evidence=files)


def discover_smi_capabilities(run: Runner) -> dict:
    caps: dict = {}
    r0 = run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], 10)
    if not r0.ok:
        reason = r0.error or r0.stderr.strip()[:200] or "nvidia-smi failed"
        return {f: {"status": "unavailable", "reason": f"nvidia-smi not usable: {reason}"} for f in SMI_FIELDS}
    for f in SMI_FIELDS:
        r = run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"], 10)
        if not r.ok:
            caps[f] = {"status": "unsupported", "reason": (r.stderr or r.stdout).strip()[:160] or "query rejected by this driver"}
            continue
        vals = [v.strip() for v in r.stdout.strip().splitlines()]
        statuses = {classify_smi_value(v) for v in vals}
        st = "supported" if statuses == {"supported"} else ("unsupported" if "unsupported" in statuses else "unavailable")
        caps[f] = {"status": st, "sample": vals[:1] if st == "supported" else vals[:1]}
    return caps


def discover_dcgm_capabilities(counters_csv: str, metrics_text: Optional[str], error: str = "") -> dict:
    requested = requested_dcgm_fields(counters_csv)
    if metrics_text is None:
        return {f: {"status": "unavailable", "reason": f"dcgm-exporter not reachable: {error}"} for f in requested}
    present = parse_prom_metric_names(metrics_text)
    return {
        f: ({"status": "supported"} if f in present else
            {"status": "unsupported", "reason": "requested but not exported by this GPU/driver/exporter combination (e.g. DCGM-only fields in nvidia-smi mode)"})
        for f in requested
    }


def pf05(mode: str, smi_caps: dict, dcgm_caps: dict) -> Check:
    name = "telemetry and diagnostic capabilities discovered (supported / unsupported / unavailable)"
    if mode != "gpu":
        return Check("PF05", name, SKIP, "cpu profile: GPU capability discovery not applicable", required=False)
    summary = {"nvidia_smi": _count(smi_caps), "dcgm": _count(dcgm_caps)}
    if all(v["status"] == "unavailable" for v in smi_caps.values()):
        return Check("PF05", name, FAIL, "nvidia-smi could not be queried; no capability data", evidence=summary)
    return Check("PF05", name, PASS, "capabilities recorded; unsupported fields will be shown as unavailable, never zero", evidence=summary)


def _count(caps: dict) -> dict:
    out = {"supported": 0, "unsupported": 0, "unavailable": 0}
    for v in caps.values():
        out[v["status"]] += 1
    return out


# ------------------------------------------------------------------ PF06 diagnostics
def gpu_is_idle(run: Runner, samples: int = 3, threshold: int = 5) -> tuple[bool, str]:
    import time

    vals = []
    for _ in range(samples):
        r = run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 10)
        if not r.ok:
            return False, f"cannot read utilization: {r.error or r.stderr.strip()[:100]}"
        try:
            vals.append(max(int(x) for x in r.stdout.split()))
        except ValueError:
            return False, f"unparsable utilization: {r.stdout.strip()[:50]}"
        time.sleep(0.5)
    return max(vals) <= threshold, f"utilization samples {vals}% (idle threshold {threshold}%)"


UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def dcgm_ids_for_uuids(discovery_text: str, uuids: list[str]) -> list[int]:
    """Map GPU UUIDs to DCGM GPU ids using `dcgmi discovery -l`. On a shared node DCGM lists every GPU
    on the node, including ones that belong to other jobs, so diagnostics must be limited to our own."""
    mapping: dict[str, int] = {}
    cur: int | None = None
    for line in discovery_text.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if m:
            cur = int(m.group(1))
        for u in UUID_RE.findall(line):
            if cur is not None:
                mapping[u] = cur
    return sorted({mapping[u] for u in uuids if u in mapping})


def parse_dcgm_diag(text: str) -> dict:
    """Count Pass/Fail/Skip/Warn verdicts in the result table rows only (not in prose or stderr)."""
    counts = {"Pass": 0, "Fail": 0, "Skip": 0, "Warn": 0}
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            for tok in re.findall(r"\b(Pass|Fail|Skip|Warn)\b", line):
                counts[tok] += 1
    return counts


def pf06(mode: str, run: Runner, level: int = 1) -> Check:
    name = "applicable readiness diagnostics run while the GPU is idle"
    if mode != "gpu":
        return Check("PF06", name, SKIP, "cpu profile: GPU diagnostics not applicable", required=False)
    if shutil.which("dcgmi") is None and run(["dcgmi", "--version"], 5).rc is None:
        return Check("PF06", name, SKIP, "dcgmi not installed on this host; diagnostic not applicable here (install DCGM to enable)")
    idle, detail = gpu_is_idle(run)
    if not idle:
        return Check("PF06", name, SKIP, f"GPU not idle, diagnostics skipped: {detail}", evidence={"idle_check": detail})

    # Only ever diagnose the GPU(s) this job owns.
    mine = run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], 10)
    uuids = UUID_RE.findall(mine.stdout) if mine.ok else []
    if not uuids:
        return Check("PF06", name, SKIP, "cannot determine this job's GPU UUID from nvidia-smi; refusing to run diagnostics on unknown GPUs",
                     evidence={"idle_check": detail})
    disc = run(["dcgmi", "discovery", "-l"], 30)
    if disc.rc is None:
        return Check("PF06", name, ERROR, f"dcgmi discovery could not run: {disc.error}", evidence={"idle_check": detail})
    if not disc.ok:
        return Check("PF06", name, SKIP, "dcgmi cannot list GPUs (no reachable DCGM host engine or insufficient privileges): "
                     + (disc.stderr or disc.stdout).strip()[:200], evidence={"idle_check": detail, "rc": disc.rc})
    ids = dcgm_ids_for_uuids(disc.stdout, uuids)
    if not ids:
        return Check("PF06", name, SKIP, "this job's GPU was not found in DCGM's device list; not running diagnostics on other GPUs",
                     evidence={"idle_check": detail, "our_uuids": uuids, "discovery_tail": disc.stdout[-800:]})

    cmd = ["dcgmi", "diag", "-r", str(level), "-i", ",".join(str(i) for i in ids)]
    r = run(cmd, 600)
    ev = {"cmd": r.cmd, "rc": r.rc, "idle_check": detail, "dcgm_gpu_ids": ids, "our_uuids": uuids,
          "stdout_tail": r.stdout[-1800:], "stderr_tail": r.stderr[-600:]}
    if r.rc is None:
        return Check("PF06", name, ERROR, f"dcgmi diag could not run: {r.error}", evidence=ev)
    counts = parse_dcgm_diag(r.stdout)
    ev["verdicts"] = counts
    if counts["Fail"]:
        return Check("PF06", name, FAIL, f"dcgmi diag -r {level} reported {counts['Fail']} failing result(s) on GPU id(s) {ids}", evidence=ev)
    if counts["Pass"]:
        warn = f" ({counts['Warn']} warning(s))" if counts["Warn"] else ""
        return Check("PF06", name, PASS, f"dcgmi diag -r {level} passed on GPU id(s) {ids}{warn}", evidence=ev)
    if r.ok and counts["Skip"] and not counts["Pass"]:
        return Check("PF06", name, SKIP, "every selected diagnostic was skipped as unsupported on this GPU/environment", evidence=ev)
    if "not supported" in (r.stdout + r.stderr).lower():
        return Check("PF06", name, SKIP, "dcgmi diag reports the test is unsupported on this GPU/environment", evidence=ev)
    return Check("PF06", name, ERROR, "dcgmi diag output could not be interpreted (no Pass/Fail rows)", evidence=ev)


# ------------------------------------------------------------------ aggregation
def summarize(checks: list[Check], mode: str) -> dict:
    counts = {s: sum(1 for c in checks if c.status == s) for s in STATUSES}
    blocking = [c.id for c in checks if c.required and c.status in (FAIL, ERROR)]
    # In GPU mode a SKIPPED required visibility/CUDA check is not acceptable evidence either.
    must_pass = {"PF02", "PF03", "PF05"} if mode == "gpu" else set()
    not_passed = [c.id for c in checks if c.id in must_pass and c.status != PASS]
    ready = not blocking and not not_passed
    return {
        "counts": counts, "blocking": blocking, "required_not_passed": not_passed,
        "gpu_ready": bool(ready) if mode == "gpu" else None,
        "cpu_ready": (not blocking) if mode == "cpu" else None,
    }
