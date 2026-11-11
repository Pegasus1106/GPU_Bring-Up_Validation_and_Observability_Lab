"""Host preflight: inventory, GPU visibility, CUDA execution, artefact checks,
capability discovery and diagnostics -> artifacts/preflight/<run>/preflight.json

    python -m preflight.run_preflight --mode gpu --image gpu-lab/app:local
    python -m preflight.run_preflight --mode cpu --no-container      # development, no Docker

Exit code 0 only if no required check failed/errored (and, for gpu mode, PF02/PF03/PF05 passed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

from common import endpoints
from common.util import REPO_ROOT, git_commit, iso, new_run_id, read_json, sha256_file, write_json
from preflight import checks as C


def run_container_checks(run: C.Runner, mode: str, image: str, no_container: bool, min_free_gb: float) -> tuple[dict | None, str]:
    """no_container=True also covers native mode: the checks run in a fresh Python process of this environment."""
    backend = "cuda" if mode == "gpu" else "cpu"
    if no_container:
        r = run([sys.executable, "-m", "preflight.container_checks", "--backend", backend,
                 "--artifacts-dir", str(REPO_ROOT / "artifacts"), "--min-free-gb", str(min_free_gb)], 300)
    else:
        cmd = ["docker", "run", "--rm"]
        if mode == "gpu":
            cmd += ["--gpus", "all"]
        cmd += ["-v", f"{REPO_ROOT/'manifests'}:/opt/lab/manifests:ro", "-v", f"{REPO_ROOT/'artifacts'}:/opt/lab/artifacts",
                "-e", "LAB_MODEL_PATH=/opt/lab/manifests/weights/resnet18-f37072fd.pth",
                "-e", "LAB_MODEL_MANIFEST=/opt/lab/manifests/model.json",
                "-e", "LAB_INPUT_DIR=/opt/lab/manifests/inputs", "-e", "LAB_INPUT_MANIFEST=/opt/lab/manifests/inputs.json",
                "--user", f"{os.getuid()}:{os.getgid()}",
                image, "python", "-m", "preflight.container_checks", "--backend", backend, "--min-free-gb", str(min_free_gb)]
        r = run(cmd, 600)
    if not r.ok:
        return None, (r.error or r.stderr.strip()[-400:] or f"rc={r.rc}")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]), ""
    except (ValueError, IndexError) as e:
        return None, f"unparsable container output: {e}: {r.stdout[-200:]}"


def native_env_identity(out_dir: Path) -> dict:
    """Native deployments have no image digest; the pinned environment is identified by its full
    package list (written next to preflight.json) plus interpreter and torch build."""
    import hashlib
    import platform
    import subprocess as sp

    freeze = sp.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env-lock.txt").write_text(freeze)
    h = hashlib.sha256(freeze.encode()).hexdigest()
    return {"image_ref": "native environment (no container image)", "image_id": f"native-env:{h[:24]}", "env_lock_sha256": h,
            "env_lock_file": "env-lock.txt", "python": sys.executable, "python_version": platform.python_version()}


def image_identity(run: C.Runner, image: str, no_container: bool) -> dict:
    if no_container:
        return {"image_ref": "(none: running in host process)", "image_id": "host-process"}
    r = run(["docker", "image", "inspect", "--format", "{{json .}}", image], 20)
    if not r.ok:
        return {"image_ref": image, "image_id": "", "error": r.error or r.stderr.strip()[:200]}
    d = json.loads(r.stdout)
    return {"image_ref": image, "image_id": d.get("Id", ""), "repo_digests": d.get("RepoDigests", []), "created": d.get("Created")}


def main(argv=None, run: C.Runner = C.real_runner) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--image", default="gpu-lab/app:local")
    ap.add_argument("--no-container", action="store_true", help="run container checks in this process (dev only; not AC02 evidence)")
    ap.add_argument("--native", action="store_true", help="native deployment (ASU Sol, no Docker); also enabled by LAB_NATIVE=1")
    ap.add_argument("--dcgm-url", default=None, help="GPU metrics exporter (DCGM Exporter, or the nvidia-smi exporter in native mode)")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/preflight")
    a = ap.parse_args(argv)
    a.dcgm_url = a.dcgm_url or endpoints.url("gpu_exporter")

    run_id = new_run_id(f"preflight-{a.mode}")
    out_dir = a.out_dir / run_id
    commit = git_commit()

    native = a.native or endpoints.native()
    if native:
        os.environ["LAB_NATIVE"] = "1"  # container_checks labels the run executed_in=native-process
        a.no_container = True
    inv = C.host_inventory(run, native=native)
    image = native_env_identity(out_dir) if native else image_identity(run, a.image, a.no_container)
    cres, cerr = run_container_checks(run, a.mode, a.image, a.no_container, a.min_free_gb)
    checks: list[C.Check] = []

    info = (cres or {}).get("info", {})
    checks.append(C.pf01(inv, info, image, commit, a.mode, native=native))
    if native:
        checks.append(C.pf02_native(a.mode, inv.get("gpus", []), (cres or {}).get("cuda")))
    else:
        checks.append(C.pf02(a.mode, run, inv.get("gpus", []), a.image, via_container=not a.no_container))
    if cres is None:
        err = C.Check("PF03", "CUDA tensor operation executed in the workload container", C.ERROR if a.mode == "gpu" else C.SKIP,
                      f"container checks did not run: {cerr}", required=a.mode == "gpu")
        checks.append(err)
        checks.append(C.Check("PF04", "model checksum, input manifest, disk space, mounted-path permissions", C.ERROR, f"container checks did not run: {cerr}"))
    else:
        checks.append(C.pf03(a.mode, cres.get("cuda")))
        checks.append(C.pf04(cres.get("files")))

    # PF05 capability discovery
    counters = (REPO_ROOT / "monitoring/dcgm/counters.csv").read_text()
    smi_caps = C.discover_smi_capabilities(run) if a.mode == "gpu" else {}
    try:
        mt = requests.get(f"{a.dcgm_url}/metrics", timeout=5).text
        dcgm_caps = C.discover_dcgm_capabilities(counters, mt)
    except requests.RequestException as e:
        dcgm_caps = C.discover_dcgm_capabilities(counters, None, type(e).__name__) if a.mode == "gpu" else {}
    checks.append(C.pf05(a.mode, smi_caps, dcgm_caps))
    checks.append(C.pf06(a.mode, run))

    summary = C.summarize(checks, a.mode)
    caps_doc = {"run_id": run_id, "created_utc": iso(), "gpus": inv.get("gpus", []), "nvidia_smi_fields": smi_caps,
                "dcgm_fields": dcgm_caps,
                "note": "unsupported/unavailable fields must be displayed as unavailable, never as zero"}
    model_manifest = REPO_ROOT / "manifests/model.json"
    doc = {
        "run_id": run_id, "commit": commit, "timestamp_utc": iso(), "configured_mode": a.mode,
        "inventory": {"host": inv, "container": info, "image": image, "tf32": (cres or {}).get("tf32")},
        "hashes": {
            "model_weights_sha256": read_json(model_manifest)["sha256"] if model_manifest.exists() else None,
            "inputs_manifest_sha256": sha256_file(REPO_ROOT / "manifests/inputs.json") if (REPO_ROOT / "manifests/inputs.json").exists() else None,
            "preprocessing_sha256": sha256_file(REPO_ROOT / "manifests/preprocessing.json"),
            "requirements_app_sha256": sha256_file(REPO_ROOT / "requirements-app.txt"),
            "image_id": image.get("image_id"),
        },
        "executed_in": (cres or {}).get("executed_in", "unknown"),
        "capabilities_file": "capabilities.json",
        "checks": [c.to_dict() for c in checks],
        "summary": summary,
    }
    write_json(out_dir / "preflight.json", doc)
    write_json(out_dir / "capabilities.json", caps_doc)
    if a.mode == "gpu" and summary["gpu_ready"]:  # never overwrite the tested manifests with a failed/no-GPU run
        write_json(REPO_ROOT / "manifests/capabilities.json", caps_doc)
        # tested-environment manifest: the exact versions behind this host's results
        write_json(REPO_ROOT / "manifests/environment.json", {
            "created_utc": doc["timestamp_utc"], "preflight_run": run_id, "commit": commit, "gpu_ready": summary["gpu_ready"],
            "executed_in": doc["executed_in"], "inventory": doc["inventory"], "hashes": doc["hashes"],
        })

    for c in checks:
        print(f"[{c.status:5}] {c.id} {c.name}" + (f"  -- {c.reason}" if c.reason else ""))
    key = "gpu_ready" if a.mode == "gpu" else "cpu_ready"
    print(f"{key}={summary[key]}  blocking={summary['blocking']}  -> {out_dir/'preflight.json'}")
    return 0 if summary[key] else 1


if __name__ == "__main__":
    sys.exit(main())
