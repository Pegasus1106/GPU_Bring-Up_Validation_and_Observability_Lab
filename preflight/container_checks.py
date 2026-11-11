"""Checks that must run *inside the workload image* (framework, CUDA execution, mounted paths).

    docker run --rm --gpus all -v $PWD/manifests:/opt/lab/manifests:ro IMAGE \
        python -m preflight.container_checks --backend cuda

Prints a single JSON document on stdout. The host orchestrator (run_preflight.py) turns it
into PF01/PF03/PF04/PF05 results.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.errors import LabError
from app.inputs import load_manifest, verify_inputs
from app.model import check_cuda_usable, configure_torch
from common.util import read_json, sha256_file


def framework_info(torch) -> dict:
    import torchvision

    return {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()), "device_count": torch.cuda.device_count(),
        "hostname": platform.node(), "in_container": os.path.exists("/.dockerenv"),
    }


def cuda_check(torch) -> dict:
    out: dict = {"ok": False}
    try:
        props = check_cuda_usable(torch)
        a = torch.randn(64, 64, device="cuda", generator=None)
        b = a @ a
        conv = torch.nn.Conv2d(3, 4, 3).cuda()
        y = conv(torch.randn(2, 3, 16, 16, device="cuda"))
        torch.cuda.synchronize()
        out.update(
            ok=bool(b.is_cuda and y.is_cuda and y.isfinite().all().item()),
            tensor_device=str(b.device), conv_output_device=str(y.device),
            expected_device_type="cuda", **props,
        )
        if not out["ok"]:
            out["reason"] = "tensors were not on CUDA or produced non-finite values"
    except LabError as e:
        out.update(reason=f"{e.reason_code}: {e.message}")
    except Exception as e:  # noqa: BLE001
        out.update(reason=f"{type(e).__name__}: {e}")
    return out


def file_checks(s: Settings, artifacts_dir: str, min_free_gb: float) -> dict:
    checks: dict = {}

    def rec(name, ok, reason="", **extra):
        checks[name] = {"ok": bool(ok), "reason": reason, **extra}

    try:
        mm = read_json(s.model_manifest)
        wp = Path(s.model_path)
        if not wp.is_file():
            rec("model_weights", False, f"missing: {wp}")
        else:
            got = sha256_file(wp)
            rec("model_weights", got == mm["sha256"], "" if got == mm["sha256"] else f"sha256 {got[:12]} != manifest {mm['sha256'][:12]}",
                sha256=got, path=str(wp))
    except Exception as e:  # noqa: BLE001
        rec("model_weights", False, f"{type(e).__name__}: {e}")
    try:
        m = load_manifest(s.input_manifest, s.input_dir)
        verify_inputs(m)
        rec("input_manifest", True, count=len(m.ids), synthetic=m.synthetic, manifest_sha256=sha256_file(s.input_manifest))
    except LabError as e:
        rec("input_manifest", False, f"{e.reason_code}: {e.message}")
    except Exception as e:  # noqa: BLE001
        rec("input_manifest", False, f"{type(e).__name__}: {e}")
    try:
        free = shutil.disk_usage(artifacts_dir if os.path.isdir(artifacts_dir) else "/").free
        rec("disk_space", free / 1e9 >= min_free_gb, f"{free/1e9:.1f} GB free < {min_free_gb} GB required" if free / 1e9 < min_free_gb else "",
            free_bytes=free, required_gb=min_free_gb)
    except OSError as e:
        rec("disk_space", False, str(e))
    rec("manifests_readable", os.access(s.input_manifest, os.R_OK) and os.access(s.model_manifest, os.R_OK), "not readable" if not os.access(s.input_manifest, os.R_OK) else "")
    try:
        with tempfile.NamedTemporaryFile(dir=artifacts_dir, prefix=".preflight-", delete=True):
            pass
        rec("artifacts_writable", True, path=artifacts_dir)
    except OSError as e:
        rec("artifacts_writable", False, f"{artifacts_dir}: {e}")
    return {"checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--artifacts-dir", default=os.environ.get("LAB_ARTIFACTS_DIR", "/opt/lab/artifacts"))
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    a = ap.parse_args(argv)
    s = Settings.from_env()
    torch = configure_torch(s)
    res = {
        "executed_in": ("native-process" if os.environ.get("LAB_NATIVE") == "1" else
                        ("container" if os.path.exists("/.dockerenv") else "host-process")),
        "info": framework_info(torch),
        "cuda": cuda_check(torch) if a.backend == "cuda" else None,
        "files": file_checks(s, a.artifacts_dir, a.min_free_gb),
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
    }
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
