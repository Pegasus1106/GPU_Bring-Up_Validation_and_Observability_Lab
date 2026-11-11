"""Write Prometheus file_sd target files for the active profiles.

    python deploy/set_targets.py gpu          # inference-gpu + dcgm-exporter
    python deploy/set_targets.py cpu
    python deploy/set_targets.py both
    python deploy/set_targets.py add-fault / remove-fault

Only configured targets are monitored, so "profile not deployed" is not an outage.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common import endpoints  # noqa: E402

# Native mode (Sol) writes into its own run directory (LAB_TARGETS_DIR) and uses runtime-chosen ports.
T = Path(os.environ.get("LAB_TARGETS_DIR") or REPO / "monitoring/targets")
_P = endpoints.ports()

GPU = {"targets": [f"127.0.0.1:{_P['gpu']}"], "labels": {"profile": "gpu", "case": "baseline"}}
CPU = {"targets": [f"127.0.0.1:{_P['cpu']}"], "labels": {"profile": "cpu", "case": "baseline"}}
FAULT = {"targets": [f"127.0.0.1:{_P['fault']}"], "labels": {"profile": "fault", "case": "fault"}}
DCGM = {"targets": [f"127.0.0.1:{_P['gpu_exporter']}"], "labels": {"profile": "gpu"}}


def write(name: str, groups: list[dict]) -> None:
    T.mkdir(parents=True, exist_ok=True)
    (T / name).write_text(yaml.safe_dump(groups))


def read(name: str) -> list[dict]:
    p = T / name
    return yaml.safe_load(p.read_text()) or [] if p.exists() else []


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("gpu", "cpu", "both", "add-fault", "remove-fault"):
        print(__doc__)
        return 2
    mode = argv[1]
    if mode in ("gpu", "cpu", "both"):
        keep_fault = [g for g in read("inference.yml") if g["labels"].get("case") == "fault"]
        write("inference.yml", ([GPU] if mode in ("gpu", "both") else []) + ([CPU] if mode in ("cpu", "both") else []) + keep_fault)
        write("dcgm.yml", [DCGM] if mode in ("gpu", "both") else [])
    elif mode == "add-fault":
        cur = [g for g in read("inference.yml") if g["labels"].get("case") != "fault"]
        write("inference.yml", cur + [FAULT])
    else:
        write("inference.yml", [g for g in read("inference.yml") if g["labels"].get("case") != "fault"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
