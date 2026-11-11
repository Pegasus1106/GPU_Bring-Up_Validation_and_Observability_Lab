"""Service endpoints.

Docker mode uses the fixed default ports. Native mode (`LAB_NATIVE=1`, used on ASU Sol where
there is no Docker and nodes are shared with other users) picks free ports at start-up and records
them in `<native dir>/ports.json`; every tool then resolves its URLs through this module, so no
port is hard-coded outside the defaults below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from common.util import REPO_ROOT

DEFAULT_PORTS = {
    "gpu": 8000,
    "cpu": 8001,
    "fault": 8010,
    "prometheus": 9090,
    "alertmanager": 9093,
    "receiver": 9094,
    "node_exporter": 9100,
    "controller": 9105,
    "gpu_exporter": 9400,
    "grafana": 3000,
}


def native() -> bool:
    return os.environ.get("LAB_NATIVE") == "1"


def native_dir() -> Path:
    return Path(os.environ.get("LAB_NATIVE_DIR") or REPO_ROOT / "artifacts/native")


def ports_file() -> Path:
    return Path(os.environ.get("LAB_PORTS_FILE") or native_dir() / "ports.json")


def ports() -> dict[str, int]:
    p = dict(DEFAULT_PORTS)
    if native() and ports_file().exists():
        try:
            p.update({k: int(v) for k, v in json.loads(ports_file().read_text()).items()})
        except (ValueError, OSError):
            pass
    return p


def port(name: str) -> int:
    return ports()[name]


def url(name: str) -> str:
    return f"http://127.0.0.1:{port(name)}"
