"""Service configuration, read from environment variables (LAB_*)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Batch sizes that may appear as a Prometheus label. Anything else is "other"
# so the label set stays bounded (spec: bounded labels only).
LABEL_BATCH_SIZES = (1, 8, 32)
MAX_BATCH = 32


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    backend: str = "cpu"  # "cpu" | "cuda"; cuda never falls back to cpu
    workload: str = "resnet18"
    precision: str = "fp32"
    model_path: str = str(REPO_ROOT / "manifests/weights/resnet18-f37072fd.pth")
    model_manifest: str = str(REPO_ROOT / "manifests/model.json")
    input_dir: str = str(REPO_ROOT / "manifests/inputs")
    input_manifest: str = str(REPO_ROOT / "manifests/inputs.json")
    max_pending: int = 32
    request_deadline_s: float = 10.0
    cpu_threads: int = 0  # 0 = leave torch default
    channels_last: bool = False  # controlled optimisation knob (off in baseline)
    stuck_after_s: float = 30.0  # worker busy on one job longer than this => not ready
    fault_dir: str = ""
    enable_fault_injection: bool = False
    log_file: str = ""
    smoke_ids: tuple[str, ...] = field(default=("img_000",))

    @staticmethod
    def from_env() -> "Settings":
        d = Settings()
        return Settings(
            backend=os.environ.get("LAB_BACKEND", d.backend).lower(),
            workload=os.environ.get("LAB_WORKLOAD", d.workload),
            precision=os.environ.get("LAB_PRECISION", d.precision).lower(),
            model_path=os.environ.get("LAB_MODEL_PATH", d.model_path),
            model_manifest=os.environ.get("LAB_MODEL_MANIFEST", d.model_manifest),
            input_dir=os.environ.get("LAB_INPUT_DIR", d.input_dir),
            input_manifest=os.environ.get("LAB_INPUT_MANIFEST", d.input_manifest),
            max_pending=_env_int("LAB_MAX_PENDING", d.max_pending),
            request_deadline_s=_env_float("LAB_REQUEST_DEADLINE_S", d.request_deadline_s),
            cpu_threads=_env_int("LAB_CPU_THREADS", d.cpu_threads),
            channels_last=_env_bool("LAB_CHANNELS_LAST", d.channels_last),
            stuck_after_s=_env_float("LAB_STUCK_AFTER_S", d.stuck_after_s),
            fault_dir=os.environ.get("LAB_FAULT_DIR", d.fault_dir),
            enable_fault_injection=_env_bool("LAB_ENABLE_FAULT_INJECTION", d.enable_fault_injection),
            log_file=os.environ.get("LAB_LOG_FILE", d.log_file),
        )

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty list == valid)."""
        problems = []
        if self.backend not in ("cpu", "cuda"):
            problems.append(f"LAB_BACKEND must be 'cpu' or 'cuda', got {self.backend!r}")
        if self.precision != "fp32":
            problems.append(f"only fp32 is supported in version one, got {self.precision!r}")
        if not (1 <= self.max_pending <= 1024):
            problems.append("LAB_MAX_PENDING must be in 1..1024")
        if self.request_deadline_s <= 0:
            problems.append("LAB_REQUEST_DEADLINE_S must be > 0")
        return problems
