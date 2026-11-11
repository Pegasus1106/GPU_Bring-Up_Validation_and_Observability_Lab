"""Model adapter interface and the pinned ResNet18 implementation.

The adapter interface lets the MTCNN + FaceNet pipeline be added later without
touching the API, worker, metrics or benchmark code (optional extension).
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.errors import LabError
from app.inputs import InputManifest, load_manifest, load_tensors, verify_inputs
from common.util import sha256_file


@dataclass
class InferResult:
    logits: np.ndarray  # float32 [N, num_classes]
    model_ms: float  # model-only time (CUDA events on cuda, perf_counter on cpu)
    backend_actual: str  # derived from the device the output tensor lived on


class ModelAdapter(ABC):
    name: str
    version: str

    @abstractmethod
    def load(self) -> None:
        """Verify artefacts, load weights onto the requested backend, preprocess inputs.
        Raises LabError with a reason code on any failure. Must never fall back
        to a different backend."""

    @abstractmethod
    def infer(self, ids: list[str]) -> InferResult: ...

    @abstractmethod
    def info(self) -> dict: ...

    @property
    @abstractmethod
    def input_ids(self) -> list[str]: ...

    @abstractmethod
    def postprocess(self, logits: np.ndarray, top_k: int = 5) -> list[list[dict]]: ...


def configure_torch(settings: Settings):
    """Apply determinism / precision settings and return the torch module."""
    import torch

    if settings.cpu_threads > 0:
        torch.set_num_threads(settings.cpu_threads)
    # Spec: disable (or record) TF32. We disable it, so FP32 means true FP32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def check_cuda_usable(torch) -> dict:
    """Prove CUDA works by executing a real tensor op; raises LabError if not."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise LabError(
            "cuda_unavailable",
            "backend=cuda requested but torch.cuda reports no usable device "
            "(refusing to fall back to CPU)",
        )
    try:
        a = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        b = (a @ a).sum()
        torch.cuda.synchronize()
        value = float(b.item())
        expected = float((torch.arange(16, dtype=torch.float32).reshape(4, 4) @ torch.arange(16, dtype=torch.float32).reshape(4, 4)).sum())
        if a.device.type != "cuda" or abs(value - expected) > 1e-3:
            raise LabError("cuda_check_failed", f"CUDA matmul gave {value}, expected {expected}")
    except LabError:
        raise
    except Exception as e:  # driver / runtime errors surface here
        raise LabError("cuda_check_failed", f"CUDA tensor operation failed: {e}") from e
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    return {
        "device_uuid": (f"GPU-{uuid}" if uuid and not str(uuid).startswith("GPU-") else (str(uuid) if uuid else None)),
        "device_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "capability": f"{props.major}.{props.minor}",
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


class ResNet18Adapter(ModelAdapter):
    name = "resnet18"

    def __init__(self, settings: Settings):
        self.s = settings
        self.version = "unloaded"
        self._model = None
        self._tensors: dict = {}
        self._manifest: InputManifest | None = None
        self._device = None
        self._torch = None
        self._categories: list[str] = []
        self._cuda_info: dict = {}
        self._model_sha = ""

    # -- loading ---------------------------------------------------------
    def load(self) -> None:
        s = self.s
        torch = self._torch = configure_torch(s)

        # 1) model artefact + checksum (PF04)
        mm = Path(s.model_manifest)
        if not mm.is_file():
            raise LabError("model_manifest_missing", f"model manifest not found: {s.model_manifest}")
        try:
            model_meta = json.loads(mm.read_text(encoding="utf-8"))
            expected_sha = model_meta["sha256"]
        except (ValueError, KeyError) as e:
            raise LabError("model_manifest_missing", f"model manifest unreadable: {e}") from e
        wp = Path(s.model_path)
        if not wp.is_file():
            raise LabError("weights_missing", f"model weights not found at {s.model_path} (no download is attempted)")
        self._model_sha = sha256_file(wp)
        if self._model_sha != expected_sha:
            raise LabError(
                "weights_checksum_mismatch",
                f"weights SHA256 {self._model_sha[:12]}… != manifest {expected_sha[:12]}…",
            )

        # 2) inputs
        self._manifest = load_manifest(s.input_manifest, s.input_dir)
        verify_inputs(self._manifest)

        # 3) backend. CUDA is checked *before* heavy work so failure is fast and explicit.
        if s.backend == "cuda":
            self._cuda_info = check_cuda_usable(torch)
            self._device = torch.device("cuda", 0)
        elif s.backend == "cpu":
            self._device = torch.device("cpu")
        else:
            raise LabError("config_invalid", f"unknown backend {s.backend!r}")

        # 4) weights -> model
        import torchvision
        from torchvision.models import ResNet18_Weights

        try:
            state = torch.load(wp, map_location="cpu", weights_only=True)
            model = torchvision.models.resnet18(weights=None)
            model.load_state_dict(state)
        except Exception as e:
            raise LabError("weights_corrupt", f"could not load weights: {e}") from e
        model.eval()
        model.to(self._device)
        if s.channels_last:
            model.to(memory_format=torch.channels_last)
        self._model = model
        self._categories = list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
        self.version = f"resnet18-{model_meta.get('weights', 'IMAGENET1K_V1')}-{self._model_sha[:12]}"

        # 5) preprocess all allowlisted inputs once (CPU, identical for every backend)
        self._tensors = load_tensors(self._manifest)

    # -- inference -------------------------------------------------------
    @property
    def input_ids(self) -> list[str]:
        return self._manifest.ids if self._manifest else []

    def infer(self, ids: list[str]) -> InferResult:
        torch = self._torch
        batch = torch.stack([self._tensors[i] for i in ids])
        if self.s.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            if self._device.type == "cuda":
                x = batch.to(self._device, non_blocking=False)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                out = self._model(x)
                end.record()
                end.synchronize()  # explicit sync: elapsed_time is only valid after the event completed
                model_ms = float(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                out = self._model(batch)
                model_ms = (time.perf_counter() - t0) * 1000.0
            backend_actual = out.device.type
            logits = out.detach().float().cpu().numpy()
        return InferResult(logits=logits, model_ms=model_ms, backend_actual=backend_actual)

    def postprocess(self, logits: np.ndarray, top_k: int = 5) -> list[list[dict]]:
        res = []
        for row in logits:
            idx = np.argsort(-row, kind="stable")[:top_k]
            res.append(
                [
                    {"class_index": int(i), "label": self._categories[int(i)], "logit": float(row[i])}
                    for i in idx
                ]
            )
        return res

    def info(self) -> dict:
        torch = self._torch
        d = {
            "model": self.name,
            "model_version": self.version,
            "weights_sha256": self._model_sha,
            "backend": self.s.backend,
            "precision": self.s.precision,
            "input_count": len(self.input_ids),
            "inputs_synthetic": bool(self._manifest.synthetic) if self._manifest else None,
            "channels_last": self.s.channels_last,
            "torch_version": torch.__version__ if torch else None,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if torch else None,
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if torch else None,
            "cpu_threads": torch.get_num_threads() if torch else None,
        }
        if self._cuda_info:
            d["cuda"] = self._cuda_info
        return d
