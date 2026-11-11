"""Test doubles. Everything here is SIMULATED: results from these fakes are never hardware evidence."""
from __future__ import annotations

import threading
import time

import numpy as np

from app.errors import LabError
from app.model import InferResult, ModelAdapter

IDS = [f"img_{i:03d}" for i in range(256)]


class FakeAdapter(ModelAdapter):
    name = "fake"

    def __init__(self, backend="cpu", delay_s=0.0, fail_load: LabError | None = None, fail_infer=False, report_backend=None):
        self.version = "fake-1"
        self.backend = backend
        self.delay_s = delay_s
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.report_backend = report_backend or backend
        self.calls = 0
        self.gate: threading.Event | None = None  # when set, infer blocks until opened

    def load(self):
        if self.fail_load:
            raise self.fail_load

    @property
    def input_ids(self):
        return IDS

    def infer(self, ids):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=20)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_infer and self.calls > 1:  # let the smoke inference succeed
            raise RuntimeError("boom")
        logits = np.zeros((len(ids), 1000), dtype=np.float32)
        for k, i in enumerate(ids):
            logits[k, int(i.split("_")[1]) % 1000] = 5.0
        return InferResult(logits, 1.0, self.report_backend)

    def postprocess(self, logits, top_k=5):
        return [[{"class_index": int(np.argmax(r)), "label": "x", "logit": float(r.max())}] for r in logits]

    def info(self):
        return {"model": "fake", "model_version": self.version, "backend": self.backend, "precision": "fp32",
                "tf32_matmul": False, "tf32_cudnn": False, "torch_version": "fake"}
