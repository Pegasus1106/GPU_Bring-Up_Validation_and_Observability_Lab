"""Loading CPU FP32 reference outputs and comparing logits against them."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.util import REPO_ROOT, read_json

REF_DIR = REPO_ROOT / "manifests/reference"
TOL_PATH = REPO_ROOT / "validation/tolerances.json"


@dataclass
class Reference:
    ids: list[str]
    logits: np.ndarray
    top1: list[int]
    margin: list[float]
    meta: dict

    def row(self, iid: str) -> int:
        return self.index[iid]

    def __post_init__(self):
        self.index = {i: k for k, i in enumerate(self.ids)}


def load_reference(ref_dir: Path = REF_DIR) -> Reference:
    meta = read_json(ref_dir / "reference.json")
    logits = np.load(ref_dir / meta["logits_file"])
    return Reference(meta["ids"], logits, meta["top1"], meta["top1_margin"], meta)


def load_tolerances(path: Path = TOL_PATH) -> dict:
    return read_json(path)


def compare(actual: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> dict:
    """Per-sample comparison. Element passes if |a-r| <= atol + rtol*|r|."""
    diff = np.abs(actual.astype(np.float64) - ref.astype(np.float64))
    bound = atol + rtol * np.abs(ref.astype(np.float64))
    max_abs = float(diff.max())
    denom = float(np.abs(ref).max()) or 1.0
    return {
        "max_abs_error": max_abs,
        "normalized_error": max_abs / denom,  # max|d| / max|ref| for this sample
        "within_tolerance": bool((diff <= bound).all()),
        "worst_excess": float((diff - bound).max()),
        "top1_actual": int(actual.argmax()),
        "top1_ref": int(ref.argmax()),
    }


def top1_disagreement_allowed(margin: float, top1_logit: float, atol: float, rtol: float) -> bool:
    """A flip is explainable only if the reference decision was inside the tolerance band."""
    return margin <= 2.0 * (atol + rtol * abs(top1_logit))
