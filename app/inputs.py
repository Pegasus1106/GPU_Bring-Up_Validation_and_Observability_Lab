"""Allowlisted fixed inputs: manifest parsing, checksum verification and
deterministic preprocessing. Only IDs present in the manifest can be requested,
so no client-supplied path ever reaches the filesystem."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.errors import LabError
from common.util import sha256_file

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@dataclass(frozen=True)
class InputManifest:
    items: dict[str, dict]  # id -> {"file","sha256",...}
    preprocessing: dict
    synthetic: bool
    base_dir: Path
    raw: dict

    @property
    def ids(self) -> list[str]:
        return list(self.items.keys())


def load_manifest(manifest_path: str, input_dir: str, preprocessing_path: str | None = None) -> InputManifest:
    mp = Path(manifest_path)
    if not mp.is_file():
        raise LabError("inputs_missing", f"input manifest not found: {manifest_path}")
    try:
        raw = json.loads(mp.read_text(encoding="utf-8"))
        items = {it["id"]: it for it in raw["items"]}
    except (KeyError, ValueError, TypeError) as e:
        raise LabError("inputs_missing", f"input manifest unreadable: {e}") from e
    for iid in items:
        if not ID_RE.match(iid):
            raise LabError("config_invalid", f"manifest contains an unsafe id: {iid!r}")
    pp = preprocessing_path or str(mp.parent / raw.get("preprocessing_file", "preprocessing.json"))
    try:
        preprocessing = json.loads(Path(pp).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LabError("config_invalid", f"preprocessing config unreadable ({pp}): {e}") from e
    return InputManifest(
        items=items,
        preprocessing=preprocessing,
        synthetic=bool(raw.get("synthetic", False)),
        base_dir=Path(input_dir),
        raw=raw,
    )


def verify_inputs(m: InputManifest) -> None:
    """PF04: every file present and matching its recorded SHA256."""
    missing, bad = [], []
    for iid, it in m.items.items():
        p = m.base_dir / it["file"]
        if not p.is_file():
            missing.append(iid)
        elif sha256_file(p) != it["sha256"]:
            bad.append(iid)
    if missing:
        raise LabError("inputs_missing", f"{len(missing)} input files missing, first: {missing[:3]}")
    if bad:
        raise LabError("inputs_checksum_mismatch", f"{len(bad)} input files fail SHA256, first: {bad[:3]}")


def build_transform(pp: dict):
    from torchvision import transforms as T

    interp = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}[
        pp.get("interpolation", "bilinear")
    ]
    return T.Compose(
        [
            T.Resize(pp["resize"], interpolation=interp),
            T.CenterCrop(pp["center_crop"]),
            T.ToTensor(),
            T.Normalize(mean=pp["mean"], std=pp["std"]),
        ]
    )


def load_tensors(m: InputManifest, ids: list[str] | None = None):
    """Preprocess inputs on CPU, identically for every backend.

    Returns a dict id -> float32 tensor [3,H,W]. Preprocessing runs once at load
    time, so request latency measures inference, not image decoding.
    """
    from PIL import Image

    tf = build_transform(m.preprocessing)
    out = {}
    for iid in ids or m.ids:
        with Image.open(m.base_dir / m.items[iid]["file"]) as im:
            out[iid] = tf(im.convert("RGB")).contiguous()
    return out
