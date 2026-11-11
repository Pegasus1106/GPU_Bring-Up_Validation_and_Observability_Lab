"""Generate CPU FP32 reference logits from the *pinned environment*.

Run it inside the workload image (`make reference`) so the reference is tied to
the tested torch/torchvision build. Output:
    manifests/reference/cpu_fp32_logits.npy   float32 [N,1000], one row per input id (batch=1)
    manifests/reference/reference.json        ids, top-1, top-1 margin, environment, SHA256s
"""
from __future__ import annotations

import argparse
import dataclasses
import platform
from pathlib import Path

import numpy as np

from app.config import Settings
from app.model import ResNet18Adapter
from common.util import REPO_ROOT, git_commit, iso, sha256_file, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "manifests/reference")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    s = dataclasses.replace(Settings.from_env(), backend="cpu", cpu_threads=a.threads)
    ad = ResNet18Adapter(s)
    ad.load()
    ids = ad.input_ids
    rows = [ad.infer([i]).logits[0] for i in ids]
    logits = np.stack(rows).astype(np.float32)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    npy = a.out_dir / "cpu_fp32_logits.npy"
    np.save(npy, logits)
    top2 = np.sort(logits, axis=1)[:, -2:]
    write_json(
        a.out_dir / "reference.json",
        {
            "created_utc": iso(),
            "commit": git_commit(),
            "backend": "cpu",
            "precision": "fp32",
            "batch_size": 1,
            "ids": ids,
            "top1": [int(x) for x in logits.argmax(1)],
            "top1_margin": [float(m) for m in (top2[:, 1] - top2[:, 0])],
            "logits_file": npy.name,
            "logits_sha256": sha256_file(npy),
            "model": ad.info(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    )
    print(f"reference for {len(ids)} inputs -> {a.out_dir}")


if __name__ == "__main__":
    main()
