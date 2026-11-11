"""Explicitly fetch and pin the ResNet18 checkpoint (the only network download).

The service never downloads weights: this script is run once (`make weights`)
before any timed run, so startup/recovery numbers contain no hidden downloads.
Writes manifests/model.json containing the SHA256 that the service enforces.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common.util import REPO_ROOT, git_commit, iso, sha256_file, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-dir", type=Path, default=REPO_ROOT / "manifests/weights")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "manifests/model.json")
    a = ap.parse_args()

    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights

    w = ResNet18_Weights.IMAGENET1K_V1
    a.dest_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(w.url).name  # resnet18-f37072fd.pth (torchvision verifies the hash prefix)
    dest = a.dest_dir / fname
    if not dest.exists():
        w.get_state_dict(progress=True, check_hash=True)  # populates the torch hub cache
        cached = Path(torch.hub.get_dir()) / "checkpoints" / fname
        shutil.copyfile(cached, dest)
    sha = sha256_file(dest)
    write_json(
        a.manifest,
        {
            "name": "resnet18",
            "weights": "IMAGENET1K_V1",
            "file": fname,
            "sha256": sha,
            "bytes": dest.stat().st_size,
            "source_url": w.url,
            "fetched_utc": iso(),
            "fetched_with": {"torch": torch.__version__, "torchvision": torchvision.__version__},
            "commit": git_commit(),
        },
    )
    print(f"{dest} sha256={sha}")


if __name__ == "__main__":
    main()
