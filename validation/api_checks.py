"""HTTP-level validation of a running inference service.

* `quick_validate` is the post-recovery revalidation used by the fault harness:
  fixed inputs must return top-1 classes matching the CPU FP32 reference
  (with the same margin policy as the full correctness run) on the expected backend.
* `contract_checks` exercises the API contract: invalid IDs, oversized batches,
  malformed bodies, real backend reporting, batch vs individual agreement,
  and bounded metric labels.

    python -m validation.api_checks --url http://127.0.0.1:8000 --expect-backend cuda
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

from common.util import REPO_ROOT, git_commit, iso, new_run_id, write_json
from validation.reference import load_reference, load_tolerances, top1_disagreement_allowed

QUICK_IDS = [f"img_{i:03d}" for i in range(0, 256, 16)]  # 16 spread-out fixed inputs


def _post(url: str, ids, timeout=15, headers=None):
    return requests.post(f"{url}/infer", json={"ids": ids}, timeout=timeout, headers=headers or {})


def quick_validate(url: str, expect_backend: str, ids: list[str] | None = None, ref=None) -> dict:
    """Return {"passed": bool, ...}. Never raises on service failure (reports it)."""
    ids = ids or QUICK_IDS
    t0 = time.monotonic()
    try:
        ref = ref or load_reference()
        tol = load_tolerances()
        r = _post(url, ids)
        if r.status_code != 200:
            return {"passed": False, "reason": f"HTTP {r.status_code}: {r.text[:200]}", "seconds": time.monotonic() - t0}
        body = r.json()
        if body["backend"] != expect_backend:
            return {"passed": False, "reason": f"backend {body['backend']} != expected {expect_backend}"}
        bad = []
        for p in body["predictions"]:
            k = ref.row(p["id"])
            got = p["top5"][0]["class_index"]
            if got != ref.top1[k]:
                logit = p["top5"][0]["logit"]
                if not top1_disagreement_allowed(ref.margin[k], logit, tol["atol"], tol["rtol"]):
                    bad.append(p["id"])
        return {
            "passed": not bad,
            "reason": "" if not bad else f"top-1 mismatch on {bad}",
            "n": len(ids),
            "backend": body["backend"],
            "model_version": body["model_version"],
            "seconds": time.monotonic() - t0,
        }
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "reason": f"{type(e).__name__}: {e}", "seconds": time.monotonic() - t0}


def contract_checks(url: str, expect_backend: str) -> list[dict]:
    res: list[dict] = []

    def add(cid, name, ok, detail=""):
        res.append({"id": cid, "name": name, "status": "PASS" if ok else "FAIL", "reason": "" if ok else detail})

    def safe(cid, name, fn):
        try:
            ok, detail = fn()
            add(cid, name, ok, detail)
        except Exception as e:  # noqa: BLE001
            res.append({"id": cid, "name": name, "status": "ERROR", "reason": f"{type(e).__name__}: {e}"})

    safe("A01", "GET /livez is 200", lambda: ((r := requests.get(f"{url}/livez", timeout=5)).status_code == 200, r.text))
    safe("A02", "GET /readyz is 200", lambda: ((r := requests.get(f"{url}/readyz", timeout=5)).status_code == 200, r.text[:200]))

    def valid():
        r = _post(url, ["img_000", "img_001"], headers={"X-Request-ID": "contract-check-1"})
        b = r.json()
        ok = (
            r.status_code == 200 and b["backend"] == expect_backend and b["request_id"] == "contract-check-1"
            and len(b["predictions"]) == 2 and b["model_version"]
        )
        return ok, f"status={r.status_code} backend={b.get('backend')}"

    safe("A03", "valid request returns real backend, model version, request id", valid)
    cases = [
        ("A04", "unknown id -> 422", ["nope"]),
        ("A05", "path-traversal id -> 422", ["../../etc/passwd"]),
        ("A06", "empty list -> 422", []),
        ("A07", "33 ids -> 422", [f"img_{i:03d}" for i in range(33)]),
    ]
    for cid, name, ids in cases:
        safe(cid, name, lambda ids=ids: ((r := _post(url, ids)).status_code == 422, f"got {r.status_code}"))
    safe(
        "A08", "malformed body -> 422",
        lambda: ((r := requests.post(f"{url}/infer", data="{bad", headers={"Content-Type": "application/json"}, timeout=5)).status_code == 422, f"got {r.status_code}"),
    )

    def batch_vs_single():
        ids = [f"img_{i:03d}" for i in range(8)]
        b = _post(url, ids).json()["predictions"]
        s = [_post(url, [i]).json()["predictions"][0] for i in ids]
        diff = [x["id"] for x, y in zip(b, s) if x["top5"][0]["class_index"] != y["top5"][0]["class_index"]]
        return not diff, f"top-1 differs for {diff}"

    safe("A09", "batch and individual top-1 agree", batch_vs_single)

    def labels_bounded():
        txt = requests.get(f"{url}/metrics", timeout=5).text
        forbidden = re.findall(r'(request_id|run_id)=', txt)
        return not forbidden, f"unbounded label(s) present: {set(forbidden)}"

    safe("A10", "metrics expose no request/run id labels", labels_bounded)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-backend", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/validation")
    a = ap.parse_args(argv)
    checks = contract_checks(a.url, a.expect_backend)
    quick = quick_validate(a.url, a.expect_backend)
    ok = all(c["status"] == "PASS" for c in checks) and quick["passed"]
    rep = {
        "run_id": new_run_id(f"api-{a.expect_backend}"), "created_utc": iso(), "commit": git_commit(),
        "url": a.url, "expect_backend": a.expect_backend, "contract_checks": checks, "quick_validate": quick, "passed": ok,
    }
    path = a.out_dir / f"{rep['run_id']}.json"
    write_json(path, rep)
    for c in checks:
        print(f"[{c['status']:5}] {c['id']} {c['name']} {c['reason']}")
    print(f"quick_validate: {quick}\n-> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
