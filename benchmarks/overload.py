"""Overload test (AC08): prove pending work and memory stay bounded and that
rejections / timeouts are explicit.

Drives far more concurrent clients than the service can admit
(concurrency > max_pending + 1), samples queue depth and RSS every second, then
evaluates: max queue depth <= capacity, rejection/timeout counts present, RSS growth
slope after the ramp below a proposed threshold, service recovers to ready and
passes quick validation afterwards.

    python -m benchmarks.overload --url http://127.0.0.1:8000 --backend cuda
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import requests

from benchmarks import aggregate as agg
from common import endpoints
from benchmarks.sampler import MetricsSampler
from common.util import REPO_ROOT, git_commit, iso, new_run_id, write_json
from loadgen.loadgen import LoadConfig, LoadRunner
from validation.api_checks import quick_validate

RSS_SLOPE_LIMIT_MB_PER_MIN = 10.0  # proposed target; record changes with rationale


def rss_slope_mb_per_min(rows: list[dict], skip_fraction: float = 0.4) -> float | None:
    pts = [(r["t_mono"], r["rss_bytes"]) for r in rows if r.get("rss_bytes") is not None]
    pts = pts[int(len(pts) * skip_fraction):]  # ignore the ramp-up
    if len(pts) < 5:
        return None
    t = np.array([p[0] for p in pts]) / 60.0
    m = np.array([p[1] for p in pts]) / 1e6
    return float(np.polyfit(t, m, 1)[0])


def evaluate(rows: list[dict], capacity: int, counts: dict, slope_limit: float = RSS_SLOPE_LIMIT_MB_PER_MIN) -> dict:
    depths = [r["queue_depth"] for r in rows if r.get("queue_depth") is not None]
    slope = rss_slope_mb_per_min(rows)
    checks = {
        "queue_bounded": {"max_observed": max(depths) if depths else None, "capacity": capacity,
                          "pass": bool(depths) and max(depths) <= capacity},
        "explicit_rejections_or_timeouts": {"rejected": counts.get("rejected", 0),
                                            "timeout": counts.get("timeout", 0) + counts.get("client_timeout", 0),
                                            "pass": (counts.get("rejected", 0) + counts.get("timeout", 0) + counts.get("client_timeout", 0)) > 0},
        "memory_not_growing": {"slope_mb_per_min": slope, "limit": slope_limit,
                               "pass": slope is not None and slope <= slope_limit,
                               "note": "unavailable (no RSS samples)" if slope is None else ""},
    }
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="default: the gpu (cuda) or cpu service, resolved via common.endpoints")
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/overload")
    a = ap.parse_args(argv)
    a.url = a.url or endpoints.url("gpu" if a.backend == "cuda" else "cpu")

    rid = new_run_id(f"overload-{a.backend}")
    d = a.out_dir / rid
    info = requests.get(f"{a.url}/readyz", timeout=5).json()
    cap_m = re.search(r"inference_queue_capacity\{[^}]*\} (\d+)", requests.get(f"{a.url}/metrics", timeout=5).text)
    if not cap_m:
        raise SystemExit("cannot read inference_queue_capacity from /metrics")
    capacity = int(cap_m.group(1))
    sampler = MetricsSampler(a.url, d / "service_samples.csv", interval_s=1.0).start()
    time.sleep(3)
    before = sampler.rows[-1] if sampler.rows else None
    s = LoadRunner(LoadConfig(a.url, a.backend, a.batch_size, a.concurrency, 0, a.duration, run_id=rid, out_dir=d)).run()
    time.sleep(5)  # drain
    sampler.stop()
    after = sampler.rows[-1] if sampler.rows else None
    ready_after = False
    for _ in range(30):
        try:
            ready_after = requests.get(f"{a.url}/readyz", timeout=3).status_code == 200
        except Exception:  # noqa: BLE001
            pass
        if ready_after:
            break
        time.sleep(1)
    qv = quick_validate(a.url, a.backend)
    checks = evaluate(sampler.rows, capacity, s["counts"])
    checks["recovered_and_valid"] = {"ready": ready_after, "quick_validate": qv, "pass": ready_after and qv["passed"]}
    out = {
        "run_id": rid, "commit": git_commit(), "created_utc": iso(), "service_info": info,
        "config": {"concurrency": a.concurrency, "batch_size": a.batch_size, "duration_s": a.duration},
        "aggregate": agg.aggregate_file(s["requests_csv"], s["measure_seconds"]),
        "before": before, "after": after, "checks": checks,
        "passed": all(c["pass"] for c in checks.values()),
    }
    write_json(d / "overload.json", out)
    for k, v in checks.items():
        print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}: { {x: y for x, y in v.items() if x != 'pass'} }")
    print(f"-> {d/'overload.json'}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
