"""30-minute baseline soak (GPU by default) with per-minute windows, queue/RSS trace
and Prometheus telemetry. Reports drift; it does not assert a speedup or a target.

    python -m benchmarks.soak --url http://127.0.0.1:8000 --backend cuda --batch-size 8 --concurrency 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks import aggregate as agg
from common import endpoints
from benchmarks import telemetry
from benchmarks.sampler import MetricsSampler
from common.util import REPO_ROOT, epoch_to_iso, git_commit, iso, new_run_id, write_json
from loadgen.loadgen import LoadConfig, LoadRunner


def drift(win: list[dict]) -> dict:
    """Compare first vs last quarter of the soak (ignoring a partial last window)."""
    full = win[:-1] if len(win) > 2 else win
    if len(full) < 4:
        return {"status": "insufficient_windows"}
    q = max(1, len(full) // 4)
    first, last = full[:q], full[-q:]

    def mean(ws, k):
        v = [w[k] for w in ws if w[k] is not None]
        return sum(v) / len(v) if v else None

    out = {}
    for k in ("images_per_s", "p95_ms"):
        a, b = mean(first, k), mean(last, k)
        out[k] = {"first_quarter": a, "last_quarter": b, "change_pct": ((b - a) / a * 100) if a else None}
    return {"status": "ok", **out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="default: the gpu (cuda) or cpu service, resolved via common.endpoints")
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=1800.0)
    ap.add_argument("--prometheus", default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/soak")
    a = ap.parse_args(argv)
    a.url = a.url or endpoints.url("gpu" if a.backend == "cuda" else "cpu")
    a.prometheus = a.prometheus or endpoints.url("prometheus")

    rid = new_run_id(f"soak-{a.backend}")
    d = a.out_dir / rid
    sampler = MetricsSampler(a.url, d / "service_samples.csv", interval_s=5.0).start()
    t0 = time.time()
    summary = LoadRunner(LoadConfig(a.url, a.backend, a.batch_size, a.concurrency, 20, a.duration, run_id=rid, out_dir=d)).run()
    t1 = time.time()
    sampler.stop()
    rows = agg.read_requests(summary["requests_csv"])
    whole = agg.aggregate_rows(rows, summary["measure_seconds"])
    win = agg.windows(rows, 60.0)
    rss = [r["rss_bytes"] for r in sampler.rows if r.get("rss_bytes") is not None]
    out = {
        "run_id": rid, "commit": git_commit(), "created_utc": iso(), "start_utc": epoch_to_iso(t0), "end_utc": epoch_to_iso(t1),
        "config": {"backend": a.backend, "batch_size": a.batch_size, "concurrency": a.concurrency, "duration_s": a.duration},
        "complete": a.duration >= 1800.0,
        "aggregate": whole, "windows_60s": win, "drift": drift(win),
        "rss_bytes": {"first": rss[0], "last": rss[-1], "max": max(rss)} if rss else "unavailable (no process_resident_memory_bytes; Linux only)",
        "telemetry": telemetry.collect(a.prometheus, t0, t1, step_s=15),
    }
    write_json(d / "soak.json", out)
    print(f"soak complete={out['complete']} attempts={whole['attempts']} success_rate={whole['success_rate']} -> {d/'soak.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
