"""Aggregate raw requests.csv files. Pure functions, unit-tested.

Percentile method: linear interpolation between closest ranks (numpy default,
"type 7"), computed over *successful* requests' client end-to-end time.
Reliability rates use *all* measured attempts in the denominator.
Image throughput uses the batch sizes of requests that actually completed.
"""
from __future__ import annotations

import csv
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

PERCENTILE_METHOD = "linear interpolation between closest ranks (numpy.percentile, method='linear')"
MIN_SUCCESSES_FOR_P99 = 1000
ERROR_OUTCOMES = {"error", "connection_error", "not_ready", "invalid"}
TIMEOUT_OUTCOMES = {"timeout", "client_timeout"}


def read_requests(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(values: Iterable[float], q: float) -> float | None:
    v = list(values)
    if not v:
        return None
    return float(np.percentile(np.asarray(v, dtype=np.float64), q, method="linear"))


def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def aggregate_rows(rows: list[dict], measure_seconds: float | None = None) -> dict:
    meas = [r for r in rows if r["phase"] == "measure"]
    attempts = len(meas)
    outcomes: dict[str, int] = {}
    for r in meas:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    ok = [r for r in meas if r["outcome"] == "success"]
    e2e = [float(r["e2e_ms"]) for r in ok]
    model = [v for v in (_f(r["model_ms"]) for r in ok) if v is not None]
    queue = [v for v in (_f(r["queue_ms"]) for r in ok) if v is not None]

    if measure_seconds is None and meas:
        measure_seconds = max(_ts(r["t_end_utc"]) for r in meas) - min(_ts(r["t_start_utc"]) for r in meas)
    images = sum(int(r["batch_size"]) for r in ok)
    n_err = sum(outcomes.get(o, 0) for o in ERROR_OUTCOMES)
    n_to = sum(outcomes.get(o, 0) for o in TIMEOUT_OUTCOMES)
    n_rej = outcomes.get("rejected", 0)
    den = attempts or None
    return {
        "attempts": attempts,
        "successes": len(ok),
        "outcomes": outcomes,
        "measure_seconds": measure_seconds,
        "requests_per_s": (len(ok) / measure_seconds) if measure_seconds else None,
        "images_per_s": (images / measure_seconds) if measure_seconds else None,
        "success_rate": (len(ok) / den) if den else None,
        "error_rate": (n_err / den) if den else None,
        "timeout_rate": (n_to / den) if den else None,
        "rejection_rate": (n_rej / den) if den else None,
        "e2e_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95), "p99": pct(e2e, 99), "mean": float(np.mean(e2e)) if e2e else None},
        "model_ms": {"p50": pct(model, 50), "p95": pct(model, 95)},
        "queue_ms": {"p50": pct(queue, 50), "p95": pct(queue, 95)},
        "p99_weak_evidence": len(ok) < MIN_SUCCESSES_FOR_P99,
        "percentile_method": PERCENTILE_METHOD,
    }


def aggregate_file(path: str | Path, measure_seconds: float | None = None) -> dict:
    return aggregate_rows(read_requests(path), measure_seconds)


def variability(values: list[float | None]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0, "mean": None, "stdev": None, "cv": None, "min": None, "max": None}
    mean = statistics.fmean(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return {"n": len(v), "mean": mean, "stdev": sd, "cv": (sd / mean) if mean else None, "min": min(v), "max": max(v)}


def cell_summary(trials: list[dict]) -> dict:
    """trials: list of aggregate_rows() results for one matrix cell."""
    return {
        "trials": len(trials),
        "images_per_s": variability([t["images_per_s"] for t in trials]),
        "requests_per_s": variability([t["requests_per_s"] for t in trials]),
        "p50_ms": variability([t["e2e_ms"]["p50"] for t in trials]),
        "p95_ms": variability([t["e2e_ms"]["p95"] for t in trials]),
        "p99_ms": variability([t["e2e_ms"]["p99"] for t in trials]),
        "success_rate": variability([t["success_rate"] for t in trials]),
        "error_rate": variability([t["error_rate"] for t in trials]),
        "timeout_rate": variability([t["timeout_rate"] for t in trials]),
        "rejection_rate": variability([t["rejection_rate"] for t in trials]),
        "total_attempts": sum(t["attempts"] for t in trials),
        "total_successes": sum(t["successes"] for t in trials),
        "p99_weak_evidence": any(t["p99_weak_evidence"] for t in trials),
    }


def windows(rows: list[dict], width_s: float = 60.0) -> list[dict]:
    """Per-window aggregates over the measured phase (used for the soak drift check)."""
    meas = [r for r in rows if r["phase"] == "measure"]
    if not meas:
        return []
    t0 = min(_ts(r["t_start_utc"]) for r in meas)
    buckets: dict[int, list[dict]] = {}
    for r in meas:
        buckets.setdefault(int((_ts(r["t_start_utc"]) - t0) // width_s), []).append(r)
    out = []
    for k in sorted(buckets):
        a = aggregate_rows(buckets[k], measure_seconds=width_s)
        out.append({"window": k, "start_offset_s": k * width_s, **{x: a[x] for x in ("attempts", "successes", "images_per_s", "error_rate", "timeout_rate", "rejection_rate")}, "p95_ms": a["e2e_ms"]["p95"], "p50_ms": a["e2e_ms"]["p50"]})
    return out


def compare(base: dict, variant: dict, higher_is_better: bool, name: str) -> dict:
    """Relative change variant vs base; reports regressions explicitly."""
    if base is None or variant is None or not base:
        return {"metric": name, "base": base, "variant": variant, "change_pct": None, "verdict": "insufficient_data"}
    change = (variant - base) / base * 100.0
    better = change > 0 if higher_is_better else change < 0
    verdict = "no_change" if math.isclose(change, 0.0, abs_tol=1e-9) else ("improvement" if better else "regression")
    return {"metric": name, "base": base, "variant": variant, "change_pct": change, "verdict": verdict}
