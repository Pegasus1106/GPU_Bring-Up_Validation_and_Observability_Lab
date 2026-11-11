"""Run the baseline benchmark matrix and write raw evidence + summary.json.

    python -m benchmarks.run_matrix --config benchmarks/matrix.yaml
    python -m benchmarks.run_matrix --resume --experiment-id baseline-v1-20261110
    python -m benchmarks.run_matrix --summarize-only --experiment-id <id>

Cells whose backend service is absent/not ready are recorded as *unsupported*
with the reason; they are never silently dropped.
"""
from __future__ import annotations

import argparse
import itertools
import os
import platform
import random
import time
from pathlib import Path

import requests
import yaml

from benchmarks import aggregate as agg
from benchmarks import telemetry
from common import endpoints
from common.util import REPO_ROOT, epoch_to_iso, git_commit, iso, new_run_id, read_json, write_json
from loadgen.loadgen import LoadConfig, LoadRunner

ART = REPO_ROOT / "artifacts/benchmark"


def cell_id(backend: str, bs: int, c: int) -> str:
    return f"{backend}-b{bs}-c{c}"


def host_controls() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        extra = {"cpu_count_logical": psutil.cpu_count(), "mem_total_bytes": vm.total, "mem_available_bytes": vm.available,
                 "cpu_percent_1s": psutil.cpu_percent(interval=1.0)}
    except Exception:  # noqa: BLE001
        extra = {}
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    return {"loadavg_1_5_15": load, "platform": platform.platform(), **extra}


def readiness(url: str) -> tuple[bool, dict]:
    try:
        r = requests.get(f"{url}/readyz", timeout=5)
        return r.status_code == 200, r.json()
    except Exception as e:  # noqa: BLE001
        return False, {"reason": f"{type(e).__name__}: {e}"}


def build_plan(cfg: dict, backends: list[str]) -> list[dict]:
    rng = random.Random(cfg["seed"])
    cells = [(b, bs, c) for b in backends for bs in cfg["batch_sizes"] for c in cfg["concurrency"]]
    plan = []
    for trial in range(1, cfg["trials"] + 1):
        order = cells[:]
        rng.shuffle(order)  # randomised within each trial round so drift is not confounded with a cell
        for b, bs, c in order:
            plan.append({"backend": b, "batch_size": bs, "concurrency": c, "trial": trial})
    for i, p in enumerate(plan):
        p["order_index"] = i
    return plan


def run_one(cfg: dict, exp_dir: Path, p: dict, info: dict) -> dict:
    be = cfg["backends"][p["backend"]]
    cid = cell_id(p["backend"], p["batch_size"], p["concurrency"])
    rid = f"{cid}-t{p['trial']}"
    rdir = exp_dir / "runs" / rid
    rdir.mkdir(parents=True, exist_ok=True)
    controls = host_controls()
    lc = LoadConfig(
        url=be["url"], backend=be["expect"], batch_size=p["batch_size"], concurrency=p["concurrency"],
        warmup_requests=cfg["warmup_requests"], duration_s=cfg["duration_s"], run_id=rid, out_dir=rdir,
    )
    t_start = time.time()
    summary = LoadRunner(lc).run()
    t_end = time.time()
    a = agg.aggregate_file(summary["requests_csv"], summary["measure_seconds"])
    tel = telemetry.collect(cfg["prometheus"], t_start, t_end)
    write_json(rdir / "telemetry.json", tel)
    meta = {
        "run_id": rid, "cell": cid, **p, "start_utc": epoch_to_iso(t_start), "end_utc": epoch_to_iso(t_end),
        "controls": controls, "service_info": info, "load_config": {
            "batch_size": lc.batch_size, "concurrency": lc.concurrency, "warmup_requests": lc.warmup_requests,
            "duration_s": lc.duration_s, "client_timeout_s": lc.timeout_s},
        "loadgen_placement": "same host, separate process (loopback)",
        "aggregate": a,
        "telemetry_available": {k: v["available"] for k, v in tel["fields"].items()},
    }
    write_json(rdir / "meta.json", meta)
    return meta


def summarize(exp_dir: Path) -> dict:
    cfg = read_json(exp_dir / "config.json")
    plan = read_json(exp_dir / "plan.json")
    unsupported = read_json(exp_dir / "unsupported.json") if (exp_dir / "unsupported.json").exists() else {}
    runs: dict[str, list[dict]] = {}
    for mp in sorted((exp_dir / "runs").glob("*/meta.json")):
        m = read_json(mp)
        runs.setdefault(m["cell"], []).append(m)

    cells, missing = {}, []
    expected = {cell_id(b, bs, c) for b in cfg["backends"] for bs in cfg["batch_sizes"] for c in cfg["concurrency"]}
    for cid in sorted(expected):
        ms = runs.get(cid, [])
        if cid in unsupported:
            missing.append({"cell": cid, "status": "unsupported", "reason": unsupported[cid]})
            continue
        if len(ms) < cfg["trials"]:
            missing.append({"cell": cid, "status": "incomplete", "reason": f"{len(ms)}/{cfg['trials']} trials present"})
        if ms:
            cells[cid] = {
                **agg.cell_summary([m["aggregate"] for m in ms]),
                "per_trial": [{"trial": m["trial"], "order_index": m["order_index"], "run_id": m["run_id"], **m["aggregate"]} for m in ms],
                "telemetry_available": {k: all(m["telemetry_available"].get(k, False) for m in ms) for k in ms[0]["telemetry_available"]},
            }

    comparisons = []
    for bs in cfg["batch_sizes"]:
        for c in cfg["concurrency"]:
            cc, gc = cells.get(cell_id("cpu", bs, c)), cells.get(cell_id("gpu", bs, c))
            if not (cc and gc):
                continue
            comparisons.append({
                "batch_size": bs, "concurrency": c,
                "images_per_s": agg.compare(cc["images_per_s"]["mean"], gc["images_per_s"]["mean"], True, "images_per_s (gpu vs cpu)"),
                "p95_ms": agg.compare(cc["p95_ms"]["mean"], gc["p95_ms"]["mean"], False, "p95 latency ms (gpu vs cpu)"),
            })
    total_trials = sum(len(v) for v in runs.values())
    summary = {
        "experiment": cfg["experiment"], "experiment_id": exp_dir.name, "commit": git_commit(),
        "experiment_matrix": {k: cfg[k] for k in ("backends", "batch_sizes", "concurrency", "trials", "warmup_requests", "duration_s", "seed")},
        "expected_cells": len(expected), "expected_runs": len(expected) * cfg["trials"],
        "completed_runs": total_trials, "completed_cells": len(cells),
        "sample_counts": {cid: {"attempts": c["total_attempts"], "successes": c["total_successes"]} for cid, c in cells.items()},
        "percentile_method": agg.PERCENTILE_METHOD,
        "p99_note": f"p99 is weak evidence when a run has fewer than {agg.MIN_SUCCESSES_FOR_P99} successful requests",
        "cells": cells, "missing": missing, "comparisons": comparisons,
        "trial_order_file": "plan.json",
    }
    write_json(exp_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "benchmarks/matrix.yaml")
    ap.add_argument("--experiment-id", default="")
    ap.add_argument("--backends", nargs="*", default=None, help="subset of backends in the config")
    ap.add_argument("--duration", type=float, default=None, help="override duration_s (NOT acceptance-valid below 60)")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text())
    if endpoints.native():  # Sol: ports are chosen at start-up
        cfg["backends"]["cpu"]["url"] = endpoints.url("cpu")
        cfg["backends"]["gpu"]["url"] = endpoints.url("gpu")
        cfg["prometheus"] = endpoints.url("prometheus")
    overridden = {}
    for k, v in (("duration_s", a.duration), ("trials", a.trials), ("batch_sizes", a.batch_sizes), ("concurrency", a.concurrency)):
        if v:
            cfg[k] = v
            overridden[k] = v
    if a.backends:
        cfg["backends"] = {b: cfg["backends"][b] for b in a.backends}
        overridden["backends"] = a.backends
    cfg["overrides"] = overridden  # recorded: an overridden run is not the frozen protocol

    exp_id = a.experiment_id or new_run_id(cfg["experiment"])
    exp_dir = ART / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        s = summarize(exp_dir)
        print(f"summary: {s['completed_runs']}/{s['expected_runs']} runs -> {exp_dir/'summary.json'}")
        return 0

    write_json(exp_dir / "config.json", cfg)
    plan = build_plan(cfg, list(cfg["backends"]))
    write_json(exp_dir / "plan.json", {"seed": cfg["seed"], "order": plan, "commit": git_commit(), "created_utc": iso()})
    unsupported: dict[str, str] = {}

    # readiness / support gate per backend
    infos = {}
    for b, be in cfg["backends"].items():
        ok, info = readiness(be["url"])
        infos[b] = info
        if not ok:
            reason = info.get("reason_code") or info.get("reason") or "not ready"
            for bs, c in itertools.product(cfg["batch_sizes"], cfg["concurrency"]):
                unsupported[cell_id(b, bs, c)] = f"backend '{b}' service at {be['url']} not ready: {reason}"
            print(f"!! backend {b} unsupported: {reason}")
    write_json(exp_dir / "unsupported.json", unsupported)

    for p in plan:
        cid = cell_id(p["backend"], p["batch_size"], p["concurrency"])
        if cid in unsupported:
            continue
        rid = f"{cid}-t{p['trial']}"
        if a.resume and (exp_dir / "runs" / rid / "meta.json").exists():
            continue
        print(f"[{p['order_index']+1}/{len(plan)}] {rid}")
        m = run_one(cfg, exp_dir, p, infos[p["backend"]])
        ag = m["aggregate"]
        print(f"    attempts={ag['attempts']} ok={ag['successes']} img/s={ag['images_per_s'] and round(ag['images_per_s'],1)} p95={ag['e2e_ms']['p95'] and round(ag['e2e_ms']['p95'],1)}ms")
        time.sleep(cfg["cooldown_s"])
    s = summarize(exp_dir)
    print(f"summary: {s['completed_runs']}/{s['expected_runs']} runs; missing={len(s['missing'])} -> {exp_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
