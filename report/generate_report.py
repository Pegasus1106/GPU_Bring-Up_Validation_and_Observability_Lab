"""Join configuration, measurements, failures and test outcomes into a reviewable report.

    python -m report.generate_report [--artifacts artifacts] [--experiment <benchmark id>]

Writes artifacts/report/{summary.json, REPORT.md, evidence_index.json}. Regenerable from the raw
files at any time. Acceptance results come only from evidence found on disk:
    PASS     evidence exists and meets the criterion
    FAIL     evidence exists and does not meet it
    ERROR    evidence exists but is unreadable / the run errored
    SKIP     approved hardware-related skip recorded in artifacts/review/skips.json
    NOT_RUN  no evidence yet (never reported as pass)
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Optional

from common.util import REPO_ROOT, git_commit, iso, read_json, read_jsonl, sha256_file, write_json

CRITERIA = {
    "AC01": "Deploy from documented commands with pinned dependencies and cached weights",
    "AC02": "GPU profile proves CUDA execution and fails when GPU access is withheld; CPU profile works independently",
    "AC03": "Fixed-fixture correctness passes frozen tolerances; invalid inputs fail clearly",
    "AC04": "18-cell, 54-trial baseline and a 30-minute soak; unsupported cells and failures flagged",
    "AC05": "5 fault scenarios x 3 trials with complete timelines and cause-specific recovery outcomes",
    "AC06": "Held service/exporter faults alert locally within 45 s; transient crashes logged within 5 s",
    "AC07": "Validated recovery within 60 s (cached restart); cold healthy start within 120 s",
    "AC08": "Overload keeps pending work bounded with explicit rejection/timeout counts and no unbounded memory growth",
    "AC09": "A second person reproduces the demo and diagnoses a saved incident from the evidence",
}


def latest(art: Path, pattern: str) -> Optional[Path]:
    files = sorted(art.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load(p: Optional[Path]) -> Any:
    try:
        return read_json(p) if p else None
    except Exception:  # noqa: BLE001
        return None


def res(status: str, details: str, evidence: list[str | Path | None] | None = None, **extra) -> dict:
    return {"status": status, "details": details, "evidence": [str(e) for e in (evidence or []) if e], **extra}


def pick_experiment(art: Path, name: str | None) -> Optional[Path]:
    if name:
        p = art / "benchmark" / name / "summary.json"
        return p if p.exists() else None
    cands = [p for p in (art / "benchmark").glob("*/summary.json")] if (art / "benchmark").exists() else []
    best, score = None, -1
    for p in cands:
        s = load(p) or {}
        if s.get("completed_runs", 0) >= score and s.get("experiment_matrix", {}).get("duration_s", 0) >= 60:
            best, score = p, s.get("completed_runs", 0)
    return best


# ------------------------------------------------------------------ acceptance evaluation
def evaluate(art: Path, bench_path: Optional[Path]) -> dict:
    skips = load(art / "review/skips.json") or {}
    out: dict[str, dict] = {}

    pf_gpu = latest(art, "preflight/preflight-gpu-*/preflight.json")
    pf_cpu = latest(art, "preflight/preflight-cpu-*/preflight.json")
    pfg, pfc = load(pf_gpu), load(pf_cpu)
    deploy_logs = sorted((art / "deploy").glob("*.log")) if (art / "deploy").exists() else []
    incidents = read_jsonl(art / "faults/incidents.jsonl")
    val_gpu = load(latest(art, "validation/validation-cuda-2*.json"))
    val_cpu = load(latest(art, "validation/validation-cpu-2*.json"))
    api_gpu = load(latest(art, "validation/api-cuda-*.json"))
    api_cpu = load(latest(art, "validation/api-cpu-*.json"))

    # AC01
    if not (pfg and deploy_logs):
        out["AC01"] = res("NOT_RUN", "needs a GPU preflight.json and a deployment log (artifacts/deploy/*.log from `make up-gpu`)", [pf_gpu])
    else:
        inv = pfg.get("inventory", {})
        ok = (pfg.get("executed_in") == "container" and inv.get("image", {}).get("image_id") and pfg.get("commit")
              and (REPO_ROOT / "requirements-app.txt").exists() and (REPO_ROOT / "requirements.lock").exists())
        if pfg.get("executed_in") == "native-process":
            why = ("native (non-container) deployment, e.g. ASU Sol: there is no image digest or Compose deployment, so this criterion as written "
                   "cannot pass. Evidence instead: deploy log, env lock (preflight env-lock.txt), commit. To accept it, record the deviation in "
                   "docs/deviations.md and add {\"AC01\": \"<reason>\"} to artifacts/review/skips.json")
            out["AC01"] = res("FAIL", why, [pf_gpu, deploy_logs[-1]])
        else:
            out["AC01"] = res("PASS" if ok else "FAIL", "deploy log, image id, lockfile, inventory and commit present" if ok else "preflight not run in container or inventory incomplete",
                              [pf_gpu, deploy_logs[-1], REPO_ROOT / "requirements.lock"])

    # AC02
    f05 = [i for i in incidents if i["scenario"] == "F05"]
    if not pfg:
        out["AC02"] = res("NOT_RUN", "no GPU preflight evidence", [])
    else:
        gpu_ok = pfg["summary"].get("gpu_ready") is True and pfg.get("executed_in") in ("container", "native-process")
        nofb = [bool(i.get("extra", {}).get("no_silent_fallback_pass")) for i in f05]
        cpu_ok = bool(pfc and pfc["summary"].get("cpu_ready")) and bool(api_cpu and api_cpu.get("passed"))
        parts = {"gpu_preflight_ready": gpu_ok, "f05_no_silent_fallback": bool(nofb) and all(nofb), "cpu_profile_independent": cpu_ok}
        if not f05 or not pfc:
            out["AC02"] = res("NOT_RUN" if not gpu_ok else "FAIL", f"incomplete evidence: {parts}", [pf_gpu, pf_cpu], parts=parts)
        else:
            out["AC02"] = res("PASS" if all(parts.values()) else "FAIL", str(parts), [pf_gpu, pf_cpu], parts=parts)

    # AC03
    if not val_gpu:
        out["AC03"] = res("NOT_RUN", "no cuda acceptance validation JSON (artifacts/validation/validation-cuda-*.json)")
    else:
        parts = {"checks_passed": val_gpu.get("checks_passed"), "tolerances_frozen": val_gpu.get("tolerances_frozen"),
                 "api_contract_passed": bool(api_gpu and api_gpu.get("passed"))}
        out["AC03"] = res("PASS" if all(parts.values()) else "FAIL", str(parts) + ("" if val_gpu.get("tolerances_frozen") else " - tolerances are still 'candidate'"), [], parts=parts)

    # AC04
    bench = load(bench_path)
    soak = load(latest(art, "soak/*/soak.json"))
    if not bench:
        out["AC04"] = res("NOT_RUN", "no benchmark summary.json with duration_s >= 60")
    else:
        missing = bench.get("missing", [])
        full = bench["completed_runs"] == bench["expected_runs"] == 54
        soak_ok = bool(soak and soak.get("complete"))
        parts = {"runs": f"{bench['completed_runs']}/{bench['expected_runs']}", "cells": f"{bench['completed_cells']}/{bench['expected_cells']}",
                 "soak_complete_30min": soak_ok, "flagged_missing": len(missing)}
        out["AC04"] = res("PASS" if full and soak_ok else "FAIL", str(parts), [bench_path], parts=parts, unsupported=[m for m in missing])

    # AC05
    if not incidents:
        out["AC05"] = res("NOT_RUN", "no incident records (artifacts/faults/incidents.jsonl)")
    else:
        counts: dict[str, int] = {}
        good: dict[str, int] = {}
        for i in incidents:
            counts[i["scenario"]] = counts.get(i["scenario"], 0) + 1
            if i["final_status"] == "RECOVERED_VALIDATED" and i.get("cleanup", {}).get("ok"):
                good[i["scenario"]] = good.get(i["scenario"], 0) + 1
        need = all(good.get(s, 0) >= 3 for s in ("F01", "F02", "F03", "F04", "F05"))
        out["AC05"] = res("PASS" if need else "FAIL", f"validated+clean trials per scenario: {good}; recorded: {counts}", [art / "faults/incidents.jsonl"], trials=good)

    # AC06
    held = [i for i in incidents if "ac06_alert_within_45s" in i]
    crash = [i for i in incidents if "ac06_crash_logged_within_5s" in i]
    if not (held or crash):
        out["AC06"] = res("NOT_RUN", "no held-fault or crash incidents recorded")
    else:
        ok = all(i["ac06_alert_within_45s"] for i in held) and all(i["ac06_crash_logged_within_5s"] for i in crash)
        vals = [i["metrics"]["fault_to_notification_s"] for i in held if i.get("metrics", {}).get("fault_to_notification_s") is not None]
        worst = max(vals) if vals else "no notification received"
        out["AC06"] = res("PASS" if ok and held and crash else ("FAIL" if not ok else "NOT_RUN"),
                          f"alert<=45s: {sum(i['ac06_alert_within_45s'] for i in held)}/{len(held)}, crash<=5s: {sum(i['ac06_crash_logged_within_5s'] for i in crash)}/{len(crash)}, worst notification {worst}s",
                          [art / "alerts/alerts.jsonl", art / "controller/faults.jsonl"])

    # AC07
    st = load(latest(art, "startup/startup-*.json"))
    f01 = [i for i in incidents if i["scenario"] == "F01"]
    if not (f01 or st):
        out["AC07"] = res("NOT_RUN", "no F01 recovery records or startup measurements")
    else:
        rec_ok = bool(f01) and all(i.get("ac07_within_60s") for i in f01)
        cold = [r for r in (st or {}).get("results", []) if r["kind"] == "cold_start"]
        cold_ok = bool(cold) and all(r["met"] for r in cold)
        parts = {"f01_recovery_within_60s": rec_ok, "cold_start_within_120s": cold_ok}
        out["AC07"] = res("PASS" if all(parts.values()) else ("NOT_RUN" if not (f01 and cold) else "FAIL"), str(parts), [], parts=parts)

    # AC08
    ov = load(latest(art, "overload/*/overload.json"))
    out["AC08"] = res("NOT_RUN", "no overload.json") if not ov else res("PASS" if ov.get("passed") else "FAIL", str({k: v["pass"] for k, v in ov["checks"].items()}), [])

    # AC09
    rv = load(art / "review/reviewer_signoff.json")
    if not rv:
        out["AC09"] = res("NOT_RUN", "needs artifacts/review/reviewer_signoff.json (see docs/reviewer_checklist.md)")
    else:
        ok = bool(rv.get("reproduced_demo")) and bool(rv.get("diagnosed_incident")) and bool(rv.get("reviewer"))
        out["AC09"] = res("PASS" if ok else "FAIL", str({k: rv.get(k) for k in ("reviewer", "reproduced_demo", "diagnosed_incident")}), [art / "review/reviewer_signoff.json"])

    for k, why in skips.items():  # approved hardware-related skips
        if k in out and out[k]["status"] != "PASS":
            out[k] = {**out[k], "status": "SKIP", "details": f"approved skip: {why}"}
    return {k: {"criterion": CRITERIA[k], **v} for k, v in sorted(out.items())}


# ------------------------------------------------------------------ incident statistics
def stat(vals: list[float]) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}


def incident_stats(incidents: list[dict]) -> dict:
    out: dict = {}
    for sid in sorted({i["scenario"] for i in incidents}):
        rows = [i for i in incidents if i["scenario"] == sid]
        m = [r.get("metrics", {}) for r in rows]
        out[sid] = {
            "name": rows[0]["scenario_name"], "trials": len(rows),
            "final_status": {s: sum(1 for r in rows if r["final_status"] == s) for s in sorted({r["final_status"] for r in rows})},
            "fault_to_first_observation_s": stat([x.get("fault_to_first_observation_s") for x in m]),
            "fault_to_notification_s": stat([x.get("fault_to_notification_s") for x in m]),
            "fault_to_recovery_s": stat([x.get("fault_to_recovery_s") for x in m]),
            "correction_to_recovery_s": stat([x.get("correction_to_recovery_s") for x in m]),
            "restarts_observed": sum(sum(w["restarts_observed"] for w in r.get("restart_count", {}).values()) for r in rows),
            "fault_class": rows[0]["fault_class"],
        }
    return out


# ------------------------------------------------------------------ markdown
def fmt(x, nd=1):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(summary: dict) -> str:
    L = [f"# GPU Workload Lab - results report", "",
         f"Generated {summary['generated_utc']} from raw artifacts at commit `{summary['commit']}`. "
         "Every number below is read from files in `artifacts/`; nothing here is a target presented as a result.", ""]
    L += ["## Acceptance status", "", "| ID | Status | Criterion | Details |", "|---|---|---|---|"]
    for k, v in summary["acceptance"].items():
        L.append(f"| {k} | **{v['status']}** | {v['criterion']} | {v['details'][:160]} |")
    pf = summary.get("environment")
    L += ["", "## Environment", ""]
    if pf:
        host, cont = pf["inventory"]["host"], pf["inventory"].get("container", {})
        gpus = host.get("gpus") or []
        L += [f"- Mode `{pf['configured_mode']}`, preflight run `{pf['run_id']}`, executed in `{pf.get('executed_in')}`",
              f"- OS `{host.get('os_pretty') or host.get('os')}`, CPU `{host.get('cpu_model')}` ({host.get('cpu_logical')} logical), RAM {fmt((host.get('ram_bytes') or 0)/2**30)} GiB",
              f"- GPU {', '.join(g['name'] + ' (' + g['uuid'] + ', driver ' + g['driver_version'] + ')' for g in gpus) or 'none detected'}",
              f"- torch `{cont.get('torch')}`, torchvision `{cont.get('torchvision')}`, CUDA runtime `{cont.get('cuda_runtime')}`, cuDNN `{cont.get('cudnn')}`",
              f"- Image `{pf['inventory'].get('image', {}).get('image_id')}`", f"- Time sync: `{host.get('time_sync')}`"]
    else:
        L.append("- No preflight evidence found.")
    b = summary.get("benchmark")
    L += ["", "## Baseline benchmark", ""]
    if b:
        L += [f"Experiment `{b['experiment_id']}`: {b['completed_runs']}/{b['expected_runs']} runs, {b['completed_cells']}/{b['expected_cells']} cells. "
              f"Percentiles: {b['percentile_method']}. {b['p99_note']}.", "",
              "| Cell | img/s (mean ± sd) | p50 / p95 / p99 ms (mean) | success | timeout | reject | attempts | note |", "|---|---|---|---|---|---|---|---|"]
        for cid, c in sorted(b["cells"].items()):
            ips, p95 = c["images_per_s"], c["p95_ms"]
            L.append(f"| {cid} | {fmt(ips['mean'])} ± {fmt(ips['stdev'])} (cv {fmt(ips['cv'], 3)}) | {fmt(c['p50_ms']['mean'])} / {fmt(p95['mean'])} / {fmt(c['p99_ms']['mean'])} | "
                     f"{fmt(c['success_rate']['mean'], 3)} | {fmt(c['timeout_rate']['mean'], 3)} | {fmt(c['rejection_rate']['mean'], 3)} | {c['total_attempts']} | {'p99 weak' if c['p99_weak_evidence'] else ''} |")
        if b["missing"]:
            L += ["", "**Missing / unsupported cells (kept, not deleted):**", ""] + [f"- `{m['cell']}`: {m['status']} - {m['reason']}" for m in b["missing"]]
        if b["comparisons"]:
            L += ["", "**GPU vs CPU (same batch / concurrency):**", "", "| batch | conc | img/s change | p95 change |", "|---|---|---|---|"]
            for c in b["comparisons"]:
                L.append(f"| {c['batch_size']} | {c['concurrency']} | {fmt(c['images_per_s']['change_pct'])}% ({c['images_per_s']['verdict']}) | {fmt(c['p95_ms']['change_pct'])}% ({c['p95_ms']['verdict']}) |")
    else:
        L.append("No benchmark with duration >= 60 s found.")
    s = summary.get("soak")
    L += ["", "## Soak", ""]
    L.append(f"Run `{s['run_id']}`: complete={s['complete']}, attempts={s['aggregate']['attempts']}, success rate={fmt(s['aggregate']['success_rate'], 4)}, drift={s['drift']}" if s else "No soak evidence found.")
    L += ["", "## Fault drills", ""]
    if summary["incidents"]:
        L += ["| Scenario | Class | Trials | Final status | Detect s (median) | Notify s (median) | Recover s (median) | Correction→recovery s (median) |", "|---|---|---|---|---|---|---|---|"]
        for sid, v in summary["incidents"].items():
            L.append(f"| {sid} {v['name']} | {v['fault_class']} | {v['trials']} | {v['final_status']} | {fmt(v['fault_to_first_observation_s'].get('median'))} | "
                     f"{fmt(v['fault_to_notification_s'].get('median'))} | {fmt(v['fault_to_recovery_s'].get('median'))} | {fmt(v['correction_to_recovery_s'].get('median'))} |")
    else:
        L.append("No incident records found.")
    ov = summary.get("overload")
    L += ["", "## Overload", ""]
    L.append("\n".join(f"- {k}: {'PASS' if v['pass'] else 'FAIL'} {({a: b for a, b in v.items() if a != 'pass'})}" for k, v in ov["checks"].items()) if ov else "No overload evidence found.")
    L += ["", "## Resume claim inputs (filled only from evidence)", ""]
    for k, v in summary["resume_inputs"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Limits of these results", "",
          "- Single node, single GPU; no multi-node or production-scale claim.",
          "- Faults are controlled application/configuration faults (F04 is a lab-only injected delay); no firmware, driver or physical hardware faults.",
          "- Synthetic input images validate numerical equivalence and performance, not classification accuracy, unless `manifests/inputs.json` says `synthetic: false`.",
          "- Targets are proposed acceptance gates; see the table above for what the evidence actually shows."]
    return "\n".join(L) + "\n"


def evidence_index(art: Path, extra: list[Path]) -> list[dict]:
    files = []
    for base in (art, ):
        skip_dirs = {"prom-data", "am-data", "grafana-data"}  # TSDB / state directories: too many files, not review evidence
        for p in sorted(base.rglob("*")):
            parts = p.relative_to(art).parts
            if p.is_file() and "report" not in parts[:1] and not (skip_dirs & set(parts)) and p.stat().st_size < 200_000_000:
                files.append(p)
    files += [p for p in extra if p.exists()]
    return [{"path": str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--experiment", default=None)
    a = ap.parse_args(argv)
    art = a.artifacts
    bench_path = pick_experiment(art, a.experiment)
    bench = load(bench_path)
    if bench:
        bench["expected_cells"] = bench.get("expected_cells")
    incidents = read_jsonl(art / "faults/incidents.jsonl")
    pf = load(latest(art, "preflight/preflight-gpu-*/preflight.json")) or load(latest(art, "preflight/preflight-cpu-*/preflight.json"))
    soak = load(latest(art, "soak/*/soak.json"))
    ov = load(latest(art, "overload/*/overload.json"))
    acc = evaluate(art, bench_path)
    stats = incident_stats(incidents)

    n_cfg = bench["completed_cells"] if bench else None
    det = [i["metrics"]["fault_to_first_observation_s"] for i in incidents if i.get("metrics", {}).get("fault_to_first_observation_s") is not None]
    rec = [i["ac07_recovery_s"] for i in incidents if i.get("ac07_recovery_s") is not None and i["final_status"] == "RECOVERED_VALIDATED"]
    summary = {
        "generated_utc": iso(), "commit": git_commit(), "artifacts_dir": str(art),
        "environment": pf, "benchmark": bench, "soak": soak, "incidents": stats, "overload": ov, "acceptance": acc,
        "percentile_method": bench["percentile_method"] if bench else None,
        "resume_inputs": {
            "[N] workload configurations evaluated": n_cfg if n_cfg is not None else "not available (no benchmark evidence)",
            "[N] controlled fault scenarios (record count)": len(incidents) if incidents else "not available",
            "[observed detection time] (median, s)": statistics.median(det) if det else "not available",
            "[observed validated recovery time] (median, s)": statistics.median(rec) if rec else "not available",
            "[measured performance change]": "see GPU-vs-CPU comparison table; quote with batch/concurrency and run-to-run variability",
        },
        "unreadable_artifacts": [],
    }
    out = art / "report"
    write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(render_md(summary))
    extra = [REPO_ROOT / p for p in ("requirements.lock", "requirements-app.txt", "validation/tolerances.json", "manifests/model.json", "manifests/inputs.json",
                                      "manifests/preprocessing.json", "monitoring/rules/alerts.yml", "monitoring/rules/baseline.yml",
                                      "monitoring/grafana/dashboards/1-readiness.json", "monitoring/grafana/dashboards/2-performance.json",
                                      "monitoring/grafana/dashboards/3-incidents.json", "deploy/versions.env", "deploy/docker-compose.yml")]
    idx = evidence_index(art, extra)
    write_json(out / "evidence_index.json", {"generated_utc": iso(), "commit": git_commit(), "files": idx})
    for k, v in acc.items():
        print(f"{k} {v['status']:8} {v['criterion']}")
    print(f"report -> {out/'REPORT.md'}  ({len(idx)} evidence files indexed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
