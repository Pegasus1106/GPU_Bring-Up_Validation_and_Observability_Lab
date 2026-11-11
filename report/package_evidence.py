"""Package evidence for a repository upload.

    python -m report.package_evidence [--artifacts artifacts] [--out evidence] [--archive raw-artifacts.tgz]

Produces
  evidence/                small, reviewable files to COMMIT (summaries, preflight, validation, incident
                           records and per-trial timelines, alerts, controller events, deploy log, report)
  raw-artifacts.tgz        EVERYTHING under artifacts/ (large request CSVs, samples, logs) to attach to a
                           GitHub Release instead of committing. artifacts/report/evidence_index.json lists
                           a SHA256 for every raw file, so the archive can be verified against the repo.

Nothing is rewritten: copies are byte-identical, so hashes in the evidence index still match.
Paths in the files contain the cluster username and node names; review before publishing a public repo.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from common.util import REPO_ROOT, iso, sha256_file, write_json

# glob patterns (relative to artifacts/) of small files that belong in the repository
INCLUDE = [
    "report/*.json", "report/*.md",
    "preflight/*/preflight.json", "preflight/*/capabilities.json", "preflight/*/env-lock.txt",
    "validation/*.json",
    "benchmark/*/config.json", "benchmark/*/plan.json", "benchmark/*/summary.json", "benchmark/*/unsupported.json",
    "benchmark/*/runs/*/meta.json", "benchmark/*/runs/*/telemetry.json",
    "soak/*/soak.json", "overload/*/overload.json", "startup/*.json",
    "faults/incidents.jsonl",
    "faults/*/*/timeline.json", "faults/*/*/controller_events.jsonl", "faults/*/*/alerts.jsonl",
    "faults/*/*/grafana_links.json", "faults/*/*/panel_data.json",
    "alerts/alerts.jsonl", "controller/faults.jsonl", "controller/state.json",
    "deploy/*.log", "review/*.json",
    "native/run/services.json", "native/run/prometheus.yml", "native/run/alertmanager.yml", "native/run/controller.yaml", "native/ports.json",
]
MAX_COMMIT_BYTES = 5_000_000  # a "small" file; anything bigger goes only to the archive


def collect(art: Path) -> tuple[list[Path], list[tuple[Path, int]]]:
    keep: list[Path] = []
    too_big: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for pat in INCLUDE:
        for p in sorted(art.glob(pat)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            size = p.stat().st_size
            if size <= MAX_COMMIT_BYTES:
                keep.append(p)
            else:
                too_big.append((p, size))
    return keep, too_big


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "evidence")
    ap.add_argument("--archive", type=Path, default=REPO_ROOT / "raw-artifacts.tgz")
    a = ap.parse_args(argv)
    art = a.artifacts
    if not art.is_dir():
        raise SystemExit(f"no artifacts directory at {art}")

    keep, too_big = collect(art)
    if a.out.exists():
        shutil.rmtree(a.out)
    for p in keep:
        dst = a.out / p.relative_to(art)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    total = sum(p.stat().st_size for p in keep)
    (a.out / "README.md").write_text(
        f"# Evidence snapshot ({iso()})\n\n"
        "Small, byte-identical copies of the lab's result files. See `report/REPORT.md` for the joined results and the acceptance table.\n\n"
        "| Folder | Content |\n|---|---|\n"
        "| `report/` | `REPORT.md`, `summary.json`, `evidence_index.json` (SHA256 of every raw artifact) |\n"
        "| `preflight/` | PF01-PF06 results, capability manifest, environment lock |\n"
        "| `validation/` | CUDA/CPU correctness and API-contract results (per-input errors) |\n"
        "| `benchmark/` | 54-run matrix: config, trial order, per-run metadata + telemetry, `summary.json` |\n"
        "| `soak/`, `overload/`, `startup/` | 30-min soak, overload test, start-up timing |\n"
        "| `faults/` | `incidents.jsonl` plus per-trial timelines, controller events, alerts, dashboard data |\n"
        "| `alerts/`, `controller/` | local alert-receiver log, controller event log and restart counters |\n"
        "| `deploy/`, `native/run/` | deployment logs and the rendered stack configuration |\n\n"
        "Large raw files (request CSVs, service samples, logs) are in the release archive `raw-artifacts.tgz`; "
        "verify them against `report/evidence_index.json`.\n"
    )
    with tarfile.open(a.archive, "w:gz") as t:
        t.add(art, arcname="artifacts", filter=lambda ti: None if "prom-data" in ti.name or "am-data" in ti.name or "grafana-data" in ti.name else ti)
    write_json(a.out / "MANIFEST.json", {"created_utc": iso(), "files": len(keep), "bytes": total,
                                         "archive": a.archive.name, "archive_sha256": sha256_file(a.archive),
                                         "skipped_over_limit": [str(p.relative_to(art)) for p, _ in too_big]})
    print(f"evidence/: {len(keep)} files, {total/1e6:.1f} MB  (skipped as too large: {len(too_big)})")
    print(f"{a.archive}: {a.archive.stat().st_size/1e6:.1f} MB  sha256 in evidence/MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
