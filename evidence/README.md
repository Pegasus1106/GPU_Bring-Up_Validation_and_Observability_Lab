# Evidence snapshot (2026-11-11T05:25:41.360Z)

Small, byte-identical copies of the lab's result files. See `report/REPORT.md` for the joined results and the acceptance table.

| Folder | Content |
|---|---|
| `report/` | `REPORT.md`, `summary.json`, `evidence_index.json` (SHA256 of every raw artifact) |
| `preflight/` | PF01-PF06 results, capability manifest, environment lock |
| `validation/` | CUDA/CPU correctness and API-contract results (per-input errors) |
| `benchmark/` | 54-run matrix: config, trial order, per-run metadata + telemetry, `summary.json` |
| `soak/`, `overload/`, `startup/` | 30-min soak, overload test, start-up timing |
| `faults/` | `incidents.jsonl` plus per-trial timelines, controller events, alerts, dashboard data |
| `alerts/`, `controller/` | local alert-receiver log, controller event log and restart counters |
| `deploy/`, `native/run/` | deployment logs and the rendered stack configuration |

Large raw files (request CSVs, service samples, logs) are in the release archive `raw-artifacts.tgz`; verify them against `report/evidence_index.json`.
