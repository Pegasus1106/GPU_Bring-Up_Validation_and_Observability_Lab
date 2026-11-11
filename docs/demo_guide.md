# Five-minute demo

Prepare beforehand (not part of the five minutes): `make up-gpu`, `make controller`, `make preflight`, a finished benchmark and fault drill saved under `artifacts/`.

| Min | Show | Command / place |
|---|---|---|
| 0:00 | Architecture and pinned environment | `docs/architecture.md`; `deploy/versions.env`, `requirements.lock`, `preflight.json` inventory (GPU UUID, driver, image digest, commit) |
| 0:45 | Environment gates pass | `make preflight` output: PF02 visibility, PF03 CUDA tensor op, PF04 hashes |
| 1:30 | Correctness | `make validate`: CUDA vs CPU reference, max abs error, top-1 agreement, batch=1/8/32; bad weights fail with specific codes |
| 2:00 | Short workload + dashboards | `python -m loadgen.loadgen --url http://127.0.0.1:8000 --backend cuda --duration 30`; Grafana *Lab 2*: throughput, p95, queue, GPU util/VRAM |
| 3:00 | One controlled failure | `python -m faults.f01_crash --trials 1` (or F02/F04): watch alert, restart and revalidation on *Lab 3* |
| 4:00 | Incident timeline | latest `artifacts/faults/F01/*/timeline.json`: injected → first observation → alert → notification → ready → validated |
| 4:30 | Full evidence | `make report` → `artifacts/report/REPORT.md` acceptance table and benchmark matrix |

Be explicit while presenting: single node; faults are controlled; any `NOT_RUN` / `FAIL` is shown, not hidden.
