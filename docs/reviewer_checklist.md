# Reviewer checklist (AC09)

A second person who did not build the lab completes this using only the repository and `docs/demo_guide.md`.

**Reproduce**
- [ ] Followed `README.md` / `docs/setup_gpu_host.md` on a prepared host without asking the author questions.
- [ ] `make up-gpu`, `make preflight` (gpu_ready=True), `make validate` succeeded.
- [ ] Short load run and the three Grafana dashboards show live data; unsupported GPU fields read “unavailable”.
- [ ] Ran one drill (F01, F02 or F04) and saw the alert in `artifacts/alerts/alerts.jsonl`.

**Diagnose a saved incident (from evidence only; do not read the scenario source first)**
- [ ] Picked one `artifacts/faults/<Fxx>/trial-*/` directory.
- [ ] Stated the fault, its first symptom, which component detected it, and who recovered it, using `timeline.json`, `controller_events.jsonl`, `alerts.jsonl`, `requests.csv`, `panel_data.json`.
- [ ] Named the runbook that applies.

**Documentation issues found and how they were resolved**

| Issue | Where | Resolution (commit / note) |
|---|---|---|
| | | |

When complete, create `artifacts/review/reviewer_signoff.json`:
```json
{"reviewer": "Name", "date": "YYYY-MM-DD", "reproduced_demo": true, "diagnosed_incident": true, "incident_dir": "artifacts/faults/F0x/trial-…", "doc_issues_resolved": 0}
```
