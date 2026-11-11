# Acceptance map

Each criterion is evaluated by `make report` from files in `artifacts/`. Status is `PASS`, `FAIL`, `ERROR`, `SKIP`
(approved hardware-related skip recorded in `artifacts/review/skips.json` as `{"AC0x": "reason"}`) or `NOT_RUN`.
Mocked CPU tests and promtool unit tests are **simulated** evidence and never satisfy a GPU gate.

| ID | How to produce the evidence | Evidence read by the report |
|---|---|---|
| AC01 | `make init weights fixtures reference up-gpu preflight` | `artifacts/deploy/up-gpu-*.log`, `preflight.json` (executed in container, image id, commit), `requirements.lock` |
| AC02 | `make preflight` (PF01–PF06), `python -m faults.f05_gpu_inaccessible`, `make up-cpu && PROFILE=cpu make validate && make preflight-cpu` | GPU `preflight.json` summary `gpu_ready`, F05 incident `no_silent_fallback_pass`, CPU preflight + API checks |
| AC03 | `make calibrate` → review → `make freeze-tolerances` → `make validate` | `validation-cuda-*.json` (`checks_passed`, `tolerances_frozen`), `api-cuda-*.json`; per-input errors are in the JSON |
| AC04 | `make up-both benchmark soak` | `benchmark/<id>/summary.json` (54 runs, 18 cells, unsupported cells listed), `soak/<id>/soak.json` (`complete`) |
| AC05 | `make fault-test` | `faults/incidents.jsonl`: ≥3 `RECOVERED_VALIDATED` + clean-up-verified trials per scenario; per-trial `timeline.json`, `panel_data.json`, `grafana_links.json` |
| AC06 | `make fault-test` | `ac06_alert_within_45s` (F02, F03, F05), `ac06_crash_logged_within_5s` (F01) from `alerts.jsonl` and `controller/faults.jsonl` |
| AC07 | `make fault-test`, `make startup` | `ac07_within_60s` (F01), `startup/*.json` cold start ≤120 s; weights are cached, no download |
| AC08 | `make overload` | `overload/<id>/overload.json`: queue ≤ capacity, explicit 429/504 counts, RSS slope, recovery + validation |
| AC09 | give the repo + `docs/demo_guide.md` to a second person; they complete `docs/reviewer_checklist.md` | `artifacts/review/reviewer_signoff.json` |

Native / Sol mode: the same commands exist as `python -m sol.lab stage <name>` (see `docs/sol.md`); AC01 needs an approved deviation there.

Screenshots: dashboards do not ship an image renderer. Open the URLs in each trial's `grafana_links.json` (exact incident window)
and save screenshots next to the trial's `timeline.json`.

## Test layers

| Layer | Command | Notes |
|---|---|---|
| Unit | `make test` | manifest parsing, tolerances, API input checks, bounded labels, alert decision logic, state transitions, recovery limits |
| promtool | `make rules-test` | absent series, low traffic, short blips, restarts shorter than the hold period |
| Container integration | `LAB_TEST_URL=… pytest -m integration` | health endpoints, wrong model path, queue rejection, evidence persistence |
| GPU integration | `make test-gpu` | CUDA use, tolerances, timing, exporter fields |
