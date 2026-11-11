# RB-006 Automatic recovery exhausted (`unresolved`)

**Alert:** `RecoveryUnresolved` · state `unresolved` in `artifacts/controller/faults.jsonl`

The controller restarted the running-but-unready workload 3 times in 5 minutes and it is still not ready. It will not restart again until an operator intervenes
or the service becomes ready by itself.

1. Collect evidence before touching anything: `artifacts/controller/faults.jsonl` (timeline), `docker logs lab-inference-<profile>`, `artifacts/logs/*.jsonl`, `/readyz` body, Grafana “Incidents” dashboard window.
2. Classify the `reason_code` seen at each restart. Same code every time ⇒ configuration/hardware (RB-002/RB-003). Different or none ⇒ look at OOM (`docker inspect … OOMKilled`), dependency drift, disk full.
3. Fix the cause. Then redeploy explicitly: `make down-workload && make up-gpu`; the controller returns to `starting → healthy` on its own when `/readyz` is 200.
4. Validate (`make validate`), and record what happened (incident note + any deviation in `docs/deviations.md`).

Out of scope for automatic or lab-level action: driver reload, GPU reset, firmware, hardware replacement.
