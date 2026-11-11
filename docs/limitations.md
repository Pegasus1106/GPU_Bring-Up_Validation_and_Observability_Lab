# Limitations and claims policy

* Single node, single GPU. No multi-node, multi-GPU, Kubernetes or Slurm result is implied.
* Faults are controlled application / container / configuration faults. F04 is a real but lab-only delay inside the worker,
  recorded as `fault_class: injected_latency`. No firmware, driver unload, GPU reset, thermal or disk-fill faults.
* Default inputs are synthetic images: they support latency, throughput and CPU-vs-GPU numerical-equivalence claims, **not** classification accuracy.
* Preprocessing runs once at start-up, so request latency excludes image decoding. State this when quoting latency.
* The p95 baseline for `LatencyRegression` is the server-side histogram (bucket-interpolated) compared against a frozen
  client-side benchmark p95 for the same backend and batch size at concurrency 4; the factor of 2 absorbs that difference. It does not cover other concurrency levels.
* `process_resident_memory_bytes` exists only on Linux; the memory-growth check reports “unavailable” elsewhere (and does not pass).
* DCGM/`nvidia-smi` field availability varies by GPU. Power, temperature and ECC may be unsupported; the lab reports that, it does not infer zero.
* Docker `restart: on-failure:3` stops retrying after three failures; F01 cleanup restarts the container explicitly and says so.
* Targets (45 s, 5 s, 60 s, 120 s, 5 % …) are proposed gates. If hardware forces a change, record it in `docs/deviations.md` with the reason.

## Resume claims

Fill the templates **only** from `artifacts/report/REPORT.md` (“Resume claim inputs”). Say “single-node”, say which faults were
injected delays versus real container/config failures, and do not claim firmware validation, hardware repair, or production-scale
cluster operation.
