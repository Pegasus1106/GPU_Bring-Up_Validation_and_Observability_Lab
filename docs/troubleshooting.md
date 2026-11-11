# Troubleshooting

| Symptom | Likely cause | Check / fix |
|---|---|---|
| `make up-gpu`: “could not select device driver nvidia” | Container Toolkit not configured | RB-003 step 2 |
| `/readyz` 503 `loading` for a long time | cold CUDA init / slow disk | `inference_startup_seconds`; first start loads 256 inputs and runs CUDA init |
| `/readyz` 503 `weights_missing` | weights not fetched / not mounted | RB-002 |
| `/readyz` 503 `cuda_unavailable` | no device mapping or driver mismatch | RB-003 |
| Grafana “unavailable” in GPU panels | unsupported field or dcgm-exporter down | `capabilities.json`, `up{job="dcgm-exporter"}` |
| No `LatencyRegression` ever fires | no frozen baseline yet, or <100 completions/min | `make freeze-baseline BASELINE=…`; the F04 precondition says so explicitly |
| `RestartObserved` never fires after F01 | controller not running → no counter | `make controller`; panel “Controller heartbeat age” |
| Prometheus cannot reach targets | host networking is Linux-only | run the stack on Linux; targets are `127.0.0.1:*` |
| Benchmark cells all “backend_mismatch” | `matrix.yaml` points the `gpu` URL at a CPU service | check ports 8000 (gpu) / 8001 (cpu) |
| `memory_not_growing` “unavailable” | RSS metric exists only on Linux | run `make overload` on the Linux host |
| Preflight PF06 SKIP | `dcgmi` not installed, or GPU not idle | install DCGM on the host or stop the load; SKIP is recorded with its reason |
| Permission denied writing `./artifacts` | container UID differs from host | use `make` (it exports `LAB_UID/LAB_GID`) and `make dirs` before first `up` |
| Sol: `missing prometheus … run setup` | binaries not fetched for this CPU architecture | `python -m sol.lab setup` on a login node (x86-64 and aarch64 are separate directories) |
| Sol: stack up but `inference-gpu` not ready, `cuda_unavailable` | job has no GPU, or torch is a CPU build | `nvidia-smi -L` in the job; `python -c "import torch;print(torch.version.cuda)"` |
| Sol: `timeout waiting for … /readyz` | cold CUDA init / first import of torch from a network filesystem | raise `--wait-s`; check `artifacts/native/logs/inference-gpu.log` |
| Sol: job killed at the time limit | stage group too long | resubmit (`benchmark` resumes); see time limits in `sol/submit.sh` |
| Port already in use | another Prometheus/Grafana on loopback | stop it or change ports in `deploy/docker-compose.yml` and `monitoring/prometheus.yml` |
