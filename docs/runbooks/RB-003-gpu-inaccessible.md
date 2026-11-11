# RB-003 GPU inaccessible or unsupported

**Alerts:** `ServiceUnavailable`, `OperatorActionRequired` · **Reason codes:** `cuda_unavailable`, `cuda_check_failed`, `unsupported_hardware` · **Drill:** F05

The GPU profile never falls back to CPU. `/readyz` is 503 and `/infer` is rejected. The controller does not restart.

1. Host: `nvidia-smi -L`. Fails → driver/hardware problem: **out of scope for this lab** (no driver repair, no GPU reset); escalate.
2. Toolkit: `docker run --rm --gpus all <image> nvidia-smi -L`. Fails → `nvidia-ctk runtime configure --runtime=docker`, restart Docker.
3. Compose: the service needs `deploy.resources.reservations.devices` (driver `nvidia`). A missing mapping is the F05 cause.
4. Container CUDA: `docker compose --profile tools-gpu run --rm tools-gpu python -m preflight.container_checks --backend cuda` → look at `cuda.reason`.
5. Version mismatch: base image CUDA runtime newer than the driver supports → choose a supported combination in `deploy/versions.env`.
6. Redeploy (`make up-gpu`), then `make preflight` (PF02/PF03 must PASS) and `make validate`.
