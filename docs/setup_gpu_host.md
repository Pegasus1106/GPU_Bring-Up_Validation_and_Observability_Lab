# GPU host setup

Planning recommendation (not a requirement): ≥4 CPU cores, 16 GB RAM, 50 GB free disk, one NVIDIA GPU with ≈8 GB VRAM, Linux.

1. **Driver** – install an NVIDIA driver supported for your GPU; `nvidia-smi` must work on the host.
2. **Docker + Compose v2**, then the **NVIDIA Container Toolkit** and `sudo nvidia-ctk runtime configure --runtime=docker`
   (see the toolkit install guide and its sample workload: `docker run --rm --gpus all <image> nvidia-smi`).
3. **Pick a consistent set**: driver ↔ CUDA runtime in `BASE_IMAGE` ↔ DCGM Exporter tag (`<exporter>-<DCGM>-<os>`).
   Edit `deploy/versions.env`, run `make pin-images` to resolve digests, record the tested set in the environment manifest
   (`preflight.json` captures it automatically).
4. `make init` → `make weights fixtures reference` (needs network once) → `make up-gpu`.
5. `make controller` (host process; needs the Docker socket, which is never mounted into a container).
6. `make preflight` must end with `gpu_ready=True` before any measurement. A skipped PF02/PF03/PF05 is **not** acceptable GPU evidence.

Cloud GPU is optional: set a personal budget, tear everything down after experiments (`make down`).

## Order of operations for the full evaluation

```
make preflight
python -m validation.correctness --backend cuda --calibrate      # inside tools-gpu; review the errors
python -m validation.correctness --freeze                        # freeze tolerances BEFORE the final run
make validate                                                    # acceptance run (cuda + API)
make up-both && make benchmark                                   # 54 runs
make freeze-baseline BASELINE=artifacts/benchmark/<id>/summary.json
make soak && make overload && make startup
make fault-test                                                  # needs controller + monitoring up
make report
```

Do not loosen tolerances after seeing final-evaluation failures; investigate. Record any hardware-driven change
to targets and its rationale in `docs/deviations.md`.
