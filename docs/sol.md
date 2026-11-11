# Running on ASU Sol (no Docker)

Sol is a shared Slurm cluster: no root, no Docker daemon. This mode runs the **same** service, controller, alert rules,
dashboards, fault scenarios and report as the Compose deployment, but as ordinary processes inside one GPU job.

| Compose deployment | Native / Sol mode |
|---|---|
| Docker daemon, `restart: on-failure:3` | `sol/supervisor.py` (user-space supervisor, same policy: ≤3 consecutive retries, counter resets after 10 s uptime, manual stop is never restarted) |
| `docker events` | the supervisor's Docker-shaped event stream (`artifacts/native/events.jsonl`), read by the unchanged controller |
| Prometheus / Alertmanager / node-exporter containers | the official release binaries, downloaded and **sha256-verified** into `.tools/<os>-<arch>/` |
| DCGM Exporter | `ops/gpu_exporter.py`: `nvidia-smi` source, **same metric names** (`DCGM_FI_DEV_*`), unsupported fields omitted (never 0) |
| Grafana container | optional Grafana binary (`--grafana`); Prometheus UI works without it |
| fixed ports on loopback | free ports chosen per job (nodes are shared), recorded in `artifacts/native/ports.json`; all listeners bind `127.0.0.1` |
| F05 "no GPU device mapping" | `CUDA_VISIBLE_DEVICES=""` for the fault-case process, then the job's own allocation restored |

Verified on a Mac (CPU profile): supervisor restart policy, F01/F03/F04 end to end, F05 failure phase, rendered configs
(promtool-checked), tool checksum verification. **Not yet exercised on Sol:** module and partition names, `pip` CUDA wheels,
the `nvidia-smi` exporter against a real GPU, Grafana, and the F02 drill (needs the GPU exporter). Expect small fixes on first contact.

## 1. One-time setup (login node; VPN required)

```bash
git clone <your repo> P_3 && cd P_3          # or rsync it; keep it on /scratch if it will hold lots of artifacts
./sol/setup_env.sh                           # mamba env "gpulab" (module load mamba/latest), torch+torchvision, pinned deps,
                                             # verified Prometheus/Alertmanager/node_exporter binaries   (WITH_GRAFANA=1 adds Grafana)
```
If the login node limits the pip step, run it inside `interactive -c 4 -t 0-01:00` instead. Check the GPU/driver you will get with
`nvidia-smi` in a job, and set `TORCH_INDEX` (default `cu128`) if the driver is older.

## 2. Stages (each `./sol/submit.sh <group>` is one Slurm job: `-p htc -q public -G a100:1 -c 8 --mem=32G`)

```bash
./sol/submit.sh bringup        # prep (weights, fixtures, CPU reference) + preflight + tolerance calibration       ~20 min
# --- read artifacts/validation/validation-cuda-calibration-*.json, then freeze (no GPU needed):
python -m sol.lab stage freeze-tolerances ; git add validation/tolerances.json && git commit -m "freeze tolerances"
./sol/submit.sh validate       # preflight + CUDA correctness vs CPU reference + API contract checks                ~20 min
./sol/submit.sh benchmark      # 54 runs (resumable: resubmit if it hits the limit) + freeze the p95 baseline       ~1.5-2 h
./sol/submit.sh soak           # 30-min soak + overload + cold/cached start timing                                   ~45 min
./sol/submit.sh faults         # F01-F05 x 3 trials under baseline load                                              ~1 h
python -m sol.lab stage report # login node is fine: artifacts/report/REPORT.md
```
Jobs write `slurm-<name>-<jobid>.out`. `python -m sol.lab run <stage...>` is what each job executes: it starts the stack, runs the stages,
and **always** stops everything, including on the time-limit SIGTERM. Use the same GPU type for every job (`-G a100:1` by default);
the benchmark baseline and the F04 latency alert are only meaningful against the same hardware. Override with `sbatch -G a100.40gb:1 ...`
via extra arguments to `submit.sh`.

Interactive use (debugging): `salloc -p htc -q public -G a100:1 -c 8 -t 0-02:00`, then
`python -m sol.lab up --profiles gpu,cpu`, `python -m sol.lab status`, `python -m sol.lab stage preflight`, `python -m sol.lab down`.

## 3. Looking at dashboards

`python -m sol.lab up --grafana` (needs Grafana fetched) prints the port and the admin password file. From your laptop, try an SSH tunnel through the
login node to the compute node (`ssh -L 3000:<node>:<port> <asurite>@sol.asu.edu`); if that is not permitted, use an Open OnDemand desktop session
or read the same data from `artifacts/…/panel_data.json` (saved per drill). Prometheus is on `ports.json → prometheus`.

## 4. Deviations to record (copy into `docs/deviations.md` as you use them)

1. No container runtime: no image digest and no Compose deployment. `make report` therefore shows **AC01 = FAIL** with an explanation until you add
   `{"AC01": "Sol has no Docker; native deployment with env lock + commit"}` to `artifacts/review/skips.json` (then it shows SKIP, not PASS).
2. PF02 compares the framework's GPU UUID with `nvidia-smi` (no host-vs-container boundary). `preflight.json` says `executed_in: native-process`.
3. GPU telemetry comes from `nvidia-smi`, not DCGM: no XID counter; PF06 (`dcgmi diag`) is normally SKIP.
4. Shared node: other users' load affects CPU numbers. `uptime` and `loadavg` are recorded per run; consider `#SBATCH --exclusive` if your QOS allows it.
   Other users on the same node can reach loopback ports too; nothing here is sensitive, but do not reuse the Grafana password.
5. Time limits (4 h in the docs' examples) split the evaluation into several jobs; the stages are resumable or short enough to fit.

## 5. Gotchas

* Sol cancels GPU jobs idle at 0 % utilisation for 4 h; the stages here keep the GPU busy or finish early.
* Idle default memory for GPU jobs is 24 GB; this setup requests 32 GB explicitly.
* `--export=NONE` is set in the sbatch header (as in the RC examples). Pass settings like `EXPERIMENT` or `FAULT_ARGS` with `sbatch --export=FAULT_ARGS="--scenarios F04 --trials 1 --first-trial 4" ...` (variable only). **Do not use `--export=ALL,...`**: it copies your interactive shell into the job and the conda environment is then not activated correctly (the job fails with `No module named 'yaml'`).
* Never run two jobs in the same checkout at once (they share `artifacts/native`); chain them with `sbatch --dependency=afterany:<jobid> ...`.
* One native stack per checkout at a time (`artifacts/native/`, `monitoring/rules/baseline.yml`). Use separate clones for parallel experiments.
* Logs per service: `artifacts/native/logs/<service>.log`; supervisor: `artifacts/native/logs/supervisor.log`; services: `python -m sol.lab status`.
