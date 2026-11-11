#!/bin/bash
# Docker-free GPU lab job for ASU Sol.   Usage:  sbatch sol/lab.sbatch <stage> [<stage> ...]
# Prefer sol/submit.sh, which picks sensible stage groups and time limits.
#
# Partition / QOS / GPU names follow the ASU RC docs (docs.rc.asu.edu/requesting-resources):
#   -p htc -q public  -G a100:1  (an explicit GPU type keeps every job on the same hardware,
#   which matters for comparing benchmark and baseline runs). Override on the command line, e.g.
#   sbatch -G a100.40gb:1 sol/lab.sbatch ...   or   sbatch -p public ...
# To pass a setting such as FAULT_ARGS or EXPERIMENT use  --export=NAME=value  (without ALL): ALL would copy your
# interactive shell into the job and break the environment activation below.
#SBATCH -J gpulab
#SBATCH -p htc
#SBATCH -q public
#SBATCH -t 0-04:00
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --export=NONE

set -o pipefail
cd "${SLURM_SUBMIT_DIR:?run this with sbatch}" || exit 1

module load mamba/latest
source activate "${LAB_ENV:-gpulab}"   # conda activate scripts are not nounset-safe, so no `set -u` before this
set -u

# fail fast, with a readable message, if the wrong Python was picked up (e.g. the environment was not activated)
if ! python -c "import yaml, requests, torch" 2>/dev/null; then
  echo "ERROR: the Python in this job ($(command -v python)) cannot import yaml/requests/torch, so the '${LAB_ENV:-gpulab}' environment is not active." >&2
  echo "       Submit with a clean environment: pass variables as  --export=NAME=value  (NOT --export=ALL,...)." >&2
  exit 1
fi

echo "job ${SLURM_JOB_ID} on $(hostname) | GPUs: ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unknown}} | stages: $*"
nvidia-smi -L || echo "WARNING: nvidia-smi found no GPU in this job"
uptime   # background load on a shared node is a benchmark control: keep it in the log

# one stack per job: starts Prometheus, Alertmanager, exporters, controller, workloads as user processes,
# runs the stages in order, and ALWAYS stops everything (also on the time-limit SIGTERM)
python -m sol.lab run "$@"
rc=$?
echo "sol.lab finished with rc=${rc}"
exit ${rc}
