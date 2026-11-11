#!/bin/bash
# Submit one stage group to Slurm.   ./sol/submit.sh <group> [extra sbatch args...]
#
#   bringup    prep preflight calibrate           1 h    weights/fixtures/reference, PF01-PF06, tolerance calibration
#   validate   preflight validate                 1 h    run AFTER `python -m sol.lab stage freeze-tolerances`
#   benchmark  benchmark freeze-baseline          4 h    18 cells x 3 trials (resumable), then freezes the p95 baseline
#   soak       soak overload startup              2 h    30-min soak, overload test, cold/cached start timing
#   faults     faults                             3 h    F01-F05 x 3 trials (needs the frozen baseline for F04)
#   cpu        cpu-check                          1 h    CPU profile alone (AC02); needs no GPU: pass  -G 0
#   report     (login node) python -m sol.lab stage report
set -euo pipefail
cd "$(dirname "$0")/.."
group="${1:?usage: $0 <bringup|validate|benchmark|soak|faults|cpu> [sbatch args]}"
shift || true
case "$group" in
  bringup)   stages="prep preflight calibrate";      t="0-01:00" ;;
  validate)  stages="preflight validate";            t="0-01:00" ;;
  benchmark) stages="benchmark freeze-baseline";     t="0-04:00" ;;
  soak)      stages="soak overload startup";         t="0-02:00" ;;
  faults)    stages="faults";                        t="0-03:00" ;;
  cpu)       stages="cpu-check";                     t="0-01:00" ;;
  *) echo "unknown group '$group'"; exit 2 ;;
esac
exec sbatch -t "$t" -J "gpulab-$group" "$@" sol/lab.sbatch $stages
