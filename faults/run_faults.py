"""Run the full fault-drill matrix: 5 scenarios x N trials (default 3) = 15 incident records.

    python -m faults.run_faults                       # F01..F05, 3 trials each
    python -m faults.run_faults --scenarios F01 F04 --trials 1
"""
from __future__ import annotations

import argparse

from faults import f01_crash, f02_exporter_outage, f03_missing_weights, f04_latency_regression, f05_gpu_inaccessible
from faults.harness import add_common_args, ctx_from_args, run_scenarios

REGISTRY = {
    "F01": f01_crash.build, "F02": f02_exporter_outage.build, "F03": f03_missing_weights.build,
    "F04": f04_latency_regression.build, "F05": f05_gpu_inaccessible.build,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--scenarios", nargs="*", default=list(REGISTRY))
    a = ap.parse_args(argv)
    return run_scenarios([REGISTRY[s]() for s in a.scenarios], a.trials, ctx_from_args(a), a.first_trial)


if __name__ == "__main__":
    raise SystemExit(main())
