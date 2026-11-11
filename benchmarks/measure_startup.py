"""Measure cold healthy start-up and cached restart time-to-validated-ready (AC07).

Cold start : container (or, in native/Sol mode, process) removed and started again, weights already cached on disk
Restart    : `docker restart` of a running container / supervisor restart of the process
Both end at a *validated* ready (fixed-input top-1 check), not just HTTP 200.

    python -m benchmarks.measure_startup --profile gpu --trials 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common.util import REPO_ROOT, git_commit, iso, new_run_id, write_json
from common import endpoints
from faults import runtime
from faults.harness import compose, docker, http_ready
from validation.api_checks import quick_validate

COLD_TARGET_S, RESTART_TARGET_S = 120.0, 60.0


def wait_validated(url: str, backend: str, timeout: float) -> tuple[float | None, dict]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        ok, info = http_ready(url, timeout=1.0)
        if ok:
            v = quick_validate(url, backend)
            return time.monotonic() - t0, {"startup_s_reported": info.get("startup_s"), "validation": v}
        time.sleep(0.2)
    return None, {"error": "not ready within timeout"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/startup")
    a = ap.parse_args(argv)
    svc, cont = f"inference-{a.profile}", f"lab-inference-{a.profile}"
    url = endpoints.url("gpu" if a.profile == "gpu" else "cpu")
    rt = runtime.get()
    backend = "cuda" if a.profile == "gpu" else "cpu"
    results = []
    for k in range(1, a.trials + 1):
        if rt.mode == "docker":
            compose(["--profile", a.profile, "rm", "-sf", svc])
            rc = compose(["--profile", a.profile, "up", "-d", "--no-build", svc]).returncode
        else:  # native: stop the supervised process, then start it again (weights stay cached on disk)
            rt.stop(svc)
            rc = rt.start(svc).returncode
        secs, detail = wait_validated(url, backend, 300)
        results.append({"kind": "cold_start", "trial": k, "seconds": secs, "target_s": COLD_TARGET_S, "met": secs is not None and secs <= COLD_TARGET_S,
                        "start_rc": rc, **detail})
        time.sleep(5)
        if rt.mode == "docker":
            docker(["restart", "-t", "10", cont], 120)
        else:
            rt.rt.restart(svc)
        secs, detail = wait_validated(url, backend, 300)
        results.append({"kind": "cached_restart", "trial": k, "seconds": secs, "target_s": RESTART_TARGET_S, "met": secs is not None and secs <= RESTART_TARGET_S, **detail})
        time.sleep(5)
    out = {"run_id": new_run_id(f"startup-{a.profile}"), "created_utc": iso(), "commit": git_commit(), "profile": a.profile,
           "note": "weights are cached on disk and mounted read-only; the service never downloads", "results": results}
    write_json(a.out_dir / f"{out['run_id']}.json", out)
    for r in results:
        print(f"{r['kind']:15} trial {r['trial']}: {r['seconds'] and round(r['seconds'], 1)}s (target {r['target_s']}s) met={r['met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
