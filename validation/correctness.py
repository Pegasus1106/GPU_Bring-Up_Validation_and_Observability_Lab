"""Model-level correctness validation (run inside the workload container).

    python -m validation.correctness --backend cuda          # acceptance run
    python -m validation.correctness --backend cuda --calibrate
    python -m validation.correctness --freeze                # freeze reviewed tolerances

Checks (per-case results are written to JSON with per-input errors):
  C01 backend executes where requested (no silent fallback), TF32 state recorded
  C02 logits vs CPU FP32 reference at batch 1, 8, 32 (all inputs)
  C03 batch vs individual prediction consistency
  C04 top-1 agreement (policy in tolerances.json)
  C05 missing weights / checksum mismatch / corrupt weights fail with the right reason code
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

from app.config import Settings
from app.errors import LabError
from app.model import ResNet18Adapter
from common.util import REPO_ROOT, git_commit, iso, new_run_id, read_json, sha256_file, write_json
from validation.reference import (
    TOL_PATH,
    compare,
    load_reference,
    load_tolerances,
    top1_disagreement_allowed,
)


def _run_batched(ad: ResNet18Adapter, ids: list[str], bs: int):
    out, backends = [], set()
    for k in range(0, len(ids), bs):
        r = ad.infer(ids[k : k + bs])
        out.append(r.logits)
        backends.add(r.backend_actual)
    return np.concatenate(out), backends


def case_result(cid: str, name: str, status: str, reason: str = "", **metrics) -> dict:
    return {"id": cid, "name": name, "status": status, "reason": reason, "metrics": metrics}


def check_weights_failures(base: Settings) -> list[dict]:
    """C05: intentional bad-weights inputs must fail clearly with specific reason codes."""
    results = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        garbage = td / "garbage.pth"
        garbage.write_bytes(b"not a checkpoint")
        import json

        good_manifest = read_json(base.model_manifest)
        corrupt_manifest = td / "corrupt_model.json"
        corrupt_manifest.write_text(json.dumps({**good_manifest, "sha256": sha256_file(garbage)}))
        cases = [
            ("missing", dict(model_path=str(td / "nope.pth")), "weights_missing"),
            ("checksum", dict(model_path=str(garbage)), "weights_checksum_mismatch"),
            ("corrupt", dict(model_path=str(garbage), model_manifest=str(corrupt_manifest)), "weights_corrupt"),
        ]
        for name, over, expect in cases:
            # always test on cpu: this is about artefact handling, not the device
            s = dataclasses.replace(base, backend="cpu", **over)
            try:
                ResNet18Adapter(s).load()
                results.append(case_result(f"C05-{name}", f"bad weights: {name}", "FAIL", "load() succeeded unexpectedly"))
            except LabError as e:
                ok = e.reason_code == expect
                results.append(
                    case_result(
                        f"C05-{name}",
                        f"bad weights: {name}",
                        "PASS" if ok else "FAIL",
                        e.message if ok else f"expected {expect}, got {e.reason_code}: {e.message}",
                        reason_code=e.reason_code,
                    )
                )
            except Exception as e:  # noqa: BLE001
                results.append(case_result(f"C05-{name}", f"bad weights: {name}", "ERROR", f"{type(e).__name__}: {e}"))
    return results


def run(backend: str, calibrate: bool, out_dir: Path, max_ids: int | None = None) -> tuple[dict, int]:
    tol = load_tolerances()
    ref = load_reference()
    atol, rtol = float(tol["atol"]), float(tol["rtol"])
    s = dataclasses.replace(Settings.from_env(), backend=backend)
    ad = ResNet18Adapter(s)
    cases: list[dict] = []
    per_input: dict[str, dict] = {}

    try:
        ad.load()
    except LabError as e:
        cases.append(case_result("C00", "load model", "FAIL", f"{e.reason_code}: {e.message}"))
        return _finish(backend, tol, cases, per_input, out_dir, calibrate, s), 1

    info = ad.info()
    ids = ad.input_ids
    if calibrate:
        ids = ids[:64]
    if max_ids:
        ids = ids[:max_ids]
    missing = [i for i in ids if i not in ref.index]
    if missing:
        cases.append(case_result("C00", "reference covers inputs", "FAIL", f"reference missing {len(missing)} ids; run `make reference`"))
        return _finish(backend, tol, cases, per_input, out_dir, calibrate, s), 1
    ref_rows = np.stack([ref.logits[ref.row(i)] for i in ids])

    # C01 backend + precision state
    r1 = ad.infer(ids[:1])
    want = "cuda" if backend == "cuda" else "cpu"
    ok = r1.backend_actual == want and not info["tf32_matmul"] and not info["tf32_cudnn"]
    cases.append(
        case_result(
            "C01", "executed on requested backend; TF32 disabled", "PASS" if ok else "FAIL",
            "" if ok else f"backend_actual={r1.backend_actual}, tf32={info['tf32_matmul']}/{info['tf32_cudnn']}",
            backend_actual=r1.backend_actual, tf32_matmul=info["tf32_matmul"], tf32_cudnn=info["tf32_cudnn"],
            torch=info["torch_version"], cuda=info.get("cuda"),
        )
    )

    # C02 + C04 per batch size
    individual = None
    for bs in (1, 8, 32):
        got, backends = _run_batched(ad, ids, bs)
        if bs == 1:
            individual = got
        errs = [compare(got[k], ref_rows[k], atol, rtol) for k in range(len(ids))]
        for k, e in enumerate(errs):
            per_input.setdefault(ids[k], {})[f"bs{bs}"] = {
                "max_abs_error": e["max_abs_error"], "normalized_error": e["normalized_error"],
                "within_tolerance": e["within_tolerance"],
            }
        n_fail = sum(not e["within_tolerance"] for e in errs)
        max_abs = max(e["max_abs_error"] for e in errs)
        max_norm = max(e["normalized_error"] for e in errs)
        worst = max(errs, key=lambda e: e["max_abs_error"])
        status = "PASS" if n_fail == 0 and backends == {want} else "FAIL"
        cases.append(
            case_result(
                f"C02-bs{bs}", f"logits within tolerance vs CPU FP32 reference, batch={bs}", status,
                "" if status == "PASS" else f"{n_fail}/{len(ids)} inputs exceed atol={atol} rtol={rtol}",
                n_inputs=len(ids), n_outside_tolerance=n_fail, max_abs_error=max_abs,
                max_normalized_error=max_norm, atol=atol, rtol=rtol, backends=sorted(backends),
                worst_input=ids[errs.index(worst)],
            )
        )
        # top-1 agreement with explicit policy
        flips = [k for k, e in enumerate(errs) if e["top1_actual"] != e["top1_ref"]]
        unexplained = []
        for k in flips:
            m = ref.margin[ref.row(ids[k])]
            if not top1_disagreement_allowed(m, float(ref_rows[k].max()), atol, rtol):
                unexplained.append(ids[k])
        agree = 1.0 - len(flips) / len(ids)
        cases.append(
            case_result(
                f"C04-bs{bs}", f"top-1 agreement vs reference, batch={bs}", "PASS" if not unexplained else "FAIL",
                "" if not unexplained else f"unexplained top-1 flips: {unexplained[:5]}",
                top1_agreement=agree, flips=[ids[k] for k in flips], unexplained_flips=unexplained,
            )
        )

    # C03 batch vs individual consistency (same backend, same tolerance)
    batched, _ = _run_batched(ad, ids, 32)
    cons = [compare(batched[k], individual[k], atol, rtol) for k in range(len(ids))]
    n_bad = sum(not c["within_tolerance"] for c in cons)
    cases.append(
        case_result(
            "C03", "batch(32) vs individual predictions consistent", "PASS" if n_bad == 0 else "FAIL",
            "" if n_bad == 0 else f"{n_bad} inputs differ beyond tolerance",
            max_abs_error=max(c["max_abs_error"] for c in cons), n_outside_tolerance=n_bad,
        )
    )

    cases.extend(check_weights_failures(Settings.from_env()))
    return _finish(backend, tol, cases, per_input, out_dir, calibrate, s), 0


def _finish(backend, tol, cases, per_input, out_dir, calibrate, s) -> dict:
    failed = [c for c in cases if c["status"] in ("FAIL", "ERROR")]
    frozen = tol.get("status") == "frozen"
    report = {
        "run_id": new_run_id(f"validation-{backend}{'-calibration' if calibrate else ''}"),
        "created_utc": iso(),
        "commit": git_commit(),
        "backend": backend,
        "mode": "calibration" if calibrate else "acceptance",
        "tolerances": tol,
        "tolerances_frozen": frozen,
        "passed": (not failed) and frozen and not calibrate,
        "checks_passed": not failed,
        "note": (
            "calibration run: informational, does not count toward acceptance" if calibrate
            else ("" if frozen else "tolerances are still 'candidate': AC03 cannot pass until frozen")
        ),
        "cases": cases,
        "per_input_errors": per_input,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['run_id']}.json"
    write_json(path, report)
    report["_path"] = str(path)
    return report


def freeze() -> None:
    tol = load_tolerances()
    tol.update(status="frozen", frozen_utc=iso(), frozen_commit=git_commit())
    write_json(TOL_PATH, tol)
    print(f"tolerances frozen at {tol['frozen_utc']} (commit {tol['frozen_commit']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/validation")
    ap.add_argument("--max-ids", type=int, default=None, help="subset for quick smoke runs (not acceptance)")
    a = ap.parse_args(argv)
    if a.freeze:
        freeze()
        return 0
    report, rc = run(a.backend, a.calibrate, a.out_dir, a.max_ids)
    for c in report["cases"]:
        print(f"[{c['status']:5}] {c['id']:10} {c['name']}  {c['reason']}")
    print(f"-> {report['_path']}  checks_passed={report['checks_passed']} frozen={report['tolerances_frozen']}")
    return 0 if report["checks_passed"] and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
