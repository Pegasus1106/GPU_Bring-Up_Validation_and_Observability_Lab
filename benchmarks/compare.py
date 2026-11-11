"""Compare a controlled-change experiment against the baseline (optimisation step).

The workload is remeasured with exactly one change (e.g. LAB_CHANNELS_LAST=1)
and the same protocol. Output checks must still pass (pass the validation JSON of
the variant); both improvements and regressions are reported. No speedup is assumed.

    python -m benchmarks.compare --base <baseline summary.json> --variant <variant summary.json> \
        --change "LAB_CHANNELS_LAST=1" --variant-validation artifacts/validation/<run>.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.aggregate import compare
from common.util import git_commit, iso, read_json, write_json


def compare_summaries(base: dict, variant: dict) -> list[dict]:
    rows = []
    for cid, bc in sorted(base["cells"].items()):
        vc = variant["cells"].get(cid)
        if vc is None:
            rows.append({"cell": cid, "status": "missing_in_variant"})
            continue
        rows.append({
            "cell": cid,
            "images_per_s": compare(bc["images_per_s"]["mean"], vc["images_per_s"]["mean"], True, "images_per_s"),
            "p95_ms": compare(bc["p95_ms"]["mean"], vc["p95_ms"]["mean"], False, "p95_ms"),
            "base_cv_images_per_s": bc["images_per_s"]["cv"],
            "variant_cv_images_per_s": vc["images_per_s"]["cv"],
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--change", required=True, help="the single controlled change, e.g. LAB_CHANNELS_LAST=1")
    ap.add_argument("--variant-validation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    val = read_json(a.variant_validation)
    rows = compare_summaries(read_json(a.base), read_json(a.variant))
    verdicts = [r[k]["verdict"] for r in rows if "images_per_s" in r for k in ("images_per_s", "p95_ms")]
    out = {
        "created_utc": iso(), "commit": git_commit(), "controlled_change": a.change,
        "output_checks_passed": bool(val.get("passed")), "validation_file": str(a.variant_validation),
        "improvements": verdicts.count("improvement"), "regressions": verdicts.count("regression"),
        "cells": rows,
        "note": "Differences within run-to-run variability (see cv fields) are not evidence of change.",
    }
    write_json(a.out, out)
    print(f"improvements={out['improvements']} regressions={out['regressions']} output_checks_passed={out['output_checks_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
