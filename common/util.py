"""Shared helpers: UTC timestamps, JSONL I/O, hashing, git commit lookup.

Convention (spec): monotonic clocks for elapsed durations inside one process,
UTC wall-clock for correlation across processes. Never subtract monotonic
values taken on different machines.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

_jsonl_lock = threading.Lock()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    """UTC ISO-8601 with millisecond precision and a trailing Z."""
    dt = dt or utc_now()
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def epoch_to_iso(ts: float) -> str:
    return iso(datetime.fromtimestamp(ts, tz=timezone.utc))


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def append_jsonl(path: str | os.PathLike, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=False, default=str)
    with _jsonl_lock:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a torn last line while a writer is active
    return out


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, p)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_commit(repo: Path = REPO_ROOT) -> str:
    env = os.environ.get("LAB_GIT_COMMIT")
    if env:
        return env
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=5
        )
        sha = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if dirty:
            sha += "+dirty"
    except Exception:
        pass
    return sha


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
