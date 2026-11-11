"""Fetch the monitoring binaries without Docker or root (run on a login node, which has internet).

    python -m sol.tools                     # prometheus, alertmanager, node_exporter, grafana
    python -m sol.tools prometheus alertmanager

Binaries land in .tools/<os>-<arch>/ (x86-64 on most Sol nodes, aarch64 on Grace Hopper).
Versions are taken from deploy/versions.env so native and Compose deployments use the same releases.
Prometheus, Alertmanager and node_exporter downloads are verified against the project's published
sha256sums.txt; Grafana against its .sha256 file.
"""
from __future__ import annotations

import hashlib
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from common.util import REPO_ROOT

TOOLS_ROOT = REPO_ROOT / ".tools"


def os_arch() -> tuple[str, str]:
    osn = {"Linux": "linux", "Darwin": "darwin"}[platform.system()]
    m = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[m]
    return osn, arch


def tools_dir() -> Path:
    o, a = os_arch()
    return TOOLS_ROOT / f"{o}-{a}"


def versions() -> dict[str, str]:
    """name -> version, parsed from the image tags in deploy/versions.env."""
    env = {}
    for line in (REPO_ROOT / "deploy/versions.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    tag = lambda key: env[key].rsplit(":", 1)[1].lstrip("v")  # noqa: E731
    return {"prometheus": tag("PROMETHEUS_IMAGE"), "alertmanager": tag("ALERTMANAGER_IMAGE"),
            "node_exporter": tag("NODE_EXPORTER_IMAGE"), "grafana": tag("GRAFANA_IMAGE")}


def spec(name: str) -> dict:
    o, a = os_arch()
    v = versions()[name]
    if name in ("prometheus", "alertmanager", "node_exporter"):
        proj = {"prometheus": "prometheus/prometheus", "alertmanager": "prometheus/alertmanager", "node_exporter": "prometheus/node_exporter"}[name]
        base = f"https://github.com/{proj}/releases/download/v{v}"
        fname = f"{name}-{v}.{o}-{a}.tar.gz"
        return {"url": f"{base}/{fname}", "sums": f"{base}/sha256sums.txt", "fname": fname, "version": v,
                "files": {"prometheus": ["prometheus", "promtool"], "alertmanager": ["alertmanager", "amtool"], "node_exporter": ["node_exporter"]}[name]}
    fname = f"grafana-{v}.{o}-{a}.tar.gz"
    return {"url": f"https://dl.grafana.com/oss/release/{fname}", "sums": None, "sha_url": f"https://dl.grafana.com/oss/release/{fname}.sha256",
            "fname": fname, "version": v, "files": []}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_sha(sp: dict) -> str | None:
    try:
        if sp.get("sums"):
            txt = urllib.request.urlopen(sp["sums"], timeout=60).read().decode()
            for line in txt.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip("*") == sp["fname"]:
                    return parts[0]
        elif sp.get("sha_url"):
            return urllib.request.urlopen(sp["sha_url"], timeout=60).read().decode().split()[0]
    except Exception as e:  # noqa: BLE001
        print(f"  warning: could not fetch checksum ({e})", file=sys.stderr)
    return None


def binary(name: str) -> Path | None:
    d = tools_dir()
    if name == "grafana":
        p = d / "grafana" / "bin" / "grafana"
        if not p.exists():
            p = d / "grafana" / "bin" / "grafana-server"
        return p if p.exists() else None
    p = d / name
    return p if p.exists() else None


def fetch(name: str, force: bool = False) -> Path:
    sp = spec(name)
    dest = tools_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if not force and binary(name):
        return binary(name)  # type: ignore[return-value]
    print(f"fetching {name} {sp['version']} ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        tgz = Path(td) / sp["fname"]
        urllib.request.urlretrieve(sp["url"], tgz)
        want = expected_sha(sp)
        got = sha256(tgz)
        if want is None:
            raise SystemExit(f"refusing to install {name}: no published checksum could be fetched to verify {got}")
        if want != got:
            raise SystemExit(f"checksum mismatch for {name}: expected {want}, got {got}")
        with tarfile.open(tgz) as t:
            members = t.getmembers()
            top = members[0].name.split("/")[0]
            t.extractall(td, members=[m for m in members if ".." not in m.name and not m.name.startswith("/")])
        src = Path(td) / top
        if name == "grafana":
            target = dest / "grafana"
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(src), str(target))
        else:
            for f in sp["files"]:
                shutil.copy2(src / f, dest / f)
                (dest / f).chmod(0o755)
    print(f"  ok -> {binary(name)}  (sha256 verified)")
    return binary(name)  # type: ignore[return-value]


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or ["prometheus", "alertmanager", "node_exporter", "grafana"]
    for n in names:
        if n not in versions():
            print(f"unknown tool {n}")
            return 2
        try:
            fetch(n)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {n}: {type(e).__name__}: {e}")
            if n in ("prometheus", "alertmanager"):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
