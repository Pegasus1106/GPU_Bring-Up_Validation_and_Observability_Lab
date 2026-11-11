"""nvidia-smi based Prometheus exporter that publishes DCGM-compatible metric names.

Used where DCGM Exporter cannot run (ASU Sol: no Docker, no privileged DCGM). The dashboards, benchmark
telemetry and capability discovery keep working unchanged because the metric names match.

Deviation to record: the source is `nvidia-smi`, not DCGM. XID errors and other DCGM-only fields are
therefore not exported. A field that the GPU reports as [N/A] / [Not Supported] is *omitted*, never
exported as 0, so panels read "unavailable".

    python -m ops.gpu_exporter --addr 127.0.0.1 --port 9400
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# nvidia-smi query field -> (DCGM metric name, help)
FIELDS = [
    ("utilization.gpu", "DCGM_FI_DEV_GPU_UTIL", "GPU utilization (in %)."),
    ("utilization.memory", "DCGM_FI_DEV_MEM_COPY_UTIL", "Memory utilization (in %)."),
    ("memory.used", "DCGM_FI_DEV_FB_USED", "Framebuffer memory used (in MiB)."),
    ("memory.free", "DCGM_FI_DEV_FB_FREE", "Framebuffer memory free (in MiB)."),
    ("temperature.gpu", "DCGM_FI_DEV_GPU_TEMP", "GPU temperature (in C)."),
    ("power.draw", "DCGM_FI_DEV_POWER_USAGE", "Power draw (in W)."),
    ("clocks.sm", "DCGM_FI_DEV_SM_CLOCK", "SM clock frequency (in MHz)."),
    ("clocks.mem", "DCGM_FI_DEV_MEM_CLOCK", "Memory clock frequency (in MHz)."),
    ("ecc.errors.uncorrected.volatile.total", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "Total double-bit volatile ECC errors."),
    ("ecc.errors.corrected.volatile.total", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "Total single-bit volatile ECC errors."),
]
IDENT = ["index", "uuid", "name"]
QUERY = ",".join(IDENT + [f[0] for f in FIELDS])


def is_missing(v: str) -> bool:
    v = v.strip()
    return v == "" or v.startswith("[") or v in ("N/A", "Not Supported")


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render_metrics(csv_text: str, hostname: str = "") -> str:
    """Pure function: nvidia-smi csv (noheader,nounits) -> Prometheus text. Missing values are omitted."""
    help_done: set[str] = set()
    lines: list[str] = []
    for row in csv_text.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < len(IDENT) + len(FIELDS):
            continue
        idx, uuid, name = parts[:3]
        labels = f'gpu="{esc(idx)}",UUID="{esc(uuid)}",modelName="{esc(name)}",Hostname="{esc(hostname)}"'
        for (_, metric, helptext), val in zip(FIELDS, parts[3:]):
            if is_missing(val):
                continue
            try:
                float(val)
            except ValueError:
                continue
            if metric not in help_done:
                lines += [f"# HELP {metric} {helptext}", f"# TYPE {metric} gauge"]
                help_done.add(metric)
            lines.append(f"{metric}{{{labels}}} {val}")
    return "\n".join(lines) + ("\n" if lines else "")


class Collector:
    def __init__(self, smi: str = "nvidia-smi", ttl_s: float = 1.0):
        self.smi, self.ttl = smi, ttl_s
        self._lock = threading.Lock()
        self._at = 0.0
        self._text = ""
        self._ok = 0
        import socket

        self.host = socket.gethostname()

    def collect(self) -> str:
        with self._lock:
            if time.monotonic() - self._at < self.ttl and self._at:
                return self._text
            try:
                r = subprocess.run([self.smi, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                self._ok = 1 if r.returncode == 0 else 0
                body = render_metrics(r.stdout, self.host) if r.returncode == 0 else ""
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                self._ok, body = 0, ""
            meta = [
                "# HELP gpu_exporter_scrape_ok 1 if nvidia-smi answered on the last collection.",
                "# TYPE gpu_exporter_scrape_ok gauge",
                f"gpu_exporter_scrape_ok {self._ok}",
                "# HELP gpu_exporter_info Source of the DCGM-compatible metrics (nvidia-smi, not DCGM).",
                "# TYPE gpu_exporter_info gauge",
                'gpu_exporter_info{source="nvidia-smi"} 1',
            ]
            self._text, self._at = body + "\n".join(meta) + "\n", time.monotonic()
            return self._text


def make_handler(col: Collector):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            body = col.collect().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9400)
    ap.add_argument("--smi", default="nvidia-smi")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer((a.addr, a.port), make_handler(Collector(a.smi)))
    print(f"gpu exporter (nvidia-smi source) on {a.addr}:{a.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
