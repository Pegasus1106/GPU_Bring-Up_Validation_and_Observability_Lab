"""Local Alertmanager webhook receiver. Writes one JSONL event per alert to
artifacts/alerts/alerts.jsonl. Nothing is sent externally.

    python -m ops.alert_receiver --addr 127.0.0.1 --port 9094 --out artifacts/alerts/alerts.jsonl
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common.util import REPO_ROOT, append_jsonl, iso


def flatten(payload: dict, received_utc: str) -> list[dict]:
    """One record per alert: status, name, labels, startsAt/endsAt from Alertmanager,
    and the local receipt time (the 'local notification receipt' for incident timing)."""
    recs = []
    for a in payload.get("alerts", []):
        recs.append({
            "received_utc": received_utc,
            "status": a.get("status"),
            "alertname": a.get("labels", {}).get("alertname"),
            "labels": a.get("labels", {}),
            "annotations": a.get("annotations", {}),
            "starts_at": a.get("startsAt"),
            "ends_at": a.get("endsAt"),
            "group_key": payload.get("groupKey"),
            "receiver": payload.get("receiver"),
        })
    return recs


def make_handler(out: Path):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return
            for rec in flatten(payload, iso()):
                append_jsonl(out, rec)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"alert receiver up")

        def log_message(self, *a):  # keep stdout quiet
            pass

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9094)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/alerts/alerts.jsonl")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer((a.addr, a.port), make_handler(a.out))
    print(f"alert receiver on {a.addr}:{a.port} -> {a.out}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
