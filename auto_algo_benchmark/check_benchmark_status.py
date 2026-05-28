#!/usr/bin/env python3
"""
Inspect benchmark progress via heartbeat.json, or serve a tiny status page in the browser.

Examples:
  python check_benchmark_status.py
  python check_benchmark_status.py --output-dir ../benchmark_results
  python check_benchmark_status.py --serve --port 8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
def _default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmark_results"


def load_heartbeat(path: Path) -> dict:
    if not path.exists():
        return {
            "error": "heartbeat.json not found",
            "hint": f"Expected at {path}. Is the benchmark running or is --output-dir wrong?",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"invalid JSON: {exc}", "path": str(path)}


def format_text(hb: dict) -> str:
    lines = []
    if "error" in hb and "status" not in hb:
        lines.append(f"ERROR: {hb.get('error')}")
        if "hint" in hb:
            lines.append(hb["hint"])
        return "\n".join(lines)

    st = hb.get("status", "?")
    lines.append(f"status:           {st}")
    if st == "finished":
        lines.append(f"finished_at:      {hb.get('finished_at', '')}")
        lines.append(f"failed:           {hb.get('failed', '')}")
        lines.append(f"skipped:          {hb.get('skipped', '')}")
        lines.append(f"total_time_sec:   {hb.get('total_time_sec', '')}")
        lines.append(f"runs_completed:   {hb.get('runs_completed', '')}")
        return "\n".join(lines)

    if st == "interrupted":
        lines.append(f"finished_at:      {hb.get('finished_at', '')}")
        lines.append(f"failed:           {hb.get('failed', '')}")
        lines.append(f"skipped:          {hb.get('skipped', '')}")
        lines.append(f"total_time_sec:   {hb.get('total_time_sec', '')}")
        lines.append(f"runs_completed:   {hb.get('runs_completed', '')}")
        lines.append(f"note:             {hb.get('note', '')}")
        return "\n".join(lines)

    lines.append(f"pid:              {hb.get('pid', '')}")
    tot = hb.get("total", "?")
    if hb.get("continuous"):
        lines.append(f" lap:             {hb.get('lap', '?')}")
    lines.append(f"progress:         {hb.get('completed_counter', '?')} / {tot}")
    lines.append(f"current:          {hb.get('framework', '')} | {hb.get('task', '')} | seed={hb.get('seed', '')}")
    lines.append(f"started_at:       {hb.get('started_at', '')}")
    lines.append(f"last_alive_at:    {hb.get('last_alive_at', '(waiting for first periodic heartbeat)')}")
    lines.append(f"heartbeat #       {hb.get('heartbeat_seq', '-')}")
    lines.append(f"parent_uptime_s:  {hb.get('parent_uptime_sec', '-')}")
    lines.append(f"interval_sec:     {hb.get('heartbeat_interval_sec', '-')}")
    if "meta" in hb:
        lines.append(f"meta:             {hb['meta']}")
    lines.append("")
    lines.append(hb.get("note", ""))
    return "\n".join(lines)


class _StatusHandler(BaseHTTPRequestHandler):
    output_dir: Path = Path(".")

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        hb_path = self.output_dir / "heartbeat.json"
        hb = load_heartbeat(hb_path)
        raw = json.dumps(hb, indent=2, ensure_ascii=False)

        if path in ("/api", "/api/heartbeat", "/heartbeat.json"):
            body = raw.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta http-equiv="refresh" content="60"/>
  <title>Benchmark status</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 1.5rem; max-width: 56rem; }}
    pre {{ background: #f6f8fa; padding: 1rem; overflow: auto; border-radius: 8px; }}
    button, a.btn {{ display: inline-block; padding: 0.5rem 1rem; margin: 0.25rem 0.5rem 0.25rem 0;
      background: #0969da; color: #fff; border: none; border-radius: 6px; cursor: pointer;
      text-decoration: none; font-size: 1rem; }}
    .muted {{ color: #57606a; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Benchmark status</h1>
  <p class="muted">Page auto-refreshes every 60 seconds. Use the button to refresh immediately.</p>
  <p>
    <a class="btn" href="/">Refresh now</a>
    <a class="btn" href="/api/heartbeat">Raw JSON</a>
  </p>
  <pre id="hb">{raw}</pre>
  <p class="muted">Output directory: {self.output_dir}</p>
</body>
</html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    p = argparse.ArgumentParser(description="Check benchmark heartbeat / serve status page.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory containing heartbeat.json (default: ../benchmark_results next to harness)",
    )
    p.add_argument("--serve", action="store_true", help="Serve http://127.0.0.1:PORT/ with live status")
    p.add_argument("--host", default="127.0.0.1", help="Bind address for --serve")
    p.add_argument("--port", type=int, default=8765, help="Port for --serve (default 8765)")
    args = p.parse_args()

    out = args.output_dir
    if out is None:
        out = _default_results_dir()
    else:
        out = out.resolve()

    hb_path = out / "heartbeat.json"

    if not args.serve:
        hb = load_heartbeat(hb_path)
        print(format_text(hb))
        if "error" in hb and hb_path.exists() is False:
            return 1
        return 0

    _StatusHandler.output_dir = out
    server = HTTPServer((args.host, args.port), _StatusHandler)
    print(f"Serving benchmark status at http://{args.host}:{args.port}/")
    print(f"Watching: {hb_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
