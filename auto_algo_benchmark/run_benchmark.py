import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from benchmark_harness.cli import run_cli
from benchmark_harness.config import load_config


def _write_skip_signal(path: Path, skip_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"skip_count": int(skip_count), "updated_at": time.time()}, indent=2), encoding="utf-8")


def _start_skip_key_listener(skip_signal_path: Path):
    if not sys.stdin or not sys.stdin.isatty():
        print("Terminal skip listener disabled because stdin is not an interactive TTY.")
        return None

    stop_event = threading.Event()

    def notify_skip(skip_count: int) -> None:
        _write_skip_signal(skip_signal_path, skip_count)
        print(f"\nSkip requested for the current generation/candidate (request {skip_count}).")
        print("The active evaluation will be penalized at the next safe checkpoint, then the run will continue.")

    def windows_listener() -> None:
        import msvcrt

        skip_count = 0
        while not stop_event.is_set():
            if msvcrt.kbhit():
                try:
                    ch = msvcrt.getwch()
                except Exception:
                    ch = ""
                if ch in ("s", "S"):
                    skip_count += 1
                    notify_skip(skip_count)
            time.sleep(0.1)

    def posix_listener() -> None:
        import select
        import termios
        import tty

        skip_count = 0
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                try:
                    ch = sys.stdin.read(1)
                except Exception:
                    ch = ""
                if ch in ("s", "S"):
                    skip_count += 1
                    notify_skip(skip_count)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    target = windows_listener if os.name == "nt" else posix_listener
    listener = threading.Thread(target=target, name="benchmark-skip-listener", daemon=True)
    listener.start()

    def stop_listener() -> None:
        stop_event.set()
        listener.join(timeout=1.0)

    print("Press 'S' in the terminal to skip the current generation/candidate.")
    return stop_listener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    config_path = Path(args.config).resolve()
    skip_signal_path = Path(tempfile.gettempdir()) / f"benchmark_skip_signal_{os.getpid()}.json"
    _write_skip_signal(skip_signal_path, 0)
    os.environ["BENCHMARK_SKIP_SIGNAL_FILE"] = str(skip_signal_path)

    stop_listener = _start_skip_key_listener(skip_signal_path)
    if stop_listener is not None:
        atexit.register(stop_listener)

    original_argv = sys.argv[:]
    try:
        sys.argv = ["run_benchmark.py", "--config", args.config]
        if args.append:
            sys.argv.append("--append")
        run_cli()
    finally:
        sys.argv = original_argv
        if stop_listener is not None:
            stop_listener()
        try:
            skip_signal_path.unlink(missing_ok=True)
        except Exception:
            pass

    if args.skip_analysis:
        return

    output_dir_cfg = cfg.get("output_dir", "benchmark_results")
    results_dir = (config_path.parent / output_dir_cfg).resolve()
    print(f"Benchmark finished. Running analysis on: {results_dir}")

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze_benchmark.py")),
            "--results-dir",
            str(results_dir),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"Analysis step failed with exit code {completed.returncode}. The benchmark results are still saved in: {results_dir}"
        )


if __name__ == "__main__":
    main()
