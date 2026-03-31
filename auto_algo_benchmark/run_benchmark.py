import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from benchmark_harness.cli import run_cli
from benchmark_harness.config import load_config


SKIP_ENV_VAR = "AUTO_BENCHMARK_SKIP_FLAG"


def _start_skip_listener(skip_flag_path: Path):
    if not sys.stdin or not sys.stdin.isatty():
        return None

    stop_event = threading.Event()

    def worker():
        if os.name == "nt":
            import msvcrt

            while not stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch and ch.lower() == "s":
                        try:
                            skip_flag_path.parent.mkdir(parents=True, exist_ok=True)
                            skip_flag_path.write_text("skip\n", encoding="utf-8")
                            print("Skip requested: the current candidate/generation will be skipped at the next safe checkpoint.")
                        except Exception:
                            pass
                time.sleep(0.05)
        else:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not stop_event.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        ch = sys.stdin.read(1)
                        if ch and ch.lower() == "s":
                            try:
                                skip_flag_path.parent.mkdir(parents=True, exist_ok=True)
                                skip_flag_path.write_text("skip\n", encoding="utf-8")
                                print("Skip requested: the current candidate/generation will be skipped at the next safe checkpoint.")
                            except Exception:
                                pass
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print("Press 'S' in the terminal to skip the current candidate/generation.")
    return stop_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = Path(args.config).resolve()
    skip_flag_path = config_path.parent / ".benchmark_skip_current.flag"
    try:
        if skip_flag_path.exists():
            skip_flag_path.unlink()
    except Exception:
        pass
    os.environ[SKIP_ENV_VAR] = str(skip_flag_path)

    stop_event = _start_skip_listener(skip_flag_path)

    sys.argv = ["run_benchmark.py", "--config", args.config]
    if args.append:
        sys.argv.append("--append")

    try:
        run_cli()
    finally:
        if stop_event is not None:
            stop_event.set()
        try:
            if skip_flag_path.exists():
                skip_flag_path.unlink()
        except Exception:
            pass

    if not args.skip_analysis:
        output_dir_cfg = cfg.get("output_dir", "benchmark_results")
        results_dir = (config_path.parent / output_dir_cfg).resolve()

        print(f"Benchmark finished. Running analysis on: {results_dir}")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_benchmark.py")),
                "--results-dir",
                str(results_dir),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
