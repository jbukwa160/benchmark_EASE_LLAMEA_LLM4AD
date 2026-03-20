
from benchmark_harness.cli import run_cli

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    # your original entry point
    run_cli()
