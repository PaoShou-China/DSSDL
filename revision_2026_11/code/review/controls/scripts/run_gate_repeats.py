"""Complete the prespecified five-seed paired gate comparison sequentially."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    logs = ROOT / "experiments/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for system in ("pendulum", "quadrotor"):
        for seed in range(5):
            summary = ROOT / f"experiments/gate/{system}/seed_{seed:04d}/summary.json"
            if summary.exists():
                print("REUSE completed pair", system, seed, flush=True)
                continue
            print("START", system, seed, flush=True)
            with (logs / f"gate_{system}_{seed:04d}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(ROOT / "scripts/run_paired_ablation.py"),
                        "gate",
                        "--system",
                        system,
                        "--seed",
                        str(seed),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            if result.returncode:
                raise SystemExit(
                    f"Failed {system} seed {seed}; inspect its log; do not omit this run"
                )
            print("DONE", system, seed, flush=True)
