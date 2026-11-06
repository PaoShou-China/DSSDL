"""Run the four remaining fixed-teacher feedback pairs without overwriting results."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    logs = ROOT / "experiments/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed in range(5800, 5805):
        summary = ROOT / f"experiments/separation/pendulum/seed_{seed:04d}/summary.json"
        if summary.exists():
            print("REUSE completed pair", seed, flush=True)
            continue
        print("START regularizer pendulum", seed, flush=True)
        with (logs / f"regularizer_pendulum_{seed:04d}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/run_paired_ablation.py"),
                    "separation",
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
            raise SystemExit(f"Failed seed {seed}; inspect log and retain the run")
        print("DONE regularizer pendulum", seed, flush=True)
