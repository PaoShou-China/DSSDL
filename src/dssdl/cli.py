"""Command-line entry point for installation checks and reproducible experiments."""

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import __version__
from .project import RELEASE, find_repository
from .tasks import TASKS, command_for

DEPENDENCIES = (
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "sympy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "higher",
)


def environment_report(root: Path) -> dict:
    """Inspect dependencies without importing heavy scientific packages."""
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name, version in versions.items() if version is None]
    return {
        "repository": str(root),
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": versions,
        "missing": missing,
        "ready": not missing and sys.version_info >= (3, 10),
        "install": 'python -m pip install -e ".[experiments]"',
    }


def execute(command: list[str], root: Path, dry_run: bool = False) -> int:
    """Run the recorded driver in a separate process, preserving its exit status."""
    print("Command:", shlex.join(command), flush=True)
    if dry_run:
        return 0
    if "--help" not in command:
        report = environment_report(root)
        if report["missing"]:
            print(
                "Missing experiment dependencies: " + ", ".join(report["missing"]), file=sys.stderr
            )
            print(report["install"], file=sys.stderr)
            return 2
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(command, cwd=root / RELEASE, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    """Return a process exit code; never load training modules for help or listing."""
    parser = argparse.ArgumentParser(prog="dssdl", description=__doc__)
    parser.add_argument("--version", action="version", version=f"DSSDL {__version__}")
    parser.add_argument("--repo-root", help="Path to a DSSDL repository checkout")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List supported experiments")
    doctor = sub.add_parser("doctor", help="Check data location and installed dependencies")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable diagnostics")
    sub.add_parser("verify", help="Check file hashes and recompute saved numerical evidence")
    sub.add_parser("check", help="Load models and validate symbolic gradients without training")
    run = sub.add_parser("run", help="Run an experiment; use 'run TASK --help' for its options")
    run.add_argument("task", choices=TASKS)
    run.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    # Forward task help to the actual scientific driver, not a duplicated option list.
    values = list(sys.argv[1:] if argv is None else argv)
    if "run" in values:
        idx = values.index("run")
        if len(values) > idx + 1 and values[idx + 1] in TASKS and "--help" in values[idx + 2 :]:
            task_help = True
            values.remove("--help")
        else:
            task_help = False
    else:
        task_help = False
    args, remaining = parser.parse_known_args(values)
    if (
        args.command != "run"
        and remaining
        and not (args.command == "verify" and remaining == ["--checksums"])
    ):
        parser.error("unrecognized arguments: " + " ".join(remaining))
    try:
        if args.command == "list":
            for name, task in TASKS.items():
                print(f"{name:24} {task.description}")
            return 0
        root = find_repository(args.repo_root)
        if args.command == "doctor":
            report = environment_report(root)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"DSSDL {__version__} | Python {report['python']} | {report['platform']}")
                print("Repository:", root)
                for name, version in report["dependencies"].items():
                    print(f"  {name:16} {version or 'MISSING'}")
                print("Ready for saved-result checks." if report["ready"] else report["install"])
                print("New symbolic searches additionally require PySR and Julia.")
            return 0 if report["ready"] else 1
        if args.command == "run":
            forwarded = ["--help"] if task_help else remaining
            return execute(command_for(root, args.task, forwarded), root, args.dry_run)
        suffix = (
            ["verify.py", "--checksums"] if args.command == "verify" else ["code/run.py", "check"]
        )
        command = [sys.executable, "-B", str(root / RELEASE / suffix[0]), *suffix[1:]]
        return execute(command, root)
    except (FileNotFoundError, PermissionError) as error:
        print(f"DSSDL: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; reference results were not overwritten.", file=sys.stderr)
        return 130
