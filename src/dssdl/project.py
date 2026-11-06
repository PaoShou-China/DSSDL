"""Locate recorded experiments without requiring a particular working directory."""

import os
from pathlib import Path

RELEASE = "revision_2026_11"


def find_repository(explicit: str | None = None) -> Path:
    """Resolve a checkout, honoring an explicit path before automatic discovery."""
    requested = explicit or os.environ.get("DSSDL_REPO_ROOT")
    if requested:
        candidates = [Path(requested).expanduser().resolve()]
    else:
        cwd = Path.cwd()
        candidates = [Path(__file__).resolve().parents[2], cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / RELEASE / "verify.py").is_file() and (
            candidate / RELEASE / "code/run.py"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "DSSDL experiment archive not found. Clone the repository, install it with "
        "'python -m pip install -e .', or pass --repo-root PATH before the command. "
        "The lightweight Python wheel does not include experimental data."
    )
