"""A small registry connecting stable user commands to recorded experiment drivers."""

import sys
from dataclasses import dataclass
from pathlib import Path

from .project import RELEASE


@dataclass(frozen=True)
class Task:
    """An experiment entry point; the underlying driver owns scientific options."""

    script: str
    prefix: tuple[str, ...]
    description: str
    trains: bool = False


TASKS: dict[str, Task] = {
    "wasserstein": Task(
        "reproduce.py", ("wasserstein",), "Recompute original-trajectory distances"
    ),
    "online": Task("reproduce.py", ("online",), "Replay the recorded time-varying scenarios"),
    "budget": Task("reproduce.py", ("budget",), "Run fixed-budget target adaptation", True),
    "gate": Task("code/run.py", ("gate",), "Run a paired dimension-selection control", True),
    "regularizer-pendulum": Task(
        "code/run.py", ("regularizer-pendulum",), "Run a measured-pendulum regularizer pair", True
    ),
    "regularizer-lorenz": Task(
        "code/run.py", ("regularizer-lorenz",), "Run a Lorenz regularizer pair", True
    ),
    "discover": Task("code/run.py", ("discover",), "Launch a new FHN/Lorenz PySR search", True),
    "feedback": Task(
        "code/run.py", ("feedback",), "Retrain with a recorded consolidated form", True
    ),
    "pendulum": Task("code/run.py", ("pendulum",), "Measured-pendulum discovery or feedback", True),
    "all-controls": Task("code/run.py", ("all-controls",), "Run all paired control repeats", True),
}


def command_for(root: Path, task: str, arguments: list[str]) -> list[str]:
    """Build a subprocess argument list without shell expansion or equation changes."""
    spec = TASKS[task]
    return [sys.executable, "-B", str(root / RELEASE / spec.script), *spec.prefix, *arguments]
