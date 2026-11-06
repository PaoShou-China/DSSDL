"""Fast checks that do not need PyTorch, GPUs, Julia or model downloads."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dssdl.cli import main
from dssdl.project import find_repository
from dssdl.tasks import TASKS, command_for

ROOT = Path(__file__).resolve().parents[1]


def test_repository_discovery_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_repository() == ROOT


def test_explicit_missing_repository_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="archive not found"):
        find_repository(str(tmp_path))


def test_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSDL_REPO_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        find_repository()


@pytest.mark.parametrize("task", TASKS)
def test_all_registered_scripts_exist(task):
    command = command_for(ROOT, task, ["--help"])
    assert Path(command[2]).is_file()
    assert command[0] == sys.executable
    assert command[-1] == "--help"


def test_dry_run_does_not_execute(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("executed a dry run"))
    assert main(["run", "online", "--dry-run", "--system", "fhn"]) == 0
    assert "--system fhn" in capsys.readouterr().out


def test_help_is_forwarded_to_the_driver(monkeypatch):
    captured = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: captured.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    assert main(["run", "online", "--help"]) == 0
    assert captured[0][-2:] == ["online", "--help"]


def test_driver_failure_propagates(monkeypatch):
    monkeypatch.setattr("dssdl.cli.environment_report", lambda root: {"missing": []})
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 7))
    assert main(["run", "online", "--system", "fhn"]) == 7


def test_missing_dependencies_are_actionable(monkeypatch, capsys):
    monkeypatch.setattr(
        "dssdl.cli.environment_report",
        lambda root: {"missing": ["torch"], "install": 'python -m pip install -e ".[experiments]"'},
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("started without dependencies")
    )
    assert main(["run", "online"]) == 2
    assert "Missing experiment dependencies: torch" in capsys.readouterr().err


def test_invalid_nonexperiment_option_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["list", "--checksums"])
    assert exc.value.code == 2


def test_list_does_not_require_scientific_imports(capsys):
    assert main(["list"]) == 0
    assert "wasserstein" in capsys.readouterr().out


def test_doctor_machine_readable(capsys):
    assert main(["doctor", "--json"]) in (0, 1)
    assert "dependencies" in json.loads(capsys.readouterr().out)


def test_no_shell_interpolation():
    command = command_for(ROOT, "online", ["--name", "a;echo injected"])
    assert command[-1] == "a;echo injected"


def test_unknown_task_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["run", "unknown"])
    assert exc.value.code == 2
