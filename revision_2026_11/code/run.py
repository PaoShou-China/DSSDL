"""Small launch/verification interface; learning algorithms remain in their source modules."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
EXP = ROOT / "review/experiments"
CONTROL = ROOT / "review/controls"


def load(name, path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum():
    checks = read(PACKAGE / "SHA256SUMS.json")
    for name, expected in checks.items():
        path = PACKAGE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Missing/changed package file: {name}")
    print(f"CHECKSUM OK: {len(checks)} files")


def verify_results():
    import numpy as np
    folder = PACKAGE / "results/controls/pairs"
    pairs = sorted(folder.glob("*/*/seed_*/summary.json"))
    assert len(pairs) == 20
    n = 0
    maximum_error = 0.
    for path in pairs:
        data = read(path)
        assert data["status"] == "completed" and data["protected_inputs_unchanged"]
        assert len({r["initialization_hash"] for r in data["results"].values()}) == 1
        reference = None
        for variant, row in data["results"].items():
            with np.load(path.parent / f"{variant}_predictions.npz") as p:
                actual = float(np.sqrt(np.mean((p["prediction"].astype(float) - p["reference"]) ** 2)))
                error = abs(actual - row["query_nrmse_after"])
                assert error < 2e-7, (path, variant, actual)
                maximum_error = max(maximum_error, error)
                if reference is not None:
                    np.testing.assert_array_equal(reference, p["reference"])
                reference = p["reference"].copy()
            assert row["steps"] == 500 and not row["query_used_for_stopping"]
            n += 1
    print(f"RESULTS OK: {len(pairs)} pairs, {n} variants, max target-RMSE difference {maximum_error:.3g}")


def check():
    import numpy as np
    import sympy as sp
    import torch
    torch.set_num_threads(2)
    paths = list(ROOT.rglob("*.py"))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    base = load("lecture_feedback", EXP / "symbolic_feedback/scripts/run_feedback.py")
    for system, dataset_name in (("fhn", "FHN"), ("lorenz", "lorenz")):
        _, dataset, module, _, _ = base.old.shared.multisystem.import_dssdl_for_system(dataset_name)
        folder = base.ROOT / "runs/pysr" / system
        spec = read(folder / "first_round_simplification.json")
        prior = base.CompactPrior(system, [sp.sympify(e) for e in spec["new_templates"]],
            spec["shared_coefficient_initial_values"], spec["coefficient_center"], spec["coefficient_scale"], dataset)
        checkpoint = ROOT / read(base.P42 / "runs" / system / "run_manifest.json")["packaged_source_checkpoint"]
        model = base.old.model_for(module, dataset, dict(device="cpu", hidden_widths=[64, 64], specific_param_dim=3),
                                   checkpoint, base.old.runner.ZeroPrior(len(base.old.NAMES[system])))
        model.f_hat_func = prior
        model.load_state_dict(torch.load(folder / "compact_model.pt", map_location="cpu", weights_only=True), strict=True)
        raw_x = np.asarray(dataset["raw"]["train"][0][0][:16])
        q = model.one_hot_to_specific_param.weight[:, 0].detach().numpy()
        x = torch.tensor(np.column_stack([raw_x, np.tile(q, (len(raw_x), 1))]), dtype=torch.float32, requires_grad=True)
        prediction = prior(x, use_raw_data=True)
        np.testing.assert_allclose(prediction.detach().numpy(), prior.numpy(x.detach().numpy()), rtol=3e-5, atol=3e-5)
        prediction.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        print(f"MODEL OK: {system}; {dataset['info']['file_counts']['train']} source tasks; prior gradients finite")
    assert not base.old.TRIPWIRE_CALLS
    pend = load("lecture_pendulum", EXP / "measured_pendulum_closure/scripts/run_closure.py")
    env = pend.original_module()
    dataset, records = pend.source_data()
    spec = read(pend.ROOT / "simplification.json")
    model = pend.make_model(env, dataset, pend.ROOT / "feedback/model.pt", "cpu", spec)
    assert torch.isfinite(next(model.parameters())).all()
    print(f"MODEL OK: real pendulum; {len(records)} source sequences; feedback checkpoint loads")
    # Confirm every included paper-system dataset can be loaded using only this tree.
    datasets = sys.modules["datasets"]
    for name in ("pendulum_real", "quadrotor", "flow-19", "flow-19-v", "flow-29-v"):
        datasets.set_current_system(name)
        assert datasets.dataset_dict is not None
        print(f"DATA OK: {name}")
    gate = load("lecture_gate_preflight", EXP / "scripts/dimension_discovery_claim_runner.py")
    for system in ("pendulum", "quadrotor"):
        bundle = gate.load_data(system, seed=0, val_fraction=.2, device=torch.device("cpu"))
        assert len(bundle.train_x) > 0 and len(bundle.target_support_x) > 0
        print(f"GATE INPUT OK: {system}; {len(bundle.train_x)} source tasks")
    lor = load("lecture_lorenz_preflight", CONTROL / "scripts/run_lorenz_regularizer.py")
    lor.prepare("cpu")
    print(f"CHECK OK: {len(paths)} Python files parsed; closure models, all packaged datasets and Lorenz control preflight passed")


def execute(script, args):
    subprocess.run([sys.executable, "-B", str(script), *args], cwd=ROOT,
                   env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"), check=True)


def fresh_output(name):
    out = ROOT / "outputs" / name
    if out.exists():
        raise SystemExit(f"Output already exists; keep it and choose another --name: {out}")
    out.mkdir(parents=True)
    return out


def discovery(args):
    import torch
    base = load("lecture_search", EXP / "label_free_discovery/scripts/run_closure.py")
    base.seed_all(0)
    torch.set_num_threads(2)
    system = args.system
    dataset_name = "FHN" if system == "fhn" else "lorenz"
    _, dataset, module, pretrain, _ = base.shared.multisystem.import_dssdl_for_system(dataset_name)
    pretrain.CONFIG_OVERRIDES["reg_rate"] = 0.
    provenance = read(base.ROOT / "runs" / system / "run_manifest.json")
    checkpoint = ROOT / provenance["packaged_source_checkpoint"]
    model = base.model_for(module, dataset, dict(device=args.device, hidden_widths=[64, 64], specific_param_dim=3),
                           checkpoint, base.runner.ZeroPrior(len(base.NAMES[system])))
    out = fresh_output(args.name or f"first_search_{system}")
    # Same search and selection functions/budget as the recorded first round.
    base.discover(model, dataset, system, out / "round_01/discovery", 1, 45)
    assert not base.TRIPWIRE_CALLS
    print(f"First search written to {out}; compare its equations with the recorded simplification before feedback")


def feedback(args):
    import shutil
    base = load("lecture_feedback_run", EXP / "symbolic_feedback/scripts/run_feedback.py")
    saved_root = base.ROOT
    out = fresh_output(args.name or f"feedback_{args.system}")
    (out / "plan").mkdir()
    shutil.copy2(saved_root / "plan/experiment-protocol.md", out / "plan/experiment-protocol.md")
    base.ROOT = out
    base.run(SimpleNamespace(system=args.system, engine="pysr", device=args.device, epochs=100))


def pendulum(args):
    base = load("lecture_pendulum_run", EXP / "measured_pendulum_closure/scripts/run_closure.py")
    saved_root = base.ROOT
    out = fresh_output(args.name or f"pendulum_{args.stage}")
    base.ROOT = out
    sys.argv = ["run_closure.py", args.stage, "--round", "1", "--device", args.device]
    if args.stage == "feedback":
        sys.argv += ["--spec", str(saved_root / "simplification.json")]
    base.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("checksum", "verify-results", "check", "all-controls", "summarize-new"):
        sub.add_parser(name)
    p = sub.add_parser("gate")
    p.add_argument("--system", choices=["pendulum", "quadrotor"], required=True)
    p.add_argument("--seed", type=int, choices=range(5), default=0)
    p.add_argument("--device", default="cuda:0")
    for name, default in (("regularizer-pendulum", 5800), ("regularizer-lorenz", 42044)):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=default)
        p.add_argument("--device", default="cuda:0")
    for name in ("discover", "feedback"):
        p = sub.add_parser(name)
        p.add_argument("--system", choices=["fhn", "lorenz"], required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--name")
    p = sub.add_parser("pendulum")
    p.add_argument("stage", choices=["discover", "feedback"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--name")
    args = parser.parse_args()
    if args.command == "checksum":
        checksum()
    elif args.command == "verify-results":
        verify_results()
    elif args.command == "check":
        check()
    elif args.command == "gate":
        execute(CONTROL / "scripts/run_paired_ablation.py", ["gate", "--system", args.system, "--seed", str(args.seed), "--device", args.device])
    elif args.command == "regularizer-pendulum":
        execute(CONTROL / "scripts/run_paired_ablation.py", ["separation", "--seed", str(args.seed), "--device", args.device])
    elif args.command == "regularizer-lorenz":
        execute(CONTROL / "scripts/run_lorenz_regularizer.py", ["--seed", str(args.seed), "--device", args.device])
    elif args.command == "all-controls":
        for name in ("run_gate_repeats.py", "run_pendulum_regularizer_repeats.py", "run_lorenz_regularizer_repeats.py", "summarize_ablation.py"):
            execute(CONTROL / "scripts" / name, [])
    elif args.command == "summarize-new":
        execute(CONTROL / "scripts/summarize_ablation.py", [])
    elif args.command == "discover":
        discovery(args)
    elif args.command == "feedback":
        feedback(args)
    elif args.command == "pendulum":
        pendulum(args)


if __name__ == "__main__":
    main()
