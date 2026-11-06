"""Portable entry points for saved-result checks and selected experiment reruns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
EXP = CODE / "review/experiments"


def load(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fresh(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("--name must be a single directory name")
    path = ROOT / "outputs" / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def wasserstein(args):
    folder = EXP / "wasserstein"
    mod = load("published_wasserstein", folder / "scripts/original_wasserstein.py")
    out = fresh(args.name or "wasserstein")
    shutil.copytree(folder / "data", out / "data")
    (out / "tables").mkdir()
    mod.ROOT = out
    frame, _, _ = mod.load_and_compute(read(out / "data/source_manifest.json"))
    frame.to_csv(out / "tables/recomputed.csv", index=False)
    mod.configure_style()
    for system in mod.SYSTEMS:
        mod.w_panel(system, frame)
    print(frame[["system", "method", "sliced_wasserstein_1", "status"]].to_string(index=False))
    print("Output:", out)


def budget(args):
    folder = EXP / "matched_budget"
    mod = load("published_budget", folder / "scripts/native_run.py")
    selected = next(
        r
        for r in read(folder / "selection_manifest.json")
        if (r["system"], r["method"], r["seed"], r["support"])
        == (args.system, args.method, args.seed, args.support)
    )
    historical = selected["restored_source_config"]
    out = fresh(args.name or f"budget_{args.system}_{args.method}_{args.seed}_{args.support}")
    checkpoint_dir = "checkpoints_historical_source" if historical else "checkpoints"
    relative = Path(checkpoint_dir) / args.system / args.method / f"seed_{args.seed}"
    shutil.copytree(folder / relative, out / relative)
    shutil.copytree(folder / "verification", out / "verification")
    shutil.copy2(folder / "original_protocol.json", out / "original_protocol.json")
    (out / "data" / args.system).mkdir(parents=True)
    shutil.copy2(
        folder / "data" / args.system / f"support_{args.support}.pkl",
        out / "data" / args.system / f"support_{args.support}.pkl",
    )
    mod.ROOT = out
    mod.PROJECT = folder / "native"
    mod.adapt(
        args.method,
        args.system,
        args.seed,
        args.support,
        "cpu",
        args.budget,
        historical_source=historical,
    )
    print("Output:", out)


def online_model(system, method):
    import numpy as np
    import sympy as sp
    import torch

    fb = load("published_feedback", EXP / "symbolic_feedback/scripts/run_feedback.py")
    _, dataset, native, _, _ = fb.old.shared.multisystem.import_dssdl_for_system(
        "FHN" if system == "fhn" else "lorenz"
    )
    manifest = read(EXP / "label_free_discovery/runs" / system / "run_manifest.json")
    checkpoint = CODE / manifest["packaged_source_checkpoint"]
    model = fb.old.model_for(
        native,
        dataset,
        dict(device="cpu", hidden_widths=[64, 64], specific_param_dim=3),
        checkpoint,
        fb.old.runner.ZeroPrior(len(fb.old.NAMES[system])),
    )
    if method == "pysr44":
        folder = EXP / "symbolic_feedback/runs/pysr" / system
        spec = read(folder / "first_round_simplification.json")
        model.f_hat_func = fb.CompactPrior(
            system,
            [sp.sympify(e) for e in spec["new_templates"]],
            spec["shared_coefficient_initial_values"],
            spec["coefficient_center"],
            spec["coefficient_scale"],
            dataset,
        )
        model.load_state_dict(
            torch.load(folder / "compact_model.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.specific_param_generator.weight.requires_grad_(True)
    with torch.no_grad():
        model.specific_param_generator.weight.copy_(
            model.one_hot_to_specific_param.weight.mean(1, keepdim=True)
        )
    normalization = dataset["info"]["normalization"]
    norm = {
        f"{part}_{stat}": np.asarray(normalization[part][stat], dtype=float)
        for part in ("input", "output")
        for stat in ("mean", "std")
    }
    return model, norm


def online(args):
    import numpy as np
    import pandas as pd
    import torch

    folder = EXP / "online"
    base = load("published_online", folder / "scripts/native_protocol.py")
    base.ROOT, base.CFG = folder, read(folder / "protocol.json")
    profile = "jump_return" if args.system == "fhn" else "slow_sine"
    out = fresh(args.name or f"online_{args.system}_{args.method}_{args.seed}")
    with np.load(folder / "data" / args.system / profile / f"seed_{args.seed:03d}.npz") as data:
        states = data["states"].copy()
    # This is the recorded 0.5-second window schedule, including its reserved 1-second horizon.
    horizon = 100
    origins = np.arange(100, len(states) - horizon, 50)
    if args.smoke:
        origins = origins[:2]
    model, norm = online_model(args.system, args.method)
    immutable = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k != "specific_param_generator.weight"
    }
    if args.method == "sindy_source":
        model = None
        ss = load("published_sindy", folder / "scripts/source_sindy.py")
        artifact = ss.load_artifact(folder / "source" / args.system)
        previous = artifact["initial"].copy()
    fitted = []
    for origin in origins:
        x, y = base.support_pairs(states, origin, 50)
        if model is None:
            previous = ss.update(x, y, previous, artifact)
            coefficients = ss.physical_coefficients(previous, artifact)
        else:
            coefficients, _, _, _, _ = base.adapt(model, norm, x, y, args.method)
        fitted.append(coefficients)
    fitted = np.asarray(fitted)
    truth = np.stack([states[o + 1 : o + horizon + 1] for o in origins])
    if model is None:
        prediction = base.sindy_forecast(
            states[origins],
            fitted,
            artifact["powers"],
            horizon,
            base.CFG["state_limit"][args.system],
        )
    else:
        prediction = base.neural_forecast(
            model, norm, states[origins], fitted, horizon, base.CFG["state_limit"][args.system]
        )
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in immutable.items())
    np.savez_compressed(
        out / "forecasts.npz", origin_index=origins, truth=truth, fitted=fitted, online=prediction
    )
    rows = []
    for index, origin in enumerate(origins):
        for h in base.CFG["horizon_s"]:
            steps = round(h / 0.01)
            rows.append(
                dict(
                    origin_s=origin * 0.01,
                    horizon_s=h,
                    **base.metrics(
                        prediction[index, :steps], truth[index, :steps], norm["input_std"]
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(out / "window_metrics.csv", index=False)
    print("Output:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("wasserstein")
    p.add_argument("--name")
    p = commands.add_parser("budget")
    p.add_argument("--system", choices=["fhn", "lorenz"], required=True)
    p.add_argument("--method", choices=["DSSDL-symbol", "DSSDL", "CAVIA", "Reptile"], required=True)
    p.add_argument("--seed", type=int, choices=range(1, 6), default=1)
    p.add_argument("--support", type=int, choices=[1, 5, 10, 20, 50, 100, 200, 500], default=100)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--name")
    p = commands.add_parser("online")
    p.add_argument("--system", choices=["fhn", "lorenz"], required=True)
    p.add_argument("--method", choices=["pysr44", "paper_no_symbol", "sindy_source"], required=True)
    p.add_argument("--seed", type=int, choices=range(3), default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--name")
    args = parser.parse_args()
    {"wasserstein": wasserstein, "budget": budget, "online": online}[args.command](args)


if __name__ == "__main__":
    main()
