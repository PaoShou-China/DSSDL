"""Original-task reruns; original modules are imported read-only.

Query diagnostics never control the fixed-budget update path. A passive
counterfactual records what the original stopping rule would have returned.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
METHODS = ["DSSDL-symbol", "DSSDL", "CAVIA", "Reptile"]
SYSTEMS = ["fhn", "lorenz"]
SUPPORTS = [1, 5, 10, 20, 50, 100, 200, 500]
MILESTONES = [0, 5, 20, 100, 200, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def literal_config(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError((path, name))


def generated_records(text):
    pattern = r"^Saved (\d+) samples \(([\d.]+)s trajectory\) for slow_param (\[[^\]]+\]) to datasets[/\\](?:FHN|lorenz)[/\\](train|support|query)"
    return [
        dict(count=int(n), duration=float(d), parameters=ast.literal_eval(p), split=s)
        for n, d, p, s in re.findall(pattern, text, re.M)
    ]


def prepare():
    import numpy as np

    locked = ROOT / "original_protocol.json"
    if locked.exists():
        manifest = json.loads(locked.read_text(encoding="utf-8"))
        changed = [
            p for p, digest in manifest["protected_hashes"].items() if sha(Path(p)) != digest
        ]
        assert not changed, changed
        assert all(
            (ROOT / "data" / s / f"support_{k}.pkl").exists() for s in SYSTEMS for k in SUPPORTS
        )
        print("Reuse locked original protocol and generated data.", flush=True)
        return
    rows, protected, records = [], {}, {}
    for method in METHODS:
        method_dir = PROJECT / "compare2" / method
        for system in SYSTEMS:
            source = method_dir / f"final_data/seed/auto_runs_{system}_by_seed.csv"
            protected[str(source)] = sha(source)
            for row in read_csv(source):
                seed, support = int(row["seed"]), int(row["support"])
                log = method_dir / row["log_path"].replace("\\", "/")
                protected[str(log)] = sha(log)
                rec = generated_records(log.read_text(encoding="utf-8-sig", errors="replace"))
                assert len(rec) == (29 if system == "fhn" else 17), (log, len(rec))
                key = (system, support)
                if key in records:
                    assert records[key] == rec, (method, system, seed, support)
                records[key] = rec
                rows.append(
                    dict(
                        method=method,
                        system=system,
                        seed=seed,
                        support=support,
                        historical_rmse=float(row.get("post_loss", row.get("post_rmse"))),
                        historical_initial_rmse=float(
                            row.get("pre_loss", row.get("pre_rmse", "nan"))
                        ),
                        log=str(log),
                    )
                )
        for path in method_dir.glob("*.py"):
            protected[str(path)] = sha(path)
        for system in ("FHN", "lorenz"):
            for path in (method_dir / "datasets" / system).glob("*.py"):
                protected[str(path)] = sha(path)
        p = method_dir / "datasets/utils.py"
        protected[str(p)] = sha(p)
    assert len(rows) == 320
    assert sorted({r["seed"] for r in rows}) == list(range(1, 6))
    assert sorted({r["support"] for r in rows}) == SUPPORTS
    for system in SYSTEMS:
        assert all(
            [r for r in records[system, k] if r["split"] == "train"]
            == [r for r in records[system, 1] if r["split"] == "train"]
            for k in SUPPORTS
        )

    data_manifest = {}
    base = PROJECT / "compare2/DSSDL-symbol"
    utils = module(base / "datasets/utils.py", "original_dataset_utils")
    for system in SYSTEMS:
        native_name = "FHN" if system == "fhn" else "lorenz"
        generator_path = base / "datasets" / native_name / "create_dataset.py"
        generator = module(generator_path, "original_generator_" + system)

        # Require identical generation functions in all four original methods.
        def function_ast(path):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            return {
                n.name: ast.dump(n, include_attributes=False)
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in ("f", "create_dataset")
            }

        canonical = function_ast(generator_path)
        for method in METHODS:
            assert (
                function_ast(
                    PROJECT / "compare2" / method / "datasets" / native_name / "create_dataset.py"
                )
                == canonical
            )
        data_root = ROOT / "data" / system
        source_records = [r for r in records[system, 1] if r["split"] == "train"]
        target = [r for r in records[system, 1] if r["split"] == "query"][0]
        ranges = literal_config(generator_path, "ranges")
        common = dict(f_handle=generator.f, ranges=ranges, num_trajectories=1)
        data_root.mkdir(parents=True, exist_ok=True)
        with (
            (data_root / "generation.log").open("w", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
        ):
            for i, rec in enumerate(source_records):
                generator.create_dataset(
                    **common,
                    slow_param=rec["parameters"],
                    save_dir=str(data_root / "train"),
                    t_span=(0.0, rec["duration"]),
                    seed=i,
                )
            generator.create_dataset(
                **common,
                slow_param=target["parameters"],
                save_dir=str(data_root / "query"),
                t_span=(0.0, target["duration"]),
                seed=4096,
            )
            for support in SUPPORTS:
                rec = next(r for r in records[system, support] if r["split"] == "support")
                dest = data_root / f"support_{support}"
                kwargs = {"sample_points": support} if system == "fhn" else {}
                generator.create_dataset(
                    **common,
                    slow_param=rec["parameters"],
                    save_dir=str(dest),
                    t_span=(0.0, rec["duration"]),
                    seed=2025,
                    **kwargs,
                )

        def load_pair(folder):
            return tuple(
                [np.loadtxt(p, ndmin=2) for p in sorted((folder / kind).glob("*.txt"))]
                for kind in ("features", "labels")
            )

        train, query = load_pair(data_root / "train"), load_pair(data_root / "query")
        im, ist = utils.calculate_normalization_params(train[0])
        om, ost = utils.calculate_normalization_params(train[1])
        info = dict(
            input_dim=len(im),
            output_dim=len(om),
            dt=generator.dt,
            normalization={
                "input": {"mean": im.tolist(), "std": ist.tolist()},
                "output": {"mean": om.tolist(), "std": ost.tolist()},
            },
        )
        for support in SUPPORTS:
            raw = dict(
                train=train, support=load_pair(data_root / f"support_{support}"), query=query
            )
            assert raw["support"][0][0].shape[0] == support
            norm = {
                s: (
                    [utils.standardize(x, im, ist) for x in xs],
                    [utils.standardize(y, om, ost) for y in ys],
                )
                for s, (xs, ys) in raw.items()
            }
            ds = dict(
                raw=raw,
                normalized=norm,
                info=dict(
                    info,
                    file_counts={s: len(x[0]) for s, x in raw.items()},
                    sample_counts={s: sum(len(a) for a in x[0]) for s, x in raw.items()},
                ),
            )
            with (data_root / f"support_{support}.pkl").open("wb") as f:
                pickle.dump(ds, f)
        data_manifest[system] = dict(
            source_records=source_records,
            query=target,
            ranges=ranges,
            normalization=info["normalization"],
            source_seed_start=0,
            support_data_seed=2025,
            query_data_seed=4096,
            supports={
                str(k): next(r for r in records[system, k] if r["split"] == "support")
                for k in SUPPORTS
            },
        )
    for path, digest in protected.items():
        assert sha(Path(path)) == digest
    dump(
        ROOT / "original_protocol.json",
        dict(rows=rows, data=data_manifest, protected_hashes=protected),
    )
    print("Prepared original data and 320 matched run identities.", flush=True)


def dataset(system, support):
    with (ROOT / "data" / system / f"support_{support}.pkl").open("rb") as f:
        return pickle.load(f)


def setup(method, system, seed, device, ds, historical_source=False):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if device == "cpu":
        torch.set_flush_denormal(True)
    native = "FHN" if system == "fhn" else "lorenz"
    data_module = types.ModuleType("datasets")
    data_module.CURRENT_SYSTEM, data_module.dataset_dict = native, ds
    sys.modules["datasets"] = data_module
    method_dir = PROJECT / "compare2" / method
    if method.startswith("DSSDL"):
        cfg = literal_config(method_dir / "pretrain.py", "CONFIG_OVERRIDES")
        cfg["device"] = device
        if historical_source:
            assert method == "DSSDL-symbol" and system == "lorenz" and seed in range(1, 5)
            fingerprint = json.loads(
                (ROOT / "verification/source_regularization_fingerprint.json").read_text(
                    encoding="utf-8"
                )
            )
            assert fingerprint["all_close"] and fingerprint["target_query_used"] is False
            cfg["reg_rate"] = 10.0
        pre = types.ModuleType("pretrain")
        pre.CONFIG_OVERRIDES = cfg
        sys.modules["pretrain"] = pre
        mod = module(method_dir / "DSSDL_.py", "original_method")
        mod.set_global_seed(seed)
        mod.config.update({k: v for k, v in cfg.items() if k in mod.config})
        # Suppress only progress rendering; no mathematical training changes.
        from tqdm import tqdm

        def quiet_progress(*a, **kw):
            if kw.get("desc", "").startswith("Epoch "):
                epoch = int(kw["desc"].split()[1])
                if epoch % 10 == 0:
                    print("Source epoch", epoch, flush=True)
            return tqdm(*a, **dict(kw, disable=True))

        mod.tqdm = quiet_progress
    elif method == "CAVIA":
        mod = module(method_dir / "cavia_workflow.py", "original_method")
        cfg = mod._merge_config(
            literal_config(method_dir / "cavia_pretrain.py", "CONFIG_OVERRIDES")
        )
        cfg.update(seed=seed, device=device)
        mod._set_seed(seed)
    else:
        mod = module(method_dir / "Reptile.py", "original_method")
        cfg = mod.config.copy()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return mod, cfg


def make_model(method, mod, cfg, ds, device):
    if method.startswith("DSSDL"):
        return mod.MetaNet(
            ds["info"]["input_dim"],
            ds["info"]["output_dim"],
            ds["info"]["file_counts"]["train"],
            cfg["specific_param_dim"],
            cfg["hidden_widths"],
            device,
            ds,
            False,
        )
    if method == "CAVIA":
        return mod._build_model(ds["info"]["input_dim"], ds["info"]["output_dim"], cfg)
    return mod.MLP(
        ds["info"]["input_dim"],
        ds["info"]["output_dim"],
        cfg["hidden_size"],
        cfg["num_layers"],
        cfg["activation"],
    ).to(device)


def pretrain(method, system, seed, device, historical_source=False):
    import torch

    folder = (
        ROOT
        / ("checkpoints_historical_source" if historical_source else "checkpoints")
        / system
        / method
        / f"seed_{seed}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = folder / "pretrain.pt"
    if ckpt.exists() and (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        assert sha(ckpt) == metadata["checkpoint_sha256"], ckpt
        assert sha(ROOT / "data" / system / "support_100.pkl") == metadata["source_data_sha256"]
        print("Reuse", ckpt, flush=True)
        return
    os.chdir(folder)
    (folder / "model").mkdir(exist_ok=True)
    ds = dataset(system, 100)
    mod, cfg = setup(method, system, seed, device, ds, historical_source)
    started = time.perf_counter()
    with (
        (folder / "pretrain.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
    ):
        if method.startswith("DSSDL"):
            train, val = mod.load_dataset_with_one_hot(
                ds, "train", cfg["batch_size"], mod.config["train_ratio"], device, False
            )
            model = make_model(method, mod, cfg, ds, device)
            trainer = mod.Trainer(
                model, learning_rate=cfg["learning_rate"], optimizer_type=cfg["optimizer"]
            )
            history = trainer.pretrain(cfg["epochs"], train, val, cfg["reg_rate"])
            torch.save(model.state_dict(), ckpt)
            dump(folder / "source_history.json", history)
        elif method == "CAVIA":
            cfg["model_path"] = str(ckpt)
            mod.meta_train(ds, cfg)
        else:
            learner = mod.Reptile(ds)
            if str(learner.device) != device and not (
                str(learner.device) == "cuda" and device == "cuda:0"
            ):
                learner.device = torch.device(device)
                learner.network.to(device)
            learner.train(n_epochs=200)
            torch.save(learner.network.state_dict(), ckpt)
    dump(
        folder / "metadata.json",
        dict(
            method=method,
            system=system,
            seed=seed,
            config=cfg,
            device=device,
            torch_version=torch.__version__,
            source_training_seconds=time.perf_counter() - started,
            checkpoint_sha256=sha(ckpt),
            original_modules_read_only=True,
            historical_source_override=historical_source,
            source_data_sha256=sha(ROOT / "data" / system / "support_100.pkl"),
        ),
    )
    print("Pretrained", system, method, seed, round(time.perf_counter() - started, 2), flush=True)


class HistoricalRule:
    """Observational only: never selects the fixed-budget model."""

    def __init__(self, method):
        self.method, self.best, self.wait = method, float("inf"), 0
        self.result = None
        self.best_metric = None

    def observe(self, step, metrics):
        if self.result is not None:
            return
        if self.method.startswith("DSSDL"):
            q = metrics["query_rmse"]
            if step > 0:
                self.wait = self.wait + 1 if abs(self.best - q) < 1e-5 else 0
            self.best = min(self.best, q)
            stop = self.wait >= 5 or step == 199
            if stop:
                self.result = dict(
                    metrics,
                    rule="original_query_rmse_stop",
                    selected_step=step,
                    stopped_early=self.wait >= 5,
                    note="Original logger evaluates before the update; final cap log is step199.",
                )
        else:
            if step == 0:
                return
            q = metrics["query_rmse"] ** 2
            if self.best - q > 1e-10:
                self.best, self.wait, self.best_metric = q, 0, metrics.copy()
            else:
                self.wait += 1
            if self.wait >= 50 or step == 2000:
                chosen = self.best_metric if self.method == "CAVIA" else metrics
                self.result = dict(
                    chosen,
                    rule="original_query_mse_stop",
                    stopped_early=self.wait >= 50,
                    stop_step=step,
                    selected_step=chosen["step"],
                    restores_best=self.method == "CAVIA",
                )


def adapt(method, system, seed, support, device, budget, diagnostic=False, historical_source=False):
    import numpy as np
    import torch
    import torch.nn.functional as F

    branch = (
        (
            "verification/diagnostic_runs_cpu_ftz"
            if device == "cpu"
            else "verification/diagnostic_runs"
        )
        if diagnostic
        else ("runs_cpu" if device == "cpu" else "runs")
    )
    if historical_source:
        assert device == "cpu" and not diagnostic
        source_proof = json.loads(
            (ROOT / "verification/historical_source_full_history_match.json").read_text(
                encoding="utf-8"
            )
        )
        assert source_proof["passed"]
        assert any(
            r["seed"] == seed and r["support"] == support and r["selected_reg_rate"] == 10
            for r in source_proof["checks"]
        )
        branch = "runs_historical_source"
    folder = ROOT / branch / system / method / f"seed_{seed}" / f"support_{support}"
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    ds = dataset(system, support)
    mod, cfg = setup(method, system, seed, device, ds, historical_source)
    model = make_model(method, mod, cfg, ds, device)
    ckpt = (
        ROOT
        / ("checkpoints_historical_source" if historical_source else "checkpoints")
        / system
        / method
        / f"seed_{seed}"
        / "pretrain.pt"
    )
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    if method.startswith("DSSDL"):
        model.set_finetune()
        params = list(model.specific_param_generator.parameters())
        optimizer = torch.optim.LBFGS(params, lr=0.1)
        penalty = mod.config["weight_decay"]

        def predict(x):
            neural, symbolic = model.finetune_forward(x)
            return neural + symbolic
    elif method == "CAVIA":
        model.reset_context_params()
        params = [model.context_params]
        optimizer, penalty = None, 0.0
        predict = model
    else:
        params = list(model.parameters())
        # Original finetune.py overrides the pretraining inner_lr to 0.001.
        optimizer = torch.optim.Adam(params, lr=0.001)
        penalty, predict = 0.0, model

    def tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    sx, sy = [tensor(v[0]) for v in ds["normalized"]["support"]]
    qx, qy = [tensor(v[0]) for v in ds["normalized"]["query"]]
    ost = tensor(ds["info"]["normalization"]["output"]["std"])
    rule = HistoricalRule(method)
    history, milestones = [], []
    compute_seconds, gradient_evaluations, start_step = 0.0, 0, 0
    resume = folder / "resume.pt"
    if resume.exists():
        if (folder / "summary.json").exists():
            previous = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            assert previous["checkpoint_sha256"] == sha(ckpt), (
                "Changed checkpoint; do not mix adaptation paths."
            )
            assert previous["dataset_sha256"] == sha(
                ROOT / "data" / system / f"support_{support}.pkl"
            )
            assert previous["device"] == device
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if method == "CAVIA":
            model.context_params = saved["context"].detach().requires_grad_(True)
            params = [model.context_params]
        else:
            optimizer.load_state_dict(saved["optimizer"])
        start_step, history, milestones = saved["step"], saved["history"], saved["milestones"]
        rule.__dict__.update(saved["rule"])
        compute_seconds, gradient_evaluations = (
            saved["compute_seconds"],
            saved["gradient_evaluations"],
        )
        if start_step >= budget and (folder / "summary.json").exists():
            print("Reuse target", method, system, seed, support, flush=True)
            return

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def metrics(step):
        with torch.no_grad():
            residual = predict(sx) - sy
            smse = F.mse_loss(residual, torch.zeros_like(residual)).item()
            objective = smse + penalty * sum(p.square().sum().item() for p in params)
            qe = predict(qx) - qy
            return dict(
                step=step,
                support_mse=smse,
                support_objective=objective,
                query_rmse=qe.square().mean().sqrt().item(),
                raw_query_rmse=(qe * ost).square().mean().sqrt().item(),
                compute_seconds=compute_seconds,
                gradient_evaluations=gradient_evaluations,
            )

    def save(step):
        payload = dict(
            step=step,
            model=model.state_dict(),
            optimizer=optimizer.state_dict() if optimizer else None,
            context=model.context_params if method == "CAVIA" else None,
            history=history,
            milestones=milestones,
            rule=rule.__dict__,
            compute_seconds=compute_seconds,
            gradient_evaluations=gradient_evaluations,
        )
        torch.save(payload, resume)
        with (folder / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    initial = metrics(start_step)
    if start_step == 0:
        history.append(initial)
        milestones.append(initial)
        rule.observe(0, initial)
    started = time.perf_counter()
    for step in range(start_step + 1, budget + 1):
        sync()
        t0 = time.perf_counter()
        if method.startswith("DSSDL"):

            def closure():
                nonlocal gradient_evaluations
                optimizer.zero_grad()
                loss = F.mse_loss(predict(sx), sy) + penalty * sum(p.square().sum() for p in params)
                loss.backward()
                gradient_evaluations += 1
                return loss.detach()

            optimizer.step(closure)
        elif method == "CAVIA":
            loss = F.mse_loss(predict(sx), sy)
            grad = torch.autograd.grad(loss, model.context_params)[0]
            model.context_params = (
                (model.context_params - 1e-6 * grad).detach().requires_grad_(True)
            )
            params = [model.context_params]
            gradient_evaluations += 1
        else:
            optimizer.zero_grad()
            loss = F.mse_loss(predict(sx), sy)
            loss.backward()
            optimizer.step()
            gradient_evaluations += 1
        sync()
        compute_seconds += time.perf_counter() - t0
        inspect = rule.result is None or step % 50 == 0 or step in MILESTONES or step == budget
        if inspect:
            entry = metrics(step)
            if not np.isfinite(entry["support_objective"]):
                dump(
                    folder / "failure.json",
                    dict(
                        step=step,
                        metrics=entry,
                        reason="nonfinite_support_objective",
                        budget=budget,
                    ),
                )
                raise FloatingPointError((method, system, seed, support, step))
            if (
                not np.isfinite(entry["query_rmse"])
                and not (folder / "nonfinite_query.json").exists()
            ):
                dump(
                    folder / "nonfinite_query.json",
                    dict(
                        step=step,
                        metrics=entry,
                        note="Recorded only; query diagnostics do not stop support optimization.",
                    ),
                )
            rule.observe(step, entry)
            if step % 50 == 0 or step in MILESTONES or step == budget:
                history.append(entry)
            if step in MILESTONES or step == budget:
                milestones.append(entry)
        if step % 1000 == 0 or step == budget:
            save(step)
        if step % 5000 == 0:
            print(
                "Adapt",
                system,
                method,
                seed,
                support,
                step,
                "support",
                entry["support_objective"],
                "query",
                entry["query_rmse"],
                flush=True,
            )
    tail = [r["support_objective"] for r in history if r["step"] >= 0.8 * budget]
    span = max(tail) - min(tail)
    relative = span / max(abs(float(np.mean(tail))), 1e-12)
    stable = relative <= 0.01 or span <= 1e-8
    manifest = json.loads((ROOT / "original_protocol.json").read_text(encoding="utf-8"))
    original = next(
        r
        for r in manifest["rows"]
        if (r["method"], r["system"], r["seed"], r["support"]) == (method, system, seed, support)
    )
    dump(
        folder / "summary.json",
        dict(
            method=method,
            system=system,
            seed=seed,
            support=support,
            budget=budget,
            fixed_budget_final=history[-1],
            milestones=milestones,
            original_rule_counterfactual=rule.result,
            historical=original,
            support_stability=dict(
                stable=stable, relative_range=relative, absolute_range=span, tail_fraction=0.2
            ),
            checkpoint_sha256=sha(ckpt),
            dataset_sha256=sha(ROOT / "data" / system / f"support_{support}.pkl"),
            device=device,
            historical_source_override=historical_source,
            source_reg_rate=cfg.get("reg_rate"),
            flush_denormal=device == "cpu",
            torch_version=torch.__version__,
            query_selected_final=False,
            early_stopping=False,
            extension_wall_seconds=time.perf_counter() - started,
        ),
    )
    print(
        "Complete",
        system,
        method,
        seed,
        support,
        "stable",
        stable,
        "seconds",
        round(compute_seconds, 1),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "pretrain", "adapt", "sweep"])
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--system", choices=SYSTEMS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--support", type=int, default=100)
    p.add_argument("--supports", nargs="+", type=int, choices=SUPPORTS, default=SUPPORTS)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--historical-source", action="store_true")
    args = p.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "pretrain":
        pretrain(args.method, args.system, args.seed, args.device, args.historical_source)
    elif args.action == "sweep":
        pretrain(args.method, args.system, args.seed, args.device, args.historical_source)
        failures = []
        for support in args.supports:
            try:
                adapt(
                    args.method,
                    args.system,
                    args.seed,
                    support,
                    args.device,
                    args.budget,
                    historical_source=args.historical_source,
                )
            except FloatingPointError as error:
                failures.append(str(error))
                print("Failed support objective", error, flush=True)
        if failures:
            raise SystemExit(1)
    else:
        adapt(
            args.method,
            args.system,
            args.seed,
            args.support,
            args.device,
            args.budget,
            args.diagnostic,
            args.historical_source,
        )


if __name__ == "__main__":
    main()
