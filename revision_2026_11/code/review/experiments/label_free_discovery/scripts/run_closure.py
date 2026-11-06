"""Two-round PySR feedback experiment with no physical parameter labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
PROJECT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import gplearn_closed_loop_symbolic_discovery as shared
import closed_loop_symbolic_discovery_runner as runner

NAMES = {"fhn": ["v", "w"], "lorenz": ["x", "y", "z"]}
THRESHOLDS = [0., .0001, .0003, .001, .003, .01, .03]
TRIPWIRE_CALLS = []


def forbidden(*args, **kwargs):
    TRIPWIRE_CALLS.append("physical_parameter_interface")
    raise RuntimeError("Physical parameters and decoder interfaces are forbidden")


shared.load_params = forbidden
shared.fit_decoder_loo = forbidden
shared.multisystem.load_transformed_params = forbidden
shared.multisystem.fit_decoder_loo = forbidden
shared.multisystem.parse_floats_from_stem = forbidden
runner.parse_params_from_stem = forbidden


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LatentPrior:
    def __init__(self, system, equations, dataset):
        self.names = NAMES[system] + ["q1", "q2", "q3"]
        symbols = {name: sp.Symbol(name) for name in self.names}
        self.expressions = [sp.sympify(eq, locals=symbols) for eq in equations]
        allowed = set(symbols.values())
        assert all(expr.free_symbols <= allowed for expr in self.expressions)
        self.dim = len(NAMES[system])
        n = dataset["info"]["normalization"]
        self.norm = {f"{part}_{stat}": np.asarray(n[part][stat])
                     for part in ("input", "output") for stat in ("mean", "std")}

    def raw_numpy(self, features):
        return np.column_stack([shared.evaluate_sympy_numpy(e, self.names, features)
                                for e in self.expressions])

    def __call__(self, inp, dataset_dict=None, use_raw_data=False):
        n = {k: torch.as_tensor(v, device=inp.device, dtype=inp.dtype)
             for k, v in self.norm.items()}
        states = inp[:, :self.dim]
        if not use_raw_data:
            states = states * n["input_std"] + n["input_mean"]
        raw = torch.cat([states, inp[:, self.dim:]], dim=1)
        env = {name: raw[:, i] for i, name in enumerate(self.names)}
        result = torch.stack([shared.evaluate_sympy_torch(e, env, states[:, 0])
                              for e in self.expressions], dim=1)
        return result if use_raw_data else (result - n["output_mean"]) / n["output_std"]


def model_for(module, dataset, cfg, checkpoint, prior):
    model = runner.build_model(module, dataset, cfg, prior)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg["device"], weights_only=True), strict=True)
    return model


def source_tensors(dataset, device):
    xs, ys = dataset["normalized"]["train"]
    X, Y, train, val = [], [], [], []
    offset = 0
    for task, (x, y) in enumerate(zip(xs, ys)):
        onehot = np.zeros((len(x), len(xs)))
        onehot[:, task] = 1
        X.append(np.c_[x, onehot])
        Y.append(y)
        cut = int(.8 * len(x))
        train.extend(range(offset, offset + cut))
        val.extend(range(offset + cut, offset + len(x)))
        offset += len(x)
    return (torch.as_tensor(np.concatenate(X), dtype=torch.float32, device=device),
            torch.as_tensor(np.concatenate(Y), dtype=torch.float32, device=device),
            np.asarray(train), np.asarray(val))


def source_scores(model, dataset, tensors):
    X, Y, train, val = tensors
    n = dataset["info"]["normalization"]["output"]
    std = np.asarray(n["std"])
    scores = {}
    model.eval()
    with torch.no_grad():
        for label, ids in (("train", train), ("val", val)):
            errors, prior_errors = [], []
            for chunk in np.array_split(ids, max(1, int(np.ceil(len(ids) / 4096)))):
                net, prior = model(X[chunk])
                errors.append((net + prior - Y[chunk]).cpu().numpy())
                prior_errors.append((prior - Y[chunk]).cpu().numpy())
            error, pe = np.concatenate(errors), np.concatenate(prior_errors)
            scores.update({f"source_{label}_rmse_norm": np.sqrt(np.mean(error**2)),
                           f"source_{label}_rmse_raw": np.sqrt(np.mean((error*std)**2)),
                           f"source_{label}_prior_rmse_norm": np.sqrt(np.mean(pe**2))})
    return scores


def train_round(model, dataset, tensors, out, epochs, symbolic, round_id):
    seed_all(42000 + round_id)
    X, Y, train, _ = tensors
    for p in model.parameters():
        p.requires_grad_(True)
    mapper = torch.optim.Adam(model.one_hot_to_specific_param.parameters(), lr=.001)
    all_opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.RandomState(42000 + round_id)
    logs, best, best_state, best_epoch = [], float("inf"), None, -1
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        ids = rng.permutation(train)
        losses = []
        for begin in range(0, len(ids), 512):
            batch = ids[begin:begin+512]
            for opt in (mapper, all_opt):
                opt.zero_grad()
                net, prior = model(X[batch])
                loss = torch.mean((net + prior - Y[batch])**2)
                if symbolic:
                    loss = loss + torch.mean((prior - Y[batch])**2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source training loss")
                loss.backward()
                opt.step()
            losses.append(float(loss.detach()))
        metric = source_scores(model, dataset, tensors)
        value = metric["source_val_rmse_norm"]
        if value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logs.append(dict(epoch=epoch, train_objective=np.mean(losses), **metric,
                         elapsed_s=time.perf_counter()-start))
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"TRAIN {out.parent.name}/{out.name} epoch={epoch} val={value:.6g}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    checkpoint = out / "model.pt"
    torch.save(best_state, checkpoint)
    pd.DataFrame(logs).to_csv(out / "training.csv", index=False)
    dump(out / "training_selection.json", dict(best_epoch=best_epoch, best_source_val=best,
         epochs=epochs, selection="source validation full-model RMSE only",
         symbolic_reg_rate=int(symbolic), wall_seconds=time.perf_counter()-start))
    return checkpoint


def collect_probe(model, dataset, system):
    rng = np.random.RandomState(420)
    raw_x = dataset["raw"]["train"][0]
    pool = np.concatenate([x[:int(.8*len(x))] for x in raw_x])
    states = pool[rng.choice(len(pool), 384, replace=False)]
    task_ids = np.arange(len(raw_x))
    order = rng.permutation(task_ids)
    train_tasks = np.sort(order[:max(1, int(.8*len(order)))])
    val_tasks = np.sort(order[max(1, int(.8*len(order))):])
    norm = dataset["info"]["normalization"]
    x = torch.as_tensor((states-norm["input"]["mean"])/norm["input"]["std"],
                        dtype=torch.float32, device=model.device)
    z = model.one_hot_to_specific_param.weight.detach().cpu().numpy().T
    features, predictions = [], []
    model.eval()
    with torch.no_grad():
        for task in task_ids:
            onehot = torch.zeros((len(x), len(task_ids)), device=x.device)
            onehot[:, task] = 1
            net, prior = model(torch.cat([x, onehot], dim=1))
            pred = (net+prior).cpu().numpy()*norm["output"]["std"] + norm["output"]["mean"]
            predictions.append(pred)
            features.append(np.c_[states, np.repeat(z[task:task+1], len(x), axis=0)])
    stack = np.stack(predictions)
    common = np.mean(stack[train_tasks], axis=0)
    indices = rng.permutation(len(states))
    return dict(features=np.concatenate(features), target=np.concatenate(predictions),
                states=states, common=common, variation=(stack-common).reshape(-1,len(NAMES[system])),
                task_ids=np.repeat(task_ids, len(states)), train_tasks=train_tasks, val_tasks=val_tasks,
                common_train=indices[:int(.8*len(indices))], common_val=indices[int(.8*len(indices)):], z=z)


def sparse_refit(expressions, names, X, y, tr, va, out):
    # Only terms actually emitted by PySR enter this data-driven dictionary.
    factor_map, groups = {}, []
    for expression in expressions:
        group = []
        for factor in shared.additive_factors(sp.expand(expression)):
            key = sp.srepr(factor)
            if key not in factor_map and len(factor_map) < 128:
                factor_map[key] = factor
            if key in factor_map and key not in group:
                group.append(key)
        if group and group not in groups:
            groups.append(group)
    columns = {}
    for key, expr in factor_map.items():
        col = shared.evaluate_sympy_numpy(expr, names, X)
        if np.isfinite(col).all() and np.sqrt(np.mean(col[tr]**2)) > 1e-12:
            columns[key] = col
    groups = [[k for k in g if k in columns] for g in groups]
    groups.append(list(columns))
    target_scale = max(float(np.std(y[tr])), 1e-9)
    candidates = []
    for group_id, group in enumerate(groups):
        if not group:
            continue
        theta = np.column_stack([columns[k] for k in group])
        scales = np.sqrt(np.mean(theta[tr]**2, axis=0))
        A = theta / scales
        for threshold in THRESHOLDS:
            active = np.ones(len(group), dtype=bool)
            beta = np.zeros(len(group))
            for _ in range(15):
                beta[:] = 0
                if not active.any():
                    break
                design = A[tr][:, active]
                beta[active] = np.linalg.lstsq(design, y[tr]/target_scale, rcond=1e-8)[0]
                keep = np.abs(beta) >= threshold
                if np.array_equal(keep, active):
                    break
                active = keep
            beta[~active] = 0
            error = float(np.sqrt(np.mean((A[va]@beta-y[va]/target_scale)**2)))
            candidates.append(dict(group=group_id, threshold=threshold, val_nrmse=error,
                                   nnz=int(np.count_nonzero(beta)), keys=group,
                                   coefficients=beta*target_scale/scales))
    if not candidates:
        raise RuntimeError("No finite PySR candidates")
    minimum = min(row["val_nrmse"] for row in candidates)
    eligible = [row for row in candidates if row["val_nrmse"] <= minimum*1.05+1e-8]
    selected = min(eligible, key=lambda row: (row["nnz"], row["val_nrmse"], row["group"]))
    expr = sum((sp.Float(float(c), 14)*factor_map[k]
                for k,c in zip(selected["keys"], selected["coefficients"]) if c != 0), sp.Integer(0))
    expr = sp.factor_terms(sp.expand(expr))
    pd.DataFrame([{k:v for k,v in row.items() if k not in ("keys","coefficients")}
                  | {"selected": row is selected} for row in candidates]).to_csv(out/"candidate_scores.csv", index=False)
    dump(out/"selection.json", dict(expression=str(expr), selected=selected,
         minimum_val_nrmse=minimum, emitted_terms=[str(x) for x in factor_map.values()],
         criterion="sparsest within 5% of minimum held-out NRMSE; no physical truth"))
    return expr


def search_component(X, y, names, tr, va, out, seed, timeout):
    from pysr import PySRRegressor
    out.mkdir(parents=True, exist_ok=True)
    center, scale = np.mean(X[tr],axis=0), np.std(X[tr],axis=0)
    scale[scale < 1e-9] = 1
    yc, ys = float(np.mean(y[tr])), max(float(np.std(y[tr])), 1e-9)
    U, target = (X-center)/scale, (y-yc)/ys
    normalized_names = [f"u{i}" for i in range(X.shape[1])]
    print(f"SEARCH {out.parent.parent.name}/{out.parent.name}/{out.name} n={len(tr)}", flush=True)
    start = time.perf_counter()
    sr = PySRRegressor(niterations=100, populations=8, population_size=32,
        binary_operators=["+","-","*"], unary_operators=[], maxsize=25,
        parsimony=.001, random_state=seed, deterministic=True, parallelism="serial",
        progress=False, verbosity=0, timeout_in_seconds=timeout,
        output_directory=str(out/"pysr"), delete_tempfiles=False)
    sr.fit(U[tr], target[tr], variable_names=normalized_names)
    subst = {sp.Symbol(name):(sp.Symbol(raw)-float(mean))/float(std)
             for name,raw,mean,std in zip(normalized_names,names,center,scale)}
    equations = [sp.expand(sp.Float(yc)+sp.Float(ys)*sr.sympy(index=i).subs(subst))
                 for i in range(len(sr.equations_))]
    np.savez_compressed(out/"search_data.npz", features=X, target=y, train_indices=tr, val_indices=va)
    dump(out/"search.json", dict(algorithm="PySR", features=names, physical_labels=False,
         seed=seed, wall_seconds=time.perf_counter()-start, centers=center, scales=scale,
         target_center=yc, target_scale=ys, raw_equations=sr.equations_["equation"].tolist(),
         inverse_standardized_equations=[str(e) for e in equations]))
    expression = sparse_refit(equations, names, X, y, tr, va, out)
    print(f"SELECT {out.name}: {expression}", flush=True)
    return expression


def discover(model, dataset, system, out, round_id, timeout):
    probe = collect_probe(model, dataset, system)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"probe.npz", **probe)
    names = NAMES[system]+["q1","q2","q3"]
    tr = np.flatnonzero(np.isin(probe["task_ids"], probe["train_tasks"]))
    va = np.flatnonzero(np.isin(probe["task_ids"], probe["val_tasks"]))
    rng = np.random.RandomState(421)
    if len(tr)>5000:
        tr = rng.choice(tr, 5000, replace=False)
    equations = []
    for j in range(len(NAMES[system])):
        common = search_component(probe["states"],probe["common"][:,j],NAMES[system],
            probe["common_train"],probe["common_val"],out/f"output{j}"/"common",
            42000+round_id*100+j*2,timeout)
        varying = search_component(probe["features"],probe["variation"][:,j],names,tr,va,
            out/f"output{j}"/"varying",42001+round_id*100+j*2,timeout)
        equations.append(str(sp.factor_terms(sp.expand(common+varying))))
    prior = LatentPrior(system,equations,dataset)
    prediction = prior.raw_numpy(probe["features"])
    std = np.asarray(dataset["info"]["normalization"]["output"]["std"])
    dump(out/"equations.json", dict(system=system,round=round_id,variables=names,equations=equations,
         teacher_rmse_raw=np.sqrt(np.mean((prediction-probe["target"])**2)),
         teacher_rmse_norm=np.sqrt(np.mean(((prediction-probe["target"])/std)**2)),
         term_counts=[len(sp.Add.make_args(sp.expand(e))) for e in prior.expressions],
         physical_parameter_labels_used=False, decoder_used=False))
    features = probe["features"][:32]
    inputs = torch.tensor(features,dtype=torch.float64,requires_grad=True)
    actual = prior(inputs,use_raw_data=True)
    expected = prior.raw_numpy(features)
    np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=1e-8,atol=1e-8)
    actual.sum().backward()
    if inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise AssertionError("Invalid symbolic gradients")
    dump(out/"prior_validation.json", dict(numpy_torch_match=True,gradient_finite=True,
        max_absolute_difference=np.max(np.abs(actual.detach().numpy()-expected))))
    return prior


# Use 02_code/run.py for the packaged execution entry.
