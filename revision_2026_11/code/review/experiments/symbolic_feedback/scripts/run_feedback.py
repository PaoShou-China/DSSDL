"""Human-assisted first-round simplification and source feedback, no true parameters."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
P42 = EXP / "label_free_discovery"
spec = importlib.util.spec_from_file_location("feedback44_old42", P42 / "scripts/run_closure.py")
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_q(expr, q):
    result = sp.lambdify(sp.symbols("q1 q2 q3"), expr, "numpy")(*q.T)
    return np.broadcast_to(np.asarray(result, dtype=float), (len(q),)).copy()


def simplify_first(system, engine, equations, old_q, train_tasks):
    states = sp.symbols(" ".join(old.NAMES[system]))
    qs = sp.symbols("q1 q2 q3")
    polynomials = [sp.Poly(sp.sympify(expr), *states) for expr in equations]

    def coef(output, monomial):
        return polynomials[output].coeff_monomial(sp.sympify(monomial))

    if system == "fhn":
        functions = [coef(0, "1"), coef(1, "1"), coef(1, "w")]
        globals_ = [coef(0, "v"), coef(0, "v**3"), coef(0, "w"), coef(1, "v")]
        templates = ["c1*v + c2*v**3 + c3*w + q1", "c4*v + q2 + q3*w"]
        retained = [{(1, 0), (3, 0), (0, 1), (0, 0)}, {(1, 0), (0, 0), (0, 1)}]
    else:
        functions = [(coef(0, "y") - coef(0, "x")) / 2, coef(1, "x")]
        if engine == "gplearn":
            functions.append(coef(2, "1"))
            globals_ = [coef(1, "x*z"), coef(2, "x*y"), coef(2, "z")]
            templates = ["q1*(y-x)", "q2*x + c1*x*z", "c2*x*y + c3*z + q3"]
            retained = [{(1, 0, 0), (0, 1, 0)}, {(1, 0, 0), (1, 0, 1)},
                        {(1, 1, 0), (0, 0, 1), (0, 0, 0)}]
            assert coef(1, "y") == 0
            assert not coef(2, "z").free_symbols
        else:
            functions.append(coef(2, "z"))
            globals_ = [coef(1, "x*z"), coef(1, "y"), coef(2, "x*y")]
            templates = ["q1*(y-x)", "q2*x + c1*x*z + c2*y", "c3*x*y + q3*z"]
            retained = [{(1, 0, 0), (0, 1, 0)}, {(1, 0, 0), (1, 0, 1), (0, 1, 0)},
                        {(1, 1, 0), (0, 0, 1)}]
            assert coef(1, "y") != 0
            assert coef(2, "z").free_symbols
    for polynomial, support in zip(polynomials, retained):
        assert support <= set(polynomial.monoms()), "A missing state term would be inserted"
    values = np.column_stack([evaluate_q(f, old_q) for f in functions])
    constants = np.array([evaluate_q(f, old_q[train_tasks]).mean() for f in globals_])
    center = values[train_tasks].mean(0)
    scales = values[train_tasks].std(0)
    assert np.all(scales > 1e-8), "A new varying coefficient was not actually discovered"
    expressions = [sp.sympify(t) for t in templates]
    c_symbols = sp.symbols(" ".join(f"c{i+1}" for i in range(len(constants))))
    substitutions = dict(zip(c_symbols, constants))
    raw_compact = [sp.expand(expr.subs(substitutions)) for expr in expressions]
    retained_exact, deleted = [], []
    for polynomial, support in zip(polynomials, retained):
        retained_expr = sum((c * sp.prod(x**p for x, p in zip(states, power))
                             for power, c in polynomial.terms() if power in support), sp.Integer(0))
        retained_exact.append(str(retained_expr))
        deleted.append(str(sp.expand(polynomial.as_expr() - retained_expr)))
    with_old_q = [str(expr.subs(dict(zip(qs, functions)), simultaneous=True)) for expr in raw_compact]
    audit = dict(original_equations=equations, new_templates=templates,
        coefficient_functions=[str(f) for f in functions],
        shared_coefficient_original_functions=[str(f) for f in globals_],
        shared_coefficient_initial_values=constants, coefficient_center=center, coefficient_scale=scales,
        compact_equations_in_Q=[str(expr) for expr in raw_compact],
        retained_state_support=[sorted(s) for s in retained], retained_before_tying=retained_exact,
        deleted_state_factors=deleted, compact_in_old_q=with_old_q,
        human_assisted=True, manual_missing_term_insertion=False, coefficient_snapping=False,
        caveat="Deleting factors, tying dx coefficients and sharing selected coefficients are approximations; independent Q relax old coefficient-function constraints.")
    return expressions, constants, center, scales, values, audit


class CompactPrior(torch.nn.Module):
    def __init__(self, system, templates, constants, center, scales, dataset):
        super().__init__()
        self.system = system
        self.names = old.NAMES[system]
        self.templates = templates
        self.global_coefficients = torch.nn.Parameter(torch.as_tensor(constants, dtype=torch.float32))
        self.register_buffer("coefficient_center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("coefficient_scale", torch.tensor(scales, dtype=torch.float32))
        norm = dataset["info"]["normalization"]
        for part in ("input", "output"):
            for stat in ("mean", "std"):
                self.register_buffer(f"{part}_{stat}", torch.tensor(norm[part][stat], dtype=torch.float32))

    def forward(self, inp, dataset_dict=None, use_raw_data=False):
        dim = len(self.names)
        states = inp[:, :dim]
        if not use_raw_data:
            states = states * self.input_std + self.input_mean
        Q = inp[:, dim:] * self.coefficient_scale + self.coefficient_center
        env = {name: states[:, j] for j, name in enumerate(self.names)}
        env.update({f"q{j+1}": Q[:, j] for j in range(3)})
        env.update({f"c{j+1}": value for j, value in enumerate(self.global_coefficients)})
        result = torch.stack([old.shared.evaluate_sympy_torch(e, env, states[:, 0]) for e in self.templates], dim=1)
        return result if use_raw_data else (result - self.output_mean) / self.output_std

    def expressions_Q(self):
        substitutions = {sp.Symbol(f"c{j+1}"): float(c) for j, c in enumerate(self.global_coefficients.detach().cpu())}
        return [sp.expand(e.subs(substitutions)) for e in self.templates]

    def expressions_z(self):
        subs = {sp.Symbol(f"q{j+1}"): float(center) + float(scale) * sp.Symbol(f"q{j+1}")
                for j, (center, scale) in enumerate(zip(self.coefficient_center.cpu(), self.coefficient_scale.cpu()))}
        return [e.subs(subs, simultaneous=True) for e in self.expressions_Q()]

    def numpy(self, features):
        names = self.names + ["q1", "q2", "q3"]
        return np.column_stack([old.shared.evaluate_sympy_numpy(e, names, features) for e in self.expressions_z()])


def source_arrays(dataset, q):
    X, Y, tasks, first = [], [], [], []
    for task, (x, y) in enumerate(zip(*dataset["raw"]["train"])):
        X.append(np.c_[x, np.repeat(q[task:task+1], len(x), axis=0)])
        Y.append(y)
        tasks.extend([task] * len(x))
        first.extend(np.arange(len(x)) < int(.8 * len(x)))
    return np.concatenate(X), np.concatenate(Y), np.array(tasks), np.array(first)


def export_stage(model, dataset, tensors, out, label):
    prior = model.f_hat_func
    z = model.one_hot_to_specific_param.weight.detach().cpu().numpy().T
    X, Y, tasks, first = source_arrays(dataset, z)
    pred = prior.numpy(X)
    std = np.asarray(dataset["info"]["normalization"]["output"]["std"])
    metrics = dict(stage=label, **old.source_scores(model, dataset, tensors))
    for name, mask in (("train", first), ("time_val", ~first)):
        metrics[f"symbol_{name}_per_output_nrmse"] = np.sqrt(np.mean(((pred[mask] - Y[mask]) / std) ** 2, axis=0))
    out.mkdir(parents=True, exist_ok=True)
    old.dump(out / f"{label}.json", dict(**metrics, equations_Q=[str(e) for e in prior.expressions_Q()],
        equations_z=[str(e) for e in prior.expressions_z()],
        learned_shared_coefficients=prior.global_coefficients.detach().cpu().numpy(),
        center=prior.coefficient_center.detach().cpu().numpy(), scale=prior.coefficient_scale.detach().cpu().numpy()))
    Q = z * prior.coefficient_scale.detach().cpu().numpy() + prior.coefficient_center.detach().cpu().numpy()
    np.savez_compressed(out / f"{label}.npz", source_z=z, source_Q=Q, symbol_prediction=pred,
        derivative_reference=Y, task_ids=tasks, first80=first)
    check_x = X[:: max(1, len(X) // 64)]
    inp = torch.tensor(check_x, dtype=torch.float64, device=model.device, requires_grad=True)
    actual = prior(inp, use_raw_data=True)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), prior.numpy(check_x), rtol=2e-6, atol=2e-6)
    grads = torch.autograd.grad(actual.sum(), [inp, prior.global_coefficients], allow_unused=False)
    assert all(torch.isfinite(g).all() for g in grads)
    return metrics


def run(args):
    torch.set_num_threads(2)
    old.seed_all(44)
    start = time.perf_counter()
    system, engine = args.system, args.engine
    out = ROOT / "runs" / engine / system
    out.mkdir(parents=True, exist_ok=True)
    manifest42 = read(P42 / "runs" / system / "run_manifest.json")
    cp = old.PROJECT / manifest42["packaged_source_checkpoint"]
    p = (EXP / "37_label_free_latent_symbolic_closure/artifacts" / system / "latent_expressions.json"
         if engine == "gplearn" else P42 / "runs" / system / "round_01/discovery/equations.json")
    source = read(p)
    equations = [r["calibrated_expression"] for r in source] if engine == "gplearn" else source["equations"]
    guards = {str(path): old.sha(path) for path in [cp, p, Path(__file__), ROOT / "plan/experiment-protocol.md",
        P42 / "scripts/run_closure.py", old.PROJECT / "compare2/DSSDL/DSSDL_.py"]}
    old.dump(out / "manifest.json", dict(status="running", protected_hashes=guards,
        args=vars(args), physical_parameter_labels_used=False, physical_decoder_used=False,
        human_assisted=True, manually_inserted_missing_terms=False))
    _, dataset, module, pretrain, _ = old.shared.multisystem.import_dssdl_for_system("FHN" if system == "fhn" else "lorenz")
    pretrain.CONFIG_OVERRIDES["reg_rate"] = 0.0
    cfg = dict(device=args.device, hidden_widths=[64, 64], specific_param_dim=3)
    model = old.model_for(module, dataset, cfg, cp, old.runner.ZeroPrior(len(old.NAMES[system])))
    qold = model.one_hot_to_specific_param.weight.detach().cpu().numpy().T.copy()
    with np.load(P42 / "runs" / system / "round_01/discovery/probe.npz") as data:
        np.testing.assert_array_equal(qold, data["z"])
        train_tasks = data["train_tasks"].copy()
    templates, constants, center, scales, Q, audit = simplify_first(system, engine, equations, qold, train_tasks)
    old.dump(out / "first_round_simplification.json", audit)
    prior = CompactPrior(system, templates, constants, center, scales, dataset).to(args.device)
    model.f_hat_func = prior
    c = prior.coefficient_center.detach().cpu().numpy()
    s = prior.coefficient_scale.detach().cpu().numpy()
    z = (Q - c) / s
    with torch.no_grad():
        model.one_hot_to_specific_param.weight.copy_(torch.tensor(z.T, device=args.device, dtype=torch.float32))
        model.specific_param_generator.weight.copy_(model.one_hot_to_specific_param.weight.mean(1, keepdim=True))
    np.savez_compressed(out / "coordinate_initialization.npz", old_q=qold, discovered_Q=Q,
        initialized_z=z, search_training_tasks=train_tasks, center=c, scale=s)
    np.testing.assert_allclose(z * s + c, Q, rtol=1e-6, atol=1e-7)
    Xold, Y, task, first = source_arrays(dataset, qold)
    names = old.NAMES[system] + ["q1", "q2", "q3"]
    original_prediction = np.column_stack([old.shared.evaluate_sympy_numpy(sp.sympify(e), names, Xold) for e in equations])
    Xnew, _, _, _ = source_arrays(dataset, z)
    compact_prediction = prior.numpy(Xnew)
    scale_y = np.asarray(dataset["info"]["normalization"]["output"]["std"])
    old.dump(out / "projection_audit.json", dict(
        original_symbol_source_time_val_nrmse=np.sqrt(np.mean(((original_prediction[~first] - Y[~first]) / scale_y) ** 2, axis=0)),
        compact_symbol_source_time_val_nrmse=np.sqrt(np.mean(((compact_prediction[~first] - Y[~first]) / scale_y) ** 2, axis=0)),
        projection_change_source_nrmse=np.sqrt(np.mean(((compact_prediction - original_prediction) / scale_y) ** 2, axis=0)),
        is_exact_algebraic_simplification=False, added_state_monomials=False))
    tensors = old.source_tensors(dataset, args.device)
    initial = export_stage(model, dataset, tensors, out, "compact_before_feedback")
    training_dir = out / "feedback"
    trained = training_dir / "model.pt"
    if trained.exists():
        model.load_state_dict(torch.load(trained, map_location=args.device, weights_only=True), strict=True)
    else:
        trained = old.train_round(model, dataset, tensors, training_dir, args.epochs, True, 44)
    final = export_stage(model, dataset, tensors, out, "compact_after_feedback")
    torch.save(model.state_dict(), out / "compact_model.pt")
    old.dump(out / "feedback_summary.json", dict(initial=initial, final=final, epochs=args.epochs,
        compact_Q_equations=[str(e) for e in prior.expressions_Q()],
        original_task_dim=3, new_task_dim=3, shared_scalar_count=len(constants),
        prior_is_used_in_original_loss=True, symbol_reg_rate=1))
    assert not old.TRIPWIRE_CALLS
    assert all(old.sha(path) == digest for path, digest in guards.items())
    logs = pd.read_csv(training_dir / "training.csv")
    selection = read(training_dir / "training_selection.json")
    assert selection["best_epoch"] == int(logs.loc[logs.source_val_rmse_norm.idxmin(), "epoch"])
    old.dump(out / "manifest.json", dict(status="feedback_complete", protected_hashes=guards,
        args=vars(args), human_assisted=True, physical_parameter_labels_used=False,
        physical_decoder_used=False, manually_inserted_missing_terms=False,
        protected_inputs_unchanged=True, coordinate_initialization_verified=True,
        numpy_torch_match=True, gradients_finite=True, forbidden_calls=old.TRIPWIRE_CALLS,
        next_symbolic_search_completed=False, wall_seconds=time.perf_counter() - start))
    print("COMPLETE FEEDBACK", engine, system, flush=True)
    print("INITIAL", initial, flush=True)
    print("FINAL", final, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["fhn", "lorenz"], required=True)
    parser.add_argument("--engine", choices=["gplearn", "pysr"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    run(parser.parse_args())
