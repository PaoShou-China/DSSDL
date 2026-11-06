"""Causal state-forecast evaluation; existing source methods remain read-only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
PROJECT = EXP.parents[1]
CFG = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parameter(t, system, profile):
    t = np.asarray(t)
    lo, hi = CFG["ranges"][system]
    elapsed = np.maximum(t - CFG["initial_hold_s"], 0)
    if profile == "constant":
        q = np.zeros_like(t)
    elif profile == "jump_return":
        a, b = CFG["jump_times_s"]
        q = ((t >= a) & (t < b)).astype(float)
    elif profile in CFG["sine_period_s"]:
        q = (1 - np.cos(2 * np.pi * elapsed / CFG["sine_period_s"][profile])) / 2
    elif profile == "ramp":
        q = np.clip(elapsed / (CFG["ramp_end_s"] - CFG["initial_hold_s"]), 0, 1)
    else:
        raise ValueError(profile)
    return lo + (hi - lo) * q


def generate(system, profile, seed):
    path = ROOT / "data" / system / profile / f"seed_{seed:03d}.npz"
    if path.exists():
        return path
    dt = CFG["dt"]
    ts = np.arange(round(CFG["duration_s"] / dt) + 1) * dt
    rng = np.random.RandomState(41000 + seed)
    if system == "fhn":
        initial = rng.uniform(-2, 2, size=2)
    else:
        pool = np.loadtxt(PROJECT / "compare2/DSSDL/datasets/lorenz/train/features/10_27_2.67.txt")
        initial = pool[rng.randint(len(pool))]

    def rhs(t, state):
        p = float(parameter(t, system, profile))
        if system == "fhn":
            v, w = state
            return [v - v**3 / 3 - w + p, 0.2 * (v + 0.7 - 0.9 * w)]
        x, y, z = state
        return [10 * (y - x), x * (p - z) - y, x * y - 2.67 * z]

    states = np.empty((len(ts), len(initial)))
    states[0] = initial
    # Integrate each smooth interval separately; only the simulator knows boundaries.
    boundaries = [0., 4., 8., 16., 20., CFG["duration_s"]]
    current = initial
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = np.flatnonzero((ts > left) & (ts <= right))
        solution = solve_ivp(
            lambda t, x: rhs(min(t, np.nextafter(right, left)), x),
            (left, right), current, t_eval=ts[indices],
            method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.05,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        states[indices] = solution.y.T
        current = solution.y[:, -1]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary, t=ts, states=states, parameter_reference=parameter(ts, system, profile))
    os.replace(temporary, path)
    return path


def origins():
    return np.arange(
        round(CFG["origin_start_s"] / CFG["dt"]),
        round((CFG["duration_s"] - max(CFG["horizon_s"])) / CFG["dt"]) + 1,
        round(CFG["update_interval_s"] / CFG["dt"]), dtype=int,
    )


def support_pairs(states, origin, steps):
    if origin < steps or origin >= len(states):
        raise ValueError("Invalid causal support")
    history = states[origin - steps:origin + 1]
    return history[:-1].copy(), np.diff(history, axis=0) / CFG["dt"]


def source_bundle(system, method):
    symbolic = method == "paper_dssdl"
    folder = PROJECT / "compare2" / ("DSSDL-symbol" if symbolic else "DSSDL")
    sys.path.insert(0, str(folder))
    import datasets
    datasets.set_current_system("FHN" if system == "fhn" else "lorenz")
    import DSSDL_ as original
    import pretrain
    pretrain.CONFIG_OVERRIDES["reg_rate"] = 1.0 if symbolic else 0.0
    original.CURRENT_SYSTEM = datasets.CURRENT_SYSTEM
    dataset = datasets.dataset_dict
    n = dataset["info"]["normalization"]
    norm = {f"{part}_{stat}": np.asarray(n[part][stat], dtype=float)
            for part in ("input", "output") for stat in ("mean", "std")}
    provenance = {"source_module": str(folder / "DSSDL_.py"), "normalization":
                  {k: v.tolist() for k, v in norm.items()}, "method": method,
                  "protocol_sha256": digest(ROOT / "protocol.json")}
    guarded = [folder / "DSSDL_.py", folder / "pretrain.py", folder / "datasets/utils.py"]
    guarded += sorted((folder / "datasets" / datasets.CURRENT_SYSTEM / "train").rglob("*.txt"))
    if method == "sindy":
        return None, norm, provenance, {str(p): digest(p) for p in guarded}
    if method == "pysr38":
        path = EXP / "38_pysr_direct_coefficient_no_decoder/scripts/run_experiment.py"
        old = load_module("exp38_readonly", path)
        checkpoint = EXP / "38_pysr_direct_coefficient_no_decoder/artifacts" / system / "model/direct_coefficient.pt"
        expressions, expression_info = old.load_pysr_expressions(system)
        prior = old.DirectCoefficientPrior(system, expressions, dataset)
        guarded += [path, EXP / "27_pysr_reproducibility_closure/exact_rerun" / system / "seed_000/tables/active_expressions.csv"]
        provenance["expressions"] = expressions
        provenance["expression_info"] = expression_info
    else:
        if system == "fhn":
            checkpoint = folder / "model/pretrain.pt"
        else:
            branch = "dssdl_symbol_scratch" if symbolic else "dssdl_scratch"
            checkpoint = EXP / "01_reproduction/lorenz" / branch / "pretrain_200/seed_000/model/pretrain.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dim = dataset["info"]["input_dim"]
    widths = [state[k].shape[0] for k in ("backbone.0.weight", "backbone.2.weight")]
    model = original.MetaNet(dim, dim, state["one_hot_to_specific_param.weight"].shape[1],
                             3, widths, "cpu", dataset, False)
    model.load_state_dict(state, strict=True)
    if method == "pysr38":
        model.f_hat_func = prior
    model.eval()
    model.set_finetune()
    for p in model.one_hot_to_specific_param.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.specific_param_generator.weight.copy_(model.one_hot_to_specific_param.weight.mean(1, keepdim=True))
    assert [k for k, p in model.named_parameters() if p.requires_grad] == ["specific_param_generator.weight"]
    provenance.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      weight_decay=CFG["pysr38_weight_decay" if method == "pysr38" else "paper_weight_decay"])
    guarded.append(checkpoint)
    return model, norm, provenance, {str(p): digest(p) for p in guarded}


def adapt(model, norm, X, derivative, method):
    xs = torch.as_tensor((X - norm["input_mean"]) / norm["input_std"], dtype=torch.float32)
    ys = torch.as_tensor((derivative - norm["output_mean"]) / norm["output_std"], dtype=torch.float32)
    z = model.specific_param_generator.weight
    before = z.detach().clone()
    opt_cfg = CFG["optimizer"]
    opt = torch.optim.LBFGS([z], lr=opt_cfg["lr"], max_iter=opt_cfg["max_iter"])
    decay = CFG["pysr38_weight_decay" if method == "pysr38" else "paper_weight_decay"]
    evaluations, stagnant, last = 0, 0, float("inf")

    def loss_value():
        net, prior = model.finetune_forward(xs)
        return torch.mean((net + prior - ys)**2) + decay * torch.sum(z**2)

    def closure():
        nonlocal evaluations
        evaluations += 1
        opt.zero_grad()
        loss = loss_value()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite adaptation loss")
        loss.backward()
        return loss

    start = time.perf_counter()
    failure = ""
    try:
        for _ in range(opt_cfg["max_outer_steps"]):
            opt.step(closure)
            with torch.no_grad():
                current = float(loss_value())
            if not np.isfinite(current):
                raise FloatingPointError("Nonfinite support loss")
            stagnant = stagnant + 1 if abs(last - current) < opt_cfg["support_tolerance"] else 0
            last = current
            if stagnant >= opt_cfg["support_patience"]:
                break
    except (RuntimeError, FloatingPointError) as error:
        failure = str(error)
        with torch.no_grad():
            z.copy_(before)
    return z.detach().numpy().reshape(-1).copy(), last, evaluations, time.perf_counter() - start, failure


def neural_forecast(model, norm, initial, latents, horizon_steps, limit):
    n = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in norm.items()}
    x = torch.as_tensor(initial, dtype=torch.float32)
    z = torch.as_tensor(latents, dtype=torch.float32)
    result = np.full((len(initial), horizon_steps, initial.shape[1]), np.nan)
    valid = torch.ones(len(initial), dtype=torch.bool)

    def rhs(states):
        inp = torch.cat([(states - n["input_mean"]) / n["input_std"], z], dim=1)
        out = model.backbone(inp) + model.f_hat_func(inp, model.dataset_dict, False)
        return out * n["output_std"] + n["output_mean"]

    # Equivalent independent forecasts with a distinct, frozen task vector per row.
    dt = CFG["dt"]
    with torch.inference_mode():
        for h in range(horizon_steps):
            k1 = rhs(x)
            k2 = rhs(x + dt * k1 / 2)
            k3 = rhs(x + dt * k2 / 2)
            k4 = rhs(x + dt * k3)
            new = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            valid &= torch.isfinite(new).all(1) & (new.abs().amax(1) <= limit)
            result[valid.numpy(), h] = new[valid].numpy()
            x = torch.where(valid[:, None], new, torch.zeros_like(new))
    return result


def sindy_forecast(initial, coefficients, powers, steps, limit):
    x = initial.copy()
    output = np.full((len(x), steps, x.shape[1]), np.nan)
    valid = np.isfinite(coefficients).all((1, 2))
    coefficients = np.nan_to_num(coefficients)
    def rhs(states):
        features = np.prod(states[:, None, :] ** powers[None, :, :], axis=-1)
        return np.einsum("nj,ndj->nd", features, coefficients)
    dt = CFG["dt"]
    with np.errstate(all="ignore"):
        for h in range(steps):
            k1 = rhs(x)
            k2 = rhs(x + dt*k1/2)
            k3 = rhs(x + dt*k2/2)
            k4 = rhs(x + dt*k3)
            new = x + dt*(k1+2*k2+2*k3+k4)/6
            valid &= np.isfinite(new).all(1) & (np.abs(new).max(1) <= limit)
            output[valid, h] = new[valid]
            x = np.where(valid[:, None], new, 0)
    return output


def metrics(predictions, reference, scale):
    failed = not np.isfinite(predictions).all()
    if failed:
        return dict(failed=True, rmse=float("inf"), nrmse=float("inf"), endpoint_rmse=float("inf"))
    error = predictions - reference
    return dict(failed=False, rmse=float(np.sqrt(np.mean(error**2))),
                nrmse=float(np.sqrt(np.mean((error / scale)**2))),
                endpoint_rmse=float(np.sqrt(np.mean(error[-1]**2))))


def worker(system, method, smoke=False, selected_profiles=None):
    work = ROOT / ("smoke" if smoke else "runs") / system / method
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    model, norm, provenance, guarded = source_bundle(system, method)
    if (work / "guarded_files.json").exists():
        previous_guards = json.loads((work / "guarded_files.json").read_text(encoding="utf-8"))
        if previous_guards != guarded:
            raise RuntimeError("Source files changed; use a new experiment directory")
    dump(work / "provenance.json", provenance)
    dump(work / "guarded_files.json", guarded)
    if model is not None:
        initial_z = model.specific_param_generator.weight.detach().clone()
        immutable = {k: v.detach().clone() for k, v in model.state_dict().items()
                     if k != "specific_param_generator.weight"}
    else:
        old = load_module("exp37_sindy_readonly", EXP / "37_label_free_latent_symbolic_closure/scripts/run_label_free_experiment.py")
        from sklearn.preprocessing import PolynomialFeatures
        powers = PolynomialFeatures(3).fit(np.zeros((1, len(norm["input_mean"])))).powers_
    oi = origins()[:3] if smoke else origins()
    H = round(max(CFG["horizon_s"]) / CFG["dt"])
    profiles = ["jump_return"] if smoke else (selected_profiles or CFG["profiles"])
    seeds = [0] if smoke else CFG["seeds"]
    windows = [0.5] if smoke else CFG["window_s"]
    audit_path = work / "audit.json"
    prior_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audited_batch = bool(prior_audit.get("batch_forward_equivalence", False) and
                         prior_audit.get("protocol_sha256") == digest(ROOT/"protocol.json"))
    for profile in profiles:
        for seed in seeds:
            data_path = generate(system, profile, seed)
            with np.load(data_path) as data:
                states = data["states"]
            for window_s in windows:
                dest = work / profile / f"seed_{seed:03d}" / f"window_{window_s:g}"
                if (dest / "complete.json").exists() and not smoke:
                    completed = json.loads((dest / "complete.json").read_text(encoding="utf-8"))
                    if completed["protocol_sha256"] != digest(ROOT/"protocol.json"):
                        raise RuntimeError("Protocol changed; use a new experiment directory")
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                if model is not None:
                    with torch.no_grad():
                        model.specific_param_generator.weight.copy_(initial_z)
                fitted, details = [], []
                for origin in oi:
                    X, derivative = support_pairs(states, origin, round(window_s / CFG["dt"]))
                    if model is not None:
                        fitted_z, loss, n_eval, elapsed, fail = adapt(model, norm, X, derivative, method)
                        fitted.append(fitted_z)
                        details.append(dict(support_loss=loss, evaluations=n_eval, update_s=elapsed, adaptation_failure=fail))
                    else:
                        start = time.perf_counter()
                        error, threshold, num_warnings = "", float("nan"), 0
                        try:
                            with warnings.catch_warnings(record=True) as ws:
                                fitted_model, coef, threshold = old.fit_sindy_support(X, derivative)
                            num_warnings = len(ws)
                            if not audited_batch:
                                theta = np.prod(X[:3, None, :] ** powers[None, :, :], -1)
                                np.testing.assert_allclose(theta @ coef.T, fitted_model.predict(X[:3]), rtol=1e-7, atol=1e-8)
                                audited_batch = True
                            loss = float(np.mean((fitted_model.predict(X) - derivative)**2))
                        except Exception as exc:
                            error = repr(exc)
                            coef = np.full((states.shape[1], len(powers)), np.nan)
                            loss = float("inf")
                        fitted.append(coef)
                        details.append(dict(support_loss=loss, evaluations=8, update_s=time.perf_counter()-start,
                                            adaptation_failure=error, threshold=threshold, warning_count=num_warnings))
                fitted = np.asarray(fitted)
                truth = np.stack([states[o+1:o+H+1] for o in oi])
                start = time.perf_counter()
                if model is not None:
                    if not audited_batch:
                        x = torch.as_tensor((states[oi[-1]:oi[-1]+1] - norm["input_mean"]) / norm["input_std"], dtype=torch.float32)
                        with torch.inference_mode():
                            net, prior = model.finetune_forward(x)
                            inp = torch.cat([x, torch.as_tensor(fitted[-1:], dtype=torch.float32)], 1)
                            torch.testing.assert_close(net+prior, model.backbone(inp)+model.f_hat_func(inp, model.dataset_dict, False))
                        audited_batch = True
                    online = neural_forecast(model, norm, states[oi], fitted, H, CFG["state_limit"][system])
                    frozen = neural_forecast(model, norm, states[oi], np.repeat(fitted[:1], len(oi), 0), H, CFG["state_limit"][system])
                    modes = {"online": online, "frozen_after_initial": frozen}
                else:
                    online = sindy_forecast(states[oi], fitted, powers, H, CFG["state_limit"][system])
                    modes = {"online": online}
                forecast_s = time.perf_counter()-start
                rows = []
                for mode, prediction in modes.items():
                    for idx, origin in enumerate(oi):
                        for horizon in CFG["horizon_s"]:
                            steps = round(horizon / CFG["dt"])
                            row = dict(system=system, method=method, profile=profile, seed=seed, window_s=window_s,
                                       mode=mode, origin_s=origin*CFG["dt"], horizon_s=horizon,
                                       support_last_time_s=origin*CFG["dt"], future_steps=steps)
                            row.update(details[idx] if mode == "online" else details[0])
                            if mode != "online" and idx > 0:
                                row["update_s"] = 0.
                                row["evaluations"] = 0
                            row.update(metrics(prediction[idx, :steps], truth[idx, :steps], norm["input_std"]))
                            rows.append(row)
                pd.DataFrame(rows).to_csv(dest / "window_metrics.csv", index=False)
                np.savez_compressed(dest / "forecasts.npz", origin_index=oi, truth=truth, fitted=fitted, **modes)
                dump(dest / "complete.json", dict(protocol_sha256=digest(ROOT/"protocol.json"), data_sha256=digest(data_path),
                                                  forecast_batch_s=forecast_s, windows=len(oi), metrics=len(rows)))
                print(f"DONE {system} {method} {profile} seed={seed} W={window_s} updates={len(oi)}", flush=True)
    if model is not None:
        for k, before in immutable.items():
            assert torch.equal(before, model.state_dict()[k]), k
    changed = [p for p, expected in guarded.items() if digest(p) != expected]
    if changed:
        raise AssertionError(f"Existing files changed: {changed}")
    dump(work / "audit.json", dict(unchanged_external_files=len(guarded), immutable_model_parameters=True,
                                   batch_forward_equivalence=audited_batch, no_decoder=True,
                                   no_target_parameter_labels=True, protocol_sha256=digest(ROOT/"protocol.json")))


def tests():
    data = np.arange(30, dtype=float).reshape(15, 2)
    X, y = support_pairs(data, 8, 5)
    modified = data.copy()
    modified[9:] += 123456
    X2, y2 = support_pairs(modified, 8, 5)
    np.testing.assert_equal(X, X2)
    np.testing.assert_equal(y, y2)
    assert X.shape == (5, 2)
    np.testing.assert_equal(X[-1], data[7])
    assert len(origins()) == 45
    assert origins()[0] >= round(max(CFG["window_s"])/CFG["dt"])
    assert origins()[-1] + round(max(CFG["horizon_s"])/CFG["dt"]) == 2400
    np.testing.assert_equal(parameter(np.array([7.999, 8., 15.999, 16.]), "lorenz", "jump_return"), [20,29,29,20])
    assert metrics(np.ones((25, 2)), np.zeros((25, 2)), np.ones(2))["rmse"] == 1
    assert metrics(np.array([[np.nan, 0.]]), np.zeros((1,2)), np.ones(2))["failed"]
    # Constant polynomial derivative: H future samples, no zero-error initial sample.
    from sklearn.preprocessing import PolynomialFeatures
    powers = PolynomialFeatures(3).fit(np.zeros((1, 2))).powers_
    coef = np.zeros((1, 2, len(powers)))
    coef[0, :, 0] = 1
    forecast = sindy_forecast(np.zeros((1, 2)), coef, powers, 25, 100)
    np.testing.assert_allclose(forecast[0, 0], .01)
    np.testing.assert_allclose(forecast[0, -1], .25)
    dump(ROOT / "tests/causality.json", dict(causal_support=True, exact_future_steps=True, common_origins=True,
                                             explicit_failure=True, jump_boundary=True))
    print("PASS causality, horizons, common origins, failures, jump boundaries", flush=True)


def run_all(smoke=False, jobs=1, only_system=None, only_method=None, split_profiles=False):
    tests()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    for system in CFG["systems"]:
        for profile in CFG["profiles"]:
            for seed in CFG["seeds"]:
                generate(system, profile, seed)
    def dispatch(pair):
        system, method, profile = pair
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--system", system, "--method", method]
        if smoke:
            command.append("--smoke")
        if profile is not None:
            command += ["--profiles", profile]
        log = logs / f"{'smoke_' if smoke else ''}{system}_{method}{'_' + profile if profile else ''}.log"
        print(f"START {system} {method} log={log}", flush=True)
        with log.open("w", encoding="utf-8") as out:
            status = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if status.returncode:
            print(log.read_text(encoding="utf-8", errors="replace")[-7000:], flush=True)
            raise RuntimeError(f"Worker failed: {system}/{method}")
        print(f"COMPLETE {system} {method}", flush=True)
    pairs = [(system, method, profile) for system in CFG["systems"] for method in CFG["methods"]
             for profile in (CFG["profiles"] if split_profiles else [None])
             if (only_system is None or system == only_system) and (only_method is None or method == only_method)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(dispatch, pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["test", "all", "worker"])
    parser.add_argument("--system", choices=CFG["systems"])
    parser.add_argument("--method", choices=CFG["methods"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--profiles", choices=CFG["profiles"], nargs="+")
    parser.add_argument("--split-profiles", action="store_true")
    args = parser.parse_args()
    if args.action == "test":
        tests()
    elif args.action == "all":
        run_all(args.smoke, args.jobs, args.system, args.method, args.split_profiles)
    else:
        worker(args.system, args.method, args.smoke, args.profiles)
