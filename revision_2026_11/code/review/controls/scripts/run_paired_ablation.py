"""Read-only reuse of audited data/models for paired round-two pilot ablations."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
EXPERIMENTS = PROJECT / "review/experiments"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def tensor_hash(values):
    h = hashlib.sha256()
    for value in values:
        value = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def state_hash(model):
    return tensor_hash([v for k, v in sorted(model.state_dict().items())])


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disjoint_query(sx, sy, qx, qy):
    supports = torch.cat([sx, sy], dim=1).detach().cpu().numpy()
    queries = torch.cat([qx, qy], dim=1).detach().cpu().numpy()
    seen = {row.tobytes() for row in supports}
    keep = np.asarray([row.tobytes() not in seen for row in queries])
    assert keep.any(), "No held-out query samples"
    keep_t = torch.as_tensor(keep, device=qx.device)
    return qx[keep_t], qy[keep_t], int((~keep).sum())


def adapt_and_score(forward, initial, sx, sy, qx, qy, steps=500):
    latent = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam([latent], lr=.03)
    with torch.no_grad():
        before = float((forward(qx, latent) - qy).square().mean().sqrt())
    trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (forward(sx, latent) - sy).square().mean()
        assert torch.isfinite(loss), f"Nonfinite support loss, step {step}"
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            trace.append(dict(step=step + 1, support_mse_pre_update=float(loss.detach())))
    with torch.no_grad():
        pred = forward(qx, latent)
        support = (forward(sx, latent) - sy).square().mean().sqrt()
        query = (pred - qy).square().mean().sqrt()
    return dict(query_nrmse_before=before, query_nrmse_after=float(query),
                support_nrmse_after=float(support), steps=steps,
                selected_iteration="final", query_used_for_stopping=False,
                latent=latent.detach(), trace=trace), pred.detach()


def gate(system, seed, device):
    folder = ROOT / f"experiments/gate/{system}/seed_{seed:04d}"
    assert not (folder / "summary.json").exists(), "Existing completed run; do not overwrite"
    path = EXPERIMENTS / "scripts/dimension_discovery_claim_runner.py"
    base = load("round2_gate_base", path)
    args = base.normalize_args(base.build_parser().parse_args([
        "--system", system, "--seed", str(seed), "--pmax", "8", "--l0-lambda", "1e-4",
        "--hidden-widths", "128,128", "--gate-init-logit", "0.5", "--model-lr", ".005",
        "--gate-lr", ".005", "--temperature-start", "1", "--temperature-end", "1",
        "--warmup-epochs", "0", "--prune-epochs", "100", "--stabilize-epochs", "70",
        "--stabilize-lr", ".001", "--samples-per-task", "512", "--batch-size", "128",
        "--device", device, "--output-root", str(folder)]))
    base.set_seed(seed)
    bundle = base.load_data(system, seed=seed, val_fraction=.2, device=args.device_obj)
    qx, qy, overlap = disjoint_query(bundle.target_support_x, bundle.target_support_y,
                                     bundle.target_query_x, bundle.target_query_y)
    dataset_dir = base.METHOD_DIR / "datasets" / bundle.dataset_name
    guards = {str(p): file_hash(p) for p in dataset_dir.rglob("*.txt")}
    guards[str(path)] = file_hash(path)
    original_class = base.GatedDSSDL

    class GateDisabled(original_class):
        def hard_mask(self, *, temperature=1., straight_through=False):
            return torch.ones_like(self.gate_logits)

    manifest = dict(experiment="stage_I_gate_ablation", system=system, seed=seed,
                    settings=vars(args), source_tasks=bundle.num_tasks,
                    source_rows=[len(a) + len(b) for a, b in zip(bundle.train_x, bundle.val_x)],
                    support_rows=len(bundle.target_support_x), query_rows_original=len(bundle.target_query_x),
                    query_rows_disjoint=len(qx), removed_support_duplicates=overlap,
                    data_hash=tensor_hash(bundle.train_x + bundle.train_y + bundle.val_x + bundle.val_y +
                                          [bundle.target_support_x, bundle.target_support_y, qx, qy]),
                    physical_labels_used=False, decoder=False, symbolic_prior=False,
                    source_code=str(path), protected_hashes=guards,
                    torch_version=torch.__version__, device=device)
    dump(folder / "manifest.json", manifest)
    rows = {}
    for variant, cls, lam in (("gate_on", original_class, 1e-4), ("gate_off", GateDisabled, 0.)):
        base.set_seed(seed)
        base.GatedDSSDL = cls
        local = copy.deepcopy(args)
        local.l0_lambda = lam
        # Reset the RNG after checking that both model initializations are identical.
        probe = cls(input_dim=bundle.input_dim, output_dim=bundle.output_dim,
                    num_tasks=bundle.num_tasks, pmax=8, hidden_widths=[128,128],
                    gate_init_logit=.5, seed=seed)
        init_hash = state_hash(probe)
        del probe
        base.set_seed(seed)
        start = time.perf_counter()
        model, selection, history = base.train_gated(bundle, local)
        mask = model.hard_mask(straight_through=False).detach()
        for p in model.parameters():
            p.requires_grad_(False)
        initial = model.embedding.weight.detach().mean(0)
        metrics, pred = adapt_and_score(lambda x, q: model.forward_latent(x, q, mask=mask),
                                        initial, bundle.target_support_x, bundle.target_support_y, qx, qy)
        latent = (model.embedding.weight * mask).detach().cpu().numpy()
        singular = np.linalg.svd(latent - latent.mean(0), compute_uv=False)**2
        pr = float(singular.sum()**2 / max(float((singular**2).sum()), 1e-30))
        row = dict(variant=variant, initialization_hash=init_hash, active_dim=int(mask.sum()),
                   pmax=8, source_val_nrmse=selection["final_source_validation"]["mean_rmse"],
                   participation_ratio=pr, wall_seconds=time.perf_counter()-start, **metrics)
        rows[variant] = row
        torch.save(model.state_dict(), folder / f"{variant}.pt")
        dump(folder / f"{variant}_history.json", dict(selection=selection, history=history, metrics=row))
        np.savez_compressed(folder / f"{variant}_predictions.npz", prediction=pred.cpu().numpy(),
                            reference=qy.cpu().numpy(), source_latents=latent, mask=mask.cpu().numpy())
        print("RESULT", system, variant, row["active_dim"], row["query_nrmse_after"], flush=True)
    assert rows["gate_on"]["initialization_hash"] == rows["gate_off"]["initialization_hash"]
    assert all(file_hash(Path(p)) == h for p, h in guards.items())
    dump(folder / "summary.json", dict(status="completed", evidence_level="pilot" if seed == 0 else "replicate",
                                      manifest=manifest, results=rows, protected_inputs_unchanged=True))


def separation(seed, device):
    folder = ROOT / f"experiments/separation/pendulum/seed_{seed:04d}"
    assert not (folder / "summary.json").exists(), "Existing completed run; do not overwrite"
    path = EXPERIMENTS / "measured_pendulum_closure/scripts/run_closure.py"
    base = load("round2_pendulum_base", path)
    base.seed_all(seed)
    env = base.original_module()
    dataset, records = base.source_data()
    train, val, arrays = base.arrays_and_loaders(env, dataset, device)
    spec_path = base.protect(base.ROOT / "simplification.json")
    spec = base.read(spec_path)
    initial = base.make_model(env, dataset, base.NC / "model/meta_train.pt", device)
    initial.f_hat_func = base.ExpressionPrior(spec, dataset).to(device)
    initial.continuous_mask.requires_grad_(False)
    mask = (initial.get_mask(hard=True).detach() > .5).to(torch.float32)
    assert int(mask.sum()) == 1
    norms = dataset["info"]["normalization"]
    target = {}
    for split in ("support", "query"):
        tensors = []
        for kind, key in (("features", "input"), ("labels", "output")):
            paths = sorted((base.NC / f"datasets/pendulum_real/{split}/{kind}").glob("*.txt"))
            assert len(paths) == 1
            array = np.loadtxt(base.protect(paths[0]))
            tensors.append(torch.tensor((array-norms[key]["mean"])/norms[key]["std"],
                                         device=device, dtype=torch.float32))
        target[split] = tensors
    sx, sy = target["support"]
    qx, qy, overlap = disjoint_query(sx, sy, *target["query"])
    initial_state = copy.deepcopy(initial.state_dict())
    manifest = dict(system="measured_pendulum", source_code=str(path), seed=seed,
                    fixed_teacher=True, physical_labels_used=False, decoder=False,
                    source_tasks=7, source_train_indices=arrays["train"], source_val_indices=arrays["val"],
                    frozen_gate=mask, epochs=100, original_trainer=True, symbolic_search_rerun=False,
                    fixed_symbolic_templates=spec["templates"], initial_constants=spec["initial_constants"],
                    source_data=records, support_rows=len(sx), query_rows=len(qx),
                    removed_support_duplicates=overlap, config=env["config"], protected_hashes=base.GUARDS)
    dump(folder / "manifest.json", manifest)
    rows = {}
    for variant, lam in (("lambda_1", 1.), ("lambda_0", 0.)):
        base.seed_all(seed)
        model = copy.deepcopy(initial)
        model.load_state_dict(initial_state)
        init_hash = state_hash(model)
        start = time.perf_counter()
        trainer = env["Trainer"](model=model, learning_rate=.01, optimizer_type="adam")
        logs = trainer.meta_train(100, train, val, reg_rate=lam)
        source_scores, predictions = base.scores(model, arrays, dataset)
        ids = arrays["val"]
        fs = predictions["normalized_prior"][ids]
        fn = predictions["normalized_full"][ids] - fs
        fs2, fn2 = float(np.mean(fs**2)), float(np.mean(fn**2))
        for p in model.parameters():
            p.requires_grad_(False)
        def forward(x, q):
            inputs = torch.cat([x, (q*mask).expand(len(x), -1)], dim=1)
            return model.backbone(inputs) + model.f_hat_func(inputs)
        qinit = model.one_hot_to_specific_param.weight.detach().mean(1)
        metrics, pred = adapt_and_score(forward, qinit, sx, sy, qx, qy)
        row = dict(variant=variant, lambda_reg=lam, initialization_hash=init_hash,
                   **source_scores, **metrics, symbolic_energy_fraction=fs2/(fs2+fn2),
                   neural_rms=fn2**.5, symbolic_rms=fs2**.5,
                   component_cross_term=2*float(np.mean(fs*fn)),
                   equations_stored_units=model.f_hat_func.current_expressions(),
                   constants=model.f_hat_func.constants.detach(), wall_seconds=time.perf_counter()-start)
        rows[variant] = row
        torch.save(model.state_dict(), folder / f"{variant}.pt")
        dump(folder / f"{variant}_training.json", logs)
        np.savez_compressed(folder / f"{variant}_predictions.npz", prediction=pred.cpu().numpy(),
                            reference=qy.cpu().numpy(), **predictions)
        dump(folder / f"{variant}_metrics.json", row)
        print("RESULT", variant, row["val_full_nrmse"], row["val_prior_nrmse"], row["query_nrmse_after"], flush=True)
    assert rows["lambda_1"]["initialization_hash"] == rows["lambda_0"]["initialization_hash"]
    assert all(file_hash(Path(p)) == h for p, h in base.GUARDS.items())
    dump(folder / "summary.json", dict(status="completed", evidence_level="fixed_teacher_pilot",
                                      manifest=manifest, results=rows, protected_inputs_unchanged=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["gate", "separation"])
    parser.add_argument("--system", choices=["pendulum", "quadrotor"], default="pendulum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.experiment == "gate":
        gate(args.system, args.seed, args.device)
    else:
        separation(args.seed, args.device)
