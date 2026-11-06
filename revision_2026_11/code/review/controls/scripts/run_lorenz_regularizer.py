"""Paired fixed-teacher, fixed-template regularizer ablation using experiment 44."""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import run_paired_ablation as common
import numpy as np
import sympy as sp
import torch

ROOT = common.ROOT
EXP = common.EXPERIMENTS


def prepare(device):
    entry = EXP / "symbolic_feedback/scripts/run_feedback.py"
    base = common.load("round2_lorenz44", entry)
    old = base.old
    old.seed_all(44)
    provenance = base.read(base.P42 / "runs/lorenz/run_manifest.json")
    checkpoint = base.old.PROJECT / provenance["packaged_source_checkpoint"]
    spec_path = base.ROOT / "runs/pysr/lorenz/first_round_simplification.json"
    init_path = spec_path.parent / "coordinate_initialization.npz"
    spec = base.read(spec_path)
    _, dataset, module, pretrain, _ = old.shared.multisystem.import_dssdl_for_system("lorenz")
    pretrain.CONFIG_OVERRIDES["reg_rate"] = 0.0
    cfg = dict(device=device, hidden_widths=[64, 64], specific_param_dim=3)
    model = old.model_for(module, dataset, cfg, checkpoint, old.runner.ZeroPrior(3))
    with np.load(init_path) as archive:
        np.testing.assert_array_equal(
            model.one_hot_to_specific_param.weight.detach().cpu().numpy().T, archive["old_q"]
        )
        z = archive["initialized_z"].copy()
        Q = archive["discovered_Q"].copy()
    prior = base.CompactPrior(
        "lorenz",
        [sp.sympify(t) for t in spec["new_templates"]],
        spec["shared_coefficient_initial_values"],
        spec["coefficient_center"],
        spec["coefficient_scale"],
        dataset,
    ).to(device)
    model.f_hat_func = prior
    with torch.no_grad():
        model.one_hot_to_specific_param.weight.copy_(
            torch.as_tensor(z.T, device=device, dtype=torch.float32)
        )
        model.specific_param_generator.weight.copy_(
            model.one_hot_to_specific_param.weight.mean(1, keepdim=True)
        )
    c = prior.coefficient_center.cpu().numpy()
    s = prior.coefficient_scale.cpu().numpy()
    np.testing.assert_allclose(z * s + c, Q, rtol=1e-6, atol=1e-7)
    raw = np.c_[dataset["raw"]["train"][0][0][:16], np.repeat(z[:1], 16, axis=0)]
    inp = torch.tensor(raw, device=device, dtype=torch.float64, requires_grad=True)
    actual = prior(inp, use_raw_data=True)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), prior.numpy(raw), rtol=2e-6, atol=2e-6
    )
    grads = torch.autograd.grad(actual.sum(), [inp, prior.global_coefficients])
    assert all(torch.isfinite(g).all() for g in grads)
    guards = {
        str(p): common.file_hash(p)
        for p in (
            entry,
            base.P42 / "scripts/run_closure.py",
            base.P42 / "runs/lorenz/run_manifest.json",
            checkpoint,
            spec_path,
            init_path,
            old.PROJECT / "compare2/DSSDL/DSSDL_.py",
            Path(sys.modules["datasets"].__file__),
        )
    }
    for p in (old.PROJECT / "compare2/DSSDL/datasets/lorenz").rglob("*.txt"):
        guards[str(p)] = common.file_hash(p)
    tensors = old.source_tensors(dataset, device)
    target = {
        split: [
            torch.as_tensor(np.concatenate(v), dtype=torch.float32, device=device)
            for v in dataset["normalized"][split]
        ]
        for split in ("support", "query")
    }
    qx, qy, overlap = common.disjoint_query(*target["support"], *target["query"])
    raw_source = np.concatenate([np.c_[x, y] for x, y in zip(*dataset["raw"]["train"])])
    source_keys = {row.tobytes() for row in raw_source}
    raw_query = np.c_[
        np.concatenate(dataset["raw"]["query"][0]), np.concatenate(dataset["raw"]["query"][1])
    ]
    source_overlap = sum(row.tobytes() in source_keys for row in raw_query)
    assert source_overlap == 0, "Query duplicates source rows; report before proceeding"
    manifest = dict(
        system="lorenz",
        fixed_teacher=True,
        source_checkpoint=str(checkpoint),
        source_code=str(entry),
        source_tasks=len(dataset["raw"]["train"][0]),
        source_rows=[len(x) for x in dataset["raw"]["train"][0]],
        source_train_indices=tensors[2],
        source_val_indices=tensors[3],
        source_selection="minimum full-model source time-validation NRMSE after 100 epochs",
        physical_labels_used=False,
        decoder=False,
        symbolic_search_rerun=False,
        source_normalization=dataset["info"]["normalization"],
        fixed_symbolic_templates=spec["new_templates"],
        initial_constants=spec["shared_coefficient_initial_values"],
        coefficient_center=c,
        coefficient_scale=s,
        epochs=100,
        batch_size=512,
        source_learning_rate=0.001,
        source_weight_decay=0.0001,
        support_rows=len(target["support"][0]),
        query_rows_original=len(target["query"][0]),
        query_rows=len(qx),
        removed_support_duplicates=overlap,
        query_exact_source_duplicates=source_overlap,
        target_configuration_unseen=False,
        target_configuration_note="The original target file identifier 10_20_2.67 also occurs in source data; observations differ. No numeric parameter labels are passed to training.",
        data_hash=common.tensor_hash(
            [tensors[0], tensors[1], tensors[2], tensors[3], *target["support"], qx, qy]
        ),
        initialization_hash=common.state_hash(model),
        protected_hashes=guards,
        torch_version=torch.__version__,
        device=device,
        numerical_prior_validation=True,
    )
    assert not old.TRIPWIRE_CALLS
    return base, model, dataset, tensors, target["support"], qx, qy, manifest


def source_predictions(model, tensors):
    X, Y, _, val = tensors
    net, prior = [], []
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(val, max(1, int(np.ceil(len(val) / 4096)))):
            n, p = model(X[batch])
            net.append(n.cpu().numpy())
            prior.append(p.cpu().numpy())
    return np.concatenate(net), np.concatenate(prior), Y[val].cpu().numpy()


def run(args):
    torch.set_num_threads(2)
    base, initial, dataset, tensors, (sx, sy), qx, qy, manifest = prepare(args.device)
    if args.audit_only:
        common.dump(ROOT / "audit/lorenz_regularizer_preflight.json", manifest)
        print(
            "PREFLIGHT OK",
            manifest["source_tasks"],
            manifest["support_rows"],
            manifest["query_rows"],
            flush=True,
        )
        return
    folder = ROOT / f"experiments/separation/lorenz/seed_{args.seed:04d}"
    assert not (folder / "summary.json").exists(), "Do not overwrite completed runs"
    manifest["seed"] = args.seed
    common.dump(folder / "manifest.json", manifest)
    rows = {}
    for variant, lam in (("lambda_1", 1.0), ("lambda_0", 0.0)):
        model = copy.deepcopy(initial)
        init_hash = common.state_hash(model)
        start = time.perf_counter()
        base.old.train_round(
            model,
            dataset,
            tensors,
            folder / variant / "feedback",
            100,
            bool(lam),
            args.seed - 42000,
        )
        source = base.old.source_scores(model, dataset, tensors)
        n, p, ref = source_predictions(model, tensors)
        fs2, fn2 = float(np.mean(p**2)), float(np.mean(n**2))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        qinit = model.one_hot_to_specific_param.weight.detach().mean(1)

        def forward(x, q):
            inputs = torch.cat([x, q.expand(len(x), -1)], dim=1)
            return model.backbone(inputs) + model.f_hat_func(inputs)

        metrics, prediction = common.adapt_and_score(forward, qinit, sx, sy, qx, qy)
        row = dict(
            variant=variant,
            lambda_reg=lam,
            initialization_hash=init_hash,
            val_full_nrmse=source["source_val_rmse_norm"],
            val_prior_nrmse=source["source_val_prior_rmse_norm"],
            active_slots=[0, 1, 2],
            source_scores=source,
            **metrics,
            symbolic_energy_fraction=fs2 / (fs2 + fn2),
            symbolic_rms=fs2**0.5,
            neural_rms=fn2**0.5,
            component_cross_term=2 * float(np.mean(n * p)),
            constants=model.f_hat_func.global_coefficients.detach(),
            equations_coefficient_coordinates=[str(e) for e in model.f_hat_func.expressions_Q()],
            wall_seconds=time.perf_counter() - start,
        )
        selection = base.read(folder / variant / "feedback/training_selection.json")
        row["source_selected_epoch"] = selection["best_epoch"]
        rows[variant] = row
        torch.save(model.state_dict(), folder / f"{variant}.pt")
        common.dump(folder / f"{variant}_metrics.json", row)
        np.savez_compressed(
            folder / f"{variant}_predictions.npz",
            prediction=prediction.cpu().numpy(),
            reference=qy.cpu().numpy(),
            source_val_prediction=n + p,
            source_val_symbolic=p,
            source_val_reference=ref,
            source_coordinates=model.one_hot_to_specific_param.weight.detach().cpu().numpy().T,
        )
        print(
            "RESULT",
            args.seed,
            variant,
            row["val_full_nrmse"],
            row["val_prior_nrmse"],
            row["query_nrmse_after"],
            flush=True,
        )
    assert rows["lambda_1"]["initialization_hash"] == rows["lambda_0"]["initialization_hash"]
    assert not base.old.TRIPWIRE_CALLS
    assert all(
        common.file_hash(Path(p)) == digest for p, digest in manifest["protected_hashes"].items()
    )
    common.dump(
        folder / "summary.json",
        dict(
            status="completed",
            evidence_level="fixed_teacher_feedback_repeat",
            manifest=manifest,
            results=rows,
            protected_inputs_unchanged=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42044)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())
