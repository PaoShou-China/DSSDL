#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finetune script built on the pretrained model.

Workflow:
- Load pretrained model
- Optional finetune on support data
- Evaluate on query data
- Record convergence epoch and training time
- Save results to results/{system}/
"""

from copy import deepcopy
import torch
import torch.nn as nn

from datasets import dataset_dict, CURRENT_SYSTEM
from DSSDL_ import (
    finetune as run_finetune,
    MetaNet,
    load_dataset_with_one_hot,
    config as BASE_CONFIG,
)
from pretrain import CONFIG_OVERRIDES

# ==================== Finetune config overrides ====================
# Override finetune-specific params on top of pretrain overrides.
FINETUNE_CONFIG_OVERRIDES = CONFIG_OVERRIDES.copy()
FINETUNE_CONFIG_OVERRIDES.update(
    {
        "epochs": 200,
        "convergence_threshold": 1e-5,
        "patience": 5,
        "batch_size": -1,
        "learning_rate": 0.1,
        "adam_lr": 0.0001,
        "optimizer": "lbfgs",
    }
)
# ===================================================================


def _build_effective_config(config_overrides):
    """Merge DSSDL_ defaults with overrides and return a safe copy."""
    cfg = deepcopy(BASE_CONFIG)
    if config_overrides:
        for key, value in config_overrides.items():
            if key in cfg:
                cfg[key] = value
    return cfg


def evaluate_query_without_finetune(dataset_dict, config_overrides):
    """
    Use the mean of the pretrained one_hot_to_specific_param outputs as the
    default slowparam (no finetune) and compute RMSE on the query set.
    """
    cfg = _build_effective_config(config_overrides)
    device = cfg.get("device", "cpu")
    specific_param_dim = config_overrides.get(
        "specific_param_dim",
        cfg.get("specific_param_dim"),
    )

    input_dim = dataset_dict["info"]["input_dim"]
    output_dim = dataset_dict["info"]["output_dim"]
    one_hot_dim = dataset_dict["info"]["file_counts"]["train"]

    query_loader, _ = load_dataset_with_one_hot(
        dataset_dict,
        split="query",
        batch_size=cfg.get("batch_size", 1),
        train_ratio=1.0,
        device=device,
        use_raw_data=cfg.get("use_raw_data", False),
    )

    model = MetaNet(
        input_dim=input_dim,
        output_dim=output_dim,
        one_hot_dim=one_hot_dim,
        specific_param_dim=specific_param_dim,
        hidden_widths=cfg["hidden_widths"],
        device=device,
        dataset_dict=dataset_dict,
        use_raw_data=cfg.get("use_raw_data", False),
    )
    state_dict = torch.load(
        "model/pretrain.pt",
        map_location=device,
        weights_only=True,
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: unexpected keys in state_dict ignored: {unexpected_keys}")
    model.eval()

    # Mean of the pretrained one-hot embedding serves as the default slowparam.
    with torch.no_grad():
        mean_embedding = model.one_hot_to_specific_param.weight.mean(dim=1, keepdim=True)
        # mean_embedding *= 0.0  # Zero mean for stability
        # mean_embedding/=mean_embedding
        model.specific_param_generator.weight.copy_(mean_embedding)

    # Save the no-finetune model for comparison with the finetuned one.
    no_finetune_path = "model/no_finetune.pt"
    torch.save(model.state_dict(), no_finetune_path)

    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in query_loader:
            outputs_net, outputs_hat = model.finetune_forward(features)
            total_loss += criterion(outputs_net + outputs_hat, labels).item()

    rmse = (total_loss / len(query_loader)) ** 0.5 if len(query_loader) > 0 else float("nan")
    mean_vector = mean_embedding.view(-1).detach().cpu().tolist()
    return rmse, mean_vector, no_finetune_path


def main():
    print(f"Current system: {CURRENT_SYSTEM}")
    print("=" * 60)

    # Baseline: no finetune, use mean slowparam from pretrained one-hot mapping.
    baseline_rmse, mean_slowparam, baseline_model_path = evaluate_query_without_finetune(
        dataset_dict=dataset_dict,
        config_overrides=FINETUNE_CONFIG_OVERRIDES,
    )
    mean_str = ", ".join(f"{v:.6f}" for v in mean_slowparam)
    print(f"Default slowparam (mean from pretrained one-hot): [{mean_str}]")
    print(f"Query RMSE without finetune: {baseline_rmse:.6f}")
    print(f"Saved no-finetune model to: {baseline_model_path}")
    print("=" * 60)

    # Run finetune (all params pulled from FINETUNE_CONFIG_OVERRIDES).
    result = run_finetune(
        dataset_dict=dataset_dict,
        config_overrides=FINETUNE_CONFIG_OVERRIDES,
    )

    print("\n" + "=" * 60)
    print("Finetune summary")
    print("=" * 60)
    max_epochs = FINETUNE_CONFIG_OVERRIDES.get("epochs", result["total_epochs"])
    ran_epochs = result["total_epochs"]  # length of log, actually executed
    converged_epoch = result["converged_epoch"]

    print(f"System: {CURRENT_SYSTEM}")
    print(f"Max epochs (config): {max_epochs}")
    print(f"Ran epochs: {ran_epochs}")
    print(f"Converged epoch: {converged_epoch}")
    print(f"Training time: {result['training_time']:.2f}s")
    print(f"Final loss: {result['final_loss']:.6e}")
    if ran_epochs < max_epochs:
        print(
            f"Converged early, saved {max_epochs - ran_epochs} epochs (stopped after epoch {converged_epoch})"
        )
    else:
        print("Ran to max epochs, no early convergence")
    print("=" * 60)


if __name__ == "__main__":
    main()
