#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified dimension-discovery experiments for the DSSDL reviewer claim.

The runner does not modify the original DSSDL implementation. It reuses the
original normalized datasets and implements the manuscript's shared binary STE
gate in an isolated experiment harness.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXP_ROOT.parents[1]
METHOD_DIR = PROJECT_ROOT / "compare2" / "DSSDL"
DEFAULT_OUTPUT_ROOT = EXP_ROOT / "22_dimension_discovery_claim"


@dataclass(frozen=True)
class SystemSpec:
    name: str
    dataset_name: str
    true_dim: int
    parameter_names: tuple[str, ...]


SPECS: dict[str, SystemSpec] = {
    "pendulum": SystemSpec(
        name="pendulum",
        dataset_name="pendulum_real",
        true_dim=1,
        parameter_names=("L",),
    ),
    "quadrotor": SystemSpec(
        name="quadrotor",
        dataset_name="quadrotor",
        true_dim=3,
        parameter_names=("I_x", "I_y", "I_z"),
    ),
    "lorenz": SystemSpec(
        name="lorenz",
        dataset_name="lorenz",
        true_dim=3,
        parameter_names=("sigma", "rho", "beta"),
    ),
    "fhn": SystemSpec(
        name="fhn",
        dataset_name="FHN",
        true_dim=3,
        parameter_names=("a", "b", "I_ext"),
    ),
}


@dataclass
class DataBundle:
    system: str
    dataset_name: str
    input_dim: int
    output_dim: int
    train_x: list[torch.Tensor]
    train_y: list[torch.Tensor]
    val_x: list[torch.Tensor]
    val_y: list[torch.Tensor]
    target_support_x: torch.Tensor
    target_support_y: torch.Tensor
    target_query_x: torch.Tensor
    target_query_y: torch.Tensor
    source_params: np.ndarray
    target_params: np.ndarray
    output_std: np.ndarray
    source_files: list[str]
    target_file: str

    @property
    def num_tasks(self) -> int:
        return len(self.train_x)


@dataclass
class ValidationMetrics:
    mean_rmse: float
    std_rmse: float
    se_rmse: float
    per_task_rmse: list[float]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def float_slug(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def parse_params(system: str, stem: str) -> list[float]:
    if system == "pendulum":
        return [float(stem)]
    if system == "quadrotor":
        match = re.fullmatch(r"0_(\d+)_0_(\d+)_0_(\d+)", stem)
        if not match:
            raise ValueError(f"Cannot parse quadrotor parameters from {stem}")
        return [float(f"0.{group}") for group in match.groups()]
    return [float(value) for value in stem.split("_")]


def import_dataset(dataset_name: str):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    import datasets

    datasets.set_current_system(dataset_name)
    return datasets.dataset_dict


def sorted_feature_paths(dataset_name: str, split: str) -> list[Path]:
    root = METHOD_DIR / "datasets" / dataset_name / split / "features"
    paths = sorted(root.glob("*.txt"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(root)
    return paths


def split_task(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) != len(y):
        raise ValueError("Feature and label lengths differ")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    val_count = max(1, int(round(len(x) * val_fraction)))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    if len(train_idx) == 0:
        raise ValueError("No training samples after validation split")
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def load_data(
    system: str,
    *,
    seed: int,
    val_fraction: float,
    device: torch.device,
) -> DataBundle:
    spec = SPECS[system]
    dataset_dict = import_dataset(spec.dataset_name)
    normalized = dataset_dict["normalized"]
    source_x, source_y = normalized["train"]
    support_x, support_y = normalized["support"]
    query_x, query_y = normalized["query"]
    source_paths = sorted_feature_paths(spec.dataset_name, "train")
    support_paths = sorted_feature_paths(spec.dataset_name, "support")

    if len(source_x) != len(source_paths):
        raise ValueError(
            f"{system}: dataset has {len(source_x)} tasks but {len(source_paths)} feature files"
        )
    if len(support_x) != 1 or len(query_x) != 1 or len(support_paths) != 1:
        raise ValueError(f"{system}: expected one support/query target task")

    train_x: list[torch.Tensor] = []
    train_y: list[torch.Tensor] = []
    val_x: list[torch.Tensor] = []
    val_y: list[torch.Tensor] = []
    for task_idx, (x_array, y_array) in enumerate(zip(source_x, source_y)):
        x_train, y_train, x_val, y_val = split_task(
            np.asarray(x_array, dtype=np.float32),
            np.asarray(y_array, dtype=np.float32),
            seed=seed * 1009 + task_idx * 9173 + 17,
            val_fraction=val_fraction,
        )
        train_x.append(torch.as_tensor(x_train, device=device))
        train_y.append(torch.as_tensor(y_train, device=device))
        val_x.append(torch.as_tensor(x_val, device=device))
        val_y.append(torch.as_tensor(y_val, device=device))

    source_params = np.asarray(
        [parse_params(system, path.stem) for path in source_paths],
        dtype=np.float64,
    )
    target_params = np.asarray(parse_params(system, support_paths[0].stem), dtype=np.float64)
    normalization = dataset_dict["info"]["normalization"]
    output_std = np.asarray(normalization["output"]["std"], dtype=np.float64)

    return DataBundle(
        system=system,
        dataset_name=spec.dataset_name,
        input_dim=int(dataset_dict["info"]["input_dim"]),
        output_dim=int(dataset_dict["info"]["output_dim"]),
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        target_support_x=torch.as_tensor(np.asarray(support_x[0], dtype=np.float32), device=device),
        target_support_y=torch.as_tensor(np.asarray(support_y[0], dtype=np.float32), device=device),
        target_query_x=torch.as_tensor(np.asarray(query_x[0], dtype=np.float32), device=device),
        target_query_y=torch.as_tensor(np.asarray(query_y[0], dtype=np.float32), device=device),
        source_params=source_params,
        target_params=target_params,
        output_std=output_std,
        source_files=[path.name for path in source_paths],
        target_file=support_paths[0].name,
    )


class GatedDSSDL(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_tasks: int,
        pmax: int,
        hidden_widths: Sequence[int],
        gate_init_logit: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.pmax = pmax
        self.embedding = nn.Embedding(num_tasks, pmax)
        generator = torch.Generator(device="cpu").manual_seed(seed + 7717)
        initial_logits = torch.full((pmax,), float(gate_init_logit))
        initial_logits += 0.01 * torch.randn(pmax, generator=generator)
        self.gate_logits = nn.Parameter(initial_logits)
        layers: list[nn.Module] = []
        width_in = input_dim + pmax
        for width in hidden_widths:
            layers.extend([nn.Linear(width_in, width), nn.SiLU()])
            width_in = width
        layers.append(nn.Linear(width_in, output_dim))
        self.backbone = nn.Sequential(*layers)

    def gate_probabilities(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits / max(float(temperature), 1e-4))

    def hard_mask(
        self,
        *,
        temperature: float = 1.0,
        straight_through: bool,
    ) -> torch.Tensor:
        probabilities = self.gate_probabilities(temperature)
        hard = (self.gate_logits >= 0).to(probabilities.dtype)
        if straight_through:
            return hard + probabilities - probabilities.detach()
        return hard

    def forward_task(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        temperature: float = 1.0,
        use_gate: bool = True,
        straight_through: bool = True,
        mask_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.embedding(task_ids)
        if mask_override is not None:
            mask = mask_override
        elif use_gate:
            mask = self.hard_mask(
                temperature=temperature,
                straight_through=straight_through,
            )
        else:
            mask = torch.ones(self.pmax, dtype=x.dtype, device=x.device)
        return self.backbone(torch.cat([x, latent * mask], dim=1))

    def forward_latent(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        *,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        expanded = latent.expand(x.shape[0], -1)
        return self.backbone(torch.cat([x, expanded * mask], dim=1))


class FixedDSSDL(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_tasks: int,
        latent_dim: int,
        hidden_widths: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_tasks, latent_dim)
        layers: list[nn.Module] = []
        width_in = input_dim + latent_dim
        for width in hidden_widths:
            layers.extend([nn.Linear(width_in, width), nn.SiLU()])
            width_in = width
        layers.append(nn.Linear(width_in, output_dim))
        self.backbone = nn.Sequential(*layers)

    def forward_task(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([x, self.embedding(task_ids)], dim=1))

    def forward_latent(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        expanded = latent.expand(x.shape[0], -1)
        return self.backbone(torch.cat([x, expanded], dim=1))


def balanced_epoch_data(
    bundle: DataBundle,
    *,
    samples_per_task: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = bundle.train_x[0].device
    for task_idx, (task_x, task_y) in enumerate(zip(bundle.train_x, bundle.train_y)):
        count = min(samples_per_task, len(task_x))
        indices = torch.randperm(len(task_x), generator=generator)[:count].to(device)
        xs.append(task_x[indices])
        ys.append(task_y[indices])
        task_ids.append(torch.full((count,), task_idx, dtype=torch.long, device=device))
    x_all = torch.cat(xs)
    y_all = torch.cat(ys)
    ids_all = torch.cat(task_ids)
    shuffle = torch.randperm(len(x_all), generator=generator).to(device)
    return x_all[shuffle], y_all[shuffle], ids_all[shuffle]


def iter_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    batch_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        yield x[start:end], y[start:end], task_ids[start:end]


@torch.no_grad()
def validate_gated(
    model: GatedDSSDL,
    bundle: DataBundle,
    *,
    mask: torch.Tensor | None = None,
) -> ValidationMetrics:
    model.eval()
    per_task: list[float] = []
    hard_mask = (
        model.hard_mask(straight_through=False) if mask is None else mask.to(bundle.val_x[0].device)
    )
    for task_idx, (x_val, y_val) in enumerate(zip(bundle.val_x, bundle.val_y)):
        ids = torch.full((len(x_val),), task_idx, dtype=torch.long, device=x_val.device)
        pred = model.forward_task(
            x_val,
            ids,
            mask_override=hard_mask,
            straight_through=False,
        )
        per_task.append(float(torch.sqrt(torch.mean((pred - y_val) ** 2)).item()))
    values = np.asarray(per_task, dtype=np.float64)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return ValidationMetrics(
        mean_rmse=float(np.mean(values)),
        std_rmse=std,
        se_rmse=std / math.sqrt(max(len(values), 1)),
        per_task_rmse=per_task,
    )


@torch.no_grad()
def validate_fixed(model: FixedDSSDL, bundle: DataBundle) -> ValidationMetrics:
    model.eval()
    per_task: list[float] = []
    for task_idx, (x_val, y_val) in enumerate(zip(bundle.val_x, bundle.val_y)):
        ids = torch.full((len(x_val),), task_idx, dtype=torch.long, device=x_val.device)
        pred = model.forward_task(x_val, ids)
        per_task.append(float(torch.sqrt(torch.mean((pred - y_val) ** 2)).item()))
    values = np.asarray(per_task, dtype=np.float64)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return ValidationMetrics(
        mean_rmse=float(np.mean(values)),
        std_rmse=std,
        se_rmse=std / math.sqrt(max(len(values), 1)),
        per_task_rmse=per_task,
    )


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_gated(
    bundle: DataBundle,
    args: argparse.Namespace,
) -> tuple[GatedDSSDL, dict[str, Any], list[dict[str, Any]]]:
    model = GatedDSSDL(
        input_dim=bundle.input_dim,
        output_dim=bundle.output_dim,
        num_tasks=bundle.num_tasks,
        pmax=args.pmax,
        hidden_widths=args.hidden_widths,
        gate_init_logit=args.gate_init_logit,
        seed=args.seed,
    ).to(args.device_obj)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.embedding.parameters()) + list(model.backbone.parameters()),
                "lr": args.model_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [model.gate_logits],
                "lr": args.gate_lr,
                "weight_decay": 0.0,
            },
        ]
    )
    history: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    total_prune_epochs = max(args.prune_epochs, 1)

    for epoch in range(args.warmup_epochs + args.prune_epochs):
        model.train()
        in_warmup = epoch < args.warmup_epochs
        prune_progress = (
            0.0 if in_warmup else min(1.0, (epoch - args.warmup_epochs + 1) / total_prune_epochs)
        )
        temperature = args.temperature_start + prune_progress * (
            args.temperature_end - args.temperature_start
        )
        lambda_now = args.l0_lambda * prune_progress
        x_epoch, y_epoch, ids_epoch = balanced_epoch_data(
            bundle,
            samples_per_task=args.samples_per_task,
            seed=args.seed * 1_000_003 + epoch * 97 + 31,
        )
        prediction_loss_sum = 0.0
        batch_count = 0
        for x_batch, y_batch, id_batch in iter_batches(
            x_epoch,
            y_epoch,
            ids_epoch,
            batch_size=args.batch_size,
        ):
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_task(
                x_batch,
                id_batch,
                temperature=temperature,
                use_gate=not in_warmup,
                straight_through=True,
            )
            prediction_loss = torch.mean((pred - y_batch) ** 2)
            if in_warmup:
                complexity = torch.zeros((), device=x_batch.device)
            else:
                complexity = model.gate_probabilities(temperature).sum()
            loss = prediction_loss + lambda_now * complexity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            prediction_loss_sum += float(prediction_loss.detach().item())
            batch_count += 1

        should_evaluate = (
            epoch == 0
            or epoch + 1 == args.warmup_epochs
            or epoch + 1 == args.warmup_epochs + args.prune_epochs
            or (epoch + 1) % args.eval_every == 0
        )
        if should_evaluate:
            metrics = validate_gated(model, bundle)
            probs = model.gate_probabilities(args.temperature_end).detach().cpu().numpy()
            active = int(np.sum(model.gate_logits.detach().cpu().numpy() >= 0.0))
            row = {
                "epoch": epoch,
                "phase": "warmup" if in_warmup else "prune",
                "temperature": float(temperature),
                "lambda": float(lambda_now),
                "train_prediction_mse": prediction_loss_sum / max(batch_count, 1),
                "source_val_mean_rmse": metrics.mean_rmse,
                "source_val_std_rmse": metrics.std_rmse,
                "source_val_se_rmse": metrics.se_rmse,
                "active_dim": active,
                "gate_probabilities": probs.tolist(),
                "gate_logits": model.gate_logits.detach().cpu().tolist(),
            }
            history.append(row)
            if not in_warmup and active > 0:
                snapshots.append({"row": copy.deepcopy(row), "state": cpu_state_dict(model)})
            print(
                f"[{bundle.system}] epoch={epoch:03d} phase={row['phase']} "
                f"active={active}/{args.pmax} val={metrics.mean_rmse:.6f} "
                f"lambda={lambda_now:.3g}",
                flush=True,
            )

    if not snapshots:
        raise RuntimeError("No nonzero gated checkpoint was produced")
    best_prediction = min(
        snapshots,
        key=lambda item: item["row"]["source_val_mean_rmse"],
    )
    cutoff = (
        best_prediction["row"]["source_val_mean_rmse"]
        + args.one_se_multiplier * best_prediction["row"]["source_val_se_rmse"]
    )
    eligible = [item for item in snapshots if item["row"]["source_val_mean_rmse"] <= cutoff]
    selected = min(
        eligible,
        key=lambda item: (
            item["row"]["active_dim"],
            item["row"]["source_val_mean_rmse"],
            -item["row"]["epoch"],
        ),
    )
    model.load_state_dict(selected["state"])
    frozen_mask = model.hard_mask(straight_through=False).detach()
    model.gate_logits.requires_grad_(False)
    stabilize_optimizer = torch.optim.AdamW(
        list(model.embedding.parameters()) + list(model.backbone.parameters()),
        lr=args.stabilize_lr,
        weight_decay=args.weight_decay,
    )
    for stabilize_epoch in range(args.stabilize_epochs):
        model.train()
        x_epoch, y_epoch, ids_epoch = balanced_epoch_data(
            bundle,
            samples_per_task=args.samples_per_task,
            seed=args.seed * 1_000_003 + stabilize_epoch * 193 + 900_001,
        )
        for x_batch, y_batch, id_batch in iter_batches(
            x_epoch,
            y_epoch,
            ids_epoch,
            batch_size=args.batch_size,
        ):
            stabilize_optimizer.zero_grad(set_to_none=True)
            pred = model.forward_task(
                x_batch,
                id_batch,
                mask_override=frozen_mask,
                straight_through=False,
            )
            loss = torch.mean((pred - y_batch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            stabilize_optimizer.step()

    final_metrics = validate_gated(model, bundle, mask=frozen_mask)
    selection = {
        "best_prediction_checkpoint": best_prediction["row"],
        "one_se_cutoff": float(cutoff),
        "eligible_checkpoint_count": len(eligible),
        "selected_checkpoint": selected["row"],
        "final_source_validation": asdict(final_metrics),
        "frozen_mask": frozen_mask.detach().cpu().tolist(),
    }
    return model, selection, history


def train_fixed(
    bundle: DataBundle,
    args: argparse.Namespace,
) -> tuple[FixedDSSDL, dict[str, Any], list[dict[str, Any]]]:
    model = FixedDSSDL(
        input_dim=bundle.input_dim,
        output_dim=bundle.output_dim,
        num_tasks=bundle.num_tasks,
        latent_dim=args.fixed_dim,
        hidden_widths=args.hidden_widths,
    ).to(args.device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.model_lr,
        weight_decay=args.weight_decay,
    )
    epochs = args.warmup_epochs + args.prune_epochs + args.stabilize_epochs
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_row: dict[str, Any] | None = None
    for epoch in range(epochs):
        model.train()
        x_epoch, y_epoch, ids_epoch = balanced_epoch_data(
            bundle,
            samples_per_task=args.samples_per_task,
            seed=args.seed * 1_000_003 + epoch * 97 + 31,
        )
        prediction_loss_sum = 0.0
        batch_count = 0
        for x_batch, y_batch, id_batch in iter_batches(
            x_epoch,
            y_epoch,
            ids_epoch,
            batch_size=args.batch_size,
        ):
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_task(x_batch, id_batch)
            loss = torch.mean((pred - y_batch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            prediction_loss_sum += float(loss.detach().item())
            batch_count += 1
        should_evaluate = epoch == 0 or epoch + 1 == epochs or (epoch + 1) % args.eval_every == 0
        if should_evaluate:
            metrics = validate_fixed(model, bundle)
            row = {
                "epoch": epoch,
                "phase": "fixed",
                "train_prediction_mse": prediction_loss_sum / max(batch_count, 1),
                "source_val_mean_rmse": metrics.mean_rmse,
                "source_val_std_rmse": metrics.std_rmse,
                "source_val_se_rmse": metrics.se_rmse,
                "active_dim": args.fixed_dim,
            }
            history.append(row)
            if best_row is None or row["source_val_mean_rmse"] < best_row["source_val_mean_rmse"]:
                best_row = copy.deepcopy(row)
                best_state = cpu_state_dict(model)
            print(
                f"[{bundle.system}] fixed-p={args.fixed_dim} epoch={epoch:03d} "
                f"val={metrics.mean_rmse:.6f}",
                flush=True,
            )
    if best_state is None or best_row is None:
        raise RuntimeError("Fixed-p training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_metrics = validate_fixed(model, bundle)
    selection = {
        "selected_checkpoint": best_row,
        "final_source_validation": asdict(final_metrics),
    }
    return model, selection, history


def adapt_target_gated(
    model: GatedDSSDL,
    bundle: DataBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mask = model.hard_mask(straight_through=False).detach()
    initial = model.embedding.weight.detach().mean(dim=0)
    latent = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([latent], lr=args.adapt_lr)
    best_support = float("inf")
    best_latent = latent.detach().clone()
    patience = 0
    trace: list[dict[str, float]] = []
    for step in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_latent(bundle.target_support_x, latent, mask=mask)
        loss = torch.mean((pred - bundle.target_support_y) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().item())
        if value < best_support - args.adapt_tolerance:
            best_support = value
            best_latent = latent.detach().clone()
            patience = 0
        else:
            patience += 1
        if step == 0 or (step + 1) % 25 == 0:
            trace.append({"step": step, "support_mse": value})
        if patience >= args.adapt_patience:
            break
    latent.data.copy_(best_latent)
    with torch.no_grad():
        support_pred = model.forward_latent(bundle.target_support_x, latent, mask=mask)
        query_pred = model.forward_latent(bundle.target_query_x, latent, mask=mask)
    return target_metrics(
        support_pred,
        bundle.target_support_y,
        query_pred,
        bundle.target_query_y,
        bundle.output_std,
        latent,
        mask,
        trace,
        step + 1,
    )


def adapt_target_fixed(
    model: FixedDSSDL,
    bundle: DataBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initial = model.embedding.weight.detach().mean(dim=0)
    latent = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([latent], lr=args.adapt_lr)
    best_support = float("inf")
    best_latent = latent.detach().clone()
    patience = 0
    trace: list[dict[str, float]] = []
    for step in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_latent(bundle.target_support_x, latent)
        loss = torch.mean((pred - bundle.target_support_y) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().item())
        if value < best_support - args.adapt_tolerance:
            best_support = value
            best_latent = latent.detach().clone()
            patience = 0
        else:
            patience += 1
        if step == 0 or (step + 1) % 25 == 0:
            trace.append({"step": step, "support_mse": value})
        if patience >= args.adapt_patience:
            break
    latent.data.copy_(best_latent)
    with torch.no_grad():
        support_pred = model.forward_latent(bundle.target_support_x, latent)
        query_pred = model.forward_latent(bundle.target_query_x, latent)
    mask = torch.ones_like(latent)
    return target_metrics(
        support_pred,
        bundle.target_support_y,
        query_pred,
        bundle.target_query_y,
        bundle.output_std,
        latent,
        mask,
        trace,
        step + 1,
    )


def target_metrics(
    support_pred: torch.Tensor,
    support_y: torch.Tensor,
    query_pred: torch.Tensor,
    query_y: torch.Tensor,
    output_std: np.ndarray,
    latent: torch.Tensor,
    mask: torch.Tensor,
    trace: list[dict[str, float]],
    steps_completed: int,
) -> dict[str, Any]:
    support_residual = support_pred - support_y
    query_residual = query_pred - query_y
    support_rmse = float(torch.sqrt(torch.mean(support_residual**2)).item())
    query_rmse = float(torch.sqrt(torch.mean(query_residual**2)).item())
    std = torch.as_tensor(output_std, dtype=query_residual.dtype, device=query_residual.device)
    query_raw_rmse = float(torch.sqrt(torch.mean((query_residual * std) ** 2)).item())
    per_output = torch.sqrt(torch.mean(query_residual**2, dim=0)).detach().cpu().tolist()
    return {
        "support_rmse_norm": support_rmse,
        "query_rmse_norm": query_rmse,
        "query_rmse_raw": query_raw_rmse,
        "query_per_output_rmse_norm": per_output,
        "adapted_latent": latent.detach().cpu().tolist(),
        "masked_adapted_latent": (latent.detach() * mask).cpu().tolist(),
        "adapt_steps_completed": steps_completed,
        "adapt_trace": trace,
    }


def linear_decoder_metrics(
    latent: np.ndarray,
    physical: np.ndarray,
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    if latent.ndim != 2 or physical.ndim != 2 or len(latent) != len(physical):
        raise ValueError("Invalid latent/physical arrays")
    centered = latent - latent.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.linalg.matrix_rank(centered))
    design = np.column_stack([latent, np.ones(len(latent))])
    coef, *_ = np.linalg.lstsq(design, physical, rcond=None)
    pred = design @ coef
    total = np.sum((physical - physical.mean(axis=0)) ** 2, axis=0)
    residual = np.sum((physical - pred) ** 2, axis=0)
    in_sample_r2 = 1.0 - residual / np.maximum(total, 1e-12)

    loo_pred = np.zeros_like(physical)
    for held_out in range(len(latent)):
        keep = np.arange(len(latent)) != held_out
        train_design = np.column_stack([latent[keep], np.ones(np.sum(keep))])
        loo_coef, *_ = np.linalg.lstsq(train_design, physical[keep], rcond=None)
        loo_pred[held_out] = np.append(latent[held_out], 1.0) @ loo_coef
    loo_residual = np.sum((physical - loo_pred) ** 2, axis=0)
    loo_r2 = 1.0 - loo_residual / np.maximum(total, 1e-12)
    loo_rmse = np.sqrt(np.mean((physical - loo_pred) ** 2, axis=0))
    return {
        "latent_shape": list(latent.shape),
        "centered_rank": rank,
        "singular_values": singular_values.tolist(),
        "in_sample_r2": {name: float(value) for name, value in zip(parameter_names, in_sample_r2)},
        "in_sample_r2_mean": float(np.mean(in_sample_r2)),
        "loo_r2": {name: float(value) for name, value in zip(parameter_names, loo_r2)},
        "loo_r2_mean": float(np.mean(loo_r2)),
        "loo_rmse": {name: float(value) for name, value in zip(parameter_names, loo_rmse)},
    }


def gate_diagnostics(model: GatedDSSDL, thresholds: Sequence[float]) -> dict[str, Any]:
    probabilities = model.gate_probabilities(temperature=1.0).detach().cpu().numpy()
    logits = model.gate_logits.detach().cpu().numpy()
    return {
        "gate_probabilities_temperature_1": probabilities.tolist(),
        "gate_logits": logits.tolist(),
        "active_dim_at_logit_zero": int(np.sum(logits >= 0)),
        "active_dim_by_probability_threshold": {
            f"{threshold:.3f}": int(np.sum(probabilities >= threshold)) for threshold in thresholds
        },
        "minimum_probability_margin_from_0p5": float(np.min(np.abs(probabilities - 0.5))),
        "mean_probability_margin_from_0p5": float(np.mean(np.abs(probabilities - 0.5))),
    }


def run_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root) / "runs" / args.kind / args.system
    if args.kind == "gated":
        root = root / f"pmax_{args.pmax:02d}" / f"lambda_{float_slug(args.l0_lambda)}"
    else:
        root = root / f"p_{args.fixed_dim:02d}"
    root = root / f"seed_{args.seed:03d}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_keys = [
        key for key, value in rows[0].items() if not isinstance(value, (list, tuple, dict))
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output_dir = run_dir(args)
    start = time.time()
    bundle = load_data(
        args.system,
        seed=args.seed,
        val_fraction=args.val_fraction,
        device=args.device_obj,
    )
    if args.kind == "gated":
        model, selection, history = train_gated(bundle, args)
        target = adapt_target_gated(model, bundle, args)
        hard_mask = model.hard_mask(straight_through=False).detach().cpu().numpy().astype(bool)
        source_latent_full = model.embedding.weight.detach().cpu().numpy()
        source_latent = source_latent_full[:, hard_mask]
        diagnostics = gate_diagnostics(model, args.thresholds)
        active_dim = int(np.sum(hard_mask))
        model_config = {
            "pmax": args.pmax,
            "l0_lambda": args.l0_lambda,
            "gate_lr": args.gate_lr,
            "gate_init_logit": args.gate_init_logit,
            "temperature_start": args.temperature_start,
            "temperature_end": args.temperature_end,
        }
    else:
        model, selection, history = train_fixed(bundle, args)
        target = adapt_target_fixed(model, bundle, args)
        source_latent = model.embedding.weight.detach().cpu().numpy()
        diagnostics = {}
        active_dim = args.fixed_dim
        model_config = {"fixed_dim": args.fixed_dim}

    decoder = linear_decoder_metrics(
        source_latent,
        bundle.source_params,
        SPECS[args.system].parameter_names,
    )
    summary = {
        "experiment_id": "dssdl_dimension_discovery_claim_v1",
        "kind": args.kind,
        "system": args.system,
        "dataset_name": bundle.dataset_name,
        "seed": args.seed,
        "true_physical_dimension": SPECS[args.system].true_dim,
        "selected_active_dimension": active_dim,
        "exact_dimension_recovery": active_dim == SPECS[args.system].true_dim,
        "device": str(args.device_obj),
        "model_config": model_config,
        "training_config": {
            "hidden_widths": args.hidden_widths,
            "warmup_epochs": args.warmup_epochs,
            "prune_epochs": args.prune_epochs,
            "stabilize_epochs": args.stabilize_epochs,
            "samples_per_task": args.samples_per_task,
            "batch_size": args.batch_size,
            "model_lr": args.model_lr,
            "stabilize_lr": args.stabilize_lr,
            "weight_decay": args.weight_decay,
            "val_fraction": args.val_fraction,
            "one_se_multiplier": args.one_se_multiplier,
        },
        "adaptation_config": {
            "adapt_steps": args.adapt_steps,
            "adapt_lr": args.adapt_lr,
            "adapt_patience": args.adapt_patience,
            "adapt_tolerance": args.adapt_tolerance,
        },
        "selection": selection,
        "target_adaptation": target,
        "gate_diagnostics": diagnostics,
        "latent_to_physical_decoder": decoder,
        "source_parameter_names": list(SPECS[args.system].parameter_names),
        "source_parameters": bundle.source_params,
        "target_parameters": bundle.target_params,
        "source_files": bundle.source_files,
        "target_file": bundle.target_file,
        "output_dir": output_dir,
        "wall_seconds": time.time() - start,
        "claim_protocol": {
            "uses_true_dimension_for_training_or_selection": False,
            "selection_rule": (
                "sparsest nonzero gate within one standard error of the best "
                "source-validation checkpoint"
                if args.kind == "gated"
                else "lowest source-validation RMSE checkpoint"
            ),
            "target_query_used_for_selection": False,
        },
    }
    torch.save(
        {
            "state_dict": cpu_state_dict(model),
            "summary_without_history": json_safe(summary),
        },
        output_dir / "checkpoint.pt",
    )
    write_history(output_dir / "training_history.csv", history)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def collect_summaries(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "runs").rglob("summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows.append(
            {
                "kind": data["kind"],
                "system": data["system"],
                "seed": data["seed"],
                "pmax": data["model_config"].get("pmax", ""),
                "l0_lambda": data["model_config"].get("l0_lambda", ""),
                "fixed_dim": data["model_config"].get("fixed_dim", ""),
                "true_dim": data["true_physical_dimension"],
                "selected_dim": data["selected_active_dimension"],
                "exact_recovery": data["exact_dimension_recovery"],
                "source_val_rmse": data["selection"]["final_source_validation"]["mean_rmse"],
                "target_query_rmse_norm": data["target_adaptation"]["query_rmse_norm"],
                "target_query_rmse_raw": data["target_adaptation"]["query_rmse_raw"],
                "decoder_loo_r2_mean": data["latent_to_physical_decoder"]["loo_r2_mean"],
                "mask_margin": data.get("gate_diagnostics", {}).get(
                    "minimum_probability_margin_from_0p5", ""
                ),
                "wall_seconds": data["wall_seconds"],
                "summary_path": str(path),
            }
        )
    return rows


def aggregate(output_root: Path) -> tuple[Path, Path]:
    rows = collect_summaries(output_root)
    detail_path = output_root / "aggregate_runs.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with detail_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["kind"],
            row["system"],
            row["pmax"],
            row["l0_lambda"],
            row["fixed_dim"],
        )
        grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        selected = np.asarray([float(row["selected_dim"]) for row in group])
        source_val = np.asarray([float(row["source_val_rmse"]) for row in group])
        query = np.asarray([float(row["target_query_rmse_norm"]) for row in group])
        exact = np.asarray([bool(row["exact_recovery"]) for row in group], dtype=float)
        aggregate_rows.append(
            {
                "kind": key[0],
                "system": key[1],
                "pmax": key[2],
                "l0_lambda": key[3],
                "fixed_dim": key[4],
                "n": len(group),
                "selected_dim_mean": float(np.mean(selected)),
                "selected_dim_std": float(np.std(selected, ddof=1)) if len(group) > 1 else 0.0,
                "exact_recovery_rate": float(np.mean(exact)),
                "source_val_rmse_mean": float(np.mean(source_val)),
                "source_val_rmse_std": (
                    float(np.std(source_val, ddof=1)) if len(group) > 1 else 0.0
                ),
                "target_query_rmse_mean": float(np.mean(query)),
                "target_query_rmse_std": float(np.std(query, ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    aggregate_path = output_root / "aggregate_groups.csv"
    if aggregate_rows:
        with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    return detail_path, aggregate_path


def suite_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    systems = [part.strip() for part in args.systems.split(",") if part.strip()]
    seeds = parse_int_list(args.seeds)
    executable = sys.executable
    script = str(Path(__file__).resolve())

    def base(system: str, seed: int) -> list[str]:
        return [
            executable,
            script,
            "--system",
            system,
            "--seed",
            str(seed),
            "--output-root",
            str(args.output_root),
            "--device",
            args.device,
            "--hidden-widths",
            ",".join(str(value) for value in args.hidden_widths),
            "--gate-init-logit",
            str(args.gate_init_logit),
            "--model-lr",
            str(args.model_lr),
            "--gate-lr",
            str(args.gate_lr),
            "--stabilize-lr",
            str(args.stabilize_lr),
            "--weight-decay",
            str(args.weight_decay),
            "--temperature-start",
            str(args.temperature_start),
            "--temperature-end",
            str(args.temperature_end),
            "--warmup-epochs",
            str(args.warmup_epochs),
            "--prune-epochs",
            str(args.prune_epochs),
            "--stabilize-epochs",
            str(args.stabilize_epochs),
            "--samples-per-task",
            str(args.samples_per_task),
            "--batch-size",
            str(args.batch_size),
            "--eval-every",
            str(args.eval_every),
            "--val-fraction",
            str(args.val_fraction),
            "--one-se-multiplier",
            str(args.one_se_multiplier),
            "--grad-clip",
            str(args.grad_clip),
            "--adapt-steps",
            str(args.adapt_steps),
            "--adapt-lr",
            str(args.adapt_lr),
            "--adapt-patience",
            str(args.adapt_patience),
            "--adapt-tolerance",
            str(args.adapt_tolerance),
            "--thresholds",
            ",".join(str(value) for value in args.thresholds),
        ]

    if args.suite in {"main", "all"}:
        for system in systems:
            for seed in seeds:
                commands.append(
                    base(system, seed)
                    + [
                        "--kind",
                        "gated",
                        "--pmax",
                        str(args.pmax),
                        "--l0-lambda",
                        str(args.l0_lambda),
                    ]
                )
    if args.suite in {"lambda", "all"}:
        for system in systems:
            for value in parse_float_list(args.lambda_grid):
                for seed in seeds:
                    commands.append(
                        base(system, seed)
                        + [
                            "--kind",
                            "gated",
                            "--pmax",
                            str(args.pmax),
                            "--l0-lambda",
                            str(value),
                        ]
                    )
    if args.suite in {"pmax", "all"}:
        for system in systems:
            for pmax in parse_int_list(args.pmax_grid):
                for seed in seeds:
                    commands.append(
                        base(system, seed)
                        + [
                            "--kind",
                            "gated",
                            "--pmax",
                            str(pmax),
                            "--l0-lambda",
                            str(args.l0_lambda),
                        ]
                    )
    if args.suite in {"fixed", "all"}:
        for system in systems:
            for dim in parse_int_list(args.fixed_dims):
                for seed in seeds:
                    commands.append(
                        base(system, seed) + ["--kind", "fixed", "--fixed-dim", str(dim)]
                    )
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for command in commands:
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            unique.append(command)
    return unique


def run_suite(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = suite_commands(args)
    print(f"Prepared {len(commands)} runs for suite={args.suite}", flush=True)
    for index, command in enumerate(commands, start=1):
        kind = command[command.index("--kind") + 1]
        system = command[command.index("--system") + 1]
        seed = command[command.index("--seed") + 1]
        if kind == "gated":
            descriptor = (
                f"pmax{command[command.index('--pmax') + 1]}_"
                f"lambda{float_slug(float(command[command.index('--l0-lambda') + 1]))}"
            )
        else:
            descriptor = f"p{command[command.index('--fixed-dim') + 1]}"
        log_path = log_dir / f"{kind}_{system}_{descriptor}_seed{seed}.log"
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        if result.returncode != 0:
            raise RuntimeError(f"Run failed with exit={result.returncode}; inspect {log_path}")
        aggregate(output_root)
    detail, groups = aggregate(output_root)
    print(f"Wrote {detail}", flush=True)
    print(f"Wrote {groups}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("gated", "fixed"), default="gated")
    parser.add_argument("--system", choices=tuple(SPECS), default="pendulum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pmax", type=int, default=8)
    parser.add_argument("--fixed-dim", type=int, default=3)
    parser.add_argument("--l0-lambda", type=float, default=1e-4)
    parser.add_argument("--hidden-widths", default="64,64")
    parser.add_argument("--gate-init-logit", type=float, default=1.5)
    parser.add_argument("--model-lr", type=float, default=3e-3)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--stabilize-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature-start", type=float, default=1.5)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--prune-epochs", type=int, default=150)
    parser.add_argument("--stabilize-epochs", type=int, default=70)
    parser.add_argument("--samples-per-task", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--one-se-multiplier", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--adapt-steps", type=int, default=500)
    parser.add_argument("--adapt-lr", type=float, default=3e-2)
    parser.add_argument("--adapt-patience", type=int, default=100)
    parser.add_argument("--adapt-tolerance", type=float, default=1e-10)
    parser.add_argument("--thresholds", default="0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--suite", choices=("main", "lambda", "pmax", "fixed", "all"))
    parser.add_argument("--systems", default="pendulum,quadrotor,lorenz,fhn")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--lambda-grid", default="0,1e-5,3e-5,1e-4,3e-4,1e-3")
    parser.add_argument("--pmax-grid", default="6,8,12")
    parser.add_argument("--fixed-dims", default="1-8")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.hidden_widths = [int(value) for value in args.hidden_widths.split(",") if value]
    args.thresholds = parse_float_list(args.thresholds)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    args.device_obj = torch.device(args.device)
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.aggregate:
        detail, groups = aggregate(Path(args.output_root))
        print(f"Wrote {detail}")
        print(f"Wrote {groups}")
        return
    if args.suite:
        run_suite(args)
        return
    run_one(args)


if __name__ == "__main__":
    main()
