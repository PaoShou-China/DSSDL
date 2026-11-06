import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
import torch
import torch.nn.functional as F

from datasets import CURRENT_SYSTEM

DEFAULT_CONFIG = {
    "seed": 0,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "use_raw_data": False,
    "num_context_params": 8,
    "hidden_sizes": [64, 64],
    "meta_lr": 1e-3,
    "inner_lr": 1e-2,
    "num_inner_updates": 5,
    "tasks_per_metaupdate": 4,
    "k_support": 10,
    "k_query": 10,
    "max_iters": 1000,
    "log_interval": 50,
    "first_order": True,
    "grad_clip": 10.0,
    "model_path": "model/cavia_pretrain.pt",
    "finetune_max_epochs": 1000,
    "finetune_patience": 50,
    "finetune_convergence_threshold": 1e-10,
    "finetune_use_query": True,
}


def _ensure_cavia_import_path():
    cavia_path = os.path.join(os.path.dirname(__file__), "cavia-master", "regression")
    if cavia_path not in sys.path:
        sys.path.append(cavia_path)


def _merge_config(overrides: Dict) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if overrides:
        cfg.update(overrides)
    return cfg


def _set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_split_data(dataset_dict: Dict, split: str, use_raw_data: bool):
    if use_raw_data:
        return dataset_dict["raw"][split]
    return dataset_dict["normalized"][split]


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _sample_support_query(
    x: torch.Tensor,
    y: torch.Tensor,
    k_support: int,
    k_query: int,
    rng: np.random.Generator,
):
    n = x.shape[0]
    if n <= 1:
        return x, y, x, y
    if k_support + k_query > n:
        k_support = min(k_support, n - 1)
        k_query = n - k_support
    indices = rng.choice(n, size=k_support + k_query, replace=False)
    support_idx = indices[:k_support]
    query_idx = indices[k_support:]
    return x[support_idx], y[support_idx], x[query_idx], y[query_idx]


def _build_model(input_dim: int, output_dim: int, cfg: Dict):
    _ensure_cavia_import_path()
    from cavia_model import CaviaModel

    model = CaviaModel(
        n_in=input_dim,
        n_out=output_dim,
        num_context_params=cfg["num_context_params"],
        n_hidden=cfg["hidden_sizes"],
        device=cfg["device"],
    )
    return model.to(cfg["device"])


def meta_train(dataset_dict: Dict, config_overrides: Dict = None):
    cfg = _merge_config(config_overrides)
    _set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg["device"]

    x_list, y_list = _get_split_data(dataset_dict, "train", cfg["use_raw_data"])
    n_tasks = len(x_list)
    if n_tasks == 0:
        raise ValueError("Empty train split in dataset_dict.")

    input_dim = dataset_dict["info"]["input_dim"]
    output_dim = dataset_dict["info"]["output_dim"]
    model = _build_model(input_dim, output_dim, cfg)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["meta_lr"])
    log = {"train_loss": []}
    time_log = []

    start = time.time()
    start_time = time.time()
    for i_iter in range(cfg["max_iters"]):
        task_count = min(cfg["tasks_per_metaupdate"], n_tasks)
        replace = cfg["tasks_per_metaupdate"] > n_tasks
        task_indices = rng.choice(n_tasks, size=task_count, replace=replace)

        meta_grads = [torch.zeros_like(p) for p in model.parameters()]
        task_losses = []

        for idx in task_indices:
            x = _to_tensor(x_list[idx], device)
            y = _to_tensor(y_list[idx], device)
            support_x, support_y, query_x, query_y = _sample_support_query(
                x,
                y,
                cfg["k_support"],
                cfg["k_query"],
                rng,
            )

            model.reset_context_params()
            for _ in range(cfg["num_inner_updates"]):
                pred = model(support_x)
                loss = F.mse_loss(pred, support_y)
                grads = torch.autograd.grad(
                    loss,
                    model.context_params,
                    create_graph=not cfg["first_order"],
                )[0]
                if cfg["first_order"]:
                    grads = grads.detach()
                model.context_params = model.context_params - cfg["inner_lr"] * grads

            pred_q = model(query_x)
            loss_q = F.mse_loss(pred_q, query_y)
            task_losses.append(loss_q.item())
            task_grads = torch.autograd.grad(loss_q, model.parameters())

            for i, grad in enumerate(task_grads):
                meta_grads[i] += grad.detach()

        for param, grad in zip(model.parameters(), meta_grads):
            grad = grad / task_count
            if cfg["grad_clip"] is not None:
                grad = grad.clamp(-cfg["grad_clip"], cfg["grad_clip"])
            param.grad = grad

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.reset_context_params()

        elapsed_seconds = time.time() - start_time
        avg_loss = float(np.mean(task_losses)) if task_losses else float("nan")
        time_log.append(
            {
                "epoch": i_iter + 1,
                "loss": avg_loss,
                "elapsed_seconds": elapsed_seconds,
            }
        )

        if i_iter == 0 or (i_iter + 1) % cfg["log_interval"] == 0:
            log["train_loss"].append((i_iter + 1, avg_loss))
            elapsed = time.time() - start
            print(f"iter {i_iter + 1}/{cfg['max_iters']} loss {avg_loss:.6e} time {elapsed:.2f}s")
            start = time.time()

    model_path = cfg["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    export_epoch_time_csv(CURRENT_SYSTEM, "pretrain", time_log)
    return model_path, log


def _adapt_context(
    model: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    cfg: Dict,
    time_log: Optional[List[Dict]] = None,
    epoch_counter: Optional[Dict[str, int]] = None,
    start_time: Optional[float] = None,
) -> torch.Tensor:
    model.reset_context_params()
    if start_time is None:
        start_time = time.time()
    for _ in range(cfg["num_inner_updates"]):
        pred = model(support_x)
        loss = F.mse_loss(pred, support_y)
        grads = torch.autograd.grad(loss, model.context_params, create_graph=False)[0]
        model.context_params = model.context_params - cfg["inner_lr"] * grads
        if time_log is not None:
            epoch_counter = epoch_counter or {"value": 0}
            epoch_counter["value"] += 1
            time_log.append(
                {
                    "epoch": epoch_counter["value"],
                    "train_loss": loss.item(),
                    "elapsed_seconds": time.time() - start_time,
                }
            )
    return model.context_params.detach()


def _adapt_context_with_early_stopping(
    model: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    cfg: Dict,
    time_log: Optional[List[Dict]] = None,
    epoch_counter: Optional[Dict[str, int]] = None,
    start_time: Optional[float] = None,
) -> torch.Tensor:
    model.reset_context_params()

    best_loss = float("inf")
    best_ctx = model.context_params.detach().clone()
    patience_counter = 0

    max_epochs = int(cfg["finetune_max_epochs"])
    patience = int(cfg["finetune_patience"])
    threshold = float(cfg["finetune_convergence_threshold"])
    if start_time is None:
        start_time = time.time()

    for epoch in range(max_epochs):
        pred = model(support_x)
        mse_loss = F.mse_loss(pred, support_y)
        grads = torch.autograd.grad(mse_loss, model.context_params, create_graph=False)[0]
        model.context_params = model.context_params - cfg["inner_lr"] * grads

        with torch.no_grad():
            val_pred = model(query_x)
            val_mse = F.mse_loss(val_pred, query_y).item()

        train_rmse = float(np.sqrt(mse_loss.item()))
        val_rmse = float(np.sqrt(val_mse))
        print(
            f"[finetune] epoch {epoch + 1}/{max_epochs} "
            f"train_rmse {train_rmse:.6e} val_rmse {val_rmse:.6e}"
        )
        if time_log is not None:
            epoch_counter = epoch_counter or {"value": 0}
            epoch_counter["value"] += 1
            time_log.append(
                {
                    "epoch": epoch_counter["value"],
                    "train_loss": mse_loss.item(),
                    "test_loss": val_mse,
                    "elapsed_seconds": time.time() - start_time,
                }
            )

        improvement = best_loss - val_mse
        if improvement > threshold:
            best_loss = val_mse
            best_ctx = model.context_params.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_ctx


def finetune_contexts(dataset_dict: Dict, config_overrides: Dict = None):
    cfg = _merge_config(config_overrides)
    _set_seed(cfg["seed"])
    device = cfg["device"]

    input_dim = dataset_dict["info"]["input_dim"]
    output_dim = dataset_dict["info"]["output_dim"]
    model = _build_model(input_dim, output_dim, cfg)

    state_dict = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    support_x_list, support_y_list = _get_split_data(
        dataset_dict,
        "support",
        cfg["use_raw_data"],
    )
    query_x_list, query_y_list = _get_split_data(
        dataset_dict,
        "query",
        cfg["use_raw_data"],
    )

    if query_x_list:
        losses = []
        with torch.no_grad():
            for x_arr, y_arr in zip(query_x_list, query_y_list):
                query_x = _to_tensor(x_arr, device)
                query_y = _to_tensor(y_arr, device)
                model.reset_context_params()
                pred = model(query_x)
                loss = F.mse_loss(pred, query_y).item()
                losses.append(loss)
        mse = float(np.mean(losses)) if losses else float("nan")
        rmse = float(np.sqrt(mse)) if losses else float("nan")
        print(f"No-finetune query RMSE: {rmse:.6e}")

    contexts = []
    time_log: List[Dict] = []
    epoch_counter = {"value": 0}
    finetune_start_time = time.time()
    for idx, (x_arr, y_arr) in enumerate(zip(support_x_list, support_y_list)):
        support_x = _to_tensor(x_arr, device)
        support_y = _to_tensor(y_arr, device)

        if cfg["finetune_use_query"] and idx < len(query_x_list):
            query_x = _to_tensor(query_x_list[idx], device)
            query_y = _to_tensor(query_y_list[idx], device)
            if query_x.numel() > 0:
                ctx = _adapt_context_with_early_stopping(
                    model,
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    cfg,
                    time_log=time_log,
                    epoch_counter=epoch_counter,
                    start_time=finetune_start_time,
                )
            else:
                ctx = _adapt_context(
                    model,
                    support_x,
                    support_y,
                    cfg,
                    time_log=time_log,
                    epoch_counter=epoch_counter,
                    start_time=finetune_start_time,
                )
        else:
            ctx = _adapt_context(
                model,
                support_x,
                support_y,
                cfg,
                time_log=time_log,
                epoch_counter=epoch_counter,
                start_time=finetune_start_time,
            )

        contexts.append(ctx.cpu())

    export_epoch_time_csv(CURRENT_SYSTEM, "finetune", time_log)
    return contexts


def evaluate_query(
    dataset_dict: Dict,
    contexts: List[torch.Tensor],
    config_overrides: Dict = None,
):
    cfg = _merge_config(config_overrides)
    _set_seed(cfg["seed"])
    device = cfg["device"]

    input_dim = dataset_dict["info"]["input_dim"]
    output_dim = dataset_dict["info"]["output_dim"]
    model = _build_model(input_dim, output_dim, cfg)

    state_dict = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    query_x_list, query_y_list = _get_split_data(
        dataset_dict,
        "query",
        cfg["use_raw_data"],
    )

    task_count = min(len(query_x_list), len(contexts))
    if task_count == 0:
        raise ValueError("No tasks available for evaluation.")

    losses = []
    with torch.no_grad():
        for i in range(task_count):
            query_x = _to_tensor(query_x_list[i], device)
            query_y = _to_tensor(query_y_list[i], device)

            context = contexts[i].to(device)
            context.requires_grad = False
            model.context_params = context

            pred = model(query_x)
            loss = F.mse_loss(pred, query_y)
            losses.append(loss.item())

    mse = float(np.mean(losses)) if losses else float("nan")
    rmse = float(np.sqrt(mse)) if losses else float("nan")
    return mse, rmse


def _select_loss_for_export(log_entry: Dict):
    if "test_loss" in log_entry and log_entry["test_loss"] is not None:
        return log_entry["test_loss"]
    if "loss" in log_entry and log_entry["loss"] is not None:
        return log_entry["loss"]
    if "train_loss" in log_entry and log_entry["train_loss"] is not None:
        return log_entry["train_loss"]
    return None


def export_epoch_time_csv(system_name: str, phase: str, training_log: List[Dict]):
    os.makedirs("final_data", exist_ok=True)
    csv_path = os.path.join("final_data", f"{system_name}_{phase}_time.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "elapsed_seconds"])
        for entry in training_log:
            writer.writerow(
                [
                    entry.get("epoch"),
                    _select_loss_for_export(entry),
                    entry.get("elapsed_seconds"),
                ]
            )
    print(f"Epoch timing CSV saved to: {csv_path}")
