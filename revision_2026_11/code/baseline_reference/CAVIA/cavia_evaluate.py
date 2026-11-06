import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

import datasets
from datasets import CURRENT_SYSTEM, dataset_dict, dt, f, slow_params_eval
from datasets.utils import get_file_list
from cavia_workflow import finetune_contexts

import evaluate as dssdl_eval


CONFIG_OVERRIDES = {
    "num_context_params": 3,
    "hidden_sizes": [64, 64],
    "inner_lr": 1e-2,
    "num_inner_updates": 5,
    "use_raw_data": False,
    "model_path": "model/cavia_pretrain.pt",
}


def _ensure_cavia_import_path():
    cavia_path = os.path.join(os.path.dirname(__file__), "cavia-master", "regression")
    if cavia_path not in sys.path:
        sys.path.append(cavia_path)


def _build_cavia_model(input_dim: int, output_dim: int, device: str, cfg: dict):
    _ensure_cavia_import_path()
    from cavia_model import CaviaModel

    model = CaviaModel(
        n_in=input_dim,
        n_out=output_dim,
        num_context_params=cfg["num_context_params"],
        n_hidden=cfg["hidden_sizes"],
        device=device,
    )
    return model.to(device)


def _parse_param_list_from_filename(path: str) -> Optional[List[float]]:
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    values = []
    try:
        for part in parts:
            values.append(float(part))
    except ValueError:
        return None
    return values


def _normalize_params(params: Sequence[float]) -> List[float]:
    return [float(x) for x in params]


def _find_context_index(
    support_feature_paths: List[str],
    target_params: Sequence[float],
    atol: float = 1e-6,
) -> Optional[int]:
    target = _normalize_params(target_params)
    for idx, path in enumerate(support_feature_paths):
        values = _parse_param_list_from_filename(path)
        if values is None:
            continue
        if len(values) != len(target):
            continue
        if np.allclose(values, target, atol=atol, rtol=0.0):
            return idx
    return None


def _select_context_for_eval(contexts: List[torch.Tensor]) -> torch.Tensor:
    support_dir = os.path.join("datasets", CURRENT_SYSTEM, "support", "features")
    support_paths = get_file_list(support_dir)

    target_params = slow_params_eval
    if not isinstance(target_params, (list, tuple, np.ndarray)):
        target_params = [target_params]

    idx = _find_context_index(support_paths, target_params)
    if idx is None or idx >= len(contexts):
        idx = 0
    return contexts[idx]


def predict_trajectory_cavia(
    x0: np.ndarray,
    T: int,
    dt_value: float,
    normalization: dict,
    context: torch.Tensor,
    cfg: dict,
    device: str,
    save_derivatives: bool = False,
    use_raw_data: bool = False,
) -> np.ndarray:
    input_dim = dataset_dict["info"]["input_dim"]
    output_dim = dataset_dict["info"]["output_dim"]
    model = _build_cavia_model(input_dim, output_dim, device, cfg)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    context = context.to(device)
    context.requires_grad = False
    model.context_params = context

    if not use_raw_data:
        input_mean = torch.tensor(
            normalization["input"]["mean"], dtype=torch.float32, device=device
        )
        input_std = torch.tensor(normalization["input"]["std"], dtype=torch.float32, device=device)
        output_mean = torch.tensor(
            normalization["output"]["mean"], dtype=torch.float32, device=device
        )
        output_std = torch.tensor(
            normalization["output"]["std"], dtype=torch.float32, device=device
        )
    else:
        input_mean = None
        input_std = None
        output_mean = None
        output_std = None

    if use_raw_data:
        input_dim_norm = input_dim
    else:
        input_dim_norm = input_mean.numel()

    x0_np = np.asarray(x0).reshape(-1)
    if x0_np.size < input_dim_norm:
        x0_np = np.pad(x0_np, (0, input_dim_norm - x0_np.size))
    elif x0_np.size > input_dim_norm:
        x0_np = x0_np[:input_dim_norm]

    x = torch.tensor(x0_np, dtype=torch.float32, device=device).reshape(1, -1)
    traj = [x.detach().cpu().numpy().copy()]

    true_derivatives = []
    pred_derivatives = []

    controller = None
    if CURRENT_SYSTEM == "quadrotor":
        controller = dssdl_eval.QuadrotorController(device=device)

    for _ in range(T):
        if controller:
            x[:, 6:] = controller.update(x)

        if datasets.CURRENT_SYSTEM == "pendulum_real":
            x_k1_scaled = x.clone()
            x_k1_scaled[:, 0] /= 20
            if not use_raw_data:
                x_k1_input = (x_k1_scaled - input_mean) / input_std
            else:
                x_k1_input = x_k1_scaled

            dx1_pred = model(x_k1_input)
            if use_raw_data:
                k1_unscaled = dx1_pred
            else:
                k1_unscaled = dx1_pred * output_std + output_mean

            k1 = k1_unscaled.clone()
            k1[:, 0] *= 100
            k1[:, 1] *= 20

            x_pred = x + k1 * dt_value
            x_k2_scaled = x_pred.clone()
            x_k2_scaled[:, 0] /= 20
            if not use_raw_data:
                x_k2_input = (x_k2_scaled - input_mean) / input_std
            else:
                x_k2_input = x_k2_scaled

            dx2_pred = model(x_k2_input)
            if use_raw_data:
                k2_unscaled = dx2_pred
            else:
                k2_unscaled = dx2_pred * output_std + output_mean

            k2 = k2_unscaled.clone()
            k2[:, 0] *= 100
            k2[:, 1] *= 20

            dx = 0.99 * k1 + 0.01 * k2
            x = x + dx * dt_value
        else:
            if not use_raw_data:
                x_input = (x - input_mean) / input_std
            else:
                x_input = x

            dx_pred = model(x_input)
            if use_raw_data:
                dx_out = dx_pred
            else:
                dx_out = dx_pred * output_std + output_mean

            out_dim = dx_out.shape[1]
            state_dim = x.shape[1]
            if out_dim == state_dim:
                dx = dx_out
            elif state_dim == 4 and out_dim == 2:
                vx_vy = x[:, 2:4]
                dx = torch.cat([vx_vy, dx_out], dim=1)
            elif CURRENT_SYSTEM == "quadrotor" and state_dim == 10 and out_dim == 3:
                angular_accel = dx_out
                phi, theta, p, q, r = x[0, 0], x[0, 1], x[0, 3], x[0, 4], x[0, 5]
                phi_dot = p + (q * torch.sin(phi) + r * torch.cos(phi)) * torch.tan(theta)
                theta_dot = q * torch.cos(phi) - r * torch.sin(phi)
                psi_dot = (q * torch.sin(phi) + r * torch.cos(phi)) / torch.cos(theta)
                full_dx_attitude = torch.stack(
                    [
                        phi_dot,
                        theta_dot,
                        psi_dot,
                        angular_accel[0, 0],
                        angular_accel[0, 1],
                        angular_accel[0, 2],
                    ]
                ).unsqueeze(0)
                dx = torch.cat([full_dx_attitude, torch.zeros((1, 4), device=device)], dim=1)
            else:
                raise RuntimeError(
                    f"Output dim {out_dim} not compatible with state dim {state_dim}"
                )

            x = x + dx * dt_value

            if save_derivatives:
                if datasets.CURRENT_SYSTEM == "quadrotor":
                    pred_derivatives.append(dx_out.detach().cpu().numpy().copy())
                else:
                    pred_derivatives.append(dx.detach().cpu().numpy().copy())
                x_np = x.detach().cpu().numpy().reshape(-1)
                true_deriv = f(x_np.reshape(1, -1), slow_params_eval).reshape(-1)
                true_derivatives.append(true_deriv.copy())

        traj.append(x.detach().cpu().numpy().copy())

    traj = np.concatenate(traj, axis=0)

    if save_derivatives and len(true_derivatives) > 0:
        true_derivatives = np.array(true_derivatives).squeeze()
        pred_derivatives = np.array(pred_derivatives).squeeze()

        derivatives_dir = f"derivatives/{CURRENT_SYSTEM}"
        os.makedirs(derivatives_dir, exist_ok=True)

        true_deriv_path = os.path.join(derivatives_dir, "true_derivatives.txt")
        pred_deriv_path = os.path.join(derivatives_dir, "pred_derivatives.txt")

        np.savetxt(true_deriv_path, true_derivatives)
        np.savetxt(pred_deriv_path, pred_derivatives)

        deriv_mae = np.mean(np.abs(true_derivatives - pred_derivatives))
        deriv_mse = np.mean((true_derivatives - pred_derivatives) ** 2)
        print(f"Saved true derivatives to: {true_deriv_path}")
        print(f"Saved predicted derivatives to: {pred_deriv_path}")
        print("Derivative error stats:")
        print(f"  MAE: {deriv_mae:.6f}")
        print(f"  MSE: {deriv_mse:.6f}")

    return traj


def _load_contexts() -> List[torch.Tensor]:
    context_path = os.path.join("results", "cavia", CURRENT_SYSTEM, "context_params.pt")
    if os.path.exists(context_path):
        payload = torch.load(context_path, map_location="cpu")
        contexts = payload["contexts"]
        print(f"Loaded {len(contexts)} context vectors from: {context_path}")
        return contexts

    print("No saved contexts found, adapting from support split.")
    return finetune_contexts(dataset_dict, CONFIG_OVERRIDES)


def main():
    print(f"Evaluating CAVIA trajectory on system: {CURRENT_SYSTEM}")
    print("=" * 50)

    steps = 5000
    device = "cpu"
    use_raw_data = CONFIG_OVERRIDES.get("use_raw_data", False)

    normalization = dataset_dict["info"]["normalization"]
    input_dim = len(normalization["input"]["mean"])
    output_dim = len(normalization["output"]["mean"])

    print(f"Input dim: {input_dim}, output dim: {output_dim}")
    print(f"dt: {dt}, steps: {steps}")
    print(f"use_raw_data: {use_raw_data}")

    contexts = _load_contexts()
    context_eval = _select_context_for_eval(contexts)

    if datasets.CURRENT_SYSTEM == "pendulum_real":
        query_features_path = os.path.join(
            "datasets",
            "pendulum_real",
            "query",
            "features",
            f"{slow_params_eval[0]:.2f}.txt",
        )
        print(f"Loading true trajectory from: {query_features_path}")
        traj_true = np.loadtxt(query_features_path)
        traj_true[:, 0] *= 20
        traj_true = traj_true[5:]
        x0_arr = traj_true[0]
        steps = len(traj_true) - 1
        print(f"True trajectory shape: {traj_true.shape}, steps: {steps}")
    elif datasets.CURRENT_SYSTEM in {"flow", "flow-19", "flow-19-v", "flow-29-v"}:
        query_features_path = os.path.join(
            "datasets",
            datasets.CURRENT_SYSTEM,
            "query",
            "features",
            f"{slow_params_eval[0]}.txt",
        )
        print(f"Loading true trajectory from: {query_features_path}")
        traj_true = np.loadtxt(query_features_path)
        x0_arr = traj_true[0]
        steps = len(traj_true) - 1
        steps = 200
        print(f"True trajectory shape: {traj_true.shape}, steps: {steps}")
    else:
        x0_arr = dssdl_eval.generate_adaptive_initial_state(CURRENT_SYSTEM, input_dim)
        print(f"Initial state: {x0_arr}")
        print("Generating true trajectory...")
        traj_true = dssdl_eval.true_trajectory(x0_arr, steps, dt)

    print("Generating predicted trajectory...")
    traj_pred = predict_trajectory_cavia(
        x0_arr,
        steps,
        dt,
        normalization,
        context_eval,
        CONFIG_OVERRIDES,
        device=device,
        save_derivatives=True,
        use_raw_data=use_raw_data,
    )

    traj_pred_noft = None
    noft_path = os.path.join("results", "cavia", CURRENT_SYSTEM, "context_params_noft.pt")
    if os.path.exists(noft_path):
        payload = torch.load(noft_path, map_location="cpu")
        contexts_noft = payload["contexts"]
        context_noft = _select_context_for_eval(contexts_noft)
        traj_pred_noft = predict_trajectory_cavia(
            x0_arr,
            steps,
            dt,
            normalization,
            context_noft,
            CONFIG_OVERRIDES,
            device=device,
            save_derivatives=False,
            use_raw_data=use_raw_data,
        )

    traj_true = traj_true[: steps + 1]
    traj_pred = traj_pred[: steps + 1]
    if traj_pred_noft is not None:
        traj_pred_noft = traj_pred_noft[: steps + 1]

    mae_overall = np.mean(np.abs(traj_true - traj_pred))
    mse_overall = np.mean((traj_true - traj_pred) ** 2)
    print("Overall error stats:")
    print(f"  MAE: {mae_overall:.6f}")
    print(f"  MSE: {mse_overall:.6f}")

    if traj_pred_noft is not None:
        mae_noft = np.mean(np.abs(traj_true - traj_pred_noft))
        mse_noft = np.mean((traj_true - traj_pred_noft) ** 2)
        print(f"  No-adaptation MAE: {mae_noft:.6f}")
        print(f"  No-adaptation MSE: {mse_noft:.6f}")

    dssdl_eval.export_trajectory_data(
        traj_true, traj_pred, dt, CURRENT_SYSTEM, tag=f"{slow_params_eval}", x0=x0_arr
    )
    if traj_pred_noft is not None:
        dssdl_eval.export_trajectory_data(
            traj_true,
            traj_pred_noft,
            dt,
            CURRENT_SYSTEM,
            tag=f"{slow_params_eval}_no_finetune",
            x0=x0_arr,
        )

    save_dir = f"pictures/{CURRENT_SYSTEM}"
    print(f"Saving figures to: {save_dir}")
    dssdl_eval.plot_trajectory_adaptive(
        traj_true, traj_pred, dt, CURRENT_SYSTEM, save_dir, traj_pred_noft=traj_pred_noft
    )
    dssdl_eval.plot_phase_diagram(traj_true, traj_pred, dt, CURRENT_SYSTEM, save_dir)


if __name__ == "__main__":
    main()
