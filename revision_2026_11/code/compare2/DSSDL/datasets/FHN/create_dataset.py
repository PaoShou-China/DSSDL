#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FitzHugh-Nagumo system dataset generation.
Equations:
    v_dot = v - v**3 / 3 - w + I_ext
    w_dot = epsilon * (v + a - b * w)
"""

import os
from typing import List, Tuple, Optional

import numpy as np
from scipy.integrate import solve_ivp

# Default timestep shared with other modules
dt = 0.01
# Fixed time-scale parameter
EPSILON = 0.20


def f(t, y, slow_param):
    """FitzHugh-Nagumo dynamics for solve_ivp."""
    v, w = y
    a, b, i_ext = slow_param
    v_dot = v - (v ** 3) / 3.0 - w + i_ext
    w_dot = EPSILON * (v + a - b * w)
    return [v_dot, w_dot]


def create_dataset(
    f_handle,
    slow_param: List[float],
    ranges: List[Tuple[float, float]],
    save_dir: str,
    num_trajectories: int = 10,
    t_span: Tuple[float, float] = (0.0, 10.0),
    dt_override: float = None,
    seed: int = 0,
    sample_points: Optional[int] = None,
):
    """
    Generate trajectories for the FitzHugh-Nagumo system.

    features: [v, w]
    labels:   [v_dot, w_dot]
    slow_param: [a, b, I_ext] with fixed epsilon=EPSILON
    sample_points: if provided, randomly subsample this many points from the
        full trajectory (without replacement) before saving.
    """
    dt_actual = dt_override if dt_override is not None else dt
    rng = np.random.RandomState(seed)

    t_eval = np.arange(t_span[0], t_span[1], dt_actual)
    n_steps = len(t_eval)
    total_samples = num_trajectories * n_steps

    features = np.zeros((total_samples, 2), dtype=np.float64)
    labels = np.zeros((total_samples, 2), dtype=np.float64)

    idx = 0

    for _ in range(num_trajectories):
        v0 = rng.uniform(ranges[0][0], ranges[0][1])
        w0 = rng.uniform(ranges[1][0], ranges[1][1])

        def dydt(t, y):
            return f_handle(t, y, slow_param)

        sol = solve_ivp(
            dydt,
            t_span,
            y0=[v0, w0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            continue

        v_traj = sol.y[0]
        w_traj = sol.y[1]

        # First-order finite differences for derivatives
        v_dot = np.empty_like(v_traj)
        w_dot = np.empty_like(w_traj)
        v_dot[:-1] = np.diff(v_traj) / dt_actual
        w_dot[:-1] = np.diff(w_traj) / dt_actual
        v_dot[-1] = v_dot[-2] if len(v_traj) > 1 else 0.0
        w_dot[-1] = w_dot[-2] if len(w_traj) > 1 else 0.0

        for i in range(n_steps):
            features[idx] = [v_traj[i], w_traj[i]]
            labels[idx] = [v_dot[i], w_dot[i]]
            idx += 1

    features = features[:idx]
    labels = labels[:idx]

    # Optional random subsampling of time points
    if sample_points is not None and sample_points < idx:
        select_idx = rng.choice(idx, size=sample_points, replace=False)
        select_idx.sort()
        features = features[select_idx]
        labels = labels[select_idx]
        idx = sample_points

    a, b, i_ext = slow_param
    slow_str = f"{a:.2f}_{b:.2f}_{i_ext:.2f}"
    features_dir = os.path.join(save_dir, "features")
    labels_dir = os.path.join(save_dir, "labels")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    np.savetxt(os.path.join(features_dir, f"{slow_str}.txt"), features)
    np.savetxt(os.path.join(labels_dir, f"{slow_str}.txt"), labels)
    print(
        f"Saved {idx} samples ({t_span[1] - t_span[0]:.2f}s trajectory) "
        f"for slow_param {slow_param} to {save_dir}"
    )


# Parameter sets for training and evaluation (epsilon fixed to EPSILON)
train_params = [
    # [0.60, 0.90, 0.40],
    # [0.70, 0.80, 0.50],
    # [0.70, 0.90, 0.70],
    # [0.80, 1.00, 0.60],
    # [0.70, 0.70, 0.80],
    # [a, b, i_ext]
    # for a in [0.6, 0.7, 0.8]
    # for b in [0.8, 0.9, 1.0]
    # for i_ext in [0.6, 0.7, 0.8]    # FHN的slowparam解释性实验训练

    [a, b, i_ext]
    for a in [0.6, 0.7, 0.8]
    for b in [0.8, 0.9, 1.0]
    for i_ext in [0.0, 0.1, 0.2] # FHN的时序误差实验/seed-support实验 不同I_ext实验训练数据 



]

unseen_params = [
    # [0.9, 1.1, 0.9],
    # [0.9, 1.1, 1.1],
    # [0.9, 1.1, 0], # FHN的不同I_ext实验fintune数据
    [0.85, 1.05, 0.85] # FHN的时序误差实验/seed-support实验
]
slow_params_eval = unseen_params[0]


if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from utils import clear_dataset_files

    clear_dataset_files("FHN")

    ranges = [
        (-2.0, 2.0),  # v range
        (-2.0, 2.0),  # w range
    ]

    base_dir = "datasets/FHN"
    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    seed_master = 0
    for i, sp in enumerate(train_params):
        create_dataset(
            f_handle=f,
            slow_param=sp,
            ranges=ranges,
            save_dir=train_dir,
            num_trajectories=1,
            t_span=(0.0, 10.0),
            seed=seed_master + i,
        )

    support_dir = os.path.join(base_dir, "support")
    os.makedirs(support_dir, exist_ok=True)
    for i, sp in enumerate(unseen_params):
        create_dataset(
            f_handle=f,
            slow_param=sp,
            ranges=ranges,
            save_dir=support_dir,
            num_trajectories=1,
            t_span=(0.0, 10.0),
            seed=2025 + i,
            sample_points=100,
            # sample_points=None,
        )

    query_dir = os.path.join(base_dir, "query")
    os.makedirs(query_dir, exist_ok=True)
    for i, sp in enumerate(unseen_params):
        create_dataset(
            f_handle=f,
            slow_param=sp,
            ranges=ranges,
            save_dir=query_dir,
            num_trajectories=1,
            t_span=(0.0, 10.0),
            seed=4096 + i,  # random subset of time points from full trajectory
        )

    print("FitzHugh-Nagumo dataset generation finished.")
