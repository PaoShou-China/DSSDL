#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorenz system dataset generation
Based on the system equations: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
from typing import Tuple, List

# 定义dt参数，供其他模块导入
dt = 0.01  # 时间步长

# =========================
#   Lorenz System
#   dx/dt = σ(y-x)
#   dy/dt = x(ρ-z)-y  
#   dz/dt = xy-βz
# =========================

def f(t, y, slow_param):
    """
    Lorenz系统动力学方程
    输入：
    - t: 时间（未使用）
    - y: 状态 [x, y, z]
    - slow_param: [σ, ρ, β]
    输出：
    - dydt: [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = y
    sigma, rho, beta = slow_param
    
    # Lorenz方程
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    return [dxdt, dydt, dzdt]


def create_dataset(
    f_handle,
    slow_param: List[float],
    ranges: List[Tuple[float, float]],
    save_dir: str,
    num_trajectories: int = 10,
    t_span: Tuple[float, float] = (0.0, 10.0),
    dt_override: float = None,
    seed: int = 0,
):
    """
    生成Lorenz系统轨迹数据集
    features: [x, y, z]
    labels:   [dx/dt, dy/dt, dz/dt]
    
    Args:
        f_handle: 动力学函数
        slow_param: [σ, ρ, β] - Lorenz系统参数
        ranges: 初始状态范围 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        save_dir: 保存目录
        num_trajectories: 轨迹数量
        t_span: 时间跨度
        dt_override: 时间步长覆盖（如果为None则使用全局dt）
        seed: 随机种子
    """
    # 使用全局dt或覆盖值
    dt_actual = dt_override if dt_override is not None else dt
    rng = np.random.RandomState(seed)

    # Always integrate a full 10s trajectory, then slice to t_span for output.
    full_t_span = (t_span[0], 10.0)
    t_eval_full = np.arange(full_t_span[0], full_t_span[1], dt_actual)
    t_eval_out = np.arange(t_span[0], t_span[1], dt_actual)
    n_steps_out = len(t_eval_out)
    total_samples = num_trajectories * n_steps_out

    features = np.zeros((total_samples, 3), dtype=np.float64)  # [x, y, z]
    labels = np.zeros((total_samples, 3), dtype=np.float64)    # numerical derivatives

    idx = 0

    for _ in range(num_trajectories):
        # 随机初始条件
        x0 = rng.uniform(ranges[0][0], ranges[0][1])  # 初始 x
        y0 = rng.uniform(ranges[1][0], ranges[1][1])  # 初始 y  
        z0 = rng.uniform(ranges[2][0], ranges[2][1])  # 初始 z

        def dydt(t, y):
            return f_handle(t, y, slow_param)

        sol = solve_ivp(
            dydt,
            full_t_span,
            y0=[x0, y0, z0],
            t_eval=t_eval_full,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            continue

        x_traj = sol.y[0]
        y_traj = sol.y[1]
        z_traj = sol.y[2]

        if len(x_traj) > 1:
            dxdt = np.gradient(x_traj, dt_actual, edge_order=2)
            dydt = np.gradient(y_traj, dt_actual, edge_order=2)
            dzdt = np.gradient(z_traj, dt_actual, edge_order=2)
        else:
            dxdt = np.zeros_like(x_traj)
            dydt = np.zeros_like(y_traj)
            dzdt = np.zeros_like(z_traj)

        # numerical derivatives via forward differences
        # dxdt = np.empty_like(x_traj)
        # dydt = np.empty_like(y_traj)
        # dzdt = np.empty_like(z_traj)
        # dxdt[:-1] = np.diff(x_traj) / dt_actual
        # dydt[:-1] = np.diff(y_traj) / dt_actual
        # dzdt[:-1] = np.diff(z_traj) / dt_actual
        # dxdt[-1] = dxdt[-2] if len(x_traj) > 1 else 0.0
        # dydt[-1] = dydt[-2] if len(y_traj) > 1 else 0.0
        # dzdt[-1] = dzdt[-2] if len(z_traj) > 1 else 0.0

        x_traj_out = x_traj[:n_steps_out]
        y_traj_out = y_traj[:n_steps_out]
        z_traj_out = z_traj[:n_steps_out]
        dxdt_out = dxdt[:n_steps_out]
        dydt_out = dydt[:n_steps_out]
        dzdt_out = dzdt[:n_steps_out]

        for i in range(n_steps_out):
            state_i = [x_traj_out[i], y_traj_out[i], z_traj_out[i]]
            deriv_i = [dxdt_out[i], dydt_out[i], dzdt_out[i]]

            features[idx] = state_i
            labels[idx] = deriv_i
            idx += 1

    # 裁剪到有效样本数
    features = features[:idx]
    labels = labels[:idx]

    # 保存
    sigma, rho, beta = slow_param
    slow_str = f"{sigma}_{rho}_{beta}"
    features_dir = os.path.join(save_dir, "features")
    labels_dir = os.path.join(save_dir, "labels")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    np.savetxt(os.path.join(features_dir, f"{slow_str}.txt"), features)
    np.savetxt(os.path.join(labels_dir, f"{slow_str}.txt"), labels)
    print(f"Saved {idx} samples ({t_span[1]-t_span[0]:.2f}s trajectory) for slow_param {slow_param} to {save_dir}")


# 参数设置（基于GS-SINDy设置）
# 训练参数：[σ, ρ, β]
all_slow_params = [
                    # [10, 28, 2.67], [9, 30, 2], [8.5, 28, 2.67], [10, 27, 2.67], [10, 27, 3], [10, 29, 2],
                    [10, 28, 2.67], [10, 30, 2.5], [8.5, 28, 2.67], [10, 27, 2.67], [11, 27, 3], [10, 29, 2.5],
                    [9, 9, 2.67], [10, 11, 3], [11, 10.5, 2.5], [10, 12, 3], [9, 11.5, 2.5], [8.5, 12.5, 3],
                    [10, 20, 2.67], [9, 20.5, 2.5], [8.5, 21, 3], [9.5, 21.5, 2.5], [10, 22, 2.67], [11, 22.5, 3], # 不同极限环实验
                    # [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 1], [1, 2, 2], [1, 2, 3], [1, 3, 1], [1, 3, 2], [1, 3, 3], 
                    # [2, 1, 1], [2, 1, 2], [2, 1, 3], [2, 2, 1], [2, 2, 2], [2, 2, 3], [2, 3, 1], [2, 3, 2], [2, 3, 3],
                    # [3, 1, 1], [3, 1, 2], [3, 1, 3], [3, 2, 1], [3, 2, 2], [3, 2, 3], [3, 3, 1], [3, 3, 2], [3, 3, 3], # 可解释性实验
                #    [10.5, 25, 2.5], [9.5, 26, 2.8], [11, 24, 3.2], [8, 31, 1.8], [12, 23, 3.5], [7.5, 32, 1.5],
                #    [11.5, 22, 4], [7, 33, 1.2], [13, 21, 4.5], [6.5, 34, 1], [14, 20, 5], [6, 35, 0.8],
                #    [15, 19, 5.5], [5.5, 36, 0.6], [16, 18, 6], [5, 37, 0.5], [17, 17, 6.5], [4.5, 38, 0.4],
                #    [18, 16, 7], [4, 39, 0.3], [19, 15, 7.5], # 时序误差实验/seed-support实验 
                #    [10.2, 28.5, 2.8]

                # [a, b, i_ext]
                #     for a in [7, 9, 11]
                #     for b in [24, 28, 32]
                #     for i_ext in [1, 4, 7]
                ]

unseen_params = [
    # [10, 10, 2.67]
    [10, 20, 2.67]
    # [10, 29, 2.67]
    # [10, 35, 2.67]  # 不同极限环实验
    # [9.9, 29.5, 2.9],  # 时序误差实验/seed-support实验
    # [1.1, 1.2, 1.3]
]
slow_params_eval = unseen_params[0]

if __name__ == "__main__":
    # Import the clear function from utils
    import sys
    import os
    
    # Add parent directory to path for utils import
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from utils import clear_dataset_files
    
    # Clear existing dataset files
    clear_dataset_files("lorenz")
    
    # ==== 参数设置 ====
    train_params = all_slow_params

    # ==== 初始条件范围 ====
    # Lorenz系统的合理初始条件范围
    # ranges = [
    #     (5, 10),   # x range
    #     (-5, -10),   # y range
    #     (20, 30),    # z range
    # ]
    ranges = [
        (5, 10),   # x range
        (-10, -5),   # y range
        (20, 30),    # z range
    ]
    # ==== 目录 ====
    base_dir = "datasets/lorenz"
    os.makedirs(base_dir, exist_ok=True)

    # ---- 训练：每个系统10s轨迹，多个初值 ----
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    seed_master = 0
    for i, sp in enumerate(train_params):
        create_dataset(
            f_handle=f,
            slow_param=sp,
            ranges=ranges,
            save_dir=train_dir,
            num_trajectories=1,     # 每个系统生成多条轨迹
            t_span=(0.0, 10.0),      # 10s轨迹
            seed=seed_master + i,
        )

    # ---- 测试support（任务自适应）：每个未见系统生成5s轨迹 ----
    support_dir = os.path.join(base_dir, "support")
    os.makedirs(support_dir, exist_ok=True)
    for i, sp in enumerate(unseen_params):
        create_dataset(
            f_handle=f,
            slow_param=sp,
            ranges=ranges,
            save_dir=support_dir,
            num_trajectories=1,     # 多个初值
            t_span=(0.0, 1),       # 5s轨迹
            seed=2025 + i,
        )

    # ---- 测试query（评估）：每个未见系统生成5s轨迹 ----
    query_dir = os.path.join(base_dir, "query")
    os.makedirs(query_dir, exist_ok=True)
    for i, sp in enumerate(unseen_params):
        create_dataset(
            f_handle=f,
            slow_param=sp,
            ranges=ranges,
            save_dir=query_dir,
            num_trajectories=1,     # 多个初值
            t_span=(0.0, 5.0),       # 5s评估
            seed=4096 + i,
        )

    print("Lorenz system dataset generated successfully.")
    print("系统参数说明：")
    print(f"- 训练参数：{len(train_params)}个实例，[σ, ρ, β]")
    print(f"- 测试参数：{len(unseen_params)}个实例，{unseen_params}")
    print("- 时间设置：训练10s轨迹，测试5s轨迹，时间步长10ms")
