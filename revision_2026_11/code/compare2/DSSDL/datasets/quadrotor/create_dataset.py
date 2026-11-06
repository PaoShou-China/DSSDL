#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadrotor system dataset generation
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
import itertools
from typing import Tuple, List

# 定义dt参数，供其他模块导入
dt = 0.02  # 时间步长

# =========================
#   Quadrotor System Dynamics
# =========================

# --- 物理参数 ---
k_T = 1e-6  # 推力系数
k_M = 1e-7  # 力矩系数
l = 0.2  # 电机到中心的距离


def attitude_dynamics(t, state, motor_speeds_sq, slow_param):
    """
    姿态动力学微分方程
    state: [φ, θ, ψ, p, q, r]
    motor_speeds_sq: [Ω1², Ω2², Ω3², Ω4²] (电机速度平方)
    slow_param: [Ix, Iy, Iz]
    """
    phi, theta, psi, p, q, r = state
    Ix, Iy, Iz = slow_param

    # 计算控制力矩
    tau_phi = l * k_T * (motor_speeds_sq[3] - motor_speeds_sq[1])  # roll
    tau_theta = l * k_T * (motor_speeds_sq[2] - motor_speeds_sq[0])  # pitch
    tau_psi = k_M * (
        motor_speeds_sq[0] - motor_speeds_sq[1] + motor_speeds_sq[2] - motor_speeds_sq[3]
    )  # yaw

    # 角加速度 (欧拉方程)
    p_dot = (Iy - Iz) / Ix * q * r + tau_phi / Ix
    q_dot = (Iz - Ix) / Iy * p * r + tau_theta / Iy
    r_dot = (Ix - Iy) / Iz * p * q + tau_psi / Iz

    # 欧拉角导数 (避免奇点: θ 接近 ±π/2)
    if abs(theta) > np.pi / 2 - 0.01:
        theta = np.sign(theta) * (np.pi / 2 - 0.01)

    phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    return np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])


def create_dataset(
    f_handle,
    slow_param: List[float],
    ranges: List[Tuple[float, float]],
    save_dir: str,
    num_trajectories: int = 10,
    t_span: Tuple[float, float] = (0.0, 0.5),
    dt_override: float = None,
    seed: int = 0,
):
    """
    生成四旋翼系统轨迹数据集
    features: [φ, θ, ψ, p, q, r, Ω1², Ω2², Ω3², Ω4²]
    labels:   [p_dot, q_dot, r_dot]

    Args:
        f_handle: 动力学函数 (未使用, 逻辑内联)
        slow_param: [Ix, Iy, Iz] - 四旋翼惯性参数
        ranges: 初始状态范围
        save_dir: 保存目录
        num_trajectories: 轨迹数量
        t_span: 时间跨度
        dt_override: 时间步长覆盖
        seed: 随机种子
    """
    dt_actual = dt_override if dt_override is not None else dt
    rng = np.random.RandomState(seed)

    t_eval = np.arange(t_span[0], t_span[1] + dt_actual, dt_actual)
    n_steps = len(t_eval)

    all_features = []
    all_labels = []

    # --- 控制器和电机参数 ---
    MIN_OMEGA_SQ = 0.0
    MAX_OMEGA_SQ = 1e-4
    Kp = np.array([0.05, 0.05, 0.02])
    Kd = np.array([0.01, 0.01, 0.005])

    for _ in range(num_trajectories):
        # 随机初始条件
        initial_state = np.array([rng.uniform(r[0], r[1]) for r in ranges])

        states = np.zeros((n_steps, 6))
        controls = np.zeros((n_steps, 4))
        derivatives = np.zeros((n_steps, 6))
        states[0] = initial_state

        for i in range(n_steps - 1):
            t = t_eval[i]
            state = states[i]
            phi, theta, psi, p, q, r = state

            # PD控制器 + 随机扰动
            angle_error = np.array([0 - phi, 0 - theta, 0 - psi])
            angle_rate_error = np.array([0 - p, 0 - q, 0 - r])
            tau_pd = Kp * angle_error + Kd * angle_rate_error

            # 随机电机输入
            random_motor_speeds_sq = rng.uniform(MIN_OMEGA_SQ, MAX_OMEGA_SQ, size=4)
            tau_rand_phi = l * k_T * (random_motor_speeds_sq[3] - random_motor_speeds_sq[1])
            tau_rand_theta = l * k_T * (random_motor_speeds_sq[2] - random_motor_speeds_sq[0])
            tau_rand_psi = k_M * (
                random_motor_speeds_sq[0]
                - random_motor_speeds_sq[1]
                + random_motor_speeds_sq[2]
                - random_motor_speeds_sq[3]
            )
            tau_total = np.array([tau_rand_phi, tau_rand_theta, tau_rand_psi]) + tau_pd

            # 控制分配
            total_thrust_base = 0.5  # 基础推力
            omega1_sq = (
                total_thrust_base / (4 * k_T)
                + tau_total[1] / (2 * l * k_T)
                - tau_total[2] / (4 * k_M)
            )
            omega2_sq = (
                total_thrust_base / (4 * k_T)
                - tau_total[0] / (2 * l * k_T)
                + tau_total[2] / (4 * k_M)
            )
            omega3_sq = (
                total_thrust_base / (4 * k_T)
                - tau_total[1] / (2 * l * k_T)
                - tau_total[2] / (4 * k_M)
            )
            omega4_sq = (
                total_thrust_base / (4 * k_T)
                + tau_total[0] / (2 * l * k_T)
                + tau_total[2] / (4 * k_M)
            )
            motor_speeds_sq = np.maximum([omega1_sq, omega2_sq, omega3_sq, omega4_sq], MIN_OMEGA_SQ)

            # 动力学演化 (RK4)
            def dydt(t, y, u):
                return attitude_dynamics(t, y, u, slow_param)

            k1 = dydt(t, state, motor_speeds_sq)
            k2 = dydt(t + dt_actual / 2, state + dt_actual / 2 * k1, motor_speeds_sq)
            k3 = dydt(t + dt_actual / 2, state + dt_actual / 2 * k2, motor_speeds_sq)
            k4 = dydt(t + dt_actual, state + dt_actual * k3, motor_speeds_sq)
            next_state = state + dt_actual / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            states[i + 1] = next_state
            controls[i] = motor_speeds_sq
            derivatives[i] = k1  # 使用k1作为当前时刻的导数

        # 组装特征和标签
        features_traj = np.concatenate([states[:-1], controls[:-1]], axis=1)
        labels_traj = derivatives[:-1, 3:6]  # [p_dot, q_dot, r_dot]
        all_features.append(features_traj)
        all_labels.append(labels_traj)

    features = np.vstack(all_features)
    labels = np.vstack(all_labels)

    # 保存
    param_str = "_".join(map(lambda x: f"{x:.3f}".replace(".", "_"), slow_param))
    features_dir = os.path.join(save_dir, "features")
    labels_dir = os.path.join(save_dir, "labels")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    np.savetxt(os.path.join(features_dir, f"{param_str}.txt"), features)
    np.savetxt(os.path.join(labels_dir, f"{param_str}.txt"), labels)
    print(f"Saved {features.shape[0]} samples for slow_param {slow_param} to {save_dir}")


# =============================
# Parameters and Configuration
# =============================
Ix_candidates = [0.018, 0.02, 0.022]
Iy_candidates = [0.023, 0.025, 0.027]
Iz_candidates = [0.028, 0.03, 0.032]

all_slow_params = list(itertools.product(Ix_candidates, Iy_candidates, Iz_candidates))
unseen_params = [(0.019, 0.024, 0.029)]
slow_params_eval = unseen_params[0]

if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from utils import clear_dataset_files

    clear_dataset_files("quadrotor")

    # ==== 参数设置 ====
    train_params = all_slow_params

    # ==== 初始条件范围 [φ, θ, ψ, p, q, r] ====
    ranges = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]

    # ==== 目录 ====
    base_dir = os.path.join("datasets", "quadrotor")
    os.makedirs(base_dir, exist_ok=True)

    # ---- 训练 ----
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    for i, sp in enumerate(train_params):
        create_dataset(
            f_handle=None,  # not used
            slow_param=list(sp),
            ranges=ranges,
            save_dir=train_dir,
            num_trajectories=100,  # 减少轨迹数量以加快生成
            t_span=(0.0, 0.5),
            seed=i,
        )

    # ---- 测试support ----
    support_dir = os.path.join(base_dir, "support")
    os.makedirs(support_dir, exist_ok=True)
    for i, sp in enumerate(unseen_params):
        create_dataset(
            f_handle=None,
            slow_param=list(sp),
            ranges=ranges,
            save_dir=support_dir,
            num_trajectories=20,
            t_span=(0.0, 0.5),
            seed=2025 + i,
        )

    # ---- 测试query ----
    query_dir = os.path.join(base_dir, "query")
    os.makedirs(query_dir, exist_ok=True)
    for i, sp in enumerate(unseen_params):
        create_dataset(
            f_handle=None,
            slow_param=list(sp),
            ranges=ranges,
            save_dir=query_dir,
            num_trajectories=100,
            t_span=(0.0, 0.5),
            seed=4096 + i,
        )

    print("\nQuadrotor system dataset generated successfully.")
