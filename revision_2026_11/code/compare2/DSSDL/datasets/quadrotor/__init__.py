#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadrotor系统数据集模块
"""

import numpy as np
import os
from ..utils import get_datasets_generic

# 获取当前模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 从create_dataset导入dt参数
try:
    from .create_dataset import dt, k_T, k_M, l
except ImportError:
    # 如果导入失败，使用默认值
    dt = 0.02
    k_T = 1e-6
    k_M = 1e-7
    l = 0.2

# Quadrotor系统参数
from .create_dataset import slow_params_eval

def f(features, slow_params):
    """
    Quadrotor系统动力学函数 (仅姿态动力学)
    输入：
    - features: 状态和控制输入数组，形状为 (N, 10)，每行为
                [φ, θ, ψ, p, q, r, Ω1², Ω2², Ω3², Ω4²]
    - slow_params: [Ix, Iy, Iz]
    输出：
    - derivatives: 角加速度数组，形状为 (N, 3)，每行为 [p_dot, q_dot, r_dot]
    """
    # 提取状态和控制输入
    p = features[:, [3]]
    q = features[:, [4]]
    r = features[:, [5]]
    omega1_sq = features[:, [6]]
    omega2_sq = features[:, [7]]
    omega3_sq = features[:, [8]]
    omega4_sq = features[:, [9]]
    
    # 提取慢变参数
    Ix, Iy, Iz = slow_params

    # 计算控制力矩
    tau_phi = l * k_T * (omega4_sq - omega2_sq)      # roll
    tau_theta = l * k_T * (omega3_sq - omega1_sq)  # pitch
    tau_psi = k_M * (omega1_sq - omega2_sq + omega3_sq - omega4_sq)  # yaw

    # 计算角加速度 (欧拉方程)
    p_dot = (Iy - Iz) / Ix * q * r + tau_phi / Ix
    q_dot = (Iz - Ix) / Iy * p * r + tau_theta / Iy
    r_dot = (Ix - Iy) / Iz * p * q + tau_psi / Iz
    
    return np.concatenate([p_dot, q_dot, r_dot], axis=1)


def get_datasets(use_normalization=True):
    """
    获取Quadrotor系统数据集
    
    Args:
        use_normalization (bool): 是否使用标准化
    
    Returns:
        dataset_dict: 标准格式的数据集字典
    """
    return get_datasets_generic(current_dir, use_normalization)


def get_raw_dataset_dict():
    """获取Quadrotor系统的原始数据集"""
    return get_datasets(use_normalization=False)


# 默认数据集（向后兼容）
dataset_dict = get_datasets(use_normalization=True)

