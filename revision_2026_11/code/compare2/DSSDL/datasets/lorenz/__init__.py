#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorenz系统数据集模块
"""

import numpy as np
import os
from ..utils import get_datasets_generic

# 获取当前模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 从create_dataset导入dt参数
try:
    from .create_dataset import dt
except ImportError:
    # 如果导入失败，使用默认值
    dt = 0.01

# Lorenz系统参数
from .create_dataset import slow_params_eval

def f(features, slow_params):
    """
    Lorenz系统动力学函数
    输入：
    - features: 状态变量数组，形状为 (N, 3)，每行为 [x, y, z]
    - slow_params: [sigma, rho, beta]
    输出：
    - derivatives: 导数数组，形状为 (N, 3)，每行为 [dx/dt, dy/dt, dz/dt]
    """
    x1, x2, x3 = features[:, [0]], features[:, [1]], features[:, [2]]
    sigma, rho, beta = slow_params
    x1_dot = sigma * (x2 - x1)
    x2_dot = x1 * (rho - x3) - x2
    x3_dot = x1 * x2 - beta * x3
    return np.concatenate([x1_dot, x2_dot, x3_dot], axis=1)


def get_datasets(use_normalization=True):
    """
    获取Lorenz系统数据集
    
    Args:
        use_normalization (bool): 是否使用标准化
    
    Returns:
        dataset_dict: 标准格式的数据集字典
    """
    return get_datasets_generic(current_dir, use_normalization)


def get_raw_dataset_dict():
    """获取Lorenz系统的原始数据集"""
    return get_datasets(use_normalization=False)


# 默认数据集（向后兼容）
dataset_dict = get_datasets(use_normalization=True)
