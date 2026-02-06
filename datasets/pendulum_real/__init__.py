#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实单摆系统数据集模块
"""

import numpy as np
import os
from ..utils import get_datasets_generic

# 获取当前模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 对于真实数据，dt 是固定的，这里我们假设一个值，或者从数据描述中获取
# 如果未知，我们设定一个通用值
from .create_dataset import dt

# 慢参数对应于数据文件的索引
all_slow_params = [[i] for i in range(1, 8)]  # train
all_slow_params.append([8]) # support/query

slow_params_eval = [8]

def f(features, slow_params):
    """
    真实单摆的动力学函数是未知的。这个函数是一个占位符。
    在实际应用中，模型将尝试从数据中学习这个函数。
    返回与输入特征相同形状的零数组，以匹配接口。
    """
    # print("Warning: The underlying dynamic function for 'pendulum_real' is unknown. Returning zeros.")
    return np.zeros_like(features)

def get_datasets(use_normalization=True):
    """
    获取真实单摆系统数据集
    
    Args:
        use_normalization (bool): 是否使用标准化
    
    Returns:
        dataset_dict: 标准格式的数据集字典
    """
    return get_datasets_generic(current_dir, use_normalization)

def get_raw_dataset_dict():
    """获取真实单摆系统的原始数据集"""
    return get_datasets(use_normalization=False)


# 默认数据集（向后兼容）
dataset_dict = get_datasets(use_normalization=True)

