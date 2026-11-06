#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流体系统数据集模块
基于不同雷诺数下的流体涡度数据
"""

import numpy as np
import os
from ..utils import get_datasets_generic

# 获取当前模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 时间步长
from .create_dataset import dt

# 雷诺数参数
# 训练集：100-350
all_slow_params = [[re_num] for re_num in [100, 125, 150, 200, 175, 225, 250, 275
                                           , 300, 325, 350
                                           ]]

# 测试集：375
slow_params_eval = [375]

def f(features, slow_params):
    """
    流体系统的动力学函数。
    对于真实数据，这个函数是未知的。这个函数是一个占位符。
    在实际应用中，模型将尝试从数据中学习这个函数。
    返回与输入特征相同形状的零数组，以匹配接口。
    
    Args:
        features: 特征数据 (涡度场)
        slow_params: 慢参数 (雷诺数)
    
    Returns:
        与features相同形状的零数组
    """
    return np.zeros_like(features)

def get_datasets(use_normalization=True):
    """
    获取流体系统数据集
    
    Args:
        use_normalization (bool): 是否使用标准化
    
    Returns:
        dataset_dict: 标准格式的数据集字典
    """
    return get_datasets_generic(current_dir, use_normalization)

def get_raw_dataset_dict():
    """获取流体系统的原始数据集"""
    return get_datasets(use_normalization=False)


# 默认数据集（向后兼容）
dataset_dict = get_datasets(use_normalization=True)

