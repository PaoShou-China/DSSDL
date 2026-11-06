#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集模块主入口
支持多种动力学系统的数据集管理
"""

import os
import importlib
from .utils import get_datasets_generic

# 配置：选择当前使用的系统
# 可选值: 'pendulum', 'lorenz', 或其他系统名称
# CURRENT_SYSTEM = 'pendulum'  # 默认使用摆系统
# CURRENT_SYSTEM = 'lorenz'
# CURRENT_SYSTEM = 'lorenz_exact'
# CURRENT_SYSTEM = 'bistable_oscillator'
# CURRENT_SYSTEM = 'double_pendulum'
# CURRENT_SYSTEM = 'van_der_pol'
# CURRENT_SYSTEM = 'pendulum_real'
# CURRENT_SYSTEM = 'flow'
# CURRENT_SYSTEM = 'flow-19'
# CURRENT_SYSTEM = 'quadrotor'
CURRENT_SYSTEM = "FHN"  # FitzHugh-Nagumo 系统


# CURRENT_SYSTEM = 'flow-19-v'
# CURRENT_SYSTEM = 'flow-29-v'
def get_system_module(system_name):
    """动态导入指定系统的模块"""
    try:
        module = importlib.import_module(f".{system_name}", package="datasets")
        return module
    except ImportError as e:
        raise ImportError(f"无法导入系统 '{system_name}': {e}")


def set_current_system(system_name):
    """设置当前使用的系统"""
    global CURRENT_SYSTEM, dataset_dict, f, dt, slow_params_eval

    CURRENT_SYSTEM = system_name
    system_module = get_system_module(system_name)

    # 重新加载系统特定的数据和函数
    dataset_dict = system_module.dataset_dict
    f = system_module.f
    dt = system_module.dt
    slow_params_eval = system_module.slow_params_eval

    print(f"已切换到系统: {system_name}")


def get_raw_dataset_dict():
    """获取当前系统的原始数据集"""
    system_module = get_system_module(CURRENT_SYSTEM)
    return system_module.get_raw_dataset_dict()


def get_datasets(use_normalization=True):
    """获取当前系统的数据集"""
    system_module = get_system_module(CURRENT_SYSTEM)
    return system_module.get_datasets(use_normalization)


def list_available_systems():
    """列出可用的系统"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    systems = []
    for item in os.listdir(current_dir):
        system_path = os.path.join(current_dir, item)
        if (
            os.path.isdir(system_path)
            and not item.startswith("__")
            and not item.startswith(".")
            and os.path.exists(os.path.join(system_path, "__init__.py"))
        ):
            systems.append(item)
    return systems


# 初始化默认系统
try:
    system_module = get_system_module(CURRENT_SYSTEM)
    dataset_dict = system_module.dataset_dict
    f = system_module.f
    dt = system_module.dt
    slow_params_eval = system_module.slow_params_eval

    # 可选：启用调试信息
    # print(f"数据集模块已加载，当前系统: {CURRENT_SYSTEM}")
    # print(f"可用系统: {list_available_systems()}")

except ImportError as e:
    print(f"警告: 无法加载默认系统 '{CURRENT_SYSTEM}': {e}")
    print(f"可用系统: {list_available_systems()}")
    # 设置默认值以避免错误
    dataset_dict = None
    f = None
    dt = 0.01
    slow_params_eval = []
