#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集通用工具函数
"""

import numpy as np
import os


def clear_dataset_files(system_name):
    """
    清除指定系统的所有数据集文件
    
    Args:
        system_name (str): 系统名称 (e.g., 'lorenz', 'double_pendulum', 'bistable_oscillator')
    """
    base_dir = f"datasets/{system_name}"
    dataset_types = ["train", "support", "query"]
    data_types = ["features", "labels"]
    
    print(f"清除 {system_name} 系统的现有数据集文件...")
    deleted_count = 0
    
    for dataset_type in dataset_types:
        for data_type in data_types:
            dir_path = os.path.join(base_dir, dataset_type, data_type)
            if os.path.exists(dir_path):
                # 删除目录中的所有.txt文件
                for file in os.listdir(dir_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(dir_path, file)
                        os.remove(file_path)
                        print(f"已删除: {file_path}")
                        deleted_count += 1
    
    if deleted_count == 0:
        print(f"没有找到需要删除的 {system_name} 数据集文件")
    else:
        print(f"共删除了 {deleted_count} 个 {system_name} 数据集文件")


def get_file_list(directory):
    """获取目录下的所有.txt文件列表"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(".txt")
    ])


def load_file_list(file_paths):
    """加载文件列表到numpy数组列表"""
    return [np.loadtxt(path, ndmin=2) for path in file_paths]


def calculate_normalization_params(data_list):
    """计算标准化参数（均值和标准差）"""
    all_data = np.concatenate(data_list, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    # 避免除零错误
    std[std == 0] = 1e-9
    return mean, std


def standardize(data, mean, std):
    """标准化数据"""
    return (data - mean) / std


def inverse_normalize_data(data, mean, std):
    """反标准化数据"""
    return data * std + mean


def get_datasets_generic(system_path, use_normalization=True):
    """
    通用数据集加载函数
    
    Args:
        system_path (str): 系统数据集路径 (e.g., 'datasets/pendulum')
        use_normalization (bool): 是否使用标准化
    
    Returns:
        dataset_dict: 标准格式的数据集字典
    """
    # 获取系统的dt参数
    system_dt = 0.01  # 默认值
    try:
        # 动态导入系统模块的dt参数
        system_name = os.path.basename(system_path)
        module = __import__(f'datasets.{system_name}', fromlist=['dt'])
        system_dt = getattr(module, 'dt', system_dt)
    except (ImportError, AttributeError):
        pass  # 使用默认值
    # 构建数据路径
    data_paths = {
        "train": {
            "features": get_file_list(os.path.join(system_path, "train", "features")),
            "labels": get_file_list(os.path.join(system_path, "train", "labels")),
        },
        "support": {
            "features": get_file_list(os.path.join(system_path, "support", "features")),
            "labels": get_file_list(os.path.join(system_path, "support", "labels")),
        },
        "query": {
            "features": get_file_list(os.path.join(system_path, "query", "features")),
            "labels": get_file_list(os.path.join(system_path, "query", "labels")),
        }
    }
    
    raw_datasets = {}
    normalized_datasets = {}
    sample_counts = {}
    file_counts = {}
    calculated_normalization = {}

    # 标准化参数占位符
    input_mean, input_std = None, None
    output_mean, output_std = None, None

    for split in ['train', 'support', 'query']:
        feat_paths = data_paths[split]['features']
        label_paths = data_paths[split]['labels']

        # 加载原始数据
        X_list = load_file_list(feat_paths)
        y_list = load_file_list(label_paths)

        # 从训练集计算标准化参数
        if split == 'train':
            input_mean, input_std = calculate_normalization_params(X_list)
            output_mean, output_std = calculate_normalization_params(y_list)
            calculated_normalization = {
                "input": {"mean": input_mean.tolist(), "std": input_std.tolist()},
                "output": {"mean": output_mean.tolist(), "std": output_std.tolist()}
            }

        # 确保标准化参数可用
        if input_mean is None or output_mean is None:
            raise RuntimeError("Normalization parameters not calculated. 'train' split must be processed first.")

        # 存储原始数据
        raw_datasets[split] = (X_list, y_list)
        
        # 创建标准化数据（如果需要）
        if use_normalization:
            X_norm_list = [standardize(X, input_mean, input_std) for X in X_list]
            y_norm_list = [standardize(y, output_mean, output_std) for y in y_list]
            normalized_datasets[split] = (X_norm_list, y_norm_list)

        # 统计信息
        total_samples = sum(X.shape[0] for X in X_list)
        assert len(feat_paths) == len(label_paths)
        file_counts[split] = len(feat_paths)
        sample_counts[split] = total_samples

    # 获取维度信息
    input_dim = X_list[0].shape[1] if X_list else 0
    output_dim = y_list[0].shape[1] if y_list else 0

    # 构建结果字典
    info = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'dt': system_dt,
        'normalization': calculated_normalization,
        'file_counts': file_counts,
        'sample_counts': sample_counts
    }

    result = {
        'raw': raw_datasets,
        'info': info
    }
    
    if use_normalization:
        result['normalized'] = normalized_datasets
        
    return result
