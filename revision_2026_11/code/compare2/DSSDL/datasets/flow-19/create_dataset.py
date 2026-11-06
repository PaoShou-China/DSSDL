#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow system dataset generation
Based on vorticity data from flow simulations
"""

import numpy as np
import os
import sys
from typing import List

# 定义dt参数，供其他模块导入
dt = 1

# 将父目录添加到系统路径中，以便导入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import clear_dataset_files

def create_dataset_from_flow_data(support_num_points=None):
    """
    从 'data_flow' 文件夹读取流体涡度数据，
    并将其划分为 train, support, 和 query 集。
    
    参数:
        support_num_points: int, 可选
            写入support集的数据点数量。如果为None，则写入全部数据点。
    """
    base_dir = current_dir
    source_data_dir = os.path.join(parent_dir, 'data_flow-19')

    # 清理旧文件
    print("Clearing existing dataset files for 'flow'...")
    clear_dataset_files("flow-19")

    # 定义目标目录
    train_dir = os.path.join(base_dir, "train")
    support_dir = os.path.join(base_dir, "support")
    query_dir = os.path.join(base_dir, "query")

    # 创建目录
    for d in [train_dir, support_dir, query_dir]:
        os.makedirs(os.path.join(d, 'features'), exist_ok=True)
        os.makedirs(os.path.join(d, 'labels'), exist_ok=True)

    # 定义雷诺数列表（从文件名推断）
    reynolds_numbers = [100, 125, 150, 175, 200, 225, 250, 275,
                         300, 325, 350, 
                         375]# 要同时修改 datasets/flow/__init__.py 中的 all_slow_params 和 slow_params_eval

    # 处理文件
    for re_num in reynolds_numbers:
        source_file_path = os.path.join(source_data_dir, f'vorticity_{re_num}.txt')

        # 检查源文件是否存在
        if not os.path.exists(source_file_path):
            print(f"Warning: Source file for Re={re_num} not found. Skipping.")
            continue

        # 读取数据
        data = np.loadtxt(source_file_path)
        
        # 计算labels（数值微分）
        # 使用numpy的gradient函数进行数值微分
        # labels = np.gradient(data, dt, axis=0, edge_order=2)
        
        # 使用欧拉(前向差分)进行数值微分
        labels = np.zeros_like(data)
        labels[:-1] = (data[1:] - data[:-1]) / dt
        labels[-1] = labels[-2]

        # 将雷诺数375分配给 support 和 query 集
        if re_num == 375:
            # 写入 support（可能只写入前N个点）
            dest_support_feature = os.path.join(support_dir, 'features', f'{re_num}.txt')
            dest_support_label = os.path.join(support_dir, 'labels', f'{re_num}.txt')
            
            if support_num_points is not None:
                # 只取前support_num_points个点
                features_subset = data[:support_num_points]
                labels_subset = labels[:support_num_points]
                
                np.savetxt(dest_support_feature, features_subset)
                np.savetxt(dest_support_label, labels_subset)
                
                print(f"Copied Re={re_num} to support set (first {len(features_subset)} points out of {len(data)} total).")
            else:
                # 复制全部数据
                np.savetxt(dest_support_feature, data)
                np.savetxt(dest_support_label, labels)
                print(f"Copied Re={re_num} to support set (all {len(data)} points).")

            # 写入 query（全部数据）
            dest_query_feature = os.path.join(query_dir, 'features', f'{re_num}.txt')
            dest_query_label = os.path.join(query_dir, 'labels', f'{re_num}.txt')
            np.savetxt(dest_query_feature, data)
            np.savetxt(dest_query_label, labels)
            print(f"Copied Re={re_num} to query set (all {len(data)} points).")
        
        # 其他文件分配给 train 集
        else:
            dest_train_feature = os.path.join(train_dir, 'features', f'{re_num}.txt')
            dest_train_label = os.path.join(train_dir, 'labels', f'{re_num}.txt')
            np.savetxt(dest_train_feature, data)
            np.savetxt(dest_train_label, labels)
            print(f"Copied Re={re_num} to train set ({len(data)} points).")

    print("\nDataset for 'flow' created successfully.")

if __name__ == '__main__':
    create_dataset_from_flow_data(1)

