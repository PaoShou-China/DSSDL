import os
import shutil
import sys

# 将父目录添加到系统路径中，以便导入 utils
dt = 0.035
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import clear_dataset_files

def create_dataset_from_real_data(support_num_points=None):
    """
    从 'data_pendulum_real' 文件夹读取真实的单摆数据，
    并将其划分为 train, support, 和 query 集。
    
    参数:
        support_num_points: int, 可选
            写入support集的数据点数量。如果为None，则写入全部数据点。
    """
    base_dir = current_dir
    source_data_dir = os.path.join(parent_dir, 'data_pendulum_real')

    # 清理旧文件
    print("Clearing existing dataset files for 'pendulum_real'...")
    clear_dataset_files("pendulum_real")

    # 定义目标目录
    train_dir = os.path.join(base_dir, "train")
    support_dir = os.path.join(base_dir, "support")
    query_dir = os.path.join(base_dir, "query")

    # 创建目录
    for d in [train_dir, support_dir, query_dir]:
        os.makedirs(os.path.join(d, 'features'), exist_ok=True)
        os.makedirs(os.path.join(d, 'labels'), exist_ok=True)

    # 处理文件
    for i in range(1, 9):  # 文件从 [1] 到 [8]
        source_feature_path = os.path.join(source_data_dir, f'[{i}]_features.txt')
        source_label_path = os.path.join(source_data_dir, f'[{i}]_labels.txt')

        # 检查源文件是否存在
        if not os.path.exists(source_feature_path) or not os.path.exists(source_label_path):
            print(f"Warning: Source files for index {i} not found. Skipping.")
            continue

        # 将文件 [8] 分配给 support 和 query 集
        if i == 8:
            # 复制到 support（可能只复制前N个点）
            dest_support_feature = os.path.join(support_dir, 'features', f'{i:.2f}.txt')
            dest_support_label = os.path.join(support_dir, 'labels', f'{i:.2f}.txt')
            
            if support_num_points is not None:
                # 读取文件并只写入前N个点
                with open(source_feature_path, 'r') as f:
                    feature_lines = f.readlines()
                with open(source_label_path, 'r') as f:
                    label_lines = f.readlines()
                
                # 只取前support_num_points个点
                feature_lines_subset = feature_lines[:support_num_points]
                label_lines_subset = label_lines[:support_num_points]
                
                with open(dest_support_feature, 'w') as f:
                    f.writelines(feature_lines_subset)
                with open(dest_support_label, 'w') as f:
                    f.writelines(label_lines_subset)
                
                print(f"Copied index {i} to support set (first {len(feature_lines_subset)} points out of {len(feature_lines)} total).")
            else:
                # 复制全部数据
                shutil.copy(source_feature_path, dest_support_feature)
                shutil.copy(source_label_path, dest_support_label)
                print(f"Copied index {i} to support set (all points).")

            # 复制到 query（全部数据）
            dest_query_feature = os.path.join(query_dir, 'features', f'{i:.2f}.txt')
            dest_query_label = os.path.join(query_dir, 'labels', f'{i:.2f}.txt')
            shutil.copy(source_feature_path, dest_query_feature)
            shutil.copy(source_label_path, dest_query_label)
            print(f"Copied index {i} to query set (all points).")
        
        # 其他文件分配给 train 集
        else:
            dest_train_feature = os.path.join(train_dir, 'features', f'{i:.2f}.txt')
            dest_train_label = os.path.join(train_dir, 'labels', f'{i:.2f}.txt')
            shutil.copy(source_feature_path, dest_train_feature)
            shutil.copy(source_label_path, dest_train_label)
            print(f"Copied index {i} to train set.")

    print("\nDataset for 'pendulum_real' created successfully.")

if __name__ == '__main__':
    create_dataset_from_real_data(50)

