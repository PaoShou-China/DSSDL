from datasets import dataset_dict
from DSSDL_ import pretrain

# ==================== 配置参数（统一在这里修改，只需修改这一个地方） ====================
CONFIG_OVERRIDES = {
    # 预训练核心参数
    "specific_param_dim": 3,
    "epochs": 200,
    "reg_rate": 1.0,
    "use_raw_data": False,
    # 轨迹采样参数（如果use_trajectory_sampling=True时生效）
    "use_trajectory_sampling": False,  # 设为False使用传统的全数据训练
    "num_trajectories": 1,  # 每个系统采样5个轨迹片段
    "trajectory_duration": 10,  # 每个片段5秒
    # 覆盖DSSDL_.py中config的配置（可选，注释掉则使用DSSDL_.py中的默认值）
    "batch_size": 128,
    # 'batch_size': -1,
    # 'train_ratio': 0.8,
    "learning_rate": 0.1,
    "hidden_widths": [64, 64],
    # 'hidden_widths': [256, 256, 256],
    # 'weight_decay': 1e-4,
    "adam_lr": 0.01,  # 5e-4,
    "optimizer": "adam",  # 'lbfgs' or 'adam'
    # 'optimizer': 'lbfgs', # 优化器类型
    # 'use_cautious_adam': False,
    "device": "cuda:0",  # 或 'cpu'
}
# ====================================================================================


def main():
    """主函数：执行预训练"""
    # 调用预训练函数（所有参数都从CONFIG_OVERRIDES中读取）
    pretrain(
        dataset_dict=dataset_dict,
        config_overrides=CONFIG_OVERRIDES,
    )
    print("预训练完成！")


if __name__ == "__main__":
    main()
