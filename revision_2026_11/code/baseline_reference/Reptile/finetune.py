import torch
import os
import json
from Reptile import Reptile, config
from datasets import dataset_dict

# Override inner loop LR for finetune
config["inner_lr"] = 0.001


# 系统名称（可以根据实际情况修改）
system_name = "Reptile"

# 创建结果目录
result_dir = os.path.join(config["result_path"], system_name)
os.makedirs(result_dir, exist_ok=True)

reptile_finetune = Reptile(dataset_dict)

# Load the pretrained model state dictionary
loaded_state_dict = torch.load(
    "pretrain.pt", map_location=reptile_finetune.device
)  # Ensure loading to correct device
reptile_finetune.network.load_state_dict(loaded_state_dict)

# Perform finetuning and get metrics
metrics = reptile_finetune.finetune()

# 准备保存的指标数据（包含中英文名称）
metrics_to_save = {
    "converged_epoch": {
        "value": metrics["converged_epoch"],
        "description_zh": "收敛轮数",
        "description_en": "Converged Epoch",
        "unit": "epochs",
    },
    "total_training_time": {
        "value": metrics["total_training_time"],
        "description_zh": "训练总耗时",
        "description_en": "Total Training Time",
        "unit": "seconds",
    },
    "final_loss": {
        "value": metrics["final_loss"],
        "description_zh": "最终损失 (RMSE)",
        "description_en": "Final Loss (RMSE)",
        "unit": "RMSE",
    },
    "convergence_threshold": {
        "value": config["convergence_threshold"],
        "description_zh": "收敛阈值",
        "description_en": "Convergence Threshold",
    },
    "patience": {
        "value": config["patience"],
        "description_zh": "耐心值",
        "description_en": "Patience",
    },
}

# 保存指标
metrics_file = os.path.join(result_dir, "finetune_metrics.json")
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)

print("\n=== Finetune Metrics ===")
print(f"收敛轮数 (Converged Epoch): {metrics['converged_epoch']}")
print(f"训练总耗时 (Total Training Time): {metrics['total_training_time']:.4f} 秒")
print(f"最终损失 (Final Loss): {metrics['final_loss']:.6f}")
print(f"\n指标已保存到: {metrics_file}")

# Save the fine-tuned model
reptile_finetune.save("finetune.pt")
