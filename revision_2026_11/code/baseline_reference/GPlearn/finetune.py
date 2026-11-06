import os
import json
from gplearn_compare import GPlearnRegressor
from datasets import dataset_dict

# 系统名称
system_name = "GPlearn"

# 创建结果目录
result_dir = os.path.join('results', system_name)
os.makedirs(result_dir, exist_ok=True)

regressor = GPlearnRegressor(dataset_dict=dataset_dict)

# Finetune the loaded model and get metrics
metrics = regressor.finetune()

# 准备保存的指标数据
metrics_to_save = {
    'final_loss': {
        'value': float(metrics['final_loss']),
        'description_zh': '最终损失 (RMSE)',
        'description_en': 'Final Loss (RMSE)',
        'unit': 'RMSE'
    }
}

# 保存指标
metrics_file = os.path.join(result_dir, 'finetune_metrics.json')
with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)

print("\n=== Finetune Metrics ===")
print(f"最终损失 (Final Loss): {metrics['final_loss']:.6f}")
print(f"\n指标已保存到: {metrics_file}")

# Save the finetuned model
regressor.save()
