import os
import torch

from datasets import CURRENT_SYSTEM, dataset_dict
from cavia_workflow import finetune_contexts


CONFIG_OVERRIDES = {
    "num_context_params": 3,
    "hidden_sizes": [64, 64],
    "inner_lr": 1e-6,
    "num_inner_updates": 5,
    "use_raw_data": False,
    "model_path": "model/cavia_pretrain.pt",
    "finetune_max_epochs": 2000,
    "finetune_patience": 50,
    "finetune_convergence_threshold": 1e-10,
    "finetune_use_query": True,
}


def main():
    print(f"Current system: {CURRENT_SYSTEM}")
    contexts = finetune_contexts(dataset_dict, CONFIG_OVERRIDES)
    save_dir = os.path.join("results", "cavia", CURRENT_SYSTEM)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "context_params.pt")
    torch.save({"contexts": contexts}, save_path)
    print(f"Saved {len(contexts)} context vectors to: {save_path}")


if __name__ == "__main__":
    main()
