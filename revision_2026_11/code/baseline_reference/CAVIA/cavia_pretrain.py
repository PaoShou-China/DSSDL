from datasets import dataset_dict, CURRENT_SYSTEM

from cavia_workflow import meta_train


CONFIG_OVERRIDES = {
    "num_context_params": 3,
    "hidden_sizes": [64, 64],
    "meta_lr": 1e-3,
    "inner_lr": 1e-4,
    "num_inner_updates": 5,
    "tasks_per_metaupdate": 4,
    "k_support": 10,
    "k_query": 10,
    "max_iters": 1000,
    "log_interval": 50,
    "first_order": True,
    "use_raw_data": False,
    "model_path": "model/cavia_pretrain.pt",
}


def main():
    print(f"Current system: {CURRENT_SYSTEM}")
    model_path, log = meta_train(dataset_dict, CONFIG_OVERRIDES)
    print(f"Saved pretrained CAVIA model to: {model_path}")
    print(f"Logged {len(log['train_loss'])} checkpoints")


if __name__ == "__main__":
    main()
