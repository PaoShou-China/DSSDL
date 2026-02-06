"""
meta_train.py - Meta-training script for sparse system identification.

This script handles the meta-training process of the MetaNet model
on multiple systems, learning shared representations and sparse masks.
"""

from datasets import dataset_dict, CURRENT_SYSTEM
from DSSDL_mask import Trainer, MetaNet, load_dataset_with_one_hot, load_dataset_with_trajectory_sampling, config, plot_and_save_loss_curve
import numpy as np
import os

# Configuration overrides for meta-training
CONFIG_OVERRIDES = {
    'specific_param_dim': 6,          # Dimension of slow/specific parameters
    'epochs': 100,                    # Number of meta-training epochs
    'reg_rate': 0.0,                  # Regularization rate for meta-learning
    'use_raw_data': False,            # Whether to use raw (unnormalized) data
    'use_trajectory_sampling': False, # Whether to use trajectory-based sampling
    'num_trajectories': 1,            # Number of trajectories to sample per system
    'trajectory_duration': 10,        # Duration of each sampled trajectory (seconds)
    'batch_size': 128,                # Batch size
    'meta_train_lr': 0.01,             # Learning rate for meta-training (LBFGS)
    'hidden_widths': [64, 64],        # Hidden layer widths for the neural network
    'fast_adapt_lr': 0.001,           # Learning rate for fast adaptation (Adam)
    'optimizer': 'adam',              # Optimizer type ('adam' or 'lbfgs')
    'device': 'cuda:0',               # Device to use
}

def meta_train(dataset_dict, config_overrides=None):
    """
    Perform meta-training on multiple systems.
    
    Args:
        dataset_dict: Dataset dictionary containing the data
        config_overrides: Configuration overrides
        
    Returns:
        None
    """
    current_config = config.copy()
    if config_overrides:
        current_config.update(config_overrides)
    
    # Extract dataset information
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']
    dt = dataset_dict['info']['dt']
    
    # Load dataset with appropriate sampling method
    if current_config.get('use_trajectory_sampling', False):
        train_loader, eval_loader = load_dataset_with_trajectory_sampling(
            dataset_dict,
            split='train',
            batch_size=current_config['batch_size'],
            train_ratio=current_config.get('train_ratio', 0.8),
            device=current_config['device'],
            use_raw_data=current_config.get('use_raw_data', False),
            num_trajectories=current_config.get('num_trajectories', 5),
            trajectory_duration=current_config.get('trajectory_duration', 1.0),
            dt=dt
        )
    else:
        train_loader, eval_loader = load_dataset_with_one_hot(
            dataset_dict,
            split='train',
            batch_size=current_config['batch_size'],
            train_ratio=current_config.get('train_ratio', 0.8),
            device=current_config['device'],
            use_raw_data=current_config.get('use_raw_data', False)
        )
    
    # Initialize model and trainer
    model = MetaNet(
        input_dim=input_dim,
        output_dim=output_dim,
        one_hot_dim=one_hot_dim,
        specific_param_dim=current_config['specific_param_dim'],
        hidden_widths=current_config['hidden_widths'],
        device=current_config['device'],
        dataset_dict=dataset_dict,
        use_raw_data=current_config.get('use_raw_data', False)
    )
    trainer = Trainer(
        model=model,
        learning_rate=current_config['meta_train_lr'],
        optimizer_type=current_config['optimizer']
    )
    
    # Perform meta-training
    training_log = trainer.meta_train(
        num_epochs=current_config['epochs'],
        train_data_loader=train_loader,
        test_data_loader=eval_loader,
        reg_rate=current_config['reg_rate']
    )
    
    # Save model and results
    trainer.save_model("meta_train.pt")
    # Use CURRENT_SYSTEM as system name since dataset_dict doesn't have 'system_name' key
    plot_and_save_loss_curve(training_log, CURRENT_SYSTEM)
    # Ensure the directory exists
    os.makedirs('final_data/meta_train', exist_ok=True)
    train_losses = [entry.get('train_loss', entry.get('loss', 0)) for entry in training_log]
    test_losses = [entry.get('test_loss', entry.get('loss', 0)) for entry in training_log]
    loss_data = np.array([train_losses, test_losses]).T
    np.save('final_data/meta_train/meta_train_loss.npy', loss_data)
    print("预训练损失已保存到 final_data/meta_train/meta_train_loss.npy")

def main():
    """
    Main function for meta-training.
    """
    meta_train(
        dataset_dict=dataset_dict,
        config_overrides=CONFIG_OVERRIDES,
    )
    print("元训练完成！")

if __name__ == '__main__':
    main()