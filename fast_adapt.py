from copy import deepcopy
import torch
import torch.nn as nn
from datasets import dataset_dict, CURRENT_SYSTEM
from DSSDL_mask import MetaNet, load_dataset_with_one_hot, Trainer, config as BASE_CONFIG
from meta_train import CONFIG_OVERRIDES
FAST_ADAPT_CONFIG_OVERRIDES = CONFIG_OVERRIDES.copy()
FAST_ADAPT_CONFIG_OVERRIDES.update({
    'epochs': 200,
    'convergence_threshold': 1e-5,
    'patience': 100,
    'batch_size': -1,
    'meta_train_lr': 0.1,
    'fast_adapt_lr': 0.0001,
    'optimizer': 'lbfgs',
})

def _build_effective_config(config_overrides):
    """
    Build effective configuration by merging base config with overrides.
    
    Args:
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        Merged configuration dictionary
    """
    cfg = deepcopy(BASE_CONFIG)
    if config_overrides:
        # Update existing keys and add new ones
        cfg.update(config_overrides)
    return cfg

def fast_adapt(dataset_dict, config_overrides=None):
    cfg = _build_effective_config(config_overrides)
    device = cfg.get('device', 'cpu')
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']
    specific_param_dim = config_overrides.get('specific_param_dim', cfg.get('specific_param_dim'))
    support_loader, _ = load_dataset_with_one_hot(dataset_dict, split='support', batch_size=cfg.get('batch_size', 1), train_ratio=1.0, device=device, use_raw_data=cfg.get('use_raw_data', False))
    query_loader, _ = load_dataset_with_one_hot(dataset_dict, split='query', batch_size=cfg.get('batch_size', 1), train_ratio=1.0, device=device, use_raw_data=cfg.get('use_raw_data', False))
    model = MetaNet(input_dim=input_dim, output_dim=output_dim, one_hot_dim=one_hot_dim, specific_param_dim=specific_param_dim, hidden_widths=cfg['hidden_widths'], device=device, dataset_dict=dataset_dict, use_raw_data=cfg.get('use_raw_data', False))
    state_dict = torch.load("model/meta_train.pt", map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: unexpected keys in state_dict ignored: {unexpected_keys}")
    trainer = Trainer(model=model, learning_rate=cfg['meta_train_lr'], optimizer_type=cfg['optimizer'])
    result = trainer.fast_adapt(num_epochs=cfg['epochs'], train_data_loader=support_loader, test_data_loader=query_loader, convergence_threshold=cfg.get('convergence_threshold', 1e-5), patience=cfg.get('patience', 5))
    torch.save(model.state_dict(), "model/fast_adapt.pt")
    return result

def evaluate_query_without_fast_adapt(dataset_dict, config_overrides):
    cfg = _build_effective_config(config_overrides)
    device = cfg.get('device', 'cpu')
    specific_param_dim = config_overrides.get('specific_param_dim', cfg.get('specific_param_dim'))
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']
    query_loader, _ = load_dataset_with_one_hot(dataset_dict, split='query', batch_size=cfg.get('batch_size', 1), train_ratio=1.0, device=device, use_raw_data=cfg.get('use_raw_data', False))
    model = MetaNet(input_dim=input_dim, output_dim=output_dim, one_hot_dim=one_hot_dim, specific_param_dim=specific_param_dim, hidden_widths=cfg['hidden_widths'], device=device, dataset_dict=dataset_dict, use_raw_data=cfg.get('use_raw_data', False))
    state_dict = torch.load("model/meta_train.pt", map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: unexpected keys in state_dict ignored: {unexpected_keys}")
    model.eval()
    with torch.no_grad():
        mean_embedding = model.one_hot_to_specific_param.weight.mean(dim=1, keepdim=True)
        model.specific_param_generator.weight.copy_(mean_embedding)
    no_fast_adapt_path = "model/no_fast_adapt.pt"
    torch.save(model.state_dict(), no_fast_adapt_path)
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in query_loader:
            outputs_net, outputs_hat = model.fast_adapt_forward(features)
            total_loss += criterion(outputs_net + outputs_hat, labels).item()
    rmse = (total_loss / len(query_loader)) ** 0.5 if len(query_loader) > 0 else float('nan')
    mean_vector = mean_embedding.view(-1).detach().cpu().tolist()
    return rmse, mean_vector, no_fast_adapt_path

def main():
    print(f"Current system: {CURRENT_SYSTEM}")
    print("=" * 60)
    baseline_rmse, mean_slowparam, baseline_model_path = evaluate_query_without_fast_adapt(dataset_dict=dataset_dict, config_overrides=FAST_ADAPT_CONFIG_OVERRIDES)
    mean_str = ", ".join(f"{v:.6f}" for v in mean_slowparam)
    print(f"Default slowparam (mean from meta-trained one-hot): [{mean_str}]")
    print(f"Query RMSE without fast adapt: {baseline_rmse:.6f}")
    print(f"Saved no-fast-adapt model to: {baseline_model_path}")
    print("=" * 60)
    result = fast_adapt(dataset_dict=dataset_dict, config_overrides=FAST_ADAPT_CONFIG_OVERRIDES)
    print("\n" + "=" * 60)
    print("Fast adapt summary")
    print("=" * 60)
    max_epochs = FAST_ADAPT_CONFIG_OVERRIDES.get('epochs', result['total_epochs'])
    ran_epochs = result['total_epochs']
    converged_epoch = result['converged_epoch']
    print(f"System: {CURRENT_SYSTEM}")
    print(f"Max epochs (config): {max_epochs}")
    print(f"Ran epochs: {ran_epochs}")
    print(f"Converged epoch: {converged_epoch}")
    print(f"Training time: {result['training_time']:.2f}s")
    print(f"Final loss: {result['final_loss']:.6e}")
    if ran_epochs < max_epochs:
        print(f"Converged early, saved {max_epochs - ran_epochs} epochs (stopped after epoch {converged_epoch})")
    else:
        print("Ran to max epochs, no early convergence")
    print("=" * 60)

if __name__ == '__main__':
    main()