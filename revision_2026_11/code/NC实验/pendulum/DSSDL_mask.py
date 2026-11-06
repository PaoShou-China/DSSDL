import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import csv
import math
from torch.utils.data import TensorDataset, DataLoader, random_split

"""
DSSDL_mask.py - Deep Sparse Slow Dynamics Learning with Masking
Main module containing the MetaNet model and training utilities for sparse system identification.
"""

# Configuration parameters for the model and training
config = {
    'specific_param_dim': 3,          # Dimension of slow/specific parameters
    'hidden_widths': [128, 128],       # Hidden layer widths for the neural network backbone
    'optimizer': 'adam',               # Optimizer type ('adam' or 'lbfgs')
    'fast_adapt_lr': 5e-3,             # Learning rate for fast adaptation (Adam)
    'meta_train_lr': 0.1,              # Learning rate for meta-training (LBFGS)
    'use_cautious_adam': False,        # Whether to use cautious Adam optimizer
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # Device to use
    'use_raw_data': False,             # Whether to use raw (unnormalized) data
    'use_trajectory_sampling': True,   # Whether to use trajectory-based sampling
    'num_trajectories': 5,             # Number of trajectories to sample per system
    'trajectory_duration': 10,         # Duration of each sampled trajectory (seconds)
}

class CautiousAdam(Adam):
    """
    Cautious Adam optimizer - a variant of Adam that adjusts learning rate based on gradient direction.
    
    This optimizer uses a cautious factor that slows down updates when the gradient direction
    changes, potentially improving stability in noisy optimization landscapes.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(CautiousAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                update = exp_avg / denom
                cautious_mask = (update * grad > 0).to(grad.dtype)
                eps_cautious = 1e-8
                cautious_factor = cautious_mask / (cautious_mask.mean() + eps_cautious)
                p.data.add_(update * cautious_factor, alpha=-step_size)
        return loss

def get_adam_optimizer(params, lr=1e-3, weight_decay=0):
    """
    Get Adam optimizer based on configuration.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay rate
        
    Returns:
        Optimizer instance (either standard Adam or CautiousAdam)
    """
    if config.get('use_cautious_adam', False):
        return CautiousAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

from datasets import CURRENT_SYSTEM

def f_hat(x, dataset_dict=None, use_raw_data=False):
    """
    Placeholder function for the unknown dynamics model.
    
    Args:
        x: Input state
        dataset_dict: Dataset dictionary
        use_raw_data: Whether to use raw data
        
    Returns:
        Zero tensor matching input shape (placeholder implementation)
    """
    from meta_train import CONFIG_OVERRIDES
    if CONFIG_OVERRIDES.get('reg_rate', 0.0) == 0.0:
        return  0.0
    return 0.0

os.makedirs("model", exist_ok=True)

def load_dataset_with_trajectory_sampling(dataset_dict, split='train', batch_size=128, train_ratio=0.7, device='cpu', use_raw_data=False, num_trajectories=5, trajectory_duration=1.0, dt=None):
    """
    Load dataset with trajectory-based sampling.
    
    Args:
        dataset_dict: Dataset dictionary containing the data
        split: Dataset split ('train', 'support', or 'query')
        batch_size: Batch size
        train_ratio: Train/validation split ratio
        device: Device to use
        use_raw_data: Whether to use raw (unnormalized) data
        num_trajectories: Number of trajectories to sample per system
        trajectory_duration: Duration of each sampled trajectory (seconds)
        dt: Time step size
        
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    if dt is None:
        dt = dataset_dict['info']['dt']
    if use_raw_data:
        X_list, y_list = dataset_dict['raw'][split]
    else:
        X_list, y_list = dataset_dict['normalized'][split]
    one_hot_dim = len(X_list)
    steps_per_trajectory = int(trajectory_duration / dt)
    feature_tensors = []
    label_tensors = []
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        total_steps = X_tensor.shape[0]
        if total_steps < steps_per_trajectory:
            print(f"Warning: System {i} has insufficient data ({total_steps} < {steps_per_trajectory}). Using all available data.")
            sampled_indices = torch.arange(total_steps, device=device)
        else:
            max_start_idx = total_steps - steps_per_trajectory
            start_indices = torch.randint(0, max_start_idx + 1, (num_trajectories,), device=device)
            sampled_indices = []
            for start_idx in start_indices:
                segment_indices = torch.arange(start_idx, start_idx + steps_per_trajectory, device=device)
                sampled_indices.append(segment_indices)
            sampled_indices = torch.cat(sampled_indices)
        feature_segment = X_tensor[sampled_indices]
        label_segment = y_tensor[sampled_indices]
        one_hot = torch.zeros((feature_segment.shape[0], one_hot_dim), device=device)
        one_hot[:, i] = 1.0
        feature_tensors.append(torch.cat((feature_segment, one_hot), dim=1))
        label_tensors.append(label_segment)
    feature_tensor_all = torch.cat(feature_tensors, dim=0)
    label_tensor_all = torch.cat(label_tensors, dim=0)
    dataset = TensorDataset(feature_tensor_all, label_tensor_all)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    eval_size = total_size - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(0))
    train_bs = max(1, len(train_dataset) if batch_size == -1 else batch_size)
    eval_bs = max(1, len(eval_dataset) if batch_size == -1 else batch_size)
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_bs, shuffle=False)
    print(f"Trajectory sampling: {num_trajectories} segments × {trajectory_duration}s = {len(feature_tensor_all)} total samples")
    return train_loader, eval_loader

def load_dataset_with_one_hot(dataset_dict, split='train', batch_size=128, train_ratio=0.7, device='cpu', use_raw_data=False):
    """
    Load dataset with one-hot encoding for system identification.
    
    Args:
        dataset_dict: Dataset dictionary containing the data
        split: Dataset split ('train', 'support', or 'query')
        batch_size: Batch size
        train_ratio: Train/validation split ratio
        device: Device to use
        use_raw_data: Whether to use raw (unnormalized) data
        
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    if use_raw_data:
        X_list, y_list = dataset_dict['raw'][split]
    else:
        X_list, y_list = dataset_dict['normalized'][split]
    one_hot_dim = len(X_list)
    feature_tensors = []
    label_tensors = []
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        feature_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        label_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        one_hot = torch.zeros((feature_tensor.shape[0], one_hot_dim), device=device)
        one_hot[:, i] = 1.0
        feature_tensors.append(torch.cat((feature_tensor, one_hot), dim=1))
        label_tensors.append(label_tensor)
    feature_tensor_all = torch.cat(feature_tensors, dim=0)
    label_tensor_all = torch.cat(label_tensors, dim=0)
    dataset = TensorDataset(feature_tensor_all, label_tensor_all)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    eval_size = total_size - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(0))
    train_bs = max(1, len(train_dataset) if batch_size == -1 else batch_size)
    eval_bs = max(1, len(eval_dataset) if batch_size == -1 else batch_size)
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_bs, shuffle=False)
    return train_loader, eval_loader

class Trainer:
    """
    Trainer class for meta-learning and fast adaptation.
    
    This class handles the training process for the MetaNet model, including:
    - Meta-training with sparsity regularization
    - Fast adaptation to new systems
    - Model evaluation and performance tracking
    """
    def __init__(self, model, learning_rate=None, optimizer_type=None):
        """
        Initialize the Trainer.
        
        Args:
            model: MetaNet model to train
            learning_rate: Learning rate for optimization
            optimizer_type: Optimizer type ('adam' or 'lbfgs')
        """
        self.model = model
        self.learning_rate = learning_rate or config['meta_train_lr']
        self.optimizer_type = optimizer_type or config['optimizer']
        self.training_log = []

    def evaluate_model(self, test_data_loader):
        test_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for data, labels in test_data_loader:
                try:
                    outputs_net, outputs_hat = self.model.forward(data)
                except:
                    outputs_net, outputs_hat = self.model.fast_adapt_forward(data)
                test_loss += criterion(outputs_net + outputs_hat, labels).item()
        return (test_loss / len(test_data_loader)) ** 0.5

    def _collect_mask_stats(self):
        if not hasattr(self.model, 'continuous_mask'):
            return None
        with torch.no_grad():
            continuous_mask = torch.sigmoid(self.model.continuous_mask)
            binary_mask = (continuous_mask > self.model.mask_threshold).float()
            return {
                'mask_mean': continuous_mask.mean().item(),
                'mask_sparsity': 1.0 - binary_mask.mean().item(),
                'mask_values': continuous_mask.detach().cpu().tolist(),
            }

    def search_specific_param_dim(self, num_epochs, train_data_loader, test_data_loader):
        original_f_hat = self.model.f_hat_func
        self.model.f_hat_func = lambda x, dataset_dict=None, use_raw_data=False: 0.0
        criterion = nn.MSELoss()
        self.training_log.clear()
        if self.optimizer_type == 'lbfgs':
            return self._search_specific_param_dim_lbfgs(num_epochs, train_data_loader, test_data_loader, criterion)
        elif self.optimizer_type == 'adam':
            return self._search_specific_param_dim_adam(num_epochs, train_data_loader, test_data_loader, criterion)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def _search_specific_param_dim_lbfgs(self, num_epochs, train_data_loader, test_data_loader, criterion):
        for epoch in range(num_epochs):
            optimizer_mapper = torch.optim.LBFGS(self.model.one_hot_to_specific_param.parameters(), lr=self.learning_rate * 3)
            optimizer_all = torch.optim.LBFGS(self.model.parameters(), lr=self.learning_rate)
            self.training_log.append({
                'epoch': epoch,
                'loss': self.evaluate_model(test_data_loader),
                'specific_parameters': self.model.specific_param_dim
            })
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            for features, labels in train_data_loader:
                def closure_mapper():
                    optimizer_mapper.zero_grad()
                    outputs_net = self.model.search_specific_param_dim_forward(features)
                    loss = criterion(outputs_net, labels)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.one_hot_to_specific_param.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                optimizer_mapper.step(closure_mapper)
                def closure_all():
                    optimizer_all.zero_grad()
                    outputs_net = self.model.search_specific_param_dim_forward(features)
                    loss = criterion(outputs_net, labels)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                loss = optimizer_all.step(closure_all)
                progress_bar.set_description(f'Epoch {epoch}: Loss {loss.item() ** 0.5:.6f} Slow param dim {self.model.specific_param_dim}')
                progress_bar.update(1)
            progress_bar.close()
        return self.training_log
    
    def _search_specific_param_dim_adam(self, num_epochs, train_data_loader, test_data_loader, criterion):
        optimizer_mapper = get_adam_optimizer(self.model.one_hot_to_specific_param.parameters(), lr=config['fast_adapt_lr'])
        optimizer_all = get_adam_optimizer(self.model.parameters(), lr=config['fast_adapt_lr'], weight_decay=config.get('weight_decay', 0))
        for epoch in range(num_epochs):
            self.training_log.append({
                'epoch': epoch,
                'loss': self.evaluate_model(test_data_loader),
                'specific_parameters': self.model.specific_param_dim
            })
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_loss = 0
            for features, labels in train_data_loader:
                optimizer_mapper.zero_grad()
                outputs_net = self.model.search_specific_param_dim_forward(features)
                loss = criterion(outputs_net, labels)
                loss.backward()
                optimizer_mapper.step()
                optimizer_all.zero_grad()
                outputs_net = self.model.search_specific_param_dim_forward(features)
                loss = criterion(outputs_net, labels)
                loss.backward()
                optimizer_all.step()
                epoch_loss += loss.item()
                progress_bar.set_description(f'Epoch {epoch}: Loss {(epoch_loss/(progress_bar.n+1)) ** 0.5:.6f} Slow param dim {self.model.specific_param_dim}')
                progress_bar.update(1)
            progress_bar.close()
        return self.training_log

    def meta_train(self, num_epochs, train_data_loader, test_data_loader, reg_rate):
        class MyLoss(nn.Module):
            def __init__(self, reg_rate):
                super(MyLoss, self).__init__()
                self.reg_rate = reg_rate
                self.mask_reg = config.get('mask_reg', 0.0)
            def forward(self, outputs_net, outputs_hat, labels, mask=None):
                mse_loss = torch.mean((outputs_net + outputs_hat - labels) ** 2)
                reg = torch.mean((outputs_hat - labels) ** 2)
                if mask is None or self.mask_reg <= 0:
                    return mse_loss + self.reg_rate * reg
                mask_sparsity = self.mask_reg * torch.sum(torch.sigmoid(mask))
                return mse_loss + self.reg_rate * reg + mask_sparsity
        criterion = MyLoss(reg_rate)
        self.training_log.clear()
        if self.optimizer_type == 'lbfgs':
            return self._meta_train_lbfgs(num_epochs, train_data_loader, test_data_loader, criterion)
        elif self.optimizer_type == 'adam':
            return self._meta_train_adam(num_epochs, train_data_loader, test_data_loader, criterion)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def _meta_train_lbfgs(self, num_epochs, train_data_loader, test_data_loader, criterion):
        for epoch in range(num_epochs):
            optimizer_mapper = torch.optim.LBFGS(self.model.one_hot_to_specific_param.parameters(), lr=self.learning_rate)
            optimizer_all = torch.optim.LBFGS(self.model.parameters(), lr=self.learning_rate)
            test_loss = self.evaluate_model(test_data_loader)
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_train_loss = 0
            batch_count = 0
            for features, labels in train_data_loader:
                def closure_mapper():
                    optimizer_mapper.zero_grad()
                    outputs_net, outputs_hat = self.model.forward(features)
                    loss = criterion(outputs_net, outputs_hat, labels, self.model.continuous_mask)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.one_hot_to_specific_param.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                optimizer_mapper.step(closure_mapper)
                def closure_all():
                    optimizer_all.zero_grad()
                    outputs_net, outputs_hat = self.model.forward(features)
                    loss = criterion(outputs_net, outputs_hat, labels, self.model.continuous_mask)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                loss = optimizer_all.step(closure_all)
                epoch_train_loss += loss.item()
                batch_count += 1
                progress_bar.set_description(f'Epoch {epoch}: Loss {loss.item() ** 0.5:.6f}')
                progress_bar.update(1)
            progress_bar.close()
            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
            log_entry = {'epoch': epoch, 'train_loss': avg_train_loss, 'test_loss': test_loss}
            mask_stats = self._collect_mask_stats()
            if mask_stats is not None:
                log_entry.update(mask_stats)
                print(f"Mask stats - mean: {mask_stats['mask_mean']:.4f}, sparsity: {mask_stats['mask_sparsity']:.3f}")
            self.training_log.append(log_entry)
        return self.training_log
    
    def _meta_train_adam(self, num_epochs, train_data_loader, test_data_loader, criterion):
        optimizer_mapper = get_adam_optimizer(self.model.one_hot_to_specific_param.parameters(), lr=config['fast_adapt_lr'])
        optimizer_all = get_adam_optimizer(self.model.parameters(), lr=config['fast_adapt_lr'], weight_decay=config.get('weight_decay', 0))
        for epoch in range(num_epochs):
            test_loss = self.evaluate_model(test_data_loader)
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_train_loss = 0
            batch_count = 0
            for features, labels in train_data_loader:
                optimizer_mapper.zero_grad()
                outputs_net, outputs_hat = self.model.forward(features)
                loss = criterion(outputs_net, outputs_hat, labels, self.model.continuous_mask)
                loss.backward()
                optimizer_mapper.step()
                optimizer_all.zero_grad()
                outputs_net, outputs_hat = self.model.forward(features)
                loss = criterion(outputs_net, outputs_hat, labels, self.model.continuous_mask)
                loss.backward()
                optimizer_all.step()
                epoch_train_loss += loss.item()
                batch_count += 1
                progress_bar.set_description(f'Epoch {epoch}: Loss {(epoch_train_loss/batch_count) ** 0.5:.6f}')
                progress_bar.update(1)
            progress_bar.close()
            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
            log_entry = {'epoch': epoch, 'train_loss': avg_train_loss, 'test_loss': test_loss}
            mask_stats = self._collect_mask_stats()
            if mask_stats is not None:
                log_entry.update(mask_stats)
                print(f"Mask stats - mean: {mask_stats['mask_mean']:.4f}, sparsity: {mask_stats['mask_sparsity']:.3f}")
            self.training_log.append(log_entry)
        return self.training_log

    def fast_adapt(self, num_epochs, train_data_loader, test_data_loader, convergence_threshold=1e-6, patience=10):
        import time
        self.model.set_fast_adapt()
        criterion = nn.MSELoss()
        self.training_log.clear()
        start_time = time.time()
        if self.optimizer_type == 'lbfgs':
            log, converged_epoch = self._fast_adapt_lbfgs(num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold, patience)
        elif self.optimizer_type == 'adam':
            log, converged_epoch = self._fast_adapt_adam(num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold, patience)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        end_time = time.time()
        training_time = end_time - start_time
        losses = [entry['loss'] for entry in log]
        np.save('model/fast_adapt_loss.npy', np.array(losses))
        print("Fast adapt loss history saved to model/fast_adapt_loss.npy")
        return {'log': log, 'converged_epoch': converged_epoch, 'total_epochs': len(log), 'training_time': training_time, 'final_loss': log[-1]['loss'] if log else None}

    def train_from_scratch(self, num_epochs, train_data_loader, test_data_loader):
        criterion = nn.MSELoss()
        self.training_log.clear()
        optimizer = get_adam_optimizer(self.model.parameters(), lr=config['fast_adapt_lr'], weight_decay=config.get('weight_decay', 0))
        for epoch in range(num_epochs):
            test_loss = self.evaluate_model(test_data_loader)
            self.training_log.append({'epoch': epoch, 'loss': test_loss})
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch} (Scratch)')
            epoch_loss = 0
            for features, labels in train_data_loader:
                optimizer.zero_grad()
                outputs_net, outputs_hat = self.model.fast_adapt_forward(features)
                loss = criterion(outputs_net + outputs_hat, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_description(f'Epoch {epoch} (Scratch): Loss {epoch_loss/(progress_bar.n+1):.6f}')
                progress_bar.update(1)
            progress_bar.close()
        losses = [entry['loss'] for entry in self.training_log]
        np.save('model/scratch_loss.npy', np.array(losses))
        print("Train from scratch loss history saved to model/scratch_loss.npy")
        return self.training_log
    
    def _fast_adapt_lbfgs(self, num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold=1e-6, patience=10):
        optimizer = torch.optim.LBFGS(self.model.specific_param_generator.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        patience_counter = 0
        converged_epoch = num_epochs
        for epoch in range(num_epochs):
            current_loss = self.evaluate_model(test_data_loader)
            self.training_log.append({'epoch': epoch, 'loss': current_loss})
            if epoch > 0:
                loss_change = abs(best_loss - current_loss)
                if loss_change < convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        converged_epoch = epoch
                        print(f"\n早停：在第 {epoch} 轮收敛 (损失变化 < {convergence_threshold})")
                        break
                else:
                    patience_counter = 0
            if current_loss < best_loss:
                best_loss = current_loss
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            for features, labels in train_data_loader:
                def closure():
                    optimizer.zero_grad()
                    outputs_net, outputs_hat = self.model.fast_adapt_forward(features)
                    loss = criterion(outputs_net + outputs_hat, labels)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.specific_param_generator.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                loss = optimizer.step(closure)
                progress_bar.set_description(f'Epoch {epoch}: Loss {loss.item():.6f}')
                progress_bar.update(1)
            progress_bar.close()
        return self.training_log, converged_epoch
    
    def _fast_adapt_adam(self, num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold=1e-6, patience=10):
        optimizer = get_adam_optimizer(self.model.specific_param_generator.parameters(), lr=config['fast_adapt_lr'], weight_decay=config.get('weight_decay', 0))
        best_loss = float('inf')
        patience_counter = 0
        converged_epoch = num_epochs
        for epoch in range(num_epochs):
            current_loss = self.evaluate_model(test_data_loader)
            self.training_log.append({'epoch': epoch, 'loss': current_loss})
            if epoch > 0:
                loss_change = abs(best_loss - current_loss)
                if loss_change < convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        converged_epoch = epoch
                        print(f"\n早停：在第 {epoch} 轮收敛 (损失变化 < {convergence_threshold})")
                        break
                else:
                    patience_counter = 0
            if current_loss < best_loss:
                best_loss = current_loss
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_loss = 0
            for features, labels in train_data_loader:
                optimizer.zero_grad()
                outputs_net, outputs_hat = self.model.fast_adapt_forward(features)
                loss = criterion(outputs_net + outputs_hat, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_description(f'Epoch {epoch}: Loss {epoch_loss/(progress_bar.n+1):.6f}')
                progress_bar.update(1)
            progress_bar.close()
        return self.training_log, converged_epoch

    def save_model(self, filename='metanet.pt'):
        """
        Save the model state dictionary.
        
        Args:
            filename: Filename to save the model (without directory prefix)
        """
        # Ensure the model directory exists
        os.makedirs("model", exist_ok=True)
        # Extract just the filename if path is provided
        filename = os.path.basename(filename)
        torch.save(self.model.state_dict(), os.path.join("model", filename))

    def load_model(self, filename='metanet.pt'):
        state_dict = torch.load(os.path.join("model", filename), weights_only=True)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            print(f"Warning: unexpected keys in state_dict ignored: {unexpected_keys}")

class MetaNet(nn.Module):
    """
    MetaNet model for sparse system identification with meta-learning.
    
    This model consists of:
    - A one-hot to specific parameter mapping layer
    - A continuous mask for sparse parameter selection
    - A neural network backbone for dynamics prediction
    - Support for both meta-training and fast adaptation
    """
    def __init__(self, input_dim, output_dim, one_hot_dim, specific_param_dim, hidden_widths, device, dataset_dict=None, use_raw_data=False):
        """
        Initialize the MetaNet model.
        
        Args:
            input_dim: Input dimension (state variables)
            output_dim: Output dimension (derivatives)
            one_hot_dim: One-hot encoding dimension (number of training systems)
            specific_param_dim: Dimension of specific (slow) parameters
            hidden_widths: List of hidden layer widths for the backbone
            device: Device to use
            dataset_dict: Dataset dictionary
            use_raw_data: Whether to use raw (unnormalized) data
        """
        super(MetaNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.one_hot_dim = one_hot_dim
        self.specific_param_dim = specific_param_dim
        self.device = device
        self.dataset_dict = dataset_dict
        self.use_raw_data = use_raw_data
        self.one_hot_to_specific_param = nn.Linear(one_hot_dim, specific_param_dim, bias=False, device=device)
        self.specific_param_generator = nn.Linear(1, specific_param_dim, bias=False, device=device)
        self.continuous_mask = nn.Parameter(torch.full((specific_param_dim,), config.get('mask_init_value', 0.5), device=device, requires_grad=True))
        self.mask_threshold = config.get('mask_threshold', 0.5)
        self.use_mask = True
        # Build neural network backbone
        layers = [nn.Linear(input_dim + specific_param_dim, hidden_widths[0]), nn.SiLU()]
        for i in range(len(hidden_widths) - 1):
            layers.append(nn.Linear(hidden_widths[i], hidden_widths[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_widths[-1], output_dim))
        self.backbone = nn.Sequential(*layers).to(device)
        self.f_hat_func = f_hat

    def get_mask(self, hard=False):
        """
        Get the mask for specific parameters.
        
        Args:
            hard: Whether to return a hard (binary) mask
            
        Returns:
            Mask tensor
        """
        continuous_mask = torch.sigmoid(self.continuous_mask)
        if hard:
            binary_mask = (continuous_mask > self.mask_threshold).float()
            return binary_mask + continuous_mask - continuous_mask.detach()
        return continuous_mask

    def forward(self, inputs):
        """
        Forward pass for meta-training.
        
        Args:
            inputs: Input tensor containing state and one-hot encoding
            
        Returns:
            Tuple of (output_net, output_hat)
        """
        x, one_hot = inputs[:, :self.input_dim], inputs[:, self.input_dim:]
        specific_params = self.one_hot_to_specific_param(one_hot)
        if self.use_mask:
            mask = self.get_mask(hard=True)
            specific_params = specific_params * mask
        x_with_specific_params = torch.cat([x, specific_params], dim=1)
        output_net = self.backbone(x_with_specific_params)
        output_hat = self.f_hat_func(x_with_specific_params, self.dataset_dict, self.use_raw_data)
        return output_net, output_hat

    def search_specific_param_dim_forward(self, inputs):
        """
        Forward pass for specific parameter dimension search.
        
        Args:
            inputs: Input tensor containing state and one-hot encoding
            
        Returns:
            Network output
        """
        x, one_hot = inputs[:, :self.input_dim], inputs[:, self.input_dim:]
        specific_params = self.one_hot_to_specific_param(one_hot)
        if self.use_mask:
            mask = self.get_mask(hard=True)
            specific_params = specific_params * mask
        x_with_specific_params = torch.cat([x, specific_params], dim=1)
        return self.backbone(x_with_specific_params)

    def fast_adapt_forward(self, inputs):
        """
        Forward pass for fast adaptation.
        
        Args:
            inputs: Input tensor containing state
            
        Returns:
            Tuple of (output_net, output_hat)
        """
        x = inputs[:, :self.input_dim]
        specific_params = self.specific_param_generator(torch.ones_like(x[:, [0]], device=self.device))
        if self.use_mask:
            mask = self.get_mask(hard=True)
            specific_params = specific_params * mask
        x_with_specific_params = torch.cat([x, specific_params], dim=1)
        output_net = self.backbone(x_with_specific_params)
        output_hat = self.f_hat_func(x_with_specific_params, self.dataset_dict, self.use_raw_data)
        return output_net, output_hat

    def set_fast_adapt(self):
        """
        Set the model for fast adaptation by freezing backbone parameters.
        """
        for param in self.backbone.parameters():
            param.requires_grad_(False)

def search_specific_param_dim(min_dim, max_dim, step, epochs, dataset_dict, optimizer_type=None):
    logger = []
    train_loader, eval_loader = load_dataset_with_one_hot(dataset_dict, split='train', batch_size=config["batch_size"], train_ratio=config["train_ratio"], device=config["device"])
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']
    for dim in range(min_dim, max_dim, step):
        print(f"Testing slow parameter dimension: {dim}")
        meta_net = MetaNet(input_dim=input_dim, output_dim=output_dim, one_hot_dim=one_hot_dim, specific_param_dim=dim, hidden_widths=config['hidden_widths'], device=config["device"], dataset_dict=dataset_dict, use_raw_data=False)
        trainer = Trainer(model=meta_net, learning_rate=config["meta_train_lr"], optimizer_type=optimizer_type)
        log_entry = trainer.search_specific_param_dim(epochs, train_loader, eval_loader)[-1]
        logger.append(log_entry)
        trainer.save_model("meta_train.pt")
    return logger

def plot_and_save_loss_curve(training_log, dataset_name, save_dir="pictures"):
    if not training_log:
        print("警告：训练日志为空，无法绘制损失图")
        return
    save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    epochs = [log['epoch'] for log in training_log]
    if 'train_loss' in training_log[0] and 'test_loss' in training_log[0]:
        train_losses = [log['train_loss'] for log in training_log]
        test_losses = [log['test_loss'] for log in training_log]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{dataset_name} - Meta Training Loss Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    else:
        losses = [log['loss'] for log in training_log]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-', label='Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{dataset_name} - Meta Training Loss Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    filename = os.path.join(save_path, 'meta_train_loss_curve.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失下降图已保存到: {filename}")
