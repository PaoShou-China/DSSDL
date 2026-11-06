import os
import pickle
import csv
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import random

config = {
    'specific_param_dim': 3,
    'hidden_widths': [128, 128],

    # Training settings
    'learning_rate': 0.1,
    'batch_size': 32,
    # 'batch_size': -1,
    'reg_rate': 1.0,
    'weight_decay': 1e-4,
    'train_ratio': 0.8,
    
    # Optimizer settings
    'optimizer': 'adam',  # 'lbfgs' or 'adam'
    # 'optimizer': 'lbfgs',  # 'lbfgs' or 'adam'
    'adam_lr': 5e-3,  # Learning rate for Adam optimizer
    # 'use_cautious_adam': True,  # Enable cautious Adam optimizer (paper: single-line modification)
    'use_cautious_adam': False,
    # Device
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'use_raw_data': False,
    'use_trajectory_sampling': True,
    'num_trajectories': 5,
    'trajectory_duration': 10,
}

def set_global_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(22222)
# ---------------------------
# Cautious Adam Optimizer (Paper Implementation)
# ---------------------------

class CautiousAdam(Adam):
    """
    Cautious Adam optimizer implementation based on the paper's single-line modification.
    
    The key modification: m = (u * g > 0).to(g.dtype)
    where u is the update direction and g is the gradient.
    Only apply updates when they align with the gradient direction.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(CautiousAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
    
    def step(self, closure=None):
        """Performs a single optimization step with cautious modification."""
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

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        # Maintains max of all exp_avg_sq till now
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

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                
                # Calculate the standard Adam update
                update = exp_avg / denom
                
                # CAUTIOUS MODIFICATION: Only apply updates that align with gradient direction
                # m = (u * g > 0).to(g.dtype)
                # p.add_(u * m/(m.mean()+eps), alpha=-lr)
                cautious_mask = (update * grad > 0).to(grad.dtype)
                eps_cautious = 1e-8
                cautious_factor = cautious_mask / (cautious_mask.mean() + eps_cautious)
                
                # Apply the cautious update
                p.data.add_(update * cautious_factor, alpha=-step_size)

        return loss


def get_adam_optimizer(params, lr=1e-3, weight_decay=0):
    """
    根据配置选择标准Adam或CautiousAdam优化器
    """
    if config.get('use_cautious_adam', False):
        return CautiousAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

from datasets import CURRENT_SYSTEM
def f_hat(x, dataset_dict=None, use_raw_data=False):
    """
    物理先验函数 - Lorenz系统方程
    
    Args:
        x: 输入特征 [x1, x2, x3, p1, p2, p3]
        dataset_dict: 数据集字典，包含标准化参数
        use_raw_data: 是否使用原始数据（不需要反标准化）
    """
    from pretrain import CONFIG_OVERRIDES
    if CONFIG_OVERRIDES.get('reg_rate', 0.0) == 0.0:
        return  0.0
    elif CURRENT_SYSTEM == "lorenz":
        
    #---------------------------------------------
        if use_raw_data or dataset_dict is None:
            # 使用原始数据，直接计算
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            p1, p2, p3 = x[:, 3], x[:, 4], x[:, 5]
        else:
            # 使用标准化数据，需要先反标准化输入
            norm_info = dataset_dict['info']['normalization']
            input_mean = torch.tensor(norm_info['input']['mean'], device=x.device, dtype=x.dtype)
            input_std = torch.tensor(norm_info['input']['std'], device=x.device, dtype=x.dtype)
            
            # 只反标准化前3维状态变量，参数维度保持不变
            # 前3维是状态变量，需要反标准化
            states = x[:, :3]  # [x1, x2, x3]
            params = x[:, 3:]  # [p1, p2, p3] - 这些是学习的参数，不需要反标准化
            
            # 反标准化状态变量
            states_denorm = states * input_std + input_mean
            x1, x2, x3 = states_denorm[:, 0], states_denorm[:, 1], states_denorm[:, 2]
            
            p1, p2, p3 = params[:, 0], params[:, 1], params[:, 2]
        
        # 计算Lorenz系统方程（原始尺度）
        x1_dot = p1 * (x2 - x1)
        x2_dot = x1 * (p2 - x3) - x2
        x3_dot = x1 * x2 - p3 * x3
        labels = torch.stack([x1_dot, x2_dot, x3_dot], dim=1)
#---------------------------------------------
    # norm_info = dataset_dict['info']['normalization']
    # input_mean = torch.tensor(norm_info['input']['mean'], device=x.device, dtype=x.dtype)
    # input_std = torch.tensor(norm_info['input']['std'], device=x.device, dtype=x.dtype)
    
    # # 只反标准化前3维状态变量，参数维度保持不变
    # # 前3维是状态变量，需要反标准化
    # states = x[:, :2]  # [x1, x2, x3]
    # params = x[:, 2:]  # [p1, p2, p3] - 这些是学习的参数，不需要反标准化
    
    # # 反标准化状态变量
    # states_denorm = states * input_std + input_mean
    # x1, x2 = states_denorm[:, 0], states_denorm[:, 1]
    
    # p1, p2, p3 = params[:, 0], params[:, 1], params[:, 2]

    # x_ddot = -(p1 * x1 + p2 * (x2 ** 3))
    # output_mean = torch.tensor(norm_info['output']['mean'], device=x.device, dtype=x.dtype)
    # output_std = torch.tensor(norm_info['output']['std'], device=x.device, dtype=x.dtype)
    # labels = torch.stack([x_ddot, x1], dim=1)
    # # 标准化输出
    # labels_norm = (labels - output_mean) / output_std
    # return labels_norm
    #--------------------------------------------- bistable oscillator符号化部分
        if use_raw_data or dataset_dict is None:
            # 使用原始数据，直接返回
            return labels
        else:
            # 使用标准化数据，需要标准化输出
            output_mean = torch.tensor(norm_info['output']['mean'], device=x.device, dtype=x.dtype)
            output_std = torch.tensor(norm_info['output']['std'], device=x.device, dtype=x.dtype)
            
            # 标准化输出
            labels_norm = (labels - output_mean) / output_std
            return labels_norm

    elif CURRENT_SYSTEM in ("FHN", "fhn"):
        try:
            from datasets.FHN.create_dataset import EPSILON as FHN_EPSILON
        except Exception:
            FHN_EPSILON = 0.20

        if use_raw_data or dataset_dict is None:
            # 浣跨敤鍘熷鏁版嵁锛岀洿鎺ヨ绠?
            v, w = x[:, 0], x[:, 1]
            a, b, i_ext = x[:, 2], x[:, 3], x[:, 4]
        else:
            # 浣跨敤鏍囧噯鍖栨暟鎹紝闇€瑕佸厛鍙嶆爣鍑嗗寲杈撳叆
            norm_info = dataset_dict['info']['normalization']
            input_mean = torch.tensor(norm_info['input']['mean'], device=x.device, dtype=x.dtype)
            input_std = torch.tensor(norm_info['input']['std'], device=x.device, dtype=x.dtype)

            # 鍙弽鏍囧噯鍖栧墠2缁寸姸鎬佸彉閲忥紝鍙傛暟缁村害淇濇寔涓嶅彉
            states = x[:, :2]  # [v, w]
            params = x[:, 2:]  # [a, b, i_ext]

            # 鍙嶆爣鍑嗗寲鐘舵€佸彉閲?
            states_denorm = states * input_std + input_mean
            v, w = states_denorm[:, 0], states_denorm[:, 1]

            a, b, i_ext = params[:, 0], params[:, 1], params[:, 2]

        # 璁＄畻FHN绯荤粺鏂圭▼锛堝師濮嬪昂搴︼級
        v_dot = v - (v ** 3) / 3.0 - w + i_ext
        w_dot = FHN_EPSILON * (v + a - b * w)
        labels = torch.stack([v_dot, w_dot], dim=1)

        if use_raw_data or dataset_dict is None:
            return labels
        else:
            # 鏍囧噯鍖栬緭鍑?
            output_mean = torch.tensor(norm_info['output']['mean'], device=x.device, dtype=x.dtype)
            output_std = torch.tensor(norm_info['output']['std'], device=x.device, dtype=x.dtype)
            labels_norm = (labels - output_mean) / output_std
            return labels_norm

    elif CURRENT_SYSTEM == "flow":
        # 动态加载所有pysr模型
        model_paths = sorted(glob(f'model/{CURRENT_SYSTEM}/pysr_model_dx*.pkl'))
        predictions = []

        # 准备反标准化（只对前7维）
        if not use_raw_data and dataset_dict is not None:
            norm_info = dataset_dict['info']['normalization']
            input_mean = torch.tensor(norm_info['input']['mean'], device=x.device, dtype=x.dtype)
            input_std = torch.tensor(norm_info['input']['std'], device=x.device, dtype=x.dtype)
            
            # 只反标准化前7维状态变量
            states = x[:, :7]  # 前7维状态变量
            other_dims = x[:, 7:]  # 其他维度（如果有的话）
            
            # 反标准化前7维
            states_denorm = states * input_std + input_mean
            
            # 合并反标准化后的状态变量和其他维度
            if other_dims.shape[1] > 0:
                x_denorm = torch.cat([states_denorm, other_dims], dim=1)
            else:
                x_denorm = states_denorm
            
            x_for_predict = x_denorm.detach().cpu().numpy()
        else:
            x_for_predict = x.detach().cpu().numpy()

        for model_path in model_paths:
            with open(model_path, 'rb') as f:
                pysr_model = pickle.load(f)
            # 预测时使用反标准化后的数据
            pred = pysr_model.predict(x_for_predict)
            predictions.append(torch.from_numpy(pred).to(x.device, dtype=x.dtype))
        
        labels = torch.stack(predictions, dim=1)

        if use_raw_data or dataset_dict is None:
            return labels
        else:
            # 标准化输出
            output_mean = torch.tensor(norm_info['output']['mean'], device=x.device, dtype=x.dtype)
            output_std = torch.tensor(norm_info['output']['std'], device=x.device, dtype=x.dtype)
            labels_norm = (labels - output_mean) / output_std
            return labels_norm
    # x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    # p1, p2, p3 = x[:, 3], x[:, 4], x[:, 5]
    # x1_dot = p1 * (x2 - x1)
    # x2_dot = x1 * (p2 - x3) - x2
    # x3_dot = x1 * x2 - p3 * x3
    # labels = torch.stack([x1_dot, x2_dot, x3_dot], dim=1)    
    # return labels  # 调试原始版本


os.makedirs("model", exist_ok=True)


# ---------------------------
# Trainer Class
# ---------------------------

from torch.utils.data import TensorDataset, DataLoader, random_split


def load_dataset_with_trajectory_sampling(dataset_dict, split='train', batch_size=128, train_ratio=0.7, device='cpu', use_raw_data=False, 
                                        num_trajectories=5, trajectory_duration=1.0, dt=None):
    """
    Loads training data with trajectory segment sampling instead of using all data.
    
    Args:
        dataset_dict (dict): Dataset dictionary returned by get_datasets().
        split (str): Which split.
        batch_size (int): Batch size for the data loaders.
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0).
        device (str): Device to move tensors to ('cpu' or 'cuda').
        use_raw_data (bool): If True, use raw data instead of normalized data.
        num_trajectories (int): Number of trajectory segments to sample per system.
        trajectory_duration (float): Duration of each trajectory segment in seconds.
        dt (float): Time step used in data generation. If None, will use dt from dataset_dict.

    Returns:
        tuple: (train_loader, eval_loader)
    """
    # Get dt from dataset_dict if not provided
    if dt is None:
        dt = dataset_dict['info']['dt']
    
    # Choose data source based on use_raw_data flag
    if use_raw_data:
        X_list, y_list = dataset_dict['raw'][split]
    else:
        X_list, y_list = dataset_dict['normalized'][split]

    one_hot_dim = len(X_list)
    steps_per_trajectory = int(trajectory_duration / dt)

    feature_tensors = []
    label_tensors = []

    for i, (X, y) in enumerate(zip(X_list, y_list)):
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        total_steps = X_tensor.shape[0]
        
        # Ensure we have enough data for sampling
        if total_steps < steps_per_trajectory:
            print(f"Warning: System {i} has insufficient data ({total_steps} < {steps_per_trajectory}). Using all available data.")
            sampled_indices = torch.arange(total_steps, device=device)
        else:
            # Randomly sample starting positions for trajectory segments
            max_start_idx = total_steps - steps_per_trajectory
            start_indices = torch.randint(0, max_start_idx + 1, (num_trajectories,), device=device)
            
            # Create index ranges for each trajectory segment
            sampled_indices = []
            for start_idx in start_indices:
                segment_indices = torch.arange(start_idx, start_idx + steps_per_trajectory, device=device)
                sampled_indices.append(segment_indices)
            sampled_indices = torch.cat(sampled_indices)

        # Sample the trajectory segments
        feature_segment = X_tensor[sampled_indices]
        label_segment = y_tensor[sampled_indices]

        # Create one-hot encoding for this system
        one_hot = torch.zeros((feature_segment.shape[0], one_hot_dim), device=device)
        one_hot[:, i] = 1.0

        # Combine features with one-hot encoding
        feature_tensors.append(torch.cat((feature_segment, one_hot), dim=1))
        label_tensors.append(label_segment)

    # Concatenate all sampled segments
    feature_tensor_all = torch.cat(feature_tensors, dim=0)
    label_tensor_all = torch.cat(label_tensors, dim=0)

    # Create dataset and split
    dataset = TensorDataset(feature_tensor_all, label_tensor_all)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    eval_size = total_size - train_size

    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(0)
    )

    # Determine batch sizes
    train_bs = max(1, len(train_dataset) if batch_size == -1 else batch_size)
    eval_bs = max(1, len(eval_dataset) if batch_size == -1 else batch_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_bs, shuffle=False)

    print(f"Trajectory sampling: {num_trajectories} segments × {trajectory_duration}s = {len(feature_tensor_all)} total samples")
    
    return train_loader, eval_loader


def load_dataset_with_one_hot(dataset_dict, split='train', batch_size=128, train_ratio=0.7, device='cpu', use_raw_data=False):
    """
    Loads training data from dataset_dict with one-hot encoding and splits into train and evaluation datasets.

    Args:
        dataset_dict (dict): Dataset dictionary returned by get_datasets().
        split (str): Which split.
        batch_size (int): Batch size for the data loaders.
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0).
        device (str): Device to move tensors to ('cpu' or 'cuda').
        use_raw_data (bool): If True, use raw data instead of normalized data.

    Returns:
        tuple: (train_loader, eval_loader)
    """
    # Choose data source based on use_raw_data flag
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

    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(0)
    )

    train_bs = max(1, len(train_dataset) if batch_size == -1 else batch_size)
    eval_bs = max(1, len(eval_dataset) if batch_size == -1 else batch_size)

    # Create DataLoaders with the determined batch sizes
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_bs, shuffle=False)

    return train_loader, eval_loader


class Trainer:
    def __init__(self, model, learning_rate=None, optimizer_type=None):
        """
        Initializes the Trainer object.

        Args:
            model (MetaNet): An instance of the MetaNet model.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_type (str): Type of optimizer ('lbfgs' or 'adam').
        """
        self.model = model
        self.learning_rate = learning_rate or config['learning_rate']
        self.optimizer_type = optimizer_type or config['optimizer']
        self.training_log = []

    def evaluate_model(self, test_data_loader):
        """
        Evaluates the model on the test dataset.

        Args:
            test_data_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            float: Root mean squared error (RMSE) on the test dataset.
        """
        test_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for data, labels in test_data_loader:
                try:
                    outputs_net, outputs_hat = self.model.forward(data)
                except:
                    outputs_net, outputs_hat = self.model.finetune_forward(data)
                test_loss += criterion(outputs_net + outputs_hat, labels).item()
        return (test_loss / len(test_data_loader)) ** 0.5

    def search_specific_param_dim(self, num_epochs, train_data_loader, test_data_loader):
        """
        Performs the search slow parameters phase of the model.

        Args:
            num_epochs (int): Number of epochs for pretraining.
            train_data_loader (DataLoader): DataLoader for the training dataset.
            test_data_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            list: Training log containing training progress.
        """
        # 在搜索阶段暂时禁用物理先验
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
        """LBFGS implementation for search_specific_param_dim."""
        for epoch in range(num_epochs):
            # Use LBFGS optimizer
            optimizer_mapper = torch.optim.LBFGS(self.model.one_hot_to_specific_param.parameters(), lr=self.learning_rate * 3)
            optimizer_all = torch.optim.LBFGS(self.model.parameters(), lr=self.learning_rate)
            self.training_log.append({
                'epoch': epoch,
                'loss': self.evaluate_model(test_data_loader),
                'specific_parameters': self.model.specific_param_dim
            })
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            for features, labels in train_data_loader:
                # LBFGS requires a closure to re-evaluate the model and loss
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

                progress_bar.set_description(
                    f'Epoch {epoch}: Loss {loss.item() ** 0.5:.6f} Slow param dim {self.model.specific_param_dim}')
                progress_bar.update(1)
            progress_bar.close()
        return self.training_log
    
    def _search_specific_param_dim_adam(self, num_epochs, train_data_loader, test_data_loader, criterion):
        """Adam implementation for search_specific_param_dim."""
        # Use Adam optimizer (or Cautious Adam based on config)
        optimizer_mapper = get_adam_optimizer(self.model.one_hot_to_specific_param.parameters(), lr=config['adam_lr'])
        optimizer_all = get_adam_optimizer(self.model.parameters(), lr=config['adam_lr'], weight_decay=config.get('weight_decay', 0))
        
        for epoch in range(num_epochs):
            self.training_log.append({
                'epoch': epoch,
                'loss': self.evaluate_model(test_data_loader),
                'specific_parameters': self.model.specific_param_dim
            })
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_loss = 0
            
            for features, labels in train_data_loader:
                # First optimize mapper
                optimizer_mapper.zero_grad()
                outputs_net = self.model.search_specific_param_dim_forward(features)
                loss = criterion(outputs_net, labels)
                loss.backward()
                optimizer_mapper.step()
                
                # Then optimize all parameters
                optimizer_all.zero_grad()
                outputs_net = self.model.search_specific_param_dim_forward(features)
                loss = criterion(outputs_net, labels)
                loss.backward()
                optimizer_all.step()
                
                epoch_loss += loss.item()
                progress_bar.set_description(
                    f'Epoch {epoch}: Loss {(epoch_loss/(progress_bar.n+1)) ** 0.5:.6f} Slow param dim {self.model.specific_param_dim}')
                progress_bar.update(1)
            progress_bar.close()
        return self.training_log

    def pretrain(self, num_epochs, train_data_loader, test_data_loader, reg_rate):
        """
        Performs the pretraining phase of the model.

        Args:
            num_epochs (int): Number of epochs for pretraining.
            train_data_loader (DataLoader): DataLoader for the training dataset.
            test_data_loader (DataLoader): DataLoader for the test dataset.
            reg_rate (float): Regulation rate for symbolic.

        Returns:
            list: Training log containing training progress.
        """

        class MyLoss(nn.Module):
            def __init__(self, reg_rate):
                super(MyLoss, self).__init__()
                self.reg_rate = reg_rate

            # def forward(self, outputs_net, outputs_hat, labels):
            #     mse_loss = torch.mean(torch.abs(outputs_net + outputs_hat - labels))
            #     reg = torch.mean(torch.abs(outputs_hat - labels))
            #     total_loss = mse_loss + self.reg_rate * reg
            #     return total_loss
            def forward(self, outputs_net, outputs_hat, labels):
                mse_loss = torch.mean((outputs_net + outputs_hat - labels) ** 2)
                reg = torch.mean((outputs_hat - labels) ** 2)
                total_loss = mse_loss + self.reg_rate * reg
                return total_loss
        criterion = MyLoss(reg_rate)
        self.training_log.clear()
        
        if self.optimizer_type == 'lbfgs':
            return self._pretrain_lbfgs(num_epochs, train_data_loader, test_data_loader, criterion)
        elif self.optimizer_type == 'adam':
            return self._pretrain_adam(num_epochs, train_data_loader, test_data_loader, criterion)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def _pretrain_lbfgs(self, num_epochs, train_data_loader, test_data_loader, criterion):
        """LBFGS implementation for pretrain."""
        start_time = time.time()
        for epoch in range(num_epochs):
            # Use LBFGS optimizer
            # optimizer_mapper = torch.optim.LBFGS(self.model.one_hot_to_specific_param.parameters(), lr=self.learning_rate * 3)
            optimizer_mapper = torch.optim.LBFGS(self.model.one_hot_to_specific_param.parameters(), lr=self.learning_rate)
            optimizer_all = torch.optim.LBFGS(self.model.parameters(), lr=self.learning_rate)
            
            # Evaluate test loss at the beginning of each epoch
            test_loss = self.evaluate_model(test_data_loader)
            
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_train_loss = 0
            batch_count = 0
            
            for features, labels in train_data_loader:
                # LBFGS requires a closure to re-evaluate the model and loss
                def closure_mapper():
                    optimizer_mapper.zero_grad()
                    outputs_net, outputs_hat = self.model.forward(features)
                    loss = criterion(outputs_net, outputs_hat, labels)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.one_hot_to_specific_param.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                optimizer_mapper.step(closure_mapper)

                def closure_all():
                    optimizer_all.zero_grad()
                    outputs_net, outputs_hat = self.model.forward(features)
                    loss = criterion(outputs_net, outputs_hat, labels)
                    if config.get('weight_decay', 0.0) > 0:
                        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                        loss = loss + config['weight_decay'] * l2_reg
                    loss.backward()
                    return loss.detach()
                loss = optimizer_all.step(closure_all)

                epoch_train_loss += loss.item()
                batch_count += 1
                progress_bar.set_description(
                    f'Epoch {epoch}: Loss {loss.item() ** 0.5:.6f}')
                progress_bar.update(1)
            progress_bar.close()
            
            # Record both train and test loss
            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
            self.training_log.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'elapsed_seconds': time.time() - start_time,
            })
        return self.training_log
    
    def _pretrain_adam(self, num_epochs, train_data_loader, test_data_loader, criterion):
        """Adam implementation for pretrain."""
        # Use Adam optimizer (or Cautious Adam based on config)
        optimizer_mapper = get_adam_optimizer(self.model.one_hot_to_specific_param.parameters(), lr=config['adam_lr'])
        optimizer_all = get_adam_optimizer(self.model.parameters(), lr=config['adam_lr'], weight_decay=config.get('weight_decay', 0))
        
        start_time = time.time()
        for epoch in range(num_epochs):
            # Evaluate test loss at the beginning of each epoch
            test_loss = self.evaluate_model(test_data_loader)
            
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_train_loss = 0
            batch_count = 0
            
            for features, labels in train_data_loader:
                # First optimize mapper
                optimizer_mapper.zero_grad()
                outputs_net, outputs_hat = self.model.forward(features)
                loss = criterion(outputs_net, outputs_hat, labels)
                loss.backward()
                optimizer_mapper.step()
                
                # Then optimize all parameters
                optimizer_all.zero_grad()
                outputs_net, outputs_hat = self.model.forward(features)
                loss = criterion(outputs_net, outputs_hat, labels)
                loss.backward()
                optimizer_all.step()
                
                epoch_train_loss += loss.item()
                batch_count += 1
                progress_bar.set_description(
                    f'Epoch {epoch}: Loss {(epoch_train_loss/batch_count) ** 0.5:.6f}')
                progress_bar.update(1)
            progress_bar.close()
            
            # Record both train and test loss
            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
            self.training_log.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'elapsed_seconds': time.time() - start_time,
            })
        return self.training_log

    def finetune(self, num_epochs, train_data_loader, test_data_loader, convergence_threshold=1e-6, patience=10):
        """
        Performs the fine-tuning phase of the model.

        Args:
            num_epochs (int): Number of epochs for fine-tuning.
            train_data_loader (DataLoader): DataLoader for the training dataset.
            test_data_loader (DataLoader): DataLoader for the test dataset.
            convergence_threshold (float): Loss change threshold for early stopping.
            patience (int): Number of epochs to wait before early stopping.

        Returns:
            dict: Training log and convergence info containing training progress.
        """
        import time
        
        self.model.set_finetune()
        criterion = nn.MSELoss()
        self.training_log.clear()
        
        # 记录开始时间
        start_time = time.time()
        
        if self.optimizer_type == 'lbfgs':
            log, converged_epoch = self._finetune_lbfgs(num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold, patience)
        elif self.optimizer_type == 'adam':
            log, converged_epoch = self._finetune_adam(num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold, patience)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        # 记录结束时间
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save loss history
        losses = [entry['loss'] for entry in log]
        np.save('model/finetune_loss.npy', np.array(losses))
        print("Finetune loss history saved to model/finetune_loss.npy")
        
        # 返回包含收敛信息的字典
        return {
            'log': log,
            'converged_epoch': converged_epoch,
            'total_epochs': len(log),
            'training_time': training_time,
            'final_loss': log[-1]['loss'] if log else None
        }

    def train_from_scratch(self, num_epochs, train_data_loader, test_data_loader):
        """
        Performs training from scratch on the same data as finetuning.
        """
        # DO NOT call self.model.set_finetune() to ensure all params are trained
        criterion = nn.MSELoss()
        self.training_log.clear()

        # Use Adam for training from scratch as it's generally more robust
        optimizer = get_adam_optimizer(self.model.parameters(), 
                                   lr=config['adam_lr'], weight_decay=config.get('weight_decay', 0))

        for epoch in range(num_epochs):
            test_loss = self.evaluate_model(test_data_loader)
            self.training_log.append({
                'epoch': epoch,
                'loss': test_loss,
            })
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch} (Scratch)')
            epoch_loss = 0
            
            for features, labels in train_data_loader:
                optimizer.zero_grad()
                # Use finetune_forward as it uses the specific_param_generator for a single system
                outputs_net, outputs_hat = self.model.finetune_forward(features)
                loss = criterion(outputs_net + outputs_hat, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_description(f'Epoch {epoch} (Scratch): Loss {epoch_loss/(progress_bar.n+1):.6f}')
                progress_bar.update(1)
            progress_bar.close()
        
        # Save loss history
        losses = [entry['loss'] for entry in self.training_log]
        np.save('model/scratch_loss.npy', np.array(losses))
        print("Train from scratch loss history saved to model/scratch_loss.npy")
        return self.training_log
    
    def _finetune_lbfgs(self, num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold=1e-6, patience=10):
        """LBFGS implementation for finetune with early stopping."""
        # Use LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.model.specific_param_generator.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        converged_epoch = num_epochs  # 默认为最大轮数
        start_time = time.time()
        
        for epoch in range(num_epochs):
            current_loss = self.evaluate_model(test_data_loader)
            
            # 检查是否收敛
            if epoch > 0:
                loss_change = abs(best_loss - current_loss)
                if loss_change < convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        converged_epoch = epoch
                        print(f"\n早停：在第 {epoch} 轮收敛 (损失变化 < {convergence_threshold})")
                        self.training_log.append({
                            'epoch': epoch,
                            'loss': current_loss,
                            'elapsed_seconds': time.time() - start_time,
                        })
                        break
                else:
                    patience_counter = 0
            
            if current_loss < best_loss:
                best_loss = current_loss
            
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            for features, labels in train_data_loader:
                # LBFGS requires a closure to re-evaluate the model and loss
                def closure():
                    optimizer.zero_grad()
                    outputs_net, outputs_hat = self.model.finetune_forward(features)
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
            
            self.training_log.append({
                'epoch': epoch,
                'loss': current_loss,
                'elapsed_seconds': time.time() - start_time,
            })
        return self.training_log, converged_epoch
    
    def _finetune_adam(self, num_epochs, train_data_loader, test_data_loader, criterion, convergence_threshold=1e-6, patience=10):
        """Adam implementation for finetune with early stopping."""
        # Use Adam optimizer (or Cautious Adam based on config)
        optimizer = get_adam_optimizer(self.model.specific_param_generator.parameters(), 
                                   lr=config['adam_lr'], weight_decay=config.get('weight_decay', 0))
        
        best_loss = float('inf')
        patience_counter = 0
        converged_epoch = num_epochs  # 默认为最大轮数
        start_time = time.time()
        
        for epoch in range(num_epochs):
            current_loss = self.evaluate_model(test_data_loader)
            
            # 检查是否收敛
            if epoch > 0:
                loss_change = abs(best_loss - current_loss)
                if loss_change < convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        converged_epoch = epoch
                        print(f"\n早停：在第 {epoch} 轮收敛 (损失变化 < {convergence_threshold})")
                        self.training_log.append({
                            'epoch': epoch,
                            'loss': current_loss,
                            'elapsed_seconds': time.time() - start_time,
                        })
                        break
                else:
                    patience_counter = 0
            
            if current_loss < best_loss:
                best_loss = current_loss
            
            progress_bar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch}')
            epoch_loss = 0
            
            for features, labels in train_data_loader:
                optimizer.zero_grad()
                outputs_net, outputs_hat = self.model.finetune_forward(features)
                loss = criterion(outputs_net + outputs_hat, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_description(f'Epoch {epoch}: Loss {epoch_loss/(progress_bar.n+1):.6f}')
                progress_bar.update(1)
            progress_bar.close()
            
            self.training_log.append({
                'epoch': epoch,
                'loss': current_loss,
                'elapsed_seconds': time.time() - start_time,
            })
        return self.training_log, converged_epoch

    def save_model(self, filename='metanet.pt'):
        """Saves the state dictionary of the model."""
        torch.save(self.model.state_dict(), os.path.join("model", filename))

    def load_model(self, filename='metanet.pt'):
        """Loads the state dictionary into the model."""
        self.model.load_state_dict(torch.load(os.path.join("model", filename), weights_only=True))




# ---------------------------
# MetaNet Model Definition
# ---------------------------

class MetaNet(nn.Module):
    def __init__(self, input_dim, output_dim, one_hot_dim, specific_param_dim, hidden_widths, device, dataset_dict=None, use_raw_data=False):
        """
        Initializes the MetaNet model.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output predictions.
            one_hot_dim (int): Dimensionality of the one-hot encoded features.
            specific_param_dim (int): Dimensionality of the slow parameters.
            hidden_widths (list of int): Widths of the hidden layers in the backbone network.
            device (str): Device on which the model will be trained.
            dataset_dict (dict): Dataset dictionary containing normalization parameters.
            use_raw_data (bool): Whether using raw data (no normalization needed).
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

        layers = [nn.Linear(input_dim + specific_param_dim, hidden_widths[0]), nn.SiLU()]
        for i in range(len(hidden_widths) - 1):
            layers.append(nn.Linear(hidden_widths[i], hidden_widths[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_widths[-1], output_dim))
        self.backbone = nn.Sequential(*layers).to(device)
        self.f_hat_func = f_hat

    def forward(self, inputs):
        """
        Forward pass of the MetaNet model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_dim + one_hot_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x, one_hot = inputs[:, :self.input_dim], inputs[:, self.input_dim:]
        specific_params = self.one_hot_to_specific_param(one_hot)
        x_with_specific_params = torch.cat([x, specific_params], dim=1)
        output_net = self.backbone(x_with_specific_params)
        # output_net = 0
        output_hat = self.f_hat_func(x_with_specific_params, self.dataset_dict, self.use_raw_data)
        return output_net, output_hat

    def search_specific_param_dim_forward(self, inputs):
        x, one_hot = inputs[:, :self.input_dim], inputs[:, self.input_dim:]
        specific_params = self.one_hot_to_specific_param(one_hot)
        x_with_specific_params = torch.cat([x, specific_params], dim=1)
        return self.backbone(x_with_specific_params)

    def finetune_forward(self, inputs):
        """
        Forward pass for fine-tuning the MetaNet model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_dim + one_hot_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = inputs[:, :self.input_dim]
        specific_params = self.specific_param_generator(torch.ones_like(x[:, [0]], device=self.device))
        x_with_specific_params = torch.cat([x, specific_params], dim=1)
        output_net = self.backbone(x_with_specific_params)
        output_hat = self.f_hat_func(x_with_specific_params, self.dataset_dict, self.use_raw_data)
        return output_net, output_hat

    def set_finetune(self):
        """Sets the model for fine-tuning by freezing the backbone network."""
        for param in self.backbone.parameters():
            param.requires_grad_(False)


# ---------------------------
# Main Execution
# ---------------------------

def search_specific_param_dim(min_dim, max_dim, step, epochs, dataset_dict, optimizer_type=None):
    """
    Search for the optimal dimension of slow parameters by evaluating different values.

    Args:
        min_dim (int): Minimum dimension to start from.
        max_dim (int): Maximum dimension to test up to.
        step (int): Step size between dimensions.
        epochs (int): Number of epochs per dimension.
        dataset_dict (dict): Dataset dictionary structure containing raw/normalized datasets and metadata.
        optimizer_type (str, optional): Type of optimizer ('lbfgs' or 'adam'). Defaults to config value.

    Returns:
        list: List of final evaluation logs for each tested dimension.
    """
    logger = []

    # Load train/eval data with one-hot encoding
    train_loader, eval_loader = load_dataset_with_one_hot(
        dataset_dict,
        split='train',
        batch_size=config["batch_size"],
        train_ratio=config["train_ratio"],
        device=config["device"]
    )

    # Extract dimensions from dataset_dict
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']

    for dim in range(min_dim, max_dim, step):
        print(f"Testing slow parameter dimension: {dim}")

        # Build model with current dimension
        meta_net = MetaNet(
            input_dim=input_dim,
            output_dim=output_dim,
            one_hot_dim=one_hot_dim,
            specific_param_dim=dim,
            hidden_widths=config['hidden_widths'],
            device=config["device"],
            dataset_dict=dataset_dict,
            use_raw_data=False
        )
        trainer = Trainer(model=meta_net, learning_rate=config["learning_rate"], optimizer_type=optimizer_type)

        # Train and log result
        log_entry = trainer.search_specific_param_dim(epochs, train_loader, eval_loader)[-1]
        logger.append(log_entry)

        # Save best model so far
        trainer.save_model("pretrain.pt")

    return logger


def plot_and_save_loss_curve(training_log, dataset_name, save_dir="pictures"):
    """
    绘制并保存损失下降图
    
    Args:
        training_log (list): 训练日志，包含每个epoch的损失信息
        dataset_name (str): 数据集名称
        save_dir (str): 保存目录
    """
    if not training_log:
        print("警告：训练日志为空，无法绘制损失图")
        return
    
    # 创建保存目录
    save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 提取损失数据
    epochs = [log['epoch'] for log in training_log]
    
    # 检查日志格式
    if 'train_loss' in training_log[0] and 'test_loss' in training_log[0]:
        # 新格式：分别记录训练和测试损失
        train_losses = [log['train_loss'] for log in training_log]
        test_losses = [log['test_loss'] for log in training_log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{dataset_name} - Pretraining Loss Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度更好地显示损失下降
        
    else:
        # 旧格式：只有一个损失值
        losses = [log['loss'] for log in training_log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-', label='Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{dataset_name} - Pretraining Loss Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度更好地显示损失下降
    
    # 保存图片
    filename = os.path.join(save_path, 'pretrain_loss_curve.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f"损失下降图已保存到: {filename}")


def _export_epoch_time_log(training_log, system_name, phase, save_dir="final_data"):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{system_name}_{phase}_time.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "elapsed_seconds"])
        for log in training_log:
            if "test_loss" in log:
                loss = log.get("test_loss")
            else:
                loss = log.get("loss", log.get("train_loss"))
            writer.writerow([
                log.get("epoch"),
                loss,
                log.get("elapsed_seconds"),
            ])
    print(f"{phase} epoch time log saved to {filename}")


def pretrain(specific_param_dim=None, epochs=None, reg_rate=None, dataset_dict=None, optimizer_type=None, 
             use_raw_data=None, use_trajectory_sampling=None, num_trajectories=None, trajectory_duration=None, 
             dt=None, config_overrides=None):
    """
    Pretrain the model using symbolic regularization.

    Args:
        specific_param_dim (int, optional): Dimension of slow parameters. If None, uses config or config_overrides.
        epochs (int, optional): Number of epochs for pretraining. If None, uses config_overrides.
        reg_rate (float, optional): Strength of symbolic regularization. If None, uses config or config_overrides.
        dataset_dict (dict): Dataset dictionary structure.
        optimizer_type (str, optional): Type of optimizer ('lbfgs' or 'adam'). Defaults to config value.
        use_raw_data (bool, optional): If True, use raw data instead of normalized data. If None, uses config_overrides.
        use_trajectory_sampling (bool, optional): If True, use trajectory segment sampling. If None, uses config_overrides.
        num_trajectories (int, optional): Number of trajectory segments to sample per system. If None, uses config_overrides.
        trajectory_duration (float, optional): Duration of each trajectory segment in seconds. If None, uses config_overrides.
        dt (float, optional): Time step used in data generation. If None, will use dt from dataset_dict.
        config_overrides (dict, optional): Dictionary of config values to override. 
            Supported keys: 'specific_param_dim', 'epochs', 'reg_rate', 'batch_size', 'train_ratio', 
            'learning_rate', 'hidden_widths', 'weight_decay', 'adam_lr', 'optimizer', 'use_cautious_adam', 
            'device', 'use_raw_data', 'use_trajectory_sampling', 'num_trajectories', 'trajectory_duration'.

    Returns:
        dict: Logging information from the pretraining phase.
    """
    # 优先从config_overrides读取参数，如果没有则使用函数参数，最后使用config默认值
    if config_overrides is not None:
        # 更新config配置
        for key, value in config_overrides.items():
            if key in config:
                config[key] = value
            else:
                print(f"警告: config中没有键 '{key}'，忽略该覆盖值")
        
        # 从config_overrides读取训练参数（如果提供了）
        specific_param_dim = config_overrides.get('specific_param_dim', specific_param_dim)
        epochs = config_overrides.get('epochs', epochs)
        reg_rate = config_overrides.get('reg_rate', reg_rate)
        use_raw_data = config_overrides.get('use_raw_data', use_raw_data)
        use_trajectory_sampling = config_overrides.get('use_trajectory_sampling', use_trajectory_sampling)
        num_trajectories = config_overrides.get('num_trajectories', num_trajectories)
        trajectory_duration = config_overrides.get('trajectory_duration', trajectory_duration)
        # optimizer_type可以从config_overrides的'optimizer_type'或'optimizer'中读取
        optimizer_type = config_overrides.get('optimizer_type', 
                                             config_overrides.get('optimizer', optimizer_type))
    
    # 验证必需参数
    if dataset_dict is None:
        raise ValueError("dataset_dict参数必须提供")
    
    # 如果函数参数仍为None，则从config中获取默认值
    if specific_param_dim is None:
        specific_param_dim = config.get('specific_param_dim', 3)
    if epochs is None:
        raise ValueError("epochs参数必须提供（通过函数参数或config_overrides）")
    if reg_rate is None:
        reg_rate = config.get('reg_rate', 1.0)
    if use_raw_data is None:
        use_raw_data = False
    if use_trajectory_sampling is None:
        use_trajectory_sampling = False
    if num_trajectories is None:
        num_trajectories = 5
    if trajectory_duration is None:
        trajectory_duration = 1.0
    
    # Load train/eval data with appropriate sampling strategy
    if use_trajectory_sampling:
        train_loader, eval_loader = load_dataset_with_trajectory_sampling(
            dataset_dict,
            split='train',
            batch_size=config["batch_size"],
            train_ratio=config["train_ratio"],
            device=config["device"],
            use_raw_data=use_raw_data,
            num_trajectories=num_trajectories,
            trajectory_duration=trajectory_duration,
            dt=dt
        )
    else:
        train_loader, eval_loader = load_dataset_with_one_hot(
            dataset_dict,
            split='train',
            batch_size=config["batch_size"],
            train_ratio=config["train_ratio"],
            device=config["device"],
            use_raw_data=use_raw_data
        )

    # Extract dimensions from dataset_dict
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']


    # Build model
    meta_net = MetaNet(
        input_dim=input_dim,
        output_dim=output_dim,
        one_hot_dim=one_hot_dim,
        specific_param_dim=specific_param_dim,
        hidden_widths=config['hidden_widths'],
        device=config["device"],
        dataset_dict=dataset_dict,
        use_raw_data=use_raw_data
    )
    trainer = Trainer(model=meta_net, learning_rate=config["learning_rate"], optimizer_type=optimizer_type)

    # Start pretraining
    logger = trainer.pretrain(epochs, train_loader, eval_loader, reg_rate)
    trainer.save_model("pretrain.pt")

    # 获取数据集名称并绘制损失图
    try:
        from datasets import CURRENT_SYSTEM
        dataset_name = CURRENT_SYSTEM
    except ImportError:
        dataset_name = dataset_dict.get('name', 'unknown_dataset')
    
    plot_and_save_loss_curve(logger, dataset_name)

    _export_epoch_time_log(logger, dataset_name, "pretrain")

    return logger


def train_from_scratch(specific_param_dim, epochs, dataset_dict, optimizer_type='adam'):
    """
    Train the model from scratch on a small amount of test data.

    Args:
        specific_param_dim (int): Dimension of slow parameters.
        epochs (int): Number of epochs for fine-tuning.
        dataset_dict (dict): Dataset dictionary structure.
        optimizer_type (str, optional): Type of optimizer ('lbfgs' or 'adam'). Defaults to 'adam'.

    Returns:
        dict: Logging information from the training phase.
    """
    # Extract dimensions from dataset_dict
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']

    # Build model but DO NOT load pretrained weights
    meta_net = MetaNet(
        input_dim=input_dim,
        output_dim=output_dim,
        one_hot_dim=one_hot_dim,
        specific_param_dim=specific_param_dim,
        hidden_widths=config['hidden_widths'],
        device=config["device"],
        dataset_dict=dataset_dict,
        use_raw_data=False
    )
    trainer = Trainer(model=meta_net, learning_rate=config["learning_rate"], optimizer_type=optimizer_type)
    # trainer.load_model("pretrain.pt") # This line is intentionally commented out

    # Load test data
    support_loader, _ = load_dataset_with_one_hot(
        dataset_dict,
        split='support',
        batch_size=config["batch_size"],
        train_ratio=1.0,
        device=config["device"]
    )
    query_loader, _ = load_dataset_with_one_hot(
        dataset_dict,
        split='query',
        batch_size=config["batch_size"],
        train_ratio=1.0,
        device=config["device"]
    )

    # Train from scratch on small data
    logger = trainer.train_from_scratch(epochs, support_loader, query_loader)
    trainer.save_model("scratch.pt")

    return logger

def finetune(specific_param_dim=None, epochs=None, dataset_dict=None, optimizer_type=None, 
             convergence_threshold=1e-6, patience=10, config_overrides=None):
    """
    Finetune the pretrained model on a small amount of test data.

    Args:
        specific_param_dim (int, optional): Dimension of slow parameters. If None, uses config or config_overrides.
        epochs (int, optional): Number of epochs for fine-tuning. If None, uses config_overrides.
        dataset_dict (dict): Dataset dictionary structure.
        optimizer_type (str, optional): Type of optimizer ('lbfgs' or 'adam'). Defaults to config value.
        convergence_threshold (float): Loss change threshold for early stopping.
        patience (int): Number of epochs to wait before early stopping.
        config_overrides (dict, optional): Dictionary of config values to override.
            Supported keys: 'specific_param_dim', 'epochs', 'batch_size', 'learning_rate', 
            'hidden_widths', 'device', 'optimizer', 'convergence_threshold', 'patience', etc.

    Returns:
        dict: Logging information from the fine-tuning phase including convergence info.
    """
    import json
    from datetime import datetime
    
    # 优先从config_overrides读取参数，如果没有则使用函数参数，最后使用config默认值
    if config_overrides is not None:
        # 更新config配置
        for key, value in config_overrides.items():
            if key in config:
                config[key] = value
        
        # 从config_overrides中读取函数参数
        if specific_param_dim is None and 'specific_param_dim' in config_overrides:
            specific_param_dim = config_overrides['specific_param_dim']
        if epochs is None and 'epochs' in config_overrides:
            epochs = config_overrides['epochs']
        if optimizer_type is None and 'optimizer' in config_overrides:
            optimizer_type = config_overrides['optimizer']
        if optimizer_type is None and 'optimizer_type' in config_overrides:
            optimizer_type = config_overrides['optimizer_type']
        if 'convergence_threshold' in config_overrides:
            convergence_threshold = config_overrides['convergence_threshold']
        if 'patience' in config_overrides:
            patience = config_overrides['patience']
    
    # Extract dimensions from dataset_dict
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']


    # Build model and load pretrained weights
    meta_net = MetaNet(
        input_dim=input_dim,
        output_dim=output_dim,
        one_hot_dim=one_hot_dim,
        specific_param_dim=specific_param_dim,
        hidden_widths=config['hidden_widths'],
        device=config["device"],
        dataset_dict=dataset_dict,
        use_raw_data=False  # finetune通常使用标准化数据
    )
    trainer = Trainer(model=meta_net, learning_rate=config["learning_rate"], optimizer_type=optimizer_type)
    trainer.load_model("pretrain.pt")

    # Load test data (split into two parts if needed)
    support_loader, _ = load_dataset_with_one_hot(
        dataset_dict,
        split='support',
        batch_size=config["batch_size"],
        train_ratio=1.0,
        device=config["device"]
    )
    query_loader, _ = load_dataset_with_one_hot(
        dataset_dict,
        split='query',
        batch_size=config["batch_size"],
        train_ratio=1.0,
        device=config["device"]
    )

    # Finetune on small data
    result = trainer.finetune(epochs, support_loader, query_loader, convergence_threshold, patience)
    trainer.save_model("finetune.pt")

    # 获取系统名称
    try:
        from datasets import CURRENT_SYSTEM
        system_name = CURRENT_SYSTEM
    except ImportError:
        system_name = 'unknown_system'
    
    # 创建结果保存目录
    results_dir = os.path.join('results', system_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备保存的结果信息
    result_info = {
        'system_name': system_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimizer': optimizer_type or config['optimizer'],
        'max_epochs': epochs,
        'converged_epoch': result['converged_epoch'],
        'total_epochs': result['total_epochs'],
        'training_time_seconds': result['training_time'],
        'training_time_formatted': f"{result['training_time']:.2f}s ({result['training_time']/60:.2f}min)",
        'final_loss': result['final_loss'],
        'convergence_threshold': convergence_threshold,
        'patience': patience,
        'specific_param_dim': specific_param_dim,
        'config': {
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'reg_rate': config['reg_rate'],
            'weight_decay': config['weight_decay'],
            'hidden_widths': config['hidden_widths']
        }
    }
    
    # 保存为JSON文件
    json_path = os.path.join(results_dir, 'finetune_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_info, f, indent=4, ensure_ascii=False)
    
    # 保存为可读的文本文件
    txt_path = os.path.join(results_dir, 'finetune_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"微调结果报告 - {system_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"时间戳: {result_info['timestamp']}\n")
        f.write(f"优化器: {result_info['optimizer']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("收敛信息:\n")
        f.write("-" * 60 + "\n")
        f.write(f"最大训练轮数: {result_info['max_epochs']}\n")
        f.write(f"实际训练轮数: {result_info['total_epochs']}\n")
        f.write(f"收敛轮数: {result_info['converged_epoch']}\n")
        if result_info['converged_epoch'] < result_info['max_epochs']:
            f.write(f"✓ 提前收敛 (节省 {result_info['max_epochs'] - result_info['converged_epoch']} 轮)\n")
        else:
            f.write(f"✗ 未提前收敛，训练至最大轮数\n")
        f.write(f"\n训练时间: {result_info['training_time_formatted']}\n")
        f.write(f"最终损失: {result_info['final_loss']:.6e}\n")
        f.write(f"收敛阈值: {result_info['convergence_threshold']:.6e}\n")
        f.write(f"耐心值: {result_info['patience']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("模型配置:\n")
        f.write("-" * 60 + "\n")
        f.write(f"慢参数维度: {result_info['specific_param_dim']}\n")
        f.write(f"学习率: {result_info['config']['learning_rate']}\n")
        f.write(f"批次大小: {result_info['config']['batch_size']}\n")
        f.write(f"正则化率: {result_info['config']['reg_rate']}\n")
        f.write(f"权重衰减: {result_info['config']['weight_decay']}\n")
        f.write(f"隐藏层宽度: {result_info['config']['hidden_widths']}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"\n{'='*60}")
    print(f"微调完成！")
    print(f"{'='*60}")
    print(f"收敛轮数: {result['converged_epoch']} / {epochs}")
    print(f"训练时间: {result['training_time']:.2f}秒 ({result['training_time']/60:.2f}分钟)")
    print(f"最终损失: {result['final_loss']:.6e}")
    print(f"\n结果已保存到:")
    print(f"  - JSON: {json_path}")
    print(f"  - TXT:  {txt_path}")
    print(f"{'='*60}\n")

    _export_epoch_time_log(result.get('log', []), system_name, "finetune")

    return result


# ---------------------------
# Usage Examples
# ---------------------------

def example_usage():
    """
    示例：如何使用不同的优化器
    
    使用方法：
    1. 修改config中的'optimizer'参数：'lbfgs' 或 'adam'
    2. 或者在调用函数时直接传递optimizer_type参数
    """
    
    # 方法1：修改全局配置
    # config['optimizer'] = 'adam'  # 切换到Adam优化器
    # config['optimizer'] = 'lbfgs'  # 切换到LBFGS优化器（默认）
    
    # 方法2：在函数调用时指定优化器
    # dataset_dict = get_datasets()  # 假设你有这个函数
    
    # 使用LBFGS优化器进行预训练
    # logger_lbfgs = pretrain(specific_param_dim=3, epochs=10, reg_rate=0.5, 
    #                        dataset_dict=dataset_dict, optimizer_type='lbfgs')
    
    # 使用Adam优化器进行预训练
    # logger_adam = pretrain(specific_param_dim=3, epochs=10, reg_rate=0.5, 
    #                       dataset_dict=dataset_dict, optimizer_type='adam')
    
    # 使用Adam优化器进行微调
    # logger_finetune = finetune(specific_param_dim=3, epochs=5, 
    #                           dataset_dict=dataset_dict, optimizer_type='adam')
    
    print("优化器切换接口已添加！")
    print("可用优化器：'lbfgs'（默认）, 'adam'")
    print("Adam优化器学习率：", config['adam_lr'])
    print("当前默认优化器：", config['optimizer'])
    print("\n=== Cautious Adam 优化器功能 ===")
    print("基于论文的单行代码修改实现的Cautious Adam优化器已集成！")
    print("使用Cautious Adam：", "是" if config.get('use_cautious_adam', False) else "否")
    print("切换方法：修改 config['use_cautious_adam'] = True/False")
    print("论文改进：只有当更新方向与梯度方向一致时才应用更新，提升训练稳定性")


if __name__ == "__main__":
    example_usage()
