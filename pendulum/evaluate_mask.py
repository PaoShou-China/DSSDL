import numpy as np
import os
import torch
import sys
import pandas as pd
import random
from datasets import f, slow_params_eval, CURRENT_SYSTEM, dt, dataset_dict
from DSSDL_mask import MetaNet
from meta_train import CONFIG_OVERRIDES

# ================= 1. 环境与种子设置 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ================= 2. 参数提取与导出函数 =================
def extract_real_slowparams(system_name):
    system_path = f"datasets/{system_name}"
    feature_dir = os.path.join(system_path, "train", "features")
    if not os.path.isdir(feature_dir):
        return [], []
    files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".txt")])
    real_params = []
    param_names = []
    if system_name in {"pendulum", "pendulum_real"}:
        for file in files:
            name = file[:-4]
            param = float(name)
            real_params.append([param])
        param_names = ["L (pendulum length)"]
    return real_params, param_names

def generate_slowparams_from_onehot(model, num_systems, device="cpu"):
    generated_params = []
    with torch.no_grad():
        for i in range(num_systems):
            one_hot = torch.zeros(1, num_systems, device=device)
            one_hot[0, i] = 1.0
            specific_params = model.one_hot_to_specific_param(one_hot)
            generated_params.append(specific_params.cpu().numpy()[0])
    return np.array(generated_params)

def export_slowparams(system_name, real_params, generated_params, param_names):
    save_dir = os.path.join("final_data", "hidden_params", system_name)
    os.makedirs(save_dir, exist_ok=True)
    def _build_header(base_names, dim, prefix):
        if base_names and len(base_names) == dim:
            return ["system_index"] + base_names
        return ["system_index"] + [f"{prefix}{i+1}" for i in range(dim)]
    
    if real_params:
        real_array = np.array(real_params, dtype=np.float64)
        header = _build_header(param_names, real_array.shape[1], "param")
        indices = np.arange(1, real_array.shape[0] + 1, dtype=np.float64).reshape(-1, 1)
        export_array = np.hstack([indices, real_array])
        np.savetxt(os.path.join(save_dir, "real_params.csv"), export_array, delimiter=",", fmt="%.10f", header=",".join(header), comments="")
    
    if generated_params is not None and generated_params.size > 0:
        learned_array = np.array(generated_params, dtype=np.float64)
        header = _build_header(None, learned_array.shape[1], "param")
        indices = np.arange(1, learned_array.shape[0] + 1, dtype=np.float64).reshape(-1, 1)
        export_array = np.hstack([indices, learned_array])
        np.savetxt(os.path.join(save_dir, "learned_params.csv"), export_array, delimiter=",", fmt="%.10f", header=",".join(header), comments="")

def export_masked_slowparams(system_name, dataset_dict, device="cpu", use_raw_data=False, model_path="model/meta_train.pt"):
    input_dim = dataset_dict['info']['input_dim']
    output_dim = dataset_dict['info']['output_dim']
    one_hot_dim = dataset_dict['info']['file_counts']['train']
    model = MetaNet(input_dim=input_dim, output_dim=output_dim, one_hot_dim=one_hot_dim, specific_param_dim=CONFIG_OVERRIDES['specific_param_dim'], hidden_widths=CONFIG_OVERRIDES['hidden_widths'], device=device, dataset_dict=dataset_dict, use_raw_data=use_raw_data)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    
    active_idx = None
    if hasattr(model, "continuous_mask"):
        with torch.no_grad():
            mask_values = torch.sigmoid(model.continuous_mask).cpu().numpy()
            active_idx = mask_values > getattr(model, "mask_threshold", 0.5)
            
    real_params, param_names = extract_real_slowparams(system_name)
    generated_params = generate_slowparams_from_onehot(model, one_hot_dim, device=device)
    if active_idx is not None:
        generated_params = generated_params[:, active_idx]
    export_slowparams(system_name, real_params, generated_params, param_names)

# ================= 3. 轨迹预测与数据导出 =================
def predict_trajectory(x0, T, dt, normalization, model_path='model/fast_adapt.pt', device='cpu', use_raw_data=False):
    if not use_raw_data:
        input_mean = torch.tensor(normalization['input']['mean'], dtype=torch.float32, device=device)
        input_std = torch.tensor(normalization['input']['std' ], dtype=torch.float32, device=device)
        output_mean = torch.tensor(normalization['output']['mean'], dtype=torch.float32, device=device)
        output_std = torch.tensor(normalization['output']['std' ], dtype=torch.float32, device=device)

    input_dim = len(normalization['input']['mean'])
    output_dim = len(normalization['output']['mean'])
    one_hot_dim = dataset_dict['info']['file_counts']['train']
    
    model = MetaNet(input_dim, output_dim, one_hot_dim, CONFIG_OVERRIDES['specific_param_dim'], CONFIG_OVERRIDES['hidden_widths'], device, dataset_dict=dataset_dict, use_raw_data=use_raw_data)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(x0, dtype=torch.float32, device=device).reshape(1, -1)
    traj = [x.cpu().numpy().copy()]

    for _ in range(T):
        with torch.no_grad():
            if CURRENT_SYSTEM == 'pendulum_real':
                # 物理缩放逻辑：索引0是速度，索引1是角度
                x_k1_scaled = x.clone()
                x_k1_scaled[:, 0] /= 20.0 
                x_k1_input = (x_k1_scaled - input_mean) / input_std
                dx1_pred, dx1_hat_pred = model.fast_adapt_forward(x_k1_input)
                k1_unscaled = (dx1_pred + dx1_hat_pred) * output_std + output_mean
                k1 = k1_unscaled.clone()
                k1[:, 0] *= 100.0; k1[:, 1] *= 20.0
                
                # RK2 步进逻辑
                x_mid = x + k1 * dt
                x_k2_scaled = x_mid.clone()
                x_k2_scaled[:, 0] /= 20.0
                x_k2_input = (x_k2_scaled - input_mean) / input_std
                dx2_pred, dx2_hat_pred = model.fast_adapt_forward(x_k2_input)
                k2_unscaled = (dx2_pred + dx2_hat_pred) * output_std + output_mean
                k2 = k2_unscaled.clone()
                k2[:, 0] *= 100.0; k2[:, 1] *= 20.0
                
                x = x + (0.996 * k1 + 0.004 * k2) * dt
            else:
                x_input = (x - input_mean) / input_std
                dx_p, dx_h = model.fast_adapt_forward(x_input)
                dx = (dx_p + dx_h) * output_std + output_mean
                x = x + dx * dt
        traj.append(x.cpu().numpy().copy())
    return np.concatenate(traj, axis=0)

def export_trajectory_data(traj_true, traj_pred, dt, system_name, tag, x0=None):
    directory = os.path.join('final_data', 'trajectories_data', system_name)
    os.makedirs(directory, exist_ok=True)
    steps = traj_true.shape[0]
    time_axis = np.arange(steps) * dt
    data_dict = {'time': time_axis}
    for idx in range(traj_true.shape[1]):
        data_dict[f'true_dim{idx}'] = traj_true[:, idx]
        data_dict[f'pred_dim{idx}'] = traj_pred[:, idx]
    pd.DataFrame(data_dict).to_csv(os.path.join(directory, f'{tag}.csv'), index=False)
    if x0 is not None:
        np.savetxt(os.path.join(directory, f'{tag}_x0.txt'), np.asarray(x0).reshape(1, -1))

# ================= 4. 继承原有的画图格式 =================
def plot_trajectory_adaptive(traj_true, traj_pred, dt, system_name, save_dir, traj_pred_noft, t):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # 统一Matplotlib样式设置
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 20,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'lines.linewidth': 3.5,
        'lines.markersize': 10,
        'axes.unicode_minus': False,
    })

    max_time = 5.0
    max_steps = min(int(max_time / dt) + 1, traj_true.shape[0])
    
    # 数据截取
    t_plot = t[:max_steps]
    true_plot = traj_true[:max_steps]
    pred_plot = traj_pred[:max_steps]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    nature_colors = ['#403990', '#80A6E2', '#FBDD85', '#F46F43', '#CF3D3E']

    # 绘制角度（索引1为 theta）
    ax.plot(t_plot, true_plot[:, 1], color=nature_colors[0], linewidth=3.5, label='Ref.')
    ax.plot(t_plot, pred_plot[:, 1], color=nature_colors[3], linewidth=3.5, linestyle='--', label='Pred.')
    if traj_pred_noft is not None:
        ax.plot(t_plot, traj_pred_noft[:max_steps, 1], color=nature_colors[1], linewidth=3.5, linestyle=':', label='w/o Adaption')

    ax.set_ylabel(r'Angle $\theta$ (rad)', fontsize=20, labelpad=8)
    ax.set_xlabel('Time (s)', fontsize=20, labelpad=8)
    ax.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 保持所有边框可见
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
    
    plt.subplots_adjust(left=0.12, right=0.95, top=0.80, bottom=0.12)

    plt.show()

# ================= 5. 主程序执行 =================
if __name__ == "__main__":
    set_seed(42)
    file_path = 'final_data/trajectories_data/pendulum_real/[8].csv'
    data_df = pd.read_csv(file_path)
    
    # 【核心维度对换点】：使用 [1, 3] 确保 0维是速度, 1维是角度
    traj_true = data_df.iloc[:, [1, 3]].values 
    t_raw = data_df['time'].values
    x0_arr = traj_true[0].copy()
    steps = len(traj_true) - 1
    
    norm_info = dataset_dict['info']['normalization']
    
    # 预测执行
    traj_pred = predict_trajectory(x0_arr, steps, dt, norm_info, model_path='model/fast_adapt.pt')
    traj_pred_noft = predict_trajectory(x0_arr, steps, dt, norm_info, model_path='model/no_fast_adapt.pt')

    # 导出参数与轨迹
    export_masked_slowparams(CURRENT_SYSTEM, dataset_dict)
    export_trajectory_data(traj_true, traj_pred, dt, CURRENT_SYSTEM, tag=f'{slow_params_eval}', x0=x0_arr)
    if traj_pred_noft is not None:
        export_trajectory_data(traj_true, traj_pred_noft, dt, CURRENT_SYSTEM, tag=f'{slow_params_eval}_no_fast_adapt', x0=x0_arr)

    # 绘图展示
    plot_trajectory_adaptive(traj_true, traj_pred, dt, CURRENT_SYSTEM, None, traj_pred_noft, t_raw)