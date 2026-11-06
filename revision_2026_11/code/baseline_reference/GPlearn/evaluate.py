import numpy as np
import pickle
import os
import pandas as pd
from datasets import f as dyn_f, dt, dataset_dict, CURRENT_SYSTEM, slow_params_eval

def true_trajectory(x0, steps, dt, slow_params=slow_params_eval):
    """
    生成系统的真实轨迹（自适应维度）。
    """
    x0 = np.asarray(x0).reshape(1, -1)
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        dx = dyn_f(x, slow_params) * dt
        x = x + dx
        traj.append(x.copy())
    traj = np.concatenate(traj, axis=0)
    return traj

def predict_trajectory_gplearn(x0, steps, dt, model_path='gplearn_model.pkl'):
    """
    使用gplearn辨识出的方程生成预测轨迹。
    """
    with open(model_path, 'rb') as f:
        gplearn_models = pickle.load(f)
    
    x = np.asarray(x0).reshape(1, -1)
    traj = [x.copy()]
    
    for _ in range(steps):
        # Predict dx using each model for each dimension
        dx_list = [model.predict(x) for model in gplearn_models]
        dx = np.concatenate(dx_list).reshape(1, -1)
        
        x = x + dx * dt
        if np.any(np.abs(x) > 1e10):
            print(f"⚠️  轨迹在步骤 {_} 发散，已提前中止预测。")
            break
        traj.append(x.copy())
    
    traj = np.concatenate(traj, axis=0)
    return traj

def generate_adaptive_initial_state(system_name, input_dim):
    """
    根据系统类型和输入维度生成合适的初始状态
    """
    if system_name == 'pendulum':
        base_state = [np.pi/4, 0.0]
    elif system_name == 'lorenz':
        base_state = [8.0, -8.0, 27.0]
    elif system_name == 'double_pendulum':
        base_state = [np.pi/4, np.pi/4, 0, 0]
    elif system_name == 'van_der_pol':
        base_state = [1.1, 0.5]
    elif system_name == 'FHN':
        base_state = [-1.0, 1.0]
    else:
        np.random.seed(42)
        base_state = np.random.uniform(-2.0, 2.0, min(input_dim, 6)).tolist()
        
    x0_arr = np.asarray(base_state).reshape(-1)
    if x0_arr.size < input_dim:
        x0_arr = np.pad(x0_arr, (0, input_dim - x0_arr.size))
    elif x0_arr.size > input_dim:
        x0_arr = x0_arr[:input_dim]
    
    return x0_arr

def export_trajectory_data(traj_true, traj_pred, dt, system_name, tag, x0=None):
    """
    Export true and predicted trajectories (truncate to common length).
    """
    # Truncate to the shortest trajectory to handle early-stop divergence
    min_len = min(len(traj_true), len(traj_pred))
    traj_true = traj_true[:min_len]
    traj_pred = traj_pred[:min_len]

    directory = os.path.join('final_data_GPlearn', 'trajectories_data', system_name)
    os.makedirs(directory, exist_ok=True)

    time_axis = np.arange(min_len) * dt

    data_dict = {'time': time_axis}
    dim = traj_true.shape[1]
    for idx in range(dim):
        data_dict[f'true_dim{idx}'] = traj_true[:, idx]
        data_dict[f'pred_dim{idx}'] = traj_pred[:, idx]

    df = pd.DataFrame(data_dict)
    file_path = os.path.join(directory, f'{tag}.csv')
    df.to_csv(file_path, index=False)

    if x0 is not None:
        x0_path = os.path.join(directory, f'{tag}_x0.txt')
        np.savetxt(x0_path, np.asarray(x0, dtype=np.float32).reshape(1, -1))
        print(f"Initial state saved: {x0_path}")

    print(f"Trajectory data saved: {file_path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    rcParams['font.family'] = 'Times New Roman'
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.unicode_minus'] = False
    rcParams['font.size'] = 16

    steps = 5000
    input_dim = dataset_dict['info']['input_dim']

    if CURRENT_SYSTEM == 'flow' or CURRENT_SYSTEM == 'flow-19':
        print("Flow system evaluation mode")
        # Load true trajectory from query set
        query_features_path = os.path.join('datasets', CURRENT_SYSTEM, 'query', 'features', f'{slow_params_eval[0]}.txt')
        print(f"Loading true trajectory from {query_features_path}...")
        traj_true = np.loadtxt(query_features_path)
        # Use the first row as the initial state
        x0 = traj_true[0]

        # Match steps to data length
        steps = len(traj_true) - 1
        print(f"Loaded trajectory: shape={traj_true.shape}, steps={steps}")
        print(f"Using first row as initial state: {x0}")
    else:
        x0 = generate_adaptive_initial_state(CURRENT_SYSTEM, input_dim)
        traj_true = true_trajectory(x0, steps, dt)
    traj_pred = predict_trajectory_gplearn(x0, steps, dt, model_path='gplearn_model.pkl')

    export_trajectory_data(traj_true, traj_pred, dt, CURRENT_SYSTEM, tag=f'{slow_params_eval}', x0=x0)

    # Truncate for plotting and metrics
    min_len = min(len(traj_true), len(traj_pred))
    traj_true = traj_true[:min_len]
    traj_pred = traj_pred[:min_len]

    t = np.arange(min_len) * dt
    D = traj_true.shape[1]
    fig, axs = plt.subplots(D, 1, figsize=(8, 2.5 * D), sharex=True)
    if D == 1:
        axs = [axs]
    
    labels = [rf'$x_{{{i+1}}}$' for i in range(D)]
    for i in range(D):
        axs[i].plot(t, traj_true[:, i], color='black', lw=2, label='True')
        axs[i].plot(t, traj_pred[:, i], color='blue', lw=2, linestyle='--', label='GPlearn Predicted')
        axs[i].set_ylabel(labels[i], fontsize=16)
        axs[i].legend(fontsize=14, loc='upper right')
        axs[i].grid(True, linestyle=':')
    
    axs[-1].set_xlabel('Time $t$', fontsize=16)
    fig.suptitle('Component-wise Trajectory Comparison', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    mse = np.mean((traj_true - traj_pred) ** 2, axis=0)
    mae = np.mean(np.abs(traj_true - traj_pred), axis=0)
    
    print("=== Error Metrics ===")
    for i in range(D):
        print(f"Dimension {i+1}:")
        print(f"  MSE: {mse[i]:.6f}")
        print(f"  MAE: {mae[i]:.6f}")
    
    print("\n=== GPlearn Discovered Equations ===")
    try:
        with open('gplearn_model.pkl', 'rb') as f:
            gplearn_models = pickle.load(f)
        for i, model in enumerate(gplearn_models):
            print(f"  x{i}' = {model._program}")
    except Exception as e:
        print(f"Error loading GPlearn model: {e}")
