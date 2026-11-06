import numpy as np
import pickle
import os
import pandas as pd
from datasets import f as dyn_f, dt, dataset_dict, CURRENT_SYSTEM, slow_params_eval


def rk4_step(x, dt, deriv_fn):
    """
    单步 RK4 积分；deriv_fn 接受 x 并返回 dx/dt
    """
    k1 = deriv_fn(x)
    k2 = deriv_fn(x + 0.5 * dt * k1)
    k3 = deriv_fn(x + 0.5 * dt * k2)
    k4 = deriv_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def true_trajectory(x0, steps, dt, slow_params=slow_params_eval, integrator="rk45"):
    """
    生成系统的真实轨迹（自适应维度）。
    """
    x0 = np.asarray(x0).reshape(1, -1)
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        if integrator == "euler":
            x = x + dyn_f(x, slow_params) * dt
        else:
            x = rk4_step(x, dt, lambda xx: dyn_f(xx, slow_params))
        traj.append(x.copy())
    traj = np.concatenate(traj, axis=0)
    return traj

def predict_trajectory_sindy(x0, steps, dt, model_path='sindy_model.pkl', integrator="rk45"):
    """
    使用SINDy辨识出的方程生成预测轨迹（无标准化）。
    """
    with open(model_path, 'rb') as f:
        sindy_model = pickle.load(f)
    
    x = np.asarray(x0).reshape(1, -1)
    traj = [x.copy()]
    
    for _ in range(steps):
        if integrator == "euler":
            # 直接在物理空间预测导数
            dx = sindy_model.predict(x)
            # 物理空间积分
            x = x + dx * dt
        else:
            # 在物理空间用 RK4 积分
            x = rk4_step(x, dt, lambda xx: sindy_model.predict(xx))

        # 检查轨迹是否发散
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
        base_state = [-1, 1]
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
    导出真实和预测的轨迹数据。
    """
    # 为了安全处理提前中止的预测，将预测轨迹裁剪到与真实轨迹相同的长度
    min_len = min(len(traj_true), len(traj_pred))
    traj_true = traj_true[:min_len]
    traj_pred = traj_pred[:min_len]

    directory = os.path.join('final_data_SINDy', 'trajectories_data', system_name)
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
        print(f"✓ 初始状态已导出: {x0_path}")

    print(f"✓ 轨迹数据已导出: {file_path}")

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
        print("流体系统评估模式")
        # 直接从query集加载真实轨迹
        query_features_path = os.path.join('datasets', CURRENT_SYSTEM, 'query', 'features', f'{slow_params_eval[0]}.txt')
        print(f"从 {query_features_path} 加载真实轨迹...")
        traj_true = np.loadtxt(query_features_path)
        # 使用真实轨迹的第一行作为初始状态
        x0 = traj_true[0]

        # 更新步数以匹配数据长度
        steps = len(traj_true) - 1
        print(f"从数据加载的轨迹: shape={traj_true.shape}, steps={steps}")
        print(f"使用数据的第一行作为初始状态: {x0}")
    else:
        x0 = generate_adaptive_initial_state(CURRENT_SYSTEM, input_dim)
        traj_true = true_trajectory(x0, steps, dt)
    traj_pred = predict_trajectory_sindy(x0, steps, dt, model_path='sindy_model.pkl')

    export_trajectory_data(traj_true, traj_pred, dt, CURRENT_SYSTEM, tag=f'{slow_params_eval}', x0=x0)

    # 裁剪轨迹以进行绘图和误差计算
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
        axs[i].plot(t, traj_pred[:, i], color='red', lw=2, linestyle='--', label='SINDy Predicted')
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
    
    print("\n=== SINDy Discovered Equations ===")
    try:
        with open('sindy_model.pkl', 'rb') as f:
            sindy_model = pickle.load(f)
        sindy_model.print()
    except Exception as e:
        print(f"Error loading SINDy model: {e}")
