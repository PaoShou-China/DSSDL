import numpy as np
import torch

from datasets import f as dyn_f

from datasets import slow_params_eval, CURRENT_SYSTEM


def true_trajectory(x0, steps, dt, slow_params=slow_params_eval):
    """
    生成系统的真实轨迹（自适应维度）。
    如果是flow系统，直接从dataset/flow/query/features读取真实轨迹。
    x0: 初始状态，形如(D,)或(1,D)（对于flow系统，此参数将被忽略）
    steps: 步数
    dt: 步长
    slow_params: 系统参数
    返回：(steps+1, D) 的轨迹
    """
    # 如果是flow系统，直接从文件读取
    if CURRENT_SYSTEM == "flow" or CURRENT_SYSTEM == "flow-19" or CURRENT_SYSTEM == "flow-29-v":
        import os

        # 获取slow_params对应的文件名（雷诺数）
        re_num = (
            int(slow_params[0]) if isinstance(slow_params, (list, np.ndarray)) else int(slow_params)
        )
        query_file = os.path.join(f"datasets", CURRENT_SYSTEM, "query", "features", f"{re_num}.txt")

        if not os.path.exists(query_file):
            raise FileNotFoundError(f"找不到{CURRENT_SYSTEM}系统的query文件: {query_file}")

        # 读取真实轨迹
        traj = np.loadtxt(query_file)

        # 如果需要的步数少于文件中的数据，截断
        if steps + 1 < len(traj):
            traj = traj[: steps + 1]
        # 如果需要的步数多于文件中的数据，发出警告
        elif steps + 1 > len(traj):
            print(
                f"警告: 请求的步数 {steps + 1} 超过了文件中的数据点数 {len(traj)}，将使用全部数据"
            )

        return traj

    # 其他系统：使用动力学函数生成轨迹
    x0 = np.asarray(x0).reshape(1, -1)
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        dx = dyn_f(x, slow_params) * dt
        x = x + dx
        traj.append(x.copy())
    traj = np.concatenate(traj, axis=0)
    return traj


def predict_trajectory_reptile(
    x0, steps, dt, normalization, model_path="finetune.pt", device="cpu"
):
    """
    使用finetune后的Reptile模型生成预测轨迹（带标准化/反标准化，自适应维度）。
    x0: 初始状态，形如(D,)或(1,D)
    steps: 步数
    dt: 步长
    normalization: dict, 包含input/output的mean/std
    model_path: finetune.pt路径
    device: 设备
    返回：(steps+1, D) 的预测轨迹
    """
    import torch
    from Reptile import Reptile
    from datasets import dataset_dict

    input_mean = torch.tensor(normalization["input"]["mean"], dtype=torch.float32, device=device)
    input_std = torch.tensor(normalization["input"]["std"], dtype=torch.float32, device=device)
    output_mean = torch.tensor(normalization["output"]["mean"], dtype=torch.float32, device=device)
    output_std = torch.tensor(normalization["output"]["std"], dtype=torch.float32, device=device)

    # 初始化Reptile模型并加载finetune参数
    model = Reptile(dataset_dict)
    model.network.load_state_dict(torch.load(model_path, map_location=device))
    model.network.to(device)  # 确保模型在正确的设备上
    model.network.eval()

    # 自适应输入维度
    input_dim = input_mean.numel()
    x = torch.tensor(x0, dtype=torch.float32, device=device).reshape(1, input_dim)
    traj = [x.cpu().numpy().copy()]
    for _ in range(steps):
        # 1. 物理空间 -> 标准化
        x_norm = (x - input_mean) / input_std
        # 2. 输入MLP
        with torch.no_grad():
            dx_norm = model.network(x_norm)
            # 3. 输出反标准化
            dx_out = dx_norm * output_std + output_mean
            # 3.1 维度自适应：
            if dx_out.shape[1] == x.shape[1]:
                dx = dx_out
            elif x.shape[1] == 4 and dx_out.shape[1] == 2:
                # 假设状态为 [x, y, vx, vy]，输出为 [ax, ay]
                vx_vy = x[:, 2:4]
                dx = torch.cat([vx_vy, dx_out], dim=1)
            else:
                raise RuntimeError(
                    f"Output dim {dx_out.shape[1]} not compatible with state dim {x.shape[1]}"
                )
            # 4. 物理空间积分
            x = x + dx * dt
        traj.append(x.cpu().numpy().copy())
    traj = np.concatenate(traj, axis=0)
    return traj


def generate_adaptive_initial_state(system_name, input_dim):
    """
    根据系统类型和输入维度生成合适的初始状态
    """
    if system_name == "pendulum":
        # 摆系统：[phi_dot, phi] 或扩展维度
        base_state = [np.pi / 4, 0.0]  # 初始角速度和角度
    elif system_name == "lorenz" or system_name == "lorenz_firstorder":
        # 洛伦兹系统：[x, y, z] 或扩展维度
        base_state = [8.0, -8.0, 27.0]  # 经典初始状态
        # base_state = [0, 1.0, 2.0]
    elif system_name == "double_pendulum":
        # 双摆系统：[theta1, theta2, omega1, omega2]
        base_state = [np.pi / 4, np.pi / 4, 0, 0]  # 初始角度和角速度
    elif system_name == "van_der_pol":
        base_state = [1.1, 0.5]
    elif system_name == "FHN":
        base_state = [-1, 1]
    else:
        # 通用系统：生成小幅度随机初始状态
        np.random.seed(42)  # 固定种子确保可重复性
        base_state = np.random.uniform(-2.0, 2.0, min(input_dim, 6)).tolist()

    # 调整到所需维度
    x0_arr = np.asarray(base_state).reshape(-1)
    if x0_arr.size < input_dim:
        # 如果维度不足，用零填充
        x0_arr = np.pad(x0_arr, (0, input_dim - x0_arr.size))
    elif x0_arr.size > input_dim:
        # 如果维度过多，截断
        x0_arr = x0_arr[:input_dim]

    return x0_arr


def export_trajectory_data(traj_true, traj_pred, dt, system_name, tag, x0=None):
    """
    导出真实和预测的轨迹数据到 final_data/trajectories_data/<system_name> 目录。

    Args:
        traj_true: numpy.ndarray, 真实轨迹（物理空间）
        traj_pred: numpy.ndarray, 预测轨迹（物理空间）
        dt: float, 时间步长
        system_name: str, 系统名称
        tag: str, 文件标识
        x0: 可选的初始状态
    """
    import os
    import pandas as pd

    if traj_true.shape != traj_pred.shape:
        raise ValueError("真实轨迹与预测轨迹的形状不一致，无法导出。")

    directory = os.path.join("final_data_Reptile", "trajectories_data", system_name)
    os.makedirs(directory, exist_ok=True)

    steps = traj_true.shape[0]
    time_axis = np.arange(steps) * dt

    data_dict = {"time": time_axis}
    dim = traj_true.shape[1]
    for idx in range(dim):
        data_dict[f"true_dim{idx}"] = traj_true[:, idx]
        data_dict[f"pred_dim{idx}"] = traj_pred[:, idx]

    df = pd.DataFrame(data_dict)
    file_path = os.path.join(directory, f"{tag}.csv")
    df.to_csv(file_path, index=False)

    if x0 is not None:
        x0_path = os.path.join(directory, f"{tag}_x0.txt")
        np.savetxt(x0_path, np.asarray(x0, dtype=np.float32).reshape(1, -1))
        print(f"✓ 初始状态已导出: {x0_path}")

    print(f"✓ 轨迹数据已导出: {file_path}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 设置科研标准字体
    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = 16

    # 参数（示例）
    steps = 5000  # FHN
    # steps = 2000 # flow
    # steps = 500
    from datasets import dt, dataset_dict

    # dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取normalization参数与输入维度
    normalization = dataset_dict["info"]["normalization"]
    input_dim = len(normalization["input"]["mean"])

    # 根据系统类型生成合适的初始状态
    if CURRENT_SYSTEM == "flow" or CURRENT_SYSTEM == "flow-19" or CURRENT_SYSTEM == "flow-29-v":
        # 对于flow系统，先读取真实轨迹，然后使用其初值
        traj_true = true_trajectory(None, steps, dt)
        x0 = traj_true[0]  # 使用真实轨迹的初值
        print(f"使用flow系统真实轨迹的初值: {x0}")
    else:
        x0 = generate_adaptive_initial_state(CURRENT_SYSTEM, input_dim)
        # 生成真实轨迹
        traj_true = true_trajectory(x0, steps, dt)

    # 生成预测轨迹
    traj_pred = predict_trajectory_reptile(
        x0, steps, dt, normalization, model_path="finetune.pt", device=device
    )

    # 计算整体误差统计
    mae_overall = np.mean(np.abs(traj_true - traj_pred))
    mse_overall = np.mean((traj_true - traj_pred) ** 2)
    print(f"\n整体误差统计:")
    print(f"  平均绝对误差 (MAE): {mae_overall:.6f}")
    print(f"  均方误差 (MSE): {mse_overall:.6f}")

    # 导出轨迹数据
    export_trajectory_data(
        traj_true, traj_pred, dt, CURRENT_SYSTEM, tag=f"{slow_params_eval}", x0=x0
    )

    # 若为3维，绘制三维轨迹
    D = traj_true.shape[1]
    if D == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            traj_true[:, 0],
            traj_true[:, 1],
            traj_true[:, 2],
            color="black",
            lw=2,
            label="True trajectory",
        )
        ax.plot(
            traj_pred[:, 0],
            traj_pred[:, 1],
            traj_pred[:, 2],
            color="red",
            lw=2,
            linestyle="--",
            label="Predicted trajectory (Reptile)",
        )
        ax.set_xlabel(r"$x_1$", fontsize=18)
        ax.set_ylabel(r"$x_2$", fontsize=18)
        ax.set_zlabel(r"$x_3$", fontsize=18)
        ax.set_title("Trajectory Comparison", fontsize=20)
        ax.legend(fontsize=16)
        plt.tight_layout()
        plt.show()

    # 分量对比曲线
    t = np.arange(steps + 1) * dt
    fig2, axs = plt.subplots(D, 1, figsize=(8, 2.5 * D), sharex=True)
    if D == 1:
        axs = [axs]
    labels = [rf"$x_{{{i + 1}}}$" for i in range(D)]
    for i in range(D):
        axs[i].plot(t, traj_true[:, i], color="black", lw=2, label="True")
        axs[i].plot(
            t, traj_pred[:, i], color="red", lw=2, linestyle="--", label="Predicted (Reptile)"
        )
        axs[i].set_ylabel(labels[i], fontsize=16)
        axs[i].legend(fontsize=14, loc="upper right")
        axs[i].grid(True, linestyle=":")
    axs[-1].set_xlabel("Time $t$", fontsize=16)
    fig2.suptitle("Component-wise Trajectory Comparison", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
