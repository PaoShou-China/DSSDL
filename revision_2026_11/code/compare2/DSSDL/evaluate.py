import numpy as np
from datasets import f, slow_params_eval, CURRENT_SYSTEM
import torch


class QuadrotorController:
    """移植自create_dataset.py的四旋翼控制器"""

    def __init__(self, device="cpu"):
        # 物理参数
        self.k_T = 1e-6
        self.k_M = 1e-7
        self.l = 0.2
        self.MIN_OMEGA_SQ = 0.0
        self.total_thrust_base = 0.5

        # PD控制器参数
        self.Kp = torch.tensor([0.05, 0.05, 0.02], device=device)
        self.Kd = torch.tensor([0.01, 0.01, 0.005], device=device)
        self.device = device

    def update(self, state_tensor):
        """
        根据当前状态计算新的电机速度平方 (Ω²)。
        state_tensor: (1, 10) 的状态张量 [φ, θ, ψ, p, q, r, Ω1², Ω2², Ω3², Ω4²]
        返回: (1, 4) 的电机速度平方张量
        """
        is_numpy = isinstance(state_tensor, np.ndarray)
        if is_numpy:
            state_tensor = torch.from_numpy(state_tensor).to(self.device)

        phi, theta, psi = state_tensor[0, 0], state_tensor[0, 1], state_tensor[0, 2]
        p, q, r = state_tensor[0, 3], state_tensor[0, 4], state_tensor[0, 5]

        # PD 控制器 (目标姿态为0)
        angle_error = torch.tensor([0 - phi, 0 - theta, 0 - psi], device=self.device)
        angle_rate_error = torch.tensor([0 - p, 0 - q, 0 - r], device=self.device)
        tau_pd = self.Kp * angle_error + self.Kd * angle_rate_error

        # 控制分配
        tau_phi, tau_theta, tau_psi = tau_pd[0], tau_pd[1], tau_pd[2]

        omega1_sq = (
            self.total_thrust_base / (4 * self.k_T)
            - tau_theta / (2 * self.l * self.k_T)
            - tau_psi / (4 * self.k_M)
        )
        omega2_sq = (
            self.total_thrust_base / (4 * self.k_T)
            + tau_phi / (2 * self.l * self.k_T)
            + tau_psi / (4 * self.k_M)
        )
        omega3_sq = (
            self.total_thrust_base / (4 * self.k_T)
            + tau_theta / (2 * self.l * self.k_T)
            - tau_psi / (4 * self.k_M)
        )
        omega4_sq = (
            self.total_thrust_base / (4 * self.k_T)
            - tau_phi / (2 * self.l * self.k_T)
            + tau_psi / (4 * self.k_M)
        )

        motor_speeds_sq = torch.stack([omega1_sq, omega2_sq, omega3_sq, omega4_sq]).unsqueeze(0)
        motor_speeds_sq = torch.clamp(motor_speeds_sq, min=self.MIN_OMEGA_SQ)

        if is_numpy:
            return motor_speeds_sq.cpu().numpy()
        return motor_speeds_sq


def true_trajectory(x0, steps, dt, slow_params=slow_params_eval):
    """
    生成系统的真实轨迹（自适应维度）。
    x0: 初始状态，形如(D,)或(1,D)
    steps: 步数
    dt: 步长
    slow_params: 系统参数
    返回：(steps+1, D) 的轨迹
    """
    # print(slow_params)
    x0 = np.asarray(x0).reshape(1, -1)
    traj = [x0.copy()]
    x = x0.copy()

    controller = None
    if CURRENT_SYSTEM == "quadrotor":
        controller = QuadrotorController(device="cpu")

    for _ in range(steps):
        if controller:
            # 实时更新控制输入
            x[0, 6:] = controller.update(x)
        # 统一的二维接口：f(features, slow_params)
        # features: (N, D) 矩阵，slow_params: 参数列表
        if CURRENT_SYSTEM == "quadrotor":
            # 四旋翼系统: f只返回角加速度[p_dot, q_dot, r_dot]
            angular_accel = f(x, slow_params)  # Shape (1, 3)

            phi, theta, p, q, r = x[0, 0], x[0, 1], x[0, 3], x[0, 4], x[0, 5]

            # 计算欧拉角导数
            phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
            theta_dot = q * np.cos(phi) - r * np.sin(phi)
            psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

            # 完整的6维姿态导数
            full_dx_attitude = np.array(
                [
                    [
                        phi_dot,
                        theta_dot,
                        psi_dot,
                        angular_accel[0, 0],
                        angular_accel[0, 1],
                        angular_accel[0, 2],
                    ]
                ]
            )

            # 只更新姿态部分的状态，控制输入部分保持不变
            x[0, :6] = x[0, :6] + full_dx_attitude * dt
        else:
            # 其他系统: 标准欧拉积分
            dx = f(x, slow_params) * dt
            x = x + dx
        traj.append(x.copy())

    traj = np.concatenate(traj, axis=0)
    return traj


def predict_trajectory(
    x0,
    T,
    dt,
    normalization,
    model_path="model/finetune.pt",
    device="cpu",
    save_derivatives=False,
    use_raw_data=False,
):
    """
    使用finetune后的MetaNet模型生成预测轨迹（带标准化/反标准化，自适应维度）。
    x0: 初始状态，形如(D,)或(1,D)
    T: 步数（与steps等价）
    dt: 步长
    normalization: dict, 包含input/output的mean/std
    model_path: finetune.pt路径
    device: 设备
    save_derivatives: 是否保存真实导数和预测导数到txt文件
    use_raw_data: 是否使用原始数据（不标准化），默认False
    返回：(T+1, D) 的预测轨迹
    """
    import torch
    import sys

    sys.path.append(".")
    from DSSDL_ import MetaNet, config

    # 根据use_raw_data决定是否加载标准化参数
    if not use_raw_data:
        input_mean = torch.tensor(
            normalization["input"]["mean"], dtype=torch.float32, device=device
        )
        input_std = torch.tensor(normalization["input"]["std"], dtype=torch.float32, device=device)
        output_mean = torch.tensor(
            normalization["output"]["mean"], dtype=torch.float32, device=device
        )
        output_std = torch.tensor(
            normalization["output"]["std"], dtype=torch.float32, device=device
        )
    else:
        # use_raw_data=True时，不需要标准化参数
        input_mean = None
        input_std = None
        output_mean = None
        output_std = None

    import os
    import pandas as pd

    # 获取当前系统的features文件夹路径
    features_dir = f"datasets/{CURRENT_SYSTEM}/train/features"
    labels_dir = f"datasets/{CURRENT_SYSTEM}/train/labels"

    # 获取features文件夹下所有文件
    feature_files = [
        f for f in os.listdir(features_dir) if os.path.isfile(os.path.join(features_dir, f))
    ]
    one_hot_dim = len(feature_files)

    # 随便选一个features文件，读取其列数作为input_dim
    if one_hot_dim > 0:
        feature_sample_path = os.path.join(features_dir, feature_files[0])
        input_dim = pd.read_csv(feature_sample_path, sep=" ").shape[1]
    else:
        raise ValueError(f"No feature files found in {features_dir}.")

    # 随便选一个labels文件，读取其列数作为output_dim
    label_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))]
    if len(label_files) > 0:
        label_sample_path = os.path.join(labels_dir, label_files[0])
        output_dim = pd.read_csv(label_sample_path, sep=" ").shape[1]
    else:
        raise ValueError(f"No label files found in {labels_dir}.")

    from pretrain import CONFIG_OVERRIDES

    specific_param_dim = CONFIG_OVERRIDES["specific_param_dim"]
    hidden_widths = CONFIG_OVERRIDES["hidden_widths"]

    model = MetaNet(
        input_dim,
        output_dim,
        one_hot_dim,
        specific_param_dim,
        hidden_widths,
        device,
        dataset_dict=dataset_dict,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 自适应输入维度：按 normalization 的 input 维度对 x0 进行补零/截断
    if use_raw_data:
        # use_raw_data=True时，直接使用input_dim作为维度
        input_dim_norm = input_dim
    else:
        input_dim_norm = input_mean.numel()

    x0_np = np.asarray(x0).reshape(-1)
    if x0_np.size < input_dim_norm:
        x0_np = np.pad(x0_np, (0, input_dim_norm - x0_np.size))
    elif x0_np.size > input_dim_norm:
        x0_np = x0_np[:input_dim_norm]

    x = torch.tensor(x0_np, dtype=torch.float32, device=device).reshape(1, -1)
    traj = [x.cpu().numpy().copy()]

    # 用于保存导数的列表
    true_derivatives = []
    pred_derivatives = []

    controller = None
    if CURRENT_SYSTEM == "quadrotor":
        controller = QuadrotorController(device=device)

    for _ in range(T):
        if controller:
            # 实时更新控制输入
            x[:, 6:] = controller.update(x)

        # 根据 use_raw_data 决定是否进行标准化
        if use_raw_data:
            x_input = x
        else:
            # 1. 物理空间 -> 标准化
            x_input = (x - input_mean) / input_std

        # 2. 构造输入
        # one_hot = torch.zeros((1, one_hot_dim), device=device) # ones?
        # inp = torch.cat([x_input, one_hot], dim=1)
        with torch.no_grad():
            import datasets

            state_dim = x.shape[1]

            # 4. 物理空间积分
            if datasets.CURRENT_SYSTEM == "pendulum_real":
                # --- pendulum_real: 带特殊缩放的改进欧拉法 ---

                # --- 计算 k1 ---
                x_k1_scaled = x.clone()
                x_k1_scaled[:, 0] /= 20
                if not use_raw_data:
                    x_k1_input = (x_k1_scaled - input_mean) / input_std
                else:
                    x_k1_input = x_k1_scaled

                dx1_pred, dx1_hat_pred = model.finetune_forward(x_k1_input)
                if use_raw_data:
                    k1_unscaled = dx1_pred + dx1_hat_pred
                else:
                    k1_unscaled = (dx1_pred + dx1_hat_pred) * output_std + output_mean

                k1 = k1_unscaled.clone()
                k1[:, 0] *= 100
                k1[:, 1] *= 20

                # --- 计算 k2 ---
                x_pred = x + k1 * dt
                x_k2_scaled = x_pred.clone()
                x_k2_scaled[:, 0] /= 20
                if not use_raw_data:
                    x_k2_input = (x_k2_scaled - input_mean) / input_std
                else:
                    x_k2_input = x_k2_scaled

                dx2_pred, dx2_hat_pred = model.finetune_forward(x_k2_input)
                if use_raw_data:
                    k2_unscaled = dx2_pred + dx2_hat_pred
                else:
                    k2_unscaled = (dx2_pred + dx2_hat_pred) * output_std + output_mean

                k2 = k2_unscaled.clone()
                k2[:, 0] *= 100
                k2[:, 1] *= 20

                # --- 最终更新 ---
                dx = 0.99 * k1 + 0.01 * k2
                x = x + dx * dt

            else:
                # --- 其他系统: 标准欧拉法 ---
                if not use_raw_data:
                    x_input = (x - input_mean) / input_std
                else:
                    x_input = x

                dx_pred, dx_hat_pred = model.finetune_forward(x_input)
                # dx_pred = torch.zeros_like(dx_pred)
                if use_raw_data:
                    dx_out = dx_pred + dx_hat_pred
                else:
                    dx_out = (dx_pred + dx_hat_pred) * output_std + output_mean

                out_dim = dx_out.shape[1]
                if out_dim == state_dim:
                    dx = dx_out
                elif state_dim == 4 and out_dim == 2:
                    vx_vy = x[:, 2:4]
                    dx = torch.cat([vx_vy, dx_out], dim=1)
                elif CURRENT_SYSTEM == "quadrotor" and state_dim == 10 and out_dim == 3:
                    # Quadrotor: model predicts [p_dot, q_dot, r_dot], state is 10D
                    angular_accel = dx_out

                    phi, theta, p, q, r = x[0, 0], x[0, 1], x[0, 3], x[0, 4], x[0, 5]

                    # Calculate euler angle derivatives (kinematics) using torch
                    phi_dot = p + (q * torch.sin(phi) + r * torch.cos(phi)) * torch.tan(theta)
                    theta_dot = q * torch.cos(phi) - r * torch.sin(phi)
                    psi_dot = (q * torch.sin(phi) + r * torch.cos(phi)) / torch.cos(theta)

                    # Combine kinematics and dynamics from model
                    full_dx_attitude = torch.stack(
                        [
                            phi_dot,
                            theta_dot,
                            psi_dot,
                            angular_accel[0, 0],
                            angular_accel[0, 1],
                            angular_accel[0, 2],
                        ]
                    ).unsqueeze(0)

                    # Create full derivative vector (control inputs' derivatives are zero)
                    dx = torch.cat([full_dx_attitude, torch.zeros((1, 4), device=device)], dim=1)
                else:
                    raise RuntimeError(
                        f"Output dim {out_dim} not compatible with state dim {state_dim}"
                    )

                x = x + dx * dt

            # 保存预测导数和真实导数 (仅在标准模式下有意义)
            if save_derivatives and datasets.CURRENT_SYSTEM != "pendulum_real":
                # 保存与f()输出维度一致的导数
                if datasets.CURRENT_SYSTEM == "quadrotor":
                    pred_derivatives.append(dx_out.cpu().numpy().copy())  # dx_out is 3D
                else:
                    pred_derivatives.append(dx.cpu().numpy().copy())
                x_np = x.cpu().numpy().reshape(-1)
                true_deriv = f(x_np.reshape(1, -1), slow_params_eval).reshape(-1)
                true_derivatives.append(true_deriv.copy())
        traj.append(x.cpu().numpy().copy())

    traj = np.concatenate(traj, axis=0)

    # 保存导数到文件
    if save_derivatives and len(true_derivatives) > 0:
        true_derivatives = np.array(true_derivatives).squeeze()
        pred_derivatives = np.array(pred_derivatives).squeeze()

        # 创建保存目录
        derivatives_dir = f"derivatives/{CURRENT_SYSTEM}"
        os.makedirs(derivatives_dir, exist_ok=True)

        # 保存到txt文件
        true_deriv_path = os.path.join(derivatives_dir, "true_derivatives.txt")
        pred_deriv_path = os.path.join(derivatives_dir, "pred_derivatives.txt")

        np.savetxt(true_deriv_path, true_derivatives)
        np.savetxt(pred_deriv_path, pred_derivatives)

        print(f"✓ 真实导数已保存到: {true_deriv_path}")
        print(f"✓ 预测导数已保存到: {pred_deriv_path}")

        # 计算导数误差统计
        deriv_mae = np.mean(np.abs(true_derivatives - pred_derivatives))
        deriv_mse = np.mean((true_derivatives - pred_derivatives) ** 2)
        print(f"导数误差统计:")
        print(f"  平均绝对误差 (MAE): {deriv_mae:.6f}")
        print(f"  均方误差 (MSE): {deriv_mse:.6f}")

    return traj


def generate_adaptive_initial_state(system_name, input_dim):
    """
    根据系统类型和输入维度生成合适的初始状态
    """
    if system_name == "pendulum":
        # 摆系统：[phi_dot, phi] 或扩展维度
        base_state = [np.pi / 4, 0.0]  # 初始角速度和角度
    elif system_name == "lorenz" or system_name == "lorenz_exact":
        # 洛伦兹系统：[x, y, z] 或扩展维度
        base_state = [8.0, -8.0, 27.0]  # 经典初始状态
        # base_state = [0.1, 0, 0]
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
        if system_name == "quadrotor":
            # Quadrotor: 根据create_dataset.py的范围生成姿态部分
            np.random.seed(42)
            attitude_state = np.random.uniform(-0.5, 0.5, 6)
            # 使用控制器计算初始的控制输入
            temp_state_10d = np.pad(attitude_state, (0, 4))  # 临时补零以匹配控制器输入维度
            controller = QuadrotorController(device="cpu")
            control_initial = controller.update(temp_state_10d.reshape(1, -1)).flatten()
            base_state = np.concatenate([attitude_state, control_initial])
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


def generate_multiple_initial_states(system_name, input_dim, num_states=4):
    """
    生成多个不同的初始状态用于相图绘制
    """
    initial_states = []

    if system_name == "pendulum":
        # 摆系统的不同初始状态
        base_states = [
            [np.pi / 6, 0.0],  # 小角度
            [np.pi / 4, 0.0],  # 中等角度
            [np.pi / 3, 0.0],  # 较大角度
            [0.0, np.pi / 4],  # 不同的初始角速度
        ]
    elif system_name == "lorenz" or system_name == "lorenz_exact":
        # 洛伦兹系统的不同初始状态
        base_states = [
            [8.0, -8.0, 27.0],  # 经典初始状态
            [-10.0, 10.0, 25.0],  # 变化1
            [-6.0, 6.0, 30.0],  # 变化2
            [-12.0, 12.0, 28.0],  # 变化3
        ]
    elif system_name == "double_pendulum":
        # 双摆系统的不同初始状态
        base_states = [
            [np.pi / 6, np.pi / 6, 0, 0],
            [np.pi / 4, np.pi / 4, 0, 0],
            [np.pi / 3, np.pi / 6, 0, 0],
            [np.pi / 6, np.pi / 3, 0, 0],
        ]
    elif system_name == "van_der_pol":
        # Van der Pol振荡器的不同初始状态
        base_states = [
            [-1.2, -2],
            [1.5, 0.0],
            [-0.5, 2.0],
            [0.8, -2.2],
            [1, 0],
            [1.5, 0],
            [2, 0],
            [2.5, 0],
            [3, 0],
            [3.5, 0],
            [4, 0],
            [4.5, 0],
            [5, 0],
            [5.5, 0],
        ]
    else:
        # 通用系统：生成不同的随机初始状态
        np.random.seed(42)
        base_states = []
        for i in range(num_states):
            np.random.seed(42 + i)  # 不同的种子
            state = np.random.uniform(-2.0, 2.0, min(input_dim, 6)).tolist()
            base_states.append(state)

    # 调整每个状态到所需维度
    for base_state in base_states:
        x0_arr = np.asarray(base_state).reshape(-1)
        if x0_arr.size < input_dim:
            x0_arr = np.pad(x0_arr, (0, input_dim - x0_arr.size))
        elif x0_arr.size > input_dim:
            x0_arr = x0_arr[:input_dim]
        initial_states.append(x0_arr)

    return initial_states[:num_states]


def plot_trajectory_adaptive(traj_true, traj_pred, dt, system_name=None, save_dir=None):
    """
    自适应绘制轨迹图，支持不同维度的系统

    Args:
        traj_true: 真实轨迹
        traj_pred: 预测轨迹
        dt: 时间步长
        system_name: 系统名称
        save_dir: 保存目录，如果为None则不保存
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import os

    # 设置科研标准字体
    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = 16

    D = traj_true.shape[1]
    steps = traj_true.shape[0] - 1
    t = np.arange(steps + 1) * dt

    print(f"绘制 {D}D 系统轨迹，系统类型: {system_name}")

    # 自适应维度的时序图（分量对比图）
    fig, axs = plt.subplots(D, 1, figsize=(10, 2.5 * D), sharex=True)
    if D == 1:
        axs = [axs]

    # 根据系统类型生成标签
    if system_name == "pendulum" and D >= 2:
        labels = [r"$\dot{\phi}$", r"$\phi$"] + [rf"$x_{{{i + 1}}}$" for i in range(2, D)]
    elif system_name == "lorenz" and D >= 3:
        labels = ["x", "y", "z"] + [rf"$x_{{{i + 1}}}$" for i in range(3, D)]
    elif system_name == "double_pendulum" and D >= 4:
        labels = ["theta1", "theta2", "omega1", "omega2"] + [
            rf"$x_{{{i + 1}}}$" for i in range(4, D)
        ]
    else:
        labels = [rf"$x_{{{i + 1}}}$" for i in range(D)]

    for i in range(D):
        axs[i].plot(t, traj_true[:, i], color="black", lw=2, label="True")
        axs[i].plot(t, traj_pred[:, i], color="red", lw=2, linestyle="--", label="Predicted")
        axs[i].set_ylabel(labels[i], fontsize=16)
        axs[i].legend(fontsize=12, loc="upper right")
        axs[i].grid(True, linestyle=":")

        # 计算误差统计
        error = np.abs(traj_true[:, i] - traj_pred[:, i])
        mae = np.mean(error)
        axs[i].text(
            0.02,
            0.98,
            f"MAE: {mae:.4f}",
            transform=axs[i].transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    axs[-1].set_xlabel("Time $t$", fontsize=16)
    title = f"{system_name.capitalize()} System" if system_name else "Dynamic System"
    fig.suptitle(f"{title} - Trajectory Comparison", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存时序对比图
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"trajectory_{system_name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"✓ 轨迹对比图已保存: {filepath}")

    plt.show()


# Override with extended version that also plots no-finetune prediction if provided.
def plot_trajectory_adaptive(
    traj_true, traj_pred, dt, system_name=None, save_dir=None, traj_pred_noft=None
):
    """
    自适应绘制轨迹图，支持 finetune 与 no-finetune 两条预测曲线。
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import os

    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = 16

    D = traj_true.shape[1]
    steps = traj_true.shape[0] - 1
    t = np.arange(steps + 1) * dt

    print(f"绘制 {D}D 系统轨迹，系统类型: {system_name}")

    fig, axs = plt.subplots(D, 1, figsize=(10, 2.5 * D), sharex=True)
    if D == 1:
        axs = [axs]

    if system_name == "pendulum" and D >= 2:
        labels = [r"$\dot{\phi}$", r"$\phi$"] + [rf"$x_{{{i + 1}}}$" for i in range(2, D)]
    elif system_name == "lorenz" and D >= 3:
        labels = ["x", "y", "z"] + [rf"$x_{{{i + 1}}}$" for i in range(3, D)]
    elif system_name == "double_pendulum" and D >= 4:
        labels = ["theta1", "theta2", "omega1", "omega2"] + [
            rf"$x_{{{i + 1}}}$" for i in range(4, D)
        ]
    else:
        labels = [rf"$x_{{{i + 1}}}$" for i in range(D)]

    for i in range(D):
        axs[i].plot(t, traj_true[:, i], color="black", lw=2, label="True")
        axs[i].plot(
            t, traj_pred[:, i], color="red", lw=2, linestyle="--", label="Predicted (finetune)"
        )
        if traj_pred_noft is not None:
            axs[i].plot(
                t,
                traj_pred_noft[:, i],
                color="blue",
                lw=2,
                linestyle=":",
                label="Predicted (no finetune)",
            )
        axs[i].set_ylabel(labels[i], fontsize=16)
        axs[i].legend(fontsize=12, loc="upper right")
        axs[i].grid(True, linestyle=":")

        error = np.abs(traj_true[:, i] - traj_pred[:, i])
        mae = np.mean(error)
        axs[i].text(
            0.02,
            0.98,
            f"MAE: {mae:.4f}",
            transform=axs[i].transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    axs[-1].set_xlabel("Time $t$", fontsize=16)
    title = f"{system_name.capitalize()} System" if system_name else "Dynamic System"
    fig.suptitle(f"{title} - Trajectory Comparison", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"trajectory_{system_name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"轨迹对比图已保存: {filepath}")

    plt.show()


def plot_phase_diagram_multiple(trajectories_data, dt, system_name=None, save_dir=None):
    """
    绘制多个初始值的相图在一个图中

    Args:
        trajectories_data: list of dict, 每个dict包含 {'traj_true': array, 'traj_pred': array, 'x0': array}
        dt: 时间步长
        system_name: 系统名称
        save_dir: 保存目录
    """
    if not trajectories_data or trajectories_data[0]["traj_true"].shape[1] < 2:
        return

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import os

    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = 14

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    for i, data in enumerate(trajectories_data):
        traj_true = data["traj_true"]
        traj_pred = data["traj_pred"]
        x0 = data["x0"]

        x = traj_true[:, 0]
        v = traj_true[:, 1]
        x_pred = traj_pred[:, 0]
        v_pred = traj_pred[:, 1]
        zeros = np.zeros_like(x)

        # 绘制轨迹 - 使用与原图相同的格式
        ax.plot(v, x, zeros, color="navy", lw=1.0, label="True" if i == 0 else None)
        ax.plot(
            v_pred,
            x_pred,
            zeros,
            color="crimson",
            lw=1.1,
            linestyle="-",
            label="Predicted" if i == 0 else None,
        )

        # 标记起始点 - 使用与原图相同的格式
        ax.scatter(v_pred[0], x_pred[0], 0, color="crimson", marker="o", s=45)

    ax.set_xlabel("V", labelpad=12)
    ax.set_ylabel("X", labelpad=12)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_zlabel("")
    ax.set_zlim(-1e-3, 1e-3)
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.set_box_aspect((1, 1, 0.0000001))

    try:
        ax.w_zaxis.line.set_lw(0.0)
    except AttributeError:
        pass
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False

    title = (
        f"{system_name.capitalize()} Phase Portrait (Multiple Initial Values)"
        if system_name
        else "Phase Portrait (Multiple Initial Values)"
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.view_init(elev=40, azim=-60)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"phase_multiple_{system_name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"✓ 多初值相图已保存: {filepath}")

    plt.show()


def plot_phase_diagram(traj_true, traj_pred, dt, system_name=None, save_dir=None):
    """
    状态维度>=2时绘制二维相图 (v-x) 并设置倾斜视角。
    """
    if traj_true.shape[1] < 2:
        return

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import os

    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = 14

    x = traj_true[:, 0]
    v = traj_true[:, 1]
    x_pred = traj_pred[:, 0]
    v_pred = traj_pred[:, 1]

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    zeros = np.zeros_like(x)
    ax.plot(v, x, zeros, color="navy", lw=1.5, label="True")
    ax.plot(v_pred, x_pred, zeros, color="crimson", lw=1.6, linestyle="--", label="Predicted")
    # ax.scatter(v[0], x[0], 0, color='navy', marker='x', s=55)
    ax.scatter(v_pred[0], x_pred[0], 0, color="crimson", marker="o", s=45)
    ax.set_xlabel("V", labelpad=12)
    ax.set_ylabel("X", labelpad=12)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_zlabel("")
    ax.set_zlim(-1e-3, 1e-3)
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.set_box_aspect((1, 1, 0.0000001))
    try:
        ax.w_zaxis.line.set_lw(0.0)
    except AttributeError:
        pass
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    title = f"{system_name.capitalize()} Phase Portrait" if system_name else "Phase Portrait"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.view_init(elev=25, azim=-60)
    # ax.grid(None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"phase_{system_name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"相图已保存: {filepath}")

    plt.show()


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

    directory = os.path.join("final_data", "trajectories_data", system_name)
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
    print(f"评估 {CURRENT_SYSTEM} 系统的轨迹预测性能")
    print("=" * 50)

    # 基本参数
    steps = 5000
    from datasets import dt, dataset_dict

    # dt = 0.01
    device = "cpu"

    # 从pretrain.py导入use_raw_data配置
    try:
        from pretrain import CONFIG_OVERRIDES

        use_raw_data = CONFIG_OVERRIDES.get("use_raw_data", False)
    except ImportError:
        use_raw_data = False

    print(f"使用原始数据模式: {use_raw_data}")

    # 获取系统信息
    normalization = dataset_dict["info"]["normalization"]
    input_dim = len(normalization["input"]["mean"])
    output_dim = len(normalization["output"]["mean"])

    print(f"系统维度: 输入{input_dim}D, 输出{output_dim}D")
    print(f"时间步长: {dt}, 预测步数: {steps}")

    import datasets
    import os

    if datasets.CURRENT_SYSTEM == "pendulum_real":
        print("真实单摆系统评估模式")
        # 直接从query集加载真实轨迹
        query_features_path = os.path.join(
            "datasets", "pendulum_real", "query", "features", f"{slow_params_eval[0]:.2f}.txt"
        )
        print(f"从 {query_features_path} 加载真实轨迹...")
        traj_true = np.loadtxt(query_features_path)
        traj_true[:, 0] *= 20
        traj_true = traj_true[5:]  # 去掉前10个点的过渡
        # 使用真实轨迹的第一行作为初始状态
        x0_arr = traj_true[0]

        # 更新步数以匹配数据长度
        steps = len(traj_true) - 1
        print(f"从数据加载的轨迹: shape={traj_true.shape}, steps={steps}")
        print(f"使用数据的第一行作为初始状态: {x0_arr}")

    elif (
        datasets.CURRENT_SYSTEM == "flow"
        or datasets.CURRENT_SYSTEM == "flow-19"
        or datasets.CURRENT_SYSTEM == "flow-19-v"
        or datasets.CURRENT_SYSTEM == "flow-29-v"
    ):
        print("流体系统评估模式")
        # 直接从query集加载真实轨迹
        query_features_path = os.path.join(
            "datasets", datasets.CURRENT_SYSTEM, "query", "features", f"{slow_params_eval[0]}.txt"
        )
        print(f"从 {query_features_path} 加载真实轨迹...")
        traj_true = np.loadtxt(query_features_path)
        # 使用真实轨迹的第一行作为初始状态
        x0_arr = traj_true[0]

        # 更新步数以匹配数据长度
        steps = len(traj_true) - 1
        steps = 200
        print(f"从数据加载的轨迹: shape={traj_true.shape}, steps={steps}")
        print(f"使用数据的第一行作为初始状态: {x0_arr}")

    else:
        print("标准系统评估模式")
        # 生成单个初始状态用于分量变化图
        x0_arr = generate_adaptive_initial_state(CURRENT_SYSTEM, input_dim)
        print(f"主要初始状态: {x0_arr}")

        # 生成轨迹用于分量变化图
        print("生成真实轨迹...")
        traj_true = true_trajectory(x0_arr, steps, dt)
    print("生成预测轨迹...")
    traj_pred = predict_trajectory(
        x0_arr,
        steps,
        dt,
        normalization,
        model_path="model/finetune.pt",
        device=device,
        save_derivatives=True,
        use_raw_data=use_raw_data,
    )
    # 不finetune的预测（若模型存在）
    try:
        traj_pred_noft = predict_trajectory(
            x0_arr,
            steps,
            dt,
            normalization,
            model_path="model/no_finetune.pt",
            device=device,
            save_derivatives=False,
            use_raw_data=use_raw_data,
        )
        has_noft = True
    except FileNotFoundError:
        traj_pred_noft = None
        has_noft = False
    print("slow_params_eval: ", slow_params_eval)
    # INSERT_YOUR_CODE
    # 按step裁剪真实轨迹和预测轨迹，只保留[steps+1]行（可能比数据实际行数少）
    traj_true = traj_true[: steps + 1]
    traj_pred = traj_pred[: steps + 1]
    if traj_pred_noft is not None:
        traj_pred_noft = traj_pred_noft[: steps + 1]
    # 计算整体误差统计
    mae_overall = np.mean(np.abs(traj_true - traj_pred))
    mse_overall = np.mean((traj_true - traj_pred) ** 2)
    print(f"\n整体误差统计:")
    print(f"  平均绝对误差 (MAE): {mae_overall:.6f}")
    print(f"  均方误差 (MSE): {mse_overall:.6f}")
    if has_noft:
        mae_noft = np.mean(np.abs(traj_true - traj_pred_noft))
        mse_noft = np.mean((traj_true - traj_pred_noft) ** 2)
        print(f"  不finetune MAE: {mae_noft:.6f}")
        print(f"  不finetune MSE: {mse_noft:.6f}")

    # 导出主要初始状态的轨迹数据
    export_trajectory_data(
        traj_true, traj_pred, dt, CURRENT_SYSTEM, tag=f"{slow_params_eval}", x0=x0_arr
    )
    if has_noft:
        export_trajectory_data(
            traj_true,
            traj_pred_noft,
            dt,
            CURRENT_SYSTEM,
            tag=f"{slow_params_eval}_no_finetune",
            x0=x0_arr,
        )

    # 创建保存目录
    save_dir = f"pictures/{CURRENT_SYSTEM}"
    print(f"图片保存目录: {save_dir}")

    # 绘制分量变化图（保持不变）
    plot_trajectory_adaptive(
        traj_true, traj_pred, dt, CURRENT_SYSTEM, save_dir, traj_pred_noft=traj_pred_noft
    )

    # 生成多个初始状态用于相图绘制
    # print("\n生成多个初始状态用于相图绘制...")
    # initial_states = generate_multiple_initial_states(CURRENT_SYSTEM, input_dim, num_states=6)
    # trajectories_data = []

    # for i, x0 in enumerate(initial_states):
    #     print(f"处理初始状态 {i+1}: {x0}")
    #     traj_true_i = true_trajectory(x0, steps, dt)
    #     traj_pred_i = predict_trajectory(x0, steps, dt, normalization, model_path='model/finetune.pt', device=device, use_raw_data=use_raw_data)

    #     export_trajectory_data(traj_true_i, traj_pred_i, dt, CURRENT_SYSTEM, tag=f'initial_{i+1}', x0=x0)

    #     trajectories_data.append({
    #         'traj_true': traj_true_i,
    #         'traj_pred': traj_pred_i,
    #         'x0': x0
    #     })

    # 绘制多初值相图
    # print("绘制多初值相图...")
    # plot_phase_diagram_multiple(trajectories_data, dt, CURRENT_SYSTEM, save_dir)

    # 也保留原来的单初值相图
    plot_phase_diagram(traj_true, traj_pred, dt, CURRENT_SYSTEM, save_dir)
