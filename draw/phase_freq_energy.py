# cdef.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# --- Global Style Settings ---
def apply_global_styles():
    """统一Matplotlib样式设置"""
    plt.rcParams.update({
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
    print("Global Matplotlib styles applied for phase_freq_energy.py.")


# --- Color Palette ---
nature_colors = ['#403990', '#80A6E2', '#F46F43', '#CF3D3E', '#008C95']
ref_color = nature_colors[3]  # 使用标准配色方案中的红色
pred_color = nature_colors[1]  # 使用标准配色方案中的蓝色
nature_cmap = LinearSegmentedColormap.from_list("nature_cmap", nature_colors)

# --- Physical Parameters (for energy calculation, can be approximate) ---
G = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0  # Pendulum length (m)
M = 0.5  # Mass (kg)


def load_data_from_csv(duration=10.0):
    """Loads, scales, and truncates trajectory data from the CSV file."""
    print(f"Loading trajectory data for the first {duration}s from CSV...")
    file_path = 'final_data/trajectories_data/pendulum_real/[8].csv'
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        mask = data[:, 0] <= duration
        t = data[mask, 0]
        true_solution = data[mask][:, [3, 1]]
        pred_solution = data[mask][:, [4, 2]]
        print(f"Successfully loaded and processed data from {file_path}")
        return t, true_solution, pred_solution
    except Exception as e:
        print(f"Could not load data from {file_path}: {e}")
        return None, None, None


# --- Functions to CREATE (but not show) each figure ---

def _set_axis_style(ax):
    # 移除背景色设置
    # 保持所有边框可见
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
    # 添加网格线，四边封口
    ax.grid(True, which='both', linestyle=':', alpha=0.6)



def create_figure_d(true_solution, pred_solution, colors):
    """Creates the phase portrait figure object comparing true vs. predicted."""
    true_theta, true_omega = true_solution.T
    pred_theta, pred_omega = pred_solution.T

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _set_axis_style(ax)
    ax.plot(true_theta, true_omega, color=ref_color, linewidth=3.5, linestyle='--', label='Ref.')
    ax.plot(pred_theta, pred_omega, color=pred_color, linestyle='-', linewidth=3.5, label='Pred.')
    ax.plot(true_theta[0], true_omega[0], 'o', color=nature_colors[1], mec='k', ms=10)
    ax.set_xlabel(r'Angle $\theta$ (rad)', fontsize=20, labelpad=3)
    ax.set_ylabel(r'Angular Velocity $\dot{\theta}$ (rad/s)', fontsize=20, labelpad=10, rotation=90, va='center')
    ax.set_ylim(-6, 7)
    ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=18)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
    plt.show()


def create_figure_e(t, true_solution, pred_solution, colors):
    """Creates the frequency spectrum figure object for the true and predicted data."""
    true_theta, _ = true_solution.T
    pred_theta, _ = pred_solution.T
    if len(t) < 2:
        print("Not enough data points for FFT after truncation.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _set_axis_style(ax)
    N, dt = len(t), t[1] - t[0]

    def _spectrum(signal):
        yf = fft(signal)
        xf = fftfreq(N, dt)[:N // 2]
        spec = 2.0 / N * np.abs(yf[0:N // 2])
        return xf, spec

    xf_true, spec_true = _spectrum(true_theta)
    xf_pred, spec_pred = _spectrum(pred_theta)

    ax.plot(xf_true, spec_true, color=ref_color, linestyle='--', linewidth=3.5, label='Ref.')
    ax.plot(xf_pred, spec_pred, color=pred_color, linestyle='-', linewidth=3.5, label='Pred.')

    peak_freq_true = xf_true[np.argmax(spec_true)]
    ax.axvline(peak_freq_true, color=ref_color, linestyle=':', linewidth=2.5)

    ax.set_xlabel('Frequency (Hz)', fontsize=20, labelpad=3)
    ax.set_ylabel(r'Amplitude $|\Theta(f)|$', fontsize=20, labelpad=10, rotation=90, va='center')
    ax.set_xlim(0, 1.5)
    ax.legend(frameon=False, loc='upper left', fontsize=18)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
    plt.show()


def moving_average(data, window_size):
    """Calculates the moving average of a 1D array using convolution."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def create_figure_f(t, true_solution, pred_solution, colors):
    """Creates the energy decay figure object for true and predicted data."""
    true_theta, true_omega = true_solution.T
    pred_theta, pred_omega = pred_solution.T
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _set_axis_style(ax)

    L_rod, m_rod, m_disk, R, d = 0.38, 0.1, 0.1, 0.1, 0.32
    I_rod = (1/3) * m_rod * L_rod**2
    I_disk = (1/4) * m_disk * R**2 + m_disk * d**2
    I_total = I_rod + I_disk
    L_cm = (m_rod * (L_rod / 2) + m_disk * d) / (m_rod + m_disk)

    def _energy(theta, omega):
        kinetic_energy = 0.5 * I_total * (omega**2)
        potential_energy = (m_rod + m_disk) * G * L_cm * (1 - np.cos(theta))
        return kinetic_energy + potential_energy

    energy_true = _energy(true_theta, true_omega)
    energy_pred = _energy(pred_theta, pred_omega)

    window_size = 100
    smoothed_true = moving_average(energy_true, window_size)
    smoothed_pred = moving_average(energy_pred, window_size)
    t_smoothed = t[window_size - 1:]

    ax.plot(t_smoothed, smoothed_true, label='Ref.', color=ref_color, linestyle='--', linewidth=3.5)
    ax.plot(t_smoothed, smoothed_pred, label='Pred.', color=pred_color, linestyle='-', linewidth=3.5)
    ax.set_xlabel('Time (s)', fontsize=20, labelpad=3)
    ax.set_ylabel('Energy (J)', fontsize=20, labelpad=10, rotation=90, va='center')
    ax.legend(frameon=False, loc='upper right', fontsize=18)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
    plt.show()


def create_and_show_all_plots_simultaneously():
    """Main function to load data and create all figures."""
    print("Generating figures from CSV data...")

    t, true_solution, pred_solution = load_data_from_csv(duration=70.0)

    if t is None or len(t) == 0:
        print("Failed to load or no data in the specified duration. Aborting plot generation.")
        return

    step_limit = min(len(t), 100)  # only the first 100 steps for c/d
    t_short = t[:step_limit]
    true_short = true_solution[:step_limit]
    pred_short = pred_solution[:step_limit]

    create_figure_d(true_short, pred_short, nature_colors)
    create_figure_e(t, true_solution, pred_solution, nature_colors)
    create_figure_f(t, true_solution, pred_solution, nature_colors)

    print("Displaying all plots. Close plot windows to continue.")


if __name__ == '__main__':
    apply_global_styles()
    create_and_show_all_plots_simultaneously()
