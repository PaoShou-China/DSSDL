# plot_L_embedding_tsne.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    print("Global Matplotlib styles applied for param_length.py.")


# --- Color Palette ---
nature_colors = ['#403990', '#80A6E2', '#FBDD85', '#F46F43', '#CF3D3E', '#008C95']


def load_learned_params():
    """Load learned parameters from the CSV file."""
    try:
        p = Path(__file__).resolve().parent.parent / 'final_data' / 'hidden_params' / 'pendulum_real' / 'learned_params.csv'
        if not p.exists():
            p = Path('final_data/hidden_params/pendulum_real/learned_params.csv')
        df = pd.read_csv(p)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find learned_params.csv at {p}")
        return None


def _set_axis_style(ax):
    """设置坐标轴样式，与其他绘图文件保持一致"""
    # 移除背景色设置
    # 保持所有边框可见
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
    # 添加网格线，四边封口
    ax.grid(True, which='both', linestyle=':', alpha=0.6)


def plot_learned_param_vs_length():
    """(b) Plot Learned Parameter vs. Actual Length."""
    print("Generating Figure: Learned Parameter vs. Actual Length...")

    df_learned = load_learned_params()
    if df_learned is None:
        return

    actual_lengths = np.linspace(9, 32, 7)
    learned_param_dim = df_learned['param1'].values

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _set_axis_style(ax)

    ax.scatter(
        learned_param_dim,
        actual_lengths,
        color=nature_colors[1],
        edgecolors='k',
        linewidths=0.4,
        s=120,
        alpha=0.9,
        label='Data Points'
    )

    coeffs = np.polyfit(learned_param_dim, actual_lengths, 1)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(learned_param_dim.min(), learned_param_dim.max(), 100)
    ax.plot(x_fit, poly(x_fit), color=nature_colors[3], linestyle='--', linewidth=3.5, label='Linear Fit')

    ax.set_xlabel('Learned Parameter Dimension 1', fontsize=20, labelpad=3)
    ax.set_ylabel('Actual Pendulum Length (cm)', fontsize=20, labelpad=10, rotation=90, va='center')
    ax.set_title('Pendulum Length vs Learned Param')

    ax.legend(frameon=False, loc='best', fontsize=18)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
    plt.show()

if __name__ == '__main__':
    apply_global_styles()
    plot_learned_param_vs_length()
