# plot_h_pretrain_loss_and_dim.py

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


# --- Global Style Settings (Nature Style) ---
# Duplicated for standalone execution
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
        'lines.linewidth': 2.2,
        'lines.markersize': 7,
        'axes.unicode_minus': False,
    })
    print("Global Matplotlib styles applied for pretrain_loss_dim.py.")


# --- Color Palette ---
# Duplicated for standalone execution
nature_colors = ['#403990', '#80A6E2', '#F46F43', '#CF3D3E', '#008C95']
nature_cmap = LinearSegmentedColormap.from_list(  # Duplicated for standalone execution
    "nature_cmap",
    nature_colors
)


def load_pretrain_loss():
    """Load pretrain loss data from CSV."""
    p = Path(__file__).resolve().parent.parent / 'final_data' / 'pretrain' / 'pendulum_real' / 'pretrain_loss.csv'
    if not p.exists():
        p = Path('final_data/pretrain/pendulum_real/pretrain_loss.csv')
    try:
        return pd.read_csv(p)
    except FileNotFoundError:
        print(f"Error: Could not find pretrain_loss.csv at {p}")
        return None


def load_mask_curve():
    """Load mask curve data from CSV."""
    p = Path(__file__).resolve().parent.parent / 'final_data' / 'mask' / 'pendulum_real' / 'mask_curve.csv'
    if not p.exists():
        p = Path('final_data/mask/pendulum_real/mask_curve.csv')
    try:
        return pd.read_csv(p)
    except FileNotFoundError:
        print(f"Warning: Could not find mask_curve.csv at {p}")
        return None


def plot_pretrain_loss_and_dim():
    """(h) Plot pretrain loss and remaining dimension curves."""
    print("Generating Figure (h): Pretrain Loss + Remaining Dim...")
    df_loss = load_pretrain_loss()
    if df_loss is None:
        return

    max_epoch = 40
    df_loss = df_loss[df_loss['epoch'] <= max_epoch]

    epoch_loss = df_loss['epoch'].values
    if 'train_rmse' in df_loss.columns:
        loss_values = df_loss['train_rmse'].values
    else:
        loss_values = df_loss['loss'].values

    df_mask = load_mask_curve()
    handles = []
    labels = []

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    if df_mask is not None:
        df_mask = df_mask[df_mask['epoch'] <= max_epoch]
        mask_cols = [c for c in df_mask.columns if c.startswith('mask_') and c not in ('mask_mean', 'mask_sparsity')]
        if mask_cols:
            threshold = 0.5
            try:
                mask_values = df_mask[mask_cols].values
                remaining_dim = (mask_values >= threshold).sum(axis=1)
                epoch_mask = df_mask['epoch'].values

                # Plot remaining dimension on left y-axis (ax1)
                ax1.plot(epoch_mask, remaining_dim, color=nature_colors[3], linestyle='--', linewidth=2.2, label='Remaining Dim')
                ax1.set_xlabel('Epoch', fontsize=20, labelpad=3)
                ax1.set_ylabel('Remaining Dimension', fontsize=20, labelpad=15, rotation=90, va='center', color=nature_colors[3])
                ax1.set_ylim(0, max(remaining_dim.max(), 1) + 0.5)

                # Create twin axis for pretrain loss on right y-axis
                ax2 = ax1.twinx()
                ax2.plot(epoch_loss, loss_values, color=nature_colors[0], linewidth=2.2, label='Pretrain Loss')
                ax2.set_ylabel('RMSE', fontsize=20, labelpad=15, rotation=270, va='center', color=nature_colors[0])
                ax2.set_yscale('log')

                # Collect legends
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h1 + h2
                labels += l1 + l2
            except Exception as e:
                print(f"Warning: Error processing mask curve data: {e}")
                # Fallback to plotting only loss
                ax1.plot(epoch_loss, loss_values, color=nature_colors[0], linewidth=2.2, label='Pretrain Loss')
                ax1.set_xlabel('Epoch', fontsize=20, labelpad=10)
                ax1.set_ylabel('RMSE', fontsize=20, labelpad=15, rotation=90, va='center', color=nature_colors[0])
                ax1.set_yscale('log')
                h1, l1 = ax1.get_legend_handles_labels()
                handles += h1
                labels += l1
    else:
        # Plot only pretrain loss if no mask data
        ax1.plot(epoch_loss, loss_values, color=nature_colors[0], linewidth=2.2, label='Pretrain Loss')
        ax1.set_xlabel('Epoch', fontsize=20, labelpad=10)
        ax1.set_ylabel('RMSE', fontsize=20, labelpad=15, rotation=90, va='center', color=nature_colors[0])
        ax1.set_yscale('log')
        h1, l1 = ax1.get_legend_handles_labels()
        handles += h1
        labels += l1

    ax1.grid(True, alpha=0.6, linestyle=':')

    # 保持所有边框可见
    for spine in ['top', 'right', 'bottom', 'left']:
        ax1.spines[spine].set_visible(True)

    plt.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.12)
    plt.show()


if __name__ == '__main__':
    apply_global_styles()
    plot_pretrain_loss_and_dim()
