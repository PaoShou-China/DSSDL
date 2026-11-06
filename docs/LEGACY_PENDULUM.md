[English](#english) | [中文](#chinese)

<a id="english"></a>

# Disentangled Shared-Specific Dynamics Learning for Cross-System Identification

This is a retained historical description, including its defaults, broad result
statements and reference list. For current verified settings and evaluation
scope, use the [getting-started guide](../GETTING_STARTED.md) and
[experiment map](../revision_2026_11/EXPERIMENTS.md).

## Current reproducibility materials

The [November 2026 experiment package](../revision_2026_11/README.md) contains the
source code, configurations, required data/checkpoints and saved numerical
results for the added component controls, symbolic discovery and feedback,
matched adaptation budgets, Wasserstein distances, and selected time-varying
experiments. See its [experiment-to-evidence map](../revision_2026_11/EXPERIMENTS.md)
for commands and result provenance.

The original files below are retained. The new package documents the original
prescribed-structure predictions, recorded PySR searches with author-confirmed
LLM-assisted consolidation, and feedback-trained deployment separately, with
the checkpoints and evaluation protocols identified for each experiment.

Quick numerical verification (no training or symbolic search):

```bash
cd revision_2026_11
python -m pip install -r code/requirements.txt
python -B verify.py --checksums
```

The remaining sections describe the original pendulum implementation.

## Additional Files

Other project files are available at: [https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing](https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing). Please download them yourself.

## Project Introduction

This project implements a deep learning-based sparse system identification method, specifically for pendulum system dynamics modeling. Through meta-learning and sparse parameter learning techniques, this method can:

- Learn shared dynamical representations from multiple similar systems
- Automatically discover key system parameters using sparse masks
- Quickly adapt to new unknown systems with only a small amount of data
- Improve model interpretability and generalization capabilities

## Technology Stack

- **Programming Language**: Python 3.9+
- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, CSV
- **Visualization**: Matplotlib
- **Optimizers**: Adam, LBFGS, custom CautiousAdam

## Project Structure

```
├── datasets/           # Dataset directory
│   ├── pendulum_real/  # Real pendulum system data
│   ├── __init__.py     # Dataset initialization
│   └── utils.py        # Data processing utilities
├── draw/               # Plotting scripts
├── final_data/         # Final result data
│   ├── hidden_params/  # Learned hidden parameters
│   ├── mask/           # Sparse mask data
│   ├── meta_train/     # Meta-training loss data
│   └── pretrain/       # Pretraining loss data
├── model/              # Model saving directory
├── pictures/           # Result pictures
├── results/            # Evaluation results
├── DSSDL_mask.py       # Core model implementation
├── evaluate_mask.py    # Mask evaluation
├── fast_adapt.py       # Fast adaptation script
├── meta_train.py       # Meta-training script
└── README.md           # Project description
```

## Core Algorithms

### 1. Meta-learning

The meta-learning phase trains the model on multiple pendulum systems to learn shared dynamical representations. Key steps include:

- **Multi-system training**: Training with data from multiple pendulum systems with different parameters
- **Parameter sharing**: Learning shared features and parameters across systems
- **Sparse regularization**: Encouraging parameter sparsity through masking mechanisms

### 2. Sparse Parameter Learning

Using learnable masks to automatically discover key system parameters:

- **Continuous mask**: Sigmoid-activated continuous value mask
- **Hard mask**: Threshold-based binary mask for true sparsity
- **Mask regularization**: Encouraging mask values to approach 0 or 1 for enhanced sparsity

### 3. Fast Adaptation

Using meta-learned initialization to quickly adapt to new systems with only a small amount of support data:

- **Parameter fine-tuning**: Only fine-tuning parameters specific to the new system
- **Early stopping**: Stopping training when loss change falls below a threshold
- **Fast convergence**: Leveraging good initialization from meta-learning to accelerate convergence

## Model Architecture

### MetaNet Model

MetaNet is the core model of this project, consisting of the following components:

1. **One-hot to specific parameter mapping**: Maps system one-hot encoding to specific parameters
2. **Continuous mask layer**: Learns sparsity mask for parameters
3. **Neural network backbone**: Composed of multiple fully connected layers and SiLU activation functions
4. **Specific parameter generator**: Generates specific parameters for new systems during fast adaptation

```python
# MetaNet model architecture
MetaNet(
  (one_hot_to_specific_param): Linear(in_features=N, out_features=D, bias=False)
  (specific_param_generator): Linear(in_features=1, out_features=D, bias=False)
  (continuous_mask): Parameter(torch.Size([D]))
  (backbone): Sequential(
    (0): Linear(in_features=I+D, out_features=64)
    (1): SiLU()
    (2): Linear(in_features=64, out_features=64)
    (3): SiLU()
    (4): Linear(in_features=64, out_features=O)
  )
)
```

Where:
- N: Number of training systems
- D: Dimension of specific parameters
- I: Dimension of input states
- O: Dimension of output derivatives

## Training Process

### 1. Meta-training

```bash
python meta_train.py
```

Meta-training process:
1. Load training data from multiple pendulum systems
2. Initialize the MetaNet model
3. Perform meta-training to learn shared representations and sparse masks
4. Save the trained model and loss curves

### 2. Fast Adaptation

```bash
python fast_adapt.py
```

Fast adaptation process:
1. Load the meta-trained model
2. Quickly fine-tune the model using support set data
3. Evaluate model performance on the query set
4. Save the adapted model and results

## Configuration Parameters

### Meta-training Configuration

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `specific_param_dim` | Dimension of specific (slow) parameters | 6 |
| `epochs` | Number of meta-training epochs | 100 |
| `reg_rate` | Regularization rate | 0.0 |
| `use_raw_data` | Whether to use raw (unnormalized) data | False |
| `use_trajectory_sampling` | Whether to use trajectory sampling | False |
| `num_trajectories` | Number of trajectories to sample per system | 1 |
| `trajectory_duration` | Duration of each trajectory (seconds) | 10 |
| `batch_size` | Batch size | 128 |
| `meta_train_lr` | Meta-training learning rate | 0.01 |
| `hidden_widths` | Hidden layer widths | [64, 64] |
| `fast_adapt_lr` | Fast adaptation learning rate | 0.001 |
| `optimizer` | Optimizer type | 'adam' |
| `device` | Device to use | 'cuda:0' |

### Fast Adaptation Configuration

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `epochs` | Number of fast adaptation epochs | 200 |
| `convergence_threshold` | Convergence threshold | 1e-5 |
| `patience` | Early stopping patience | 100 |
| `batch_size` | Batch size | -1 (use full batch) |
| `optimizer` | Optimizer type | 'lbfgs' |

## Dataset Format

The dataset format used in this project is as follows:

```
datasets/
└── pendulum_real/
    ├── train/         # Training set (multiple systems)
    │   ├── features/  # State features
    │   └── labels/    # Derivative labels
    ├── support/       # Support set (new system)
    │   ├── features/  # State features
    │   └── labels/    # Derivative labels
    └── query/         # Query set (new system)
        ├── features/  # State features
        └── labels/    # Derivative labels
```

Each system's data is stored as text files, where:
- Feature files contain the system's state variables (e.g., angle, angular velocity)
- Label files contain the derivatives of the state variables

## Result Analysis

### 1. Loss Curves

Loss curves during training are saved in the `pictures/` directory, including:
- Meta-training loss curve (`meta_train_loss_curve.png`)
- Pretraining loss curve (`pretrain_loss_curve.png`)
- Fast adaptation loss curve

### 2. Sparse Mask Analysis

Learned sparse masks are saved in the `final_data/mask/` directory, and system key parameters can be analyzed through mask values.

### 3. Trajectory Prediction

Model trajectory prediction results on new systems are saved in the `pictures/` directory, such as `trajectory_pendulum_real.png`.

### 4. Hidden Parameter Analysis

Learned system hidden parameters are saved in the `final_data/hidden_params/` directory, which can be used to analyze the physical characteristics of the system.

## Technical Highlights

1. **Sparse Parameter Learning**: Automatically discovering system key parameters through learnable masks, improving model interpretability.

2. **Meta-learning**: Learning shared representations from multiple systems, improving model generalization ability.

3. **Fast Adaptation**: Quickly adapting to new systems with only a small amount of data, reducing data requirements.

4. **Custom Optimizer**: Implementing the CautiousAdam optimizer that adjusts learning rate based on gradient direction, improving training stability.

5. **Trajectory Sampling**: Supporting trajectory-based sampling methods to better capture system dynamic characteristics.

## Experimental Results

Experimental results on pendulum systems show:

- Meta-learning significantly improves model generalization ability
- Sparse masks successfully discover system key parameters
- Fast adaptation achieves good performance with only a small amount of data
- The model has higher prediction accuracy on unknown systems compared to traditional methods

## Code Usage Guide

### 1. Prepare Dataset

Organize pendulum system data into the `datasets/` directory according to the format described above.

### 2. Meta-training

```bash
# Execute meta-training
python meta_train.py

# Meta-training results will be saved to:
# - model/meta_train.pt (model weights)
# - final_data/meta_train/meta_train_loss.npy (loss data)
# - pictures/pendulum_real/meta_train_loss_curve.png (loss curve)
```

### 3. Fast Adaptation

```bash
# Execute fast adaptation
python fast_adapt.py

# Fast adaptation results will be saved to:
# - model/fast_adapt.pt (adapted model)
# - model/fast_adapt_loss.npy (adaptation loss)
# - results/pendulum_real/ (evaluation results)
```

### 4. Custom Configuration

You can adjust model parameters and training settings by modifying the configuration dictionaries in `meta_train.py` and `fast_adapt.py`.

## Extension and Application

This method is not only applicable to pendulum systems but can also be extended to other types of dynamical systems, such as:

- Spring-mass systems
- Robot dynamics models
- Chemical reaction dynamics
- Financial time series modeling

By adjusting the model's input dimension and hidden layer widths, it can adapt to systems of different complexities.

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy
- Matplotlib
- tqdm

## References

1. Meta-Learning for Few-Shot Learning. Chelsea Finn, Pieter Abbeel, Sergey Levine. 2017.
2. Sparse Identification of Nonlinear Dynamical Systems. Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz. 2016.
3. Deep Sparse Regularization for Neural Networks. Zhang et al. 2018.

## Code Description

### Core Modules

1. **DSSDL_mask.py**: Implements the core MetaNet model and trainer class, including sparse mask learning and meta-learning functionality.

2. **meta_train.py**: Implements the meta-training process, training the model on multiple systems.

3. **fast_adapt.py**: Implements the fast adaptation process, quickly adapting to new systems with small amounts of data.

4. **evaluate_mask.py**: Used for evaluating and analyzing learned sparse masks.

### Data Processing

Dataset loading and processing functionality is located in the `datasets/` directory, supporting:
- Data normalization
- Trajectory sampling
- One-hot encoding
- Batch loading

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce batch size `batch_size`
   - Use smaller hidden layer widths `hidden_widths`
   - Switch to CPU device `device='cpu'`

2. **Training Instability**:
   - Adjust learning rates `meta_train_lr` and `fast_adapt_lr`
   - Use CautiousAdam optimizer `use_cautious_adam=True`
   - Increase regularization rate `reg_rate`

3. **Poor Fast Adaptation Performance**:
   - Increase support set data amount
   - Adjust `convergence_threshold` and `patience` parameters
   - Extend fast adaptation epochs `epochs`

## Contact

If you have any questions or suggestions, please feel free to contact the project maintainers.

---

**Project Version**: 1.0.0
**Last Updated**: 2026-02-06
**License**: MIT

---

<a id="chinese"></a>

# 单摆历史实现说明

本节译自上方保留的历史说明。其默认参数、效果概述及参考文献属于该历史版本，不作为当前实验包的新结论。当前可核验配置、边界和入口以[使用指南](../GETTING_STARTED.md)及[实验索引](../revision_2026_11/EXPERIMENTS.md)为准。

## 当前复现材料

[2026 年 11 月实验包](../revision_2026_11/README.md)包含新增部件对照、符号发现/回填、对齐适配预算、Wasserstein 距离及所选时变实验的源码、配置、所需数据/模型和结果。[对应索引](../revision_2026_11/EXPERIMENTS.md)提供命令及来源。

原文件仍保留。新包分别说明原给定结构预测、已有 PySR 搜索及作者确认的 LLM 辅助整理、回填部署，并标明各实验模型与协议。从仓库根目录可快速核验（不训练、不搜索）：

```bash
cd revision_2026_11
python -m pip install -r code/requirements.txt
python -B verify.py --checksums
```

以下描述原单摆实现。

## 附加文件与项目介绍

其他项目文件见原 [Google Drive 数据目录](https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing)。

项目使用深度学习进行稀疏系统辨识，原实现面向单摆动力学。通过元学习和稀疏参数学习，从相似系统学习共享表示，利用稀疏掩码选择关键坐标，以少量数据适配未知系统，并研究可解释性与泛化。

## 技术栈

历史说明列示 Python3.9+、PyTorch、NumPy/CSV、Matplotlib，以及 Adam、LBFGS、自定义 CautiousAdam。当前锁定环境另见新包说明。

## 目录

| 路径 | 内容 |
|---|---|
| `datasets/pendulum_real/` | 实测单摆数据 |
| `datasets/__init__.py`、`utils.py` | 初始化与数据处理 |
| `draw/` | 绘图脚本 |
| `final_data/hidden_params/` | 学到的隐参数 |
| `final_data/mask/` | 稀疏掩码 |
| `final_data/meta_train/`、`pretrain/` | 训练损失 |
| `model/`、`pictures/`、`results/` | 模型、图片、评价结果 |
| `DSSDL_mask.py` | 核心模型 |
| `evaluate_mask.py` | 掩码评价 |
| `fast_adapt.py`、`meta_train.py` | 快速适配、元训练 |
| `README.md` | 项目说明 |

## 核心算法

1. **元学习：** 多参数单摆数据联合训练，学习共享特征与参数，并以掩码正则鼓励稀疏。
2. **稀疏参数学习：** Sigmoid 连续掩码、按阈值二值化的硬掩码，以及鼓励接近0/1的正则。
3. **快速适配：** 从元学习初始化出发，仅调整新任务特性参数；历史实现按损失变化阈值早停，并利用初始化加快收敛。当前固定预算补充实验另有明确协议。

## 模型结构

MetaNet 包括 one-hot 到特性参数的线性映射、连续掩码、带 SiLU 的全连接骨干，以及新任务特性参数生成器。结构与上方代码相同：

```text
one_hot_to_specific_param: Linear(N, D, bias=False)
specific_param_generator: Linear(1, D, bias=False)
continuous_mask: Parameter([D])
backbone: Linear(I+D,64) → SiLU → Linear(64,64) → SiLU → Linear(64,O)
```

N 为训练系统数，D 为特性维数，I 为状态输入维数，O 为导数输出维数。

## 训练过程

在原根目录运行 `python meta_train.py`：加载多系统数据、初始化 MetaNet、训练共享表示与稀疏掩码、保存模型和损失。运行 `python fast_adapt.py`：加载元训练模型、用 support 微调、在 query 评价、保存适配模型和结果。

## 历史默认配置

| 元训练参数 | 含义 | 默认值 |
|---|---|---|
| `specific_param_dim` | 特性（慢）参数维数 | 6 |
| `epochs` | 元训练轮数 | 100 |
| `reg_rate` | 正则权重 | 0.0 |
| `use_raw_data` | 使用未标准化数据 | False |
| `use_trajectory_sampling` | 使用轨迹采样 | False |
| `num_trajectories` | 每系统采样轨迹数 | 1 |
| `trajectory_duration` | 每轨迹秒数 | 10 |
| `batch_size` | 批大小 | 128 |
| `meta_train_lr` | 元训练学习率 | 0.01 |
| `hidden_widths` | 隐层宽度 | [64,64] |
| `fast_adapt_lr` | 快速适配学习率 | 0.001 |
| `optimizer` | 优化器 | adam |
| `device` | 设备 | cuda:0 |

| 适配参数 | 含义 | 默认值 |
|---|---|---|
| `epochs` | 适配轮数 | 200 |
| `convergence_threshold` | 收敛阈值 | 1e-5 |
| `patience` | 早停耐心值 | 100 |
| `batch_size` | 批大小 | -1，全批 |
| `optimizer` | 优化器 | lbfgs |

## 数据格式

`datasets/pendulum_real/` 下分别为 `train/`（多训练系统）、`support/`（新系统支持集）、`query/`（新系统查询集），每个子集含 `features/` 与 `labels/`。文本特征文件存状态（如角度、角速度）；标签文件存状态导数。

## 结果分析

- 损失曲线位于 `pictures/`，包括 `meta_train_loss_curve.png`、`pretrain_loss_curve.png` 和适配损失。
- 掩码保存在 `final_data/mask/`，可分析被保留的特性坐标。
- 轨迹图片位于 `pictures/`，如 `trajectory_pendulum_real.png`。
- 隐参数位于 `final_data/hidden_params/`，用于分析系统特性。

## 历史技术特点与结果概述

原说明强调可学习掩码的稀疏表示、跨系统元学习、少样本适配、按梯度方向调整更新的 CautiousAdam，以及轨迹采样。其原结果概述称元学习改善泛化、掩码识别关键参数、少数据适配有效，并比传统方法有更高未知系统预测精度。此处保留历史概述的译义；当前各比较的具体结果和允许结论应查实验索引，不能从该概述推出所有任务均占优。

## 使用

先按上述格式组织数据，再从原根目录运行：

```bash
python meta_train.py
python fast_adapt.py
```

元训练输出：`model/meta_train.pt`、`final_data/meta_train/meta_train_loss.npy`、`pictures/pendulum_real/meta_train_loss_curve.png`。适配输出：`model/fast_adapt.pt`、`model/fast_adapt_loss.npy`、`results/pendulum_real/`。自定义配置位于两脚本的配置字典。

## 扩展与依赖

历史说明列举弹簧质量系统、机器人动力学、化学反应、金融时间序列等潜在扩展，并建议调整输入维数和隐层宽度；这不表示该包已包含这些扩展实验。历史依赖列为 Python3.9+、PyTorch2.0+、NumPy、Matplotlib、tqdm。

## 原说明参考条目

以下保留原英文列示，不把翻译视作重新核验的文献元数据：

1. Meta-Learning for Few-Shot Learning. Chelsea Finn, Pieter Abbeel, Sergey Levine. 2017.
2. Sparse Identification of Nonlinear Dynamical Systems. Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz. 2016.
3. Deep Sparse Regularization for Neural Networks. Zhang et al. 2018.

## 模块与数据处理

`DSSDL_mask.py` 实现 MetaNet、训练器及掩码学习；`meta_train.py` 执行多系统训练；`fast_adapt.py` 执行少样本适配；`evaluate_mask.py` 分析掩码。`datasets/` 支持标准化、轨迹采样、one-hot 编码和批次加载。

## 排错与联系

- CUDA 内存不足：减小 `batch_size`、`hidden_widths` 或改为 `device='cpu'`。
- 训练不稳定：检查 `meta_train_lr`、`fast_adapt_lr`，原说明还建议 CautiousAdam（`use_cautious_adam=True`）及调整 `reg_rate`。
- 适配效果不佳：增加 support、检查 `convergence_threshold`、`patience` 或延长 `epochs`。正式比较的调参必须遵循相应实验协议。

问题和建议请联系项目维护者。历史说明版本：1.0.0；更新日期：2026-02-06；列示许可：MIT。
