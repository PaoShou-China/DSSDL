[English](#english) | [中文](#chinese)

<a id="english"></a>

# DSSDL
### Disentangled Shared-Specific Dynamics Learning for Cross-System Identification

![DSSDL scientific identity](docs/assets/hero.png)

**Shared structure. Specific dynamics.**

A research codebase for learning shared dynamics across related systems, adapting
low-dimensional task coordinates from limited observations, and studying symbolic
discovery with feedback training.

[Getting started / 使用指南](GETTING_STARTED.md) ·
[Project website](https://paoshou-china.github.io/DSSDL/) ·
[Experiment map](revision_2026_11/EXPERIMENTS.md) ·
[Contributing](CONTRIBUTING.md)

## Start here

Python 3.10 or 3.11 is recommended for the recorded scientific environment.

```bash
git clone https://github.com/PaoShou-China/DSSDL.git
cd DSSDL
python -m pip install -e ".[experiments]"
dssdl doctor
dssdl verify
```

Use a virtual environment. For a specific CPU/CUDA build, install PyTorch first
using its [official instructions](https://pytorch.org/get-started/locally/).
The lightweight `pip install -e .` installs the CLI alone; it does not download
PyTorch. `python -m dssdl` is equivalent to the `dssdl` command.

## One interface, recorded experiments

```bash
dssdl list
dssdl run wasserstein --name my-distance-check
dssdl run online --system fhn --method pysr44 --seed 0 --smoke --name my-online-check
dssdl run online --help
```

`verify` recalculates saved metrics and checks file integrity. `check` loads
models and checks symbolic gradients. Neither retrains a model.
`--smoke` runs two online windows as an installation check; remove it for the
full recorded schedule. Use `--dry-run` to inspect a command before execution.

| Evidence | Included records |
|---|---|
| Dimension selection and symbolic regularization | 20 paired comparisons, 40 variants |
| Matched adaptation budgets | 320 records, 64 aggregate rows |
| Fixed-parameter trajectory distributions | 12 exports across FHN and Lorenz |
| Selected time-varying prediction protocols | 810 forecast windows |
| Symbolic discovery and feedback | FHN, Lorenz and measured pendulum |

Inputs, checkpoints and saved results for these paths are included. The complete
raw CFD/POD-generation workflow and every historical trial are not bundled.

## Algorithm

![DSSDL workflow overview](docs/assets/overview.png)

The symbolic workflow separates search, consolidation and feedback. The recorded
consolidation uses LLM assistance with author confirmation. Exact coefficient
reparameterizations and approximate reductions are distinguished in the
[workflow record](revision_2026_11/SYMBOLIC_WORKFLOW.md).

Original prescribed-structure predictions, no-symbol models and feedback-trained
models have distinct checkpoints. Their performance records are not interchangeable.
See the [experiment map](revision_2026_11/EXPERIMENTS.md) for the evaluation scope.

## Repository

```text
src/dssdl/           Unified CLI, task registry and environment checks
tests/              Fast command-line tests and website checks
scripts/            Scientific-integrity checks and website data export
docs/               Static project website and local visual assets
revision_2026_11/    Recorded sources, data, models, results and provenance
ci/templates/       CI definitions awaiting workflow authorization
```

The original root-level pendulum implementation remains available. Its original
instructions are preserved in [the legacy guide](docs/LEGACY_PENDULUM.md).
Older project downloads remain at the original
[project data folder](https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff format --check .
python -m ruff check .
```

The style cleanup preserves executable syntax for all 80 pre-existing Python
files. Numerical records are checked separately. See
[CONTRIBUTING.md](CONTRIBUTING.md) before modifying an experiment.

Local code checks pass and the project website is published on GitHub Pages.
GitHub CI activation still requires workflow authorization; see [CI setup](ci/README.md).

The project page uses Beihang-inspired colors and an original project emblem,
not the university seal. Concept imagery is distinct from plotted experimental
data. Asset provenance is documented [here](docs/assets/ASSETS.md).

The website has a Chinese/English switch. Repository documentation provides both
languages in the same file; use the language links at the top.

---

<a id="chinese"></a>

# DSSDL
### 面向跨系统辨识的共享与特性解耦动力学学习

**共享结构，特定动力学。**

本研究代码库从相关系统中学习共享动力学，用少量观测适配低维任务坐标，并研究符号发现与回填训练。

[使用指南](GETTING_STARTED.md) · [项目网页](https://paoshou-china.github.io/DSSDL/) · [实验索引](revision_2026_11/EXPERIMENTS.md) · [开发指南](CONTRIBUTING.md)

## 快速开始

复现已记录的科学计算环境，建议使用 Python 3.10 或 3.11。在独立虚拟环境中执行：

```bash
git clone https://github.com/PaoShou-China/DSSDL.git
cd DSSDL
python -m pip install -e ".[experiments]"
dssdl doctor
dssdl verify
```

需要指定 CPU/CUDA 版本时，先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装。轻量命令 `pip install -e .` 仅安装命令行工具，不下载 PyTorch。`python -m dssdl` 与 `dssdl` 等价。

## 统一入口与已有实验

```bash
dssdl list
dssdl run wasserstein --name my-distance-check
dssdl run online --system fhn --method pysr44 --seed 0 --smoke --name my-online-check
dssdl run online --help
```

`verify` 重算保存指标并检查文件完整性；`check` 加载模型、检查符号梯度。两者都不重新训练。`--smoke` 只运行两个在线窗口，供安装检查；移除该选项可运行完整的已记录时序。`--dry-run` 只显示将执行的命令。

| 证据 | 包含记录 |
|---|---|
| 维度选择与符号正则 | 20 组成对对照、40 个变体 |
| 对齐适配预算 | 320 条记录、64 行汇总 |
| 固定参数轨迹分布 | FHN 与 Lorenz 共 12 份导出 |
| 所选时变预测协议 | 810 个预测窗口 |
| 符号发现与回填 | FHN、Lorenz 与实测单摆 |

这些复现路径所需的输入、模型和结果已包含。完整原始 CFD/POD 生成流程及全部历史尝试不在本包范围内。

## 算法

上方总览图中的 Learn、Discover、Consolidate、Adapt 分别对应学习、发现、整理和适配。

符号流程区分搜索、整理和回填。已有整理记录采用 LLM 辅助并经作者确认；精确系数重参数化与近似删项在[流程记录](revision_2026_11/SYMBOLIC_WORKFLOW.md)中分开说明。

原给定结构模型、无符号模型与回填模型使用不同检查点，性能记录不能相互替代。各自评价范围见[实验索引](revision_2026_11/EXPERIMENTS.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/dssdl/` | 统一命令行、任务注册和环境检查 |
| `tests/` | 命令行测试与网页检查 |
| `scripts/` | 科学结果完整性检查、网页数据导出 |
| `docs/` | 静态双语网页与本地视觉资源 |
| `revision_2026_11/` | 实验源码、数据、模型、结果与来源 |
| `ci/templates/` | 等待工作流授权的自动检查配置 |

原根目录单摆实现仍保留，其说明见[历史指南](docs/LEGACY_PENDULUM.md)。早期下载链接仍为原[项目数据目录](https://drive.google.com/drive/folders/1UlD6YIVsCEYH2_ZC5PC16jn9Kqm4-Q5L?usp=sharing)。

## 开发与核验

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff format --check .
python -m ruff check .
```

格式整理保留了全部 80 个原有 Python 文件的可执行语法结构；数值记录另行核验。修改实验前请阅读[开发指南](CONTRIBUTING.md)。

本地代码检查已通过，网页已发布至 GitHub Pages。GitHub 自动检查仍待工作流授权，见[配置说明](ci/README.md)。

网页提供中文与英文切换；本文及其他说明在同一文件内提供两个语言入口。页面使用北航风格配色和原创项目标识，并非学校校徽或官方网站。概念图与实验数据分开标注，来源见[视觉资源说明](docs/assets/ASSETS.md)。
