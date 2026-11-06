[English](#english) | [中文](#chinese)

<a id="english"></a>

# Getting started / 使用指南

The checked entry points use the included data and checkpoints. No download from
a personal drive is needed for the selected experiments in `revision_2026_11`.
Run commands from a cloned checkout; install the CLI in editable mode.

## 1. Install

Use Python 3.10 or 3.11 for the pinned scientific environment. The lightweight
CLI and its unit tests also support newer Python versions.

```bash
git clone https://github.com/PaoShou-China/DSSDL.git
cd DSSDL
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -e ".[experiments]"
dssdl doctor
```

For a particular CPU/CUDA build, install PyTorch using its
[official selector](https://pytorch.org/get-started/locally/) first. Keep the
recorded PyTorch version for comparisons with saved results. The base command
`python -m pip install -e .` installs only the CLI, without scientific packages.

## 2. Verify before training

```bash
dssdl verify
dssdl check
dssdl list
```

- `verify`: checks SHA-256 hashes and recalculates the saved numerical evidence.
- `check`: loads the archived datasets and models and tests symbolic gradients.
- `list`: shows the ten supported experiment entry points.

These commands do not start training or a PySR search. `python -m dssdl` is an
alternative when your shell does not find the `dssdl` executable.

## 3. Reproduce one result

```bash
dssdl run wasserstein --name w-example
dssdl run online --system fhn --method pysr44 --seed 0 --smoke --name online-example
dssdl run budget --system lorenz --method CAVIA --seed 1 --support 100 --name budget-example
```

Remove `--smoke` for the full saved online schedule. The budget run defaults to
10,000 target updates. Add `--dry-run` to print the resolved command without
running it. `dssdl run online --help` exposes the original driver's options.
New runs refuse to overwrite an existing output name.

## 4. Run controls or symbolic discovery

```bash
dssdl run gate --system pendulum --seed 0 --device cpu
dssdl run regularizer-pendulum --seed 5800 --device cpu
dssdl run regularizer-lorenz --seed 42044 --device cpu
python -m pip install -e ".[symbolic]"
dssdl run discover --system lorenz --device cpu --name new-search
dssdl run feedback --system lorenz --device cpu --name recorded-feedback
```

PySR needs Julia; its first setup can download a runtime and packages. Feedback
uses the recorded, author-confirmed template, not the output of an arbitrary new
search. See the [symbolic workflow](revision_2026_11/SYMBOLIC_WORKFLOW.md).
The `all-controls` convenience launcher retains the original GPU configuration;
use the individual commands above when running CPU-only.

## Where files go

| Content | Location |
|---|---|
| Original results, never overwritten by reruns | `revision_2026_11/results/` |
| Online, budget and Wasserstein reruns | `revision_2026_11/outputs/` |
| Discovery and feedback reruns | `revision_2026_11/code/outputs/` |
| Paired control reruns | `revision_2026_11/code/review/controls/experiments/` |
| Paper-to-experiment map | `revision_2026_11/EXPERIMENTS.md` |

## Troubleshooting

- **Where to run:** the installed CLI locates the source checkout. Use `dssdl --repo-root "repository-path" verify` or `DSSDL_REPO_ROOT` to specify it explicitly.
- **Command not found:** use `python -m dssdl` from the installation environment; do not mix Python environments.
- **Missing dependencies:** run `dssdl doctor` and follow its suggestions without replacing an existing research environment.
- **Output exists:** choose another `--name` to retain the previous result.
- **No GPU:** verification, Wasserstein and the selected online runs can use CPU. Pass `--device cpu` explicitly for training.
- **Small numerical differences:** operating systems, hardware and dependency versions can affect floating-point computations and random searches. Verify saved results first; do not overwrite them with new runs.
- **SINDy dependencies:** online replay uses saved source structure and does not need PySINDy. Check the package README for historical NumPy constraints before installing optional dependencies.

## Website

`docs/index.html` is a standalone static page; no frontend build is required.
The interactive plots come from saved CSVs and are exported by
`python scripts/build_site_data.py`. The current Pages deployment uses `main:/` with a root redirect to `docs/`.
The Chinese/English switch translates the whole page and remembers the choice;
`?lang=en` or `?lang=zh` selects the language when opening a link.

---

<a id="chinese"></a>

# 使用指南

所选入口使用包内数据与模型，无需从个人网盘下载 `revision_2026_11` 所选实验的依赖文件。从克隆的仓库目录安装，建议使用可编辑安装模式。

## 1. 安装

锁定的科学计算环境使用 Python 3.10 或 3.11；轻量命令行及单元测试也支持更新版本。

```bash
git clone https://github.com/PaoShou-China/DSSDL.git
cd DSSDL
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活；Linux/macOS 使用 `source .venv/bin/activate`。随后执行：

```bash
python -m pip install -e ".[experiments]"
dssdl doctor
```

如需指定 CPU/CUDA 构建，请先用 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)安装。与保存结果比较时，保留记录中的 PyTorch 版本。`python -m pip install -e .` 仅安装命令行，不安装科学计算依赖。

## 2. 训练前先核验

```bash
dssdl verify
dssdl check
dssdl list
```

- `verify`：检查 SHA-256 并重算保存的数值证据。
- `check`：加载归档数据与模型，检查符号梯度。
- `list`：列出十个实验入口。

这些命令不会启动训练或 PySR 搜索。若终端找不到 `dssdl`，使用同一环境中的 `python -m dssdl`。

## 3. 复现一个结果

```bash
dssdl run wasserstein --name w-example
dssdl run online --system fhn --method pysr44 --seed 0 --smoke --name online-example
dssdl run budget --system lorenz --method CAVIA --seed 1 --support 100 --name budget-example
```

移除 `--smoke` 可运行完整保存时序。预算对比默认执行 10,000 次目标更新。`--dry-run` 只打印实际命令；`dssdl run online --help` 查看原驱动选项。新运行拒绝覆盖已有输出名称。

## 4. 对照与符号发现

```bash
dssdl run gate --system pendulum --seed 0 --device cpu
dssdl run regularizer-pendulum --seed 5800 --device cpu
dssdl run regularizer-lorenz --seed 42044 --device cpu
python -m pip install -e ".[symbolic]"
dssdl run discover --system lorenz --device cpu --name new-search
dssdl run feedback --system lorenz --device cpu --name recorded-feedback
```

PySR 需要 Julia，首次设置可能下载运行时和依赖。回填使用已记录、经作者确认的模板，并不会自动采用任意一次新搜索结果，详见[符号流程](revision_2026_11/SYMBOLIC_WORKFLOW.md)。`all-controls` 保留原 GPU 配置；仅用 CPU 时请运行上面的独立命令。

## 输出位置

| 内容 | 路径 |
|---|---|
| 原结果，重跑不覆盖 | `revision_2026_11/results/` |
| 在线、预算、Wasserstein 重跑 | `revision_2026_11/outputs/` |
| 发现与回填重跑 | `revision_2026_11/code/outputs/` |
| 成对对照重跑 | `revision_2026_11/code/review/controls/experiments/` |
| 论文与实验对应关系 | `revision_2026_11/EXPERIMENTS.md` |

## 常见问题

- **从哪里运行：** 安装后 CLI 会定位本地源码。也可使用 `dssdl --repo-root "仓库路径" verify` 或设置 `DSSDL_REPO_ROOT`。
- **命令找不到：** 使用安装环境中的 `python -m dssdl`，不要混用不同 Python。
- **依赖缺失：** 先运行 `dssdl doctor` 并按提示安装；不要覆盖已有研究环境。
- **输出已存在：** 换一个 `--name`，保留原结果。
- **没有 GPU：** 核验、W 距离和所选在线运行可用 CPU；训练命令显式指定 `--device cpu`。
- **数值略有不同：** 操作系统、硬件和依赖版本会影响浮点计算与随机搜索。先核验已有结果，不用新结果覆盖原记录。
- **SINDy 依赖：** 在线回放加载保存的源任务结构，无需 PySINDy。其历史 NumPy 版本约束见实验包说明，不要直接混装。

## 网页

`docs/index.html` 是独立静态页，不需要前端构建。中文与英文按钮切换整页内容并记住选择；`?lang=en` 或 `?lang=zh` 可指定首次打开的语言。交互图由保存的 CSV 经 `python scripts/build_site_data.py` 导出。当前 Pages 使用 `main:/`，根页面跳转到 `docs/`。
