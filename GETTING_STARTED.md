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

## 常见问题

- **不知道从哪里运行：** 安装后可在其他目录运行，CLI 会定位本地源码。
  也可使用 `dssdl --repo-root "仓库路径" verify` 或设置 `DSSDL_REPO_ROOT`。
- **命令找不到：** 使用安装环境中的 `python -m dssdl`，不要混用不同 Python。
- **依赖缺失：** 先运行 `dssdl doctor`，按照提示安装；不要覆盖原有研究环境。
- **输出目录已存在：** 换一个 `--name`，保留原结果，避免误覆盖。
- **没有 GPU：** 核验、W 距离和所选在线运行可用 CPU；训练命令显式指定 `--device cpu`。
- **数值略有不同：** 操作系统、硬件和依赖版本可影响浮点数和随机搜索。
  先核对已保存结果，再讨论新的训练结果，不以新结果覆盖原记录。
- **SINDy 依赖：** 在线回放使用保存的源任务结构，不需要安装 PySINDy。
  历史 PySINDy 与 NumPy 的版本约束见实验目录 README，不要直接混装。

## Website

`docs/index.html` is a standalone static page; no frontend build is required.
The interactive plots come from saved CSVs and are exported by
`python scripts/build_site_data.py`. GitHub Pages can publish `main:/docs`.
