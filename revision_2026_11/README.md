[English](#english) | [中文](#chinese)

<a id="english"></a>

# DSSDL: November 2026 reproducibility package

This directory contains the selected experiment sources and numerical records
used in the current manuscript. Original implementations elsewhere in the
repository remain unchanged. No peer-review correspondence is included.

## Contents

| Directory | Contents |
|---|---|
| `code/` | Original model modules, portable experiment drivers, source data and required checkpoints |
| `results/` | Saved component-control predictions, 320 matched-budget records, original-trajectory distances and selected online forecasts |
| `records/` | Protocols, source provenance and the corresponding numerical table sources |
| `outputs/` | Created locally by reruns; not part of the reference results |

See [EXPERIMENTS.md](EXPERIMENTS.md) for the mapping from each experiment to the
manuscript and SI, including the limits of each comparison.

## Environment and quick checks

The tested environment uses Python 3.10, PyTorch 2.8.0, NumPy 1.26.4 and SciPy
1.15.3. Full versions are in `code/environment_tested.json`. Choose the appropriate
PyTorch CPU/CUDA build for your machine. Run commands from this directory:

```bash
python -m pip install -r code/requirements.txt
python -B verify.py --checksums
python -B code/run.py check
```

`verify.py` recomputes 40 target errors from the paired-control predictions,
checks all 64 fixed-budget aggregate rows against 320 saved runs, recomputes the
two-system Wasserstein results, and checks the selected online forecast errors.
`check` loads the included data and feedback models and checks the symbolic
component's numerical evaluation and gradients. Neither command trains models.

## Component controls

```bash
python -B code/run.py gate --system pendulum --seed 0
python -B code/run.py gate --system quadrotor --seed 0
python -B code/run.py regularizer-pendulum --seed 5800
python -B code/run.py regularizer-lorenz --seed 42044
```

Gate seeds are 0-4; pendulum feedback seeds are 5800-5804; Lorenz feedback seeds
are 42044-42048. Add `--device cpu` for CPU execution. `all-controls` runs the
complete set. The regularizer comparison retains the same symbolic template in
both arms; it is not the no-symbol ablation. Five feedback repeats share one
teacher and template and are not five independent PySR searches.

## Discovery and feedback

Install `code/requirements-symbolic.txt` before launching a new PySR search.
PySR also requires its Julia runtime and packages; initial setup may download
them. Saved-result checks do not need Julia.

```bash
python -B code/run.py discover --system fhn
python -B code/run.py feedback --system fhn
python -B code/run.py discover --system lorenz
python -B code/run.py feedback --system lorenz
python -B code/run.py pendulum discover
python -B code/run.py pendulum feedback
```

The `feedback` commands reuse the recorded first-round fits and author-confirmed
consolidation. They do not silently replace that template with a new stochastic
search result. To use a newly searched expression, inspect its coefficient
mapping and proposed reductions before creating a separate feedback record.
See [SYMBOLIC_WORKFLOW.md](SYMBOLIC_WORKFLOW.md).

## Fixed-budget adaptation

```bash
python -B reproduce.py budget --system fhn --method DSSDL-symbol --seed 1 --support 100
python -B reproduce.py budget --system lorenz --method CAVIA --seed 1 --support 100
```

The default budget is 10,000 CPU updates. Methods are `DSSDL-symbol` (the original
prescribed-structure model), `DSSDL` (no symbol), `CAVIA`, and `Reptile`. The driver
loads the saved source checkpoint and uses the matched historical source setting
for each record. Query errors do not choose the final iterate. The parameter
`--budget` can be changed for an explicitly separate diagnostic run. These are
normalized derivative-response errors, not state-trajectory errors.

## Distributional and online results

```bash
python -B reproduce.py wasserstein
python -m pip install -r code/requirements-online.txt
python -B reproduce.py online --system fhn --method pysr44 --seed 0
python -B reproduce.py online --system lorenz --method paper_no_symbol --seed 0
python -B reproduce.py online --system lorenz --method sindy_source --seed 0
```

The online package includes the two displayed scenarios: FHN jump-and-return
over [0, 0.9] and Lorenz slow sinusoidal variation over [20, 29], with three
initial conditions each. `pysr44` is the feedback-trained model labeled DSSDL in
the online panels. `paper_no_symbol` is the original neural-only counterpart.
`sindy_source` is the source-structured SINDy baseline, not independent
window-wise STLSQ. All use a trailing 0.5-second observation window. Forecasts
start from the currently observed state; they are not one unbroken free rollout.
`--smoke` runs only two update windows as an installation check, not a paper result.

Online replay loads the saved source-selected support and does not need PySINDy.
The archived source-selection function used PySINDy 2.0.0. Its dependency metadata
requires NumPy >=2.0, whereas the historical experiment environment used NumPy
1.26.4; do not install that optional package over the pinned replay environment.
Recreating the offline STLSQ search is separate from the documented replay commands.

Reruns refuse to reuse an existing output name. Use `--name` to select a new name.
The reference files under `results/` are never overwritten by these entry points.

## Provenance and scope

- Input observations, fitted checkpoints and published numerical results are
  copied from the recorded experiments; packaging does not create new evidence.
- Local filesystem references in historical JSON metadata are made relative.
  `records/source_manifest.json` identifies the original record and original and
  packaged hashes. `SHA256SUMS.json` covers the files in this release.
- The published table sources identify the exact numerical tables. Experiment
  result identities are retained: the static performance, discovery, paired
  controls and online deployment do not share a single universal checkpoint.
- The complete raw CFD acquisition/POD-generation workflow and every historical
  trial are outside this package. Existing project downloads remain linked from
  the repository's main README.
- Model files and dataset pickles should be loaded only from a trusted copy of
  this release. No additional license is assigned to third-party dependencies.

---

<a id="chinese"></a>

# DSSDL：2026 年 11 月复现包

本目录包含当前论文使用的所选实验源码与数值记录。仓库其他原实现保留，不含审稿通信。后续格式维护不改变实验逻辑。

## 内容

| 目录 | 内容 |
|---|---|
| `code/` | 原模型模块、可移植驱动、源数据和所需模型 |
| `results/` | 部件对照预测、320 条对齐预算记录、原轨迹距离与所选在线预测 |
| `records/` | 协议、来源追踪及数值表来源 |
| `outputs/` | 本地重跑生成，不属于参考结果 |

实验与正文、SI 的对应及比较边界见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 环境与快速核验

测试环境为 Python 3.10、PyTorch 2.8.0、NumPy 1.26.4、SciPy 1.15.3；完整版本见 `code/environment_tested.json`。按机器选择 PyTorch CPU/CUDA 构建。在本目录执行：

```bash
python -m pip install -r code/requirements.txt
python -B verify.py --checksums
python -B code/run.py check
```

`verify.py` 重算成对预测的 40 个目标误差，核对 320 次运行与 64 行固定预算汇总，重算两个系统的 Wasserstein 结果并检查在线预测误差。`check` 加载包内数据与回填模型，检查符号值及梯度。两者都不训练模型。

## 部件对照

```bash
python -B code/run.py gate --system pendulum --seed 0
python -B code/run.py gate --system quadrotor --seed 0
python -B code/run.py regularizer-pendulum --seed 5800
python -B code/run.py regularizer-lorenz --seed 42044
```

门控种子为 0–4，单摆回填种子为 5800–5804，Lorenz 为 42044–42048。CPU 执行加 `--device cpu`；`all-controls` 运行整组。正则对照两臂保留相同符号模板，不是无符号消融。五次回填共用同一 teacher 和模板，并非五次独立 PySR 搜索。

## 发现与回填

新 PySR 搜索前安装 `code/requirements-symbolic.txt`。PySR 还需要 Julia 运行时及依赖，首次设置可能下载；核验已保存结果不需要 Julia。

```bash
python -B code/run.py discover --system fhn
python -B code/run.py feedback --system fhn
python -B code/run.py discover --system lorenz
python -B code/run.py feedback --system lorenz
python -B code/run.py pendulum discover
python -B code/run.py pendulum feedback
```

`feedback` 复用已有首轮拟合及作者确认的整理，不自动替换成新随机搜索结果。若使用新表达式，应先检查系数映射和删项，再建立独立回填记录。详见 [SYMBOLIC_WORKFLOW.md](SYMBOLIC_WORKFLOW.md)。

## 固定预算适配

```bash
python -B reproduce.py budget --system fhn --method DSSDL-symbol --seed 1 --support 100
python -B reproduce.py budget --system lorenz --method CAVIA --seed 1 --support 100
```

默认执行 10,000 次 CPU 更新。方法名为 `DSSDL-symbol`（原给定结构模型）、`DSSDL`（无符号）、`CAVIA`、`Reptile`。驱动加载已保存源检查点并匹配各记录的历史源设置；query 误差不选择最终迭代。`--budget` 可用于单独标注的诊断运行。这里评价标准化导数响应，而非状态轨迹。

## 分布与在线结果

```bash
python -B reproduce.py wasserstein
python -m pip install -r code/requirements-online.txt
python -B reproduce.py online --system fhn --method pysr44 --seed 0
python -B reproduce.py online --system lorenz --method paper_no_symbol --seed 0
python -B reproduce.py online --system lorenz --method sindy_source --seed 0
```

在线包包含两种展示工况：FHN 在 [0, 0.9] 间跳变后返回；Lorenz 在 [20, 29] 间慢正弦变化；每种有三个初值。`pysr44` 为在线图标作 DSSDL 的回填模型；`paper_no_symbol` 为原纯神经模型；`sindy_source` 为源任务选结构的 SINDy，不是每窗独立 STLSQ。

全部采用向后 0.5 秒观测窗。每段预测从当前观测状态出发，并非一条不断重置误差以外的完整自由轨迹。`--smoke` 只跑两个更新窗，不能作为论文结果。

在线回放加载已保存源结构，不依赖 PySINDy。历史源选择函数使用 PySINDy 2.0.0；其依赖声明要求 NumPy ≥2.0，而实验环境为 NumPy 1.26.4，不要把该可选包覆盖安装到锁定回放环境。重建离线 STLSQ 搜索与这里的回放命令不同。

重跑拒绝复用已有输出名；使用 `--name` 指定新名称。入口不覆盖 `results/` 下的参考文件。

## 来源与范围

- 输入观测、拟合模型及发表数值来自原实验；打包不产生新证据。
- 历史 JSON 本地路径已相对化。`records/source_manifest.json` 记录原记录及原始/打包哈希；`SHA256SUMS.json` 覆盖发布文件。
- 表格来源指明对应数值。静态性能、发现、成对对照与在线部署并不共用一个通用检查点。
- 完整原始 CFD 获取/POD 生成流程和全部历史尝试不在包内；旧下载见根 README。
- 仅从可信发布副本加载模型及 pickle 数据；第三方依赖仍遵循各自许可。
