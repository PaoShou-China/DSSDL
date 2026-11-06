[English](#english) | [中文](#chinese)

<a id="english"></a>

# Experiment 42: parameter-label-free PySR closure

Locked before execution, 2026-09-04. This is an exploratory implementation audit, not a final multi-seed claim.

## Information boundary

- FHN and Lorenz start from the same existing no-symbol source checkpoints used by experiment 41.
- Observed source states, derivative targets, and opaque task IDs are available. Physical parameter values, parameter-bearing filename parsing, physical decoders, analytic equations, and historical discovered equations are forbidden inputs to discovery/training.
- Source checkpoint pretraining already saw the source tasks/data. Held-out source partitions below assess symbolic distillation and continuation, not fully unseen-task generalization.
- No physical coordinate calibration or post-hoc scaling is performed. Learned coordinates remain q1, q2, q3.

## Automatic loop

1. Query the current full neural-plus-symbolic model on shared source-state probes for every learned source task vector.
2. Split the response into the mean over search-training tasks and the task-dependent remainder. This generic decomposition prevents the common response from hiding smaller task effects; it uses no equations or physical labels.
3. PySR searches common response from states and remainder from states plus learned coordinates. Operators are +, -, *. Standardization is estimated on search-training data only and inverted algebraically, not physically calibrated.
4. Expand PySR expressions with SymPy. Candidate terms must have been emitted by PySR; no analytic polynomial library is supplied. Refit coefficients and remove small contribution terms with a fixed threshold grid. Choose a sparse expression within 5% of the minimum held-out normalized RMSE. Save every candidate and selection decision. No integer snapping.
5. Inject the resulting differentiable expression into the original MetaNet interface and original loss: MSE(net+symbol, labels) + lambda MSE(symbol, labels), lambda=1. Resume source training with the original two-step Adam pattern (task mapper, then all model weights).
6. Repeat discovery on the updated model for a fixed two-round pilot budget. Save all rounds, including worse ones. Select the final available checkpoint by source validation derivative RMSE only, allowing round 0 to win. This fixed-budget pilot does not claim convergence.

## Fixed splits / budgets

- Seed 0, one source checkpoint per system. Two discovery/feedback rounds, 60 source continuation epochs per round, batch 512, Adam lr 0.001, weight decay 0.0001.
- Source continuation uses the first 80% of each source trajectory; last 20% is validation. Best post-update epoch per round is chosen by full-model source validation MSE.
- 384 common state probes, drawn from first-80% source states. A fixed 80% source-task partition is used for task-varying symbolic regression, and a fixed 80% probe partition for common-response regression.
- PySR per component/output: up to 100 iterations, 8 populations of 32, maxsize 25, serial execution, fixed seed, 45-second search timeout after Julia setup. Actual elapsed time and equation archives are saved.
- Sparsity thresholds are relative contribution levels: 0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03. Coefficients are refitted in normalized candidate columns.

## Comparisons and evaluation

- Initial no-symbol checkpoint; both closed-loop rounds; no-symbol continuation at the same cumulative epoch budgets. The no-symbol control shares data, splits, optimizer settings and seeds; it does not receive a symbolic penalty.
- Report full-model and symbol-only derivative RMSE in source normalization and raw units. Record teacher-distillation error separately; low distillation error is not proof of a true physical equation.
- Static target: adapt only task vector on the existing support trajectory, report the existing query derivative error. No query values affect search, training, adaptation or selection.
- Time-varying exploratory check: experiment-41 fast-sine target, seed 0, W=0.5 s, H=0.5 s, same 45 causal origins and support-only LBFGS settings. Reuse t/states only, never parameter_reference. Report state RMSE and failed forecasts; compare selected closure, initial no-symbol, and matched no-symbol continuation. The FHN amplitude remains that of experiment 41, not experiment 38.
- State forecasts reset to the observed state at each origin, freeze the adapted vector over the forecast, and do not read future observations until scoring.
- No target metric selects a round or search setting. No multi-seed robustness, full window grid, noise sweep, or physical recovery claim is made in this pilot.

## Verification

Runtime tripwires reject known parameter parsers/decoder fitters. Save checkpoint/input hashes, source split IDs, search inputs, expressions, training logs and evaluation arrays. Numerically compare differentiable symbolic evaluation to NumPy, test gradients, and verify protected inputs are unchanged.

---

<a id="chinese"></a>

# 实验 42：无物理参数标签的 PySR 闭环

执行前于 2026-09-04 锁定。这是探索性实现核查，不是最终多种子结论。

## 信息边界

- FHN、Lorenz 从实验 41 使用的同一批已有无符号源检查点启动。
- 允许使用源状态、导数监督、不透明任务 ID；发现/训练禁止使用物理参数值、含参数文件名解析、物理 decoder、解析真方程或历史已发现方程。
- 源检查点预训练已见过这些任务/数据；下述源留出评价符号蒸馏与续训，不代表全新任务泛化。
- 不做物理坐标校准或后验缩放，坐标仍为 q1、q2、q3。

## 自动循环

1. 在各学到的源任务向量下，用共享源状态探针查询当前神经加符号完整模型。
2. 将响应分成搜索训练任务上的均值和任务相关余项。该通用分解避免共性淹没较小任务差异，不使用方程或物理标签。
3. PySR 用状态搜索共性响应，用状态加学到的坐标搜索余项。运算符为 +、-、*；标准化只由搜索训练数据估计，再以代数方式还原，不做物理校准。
4. SymPy 展开表达式。候选项必须来自 PySR 输出，不提供解析多项式库。重拟合系数，并按固定阈值网格删去贡献小的项；在最低留出 NRMSE 的 5% 范围内选稀疏表达式。保存全部候选与选择，不吸附到整数。
5. 将可微表达式注入原 MetaNet 接口和原损失：MSE(net+symbol, labels) + lambda MSE(symbol, labels)，lambda=1。按原两步 Adam（任务映射、全模型权重）续训。
6. 固定两轮探索预算，对更新后的模型再次发现。所有轮次包括变差的轮次均保存；仅按源验证导数 RMSE 选择可用检查点，允许第 0 轮胜出。固定预算不构成收敛声明。

## 固定切分与预算

- 种子 0，每系统一个源检查点；两轮发现/回填，每轮续训 60 个源 epoch，batch512，Adam lr0.001，weight_decay0.0001。
- 每条源轨迹前 80% 续训、后 20% 验证；每轮选更新后完整模型源验证 MSE 最低的 epoch。
- 从前 80% 源状态抽 384 个共用探针；任务相关回归固定 80% 源任务划分，共性响应固定 80% 探针划分。
- 每分量/输出 PySR 最多 100 次迭代，8 个种群、每群 32，maxsize25，串行、固定种子；Julia 设置后搜索超时 45 秒，保存实际耗时与方程。
- 相对贡献阈值为 0、0.0001、0.0003、0.001、0.003、0.01、0.03；在标准化候选列上重拟合系数。

## 比较与评价

- 比较初始无符号模型、两轮闭环、相同累计 epoch 的无符号续训。无符号对照共用数据、切分、优化器和种子，不施加符号惩罚。
- 报告源标准化及原单位下完整/符号单独导数 RMSE；teacher 蒸馏误差单列，低蒸馏误差不证明物理真方程。
- 静态目标只在原 support 上适配任务向量，报告原 query 导数误差；query 不影响搜索、训练、适配或筛选。
- 时变探索沿用实验 41 快正弦目标、种子 0、W=0.5 秒、H=0.5 秒，同 45 个因果起点及 support-only LBFGS。只复用 t/states，不读 parameter_reference。报告状态 RMSE 和失败预测；比较选定闭环、初始无符号及预算匹配无符号续训。FHN 幅值沿用实验 41，不是实验 38。
- 每段从当前观测状态重置，预测期间冻结适配向量；评分之前不读未来观测。
- 目标指标不选择轮次或搜索设置。本探索不声明多种子稳健性、完整窗口网格、噪声扫描或物理恢复。

## 核验

运行时检查拒绝已知参数解析器/decoder 拟合函数。保存检查点和输入哈希、源切分 ID、搜索输入、表达式、日志、评价数组。比较可微符号与 NumPy 值，检查梯度与受保护输入不变。
