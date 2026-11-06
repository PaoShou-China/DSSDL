[English](#english) | [中文](#chinese)

<a id="english"></a>

# Experiment 44: human-assisted symbolic coefficient collapse

Locked before execution on 2026-09-04 following the user's explicit request.
The user permits human simplification but explicitly forbids filling missing
terms from the known system. This is NOT fully automatic structure discovery.

## Input and scope

- First label-free GPLearn expressions from experiment 37 and first label-free
  PySR expressions from experiment 42, FHN and Lorenz.
- Learned source task coordinates from the original no-symbol checkpoint.
- Existing source states and derivative supervision. No source/target physical
  parameter labels, physical decoders, target data or analytic coefficient values.
- Preserve the original code, old checkpoints/results, and manuscript. All new
  work lives here. Experiment 43 remains prepared but unexecuted.

## Manual structural choices (not truth-free automatic selections)

Numbers are not snapped. All shared numerical coefficients start at the mean
of their discovered coefficient function on search-training source tasks.
The choices below are manually selected from already present state factors.
They must NOT be described as uniquely minimal or algebraically equivalent
to the complete first-round expression.

FHN, both engines:

    dv = a*v + b*v^3 + c*w + Q1
    dw = d*v + Q2 + Q3*w

Q1 is the complete discovered state-independent coefficient of dv. Q2 is the
state-independent coefficient of dw. Q3 is the coefficient of w in dw.
Other state factors are removed. Weak q dependence in the coefficients of
v/v^3/w in dv and v in dw is projected to shared scalars a,b,c,d.

Lorenz, GPLearn:

    dx = Q1*(y-x)
    dy = Q2*x + a*x*z
    dz = b*x*y + c*z + Q3

Q1 = (coefficient_y(dx) - coefficient_x(dx))/2. This manually ties two nearly
opposite coefficients; the leftover common-sign component is removed.
Q2 = coefficient_x(dy). Q3 = the complete additive coefficient of dz.
No y term is inserted in dy, and c*z remains a shared, not task-dependent,
coefficient. Other first-round factors/biases are removed.

Lorenz, PySR:

    dx = Q1*(y-x)
    dy = Q2*x + a*x*z + b*y
    dz = c*x*y + Q3*z

Q1 is tied as above; Q2 = coefficient_x(dy); Q3 = coefficient_z(dz), which was
present and task-dependent in the FIRST PySR round. The task dependence of
the retained x*z and y coefficients is projected to shared scalars. Additional
state factors and additive bias functions are removed, even if their effect
is non-negligible; report the initial projection error explicitly.

## Coordinate and optimization changes

- The three new Q functions are evaluated on old learned source q coordinates
  to initialize three new task coordinates. No physical labels are used.
- For numerical conditioning only, use Q_j = mean_j + std_j*z_j with moments
  computed from these discovered coefficient values on source search-training
  tasks. These statistics are not physical calibration. Equations are reported
  in Q coordinates; model inputs use z coordinates.
- Treating Q functions as independent coordinates relaxes their old algebraic
  dependencies. This is reparameterization, not just renaming notation.
- Keep the original MetaNet hidden widths and three-dimensional task vector.
  Warm-start existing network weights; replace source task-vector values with
  normalized Q values. No trained inverse/physical decoder is fitted.
- Shared numerical coefficients are additional trainable scalar parameters in
  the new prior module, initialized only from first-round expressions. This is
  a disclosed extension; it is not the unchanged original fixed-prior method.
- Original two-step optimization and loss are reused:
  MSE(network + symbol, derivative) + MSE(symbol, derivative).
  100 source epochs, lr .001, batch 512; best source-time-validation full-model
  RMSE epoch. First80% source trajectories train, last20% validate. Pretraining
  already saw these sources: no unseen-neural-task claim.

## Next symbolic search and reporting

- The compact prior is fed back for 100 epochs. Save before/after exact prior,
  learned task coordinates, coefficient values, source symbol-only errors,
  pruning residual and retained/deleted state-factor records.
- A following symbolic search, when run, must use the same engine for its
  branch. It receives model responses, observed states and new z coordinates,
  never the known true equation. Save raw outputs without manually claiming
  they stayed compact. Search budgets must be recorded explicitly.
- Distinguish first-round simplification, feedback optimization and following
  symbolic rediscovery. A feedback checkpoint alone is not a second search.
- Do not make prediction-superiority or fully-automatic-recovery claims.
- Verify protected hashes, permitted state support, coefficient-function
  initialization, differentiable/numpy agreement, finite gradients, selection
  epoch, and source-only information access.

---

<a id="chinese"></a>

# 实验 44：人工辅助的符号系数整理

按用户明确请求于 2026-09-04 执行前锁定。允许人工化简，但禁止根据已知系统补齐未发现项；不是完全自动结构发现。

## 输入与范围

- 实验 37 首轮无标签 GPLearn、实验 42 首轮无标签 PySR，系统为 FHN 与 Lorenz。
- 原无符号检查点学到的源任务坐标。
- 已有源状态与导数监督；不使用源/目标物理参数标签、物理 decoder、目标数据或解析系数。
- 保留原代码、旧模型/结果和论文。新增工作放独立目录。实验 43 当时仅准备、未执行。

## 人工结构选择

数值不吸附。共享系数初值为发现系数函数在搜索训练源任务上的均值。以下选择只取已有状态因子，但不能称为唯一最简或与完整首轮式恒等。

FHN（两种引擎）：

```text
dv = a*v + b*v^3 + c*w + Q1
dw = d*v + Q2 + Q3*w
```

Q1 为 dv 完整的状态无关系数；Q2 为 dw 的状态无关系数；Q3 为 dw 中 w 的系数。移除其他状态因子；dv 中 v/v^3/w 及 dw 中 v 系数的弱 q 依赖投影为共享 a、b、c、d。

Lorenz（GPLearn）：

```text
dx = Q1*(y-x)
dy = Q2*x + a*x*z
dz = b*x*y + c*z + Q3
```

Q1=(coefficient_y(dx)-coefficient_x(dx))/2，人工绑定两个近似相反系数并移除同号余项。Q2=coefficient_x(dy)，Q3 为 dz 的完整加性系数。不向 dy 补 y 项，c*z 保持共享而非任务相关。其他首轮因子/偏置删去。

Lorenz（PySR）：

```text
dx = Q1*(y-x)
dy = Q2*x + a*x*z + b*y
dz = c*x*y + Q3*z
```

Q1 同上；Q2=coefficient_x(dy)；Q3=coefficient_z(dz)，该任务相关项已在 PySR 首轮出现。保留 x*z、y 系数的任务依赖投影为共享量。额外状态项和加性偏置删去，即使作用不可忽略也要明确报告初始投影误差。

## 坐标与优化变化

- 在旧源 q 上计算三个 Q 函数，以初始化三个新任务坐标；无物理标签。
- 仅为数值条件改善，采用 Q_j=mean_j+std_j*z_j；均值/标准差由发现系数在搜索训练源任务上的值计算，不是物理校准。报告 Q 坐标方程，网络输入 z。
- 将 Q 视为独立坐标会放松原代数依赖，不只是换名字。
- 保留原 MetaNet 宽度和三维任务向量；网络暖启动，源向量换成标准化 Q；不拟合逆映射/物理 decoder。
- 新先验模块增加可训练共享标量，只由首轮式初始化。这是明确披露的扩展，不是原固定先验方法完全不变。
- 复用两步优化与损失 MSE(network+symbol, derivative)+MSE(symbol, derivative)。100 源 epoch、lr0.001、batch512；前 80% 轨迹训练、后 20% 验证，选完整模型源验证 RMSE 最低轮。预训练已见源任务，不能称神经网络未见任务泛化。

## 后续搜索与报告

- 紧凑先验回填 100 epoch，保存前后精确先验、任务坐标、系数、源符号误差、删项残差及保留/删除项。
- 若执行后续搜索，各分支使用原引擎；只输入模型响应、观测状态、新 z，不输入真方程。保存原始结果，不能自行宣称仍紧凑；搜索预算明确记录。
- 区分首轮整理、回填优化及后续再发现。只有回填检查点不算第二次搜索。
- 不声称预测优势或完全自动恢复。
- 核验受保护哈希、允许状态项、系数初始化、可微/NumPy 一致性、有限梯度、选中 epoch 及仅源信息访问。
