[English](#english) | [中文](#chinese)

<a id="english"></a>

# Original Real-Pendulum PySR Closure

Locked before first search, 2026-09-05.

- Start from `NC实验/pendulum/model/meta_train.pt`, original MetaNet class, 6 latent slots, 64x64 SiLU backbone and original hard/straight-through gate. The checkpoint initially activates zero-based slot 4 only.
- Read source tasks 1-7 only. Preserve saved columns `[u=omega/20,theta]` -> `[alpha/100,omega/20]`, dt=.035, all-source population mean/std, and original random point split (`torch.Generator().manual_seed(0)`, 80/20).
- No measured lengths, physical parameter labels, decoder, synthetic dynamics, derivative recomputation, row shifts or cross-task state/embedding combinations. Task filenames are identifiers only.
- Extract source teacher predictions only at observed state/task pairs. PySR receives stored state values and the checkpoint's active task coordinate; target is the teacher's output in original output-normalized coordinates. Convert selected expressions back to stored output units before reporting/injection.
- New search settings: operators +,-,*,sin,cos; 16 populations of 40; maxsize 25; 200 iterations; serial seed 5800+round*10+output; timeout 120 seconds/output, 350 observed training samples/task maximum. These are search settings, not changes to the original neural model. All Pareto candidates are retained and evaluated on the original source validation split. Select smallest candidate within 5% of minimum validation teacher NRMSE (absolute tolerance .002 normalized units).
- Human assistance after round 1 may algebraically expand/collect/simplify actual emitted expressions, rename coefficient-only functions and prune existing terms with an explicitly logged approximation error. No missing sin/velocity factors may be inserted based on the known pendulum equation. A different Pareto candidate may be selected with a written reason and its measured error.
- Feedback uses the original `Trainer.meta_train` implementation and original loaders/batch 128/100 epochs. Original Adam code reads global `fast_adapt_lr=.005`; preserve that executed behavior, despite the different override name in meta_train.py. Keep original gating behavior and record any active-slot changes. Turn on existing symbolic regularization at lambda=1; this is the necessary closure intervention. Numeric constants already in the simplified expression may train jointly with q. Preserve q coordinates; do not calibrate to physical parameters.
- Save final epoch as the primary feedback checkpoint, consistent with original training. Re-run the same search protocol on the updated full teacher for round 2. Keep all failed or inferior results.
- Main question: whether discovery supplies a pure-angle restoring term and a nonzero separately additive velocity term, and how simplification/feedback change fit. Report stored output units and correctly rescaled update units. A mixed-angle sine is not silently replaced by sin(theta).
- Metrics: pooled original-source validation NRMSE (full model and symbol-only), per-output NRMSE, normalized teacher approximation error, equation complexity, active latent dimensions, wall time. Original target support/query may be evaluated only after structure decisions are locked, and must not influence selection.
- No prediction-superiority claim from this exploratory single-checkpoint run. Source validation is reused for candidate ranking; it is not a new untouched generalization test. A clean continuous physical-law recovery claim remains subject to the known finite-difference time-position ambiguity documented in round57.

If the result needs a different search constraint, extra round or changed hyperparameter, append the change and reason before running that branch. Do not overwrite previous candidates.

## Human Intervention After First Search

The final automatic output0 choice is a mixed-angle sine plus u^2 (symbol-only pooled validation NRMSE .04427). A complexity-16 intermediate Pareto entry, observed while the same search was running, already contained `(sin(theta)+u)*(a*q+b)+u-u^2`. It is preserved as an explicitly transcribed intermediate observation, not mislabeled as the final Pareto entry. Its recomputed search-training MSE agrees with the observed PySR loss within 1e-8.

Under the user's permission for human simplification, use that separated candidate for interpretability. Algebraically collect its existing sin(theta) and u factors, remove its u^2 term and tiny offsets, and use the actually emitted near-identity linear candidate for output1. This increases pooled source-validation symbol NRMSE from .04717 (intermediate unpruned) to .05054 (pruned), compared with .04427 for the automatic final winner. It is an explicit approximation/tradeoff, not an equivalent simplification or automatic accuracy selection.

The injected form is `(c0*q+c1)*sin(theta)+(c2*q+c3)*u`, `c4*u`. Initially the two slopes share the value emitted by PySR. During feedback they may train separately; this relaxes an existing coefficient tie, does not add state factors or new task coordinates. The q coordinate remains the original active latent slot; no length calibration is used. The first search's grammar is unchanged.

---

<a id="chinese"></a>

# 原实测单摆 PySR 闭环

于首次搜索前 2026-09-05 锁定。

- 从 `NC实验/pendulum/model/meta_train.pt` 和原 MetaNet 启动：6 个潜在槽位、64×64 SiLU、原 hard/straight-through 门；初始只激活零基索引 4。
- 仅读源任务 1–7；保留列 [u=omega/20,theta] → [alpha/100,omega/20]、dt=.035、全源总体均值/标准差，以及原随机点 80/20 切分（torch.Generator().manual_seed(0)）。
- 不用实测摆长、物理参数标签、decoder、合成动力学、导数重算、行移位或跨任务状态/嵌入组合。文件名仅为 ID。
- 只在真实观测的状态/任务配对提取 teacher 预测。PySR 输入存储状态及激活任务坐标，目标是原输出标准化坐标中的 teacher 输出；报告/注入前还原到存储输出单位。
- 搜索算符 +、-、*、sin、cos；16 个种群、每群 40，maxsize25，200 迭代，串行；种子 5800+round*10+output；每输出超时 120 秒、每任务最多 350 个训练观测。这些是搜索设置，不改变原网络。保留全部 Pareto 候选并在原源验证集评价；在最低验证 teacher NRMSE 的 5% 范围（绝对容差 .002）内选最小式。
- 首轮后允许人工对实际输出展开、归并、化简、重命名纯系数函数，并在记录近似误差后剪项；不能按真单摆方程补入未发现的 sin/速度因子。改选另一 Pareto 候选必须记录理由与实测误差。
- 回填用原 Trainer.meta_train、原加载器、batch128、100 epoch。原 Adam 实际读取全局 fast_adapt_lr=.005，虽 meta_train.py 有不同覆盖名仍保留该执行行为。保持原门控并记录激活槽变化；开启已有符号正则 lambda=1，这是闭环所需干预。整理式已有数值常数可与 q 联训；保留 q，不做物理校准。
- 按原训练保留最终 epoch 为主检查点。第二轮对更新完整 teacher 使用同搜索协议；失败和较差结果也保存。
- 主要问题是能否得到纯角度恢复项与独立加性非零速度项，以及整理/回填如何改变拟合。报告存储单位和正确还原的更新单位；不把混合角度正弦悄悄替换为 sin(theta)。
- 指标：合并源验证 NRMSE（完整与符号单独）、逐输出 NRMSE、标准化 teacher 逼近误差、复杂度、激活维数、耗时。目标 support/query 只在结构锁定后评价，不影响选择。
- 单检查点探索不宣称预测优势；源验证被重复用于候选排序，不是未触碰泛化集。连续物理规律恢复仍受第 57 轮记录的有限差分时间位置歧义约束。

若要改变搜索约束、增加轮次或改超参数，先追加变更与原因再运行，不覆盖旧候选。

## 首轮后人工干预

output0 最终自动选择为混合角度正弦加 u^2，符号合并验证 NRMSE .04427。同次搜索运行中曾出现复杂度 16 的中间 Pareto 项，已含 `(sin(theta)+u)*(a*q+b)+u-u^2`。它以明确转录的中间观察保存，不冒充最终 Pareto 项；重算训练 MSE 与当时 PySR loss 在 1e-8 内一致。

获得人工化简许可后，为可解释性选用该分离候选。归并已有 sin(theta)、u，删去 u^2 和微小偏置，output1 用实际输出的近恒等线性候选。符号合并源验证 NRMSE 从 .04717（未剪中间候选）升至 .05054（剪后），自动最终胜出者为 .04427。这是明确的近似权衡，不是等价化简或自动精度选择。

注入形式为 `(c0*q+c1)*sin(theta)+(c2*q+c3)*u`、`c4*u`。两斜率初值来自 PySR 同一数值，回填时允许分别训练，放松系数绑定但不增加状态因子或任务坐标。q 保留原激活潜在槽位，不做摆长校准；首轮语法不变。
