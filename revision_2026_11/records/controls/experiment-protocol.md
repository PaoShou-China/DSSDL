[English](#english) | [中文](#chinese)

<a id="chinese"></a>

# 二轮补充实验协议 v1

锁定日期：2026-10-27。原文件只读；所有新输出在本工作目录。该协议先于本轮结果建立。目的不是制造所有消融均变差的排序。

## D0：数据与任务

1. 门控对照使用第22轮已核查的真实数据读取器、原保存特征/标签和source标准化。实测单摆为文件1–7的source和文件8的target，不生成新的解析单摆，不添加第8个训练摆长；四旋翼为原27任务惯量网格和(.02,.05,.10)目标。均不新增工况。
2. 门控网络/预算沿用第22轮主配置：候选8维，128×128 SiLU，门logit初值0.5，温度1，模型与门lr=.005，AdamW weight_decay=1e-4；每任务每轮512点，batch128；门阶段100轮，source留出20%，1-SE规则；固定门后70轮、lr=.001。27是四旋翼任务数量，不是本试验候选维数。
3. 符号重构正则对照先使用第58轮实测单摆：原teacher、原随机80/20切分、20/100缩放、已记录的PySR候选整理模板。保留相同表达式，不重新给定真值方程、不新建库头。两臂同初始化，只切换lambda_reg=1/0，固定门，以免门变化混杂正则贡献。
4. 后续Lorenz正则对照应沿用第44轮teacher、整理模板与source/target观测，先审计更新对象再执行，不使用新目录临时生成的仅rho变化6任务。

## D1：单因素对照

- Gate-on vs gate-off：只改变门为全1并关闭门惩罚。其余架构、初始权重、随机抽样、source选择规则、训练预算及目标适配一致。门控阶段fs=0，故这一对照检验Stage I维数压缩，不冒称检验完整PySR流程。
- lambda=1 vs lambda=0：两臂均保留同一个符号式和网络；两臂均使用原Trainer相同的坐标更新和共享更新步骤。只比较反馈训练中的符号重构约束。该实验不等于w/o symbol，也不测试PySR自由搜索是否丢项。
- 本轮目标适配统一固定500步、Adam lr=.03、source平均坐标初始化；只用support拟合；最终迭代评价query，不用query早停/择优。门控source选择仅用source验证。
- 原query若重复包含support中的同一输入-标签行，本轮主指标排除这些重复行，并记录原样本数、剔除数及最终样本数；两臂使用相同剩余query。这是防止观测重复计分，不是新增工况，不能与旧全query数字混报。
- 没有基于目标结果的参数调整、seed剔除、保底强留一维或新加物理标签。门全部关闭应标记失败/退化，保留其计数。

## D2：执行级别

先运行门控seed0在两系统的成对技术验证，和实测单摆固定teacher、seed5800的成对反馈实验。pilot不用于五种子统计推断。

正式门控计划seed0–4；正则反馈可做seed5800–5804，但它们只重复固定teacher之后的反馈，不是5个独立端到端发现种子；表头必须如此说明。正式预算与本协议一致，除非实现错误需先修订协议、完整重跑两臂。

## 指标

- 标准化导数输出RMSE（source留出、目标适配前/后），明确不是状态轨迹RMSE。
- 门候选/有效维数、门全关次数、latent covariance participation ratio；不把坐标逐维相关性最大值当作恢复全部物理参数的证明。实物摆长近似，不作为正式精确真值指标。
- 符号模型单独NRMSE、完整输出NRMSE、神经项RMS、符号/神经分量内积。可附 ||fs||²/(||fs||²+||fn||²)，但该比值受分量抵消和标准化影响，不能单独证明可辨识性。
- 正则对照报告初始/最终系数及方程；固定结构下不能声称发现/丢失某个项，除非后续同预算PySR重搜索确实执行。
- 运行时间、硬件、source更新与target步数记录，但不把同epoch数写成严格等算力。

## 成功判据与允许结论

门控若显著压缩维数而精度相近，可支持紧凑表示，不必强求RMSE更低。若压缩同时损害精度，应如实报告权衡。正则若改善符号单独拟合而不改变完整预测精度，可支持符号可解释部分的约束作用，不扩大为所有外推必然提升。只有真实运行后才允许向正文和回复填数。

数据访问、初始权重哈希、输入文件哈希、split/批次指纹、参数、软件和输出明细随每次运行保存。原程序不修改，原结果不覆盖。

## 执行核对补记（非事后调整）

原数据读取后确认：实测单摆7个source文件分别为6230、5545、4849、4313、3714、3390、2731行；target support为50行，原query为2168行，其中50行与support完全相同，按上述预定规则剔除后评价2118行。门控与正则对照一致。

四旋翼27个source任务各2500行；target support为500行，query为2500行，无完全重复support行被剔除。每个系统仍只有原来的一个目标配置，本轮多种子统计并未增加不同物理参数的目标系统。

## Lorenz执行协议（训练前锁定，2026-10-27）

直接复用第44轮PySR首次整理后的模板、共享系数初值与坐标初始化，以及第42轮记录的无符号teacher；不重新生成数据、不补任何未发现项、不使用物理参数或decoder。模板为`Q1*(y-x)`、`Q2*x+c1*x*z+c2*y`、`c3*x*y+Q3*z`。中心与尺度来自已发现系数的source统计，不是真值参数拟合。

沿用第44轮反馈训练：原64×64网络，3维任务坐标；每个source文件前80%训练、后20%验证；100轮，batch512；mapper Adam lr=.001以及全参数Adam lr=.001、weight_decay=.0001依次更新。两臂均保留同一模板与可训练共享数值系数，只令符号重构权重为1或0。双方完整跑完100轮，再按相同source完整模型验证误差选检查点，不用target选择。此选择规则与单摆原Trainer的最终轮不同，分系统明确报告。

反馈随机种子预设为42044–42048：第一项与第44轮原训练种子一致，其余依次增加；只改变固定teacher之后的批次顺序。每对初值、source切分和target数据相同，保存指纹及逐轮记录。目标适配仍用本轮统一的500步Adam、lr=.03、源坐标均值初值，只优化目标坐标，query不参与早停。最终指标保留符号单独与完整模型的source/target标准化导数RMSE，报告mean±样本SD。单摆和Lorenz分别汇总，不将两种source划分/选择规则混称一致。

训练前数据核对：Lorenz为18个source文件，各1000行；目标support100行、query500行。目标标识10_20_2.67也在source列表内，观测行无完全重合，所以这是同参数配置下的新观测适配，不作为未见物理参数外推结果。主消融结论依据source完整/符号误差，target观测误差分列描述，不更换到一个新工况。

---

<a id="english"></a>

# Second-round supplementary experiment protocol v1

Locked on 2026-10-27, before these results. Original files remain read-only; new outputs go to the working directory. The purpose is not to force every ablation to rank worse.

## D0: Data and tasks

1. Gating controls reuse the audited experiment-22 real-data loaders, saved features/labels and source normalization. Measured pendulum sources are files 1–7 and the target is file 8; no synthetic pendulum or eighth training length is added. Quadrotor uses the original 27-task inertia grid and target (.02,.05,.10). No new scenario is introduced.
2. The gating configuration follows experiment 22: 8 candidate coordinates, 128×128 SiLU, initial gate logit .5, temperature 1, model/gate lr .005, AdamW weight decay 1e-4; 512 samples/task/epoch, batch128; 100 gating epochs, 20% source holdout and the 1-SE rule; then 70 fixed-gate epochs at lr .001. The number 27 denotes quadrotor tasks, not candidate dimensions.
3. Symbolic-reconstruction controls first use experiment-58 measured pendulum: original teacher, random 80/20 split, scaling by 20/100 and recorded consolidated PySR template. Keep the same expression; do not supply a new true equation or library head. Both arms share initialization; only lambda_reg=1/0 changes, with the gate fixed to avoid confounding.
4. The later Lorenz control must reuse the experiment-44 teacher/template and source/target observations, auditing update targets before execution. Do not use the newly generated six-task rho-only setting.

## D1: Single-factor controls

- Gate-on/off: set all gates to 1 and disable the gate penalty only. Architecture, initial weights, sampling, source selection, budget and target adaptation otherwise match. fs=0 during gating, so this tests Stage-I compression, not the full PySR workflow.
- Lambda=1/0: both retain the same symbolic expression and network and use the original Trainer's coordinate/shared update steps. This isolates the feedback reconstruction constraint, not the no-symbol ablation or term loss in a fresh PySR search.
- Target adaptation uses exactly 500 Adam steps, lr .03, initialized at the mean source coordinate. Fit support only; score the final iterate on query. No query early stopping or selection. Gating selection uses source validation only.
- If query includes identical input–label rows from support, exclude them from the new primary score and record original, removed and final counts. Both arms use the same remaining query. This prevents duplicate scoring; it is not a new scenario and must not be mixed with old full-query numbers.
- No target-driven tuning, seed exclusion, forced retention of at least one dimension or added physical labels. If all gates close, retain and count the failed/degenerate case.

## D2: Execution stages

First run paired technical checks for gating seed0 on both systems and pendulum feedback with the fixed teacher and seed5800. Pilots do not establish five-seed statistics.

The formal plan uses gating seeds0–4 and feedback seeds5800–5804. These repeat feedback after a fixed teacher, not five independent end-to-end discovery searches; table headings must say so. Budgets follow this protocol unless an implementation error requires a prior amendment and rerunning both arms.

## Metrics

- Normalized derivative-output RMSE on source holdout and target before/after adaptation; explicitly not state-trajectory RMSE.
- Candidate/effective dimensions, all-gates-off counts and latent covariance participation ratio. Maximum coordinate-wise correlation does not prove recovery of every physical parameter. Approximate pendulum lengths are not precise ground truth.
- Symbol-only NRMSE, full-output NRMSE, neural RMS and symbolic/neural inner product. The optional ratio ||fs||²/(||fs||²+||fn||²) depends on cancellation and normalization; it cannot alone prove identifiability.
- Initial/final coefficients and equations. Under a fixed structure, do not claim term discovery/loss unless a subsequent equal-budget PySR search was actually run.
- Record runtime, hardware and source/target update counts, but do not equate matching epochs with strictly equal compute.

## Success criteria and permitted conclusions

Reduced dimension with similar accuracy supports compact representation; RMSE need not improve. Report any accuracy/compression trade-off. Better symbol-only fitting without better full prediction supports the constraint's role in the interpretable component, not universal extrapolation gains. Insert numbers into the manuscript/response only after execution.

Save data access, initial-weight/input hashes, split/batch fingerprints, settings, software and outputs for every run. Do not modify original programs or overwrite original results.

## Execution audit note (not a retrospective adjustment)

The seven measured-pendulum source files contain 6230, 5545, 4849, 4313, 3714, 3390 and 2731 rows. Target support has 50 rows; the original query has 2168, including 50 exact support duplicates. The prespecified exclusion leaves 2118 query rows, consistently for gating and regularizer controls.

Each of the 27 quadrotor source tasks has 2500 rows. Target support has 500 and query 2500, with no exact support duplicates removed. Each system still has its original single target configuration; multiple seeds do not add target systems with different physical parameters.

## Lorenz execution protocol (locked before training, 2026-10-27)

Reuse experiment44's first consolidated PySR template, shared coefficient initialization and coordinate initialization, and the experiment42 no-symbol teacher. Do not regenerate data, add missing terms or use physical parameters/decoders. The template is `Q1*(y-x)`, `Q2*x+c1*x*z+c2*y`, `c3*x*y+Q3*z`. Centers/scales come from source statistics of discovered coefficients, not ground-truth fitting.

Follow experiment44 feedback: original64×64 network, three task coordinates; first80% of each source file for training, last20% for validation; 100epochs, batch512; mapper Adam lr .001 followed by all-parameter Adam lr .001, weight decay .0001. Both arms retain the same template and trainable shared coefficients, changing only reconstruction weight1/0. Complete all100epochs in both arms, then select by the same source full-model validation error, never target error. This differs from the pendulum Trainer's final-epoch rule and is reported per system.

Feedback seeds are42044–42048. The first matches experiment44; the rest increment it, changing only batching after the fixed teacher. Each pair shares initial weights, source split and target data; save fingerprints and epoch logs. Target adaptation uses500Adam steps, lr .03 and mean source-coordinate initialization, optimizing only the target coordinates without query early stopping. Report source/target normalized derivative RMSE for symbol-only/full outputs as mean±sample SD. Summarize pendulum and Lorenz separately; do not call their source splitting/selection identical.

Before training, the Lorenz audit found18source files,1000rows each; target support100rows and query500rows. Target ID10_20_2.67 is also a source configuration, but observations do not exactly overlap. This is adaptation to new observations at the same physical configuration, not unseen-parameter extrapolation. Base the primary ablation conclusions on source full/symbol errors, list target errors separately, and do not replace the scenario.
