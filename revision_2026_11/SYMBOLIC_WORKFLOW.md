[English](#english) | [中文](#chinese)

<a id="english"></a>

# Recorded symbolic discovery and feedback

The first-round PySR searches use state observations and learned task coordinates,
not physical parameter labels. The frozen neural model supplies the responses
for the search. FHN and Lorenz search mean and task-dependent responses separately;
the candidate terms are refitted and selected using the recorded source validation
and complexity rule. Search settings and expressions are in
`records/published_tables/search_settings.tex` and `first_round_expressions.tex`.

## Consolidation

For a term `a(q)*psi(x)`, the substitution `Q=a(q)` is exact. Removing terms,
tying coefficients or making previously tied coefficients independent changes
the model and is recorded as approximate consolidation. The feedback templates
were proposed with LLM assistance and confirmed by the authors. The code in
`symbolic_feedback/scripts/run_feedback.py` implements those recorded choices;
it is not a general-purpose autonomous simplifier.

- FHN retains `v`, `v**3`, `w` and an offset in the first equation, and `v`, `w`
  and an offset in the second. Learned coefficient functions initialize new task
  coordinates; shared numerical coefficients remain trainable.
- Lorenz retains `Q1*(y-x)`, `Q2*x+c1*x*z+c2*y` and `c3*x*y+Q3*z`. Tying the two
  linear coefficients in the first equation is approximate, not an identity.
- The measured pendulum uses an earlier candidate from the same search with
  separate `sin(theta)` and velocity terms, rather than the final automatic
  candidate containing `sin(theta+0.70709*u)`. Its quadratic velocity and constant
  terms are removed, and initially tied slopes may vary independently.

After consolidation, the structure stays fixed while task coordinates, shared
numerical coefficients and neural weights are fitted to observations. The saved
before/after models and equations preserve the sequence. Feedback errors assess
these chosen templates; repeated fixed-template training is not repeated discovery.

## General LLM prompt

The following prompt was tested retrospectively on anonymized first-round fits.
It was not the automated selection rule used to obtain all feedback templates.

```text
Read the fitted expressions as if you had to explain them to a colleague
who does not know the system. What is the simplest account that still
explains the distinctions visible in the input?

First rewrite without approximation: group repeated patterns and name a
combination only when the name makes a relationship easier to see.
Then consider what each proposed simplification would lose. Does it
merely change notation, or would two cases the fit distinguishes now
look the same? Use the supplied examples to judge that trade-off.
Prefer the simpler account when the distinction has little support;
a small number alone is not evidence of a small effect.

Give one preferred form for each output, define any new symbols, and
show what was discarded (original minus proposal). Briefly explain the
most consequential choice and what observation could overturn it.
Do not assume an underlying law or add terms absent from the input.
```

The prompt check retained extra terms and did not reproduce all author-confirmed
templates. For example, `a(q)*x+b(q)*y` can be collected exactly into
`Q*(y-x)+D*(x+y)`, with `Q=(b-a)/2` and `D=(a+b)/2`. Dropping the second term is
a separate approximation. This example explains the mapping; it was not supplied
as a hint in the anonymized prompt test.

---

<a id="chinese"></a>

# 已记录的符号发现与回填

首轮 PySR 使用状态观测和学到的任务坐标，不使用物理参数标签。冻结神经模型提供搜索响应。FHN、Lorenz 分别搜索平均响应与任务相关响应，再按记录的源验证及复杂度规则重拟合、筛选候选项。搜索设置和表达式见 `records/published_tables/search_settings.tex`、`first_round_expressions.tex`。

## 结构整理

对于 `a(q)*psi(x)`，令 `Q=a(q)` 是精确代换。删项、绑定系数或使原绑定系数独立会改变模型，记作近似整理。回填模板经 LLM 辅助提出、作者确认；`symbolic_feedback/scripts/run_feedback.py` 实现这些已记录选择，不是通用自主化简器。

- FHN 第一式保留 `v`、`v**3`、`w` 和偏置；第二式保留 `v`、`w` 和偏置。发现的系数函数初始化新任务坐标，共享数值系数保持可训练。
- Lorenz 保留 `Q1*(y-x)`、`Q2*x+c1*x*z+c2*y`、`c3*x*y+Q3*z`。第一式将两个线性系数绑定是近似处理，不是恒等式。
- 实测单摆采用同次搜索中更早、具有独立 `sin(theta)` 与速度项的候选，而非最终自动选中的 `sin(theta+0.70709*u)`。移除速度平方与常数项，初始绑定斜率之后可独立变化。

整理后结构固定，任务坐标、共享数值系数和神经权重继续拟合观测。前后模型与方程分别保存。回填误差评价这些选定模板；重复固定模板训练不等于重复发现。

## 通用 LLM 提示词

下列中文是上方英文提示词的译文。实际回溯测试使用英文提示词，输入为匿名化首轮拟合；它不是获得所有回填模板的自动筛选规则。此处翻译不代表又执行了一次中文提示词试验。

```text
阅读这些拟合表达式，设想需要向一位不了解该系统的同事解释它们。
在仍能解释输入中可见差异的前提下，最简洁的说明是什么？

首先进行不带近似的改写：归并重复模式；只有在新名称能使关系更易理解时，
才给某个组合命名。然后考虑每个化简会损失什么：它只是改变记号，
还是会让原拟合可区分的两种情形变得相同？用给定示例判断这种权衡。
当某种差异缺少支持时，优先采用更简单的说明；数值小本身不证明作用小。

为每个输出给出一个推荐形式，定义新符号，并列出被丢弃的部分
（原式减去建议式）。简述影响最大的选择，以及什么观测可能推翻它。
不要假设潜在真实定律，也不要加入输入中没有的项。
```

提示词检查保留了额外项，未复现全部作者确认模板。例如 `a(q)*x+b(q)*y` 可精确整理成 `Q*(y-x)+D*(x+y)`，其中 `Q=(b-a)/2`、`D=(a+b)/2`；删去第二项是另一步近似。该例仅解释映射，并未作为提示输入匿名测试。
