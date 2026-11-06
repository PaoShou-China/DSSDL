[English](#english) | [中文](#chinese)

<a id="chinese"></a>

# 原始核心实现

DSSDL_.py、DSSDL_mask.py、pretrain.py、finetune.py、evaluate.py 及数据模块最初按原样归档。后续全仓库格式维护保留其可执行语法，而非字节完全相同，见根目录 CHANGELOG。

pretrain_analytic_reference.py 是原 DSSDL-symbol/pretrain.py 的配置参考；两目录的 DSSDL_.py 原本完全相同，故不重复携带。不要仅切换 λ 就把含符号项的模型称为 w/o symbol。原入口可能包含历史早停和默认工况；当前配对消融从仓库根目录的 `revision_2026_11/code/run.py` 运行。

---

<a id="english"></a>

# Original core implementation

The archival note above describes the originally copied model and data modules: `DSSDL_.py`, `DSSDL_mask.py`, `pretrain.py`, `finetune.py` and `evaluate.py`. The subsequent repository-wide formatting pass preserves their executable syntax, not byte identity; see the root CHANGELOG.

`pretrain_analytic_reference.py` records the configuration from the original `DSSDL-symbol/pretrain.py`. The two original `DSSDL_.py` implementations were identical, so only one is included. A model retaining the symbolic term must not be called “w/o symbol” merely because its regularization weight is changed. Historical entry points may retain early stopping and their original default scenarios. For current paired controls, use `revision_2026_11/code/run.py` from the repository root.
