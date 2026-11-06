# 原始核心实现

DSSDL_.py、DSSDL_mask.py、pretrain.py、finetune.py、evaluate.py及数据模块均按字节原样保留。pretrain_analytic_reference.py 是原 DSSDL-symbol/pretrain.py 的配置参考；两目录的 DSSDL_.py 完全相同，故不重复携带。不要仅切换 λ 就把含符号项的模型称为 w/o symbol。原入口可能包含历史早停和默认工况；本包最新配对消融从 02_code/run.py 运行。
