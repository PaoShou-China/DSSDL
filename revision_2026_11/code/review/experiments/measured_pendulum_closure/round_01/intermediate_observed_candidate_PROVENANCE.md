[English](#english) | [中文](#chinese)

<a id="english"></a>

# Intermediate Candidate Provenance

`intermediate_observed_candidate.csv` is a transcription of the complexity-16 row that the agent read from the running first-round output0 `hall_of_fame.csv`, before this run finished. It is not the final Pareto candidate at complexity 16: later search replaced that entry with a lower-error mixed-angle sine. The source text and loss appeared in the tool output of this conversation. This transcription is explicitly distinguished from PySR's final native `pysr/*/hall_of_fame.csv` and `candidates.json` artifacts.

No expression term or coefficient was inferred from a reference pendulum equation. The intermediate row already contains sin(theta), u, q-dependent amplitudes, and u^2. The agent selects this actual emitted intermediate row for interpretability, under the user's authorization for human intervention. Its score is recomputed on archived search_data before any simplification. The final automatic choice remains separately archived.

The original running file was `round_01/output0/pysr/20260905_172710_AdPnt6/hall_of_fame.csv`. That native file now contains the final, replaced entry. This separate transcription does not modify it. Recomputed training MSE is 0.0018175566313107302, agreeing with the observed native float32 loss 0.0018175564.

The feedback result is therefore human-assisted candidate selection and approximate simplification, not fully automatic final-Pareto selection. Reproducing the downstream feedback uses this saved candidate directly and does not depend on re-capturing transient search files.

---

<a id="chinese"></a>

# 中间候选的来源

`intermediate_observed_candidate.csv` 转录自首轮 output0 搜索运行中的 `hall_of_fame.csv` 复杂度 16 行，当时运行尚未结束。它不是最终复杂度 16 的 Pareto 项；之后该项被更低误差的混合角度正弦替换。原式与 loss 曾出现在工具输出中。此转录与 PySR 最终原生 `pysr/*/hall_of_fame.csv`、`candidates.json` 明确区分。

未根据参考单摆方程推断任何项或系数。中间项已包含 sin(theta)、u、q 相关幅值及 u^2。按用户授权，为可解释性选择这个实际输出过的中间项；任何化简前先在归档 search_data 上重算分数。最终自动选择另存。

当时文件为 `round_01/output0/pysr/20260905_172710_AdPnt6/hall_of_fame.csv`，当前原生文件已保存最终替代项，单独转录不修改它。重算训练 MSE 为 0.0018175566313107302，与观察到的原生 float32 loss 0.0018175564 一致。

因此回填是人工辅助候选选择与近似整理，不是完全自动的最终 Pareto 选择。复现后续回填直接使用保存候选，不依赖再次捕获短暂中间文件。
