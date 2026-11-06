[English](#english) | [中文](#chinese)

<a id="english"></a>

# Third-party code

The two `cavia-master` subdirectories contain `regression/cavia_model.py` from
the project's existing CAVIA baseline. Each copy retains its original MIT
license (Copyright 2019 Luisa M. Zintgraf). The model is included so that the
matched-budget CAVIA entry point does not depend on an external checkout.

Scientific Python packages are installed separately and retain their own
licenses. The archival modules in `code/baseline_reference/GPlearn` and `SINDy`
also require their corresponding packages if used directly; neither is required
to recompute the distances from saved trajectories. Their presence does not
imply that every historical baseline script is a supported standalone launcher.
Use the commands in the package README for the checked reproduction paths.

---

<a id="chinese"></a>

# 第三方代码

两个 `cavia-master` 子目录包含原 CAVIA 基线的 `regression/cavia_model.py`。各副本保留原 MIT 许可证（Copyright 2019 Luisa M. Zintgraf）。包含模型的目的是让对齐预算 CAVIA 入口不依赖外部仓库副本。

科学计算 Python 依赖另行安装并遵循各自许可。直接使用 `code/baseline_reference/GPlearn` 和 `SINDy` 的归档模块时，也需要相应依赖；但从保存轨迹重算距离无需安装它们。归档源码的存在不意味着每个历史基线脚本都是受支持的独立运行入口。请使用实验包 README 中已核查的复现命令。

第三方许可证保留原始法律文本，不以本中文说明替换。
