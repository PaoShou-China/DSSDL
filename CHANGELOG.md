[English](#english) | [中文](#chinese)

<a id="english"></a>

# Changes

## Bilingual website and documentation

Added Chinese/English page switching, a remembered language preference, and
translations for controls, captions and status messages. Documentation and
archival protocols now include both languages in the same file. Commands,
equations, numerical data, checkpoints and experiment logic are unchanged.
Third-party license texts remain verbatim. The bilingual checks extend the
initial 25-test release; the current count is reported by the test runner.

## 0.2.0 - Research software usability and project page

- Added an installable, dependency-light `dssdl` command with ten experiment
  entry points, environment diagnostics, dry runs and preserved driver options.
- Standardized first-party Python formatting and added local quality checks and CI templates.
  Executable syntax trees of all 80 pre-existing Python files match the archive
  at commit `330d311`; vendored CAVIA code and all numerical data remain intact.
- Added 25 fast tests covering command behavior, archive discovery, static assets
  and website-to-record numerical agreement. Package builds as an sdist and wheel.
- Rechecked 20 paired controls, 320 fixed-budget records, 12 trajectory exports
  and 810 forecast windows. Model/data loading and symbolic gradients passed.
- Replayed six two-window online checks and four two-update neural baseline
  checks. Maximum online-state deviation from the archived records was below
  `6e-6`. These are installation/regression checks, not new training repetitions.
- Added a static Beihang-inspired project page, an original project emblem,
  concept hero image and generic workflow overview. Displayed numerical plots
  use the recorded finite trajectory intervals, not generated illustration data.
- Added installation, contribution and asset-provenance documentation.
- Published the website on GitHub Pages. Interactive checks at 1440 x 1000 and
  390 x 844 confirmed image loading, no horizontal overflow, system switching,
  trajectory playback and command selection/copying, with no browser errors.
  A root redirect supports the repository's root-directory Pages configuration.

Full model retraining and new symbolic searches were not performed in this
maintenance release. The committed standalone browser test suite has not yet
run in CI; the browser checks above were performed against the published site.
GitHub workflow activation is pending repository authorization. The inactive
definitions are under `ci/templates/` and are not claimed as completed CI runs.

---

<a id="chinese"></a>

# 更新记录

## 双语网页与文档

网页新增中文/英文切换、语言选择记忆，以及控件、图注、状态消息的完整翻译。项目说明与历史协议在同一文件内增加另一语言版本。命令、方程、数据、模型和实验逻辑均不因翻译改变。第三方许可原文保留。

## 0.2.0：研究代码易用性与项目网页

- 新增可安装、轻依赖的 `dssdl` 命令：十个实验入口、环境诊断、试运行预览，并保留原驱动选项。
- 统一第一方 Python 格式，增加本地检查和 CI 模板；80 个原 Python 文件的可执行语法与 `330d311` 一致。第三方 CAVIA 和所有数值数据保留。
- 此版本初次发布时加入 25 项快速测试，覆盖命令、归档定位、资源和网页数值一致性；可构建 sdist、wheel。之后双语检查继续扩展测试，当前数量以测试输出为准。
- 复核 20 组成对对照、320 条固定预算记录、12 份轨迹、810 个预测窗口；模型/数据加载和符号梯度通过。
- 回放六组双窗口在线检查和四组双更新神经基线检查；与原在线状态最大差异小于 `6e-6`。这是安装/回归检查，不是新训练重复。
- 增加北航风格静态页、原创标识、概念封面与通用流程总览。数值图来自记录的有限轨迹，而非生成图片。
- 增加安装、开发与资源来源说明。
- 网页发布至 GitHub Pages。1440×1000、390×844 交互检查确认图片正常、无水平溢出，切换、播放和命令复制可用，无浏览器错误；根目录跳转支持现有 Pages 配置。

维护版本未执行完整重训练或新符号搜索。独立浏览器测试套件尚未在 CI 中运行；上面的交互核查针对已发布页面。工作流仍待授权，`ci/templates/` 中的模板未启用，不能称为已完成 CI。
