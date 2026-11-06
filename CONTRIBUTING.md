[English](#english) | [中文](#chinese)

<a id="english"></a>

# Contributing

Start with an issue describing the observed problem and the experiment affected.
Keep scientific changes separate from packaging, formatting and website changes.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m ruff check src scripts tests --select E,F,I,UP,B --ignore E501
```

The code uses four-space indentation, 100-character formatting and Python 3.10
syntax. New interfaces use type annotations, concise docstrings and actionable
errors. Avoid importing PyTorch or starting an experiment when importing the CLI.
Pass subprocess arguments as lists; do not construct shell commands from options.
Do not automatically reorder historical imports with runtime side effects.

## Scientific integrity

- Preserve observations, splits, seeds, normalization and saved predictions.
- For formatting changes to historical code, run
  `python scripts/check_scientific_ast.py`. It compares executable syntax against
  the pre-cleanup release; docstring whitespace is ignored.
- Run `dssdl verify` and `dssdl check` after a source change. The saved-result check
  is not a substitute for a new training experiment when an algorithm changes.
- Refresh hashes only after reviewing intentional changes with
  `python scripts/refresh_checksums.py`. This updates packaged hashes, not original
  source hashes. Never use it to hide a failed check or changed observation.
- Write new outputs to a new directory. Record the configuration and metrics;
  do not replace historical evidence or change model identities.
- Describe approximate symbolic consolidation separately from exact algebra.

## Scope of quality checks

The complete first-party Python tree is formatted and checked for syntax-related
lint errors. New maintenance code (`src`, `scripts`, `tests`) additionally passes
the broader E/F/I/UP/B rules. Vendored CAVIA source remains unchanged under its
original license. This is deliberate provenance preservation, not a claim that
every historical script has been redesigned into a general-purpose library.

The conventions follow the practical separation of formatting, tests and
contribution review used in the
[scikit-learn developer guide](https://scikit-learn.org/stable/developers/index.html)
and [Ruff documentation](https://docs.astral.sh/ruff/).

## Website

The page is static HTML/CSS/JavaScript with local assets. Do not add tracking,
external font requests or mandatory network dependencies. Keep keyboard focus,
mobile overflow and reduced-motion preferences working. Concept imagery must
not be presented as experimental evidence. New numerical plots must identify
their source records and evaluation protocol.

Use `npm ci`, then `npm run format:check` and `npm run test:site` for website
checks. Browser tests use the public site by default; `DSSDL_SITE_URL` can point
to an explicitly authorized test deployment. They do not load arbitrary local
files into the browser.

Keep website translations in `docs/i18n.js`, with Chinese and English for every
entry. Do not translate commands, equations or numerical data. Maintain both
language sections in each document. Preserve third-party license texts verbatim.

---

<a id="chinese"></a>

# 开发指南

先提交 issue，说明问题及受影响实验。科学方法改动与打包、格式整理、网页改动分开提交。

## 开发检查

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m ruff check src scripts tests --select E,F,I,UP,B --ignore E501
```

采用四空格缩进、100 字符格式宽度与 Python 3.10 语法。新接口使用类型标注、简明文档字符串及可操作的错误提示。导入 CLI 时不要导入 PyTorch 或启动实验。子进程参数使用列表，不通过选项拼接 shell 命令；不要自动调整具有运行副作用的历史导入顺序。

## 科学结果保护

- 保留观测、切分、随机种子、标准化与保存预测。
- 历史代码格式改动后运行 `python scripts/check_scientific_ast.py`，与整理前版本比较可执行语法；忽略文档字符串空白。
- 源码修改后运行 `dssdl verify` 和 `dssdl check`。若算法改变，保存结果核验不能代替新的训练实验。
- 只有审查确认了有意改动后，才运行 `python scripts/refresh_checksums.py` 更新打包哈希。原来源哈希不变；禁止用刷新哈希掩盖检查失败或观测改动。
- 新输出放新目录，记录配置和指标，不替换历史证据或混淆模型版本。
- 近似符号整理与精确代数变换分开说明。

## 检查范围

所有第一方 Python 代码统一格式并通过语法相关静态检查；新增维护代码（`src`、`scripts`、`tests`）另通过 E/F/I/UP/B 规则。第三方 CAVIA 源码按原许可保留，不做改写。此举旨在保留来源，并不表示所有历史脚本都已重构为通用库。

格式、测试与贡献审查的分离参考 [scikit-learn 开发指南](https://scikit-learn.org/stable/developers/index.html)和 [Ruff 文档](https://docs.astral.sh/ruff/)。

## 网页与双语维护

网页是带本地资源的静态 HTML/CSS/JavaScript。不要增加跟踪、外部字体请求或强制联网依赖。保留键盘焦点、手机布局及减少动态效果设置。概念图不能充当实验证据；新增数值图须注明记录来源和评价协议。

执行 `npm ci`，再运行 `npm run format:check` 和 `npm run test:site`。浏览器测试默认访问公开网站；`DSSDL_SITE_URL` 可指定明确获准的测试站点，不向浏览器加载任意本地文件。

网页词条统一放在 `docs/i18n.js`，每项依次提供中文与英文；不要改动公式、命令或实验数值来适应语言。文档保留同一文件的两个语言入口；新增说明需同步翻译。第三方许可证原文不翻译替换。
