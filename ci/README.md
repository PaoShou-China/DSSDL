[English](#english) | [中文](#chinese)

<a id="english"></a>

# Continuous integration setup

The current publishing credential can push repository contents but cannot create
GitHub Actions workflows. The files in `templates/` are therefore **inactive**.
They are ordinary versioned configuration examples, not an alternative execution
mechanism and not a successful CI status.

After a repository administrator authorizes workflow changes, copy the two YAML
files into `.github/workflows/` and commit that change. The code job covers
Windows/Linux and Python 3.10/3.12; the scientific job checks the recorded data
and model inputs on Python 3.10. The website job runs in an isolated GitHub runner
and serves only the public `docs/` tree before testing desktop and mobile layouts.

GitHub Pages is published using **Deploy from a branch**, branch **main**, folder
**/ (root)**. The root page redirects to the static `docs/` site; no application
build is required. The public entry point is `https://paoshou-china.github.io/DSSDL/`.
The Pages deployment is separate from the inactive quality-check templates above.

Local code checks already have independent entry points in CONTRIBUTING.md.
The browser suite accepts only an HTTP(S) test deployment, never `file://`.

---

<a id="chinese"></a>

# 自动检查配置

当前发布凭据能够推送仓库内容，但不能创建 GitHub Actions 工作流。因此 `templates/` 中的配置**未启用**，只是版本化配置模板，并非替代执行机制，也不代表 CI 已成功运行。

仓库管理员授权工作流修改后，将两个 YAML 文件复制到 `.github/workflows/` 并提交。代码任务覆盖 Windows/Linux 和 Python 3.10/3.12；科学核验任务在 Python 3.10 下检查已记录数据与模型输入。网页任务在隔离的 GitHub runner 中仅提供公开 `docs/` 文件，再检查桌面与手机布局。

GitHub Pages 当前设置为 **Deploy from a branch**、**main**、**/ (root)**。根页面跳转到静态 `docs/` 页面，无需应用构建。公开入口为 [DSSDL 网页](https://paoshou-china.github.io/DSSDL/)。Pages 部署与上述未启用的质量检查模板是两回事。

本地代码检查入口见 CONTRIBUTING.md。浏览器套件仅接受 HTTP(S) 测试站点，不接受 `file://`。
