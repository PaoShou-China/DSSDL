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

For GitHub Pages, choose **Deploy from a branch**, branch **main**, folder
**/docs** in Settings > Pages. The site is static and needs no build command.
The intended URL is `https://paoshou-china.github.io/DSSDL/`; it should not be
advertised as live until deployment has completed and the URL has been checked.

Local code checks already have independent entry points in CONTRIBUTING.md.
The browser suite accepts only an HTTP(S) test deployment, never `file://`.
