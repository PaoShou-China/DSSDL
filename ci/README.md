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
