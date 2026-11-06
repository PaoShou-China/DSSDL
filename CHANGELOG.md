# Changes

## 0.2.0 - Research software usability and project page

- Added an installable, dependency-light `dssdl` command with ten experiment
  entry points, environment diagnostics, dry runs and preserved driver options.
- Standardized first-party Python formatting and added local quality checks and CI templates.
  Executable syntax trees of all 80 pre-existing Python files match the archive
  at commit `330d311`; vendored CAVIA code and all numerical data remain intact.
- Added 24 fast tests covering command behavior, archive discovery, static assets
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

Full model retraining and new symbolic searches were not performed in this
maintenance release. Browser layout/interaction checks are a separate test suite;
passing static checks alone does not establish their outcome.
GitHub workflow activation is pending repository authorization. The inactive
definitions are under `ci/templates/` and are not claimed as completed CI runs.
