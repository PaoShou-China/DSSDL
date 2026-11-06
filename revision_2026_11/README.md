# DSSDL: November 2026 reproducibility package

This directory contains the selected experiment sources and numerical records
used in the current manuscript. Original implementations elsewhere in the
repository remain unchanged. No peer-review correspondence is included.

## Contents

| Directory | Contents |
|---|---|
| `code/` | Original model modules, portable experiment drivers, source data and required checkpoints |
| `results/` | Saved component-control predictions, 320 matched-budget records, original-trajectory distances and selected online forecasts |
| `records/` | Protocols, source provenance and the corresponding numerical table sources |
| `outputs/` | Created locally by reruns; not part of the reference results |

See [EXPERIMENTS.md](EXPERIMENTS.md) for the mapping from each experiment to the
manuscript and SI, including the limits of each comparison.

## Environment and quick checks

The tested environment uses Python 3.10, PyTorch 2.8.0, NumPy 1.26.4 and SciPy
1.15.3. Full versions are in `code/environment_tested.json`. Choose the appropriate
PyTorch CPU/CUDA build for your machine. Run commands from this directory:

```bash
python -m pip install -r code/requirements.txt
python -B verify.py --checksums
python -B code/run.py check
```

`verify.py` recomputes 40 target errors from the paired-control predictions,
checks all 64 fixed-budget aggregate rows against 320 saved runs, recomputes the
two-system Wasserstein results, and checks the selected online forecast errors.
`check` loads the included data and feedback models and checks the symbolic
component's numerical evaluation and gradients. Neither command trains models.

## Component controls

```bash
python -B code/run.py gate --system pendulum --seed 0
python -B code/run.py gate --system quadrotor --seed 0
python -B code/run.py regularizer-pendulum --seed 5800
python -B code/run.py regularizer-lorenz --seed 42044
```

Gate seeds are 0-4; pendulum feedback seeds are 5800-5804; Lorenz feedback seeds
are 42044-42048. Add `--device cpu` for CPU execution. `all-controls` runs the
complete set. The regularizer comparison retains the same symbolic template in
both arms; it is not the no-symbol ablation. Five feedback repeats share one
teacher and template and are not five independent PySR searches.

## Discovery and feedback

Install `code/requirements-symbolic.txt` before launching a new PySR search.
PySR also requires its Julia runtime and packages; initial setup may download
them. Saved-result checks do not need Julia.

```bash
python -B code/run.py discover --system fhn
python -B code/run.py feedback --system fhn
python -B code/run.py discover --system lorenz
python -B code/run.py feedback --system lorenz
python -B code/run.py pendulum discover
python -B code/run.py pendulum feedback
```

The `feedback` commands reuse the recorded first-round fits and author-confirmed
consolidation. They do not silently replace that template with a new stochastic
search result. To use a newly searched expression, inspect its coefficient
mapping and proposed reductions before creating a separate feedback record.
See [SYMBOLIC_WORKFLOW.md](SYMBOLIC_WORKFLOW.md).

## Fixed-budget adaptation

```bash
python -B reproduce.py budget --system fhn --method DSSDL-symbol --seed 1 --support 100
python -B reproduce.py budget --system lorenz --method CAVIA --seed 1 --support 100
```

The default budget is 10,000 CPU updates. Methods are `DSSDL-symbol` (the original
prescribed-structure model), `DSSDL` (no symbol), `CAVIA`, and `Reptile`. The driver
loads the saved source checkpoint and uses the matched historical source setting
for each record. Query errors do not choose the final iterate. The parameter
`--budget` can be changed for an explicitly separate diagnostic run. These are
normalized derivative-response errors, not state-trajectory errors.

## Distributional and online results

```bash
python -B reproduce.py wasserstein
python -m pip install -r code/requirements-online.txt
python -B reproduce.py online --system fhn --method pysr44 --seed 0
python -B reproduce.py online --system lorenz --method paper_no_symbol --seed 0
python -B reproduce.py online --system lorenz --method sindy_source --seed 0
```

The online package includes the two displayed scenarios: FHN jump-and-return
over [0, 0.9] and Lorenz slow sinusoidal variation over [20, 29], with three
initial conditions each. `pysr44` is the feedback-trained model labeled DSSDL in
the online panels. `paper_no_symbol` is the original neural-only counterpart.
`sindy_source` is the source-structured SINDy baseline, not independent
window-wise STLSQ. All use a trailing 0.5-second observation window. Forecasts
start from the currently observed state; they are not one unbroken free rollout.
`--smoke` runs only two update windows as an installation check, not a paper result.

Online replay loads the saved source-selected support and does not need PySINDy.
The archived source-selection function used PySINDy 2.0.0. Its dependency metadata
requires NumPy >=2.0, whereas the historical experiment environment used NumPy
1.26.4; do not install that optional package over the pinned replay environment.
Recreating the offline STLSQ search is separate from the documented replay commands.

Reruns refuse to reuse an existing output name. Use `--name` to select a new name.
The reference files under `results/` are never overwritten by these entry points.

## Provenance and scope

- Input observations, fitted checkpoints and published numerical results are
  copied from the recorded experiments; packaging does not create new evidence.
- Local filesystem references in historical JSON metadata are made relative.
  `records/source_manifest.json` identifies the original record and original and
  packaged hashes. `SHA256SUMS.json` covers the files in this release.
- The published table sources identify the exact numerical tables. Experiment
  result identities are retained: the static performance, discovery, paired
  controls and online deployment do not share a single universal checkpoint.
- The complete raw CFD acquisition/POD-generation workflow and every historical
  trial are outside this package. Existing project downloads remain linked from
  the repository's main README.
- Model files and dataset pickles should be loaded only from a trusted copy of
  this release. No additional license is assigned to third-party dependencies.
