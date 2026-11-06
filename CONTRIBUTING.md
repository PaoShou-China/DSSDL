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
