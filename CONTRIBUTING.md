# Contributing to TRACER

TRACER is under **active development**. We are actively expanding platform support,
documentation, reproducibility scripts, and community-facing functionality — and we
welcome contributions.

If you would like to contribute, please open a GitHub Issue or Pull Request. For larger
features, new platform support, benchmark additions, or collaborations, please email us
first (see [README](README.md#contact)) so we can coordinate design and avoid
duplicated effort.

## Reporting bugs

Open a GitHub Issue with:

- what you ran (command/API call) and the TRACER version (`python -c "import tracer; print(tracer.__version__)"`),
- the expected vs. actual behavior,
- a minimal reproducer if possible (the synthetic generators in `tests/` are a good starting point),
- the full traceback and your OS / Python version.

## Requesting features or platform support

Open an Issue describing the platform or feature and, for new platforms, the input
format (transcript columns or matrix layout) and a small example if you can share one.
For substantial additions, email us first so we can align on design.

## Development setup

```bash
git clone https://github.com/imlong4real/TRACER.git
cd TRACER
pip install -e ".[dev]"     # editable install with dev/test extras
```

A C compiler is required (TRACER ships Cython extensions). On macOS run
`xcode-select --install` first.

## Code style and tests

- Match the style of the surrounding code; keep changes focused.
- Add or update tests for any behavior change.
- Run the suite locally before opening a PR:

```bash
python -m pytest
```

The suite currently runs **212 passed / 4 skipped** (the skips are optional, data-dependent
checks). CI runs `pytest` on Ubuntu + macOS across supported Python versions; tests must
pass before merge. They cover the synthetic generator, both pipelines (segmented and
no-segmentation), and snapshot regression of pipeline outputs.

If you intentionally change pipeline outputs, regenerate the snapshot references and
commit the updated `tests/references/*.json`:

```bash
TRACER_UPDATE_REFERENCES=1 python -m pytest tests/test_pipeline_regression.py
```

## Pull request checklist

- [ ] Tests pass locally (`python -m pytest`).
- [ ] New/changed behavior is covered by tests.
- [ ] Snapshot references updated if pipeline outputs intentionally changed.
- [ ] No large data, outputs, or generated files committed (see `.gitignore`).
- [ ] Public-facing changes documented in the README where relevant.
- [ ] Clear, descriptive commit messages and PR description.

Thank you for helping improve TRACER!
