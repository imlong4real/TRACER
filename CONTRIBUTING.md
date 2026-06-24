# Contributing

## Tests

CI runs `pytest tests/` on Ubuntu + macOS × Python 3.10/3.11. Tests
must pass before merge. They cover the synthetic generator, both
pipeline workflows (segmented + no-seg) on the full volume and on a
5 µm tissue section, and snapshot regression of pipeline outputs.

To regenerate snapshot references after an intentional pipeline
change:

```
TRACER_UPDATE_REFERENCES=1 pytest tests/test_pipeline_regression.py
```

Then commit the updated `tests/references/*.json`.

## Benchmark log

Before opening a PR, run the benchmark and append the result to
`BENCHMARKS.md`:

```
python benchmarks/pr_benchmark.py 2>/dev/null >> BENCHMARKS.md
```

This measures TRACER recovery quality on synthetic data under two
scenarios: full-volume + ground-truth input (easy mode) and
sectioned + DAPI/Voronoi simulated segmentation (realistic mode).
The benchmark is **advisory, not gating** — but a reviewer will
expect the entry, and substantial regressions are reasonable
grounds to ask for changes before merge.

See `BENCHMARKS.md` for the format and the rationale.
