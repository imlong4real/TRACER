"""Reproducibility helpers shared across tracer submodules."""

import os
import random

import numpy as np
import pandas as pd

# ---------- Reproducibility helpers ----------
_REPRO_SEEDED = False


def set_reproducibility_seed(seed: int = 42) -> None:
    """
    Best-effort reproducibility seed setter.
    Note: PYTHONHASHSEED is honored only at interpreter start; we still
    set it for clarity and for subprocesses.
    """
    global _REPRO_SEEDED
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    _REPRO_SEEDED = True


def _ensure_reproducibility_seed(seed: int = 42) -> None:
    """Call once per process to stabilize hashes and RNG usage."""
    if not _REPRO_SEEDED:
        set_reproducibility_seed(seed)


def reproducibility_smoke_test(seed: int = 42) -> bool:
    """
    Small deterministic smoke test for pipeline reproducibility.

    Runs a tiny synthetic dataset twice and asserts bitwise-identical outputs
    for graph construction and spatial coherence labeling.
    """
    _ensure_reproducibility_seed(seed)

    # Lazy imports to avoid circular dependencies: graph and spatial both
    # import _ensure_reproducibility_seed from this module.
    from .graph import build_graph
    from .spatial import enforce_spatial_coherence_fast

    # Synthetic, deterministic coordinates with near-ties
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5000001, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["feature_name"] = ["A", "B", "C", "D", "A", "B"]
    df["transcript_id"] = np.arange(len(df)).astype(str)
    # The redesigned pipeline uses a single `tracer_id` column updated
    # in place by each stage. Seed it with a single label here.
    df["tracer_id"] = ["C1", "C1", "C1", "C1", "C1", "C1"]

    data1 = build_graph(df, k=3, dist_threshold=2.0, coord_cols=("x", "y", "z"))
    data2 = build_graph(df, k=3, dist_threshold=2.0, coord_cols=("x", "y", "z"))

    if not np.array_equal(data1.edge_index.numpy(), data2.edge_index.numpy()):
        raise AssertionError("build_graph is not deterministic")

    out1 = enforce_spatial_coherence_fast(
        df, build_graph_fn=build_graph, k=3, dist_threshold=2.0, show_progress=False
    )
    out2 = enforce_spatial_coherence_fast(
        df, build_graph_fn=build_graph, k=3, dist_threshold=2.0, show_progress=False
    )

    # Default `out_col` is now `tracer_id`; both stages mutated it in
    # place. Determinism check now reads the canonical column.
    if not out1["tracer_id"].equals(out2["tracer_id"]):
        raise AssertionError("spatial coherence labeling is not deterministic")

    return True
