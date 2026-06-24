"""Shared pytest fixtures and path setup for the TRACER test suite.

Adds the repo root to ``sys.path`` so ``tests.synthetic`` is importable
even when the package is installed in a way that doesn't include the
test fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also ensure src/ is importable for editable installs that may not yet
# be built. Normal `pip install -e .` configures this, so this is just a
# defensive fallback.
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Lightweight-environment fallback: ``tracer/__init__.py`` pulls the full geo/
# torch/open3d stack. Metrics-only tests (PMI bootstrap) need just
# ``tracer.metrics`` (numpy/pandas/scipy/numba). When the full package can't be
# imported, register a minimal ``tracer`` namespace and load ``tracer.metrics``
# standalone (stubbing the unused ``geopandas`` top-level import in metrics.py).
# No-op when the full package imports cleanly (e.g. CI with everything installed).
try:  # pragma: no cover - environment-dependent
    import tracer  # noqa: F401
except Exception:
    import types as _types
    import importlib.util as _ilu

    sys.modules.setdefault("geopandas", _types.ModuleType("geopandas"))
    _pkg = _types.ModuleType("tracer")
    _pkg.__path__ = [str(_SRC / "tracer")]
    sys.modules["tracer"] = _pkg
    _spec = _ilu.spec_from_file_location(
        "tracer.metrics", str(_SRC / "tracer" / "metrics.py")
    )
    _metrics_mod = _ilu.module_from_spec(_spec)
    sys.modules["tracer.metrics"] = _metrics_mod
    _spec.loader.exec_module(_metrics_mod)


import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seed() -> int:
    """Default reproducibility seed for tests."""
    return 42


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a tmp ``<project>/data/`` containing a single synthetic
    parquet, suitable for ``tracer.data.discover_data_files`` tests.

    Returns
    -------
    Path to the project root (i.e. the parent of ``data/``).
    """
    proj = tmp_path / "syntheticproj"
    data = proj / "data"
    data.mkdir(parents=True)

    df = pd.DataFrame({
        "transcript_id": ["t0", "t1", "t2"],
        "feature_name": ["A", "B", "A"],
        "cell_id": ["c0", "c0", "c1"],
        "x": np.array([0.0, 1.0, 5.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 5.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    })
    df.to_parquet(data / "syntheticproj_df.parquet")
    return proj
