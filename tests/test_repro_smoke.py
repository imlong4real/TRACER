"""Wrap ``tracer._repro.reproducibility_smoke_test`` as a pytest test.

The smoke test runs ``build_graph`` and ``enforce_spatial_coherence_fast``
twice on a synthetic 6-point dataset and asserts bitwise-identical
outputs. Verifies the determinism guarantee TRACER's reproducibility
helpers provide.
"""
from __future__ import annotations

from tracer._repro import reproducibility_smoke_test


def test_pipeline_determinism():
    """build_graph + enforce_spatial_coherence_fast are deterministic."""
    assert reproducibility_smoke_test(seed=42) is True
