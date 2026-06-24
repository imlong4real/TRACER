"""Tests for ``tracer.data`` (project-folder discovery + load helpers).

These exercise the new generic loader without depending on real data:
the ``tmp_project_dir`` fixture (in ``conftest.py``) creates a fresh
``<project>/data/`` with one synthetic parquet for each test.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracer.data import discover_data_files, load_full_df


def test_discover_unique_parquet(tmp_project_dir: Path):
    parquet, npmi_cache = discover_data_files(tmp_project_dir)
    assert parquet.suffix == ".parquet"
    assert parquet.is_file()
    # The fixture doesn't write a npmi_bs cache.
    assert npmi_cache is None


def test_discover_disambiguates_by_project_name(tmp_project_dir: Path):
    """When multiple parquets exist, prefer the one whose name contains
    the project folder name."""
    data = tmp_project_dir / "data"
    # Add a second, unrelated parquet
    other = pd.DataFrame({"x": [0.0]})
    other.to_parquet(data / "unrelated.parquet")

    parquet, _ = discover_data_files(tmp_project_dir)
    assert "syntheticproj" in parquet.name


def test_discover_raises_on_no_parquet(tmp_path: Path):
    proj = tmp_path / "empty"
    (proj / "data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        discover_data_files(proj)


def test_discover_raises_on_no_data_dir(tmp_path: Path):
    proj = tmp_path / "no_data"
    proj.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_data_files(proj)


def test_discover_finds_npmi_cache_via_glob(tmp_project_dir: Path):
    data = tmp_project_dir / "data"
    cache = data / "npmi_bs_full_pmi.csv"
    cache.write_text("gene_i,gene_j,NPMI\nA,B,0.5\n")

    parquet, npmi_cache = discover_data_files(tmp_project_dir)
    assert npmi_cache == cache


def test_load_full_df_round_trip(tmp_project_dir: Path):
    df = load_full_df(project_dir=tmp_project_dir)
    assert "transcript_id" in df.columns
    assert "feature_name" in df.columns
    assert len(df) == 3  # the fixture writes 3 rows


def test_load_full_df_requires_one_arg():
    with pytest.raises(ValueError):
        load_full_df()  # no project_dir, no parquet_path


def test_load_full_df_with_explicit_parquet_path(tmp_project_dir: Path):
    parquet, _ = discover_data_files(tmp_project_dir)
    df = load_full_df(parquet_path=parquet)
    assert len(df) == 3
