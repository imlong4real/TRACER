"""Project-folder data loading utilities.

The TRACER pipeline functions are DataFrame-in / DataFrame-out — file
IO is the caller's responsibility. This module is a thin convenience
layer for the common case of a project laid out by convention:

  <project>/
    data/
      <name>.parquet           # input transcript table
      npmi_bs*.csv             # bootstrap PMI cache (optional)

These helpers locate those files and load the full transcript table
into a DataFrame. ROI subsetting and VHD-style perturbations are
intentionally NOT here — those are bench-time tools, not production
transformations.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def discover_data_files(project_dir: Path | str) -> tuple[Path, Path | None]:
    """Locate the input parquet and (optional) bootstrap NPMI cache.

    Looks inside ``<project_dir>/data/`` for:
      - exactly one ``*.parquet``. If multiple, prefers the one whose
        filename contains the project folder name; otherwise raises.
      - any file matching ``npmi_bs*.csv`` (the bootstrap PMI cache).
        Returns the first match or ``None`` if no cache yet.

    Returns
    -------
    (parquet_path, npmi_cache_path_or_None)

    Raises
    ------
    FileNotFoundError
        If ``<project_dir>/data/`` doesn't exist or contains no parquet.
    ValueError
        If multiple parquets exist and project-name disambiguation
        cannot pick a single one.
    """
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"No data folder at {data_dir}; expected <project>/data/ with "
            "a .parquet input file."
        )
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No .parquet found in {data_dir}; expected the input transcript "
            "table here."
        )
    if len(parquets) == 1:
        parquet = parquets[0]
    else:
        proj_name = project_dir.name
        matching = [p for p in parquets if proj_name in p.name]
        if len(matching) == 1:
            parquet = matching[0]
        else:
            raise ValueError(
                f"Multiple .parquet files in {data_dir} and project-name "
                f"disambiguation failed (matched: {matching}); rename or "
                "remove duplicates so exactly one parquet remains."
            )
    npmi_caches = sorted(data_dir.glob("npmi_bs*.csv"))
    npmi_cache = npmi_caches[0] if npmi_caches else None
    return parquet, npmi_cache


def load_full_df(project_dir: Path | str | None = None,
                 parquet_path: Path | str | None = None) -> pd.DataFrame:
    """Load the full transcript table for a project.

    Pass exactly one of:
      - ``project_dir``: auto-discover ``data/*.parquet`` via
        :func:`discover_data_files`.
      - ``parquet_path``: direct file path (skips discovery).

    Returns
    -------
    DataFrame
    """
    if (parquet_path is None) == (project_dir is None):
        raise ValueError(
            "Provide exactly one of project_dir or parquet_path "
            "(got both or neither)."
        )
    if parquet_path is not None:
        return pd.read_parquet(parquet_path)
    parquet, _ = discover_data_files(project_dir)
    return pd.read_parquet(parquet)
