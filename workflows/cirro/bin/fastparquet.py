"""Narrow fastparquet API adapter used by TRACER's Xenium preprocessor.

The pinned TRACER 0.1.1 image includes PyArrow but omits the fastparquet
package imported by ``scripts/preprocess_xenium.py``. That script only uses
``ParquetFile.columns``, ``row_groups``, and ``iter_row_groups``. Implementing
that small interface over the already-pinned PyArrow runtime keeps the core
script unchanged and avoids installing packages during a Cirro run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


class ParquetFile:
    """Compatibility subset of :class:`fastparquet.ParquetFile`."""

    def __init__(self, path: str | Path):
        self._file = pq.ParquetFile(path)
        self.columns = list(self._file.schema_arrow.names)
        self.row_groups = list(range(self._file.num_row_groups))

    def iter_row_groups(self, columns: Iterable[str] | None = None):
        selected = list(columns) if columns is not None else None
        for index in self.row_groups:
            yield self._file.read_row_group(index, columns=selected).to_pandas()
