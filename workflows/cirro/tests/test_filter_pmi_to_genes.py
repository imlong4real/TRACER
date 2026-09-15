from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "bin" / "filter_pmi_to_genes.py"


def test_filters_without_retuning_and_records_overlap(tmp_path: Path) -> None:
    source = tmp_path / "panel.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("gene_i,gene_j,PMI,O,E,z\n")
        handle.write("A,B,0.125000,3,2.0,1.5\n")
        handle.write("A,C,-0.250000,1,2.0,-1.0\n")
        handle.write("B,C,0.500000,4,2.0,2.0\n")
        handle.write("A,D,9.000000,9,1.0,9.0\n")

    genes = tmp_path / "gene_counts.tsv"
    genes.write_text(
        "feature_name\tn_transcripts\nA\t10\nB\t8\nC\t4\nX\t1\n",
        encoding="utf-8",
    )
    output = tmp_path / "effective.csv.gz"
    receipt = tmp_path / "receipt.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pmi",
            str(source),
            "--gene-counts",
            str(genes),
            "--out",
            str(output),
            "--receipt",
            str(receipt),
            "--source",
            "cirro://reference/fixed",
            "--source-sha256",
            "a" * 64,
            "--chunksize",
            "2",
        ],
        check=True,
    )

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert handle.read().splitlines() == [
            "gene_i,gene_j,PMI,O,E,z",
            "A,B,0.125000,3,2.0,1.5",
            "A,C,-0.250000,1,2.0,-1.0",
            "B,C,0.500000,4,2.0,2.0",
        ]

    record = json.loads(receipt.read_text(encoding="utf-8"))
    assert record["panel_rebuilt_or_retuned"] is False
    assert record["source_pmi"]["pair_rows"] == 4
    assert record["platform"]["gene_count"] == 4
    assert record["overlap"]["gene_count"] == 3
    assert record["overlap"]["genes"] == ["A", "B", "C"]
    assert record["effective_pmi"]["gene_pair_edge_count"] == 3
