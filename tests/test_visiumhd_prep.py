"""VisiumHD seg-input prep: registration guard + CLI surface.

The bin->polygon overlay is only valid when the binned and segmented
outputs share a coordinate frame. They do when both come from the same
spaceranger run; they did NOT in one copy of the PDAC sample (binned
3.0.1 @ 5.7499 um/px vs segmented 4.0.1 @ 0.46428 um/px), where a direct
overlay silently mis-registers every bin. The prep now reads
microns_per_pixel from both spatial dirs and refuses to run on a mismatch
unless explicitly overridden.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_spec = importlib.util.spec_from_file_location(
    "prep_vhd", ROOT / "scripts" / "prepare_visiumhd_seg_input.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _spatial(tmp_path: Path, mpp) -> Path:
    d = tmp_path / "spatial"
    d.mkdir(parents=True, exist_ok=True)
    payload = {} if mpp is None else {"microns_per_pixel": mpp}
    (d / "scalefactors_json.json").write_text(json.dumps(payload))
    return d


def test_reads_microns_per_pixel(tmp_path):
    assert prep._read_microns_per_pixel(_spatial(tmp_path, 0.4642835)) == pytest.approx(0.4642835)


def test_missing_scalefactors_returns_none(tmp_path):
    d = tmp_path / "spatial"
    d.mkdir()
    assert prep._read_microns_per_pixel(d) is None


def test_missing_key_returns_none_not_crash(tmp_path):
    assert prep._read_microns_per_pixel(_spatial(tmp_path, None)) is None


def test_same_frame_and_mismatch_are_distinguishable(tmp_path):
    """The guard's decision variable: scale = binned_mpp / segmented_mpp."""
    same = prep._read_microns_per_pixel(_spatial(tmp_path / "a", 0.4642835))
    other = prep._read_microns_per_pixel(_spatial(tmp_path / "b", 0.4642835))
    assert abs(same / other - 1.0) < 0.02, "identical mpp must read as same frame"

    binned = prep._read_microns_per_pixel(_spatial(tmp_path / "c", 5.7499))
    seg = prep._read_microns_per_pixel(_spatial(tmp_path / "d", 0.4642835))
    scale = binned / seg
    assert abs(scale - 1.0) >= 0.02, "version mismatch must NOT read as same frame"
    assert scale == pytest.approx(12.38, abs=0.01), (
        "the PDAC tutorials-copy mismatch is the 5.7499/0.46428 = 12.38x case")


def test_cli_exposes_the_new_flags():
    """--cell-geojson is what makes prune_scope='cell' meaningful on VHD:
    without it cell_id == nucleus id, so the 'whole cell' IS the nucleus."""
    import argparse
    sys.argv = ["prep", "--matrix-dir", "m", "--spatial-dir", "s",
                "--geojson", "g", "--out", "o"]
    args = prep._parse_args()
    assert args.cell_geojson is None
    assert args.panel_genes_only is False      # default: explode ALL genes
    assert args.allow_frame_mismatch is False  # default: refuse a mismatch
