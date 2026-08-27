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


# ---------------------------------------------------------------------------
# Review follow-ups: guards must not fail open, docs must match behaviour
# ---------------------------------------------------------------------------
def test_unverifiable_frame_warns_instead_of_passing_silently(tmp_path, capsys):
    """A missing/malformed scalefactors file must not read as "verified".

    The guard exists because a frame mismatch is *silent*. Skipping it without
    a word whenever either file is unreadable reproduces exactly the failure
    mode it was added to prevent, so the unverifiable case has to announce
    itself.
    """
    readable = _spatial(tmp_path / "binned", 0.4642835)
    missing = tmp_path / "segmented" / "spatial"       # never created
    assert prep._read_microns_per_pixel(readable) is not None
    assert prep._read_microns_per_pixel(missing) is None

    src = (ROOT / "scripts" / "prepare_visiumhd_seg_input.py").read_text()
    guard = src.split("_bmpp = _read_microns_per_pixel", 1)[1]
    assert "if _bmpp is None or _smpp is None:" in guard, (
        "the guard must branch explicitly on unreadable scalefactors rather "
        "than falling through a truthiness test")
    assert "UNVERIFIED" in guard, (
        "the unverifiable branch must warn; silence is indistinguishable "
        "from a passed check")


def test_orphan_nuclear_bins_are_detected_not_excluded():
    """Agreement measured only where both masks assigned is blind to the
    failure that matters: a cell mask covering nothing still "agrees"."""
    src = (ROOT / "scripts" / "prepare_visiumhd_seg_input.py").read_text()
    assert 'orphan = int(((nuc_only != "-1") & (bin_cell_id == "-1")).sum())' in src, (
        "the cross-check must count nuclear bins the cell mask failed to "
        "cover, not just disagreements among co-assigned bins")
    assert "orphan_frac > 0.05" in src, (
        "wholesale cell-mask failure should abort, not warn")


def test_metadata_reports_nuclei_and_cells_separately():
    """`n_nuclei_seeded` derived from `bin_cell_id` counted CELLS under
    --cell-geojson, because that variable holds the cell mask there."""
    src = (ROOT / "scripts" / "prepare_visiumhd_seg_input.py").read_text()
    meta = src.split('meta = {', 1)[1].split('\n    }', 1)[0]
    assert '"n_cells_seeded"' in meta and '"cell_id_source"' in meta, (
        "metadata must distinguish the cell count and name the seeding mask")
    n_nuclei_line = next(l for l in meta.splitlines() if '"n_nuclei_seeded"' in l)
    idx = meta.splitlines().index(n_nuclei_line)
    block = "\n".join(meta.splitlines()[idx:idx + 3])
    assert "cell_id_arr" in block, (
        "n_nuclei_seeded must come from the NUCLEUS overlay (cell_id_arr), "
        "not from bin_cell_id which is the cell mask under --cell-geojson")


def test_npmi_help_does_not_claim_to_restrict_the_explode():
    """--npmi stopped gating the explode; --panel-genes-only does."""
    parser_src = (ROOT / "scripts" / "prepare_visiumhd_seg_input.py").read_text()
    # Slice to the NEXT add_argument: the help string itself contains "(.gz)",
    # so splitting on the first ")" truncates mid-sentence.
    npmi_help = parser_src.split('p.add_argument("--npmi"', 1)[1] \
                          .split("p.add_argument(", 1)[0]
    assert "panel-genes-only" in npmi_help, (
        "--npmi help must point at the flag that actually restricts the explode")
    assert "does NOT restrict" in npmi_help, (
        "--npmi help must state plainly that it no longer gates the explode")
    assert "strongly recommended" not in npmi_help, (
        "stale claim that --npmi restricts the explode")


def test_readme_documents_cell_geojson_for_whole_cell_seeding():
    """Following the README must exercise the feature the PR ships.

    Without --cell-geojson, cell_id is nucleus-derived and prune_scope="cell"
    is a no-op on VisiumHD — the documented workflow would silently opt out of
    the default it is meant to demonstrate.
    """
    readme = (ROOT / "README.md").read_text()
    block = readme.split("prepare_visiumhd_seg_input.py", 1)[1][:1500]
    assert "--cell-geojson" in block, (
        "the documented VisiumHD invocation must pass --cell-geojson")
