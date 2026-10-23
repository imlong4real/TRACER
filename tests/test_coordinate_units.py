"""Coordinate units: pixel / z-plane-index input is converted ONCE to microns.

Every TRACER threshold is in microns. These tests pin the conversion contract
(``tracer.units``) and its wiring through ``preprocess_xenium.py`` and
``run_tracer.py``:

* physically equivalent pixel and micron inputs give the same canonical
  coordinates AND the same pipeline result;
* nothing is scaled twice (stamped inputs; a pixel scale on micron input);
* missing, contradictory or invalid calibration fails with a fix;
* discrete z-planes: plane index -> microns, after which ``g_z_um = "auto"``
  bins at the plane spacing (it does not convert by itself);
* 2D input stays 2D.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracer.config import CoordinatesConfig, load_config
from tracer.units import (
    METADATA_KEY,
    CalibrationError,
    read_micron_to_pixel_transform,
    read_parquet_stamp,
    resolve_calibration,
    to_microns,
)
from tracer.zscale import resolve_g_z_um

ROOT = Path(__file__).resolve().parents[1]

UM_PER_PX = 0.18      # arbitrary test calibration, not a platform default
Z_STEP_UM = 0.8


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclasses resolve their module here
    spec.loader.exec_module(mod)
    return mod


def _pixel_table(n=400, seed=0):
    """A CosMx-like table: x/y in pixels, z as integer plane indices."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.uniform(0, 5000, n),
        "y": rng.uniform(0, 5000, n),
        "z": rng.integers(0, 11, n).astype(np.int64),
        "feature_name": rng.choice(list("ABCDEF"), n),
    })


def _micron_twin(px: pd.DataFrame) -> pd.DataFrame:
    """The same transcripts expressed in microns."""
    um = px.copy()
    um["x"] = UM_PER_PX * px["x"].to_numpy(np.float64)
    um["y"] = UM_PER_PX * px["y"].to_numpy(np.float64)
    um["z"] = Z_STEP_UM * px["z"].to_numpy(np.float64)
    return um


CAL = CoordinatesConfig(xy_units="px", um_per_px=UM_PER_PX,
                        z_units="plane", z_step_um=Z_STEP_UM)


# --------------------------------------------------------------------------
# conversion contract
# --------------------------------------------------------------------------
def test_default_declaration_is_the_identity():
    df = _micron_twin(_pixel_table())
    out, rec = to_microns(df)
    for c in ("x", "y", "z"):
        np.testing.assert_array_equal(out[c].to_numpy(), df[c].to_numpy())
    assert rec["applied"]["xy"]["op"] == "none"
    assert rec["applied"]["z"]["op"] == "none"
    assert rec["input_units"] == {"xy": "um", "z": "um"}


def test_pixel_and_micron_inputs_are_equivalent():
    px = _pixel_table()
    before = px.copy()
    out, rec = to_microns(px, CAL, source="cli")
    um = _micron_twin(px)
    for c in ("x", "y", "z"):
        np.testing.assert_array_equal(out[c].to_numpy(), um[c].to_numpy())
    pd.testing.assert_frame_equal(px, before)        # input left untouched
    assert rec["calibration_source"] == "cli"
    assert rec["calibration"] == {"um_per_px": [UM_PER_PX, UM_PER_PX],
                                  "z_step_um": Z_STEP_UM}
    assert rec["applied"]["xy"]["op"] == "scale"
    assert rec["applied"]["z"] == {"op": "plane_index", "z_step_um": Z_STEP_UM,
                                   "formula": "z_um = 0.8 * z_plane"}


def test_anisotropic_pixels():
    px = _pixel_table()
    out, rec = to_microns(px, CoordinatesConfig(xy_units="px", um_per_px=[0.2, 0.25]))
    np.testing.assert_allclose(out["x"], 0.2 * px["x"])
    np.testing.assert_allclose(out["y"], 0.25 * px["y"])
    assert rec["applied"]["xy"]["um_per_unit"] == [0.2, 0.25]


def test_si_units_are_fixed_factors():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [0.5, 0.0], "z": [0.001, 0.002]})
    out, _ = to_microns(df, CoordinatesConfig(xy_units="mm", z_units="mm"))
    np.testing.assert_allclose(out[["x", "y", "z"]].to_numpy(),
                               [[1000.0, 500.0, 1.0], [2000.0, 0.0, 2.0]])


def _write_transform(path: Path, m: np.ndarray) -> Path:
    path.write_text("\n".join(" ".join(f"{v:.10g}" for v in row) for row in m) + "\n")
    return path


MERSCOPE_LIKE = np.array([[9.259259, 0.0, 3550.5],
                          [0.0, 9.259259, 1293.75],
                          [0.0, 0.0, 1.0]])          # micron -> mosaic pixel


def test_vendor_transform_maps_mosaic_pixels_into_microns(tmp_path):
    t = _write_transform(tmp_path / "micron_to_mosaic_pixel_transform.csv",
                         MERSCOPE_LIKE)
    rng = np.random.default_rng(1)
    um = rng.uniform(-500, 500, size=(200, 2))
    px = um @ MERSCOPE_LIKE[:2, :2].T + MERSCOPE_LIKE[:2, 2]
    df = pd.DataFrame({"x": px[:, 0], "y": px[:, 1]})
    out, rec = to_microns(df, CoordinatesConfig(xy_units="px", pixel_transform=str(t)))
    np.testing.assert_allclose(out[["x", "y"]].to_numpy(), um, atol=1e-9)
    assert rec["calibration_source"] == "vendor:micron_to_mosaic_pixel_transform.csv"
    assert len(rec["calibration"]["pixel_transform_sha1"]) == 40
    assert rec["applied"]["xy"]["op"] == "affine"
    # an explicit scale that AGREES with the transform is accepted
    to_microns(df, CoordinatesConfig(xy_units="px", um_per_px=1 / 9.259259,
                                     pixel_transform=str(t)))


def test_transform_and_scale_that_disagree_are_contradictory(tmp_path):
    t = _write_transform(tmp_path / "t.csv", MERSCOPE_LIKE)
    with pytest.raises(CalibrationError, match="contradictory"):
        resolve_calibration(CoordinatesConfig(xy_units="px", um_per_px=0.18,
                                              pixel_transform=str(t)))


@pytest.mark.parametrize("text, match", [
    ("1 0 0\n0 1 0\n", "3x3"),
    ("1 0 0\n0 1 0\n1 0 1\n", "last row"),
    ("0 0 0\n0 0 0\n0 0 1\n", "singular"),
    ("a b c\n0 1 0\n0 0 1\n", "parse"),
])
def test_malformed_transform_files_are_rejected(tmp_path, text, match):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(CalibrationError, match=match):
        read_micron_to_pixel_transform(p)


def test_missing_transform_file_is_reported(tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        resolve_calibration(CoordinatesConfig(
            xy_units="px", pixel_transform=str(tmp_path / "nope.csv")))


# --------------------------------------------------------------------------
# missing / contradictory / invalid calibration
# --------------------------------------------------------------------------
def test_pixels_without_calibration_fail_with_the_fix():
    with pytest.raises(CalibrationError) as e:
        to_microns(_pixel_table(), CoordinatesConfig(xy_units="px"))
    msg = str(e.value)
    assert "--um-per-px" in msg and "--pixel-transform" in msg
    assert "does not guess" in msg


def test_planes_without_step_fail_and_say_auto_is_not_a_conversion():
    with pytest.raises(CalibrationError) as e:
        resolve_calibration(CoordinatesConfig(z_units="plane"))
    assert "--z-step-um" in str(e.value) and "auto" in str(e.value)


@pytest.mark.parametrize("cfg, match", [
    (CoordinatesConfig(um_per_px=0.18), "twice"),             # micron input + px scale
    (CoordinatesConfig(xy_units="mm", um_per_px=0.18), "twice"),
    (CoordinatesConfig(z_step_um=0.8), "plane"),              # um z + plane step
    (CoordinatesConfig(z_um_per_px=0.8), "z_units = 'px'"),
    (CoordinatesConfig(z_units="px"), "z_um_per_px"),
])
def test_contradictory_declarations_are_rejected(cfg, match):
    with pytest.raises(CalibrationError, match=match):
        resolve_calibration(cfg)


@pytest.mark.parametrize("kwargs", [
    {"um_per_px": 0.0}, {"um_per_px": -0.18}, {"um_per_px": float("nan")},
    {"um_per_px": "abc"}, {"um_per_px": [0.1, 0.2, 0.3]},
    {"z_step_um": 0.0}, {"z_um_per_px": float("inf")},
    {"xy_units": "pixels"}, {"z_units": "slice"},
])
def test_invalid_values_are_rejected_at_config_time(kwargs):
    with pytest.raises(ValueError):
        CoordinatesConfig(**kwargs)


def test_non_integer_plane_indices_contradict_the_declaration():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0] * 3, "z": [-0.8, 0.0, 0.8]})
    with pytest.raises(CalibrationError, match="non-integer"):
        to_microns(df, CoordinatesConfig(z_units="plane", z_step_um=0.8))


def test_toml_declaration_round_trips(tmp_path):
    p = tmp_path / "u.toml"
    p.write_text('[coordinates]\nxy_units = "px"\num_per_px = [0.2, 0.25]\n'
                 'z_units = "plane"\nz_step_um = 1.5\n')
    c = load_config(path=p).coordinates
    assert c.um_per_px == (0.2, 0.25) and c.z_step_um == 1.5
    cal = resolve_calibration(c)
    assert cal.xy_affine == ((0.2, 0.0, 0.0), (0.0, 0.25, 0.0)) and cal.z_scale == 1.5


# --------------------------------------------------------------------------
# no double scaling
# --------------------------------------------------------------------------
def test_converted_frame_refuses_a_second_conversion():
    out, _ = to_microns(_pixel_table(), CAL)
    assert out.attrs[METADATA_KEY]["frame"] == "um"
    with pytest.raises(CalibrationError, match="second time"):
        to_microns(out, CAL)
    again, rec = to_microns(out)               # the default is still fine
    np.testing.assert_array_equal(again["x"].to_numpy(), out["x"].to_numpy())
    assert rec["already_canonical"] is True


# --------------------------------------------------------------------------
# discrete z-planes and 2D input
# --------------------------------------------------------------------------
def test_auto_bins_at_the_plane_spacing_only_after_conversion():
    px = _pixel_table()
    out, _ = to_microns(px, CAL)
    r = resolve_g_z_um(out["z"].to_numpy(), "auto")
    assert r.mode == "auto-discrete" and r.g_z_um == pytest.approx(Z_STEP_UM)
    # On raw indices "auto" measures a spacing of 1 and would treat the
    # planes as 1 um apart: auto-sizing is not a unit conversion.
    raw = resolve_g_z_um(px["z"].to_numpy(), "auto")
    assert raw.g_z_um == pytest.approx(1.0)


def test_two_d_input_converts_xy_only():
    px = _pixel_table().drop(columns="z")
    out, rec = to_microns(px, CoordinatesConfig(xy_units="px", um_per_px=UM_PER_PX))
    assert "z" not in out.columns
    assert rec["z_present"] is False and rec["applied"]["z"]["op"] == "absent"
    with pytest.raises(CalibrationError, match="no 'z' column"):
        to_microns(px, CAL)


@pytest.mark.parametrize("platform", ["cosmx", "merfish"])
def test_discrete_plane_presets_use_auto_and_imply_no_calibration(platform):
    cfg = load_config(platform=platform)
    assert cfg.stitch.g_z_um == "auto"
    assert cfg.coordinates == CoordinatesConfig()     # nothing from the name


def test_pipeline_result_is_unit_invariant():
    """Same tissue in pixels/planes and in microns -> same partition."""
    from tests.synthetic import (make_synthetic_npmi_panel_for_transcripts,
                                 make_synthetic_transcripts)
    from tracer.pipeline import run_segmented_pipeline

    df, gt = make_synthetic_transcripts(n_cells=8, voxels_per_cell_mean=80,
                                        tx_per_cell=25, n_genes=12, n_types=3,
                                        domain_z_um=8.0, seed=7)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    px = df.copy()
    px["x"] = df["x"].to_numpy(np.float64) / UM_PER_PX
    px["y"] = df["y"].to_numpy(np.float64) / UM_PER_PX
    px["z"] = np.round(df["z"].to_numpy(np.float64) / Z_STEP_UM).astype(np.int64)
    um = _micron_twin(px)

    conv, _ = to_microns(px, CAL)
    cfg = load_config(platform="cosmx")
    runs = []
    for frame in (um, conv):
        d = {}
        frame = frame.astype({"x": np.float32, "y": np.float32, "z": np.float32})
        out, prog = run_segmented_pipeline(frame, panel, cfg=cfg, diagnostics=d)
        runs.append((out, prog, d))
    (o1, p1, d1), (o2, p2, d2) = runs
    assert list(o1["tracer_id"]) == list(o2["tracer_id"])
    strip = lambda p: [{k: v for k, v in s.items() if k not in ("_ts", "stage_seconds")}
                       for s in p]
    assert strip(p1) == strip(p2)
    assert d1["z_binning"] == d2["z_binning"]
    assert d1["z_binning"]["mode"] == "auto-discrete"
    assert d1["z_binning"]["resolved_g_z_um"] == pytest.approx(Z_STEP_UM)


# --------------------------------------------------------------------------
# entry points: preprocess stamps, run_tracer records and refuses to re-scale
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pixel_parquet(tmp_path_factory):
    d = tmp_path_factory.mktemp("units")
    df = _pixel_table(n=300, seed=3)
    df["cell_id"] = np.where(np.arange(len(df)) % 5 == 0, "-1",
                             (np.arange(len(df)) // 20).astype(str))
    df["overlaps_nucleus"] = (np.arange(len(df)) % 2).astype(np.uint8)
    p = d / "tx_px.parquet"
    df.to_parquet(p, index=False)
    return d, p, df


def test_preprocess_converts_once_and_stamps(pixel_parquet):
    d, p, df = pixel_parquet
    pre = _load_script("preprocess_xenium")
    out = d / "std.parquet"
    assert pre.main(["--input", str(p), "--out", str(out),
                     "--summary-dir", str(d / "pre"),
                     "--xy-units", "px", "--um-per-px", str(UM_PER_PX),
                     "--z-units", "plane", "--z-step-um", str(Z_STEP_UM)]) == 0
    std = pd.read_parquet(out)
    np.testing.assert_allclose(std["x"], (UM_PER_PX * df["x"]).astype(np.float32))
    np.testing.assert_allclose(std["z"], (Z_STEP_UM * df["z"]).astype(np.float32))
    stamp = read_parquet_stamp(out)
    assert stamp["frame"] == "um" and stamp["stamped_by"] == "preprocess_xenium.py"
    summ = json.loads((d / "pre" / "preprocessing_summary.json").read_text())
    co = summ["coordinates"]
    assert co["input_units"] == {"xy": "px", "z": "plane"}
    assert co["calibration_source"] == "cli"
    assert co["calibration"]["z_step_um"] == Z_STEP_UM
    assert co["z_planes_um"]["n_planes"] == 11
    assert co["z_planes_um"]["g_z_um_if_auto"] == pytest.approx(Z_STEP_UM)

    # run_tracer: the stamped table is microns; a pixel declaration on it
    # would scale twice and is refused.
    rt = _load_script("run_tracer")
    import logging
    log = logging.getLogger("test_units")
    loaded, rec = rt.load_transcripts(out, log)
    assert rec["already_canonical"] is True
    assert rec["calibration_source"] == "cli"
    with pytest.raises(SystemExit, match="second time"):
        rt.load_transcripts(out, log, coords=CAL, coords_source="cli")
    # ... and so does preprocessing the stamped table again
    with pytest.raises(SystemExit, match="second time"):
        pre.main(["--input", str(out), "--out", str(d / "twice.parquet"),
                  "--xy-units", "px", "--um-per-px", "0.18"])


def test_preprocess_fails_fast_without_calibration(pixel_parquet, tmp_path):
    _, p, _ = pixel_parquet
    pre = _load_script("preprocess_xenium")
    with pytest.raises(SystemExit, match="um-per-px"):
        pre.main(["--input", str(p), "--out", str(tmp_path / "o.parquet"),
                  "--xy-units", "px"])
    assert not (tmp_path / "o.parquet").exists()


def test_run_tracer_records_units_and_resolved_z_bin(pixel_parquet, tmp_path,
                                                     monkeypatch):
    from tests.synthetic import (make_synthetic_npmi_panel_for_transcripts,
                                 make_synthetic_transcripts)

    df, gt = make_synthetic_transcripts(n_cells=6, voxels_per_cell_mean=60,
                                        tx_per_cell=20, n_genes=9, n_types=3,
                                        domain_z_um=8.0, seed=11)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    px = df.drop(columns="is_nuclear").copy()
    px["x"] = df["x"].to_numpy(np.float64) / UM_PER_PX
    px["y"] = df["y"].to_numpy(np.float64) / UM_PER_PX
    px["z"] = np.round(df["z"].to_numpy(np.float64) / Z_STEP_UM).astype(np.int64)
    tx = tmp_path / "tx.parquet"
    px.to_parquet(tx, index=False)
    npmi = tmp_path / "panel.csv.gz"
    panel.to_csv(npmi, index=False)

    rt = _load_script("run_tracer")
    outdir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "run_tracer.py", "--transcripts", str(tx), "--npmi", str(npmi),
        "--platform", "cosmx", "--outdir", str(outdir), "--sample-name", "u",
        "--xy-units", "px", "--um-per-px", str(UM_PER_PX),
        "--z-units", "plane", "--z-step-um", str(Z_STEP_UM)])
    assert rt.main() == 0
    rec = json.loads((outdir / "config_receipt.json").read_text())
    co = rec["coordinates"]
    assert co["input_units"] == {"xy": "px", "z": "plane"}
    assert co["calibration_source"] == "cli"
    assert co["applied"]["z"]["op"] == "plane_index"
    assert rec["config"]["coordinates"]["xy_units"] == "px"
    zb = rec["z_binning"]
    assert zb["requested_g_z_um"] == "auto" and zb["mode"] == "auto-discrete"
    assert zb["resolved_g_z_um"] == pytest.approx(Z_STEP_UM)
    refined = pd.read_parquet(outdir / "outputs" / "transcripts_tracer_refined.parquet")
    np.testing.assert_allclose(np.sort(refined["x"].to_numpy()),
                               np.sort((UM_PER_PX * px["x"]).to_numpy(np.float32)),
                               rtol=1e-6)
    assert "g_z_um = 0.8" in (outdir / "run_summary.md").read_text()


def test_preprocess_z_summary_on_continuous_z_stays_small():
    """Xenium-like continuous z: the streaming tracker gives up at the cap
    instead of holding every distinct value, and reports what "auto" does."""
    pre = _load_script("preprocess_xenium")
    t = pre._ZPlaneTracker()
    t.update(np.random.default_rng(0).uniform(10.0, 30.0, 100_000))
    r = t.report()
    assert t.overflow and len(t.values) == 0
    assert r["mode"] == "auto-continuous" and r["g_z_um_if_auto"] == 1.0
