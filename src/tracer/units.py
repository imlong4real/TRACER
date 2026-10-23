"""Convert transcript coordinates into TRACER's canonical micron frame.

Every spatial threshold in TRACER is in microns (``rescue.bin_size_um``,
``stitch.dist_threshold_um``, ``split_phase1.dz_threshold_um``, the z-bin
``stitch.g_z_um``, ...). This module owns the ONE conversion from an input's
native units into that frame. Two operations that are easy to conflate are
kept apart:

* **Unit conversion** (here): pixels, z-plane indices, mm or nm -> microns.
  It needs a calibration, taken from a vendor metadata file when the vendor
  documents one (MERSCOPE ``micron_to_mosaic_pixel_transform.csv``) or given
  explicitly (``um_per_px``, ``z_step_um``). It is never inferred from
  coordinate ranges or from the platform name.
* **Bin sizing** (``tracer.zscale``, the ``*_bin_size_um`` knobs): choosing grid
  sizes for coordinates that are ALREADY microns. ``stitch.g_z_um = "auto"``
  measures the z-plane spacing of micron data; it converts nothing. On
  plane-index input it would measure a spacing of "1" and silently treat
  planes as 1 µm apart, which is why plane indices must be declared here.

Canonical frame
---------------
``x``, ``y``, ``z`` in microns, in the input's own orientation and origin
(scale-only calibrations) or in the vendor's micron frame (affine
transforms). Only relative geometry matters to the pipeline. Other geometric
inputs, such as segmentation polygons, must be supplied in the same frame.

Converting twice is prevented by a stamp: ``to_microns`` records the frame in
``DataFrame.attrs[METADATA_KEY]``, and ``preprocess_xenium.py`` writes the same
stamp into the output parquet's schema metadata. A stamped input accepts only
the default (micron) declaration.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tracer.config import CoordinatesConfig

__all__ = [
    "METADATA_KEY",
    "CalibrationError",
    "Calibration",
    "resolve_calibration",
    "read_micron_to_pixel_transform",
    "to_microns",
    "check_plane_indices",
    "read_parquet_stamp",
    "schema_with_stamp",
    "add_coordinate_args",
    "coordinates_from_args",
]

#: Key of the frame stamp in ``DataFrame.attrs`` and in parquet schema metadata.
METADATA_KEY = "tracer.coordinates"

_SI_TO_UM = {"um": 1.0, "mm": 1000.0, "nm": 1e-3}
#: Two calibrations of the same pixels may differ by rounding in a vendor
#: file, not by more than this (relative).
_AGREE_RTOL = 1e-3
_PLANE_ATOL = 1e-6


class CalibrationError(ValueError):
    """Missing, contradictory or invalid coordinate calibration."""


@dataclass(frozen=True)
class Calibration:
    """A complete calibration: how input x/y/z map into microns.

    ``xy_affine`` is the 2x3 matrix taking input ``(x, y, 1)`` to microns, or
    ``None`` when x/y are already microns. ``z_scale`` is microns per input z
    unit, or ``None`` when z is already microns.
    """

    xy_units: str
    z_units: str
    xy_affine: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    z_scale: float | None
    source: str
    detail: dict = field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        return self.xy_affine is None and self.z_scale is None

    def describe_xy(self) -> dict:
        if self.xy_affine is None:
            return {"op": "none", "reason": "x/y already in microns"}
        a = self.xy_affine
        if a[0][1] == 0.0 and a[1][0] == 0.0 and a[0][2] == 0.0 and a[1][2] == 0.0:
            return {"op": "scale", "um_per_unit": [a[0][0], a[1][1]],
                    "formula": f"x_um = {a[0][0]:g} * x; y_um = {a[1][1]:g} * y"}
        return {"op": "affine", "matrix_to_um": [list(a[0]), list(a[1])],
                "formula": "[x_um, y_um] = M @ [x, y, 1]"}

    def describe_z(self, has_z: bool) -> dict:
        if not has_z:
            return {"op": "absent", "reason": "no z column (2D input)"}
        if self.z_scale is None:
            return {"op": "none", "reason": "z already in microns"}
        if self.z_units == "plane":
            return {"op": "plane_index", "z_step_um": self.z_scale,
                    "formula": f"z_um = {self.z_scale:g} * z_plane"}
        return {"op": "scale", "um_per_unit": self.z_scale,
                "formula": f"z_um = {self.z_scale:g} * z"}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_micron_to_pixel_transform(path: str | Path) -> np.ndarray:
    """Read a vendor 3x3 micron->pixel affine (MERSCOPE format).

    MERSCOPE ships ``images/micron_to_mosaic_pixel_transform.csv``: three rows
    of three numbers mapping ``(x_um, y_um, 1)`` to mosaic pixels. Whitespace
    or comma separated. Returns the matrix as given (micron -> pixel).
    """
    p = Path(path)
    if not p.is_file():
        raise CalibrationError(f"pixel transform file not found: {p}")
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(line.replace(",", " ").split())
    try:
        m = np.array(rows, dtype=np.float64)
    except ValueError:
        raise CalibrationError(
            f"{p}: expected three rows of three numbers (a 3x3 micron->pixel "
            f"affine); could not parse {rows!r}") from None
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise CalibrationError(
            f"{p}: expected a finite 3x3 micron->pixel affine; got shape {m.shape}")
    if not np.allclose(m[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise CalibrationError(
            f"{p}: last row must be [0, 0, 1] for an affine transform; got {m[2].tolist()}")
    if abs(np.linalg.det(m[:2, :2])) < 1e-12:
        raise CalibrationError(f"{p}: transform is singular and cannot be inverted")
    return m


def resolve_calibration(coords: CoordinatesConfig | None = None, *,
                        source: str = "config") -> Calibration:
    """Turn a declaration into a complete calibration, or fail with a fix.

    ``source`` labels where the declaration came from (``"default"``,
    ``"config"``, ``"cli"``) for the receipt. A vendor transform file
    overrides it with ``"vendor:<file name>"``.
    """
    c = coords if coords is not None else CoordinatesConfig()

    if c.xy_units != "px":
        given = [k for k in ("um_per_px", "pixel_transform") if getattr(c, k) is not None]
        if given:
            raise CalibrationError(
                f"{' and '.join(given)} calibrate PIXEL coordinates, but "
                f"xy_units = {c.xy_units!r}. Remove {' and '.join(given)}, or "
                f"set xy_units = 'px' if x/y really are pixels. Applying a "
                f"pixel scale to coordinates that are already physical would "
                f"scale them twice.")
    if c.z_units != "plane" and c.z_step_um is not None:
        raise CalibrationError(
            f"z_step_um converts z-plane INDICES, but z_units = {c.z_units!r}. "
            f"Remove z_step_um, or set z_units = 'plane' if z holds plane "
            f"indices.")
    if c.z_units != "px" and c.z_um_per_px is not None:
        raise CalibrationError(
            f"z_um_per_px converts z in PIXELS, but z_units = {c.z_units!r}. "
            f"Remove z_um_per_px, or set z_units = 'px'.")

    detail: dict[str, Any] = {}
    xy_affine = None
    if c.xy_units == "px":
        user = c.um_per_px
        if c.pixel_transform is not None:
            p = Path(c.pixel_transform)
            m = read_micron_to_pixel_transform(p)
            inv = np.linalg.inv(m)
            xy_affine = (tuple(float(v) for v in inv[0]),
                         tuple(float(v) for v in inv[1]))
            # Microns per pixel along each INPUT axis: column norms of the
            # linear part of the pixel->micron map.
            sx = float(math.hypot(inv[0, 0], inv[1, 0]))
            sy = float(math.hypot(inv[0, 1], inv[1, 1]))
            if user is not None and not (
                    math.isclose(sx, user[0], rel_tol=_AGREE_RTOL)
                    and math.isclose(sy, user[1], rel_tol=_AGREE_RTOL)):
                raise CalibrationError(
                    f"contradictory calibration: um_per_px = {list(user)} but "
                    f"{p.name} implies {sx:.6g} x {sy:.6g} um/px. Keep one of "
                    f"them; the vendor transform is the documented value for "
                    f"its own mosaic pixels.")
            source = f"vendor:{p.name}"
            detail.update(pixel_transform=str(p), pixel_transform_sha1=_sha1(p),
                          micron_to_pixel=m.tolist(),
                          um_per_px_from_transform=[sx, sy])
            if user is not None:
                detail["um_per_px"] = list(user)
        elif user is not None:
            xy_affine = ((user[0], 0.0, 0.0), (0.0, user[1], 0.0))
            detail["um_per_px"] = list(user)
        else:
            raise CalibrationError(
                "x/y are declared in pixels (xy_units = 'px') but no calibration "
                "was given. Set coordinates.um_per_px (CLI --um-per-px) to the "
                "micron-per-pixel value from your instrument's documentation or "
                "run metadata, or pass the vendor transform file "
                "(coordinates.pixel_transform, CLI --pixel-transform). TRACER "
                "does not guess it from coordinate ranges or the platform.")
    elif c.xy_units in _SI_TO_UM and c.xy_units != "um":
        f = _SI_TO_UM[c.xy_units]
        xy_affine = ((f, 0.0, 0.0), (0.0, f, 0.0))

    z_scale = None
    if c.z_units == "plane":
        if c.z_step_um is None:
            raise CalibrationError(
                "z is declared as z-plane indices (z_units = 'plane') but "
                "z_step_um is missing. Set coordinates.z_step_um (CLI "
                "--z-step-um) to the plane spacing in microns from the "
                "acquisition settings. stitch.g_z_um = 'auto' is not a "
                "substitute: it measures spacing, it does not convert units.")
        z_scale = float(c.z_step_um)
        detail["z_step_um"] = z_scale
    elif c.z_units == "px":
        if c.z_um_per_px is None:
            raise CalibrationError(
                "z is declared in pixels (z_units = 'px') but z_um_per_px is "
                "missing. Set coordinates.z_um_per_px (CLI --z-um-per-px).")
        z_scale = float(c.z_um_per_px)
        detail["z_um_per_px"] = z_scale
    elif c.z_units != "um":
        z_scale = _SI_TO_UM[c.z_units]

    return Calibration(xy_units=c.xy_units, z_units=c.z_units,
                       xy_affine=xy_affine, z_scale=z_scale,
                       source=source, detail=detail)


def check_plane_indices(z: np.ndarray) -> None:
    """Plane indices are integers; anything else contradicts the declaration."""
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return
    off = np.abs(z - np.round(z))
    if off.max() > _PLANE_ATOL:
        bad = float(z[np.argmax(off)])
        raise CalibrationError(
            f"z is declared as z-plane indices, but it holds non-integer "
            f"values (e.g. {bad:g}). Plane indices are integers; if z is "
            f"already physical, declare its unit (z_units = 'um') instead of "
            f"converting it again.")


def _extent(v: np.ndarray) -> list[float] | None:
    v = v[np.isfinite(v)]
    return [float(v.min()), float(v.max())] if v.size else None


def to_microns(
    df: pd.DataFrame,
    coords: CoordinatesConfig | None = None,
    *,
    source: str = "config",
    columns: tuple[str, str, str] = ("x", "y", "z"),
    input_stamp: dict | None = None,
    stamped_by: str = "tracer.units.to_microns",
) -> tuple[pd.DataFrame, dict]:
    """Return ``(df_in_microns, receipt)``.

    Converts ``columns`` (x, y and optional z) once, using the calibration
    resolved from ``coords``. Other columns pass through untouched. A table
    that is already stamped as microns (``input_stamp``, or the stamp in
    ``df.attrs``) only accepts the default micron declaration, so it cannot
    be scaled twice.

    The receipt records the input units, the calibration and its source, the
    conversion actually applied, and the resulting extent in microns.
    """
    cal = resolve_calibration(coords, source=source)
    x_col, y_col, z_col = columns
    missing = [c for c in (x_col, y_col) if c not in df.columns]
    if missing:
        raise KeyError(f"coordinate columns missing: {missing}")
    has_z = z_col in df.columns

    stamp = input_stamp if input_stamp is not None else df.attrs.get(METADATA_KEY)
    if stamp and stamp.get("frame") == "um" and not cal.is_identity:
        raise CalibrationError(
            f"these coordinates are already in microns (stamped by "
            f"{stamp.get('stamped_by', 'an earlier conversion')}; calibration "
            f"source {stamp.get('calibration_source', 'unknown')!r}). Declaring "
            f"xy_units = {cal.xy_units!r}, z_units = {cal.z_units!r} would "
            f"convert them a second time. Drop the unit declaration for this "
            f"input.")
    if cal.z_units != "um" and not has_z:
        raise CalibrationError(
            f"z_units = {cal.z_units!r} was declared but the input has no "
            f"{z_col!r} column. For 2D input leave z_units at its default.")

    # Shallow copy: converted columns are new arrays, everything else is
    # shared, and the caller's frame is never modified.
    out = df.copy(deep=False)
    if cal.xy_affine is not None:
        a = cal.xy_affine
        x = df[x_col].to_numpy(dtype=np.float64)
        y = df[y_col].to_numpy(dtype=np.float64)
        out[x_col] = a[0][0] * x + a[0][1] * y + a[0][2]
        out[y_col] = a[1][0] * x + a[1][1] * y + a[1][2]
    if has_z and cal.z_scale is not None:
        z = df[z_col].to_numpy(dtype=np.float64)
        if cal.z_units == "plane":
            check_plane_indices(z)
        out[z_col] = z * cal.z_scale

    applied = {"xy": cal.describe_xy(), "z": cal.describe_z(has_z)}
    receipt = {
        "canonical_frame": "um",
        "input_units": {"xy": cal.xy_units, "z": cal.z_units if has_z else None},
        "z_present": bool(has_z),
        "calibration_source": (stamp.get("calibration_source", "input stamp")
                               if stamp and cal.is_identity else cal.source),
        "calibration": cal.detail,
        "applied": applied,
        "already_canonical": bool(stamp and stamp.get("frame") == "um"),
        "input_stamp": stamp or None,
        "extent_um": {
            "x": _extent(out[x_col].to_numpy(dtype=np.float64)),
            "y": _extent(out[y_col].to_numpy(dtype=np.float64)),
            "z": _extent(out[z_col].to_numpy(dtype=np.float64)) if has_z else None,
        },
    }
    out.attrs[METADATA_KEY] = _json_safe(stamp if (stamp and cal.is_identity) else {
        "frame": "um",
        "input_units": receipt["input_units"],
        "calibration_source": receipt["calibration_source"],
        "applied": applied,
        "stamped_by": stamped_by,
    })
    return out, _json_safe(receipt)


def _json_safe(obj: Any) -> Any:
    """Plain Python types only, so the stamp survives json and parquet."""
    return json.loads(json.dumps(obj, default=float))


# --------------------------------------------------------------------------- #
# CLI plumbing shared by run_tracer.py and preprocess_xenium.py
# --------------------------------------------------------------------------- #
_CLI_FIELDS = {
    "xy_units": "xy_units", "um_per_px": "um_per_px", "z_units": "z_units",
    "z_step_um": "z_step_um", "z_um_per_px": "z_um_per_px",
    "pixel_transform": "pixel_transform",
}


def add_coordinate_args(parser) -> None:
    """Add the coordinate-unit flags. Each overrides the [coordinates] key of
    the same name; unset flags leave the config value alone."""
    from tracer.config import COORD_XY_UNITS, COORD_Z_UNITS

    g = parser.add_argument_group(
        "coordinate units",
        "Declare the units of the INPUT x/y/z; they are converted to microns "
        "once, before any micron threshold applies. Default: already microns. "
        "Calibration is never inferred from coordinate ranges or the platform.")
    g.add_argument("--xy-units", choices=COORD_XY_UNITS, default=None,
                   help="Units of x/y (default from [coordinates], i.e. um).")
    g.add_argument("--um-per-px", type=float, nargs="+", default=None,
                   metavar="UM",
                   help="Microns per xy pixel, from the instrument or run "
                        "metadata: one value, or two for x and y. Requires "
                        "--xy-units px.")
    g.add_argument("--pixel-transform", default=None, metavar="CSV",
                   help="Vendor 3x3 micron->pixel affine, e.g. MERSCOPE "
                        "images/micron_to_mosaic_pixel_transform.csv. "
                        "Requires --xy-units px.")
    g.add_argument("--z-units", choices=COORD_Z_UNITS, default=None,
                   help="Units of z; 'plane' = integer z-plane index.")
    g.add_argument("--z-step-um", type=float, default=None, metavar="UM",
                   help="Microns between z-planes. Requires --z-units plane.")
    g.add_argument("--z-um-per-px", type=float, default=None, metavar="UM",
                   help="Microns per z pixel. Requires --z-units px.")


def coordinates_from_args(args, base: CoordinatesConfig | None = None
                          ) -> tuple[CoordinatesConfig, str]:
    """Apply the CLI flags over ``base``; return ``(config, source)``.

    ``source`` is ``"cli"`` when any flag was given, ``"config"`` when the
    TOML layers declared something other than the default, else ``"default"``.
    """
    import dataclasses

    base = base if base is not None else CoordinatesConfig()
    over = {}
    for attr, fld in _CLI_FIELDS.items():
        v = getattr(args, attr, None)
        if v is None:
            continue
        if attr == "um_per_px":
            if len(v) not in (1, 2):
                raise CalibrationError("--um-per-px takes one value or two (x y)")
            v = v[0] if len(v) == 1 else tuple(v)
        over[fld] = v
    cfg = dataclasses.replace(base, **over) if over else base
    if over:
        return cfg, "cli"
    return cfg, ("default" if cfg == CoordinatesConfig() else "config")


def read_parquet_stamp(path: str | Path) -> dict | None:
    """The coordinate stamp in a parquet file's schema metadata, if any."""
    import pyarrow.parquet as pq

    md = pq.read_schema(str(path)).metadata or {}
    raw = md.get(METADATA_KEY.encode())
    return json.loads(raw) if raw else None


def schema_with_stamp(schema, stamp: dict):
    """Return a pyarrow schema carrying ``stamp`` under ``METADATA_KEY``."""
    md = dict(schema.metadata or {})
    md[METADATA_KEY.encode()] = json.dumps(_json_safe(stamp)).encode()
    return schema.with_metadata(md)
