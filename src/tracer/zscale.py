"""Platform-aware resolution of the z-bin size (``g_z_um``).

Why this module exists
----------------------
TRACER bins transcripts in z by ``bz = floor(z / g_z_um)`` and connects
entities whose z-bins lie within ``±z_neighbor_depth`` (see
``graph.build_grid_graph_xyz`` and ``stitching``). The historical default
``g_z_um = 1.0`` µm was tuned for **Xenium**, whose z is a near-continuous
optical coordinate (~20 µm span, >1e6 distinct values). It silently fails
on platforms that image **discrete z-planes**:

* **MERFISH** (mouse ileum ROI): 9 planes, spacing **1.5 µm**. With
  ``g_z_um = 1.0`` and ``z_neighbor_depth = 1``, adjacent planes land up to
  *two* bins apart (``floor(2.5/1) = 2`` vs ``floor(4.0/1) = 4``), so they
  never connect — every z-plane is an island and within-cell z structure is
  destroyed.
* **CosMx** (NSCLC ROI): 11 planes, spacing **0.8 µm**. ``g_z_um = 1.0`` is
  misaligned with the 0.8 µm grid: some adjacent planes collapse into the
  same bin while others split, giving uneven z-connectivity.

The fix is to make the z-bin size match the platform's z-plane spacing so
consecutive planes map to consecutive bins (1 bin apart) and therefore
connect at ``z_neighbor_depth = 1``. This module resolves the requested
``g_z_um`` (explicit float, ``"auto"``, or ``None``/legacy) against the
**observed** z distribution and returns the value to use plus a
human-readable reason and any warnings.

Heuristic (documented + explicit)
---------------------------------
1. **Degenerate z** (absent / all-NaN / a single distinct value): run in
   2D — return ``z_neighbor_depth_override = 0`` and warn. (With constant z
   every tx already shares one z-bin, so this is graceful, not fatal.)
2. **Explicit positive float**: honor it. If z is discrete *and* the value
   is **smaller than the median plane spacing**, warn that adjacent planes
   may not connect and recommend ``"auto"`` or a platform preset.
3. ``"auto"``:
   * **Discrete planes** (``n_unique <= max_discrete_planes``): set
     ``g_z_um = median positive plane spacing``. Adjacent planes then sit
     exactly one bin apart.
   * **Continuous z**: keep ``continuous_default_um`` (1.0 µm), the Xenium
     behavior — no change for existing Xenium/Atera runs.
4. ``None``: legacy — defer to the caller's within-cell ``auto_Gz``
   estimator (``stitching.estimate_within_cell_dz_threshold``). Preserves
   the pre-existing ``g_z_um = null`` semantics bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ZScaleResolution", "resolve_g_z_um", "z_plane_stats"]


@dataclass(frozen=True)
class ZScaleResolution:
    """Outcome of resolving a requested ``g_z_um`` against observed z.

    Attributes
    ----------
    g_z_um
        The z-bin size (µm) the pipeline should use, OR ``None`` to signal
        "fall back to the legacy within-cell ``auto_Gz`` estimator" (only
        emitted for the ``requested is None`` legacy path).
    z_neighbor_depth_override
        ``None`` to keep the configured depth, or ``0`` to force 2D
        behavior (degenerate z).
    mode
        One of ``"explicit"``, ``"auto-discrete"``, ``"auto-continuous"``,
        ``"legacy-auto"``, ``"degenerate-2d"``.
    reason
        Human-readable one-liner describing how ``g_z_um`` was chosen.
    n_planes
        Count of distinct finite z values observed.
    median_spacing
        Median positive spacing between consecutive distinct z values
        (``nan`` when fewer than 2 planes).
    warnings
        Tuple of warning strings (empty when none).
    """

    g_z_um: float | None
    z_neighbor_depth_override: int | None
    mode: str
    reason: str
    n_planes: int
    median_spacing: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def z_plane_stats(z_values) -> tuple[int, float]:
    """Return ``(n_distinct_finite, median_positive_spacing)``.

    ``median_positive_spacing`` is ``nan`` when fewer than two distinct
    finite z values exist. Distinct values are de-duplicated with a small
    tolerance so floating-point jitter on nominally-identical planes
    doesn't inflate the plane count.
    """
    if z_values is None:
        return 0, float("nan")
    z = np.asarray(z_values, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0, float("nan")
    uz = np.unique(z)
    if uz.size >= 2:
        # Collapse near-duplicate planes (jitter below 1e-6 of the range).
        tol = max(1e-9, (uz[-1] - uz[0]) * 1e-6)
        keep = np.concatenate(([True], np.diff(uz) > tol))
        uz = uz[keep]
    n_unique = int(uz.size)
    if n_unique < 2:
        return n_unique, float("nan")
    diffs = np.diff(uz)
    diffs = diffs[diffs > 0]
    median_spacing = float(np.median(diffs)) if diffs.size else float("nan")
    return n_unique, median_spacing


def resolve_g_z_um(
    z_values,
    requested: float | str | None,
    *,
    max_discrete_planes: int = 64,
    continuous_default_um: float = 1.0,
    z_neighbor_depth: int = 1,
) -> ZScaleResolution:
    """Resolve a requested ``g_z_um`` against the observed z distribution.

    Parameters
    ----------
    z_values
        Array / Series of transcript z coordinates (µm), or ``None``.
    requested
        ``cfg.stitch.g_z_um``: a positive float (explicit), the string
        ``"auto"`` (adaptive), or ``None`` (legacy within-cell estimator).
    max_discrete_planes
        z with at most this many distinct values is treated as
        discrete-plane data; above it, continuous.
    continuous_default_um
        ``g_z_um`` used for continuous z under ``"auto"`` (Xenium default).
    z_neighbor_depth
        The configured z-neighbor depth — used only to phrase the
        connectivity warning for explicit-float-vs-spacing mismatches.

    Returns
    -------
    ZScaleResolution
    """
    n_planes, median_spacing = z_plane_stats(z_values)

    # 1. Degenerate z → 2D.
    if n_planes <= 1:
        return ZScaleResolution(
            g_z_um=None,
            z_neighbor_depth_override=0,
            mode="degenerate-2d",
            reason=(
                "z is missing or single-valued — running in 2D "
                "(z_neighbor_depth forced to 0; g_z_um ignored)."
            ),
            n_planes=n_planes,
            median_spacing=median_spacing,
            warnings=(
                "No usable z variation found; z-aware stages disabled "
                "(2D mode). If this is 3D data, check the z column/units.",
            ),
        )

    is_discrete = n_planes <= int(max_discrete_planes)

    # 2. Explicit positive float.
    if requested is not None and not isinstance(requested, str):
        g = float(requested)
        warns: list[str] = []
        # Warn only when g is *meaningfully* below the plane spacing (>5%),
        # so a preset that equals the spacing doesn't trip its own warning
        # on float32-jittered coordinates (e.g. CosMx 0.8 vs 0.80000001).
        if (is_discrete and np.isfinite(median_spacing)
                and g < median_spacing * 0.95):
            warns.append(
                f"g_z_um={g:g} µm is smaller than the observed z-plane "
                f"spacing ({median_spacing:g} µm) across {n_planes} planes; "
                f"adjacent z-planes will sit >1 bin apart and may not "
                f"connect at z_neighbor_depth={z_neighbor_depth}. Consider "
                f"g_z_um=\"auto\" or a platform preset (cosmx/merfish)."
            )
        return ZScaleResolution(
            g_z_um=g,
            z_neighbor_depth_override=None,
            mode="explicit",
            reason=(
                f"g_z_um={g:g} µm (explicit config/CLI; observed "
                f"{n_planes} z-planes, median spacing "
                f"{median_spacing:g} µm)."
            ),
            n_planes=n_planes,
            median_spacing=median_spacing,
            warnings=tuple(warns),
        )

    # 3. "auto".
    if isinstance(requested, str):
        if requested.strip().lower() != "auto":
            raise ValueError(
                f"g_z_um string must be 'auto'; got {requested!r}"
            )
        if is_discrete and np.isfinite(median_spacing) and median_spacing > 0:
            g = round(median_spacing, 4)
            return ZScaleResolution(
                g_z_um=g,
                z_neighbor_depth_override=None,
                mode="auto-discrete",
                reason=(
                    f"g_z_um={g:g} µm (auto: {n_planes} discrete z-planes, "
                    f"median spacing {median_spacing:g} µm → adjacent planes "
                    f"1 bin apart)."
                ),
                n_planes=n_planes,
                median_spacing=median_spacing,
            )
        return ZScaleResolution(
            g_z_um=float(continuous_default_um),
            z_neighbor_depth_override=None,
            mode="auto-continuous",
            reason=(
                f"g_z_um={float(continuous_default_um):g} µm (auto: "
                f"continuous z, {n_planes} distinct values → "
                f"Xenium-style default)."
            ),
            n_planes=n_planes,
            median_spacing=median_spacing,
        )

    # 4. None → legacy within-cell estimator (caller substitutes auto_Gz).
    return ZScaleResolution(
        g_z_um=None,
        z_neighbor_depth_override=None,
        mode="legacy-auto",
        reason=(
            "g_z_um=null — using the legacy within-cell Δz estimator "
            "(auto_Gz)."
        ),
        n_planes=n_planes,
        median_spacing=median_spacing,
    )
