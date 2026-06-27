"""Derive a TRACER seg-mode ``overlaps_nucleus`` label from VisiumHD-style
nucleus segmentation polygons.

VisiumHD ships only a square-bin expression matrix (no per-molecule cell
assignment), but 10x also exports nucleus polygons
(``*_nucleus_segmentations.geojson``) in **full-resolution pixel** space —
the same frame as ``tissue_positions``' ``pxl_row/col_in_fullres``. This
module overlays bin centers onto those polygons so each bin gets:

* ``cell_id``           — the nucleus id its center falls in, or ``"-1"``;
* ``overlaps_nucleus``  — 1 if it falls in a nucleus, else 0.

The nucleus footprint then acts as a TRACER **seed** (analogous to
``overlaps_nucleus`` in imaging data): ``run_segmented_pipeline`` anchors
each cell on its nucleus bins and Rescue grows the cytoplasm from the
surrounding ``"-1"`` bins. Unassigned bins are preserved so residual /
partial profiles can still be reconstructed.

Pure-geometry + numpy/shapely so it is unit-testable with synthetic
polygons (no scanpy / VisiumHD IO needed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "NucleusPolygons",
    "load_nucleus_polygons",
    "assign_bins_to_nuclei",
]

# Sentinel used everywhere in TRACER for "no assignment".
UNASSIGNED = "-1"


@dataclass
class NucleusPolygons:
    """Loaded nucleus polygons in full-res pixel space."""
    geoms: list           # list[shapely Polygon]
    cell_ids: np.ndarray  # object array of str ids, aligned with geoms
    centroids: np.ndarray  # (N, 2) float — polygon centroids (px)


def load_nucleus_polygons(
    geojson_path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    id_field: str = "cell_id",
) -> NucleusPolygons:
    """Load nucleus polygons from a 10x segmentation GeoJSON.

    Parameters
    ----------
    geojson_path
        Path to ``*_nucleus_segmentations.geojson`` (FeatureCollection of
        Polygons with ``properties[id_field]``).
    bbox
        Optional ``(xmin, ymin, xmax, ymax)`` in pixel space; polygons whose
        centroid lies outside are skipped (keeps the overlay cheap for ROIs).
    id_field
        Property holding the nucleus id (10x uses ``"cell_id"``).
    """
    from shapely.geometry import shape

    with open(geojson_path) as f:
        gj = json.load(f)
    feats = gj["features"] if isinstance(gj, dict) else gj

    geoms: list = []
    ids: list[str] = []
    cents: list[tuple[float, float]] = []
    for feat in feats:
        geom = shape(feat["geometry"])
        c = geom.centroid
        if bbox is not None:
            if not (bbox[0] <= c.x <= bbox[2] and bbox[1] <= c.y <= bbox[3]):
                continue
        cid = feat.get("properties", {}).get(id_field, feat.get("id"))
        geoms.append(geom)
        ids.append(str(cid))
        cents.append((c.x, c.y))
    return NucleusPolygons(
        geoms=geoms,
        cell_ids=np.asarray(ids, dtype=object),
        centroids=np.asarray(cents, dtype=np.float64).reshape(-1, 2),
    )


def assign_bins_to_nuclei(
    bin_x: np.ndarray,
    bin_y: np.ndarray,
    nuclei: NucleusPolygons,
    *,
    multi_rule: str = "nearest_centroid",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Assign each bin center to a nucleus by point-in-polygon overlay.

    Parameters
    ----------
    bin_x, bin_y
        Bin-center coordinates **in the polygon frame** (full-res pixels).
    nuclei
        Loaded polygons (see :func:`load_nucleus_polygons`).
    multi_rule
        Deterministic tie-break when a bin center falls inside more than one
        polygon (rare — 10x nuclei seldom overlap):

        * ``"nearest_centroid"`` (default): the polygon whose centroid is
          closest to the bin center; ties broken by smallest cell_id.
        * ``"smallest_id"``: the lexicographically smallest cell_id.

    Returns
    -------
    cell_id : object ndarray of str
        Nucleus id per bin, or ``"-1"`` when the bin center is in no nucleus.
    overlaps_nucleus : uint8 ndarray
        1 where assigned, 0 otherwise.
    stats : dict
        ``n_bins, n_assigned, n_unassigned, frac_assigned, n_ambiguous,
        ambiguity_rate, n_nuclei``.
    """
    from shapely import STRtree, points as _points

    n = int(bin_x.shape[0])
    cell_id = np.full(n, UNASSIGNED, dtype=object)
    overlaps = np.zeros(n, dtype=np.uint8)

    if n == 0 or len(nuclei.geoms) == 0:
        stats = dict(n_bins=n, n_assigned=0, n_unassigned=n,
                     frac_assigned=0.0, n_ambiguous=0, ambiguity_rate=0.0,
                     n_nuclei=len(nuclei.geoms))
        return cell_id, overlaps, stats

    pts = _points(np.column_stack([bin_x, bin_y]))
    tree = STRtree(nuclei.geoms)
    # For each bin point, which polygons it falls in. shapely 2.x returns a
    # (2, M) array of [input_point_idx, tree_geom_idx] pairs. Use
    # "intersects" (point-in/on-polygon): predicate is tested as
    # input.predicate(tree), so "contains" (point.contains(polygon)) is
    # always False here — "intersects" is the point-in-polygon test.
    pairs = tree.query(pts, predicate="intersects")
    pt_idx = np.asarray(pairs[0])
    poly_idx = np.asarray(pairs[1])

    if pt_idx.size == 0:  # no bin center falls in any nucleus
        stats = dict(n_bins=n, n_assigned=0, n_unassigned=n,
                     frac_assigned=0.0, n_ambiguous=0, ambiguity_rate=0.0,
                     n_nuclei=len(nuclei.geoms))
        return cell_id, overlaps, stats

    # Group candidate polygons per bin point.
    n_ambiguous = 0
    order = np.argsort(pt_idx, kind="stable")
    pt_idx_s = pt_idx[order]
    poly_idx_s = poly_idx[order]
    boundaries = np.r_[0, np.where(np.diff(pt_idx_s) != 0)[0] + 1, pt_idx_s.size]
    for b in range(len(boundaries) - 1):
        s, e = boundaries[b], boundaries[b + 1]
        bi = int(pt_idx_s[s])
        cands = poly_idx_s[s:e]
        if cands.size == 1:
            chosen = int(cands[0])
        else:
            n_ambiguous += 1
            chosen = _resolve_multi(
                bi, cands, bin_x, bin_y, nuclei, multi_rule)
        cell_id[bi] = nuclei.cell_ids[chosen]
        overlaps[bi] = 1

    n_assigned = int(overlaps.sum())
    stats = dict(
        n_bins=n,
        n_assigned=n_assigned,
        n_unassigned=n - n_assigned,
        frac_assigned=float(n_assigned / n) if n else 0.0,
        n_ambiguous=int(n_ambiguous),
        ambiguity_rate=float(n_ambiguous / n_assigned) if n_assigned else 0.0,
        n_nuclei=int(len(nuclei.geoms)),
    )
    return cell_id, overlaps, stats


def _resolve_multi(bi, cands, bin_x, bin_y, nuclei, rule) -> int:
    """Deterministically pick one polygon among several containing a bin."""
    if rule == "smallest_id":
        return int(cands[np.argmin([nuclei.cell_ids[c] for c in cands])])
    if rule == "nearest_centroid":
        cx, cy = bin_x[bi], bin_y[bi]
        cents = nuclei.centroids[cands]
        d2 = (cents[:, 0] - cx) ** 2 + (cents[:, 1] - cy) ** 2
        best = np.where(d2 == d2.min())[0]
        if best.size > 1:  # tie → smallest id, fully deterministic
            ids = np.asarray([nuclei.cell_ids[cands[k]] for k in best])
            return int(cands[best[np.argmin(ids)]])
        return int(cands[best[0]])
    raise ValueError(f"unknown multi_rule {rule!r}")
