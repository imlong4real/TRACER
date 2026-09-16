"""Entity-type categorical column — canonical kind classification.

Replaces label-string parsing (see `stitching.infer_entity_type`) as the
canonical mechanism for asking "what kind of entity is this row".

The `_etype` column is populated by every stage that emits or transforms
entities; readers consume it directly via `infer_entity_type_etype` and
related sibling helpers, without parsing the label.

Categories (string-valued for readability; backed by uint8 codes):
  - ``cell``       — main entity for an input cell_id (or a cascade main).
  - ``partial``    — sub-seed emitted by Phase 1c, or a cascade partial.
  - ``component``  — UNASSIGNED_<n> (legacy spatial-CC Group fallback) or
                     similar pseudo-cells.
  - ``drop``       — explicitly demoted entity. Reserved; not produced
                     by any stage today but kept for symmetry.
  - ``unknown``    — unassigned tx or unrecognized; sentinel values
                     like "-1", "DROP", "UNASSIGNED", "nan", "*_rejected".

Memory: 5 categories → uint8 codes; 20.7M tx × 1 byte ≈ 20 MB. Negligible.

See `docs/superpowers/specs/2026-05-11-etype-column-design.md` for the
full migration plan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ETYPE_CATEGORIES: list[str] = ["cell", "partial", "component", "drop", "unknown"]

ETYPE_DTYPE: pd.CategoricalDtype = pd.CategoricalDtype(
    categories=ETYPE_CATEGORIES, ordered=False
)


# ---------------------------------------------------------------------------
# Entity-hierarchy delimiter
#
# All TRACER-produced partial / sub-partial labels use the unique
# `-tr-` delimiter between the input cell_id and the partial-index
# tree. Mains keep their bare cell_id label.
#
#   main          : `{cell_id}`                       e.g. "42" or "adohnpem-1"
#   partial       : `{cell_id}{ENTITY_DELIMITER}{k}`  e.g. "42-tr-1"
#   sub-partial   : `{cell_id}{ENTITY_DELIMITER}{k}{ENTITY_DELIMITER}{j}`
#                                                       e.g. "42-tr-1-tr-1"
#
# This sidesteps the ambiguity on Xenium FFPE / IO data where the
# input cell_id natively contains dashes (`adohnpem-1`). Splitting on
# `-tr-` yields `[cell_id, k, j]` uniquely regardless of cell_id
# content.
#
# When changing this constant, **regenerate all reference partitions**
# (`tests/references/*.json`) and audit any external tooling that
# pattern-matches on the legacy bare `-` delimiter.
# ---------------------------------------------------------------------------
ENTITY_DELIMITER: str = "-tr-"


def make_partial_label(cell_id: str, idx: int) -> str:
    """Construct a depth-1 partial label: `{cell_id}-tr-{idx}`."""
    return f"{cell_id}{ENTITY_DELIMITER}{idx}"


def make_subpartial_label(parent_partial: str, idx: int) -> str:
    """Construct a sub-partial label under an existing partial:
    `{parent_partial}-tr-{idx}`. `parent_partial` is expected to be a
    depth-1 partial label."""
    return f"{parent_partial}{ENTITY_DELIMITER}{idx}"


def split_entity_label(label: str) -> tuple[str, list[int]]:
    """Decompose a TRACER entity label into (cell_id, depth_indices).

    - main:        ("42",          [])
    - partial:     ("42",          [1])
    - sub-partial: ("42",          [1, 1])
    - PDAC main:   ("adohnpem-1",  [])
    - PDAC partial:("adohnpem-1",  [1])

    Raises ValueError if a suffix piece isn't a non-negative integer.
    Returns (label, []) for any label that doesn't contain
    `ENTITY_DELIMITER` (treated as a main).
    """
    if ENTITY_DELIMITER not in label:
        return label, []
    parts = label.split(ENTITY_DELIMITER)
    cell_id = parts[0]
    indices = [int(p) for p in parts[1:]]
    return cell_id, indices


def empty_etype(n: int) -> pd.Categorical:
    """Build an all-`unknown` etype column of length ``n``."""
    return pd.Categorical(["unknown"] * n, dtype=ETYPE_DTYPE)


def etype_from_codes(codes: np.ndarray) -> pd.Categorical:
    """Map Cython per-tx codes from ``prune_cells_nuclear_seed`` to etypes.

    Codes returned by the kernel:
      0 = main             → ``cell``
      1 = partial          → ``partial``
      2 = unassigned       → ``unknown``
      3 = fallback-needed  → ``unknown`` (caller handles fallback path)
    """
    cat_codes = np.full(
        codes.shape, ETYPE_CATEGORIES.index("unknown"), dtype=np.int8
    )
    cat_codes[codes == 0] = ETYPE_CATEGORIES.index("cell")
    cat_codes[codes == 1] = ETYPE_CATEGORIES.index("partial")
    return pd.Categorical.from_codes(cat_codes, dtype=ETYPE_DTYPE)


def infer_etype_from_label(labels) -> pd.Categorical:
    """Parity helper: classify a label series via the same rules as
    `stitching.infer_entity_type`. Used during migration to verify
    stage emitters produce a column consistent with legacy parsing
    *on integer cell_ids*.

    On dash-containing cell_ids (Xenium FFPE / IO), the legacy rule
    misclassifies mains as partials — this helper preserves that
    behavior intentionally so it can be used as a regression baseline.
    The bug is fixed in production by stage emitters that write the
    correct `_etype` directly from kernel codes / stage semantics,
    not by changing the parsing rule here.

    Categories returned:
      - sentinels (``-1``, ``DROP``, ``UNASSIGNED``, ``nan``,
        ``*_rejected``) → ``unknown``
      - starts with ``UNASSIGNED_``                       → ``component``
      - contains ``-``                                    → ``partial``
      - else                                              → ``cell``

    NOTE: this still uses the legacy bare-dash rule for parity with
    existing code. The `-tr-` delimiter defined in this module is the
    target convention; full migration happens in a follow-up commit
    that updates every emitter + parser + regenerates reference
    partitions in lockstep.
    """
    s = pd.Series(labels).astype(str).reset_index(drop=True)
    out = np.full(len(s), "unknown", dtype=object)

    is_sentinel = s.isin({"-1", "DROP", "UNASSIGNED", "nan"}) | s.str.endswith(
        "_rejected"
    )
    is_component = ~is_sentinel & s.str.startswith("UNASSIGNED_")
    is_partial = (
        ~is_sentinel & ~is_component & s.str.contains("-", regex=False)
    )
    is_cell = ~is_sentinel & ~is_component & ~is_partial

    out[is_sentinel.to_numpy()] = "unknown"
    out[is_component.to_numpy()] = "component"
    out[is_partial.to_numpy()] = "partial"
    out[is_cell.to_numpy()] = "cell"

    return pd.Categorical(out, dtype=ETYPE_DTYPE)


def infer_entity_type_etype(
    df: pd.DataFrame, type_col: str = "_etype"
) -> pd.Series:
    """Sibling reader: return entity kind from the ``_etype`` column.

    Drop-in for the label-parsing ``stitching.infer_entity_type`` at
    call sites that have access to the DataFrame. Returns a string
    Series with the same vocabulary as the legacy helper.
    """
    return df[type_col].astype(str)


# ─────────────────────────────────────────────────────────────────────
# Post-merge etype homogenization
#
# Any pipeline stage that merges entities (Phase1-Maha-Remerge, Stitch,
# anything else that calls a DSU union and remaps `entity_col`) must
# leave the resulting entity with a single `_etype` value across all
# its rows. Without this invariant, `df.groupby(...)["_etype"].first()`
# in `build_entity_table` and downstream is non-deterministic, which
# can silently misclassify a merged entity and break the cell-cell
# merge gate (`stitching.can_merge`).
#
# Convention: when multiple etypes are present, the highest-priority
# one wins. The order reflects the value of the assignment as
# evidence for the entity's identity — a real cell beats a partial
# sub-program; a component beats a partial; demoted/unknown trails
# everything.
#
# Use `homogenize_etype_for_entity` per affected root after each
# union — O(merges) work, not O(N entities).
# ─────────────────────────────────────────────────────────────────────
_ETYPE_PRIORITY = {
    "cell": 0,       # strongest evidence — full nuclear-anchored cell
    "partial": 1,    # sub-partition of a real cell, also nuclear-anchored
    "component": 2,  # UNASSIGNED_* group — clustered residual tx
    "drop": 3,       # explicitly demoted
    "unknown": 4,    # unclassified / sentinel
}


def homogenize_etype_for_entities(
    df: pd.DataFrame,
    entity_labels,
    *,
    entity_col: str = "tracer_id",
    etype_col: str = "_etype",
) -> None:
    """In-place batch homogenization: give every entity in
    ``entity_labels`` a single ``_etype`` (the highest-priority etype
    present within that entity).

    Cost: **one** O(N) scan of ``entity_col`` to locate the affected
    rows, then the priority resolution runs over **only** those rows
    (O(rows-in-affected)). This replaces the per-entity pattern that
    re-scanned the whole column once *per entity* — so it is the
    preferred form at merge call sites that touch many roots.

    No-op when ``etype_col`` is absent, ``entity_labels`` is empty, or no
    row matches. Idempotent: passing already-homogeneous entities (or
    re-running) leaves the column unchanged.
    """
    if etype_col not in df.columns:
        return
    labels = {str(x) for x in entity_labels}
    if not labels:
        return
    ent = df[entity_col].astype(str)
    mask = ent.isin(labels).to_numpy()
    if not mask.any():
        return
    # Mask first, then string-cast only the affected rows (avoids casting
    # the full _etype column when few entities are touched).
    sub_ent = ent.to_numpy()[mask]
    sub_et = df[etype_col].to_numpy()[mask].astype(str)
    prio = np.fromiter(
        (_ETYPE_PRIORITY.get(e, 99) for e in sub_et),
        dtype=np.int16, count=sub_et.shape[0],
    )
    # Winner per entity = the etype carried by its highest-priority
    # (lowest-rank) row. Stable sort → deterministic; groupby-first picks
    # the winner; reindex broadcasts it back to every affected row.
    tmp = pd.DataFrame({"_ent": sub_ent, "_prio": prio, "_et": sub_et})
    winners = (
        tmp.sort_values("_prio", kind="stable")
        .groupby("_ent", sort=False)["_et"]
        .first()
    )
    df.loc[mask, etype_col] = winners.reindex(sub_ent).to_numpy()


def homogenize_etype_for_entity(
    df: pd.DataFrame,
    entity_label: str,
    *,
    entity_col: str = "tracer_id",
    etype_col: str = "_etype",
) -> None:
    """In-place: ensure every row of ``entity_label`` shares a single
    ``_etype`` value (the highest-priority etype present in the group).

    No-op when ``etype_col`` is absent, no row matches ``entity_label``,
    or the entity is already homogeneous. Thin single-entity wrapper over
    :func:`homogenize_etype_for_entities`; idempotent.
    """
    homogenize_etype_for_entities(
        df, (entity_label,), entity_col=entity_col, etype_col=etype_col,
    )


def set_entity_etype(
    df: pd.DataFrame,
    entity_label: str,
    etype: str,
    *,
    entity_col: str = "tracer_id",
    etype_col: str = "_etype",
) -> None:
    """In-place: set ``_etype`` to ``etype`` for EVERY tx of ``entity_label``.

    The all-tx counterpart to a per-row etype write: whenever an entity's
    type changes (promotion, demotion, relabel), every one of its tx must
    carry the new etype so the entity never goes mixed. No-op when the etype
    column is absent or no row matches.
    """
    if etype_col not in df.columns:
        return
    mask = (df[entity_col].astype(str) == str(entity_label)).to_numpy()
    if mask.any():
        df.loc[mask, etype_col] = etype


def swap_entity_etypes(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
    *,
    entity_col: str = "tracer_id",
    etype_col: str = "_etype",
) -> None:
    """In-place: swap the ``_etype`` of two entities — every tx of ``label_a``
    takes ``label_b``'s etype and vice versa (all-tx, so neither side is left
    mixed). Each side's etype is its dominant (most common) value, ties broken
    by priority. No-op if either entity is absent or the etype column missing.
    """
    if etype_col not in df.columns:
        return
    ent = df[entity_col].astype(str)
    a = (ent == str(label_a)).to_numpy()
    b = (ent == str(label_b)).to_numpy()
    if not a.any() or not b.any():
        return
    et = df[etype_col].astype(str).to_numpy()

    def _dom(vals):
        c = pd.Series(vals).value_counts()
        top = c[c == c.max()].index
        return min(top, key=lambda e: _ETYPE_PRIORITY.get(e, 99))

    dom_a, dom_b = _dom(et[a]), _dom(et[b])
    df.loc[a, etype_col] = dom_b
    df.loc[b, etype_col] = dom_a


# Label sentinels that are never a real entity (mirror of spatial.UNASSIGNED_LABELS
# kept local so this module has no upward import). A member carrying one of these
# labels can never define a target entity's type.
_ETYPE_SENTINELS: frozenset = frozenset(
    {"-1", "DROP", "UNASSIGNED", "nan", "__GUARD_SKIP__",
     "group_rejected", "demote_rejected", "None", ""}
)


def _dominant_etype(values) -> str:
    """Most-common etype among ``values``; ties broken by ``_ETYPE_PRIORITY``
    (cell < partial < component < drop < unknown)."""
    c = pd.Series(values).value_counts()
    top = c[c == c.max()].index
    return min(top, key=lambda e: _ETYPE_PRIORITY.get(e, 99))


def propagate_etype_to_moved(
    df: pd.DataFrame,
    moved_positions,
    target_labels,
    *,
    entity_col: str = "tracer_id",
    etype_col: str = "_etype",
) -> None:
    """Set ``_etype`` for rows just relabeled into a target entity (by a rescue
    move) so the target stays HOMOGENEOUS.

    The target's etype is the dominant non-``unknown``, non-sentinel etype among
    its **pre-existing** members — rows NOT in ``moved_positions``. Moved rows
    then adopt that etype. Because moved rows are excluded when deriving the
    type, a stale/stray etype riding in on a moved row can neither define the
    target nor survive the write, so a single contaminant can never flip (or
    mix) a target's type.

    Parameters
    ----------
    df : DataFrame mutated in place.
    moved_positions : positional (iloc) row indices that were just relabeled.
    target_labels : the target entity label for each moved row (same length /
        order as ``moved_positions``).

    Moved rows whose target has no pre-existing typed member are left unchanged
    (no fabricated etype). No-op if the etype column is absent or nothing moved.
    """
    if etype_col not in df.columns:
        return
    moved_positions = np.asarray(moved_positions)
    if moved_positions.size == 0:
        return
    target_labels = np.asarray(target_labels).astype(str)

    n = len(df)
    is_moved = np.zeros(n, dtype=bool)
    is_moved[moved_positions] = True

    lab = df[entity_col].astype(str).to_numpy()
    et = df[etype_col].astype(str).to_numpy()
    # Pre-existing, typed, non-sentinel members define each target's type.
    elig = (~is_moved) & (et != "unknown") & (~np.isin(lab, list(_ETYPE_SENTINELS)))
    if not elig.any():
        return
    dom = (
        pd.DataFrame({"lab": lab[elig], "et": et[elig]})
        .groupby("lab")["et"].agg(_dominant_etype).to_dict()
    )
    new = np.array([dom.get(t) for t in target_labels], dtype=object)
    ok = np.array([v is not None for v in new])
    if ok.any():
        rows = moved_positions[ok]
        df.iloc[rows, df.columns.get_loc(etype_col)] = new[ok].astype(str)
