"""Wrap the production segmented + noseg TRACER pipelines into thin
runners suitable for tests.

Both runners return ``(df_final, stage_progression)`` where
``stage_progression`` is a list of dicts ``{stage, n_cells, n_partials,
n_components, n_unassigned_tx}`` recording the pipeline state after
each stage. The final state is the last entry.

Mirrors the configuration used by the segmented_workflow.ipynb and
noseg_workflow.ipynb notebooks: PMI metric, mean-PMI rescue veto,
grid_3d Stage 4, same-bin Stage 2 at G=8, post-S5 G=2.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from tracer.graph import build_grid_graph_xy, build_grid_graph_xyz
from tracer.zscale import resolve_g_z_um
from tracer.pruning import prune_transcripts_fast, prune_genes_by_pmi_greedy
from tracer.spatial import (
    annotate_unassigned_components_fast,
    enforce_spatial_coherence_fast,
    guarded_rescue,
    reassign_unassigned_grid_pool,
    demote_small_entities,
    finalize_unassigned,
)
from tracer.pruning import prune_transcripts_nuclear_seed
from tracer.stitching import (
    apply_stitching_to_transcripts_memory_efficient,
    coherence,
    estimate_within_cell_dz_threshold,
)
from tracer.density_cascade import cascade_as_residual_handler

_log = logging.getLogger("tracer.pipeline")


def _resolve_stitch_g_z(df: pd.DataFrame, cfg, auto_Gz: float,
                        coord_cols=("x", "y", "z")) -> tuple[float, int]:
    """Resolve the Stitch z-bin (``g_z_um``) and z-neighbor depth from the
    config request against the observed z distribution.

    Centralizes the platform-aware z-scaling for both the segmented and
    no-seg runners. Returns ``(g_z_um, z_neighbor_depth)`` ready to hand to
    ``apply_stitching_to_transcripts_memory_efficient``. The resolved value
    and the reason it was chosen are logged; any warnings (e.g. an explicit
    g_z_um smaller than the platform's z-plane spacing) are surfaced at
    WARNING level so misconfigured discrete-plane runs are not silent.
    """
    z_col = coord_cols[2] if len(coord_cols) >= 3 else None
    z_vals = df[z_col].to_numpy() if (z_col and z_col in df.columns) else None
    res = resolve_g_z_um(
        z_vals, cfg.stitch.g_z_um,
        z_neighbor_depth=cfg.stitch.z_neighbor_depth,
    )
    _log.info("z-scale resolution: %s", res.reason)
    for w in res.warnings:
        _log.warning("z-scale: %s", w)
    # None g_z_um means "use the legacy within-cell estimator" (auto_Gz).
    g_z = res.g_z_um if res.g_z_um is not None else float(auto_Gz)
    z_depth = (
        res.z_neighbor_depth_override
        if res.z_neighbor_depth_override is not None
        else cfg.stitch.z_neighbor_depth
    )
    return float(g_z), int(z_depth)


# Modern config — matches segmented_workflow.ipynb / noseg_workflow.ipynb.
# PMI_THR relaxed to 1e-5 ("essentially zero positive PMI") on the
# strength of the cell-37742 EMT analysis: log(1.5) ≈ 0.405 sat ABOVE
# the in-cell max NPMI for that cell, blowing up its prune. With NaN→0
# fill in nuclear-seed Prune, threshold ≈ 0 admits any non-negative
# evidence to the seed, which gave +29 % ARI(vs Xenium cell_id) on the
# 50×50 µm validation crop (0.442 → 0.573).
#
# 2026-05-13: PMI_THR raised 0.05 → 0.2 (= 1.22× chance, real enrichment
# cutoff in natural-log PMI space). Validated on PDAC + lung full-tissue:
# small retention cost (−0.3 pp) for major coherence gains (cell C
# mean 0.80→0.93, p10 0.57→0.82). RESCUE_NEG_THR / ANNOTATE_NEG_THR
# scale with this. See benchmarks/pdac_pmi_sweep/ and
# benchmarks/pdac_full_seq{,_thr0}_strict/ for the validation suite.
PMI_THR = 0.2
SEED_COHERENCE_FLOOR = 0.10
TX_WEIGHTED_PRUNE = True   # tx-weighted greedy bad-edge prune (1a/1c)
SPLIT_PHASE1_DZ = 2.0      # µm; if consecutive z-sorted tx gap > this,
                            #     split entity at that point.
SPLIT_PHASE1_MIN_TX = 1     # min tx per sub-group during the split itself
                            #     (1 = keep singletons; QC pass below
                            #     handles the actual demotion).
SPLIT_PHASE1_MIN_ENTITY = 2 # minimum entity size to consider (2 = bare
                            #     minimum to compute a diff).
PHASE1_QC_MIN_TX = 3        # post-split QC: any Phase 1 entity (main or
                            #     partial) with < this tx → unassigned.
                            #     Prevents 1- and 2-tx degenerate seeds
                            #     from anchoring Rescue admissions.


def _qc_demote_small_phase1_entities(df_in: pd.DataFrame, *,
                                       entity_col: str,
                                       min_size: int = PHASE1_QC_MIN_TX,
                                       unassigned_id: str = "-1"
                                       ) -> tuple[pd.DataFrame, dict]:
    """Demote any Phase 1 entity (main `{cell}` or partial) with < min_size tx
    to unassigned. Skips already-unassigned labels (`-1`, `UNASSIGNED`,
    `UNASSIGNED_*`) — these are not seeded entities."""
    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    counts = df_out[entity_col].value_counts()
    bad = []
    for ent, n in counts.items():
        if n >= min_size:
            continue
        if ent == unassigned_id or ent == "UNASSIGNED" or ent.startswith("UNASSIGNED_"):
            continue
        bad.append(ent)
    n_demoted = 0
    if bad:
        mask = df_out[entity_col].isin(bad)
        df_out.loc[mask, entity_col] = unassigned_id
        n_demoted = int(mask.sum())
        # Mirror the demotion in _etype if the column exists.
        if "_etype" in df_out.columns:
            df_out.loc[mask, "_etype"] = "unknown"
    return df_out, {
        "entities_demoted": len(bad),
        "tx_demoted": n_demoted,
    }


def _phase1_rerank_within_parent_etype(df_in: pd.DataFrame, *,
                                         entity_col: str,
                                         cell_id_col: str = "cell_id",
                                         nuclear_col: str = "overlaps_nucleus",
                                         margin_tx: int = 1,
                                         ) -> tuple[pd.DataFrame, dict]:
    """Sibling of `_phase1_rerank_within_parent` that uses the input
    `cell_id_col` for parent identity (works regardless of cell_id
    format — integer or FFPE-style dash-containing) instead of regex-
    parsing the label string.

    Depth is determined by parsing the suffix that follows the cell_id
    prefix. A tx with label ``L`` and cell_id ``C`` is:

      - main (depth 0):       ``L == C``
      - depth-1 partial:      ``L == C + "-{k}"`` with k an integer
      - sub-partial (depth 2):``L == C + "-{k}-{j}"``

    Cell_ids that natively contain dashes (e.g. PDAC's ``adohnpem-1``)
    are handled correctly: the suffix-after-cell_id (``""`` /
    ``"-1"`` / ``"-1-1"``) is parsed, not the cell_id itself.

    Output semantics identical to the legacy version: sub-partials
    follow their depth-1 ancestor's renaming, bump-on-collision for
    deposed mains, strict `>` margin gate.

    Reads from the `_etype` column when present to filter rerank
    candidates (only ``cell`` / ``partial`` types are considered;
    cascade ``component`` entities and ``unknown`` sentinels are
    skipped). On dataframes without `_etype`, falls back to a label-
    structure check that treats any pattern ``{cell_id}(-\\d+){0,2}``
    as a rerank candidate.
    """
    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    labels = df_out[entity_col].to_numpy(dtype=object).copy()
    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)
    cell_ids = df_out[cell_id_col].astype(str).to_numpy()
    has_etype = "_etype" in df_out.columns
    etype_arr = (
        df_out["_etype"].astype(str).to_numpy() if has_etype else None
    )

    # Set of labels classified as TRACER-managed entities (cell or
    # partial). On dataframes with `_etype` we read directly; otherwise
    # fall back to checking that the suffix-after-cell_id matches the
    # `{cell_id}(-\\d+){0,2}` shape.
    UNASSIGNED_SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}

    def _suffix_indices(lab: str, cid: str) -> list[int] | None:
        """Return the integer suffix-after-cell_id components, or None
        if the label doesn't follow the expected form for parent ``cid``.

        Examples:
          ('42',         '42')        → []
          ('42-1',       '42')        → [1]
          ('42-1-1',     '42')        → [1, 1]
          ('adohnpem-1', 'adohnpem-1')→ []
          ('adohnpem-1-1','adohnpem-1')→ [1]
          ('cascade_3-1','42')        → None (different parent)
        """
        if lab == cid:
            return []
        if not lab.startswith(cid + "-"):
            return None
        suffix = lab[len(cid) + 1:]
        parts = suffix.split("-")
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None

    # Bucket tx by (parent_cell_id, depth_1_label). Depth-1 label is
    # `cid` for mains, `cid-{k}` for depth-1 partials, and for sub-
    # partials we use their depth-1 ancestor (`cid-{k}`).
    parent_to_depth1_rows: dict[str, dict[str, list[int]]] = {}
    for i, (lab, cid) in enumerate(zip(labels, cell_ids)):
        if cid in UNASSIGNED_SENTINELS:
            continue
        if has_etype and etype_arr[i] not in ("cell", "partial"):
            continue
        suffix_idx = _suffix_indices(str(lab), str(cid))
        if suffix_idx is None or len(suffix_idx) > 2:
            continue  # not a Phase 1-style entity (e.g., cascade)
        depth1 = str(cid) if not suffix_idx else f"{cid}-{suffix_idx[0]}"
        parent_to_depth1_rows.setdefault(str(cid), {}).setdefault(
            depth1, []
        ).append(i)

    stats = {"n_parents_reranked": 0, "n_tx_relabeled": 0}
    # Accumulate _etype relabels across ALL parents, applied once after the loop
    # (the old per-parent `df.loc[mask, "_etype"]=...` was a full-column
    # categorical setitem per reranked parent → O(n_parents × n_tx)).
    all_to_cell: list[int] = []
    all_to_partial: list[int] = []

    for parent, depth1_map in parent_to_depth1_rows.items():
        if len(depth1_map) < 2:
            continue

        sizes: list[tuple[str, int]] = []
        current_main = parent if parent in depth1_map else None
        for d1, rows in depth1_map.items():
            n_nuc = int(sum(1 for r in rows if is_nuclear[r]))
            sizes.append((d1, n_nuc))

        def _sort_key(d1_size: tuple[str, int], _cm=current_main) -> tuple:
            d1, n = d1_size
            return (-n, -(d1 == _cm), d1)
        sizes.sort(key=_sort_key)

        n_largest = sizes[0][1]
        n_runner_up = sizes[1][1]
        if (n_largest - n_runner_up) < margin_tx:
            continue
        if sizes[0][0] == current_main:
            continue

        # Count rank-0's sub-partials to reserve their suffix slots.
        rank0_old_d1 = sizes[0][0]
        rank0_subs: set[int] = set()
        for r in depth1_map[rank0_old_d1]:
            suffix_idx = _suffix_indices(str(labels[r]), parent)
            assert suffix_idx is not None
            if len(suffix_idx) == 2:
                rank0_subs.add(suffix_idx[1])
        n_rank0_subs = len(rank0_subs)

        # Build the depth-1 rename map.
        new_depth1: dict[str, str] = {}
        for k, (d1, _) in enumerate(sizes):
            new_depth1[d1] = parent if k == 0 else f"{parent}-{k + n_rank0_subs}"

        # Sub-suffix renumber: uniform per old depth-1, starting at 1.
        sub_rename: dict[tuple[str, int], int] = {}
        for d1, _ in sizes:
            old_d2js: list[int] = []
            for r in depth1_map[d1]:
                suffix_idx = _suffix_indices(str(labels[r]), parent)
                assert suffix_idx is not None
                if len(suffix_idx) == 2 and suffix_idx[1] not in old_d2js:
                    old_d2js.append(suffix_idx[1])
            old_d2js.sort()
            for new_idx, old_d2j in enumerate(old_d2js, start=1):
                sub_rename[(d1, old_d2j)] = new_idx

        all_rows = [r for rows in depth1_map.values() for r in rows]
        for r in all_rows:
            suffix_idx = _suffix_indices(str(labels[r]), parent)
            assert suffix_idx is not None
            old_d1 = parent if not suffix_idx else f"{parent}-{suffix_idx[0]}"
            new_d1 = new_depth1[old_d1]
            if len(suffix_idx) < 2:
                labels[r] = new_d1
                if new_d1 == parent:
                    all_to_cell.append(r)
                else:
                    all_to_partial.append(r)
            else:
                new_d2 = sub_rename[(old_d1, suffix_idx[1])]
                labels[r] = f"{new_d1}-{new_d2}"
                # sub-partial keeps its existing "partial" etype

        stats["n_parents_reranked"] += 1
        stats["n_tx_relabeled"] += len(all_rows)

    # Apply all _etype changes in ONE positional pass. Row sets are globally
    # disjoint (each tx belongs to one parent and is relabeled to cell OR
    # partial at most once), so this is bit-identical to the per-parent
    # masked assignment it replaces.
    if has_etype:
        _etype_pos = df_out.columns.get_loc("_etype")
        if all_to_cell:
            df_out.iloc[all_to_cell, _etype_pos] = "cell"
        if all_to_partial:
            df_out.iloc[all_to_partial, _etype_pos] = "partial"
    df_out[entity_col] = labels
    return df_out, stats


def _spatial_split_phase1_entities(df_in: pd.DataFrame, *,
                                     entity_col: str,
                                     coord_cols=("x", "y", "z"),
                                     dz_threshold: float = SPLIT_PHASE1_DZ,
                                     min_size: int = SPLIT_PHASE1_MIN_TX,
                                     min_entity_size: int = SPLIT_PHASE1_MIN_ENTITY,
                                     unassigned_id: str = "-1") -> tuple[pd.DataFrame, dict]:
    """Sort tx by z within each entity; split where Δz > dz_threshold.

    Applies to all main cells and partials (≥ min_entity_size tx).
    Largest sub-group keeps the original label; smaller groups get a
    fresh appended suffix; groups < min_size demoted to unassigned_id.

    Returns (df_out, stats) where stats reports counts of entities
    examined / split / demoted etc.
    """
    import re as _re_inner
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    coords_arr = df_out[list(coord_cols)].to_numpy(dtype=np.float64)

    # Prefer the `_etype` column for cell/partial classification — it is set
    # upstream from kernel codes (see pruning.etype_from_codes), so it stays
    # correct on alphanumeric cell_ids (e.g. "jikammne-1") where the input
    # cell_id natively contains dashes and label parsing is ambiguous. Falls
    # back to the legacy numeric-label regex when the column is absent.
    has_etype = "_etype" in df_out.columns

    # Pre-compute the next collision-free child suffix per parent label, so
    # split-emitted labels don't collide with existing children. A child is
    # any label of the form "{parent}-{int}"; we split on the *last* dash so
    # that dash-containing cell_ids and arbitrary depths are handled
    # uniformly:
    #   "37962-1"        → next_child["37962"]   = max(...,1)
    #   "37962-1-2"      → next_child["37962-1"] = max(...,2)
    #   "jikammne-1-1"   → next_child["jikammne-1"] = max(...,1)
    next_child: dict[str, int] = {}
    for lab in df_out[entity_col].unique():
        m = _re.match(r"^(.*)-(\d+)$", str(lab))
        if m:
            prefix, k = m.group(1), int(m.group(2))
            next_child[prefix] = max(next_child.get(prefix, 0), k)

    out_labels = df_out[entity_col].to_numpy().copy()
    z_arr = df_out[coord_cols[2]].to_numpy(dtype=np.float64)

    stats = {
        "entities_examined": 0,
        "entities_split": 0,
        "subgroups_minted": 0,
        "tx_demoted_singletons": 0,
        "tx_total_relabelled": 0,
    }
    # Accumulate _etype relabels across ALL split entities, applied once after
    # the loop (the old per-entity `df.loc[mask, "_etype"]=...` was a full-column
    # categorical setitem + len(df) bool-mask alloc per split entity →
    # O(n_split × n_tx)).
    all_to_unknown: list[np.ndarray] = []
    all_to_partial: list[np.ndarray] = []

    for ent, group in df_out.groupby(entity_col, sort=False):
        if ent == unassigned_id or ent == "UNASSIGNED" or ent.startswith("UNASSIGNED_"):
            continue
        if has_etype:
            if str(group["_etype"].iloc[0]) not in ("cell", "partial"):
                continue  # cascade component / unknown — not split here
        elif not _re.match(r"^\d+(-\d+){0,2}$", ent):
            continue  # legacy fallback: numeric cell/partial labels only
        rows = group.index.to_numpy()
        if len(rows) < min_entity_size:
            continue
        stats["entities_examined"] += 1

        # Sort tx by z; find gaps > dz_threshold in the sorted sequence.
        z_vals = z_arr[rows]
        sort_order = np.argsort(z_vals, kind="stable")
        rows_sorted = rows[sort_order]
        z_sorted = z_vals[sort_order]
        diffs = np.diff(z_sorted)
        split_positions = np.where(diffs > dz_threshold)[0]
        if len(split_positions) == 0:
            continue

        # Slice rows_sorted into contiguous z-groups at split positions.
        # split_positions[i] is the last index of group i (group boundaries
        # are between i and i+1 in the sorted array).
        groups_rows = []
        prev = 0
        for sp in split_positions:
            groups_rows.append(rows_sorted[prev : sp + 1])
            prev = sp + 1
        groups_rows.append(rows_sorted[prev:])

        # Rank groups by size descending — largest keeps original label.
        groups_rows.sort(key=lambda a: -len(a))
        stats["entities_split"] += 1

        for k, gr in enumerate(groups_rows):
            sz = len(gr)
            if sz < min_size:
                out_labels[gr] = unassigned_id
                stats["tx_demoted_singletons"] += sz
                all_to_unknown.append(gr)
                continue
            if k == 0:
                continue  # largest keeps original label (and its existing _etype)

            # Mint a fresh child label by appending the next collision-free
            # suffix to the parent label. Prefix-agnostic: works for numeric
            # ("37962" → "37962-1") and dash-containing cell_ids
            # ("jikammne-1" → "jikammne-1-1") alike.
            next_child[ent] = next_child.get(ent, 0) + 1
            new_label = f"{ent}-{next_child[ent]}"

            out_labels[gr] = new_label
            stats["subgroups_minted"] += 1
            stats["tx_total_relabelled"] += sz
            # New sub-group inherits "partial" etype regardless of whether
            # the parent was a main (depth-1 partial emitted) or a partial
            # (sub-partial emitted) — per the design, sub-partials are
            # "partial" flat.
            all_to_partial.append(gr)

    # Apply all _etype changes in ONE positional pass. Row sets are globally
    # disjoint (each tx belongs to one split entity and one sub-group), so this
    # is bit-identical to the per-entity masked assignment it replaces.
    if "_etype" in df_out.columns:
        _etype_pos = df_out.columns.get_loc("_etype")
        if all_to_unknown:
            df_out.iloc[np.concatenate(all_to_unknown), _etype_pos] = "unknown"
        if all_to_partial:
            df_out.iloc[np.concatenate(all_to_partial), _etype_pos] = "partial"
    df_out[entity_col] = out_labels
    return df_out, stats


def _vectorized_mean_pmi_excl_self(W: np.ndarray,
                                     query_gene_idx: np.ndarray,
                                     seed_gene_idx: np.ndarray
                                     ) -> np.ndarray:
    """For each gene index in `query_gene_idx`, compute the mean PMI
    against `seed_gene_idx`, excluding any seed entry equal to the
    query gene itself (matches the legacy `_mean_pmi(g, S)` semantics
    where `others = [x for x in S if x != g]`).

    Parameters
    ----------
    W : (G, G) float32 PMI matrix.
    query_gene_idx : (N,) int array, values in [0, G). All entries
        must be ≥ 0.
    seed_gene_idx : (M,) int array, the seed gene-set indices. Order
        is irrelevant; duplicates are allowed but treated as one slot.

    Returns
    -------
    (N,) float64 array. NaN where the seed has no finite-PMI entries
    after self-exclusion.
    """
    if query_gene_idx.size == 0 or seed_gene_idx.size == 0:
        return np.full(query_gene_idx.shape, np.nan, dtype=np.float64)
    # (N, M) PMI slice
    pmi = W[np.ix_(query_gene_idx, seed_gene_idx)].astype(np.float64, copy=False)
    # (N, M) self-exclusion mask: seed_j == query_i
    self_mask = seed_gene_idx[None, :] == query_gene_idx[:, None]
    valid = np.isfinite(pmi) & ~self_mask
    # safe mean: sum / count
    pmi_clean = np.where(valid, pmi, 0.0)
    count = valid.sum(axis=1)
    total = pmi_clean.sum(axis=1)
    out = np.full(query_gene_idx.shape, np.nan, dtype=np.float64)
    nonzero = count > 0
    out[nonzero] = total[nonzero] / count[nonzero]
    return out


def _reassign_nuclear_post_1c_etype(df_in: pd.DataFrame, *,
                                       entity_col: str,
                                       aux: dict,
                                       cell_id_col: str = "cell_id",
                                       gene_col: str = "feature_name",
                                       nuclear_col: str = "overlaps_nucleus",
                                       margin: float = 0.05,
                                       threshold: float = 0.05,
                                       ) -> tuple[pd.DataFrame, dict]:
    """Sibling of `_reassign_nuclear_post_1c` that uses the input
    `cell_id_col` for parent identity instead of regex-parsing the
    label string. Works on any cell_id format — integer (lung cancer)
    or alphanumeric / dash-containing (Xenium FFPE / IO).

    All other semantics identical to the legacy / regex version:
    sequential margin walk, `mp > best_mean + margin` admission rule,
    nuclear-only tx considered.
    """
    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    label_arr = df_out[entity_col].to_numpy(dtype=object).copy()

    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]
    # Two gene-fit backends:
    #   - dense  (G,G) float32: the legacy path, NaN marks unobserved
    #     pairs and `_mean_pmi_excl_self` skips them.
    #   - sparse CSR upper-triangle (the bootstrap's `W_sparse`): scales
    #     to whole-transcriptome panels where a dense (G,G) is ~1.6 GB.
    #     Structurally-absent pairs are skipped (NOT filled with 0 — that
    #     is the coherence convention and the wrong one here). We do NOT
    #     densify, since `.todense()` turns absent→0 and re-introduces the
    #     NaN-vs-0 gene-fit bug.
    import scipy.sparse as _sp
    W_is_sparse = _sp.issparse(W)
    if W_is_sparse:
        # Symmetrize the upper-triangle CSR via COO-stacking. NOT `W + W.T`:
        # scipy's sparse add eliminates explicit zeros, which would drop an
        # observed PMI of exactly 0.0. Never call eliminate_zeros() here for
        # the same reason.
        Wu = W.tocoo()
        Wsym = _sp.csr_matrix(
            (
                np.concatenate([Wu.data, Wu.data]).astype(np.float32),
                (
                    np.concatenate([Wu.row, Wu.col]),
                    np.concatenate([Wu.col, Wu.row]),
                ),
            ),
            shape=W.shape,
            dtype=np.float32,
        )
        Wsym.sort_indices()
        W_sp_indptr = Wsym.indptr.astype(np.int32)
        W_sp_indices = Wsym.indices.astype(np.int32)
        W_sp_data = Wsym.data.astype(np.float32)
    else:
        if not isinstance(W, np.ndarray):
            W = np.asarray(W, dtype=np.float32)
        if W.dtype != np.float32:
            W = W.astype(np.float32)

    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)
    nuc_genes = df_out[gene_col].astype(str).to_numpy()
    cell_id_arr = df_out[cell_id_col].astype(str).to_numpy()
    nuc_idx = np.where(is_nuclear)[0]

    has_etype = "_etype" in df_out.columns
    etype_arr = (
        df_out["_etype"].astype(str).to_numpy() if has_etype else None
    )

    # Build per-entity nuclear-gene set via pandas groupby.
    _UN = {"-1", "DROP", "UNASSIGNED", "nan"}
    label_s = pd.Series(label_arr)
    is_un_label = (
        label_s.isin(_UN) | label_s.str.startswith("UNASSIGNED_", na=False)
    )
    nuc_with_entity_mask = is_nuclear & (~is_un_label.to_numpy())
    if nuc_with_entity_mask.any():
        gi_arr_all = (
            pd.Series(nuc_genes).map(gene_to_idx)
            .fillna(-1).astype(np.int64).to_numpy()
        )
        valid_for_build = nuc_with_entity_mask & (gi_arr_all >= 0)
        if valid_for_build.any():
            build_df = pd.DataFrame({
                "entity": label_arr[valid_for_build].astype(str),
                "gi":     gi_arr_all[valid_for_build],
            }).drop_duplicates()
            entity_to_genes_lists = (
                build_df.groupby("entity", sort=False)["gi"]
                .apply(np.array).to_dict()
            )
            entity_to_genes: dict[str, set[int]] = {
                e: set(int(x) for x in arr)
                for e, arr in entity_to_genes_lists.items()
            }
        else:
            entity_to_genes = {}
    else:
        entity_to_genes = {}

    # Build parent_to_entities using the cell_id + _etype columns.
    # Previous version did `np.where(label_arr == ent)` once per
    # entity — O(N_entities * N_tx). On dense PDAC scale (~5K
    # entities × ~500K tx) that was ~2.5B compares, dominating wall.
    #
    # New approach: single groupby('cell_id') over tx that are in
    # a Phase 1 cell-or-partial entity (etype ∈ {cell, partial} when
    # the column is present; falls back to label-prefix check otherwise).
    # Each parent's entity list is unique(tracer_id) inside its group.
    # No per-entity scan; O(N_tx · log N_tx).
    SENT = _UN
    parent_to_entities: dict[str, list[str]] = {}

    if has_etype:
        # Fast path: etype filter on column.
        keep_mask = np.isin(etype_arr, ("cell", "partial"))
    else:
        # Fallback: label is either the cell_id itself or starts with cell_id+'-'.
        # Build a per-tx boolean: label==cid OR label.startswith(cid+'-').
        lab_s = pd.Series(label_arr, dtype=str)
        cid_s = pd.Series(cell_id_arr, dtype=str)
        keep_mask = (lab_s == cid_s) | (
            lab_s.str.startswith(cid_s.add("-"), na=False)
        )
        keep_mask = keep_mask.to_numpy()

    # Drop sentinel cell_ids too.
    cid_str = cell_id_arr.astype(str)
    keep_mask = keep_mask & (~np.isin(cid_str, list(SENT)))

    if keep_mask.any():
        sub_df = pd.DataFrame({
            "parent": cid_str[keep_mask],
            "ent":    pd.Series(label_arr, dtype=str).to_numpy()[keep_mask],
        }).drop_duplicates()
        for parent, grp in sub_df.groupby("parent", sort=False):
            ent_list = [e for e in grp["ent"].tolist()
                        if e in entity_to_genes]
            if ent_list:
                parent_to_entities[parent] = ent_list

    # Candidate tx per parent: nuclear, in main entity (label == cell_id),
    # AND etype == "cell" if column present.
    cand_by_parent: dict[str, np.ndarray] = {}
    if nuc_idx.size > 0:
        nuc_label = label_arr[nuc_idx].astype(str)
        nuc_cid = cell_id_arr[nuc_idx]
        is_cand = nuc_label == nuc_cid
        if has_etype:
            is_cand &= (etype_arr[nuc_idx] == "cell")
        if is_cand.any():
            cand_tx_idx = nuc_idx[is_cand]
            cand_parent = nuc_cid[is_cand]
            order = np.argsort(cand_parent, kind="stable")
            cand_tx_idx_sorted = cand_tx_idx[order]
            cand_parent_sorted = cand_parent[order]
            change = np.r_[True, cand_parent_sorted[1:] != cand_parent_sorted[:-1]]
            starts = np.where(change)[0]
            ends = np.r_[starts[1:], cand_parent_sorted.size]
            for s, e in zip(starts, ends):
                cand_by_parent[str(cand_parent_sorted[s])] = cand_tx_idx_sorted[s:e]

    # Build CSR-style flat arrays for the Cython kernel. Only enumerate
    # parents that have ≥2 entities AND have candidate tx.
    active_parents: list[str] = []
    partials_per_parent: list[list[str]] = []
    for parent, ent_list in parent_to_entities.items():
        if len(ent_list) < 2:
            continue
        main = parent
        if main not in entity_to_genes:
            continue
        partials = [e for e in ent_list if e != main]
        if not partials:
            continue
        if parent not in cand_by_parent or cand_by_parent[parent].size == 0:
            continue
        active_parents.append(parent)
        partials_per_parent.append(partials)

    n_parents_with_partials = len(active_parents)

    if n_parents_with_partials == 0:
        df_out[entity_col] = label_arr
        return df_out, {
            "n_tx_moved": 0,
            "n_parents_with_partials": 0,
        }

    # Vectorize gene-string → gene-index for ALL nuclear tx in one pass
    # (rather than per-parent dict lookups). The result is a global
    # int32 array we can index into cheaply per parent.
    nuc_gene_idx_global = (
        pd.Series(nuc_genes).map(gene_to_idx)
        .fillna(-1).astype(np.int32).to_numpy()
    )

    # Pre-convert entity_to_genes to a single concatenated int32 array
    # plus offsets, with a name → (start, end) lookup. Avoids the
    # per-parent np.fromiter cost.
    entity_names = list(entity_to_genes.keys())
    name_to_eindex = {n: i for i, n in enumerate(entity_names)}
    entity_gene_arrays = [
        np.fromiter(entity_to_genes[n], dtype=np.int32)
        for n in entity_names
    ]
    entity_gene_offsets = np.zeros(len(entity_names) + 1, dtype=np.int32)
    if entity_gene_arrays:
        entity_gene_offsets[1:] = np.cumsum(
            [a.size for a in entity_gene_arrays], dtype=np.int32,
        )
        entity_gene_flat = np.concatenate(entity_gene_arrays).astype(np.int32)
    else:
        entity_gene_flat = np.zeros(0, dtype=np.int32)

    def _entity_gene_slice(name: str) -> tuple[int, int]:
        ei = name_to_eindex[name]
        return int(entity_gene_offsets[ei]), int(entity_gene_offsets[ei + 1])

    # Build CSR arrays for the kernel. The loops below are now light:
    # just offset arithmetic + array indexing, no dict / map calls.
    cand_offsets = np.zeros(n_parents_with_partials + 1, dtype=np.int32)
    cand_sizes = np.fromiter(
        (cand_by_parent[p].size for p in active_parents),
        dtype=np.int32, count=n_parents_with_partials,
    )
    cand_offsets[1:] = np.cumsum(cand_sizes, dtype=np.int32)
    cand_tx_arr = np.concatenate(
        [cand_by_parent[p] for p in active_parents]
    ).astype(np.int32)
    cand_gene_arr = nuc_gene_idx_global[cand_tx_arr]

    main_sizes = np.fromiter(
        (
            entity_gene_offsets[name_to_eindex[p] + 1]
            - entity_gene_offsets[name_to_eindex[p]]
            for p in active_parents
        ),
        dtype=np.int32, count=n_parents_with_partials,
    )
    main_offsets = np.zeros(n_parents_with_partials + 1, dtype=np.int32)
    main_offsets[1:] = np.cumsum(main_sizes, dtype=np.int32)
    main_genes_arr = np.concatenate([
        entity_gene_flat[
            entity_gene_offsets[name_to_eindex[p]]:
            entity_gene_offsets[name_to_eindex[p] + 1]
        ]
        for p in active_parents
    ]).astype(np.int32) if active_parents else np.zeros(0, dtype=np.int32)

    # Partials: enumerate flat per-parent, then per-partial CSR.
    partial_counts = np.fromiter(
        (len(partials) for partials in partials_per_parent),
        dtype=np.int32, count=n_parents_with_partials,
    )
    partial_offsets = np.zeros(n_parents_with_partials + 1, dtype=np.int32)
    partial_offsets[1:] = np.cumsum(partial_counts, dtype=np.int32)
    flat_partial_names = [
        pe for partials in partials_per_parent for pe in partials
    ]
    partial_sizes = np.fromiter(
        (
            (entity_gene_offsets[name_to_eindex[pe] + 1]
             - entity_gene_offsets[name_to_eindex[pe]])
            if pe in name_to_eindex else 0
            for pe in flat_partial_names
        ),
        dtype=np.int32, count=len(flat_partial_names),
    )
    partial_gene_offsets_arr = np.zeros(len(flat_partial_names) + 1, dtype=np.int32)
    partial_gene_offsets_arr[1:] = np.cumsum(partial_sizes, dtype=np.int32)
    partial_genes_flat_chunks = []
    for pe in flat_partial_names:
        if pe in name_to_eindex:
            s = entity_gene_offsets[name_to_eindex[pe]]
            e = entity_gene_offsets[name_to_eindex[pe] + 1]
            partial_genes_flat_chunks.append(entity_gene_flat[s:e])
        else:
            partial_genes_flat_chunks.append(np.zeros(0, dtype=np.int32))
    partial_genes_arr = (
        np.concatenate(partial_genes_flat_chunks).astype(np.int32)
        if partial_genes_flat_chunks else np.zeros(0, dtype=np.int32)
    )

    # Call the kernel — per-candidate local-partial-index decision. The
    # sparse and dense kernels share identical offset-loop / margin logic
    # and agree bit-for-bit on a fully-dense panel; they differ only in how
    # the per-seed mean PMI is gathered (CSR binary-search vs direct index).
    if W_is_sparse:
        from tracer._cy_reassign import reassign_nuclear_post_1c_kernel_sparse
        out_partial_local_idx = reassign_nuclear_post_1c_kernel_sparse(
            W_sp_indptr,
            W_sp_indices,
            W_sp_data,
            cand_offsets,
            cand_gene_arr,
            main_offsets,
            main_genes_arr,
            partial_offsets,
            partial_gene_offsets_arr,
            partial_genes_arr,
            float(margin),
        )
    else:
        # Ensure W is C-contiguous float32 (kernel requirement).
        if not (W.flags["C_CONTIGUOUS"] and W.dtype == np.float32):
            W = np.ascontiguousarray(W, dtype=np.float32)
        from tracer._cy_reassign import reassign_nuclear_post_1c_kernel
        out_partial_local_idx = reassign_nuclear_post_1c_kernel(
            W,
            cand_offsets,
            cand_gene_arr,
            main_offsets,
            main_genes_arr,
            partial_offsets,
            partial_gene_offsets_arr,
            partial_genes_arr,
            float(margin),
        )

    # Apply moves: for each candidate with out>=0, relabel its tx to the
    # corresponding partial.
    moved_tx_indices: list[int] = []
    n_moves = 0
    for i in range(n_parents_with_partials):
        s, e = int(cand_offsets[i]), int(cand_offsets[i + 1])
        if s == e:
            continue
        choices = out_partial_local_idx[s:e]
        moved_mask = choices >= 0
        if not moved_mask.any():
            continue
        local_partials = partials_per_parent[i]
        tx_slice = cand_tx_arr[s:e]
        for n_local in np.where(moved_mask)[0]:
            kk = int(choices[n_local])
            tx_global = int(tx_slice[n_local])
            label_arr[tx_global] = local_partials[kk]
            moved_tx_indices.append(tx_global)
        n_moves += int(moved_mask.sum())

    df_out[entity_col] = label_arr
    if has_etype and moved_tx_indices:
        mask = np.zeros(len(df_out), dtype=bool)
        mask[moved_tx_indices] = True
        df_out.loc[mask, "_etype"] = "partial"
    return df_out, {
        "n_tx_moved": n_moves,
        "n_parents_with_partials": n_parents_with_partials,
    }


def _split_unassigned_components(df_in: pd.DataFrame, *,
                                   entity_col: str,
                                   coord_cols=("x", "y", "z"),
                                   dz_threshold: float,
                                   min_size: int = 1,
                                   min_entity_size: int = 2,
                                   unassigned_id: str = "-1"
                                   ) -> tuple[pd.DataFrame, dict]:
    """Sort tx by z within each UNASSIGNED_* component; split where
    consecutive Δz > dz_threshold.

    Mirrors `_spatial_split_phase1_entities` but operates on
    UNASSIGNED_<base> labels emitted by Group. Group's spatial graph
    is essentially 2D (xy bins + d ≤ 1.5 µm in xy), so its components
    can span large z gaps; this stage catches that.

    Largest sub-cluster keeps the original label; smaller ones get a
    fresh suffix `UNASSIGNED_<base>-<k>` for k = 1, 2, ...
    Sub-clusters smaller than ``min_size`` demote to ``unassigned_id``.
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    out_labels = df_out[entity_col].to_numpy().copy()
    z_arr = df_out[coord_cols[2]].to_numpy(dtype=np.float64)

    stats = {
        "components_examined": 0,
        "components_split": 0,
        "subcomps_minted": 0,
        "tx_demoted_singletons": 0,
        "tx_total_relabelled": 0,
    }

    # Track existing UNASSIGNED_<base>-<k> suffixes to avoid collisions
    next_subidx: dict[str, int] = {}
    for lab in df_out[entity_col].unique():
        m = _re.match(r"^(UNASSIGNED_\d+)-(\d+)$", str(lab))
        if m:
            base, k = m.group(1), int(m.group(2))
            next_subidx[base] = max(next_subidx.get(base, 0), k)

    for ent, group in df_out.groupby(entity_col, sort=False):
        if not isinstance(ent, str) or not ent.startswith("UNASSIGNED_"):
            continue
        # Only base labels — already-suffixed labels (e.g. UNASSIGNED_42-1)
        # were emitted by a prior split; skip to avoid recursion.
        if "-" in ent.replace("UNASSIGNED_", "", 1):
            continue
        rows = group.index.to_numpy()
        if len(rows) < min_entity_size:
            continue
        stats["components_examined"] += 1

        z_vals = z_arr[rows]
        sort_order = np.argsort(z_vals, kind="stable")
        rows_sorted = rows[sort_order]
        z_sorted = z_vals[sort_order]
        diffs = np.diff(z_sorted)
        split_positions = np.where(diffs > dz_threshold)[0]
        if split_positions.size == 0:
            continue

        groups_rows = []
        prev = 0
        for sp in split_positions:
            groups_rows.append(rows_sorted[prev: sp + 1])
            prev = sp + 1
        groups_rows.append(rows_sorted[prev:])

        # Sort by size desc — largest keeps the original label
        groups_rows.sort(key=lambda a: -len(a))
        stats["components_split"] += 1

        demoted_rows_local: list[np.ndarray] = []
        for k, gr in enumerate(groups_rows):
            sz = len(gr)
            if sz < min_size:
                out_labels[gr] = unassigned_id
                stats["tx_demoted_singletons"] += sz
                demoted_rows_local.append(gr)
                continue
            if k == 0:
                continue  # largest keeps original label
            next_subidx[ent] = next_subidx.get(ent, 0) + 1
            new_label = f"{ent}-{next_subidx[ent]}"
            out_labels[gr] = new_label
            stats["subcomps_minted"] += 1
            stats["tx_total_relabelled"] += sz
            # New sub-component label is still a Group component; _etype
            # stays "component" — no update needed.
        if demoted_rows_local and "_etype" in df_out.columns:
            rows_concat = np.concatenate(demoted_rows_local)
            mask = np.zeros(len(df_out), dtype=bool)
            mask[rows_concat] = True
            df_out.loc[mask, "_etype"] = "unknown"

    df_out[entity_col] = out_labels
    return df_out, stats


def _qc_demote_low_coherence(df_in: pd.DataFrame, *,
                               entity_col: str,
                               aux: dict,
                               min_C: float,
                               min_n_genes: int = 2,
                               threshold: float = 0.05,
                               metric: str = "pmi",
                               unassigned_id: str = "-1",
                               real_signal_threshold: float | None = None,
                               ) -> tuple[pd.DataFrame, dict]:
    """Demote any entity (cell, partial, or component) whose internal
    coherence is below ``min_C``, OR whose distinct-gene count is
    below ``min_n_genes``. The latter forces single-gene entities
    to fail (coherence is undefined for n_genes < 2).

    Uses the Cython batch coherence kernel
    (`tracer._cy_prune.coherence_count_per_entity_batch`) for the
    per-entity coherence computation. ~50–100x faster than a Python
    groupby + per-entity coherence call on large entity sets.

    Returns (df_out, stats). When ``min_C <= 0`` AND
    ``min_n_genes <= 1``, the function is a no-op.
    """
    if min_C <= 0 and min_n_genes <= 1:
        return df_in.copy(), {
            "entities_examined": 0, "entities_demoted_low_C": 0,
            "entities_demoted_few_genes": 0, "tx_demoted": 0,
        }

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]

    # Densify W to float32 contiguous if needed (Cython expects this)
    if not isinstance(W, np.ndarray):
        # sparse → dense
        W = np.asarray(W.todense() if hasattr(W, "todense") else W,
                       dtype=np.float32)
    if W.dtype != np.float32:
        W = W.astype(np.float32)
    if not W.flags.c_contiguous:
        W = np.ascontiguousarray(W)

    # Map each tx's gene to its W-index in one vectorized pass.
    gene_arr = df_out["feature_name"].astype(str).to_numpy()
    gene_idx_arr = np.array(
        [gene_to_idx.get(g, -1) for g in gene_arr], dtype=np.int32,
    )
    label_arr = df_out[entity_col].to_numpy()
    label_str = label_arr.astype(str)

    # Filter: drop unassigned-class sentinels and tx with unknown gene.
    # NOTE: UNASSIGNED_<n> components ARE valid entities — they're kept.
    # Only the literal sentinels are excluded.
    drop_set = {str(unassigned_id), "UNASSIGNED", "DROP", "nan"}
    is_unassigned_sentinel = np.isin(label_str, list(drop_set))
    keep = (~is_unassigned_sentinel) & (gene_idx_arr >= 0)

    sub_labels = label_str[keep]
    sub_genes = gene_idx_arr[keep]
    if sub_labels.size == 0:
        return df_out, {
            "entities_examined": 0, "entities_demoted_low_C": 0,
            "entities_demoted_few_genes": 0, "tx_demoted": 0,
        }

    # Build CSR (entity → unique gene indices). Pandas sort + drop_duplicates.
    tmp = pd.DataFrame({"label": sub_labels, "gene": sub_genes})
    tmp = tmp.drop_duplicates(subset=["label", "gene"], keep="first")
    tmp = tmp.sort_values(["label", "gene"], kind="stable")
    counts = tmp.groupby("label", sort=False).size().to_numpy(dtype=np.int32)
    entity_ids = tmp.groupby("label", sort=False).size().index.to_numpy()
    offsets = np.empty(counts.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    # `to_numpy(dtype=...)` returns a read-only view in some
    # pandas/numpy combinations (notably pandas + numpy<2). The
    # Cython kernel requests a writable typed memoryview, so we
    # need an explicit writable copy here.
    flat_genes = np.ascontiguousarray(
        tmp["gene"].to_numpy(dtype=np.int32, copy=True)
    )

    # Cython batch call. When real_signal_threshold > 0, the kernel
    # uses the "real players" denominator (only pairs with |W| above
    # the noise floor count) — making C panel-shape-agnostic across
    # dense (legacy) and sparse (bootstrap, Visium HD) W matrices.
    # Informative-edges denominator by default (rst = tau) so the coherence
    # floor is panel-shape-agnostic (purity+conflict=1 over informative pairs).
    rst = float(threshold) if real_signal_threshold is None else float(real_signal_threshold)
    from tracer._cy_prune import coherence_count_per_entity_batch
    C_arr, _P_arr, _N_arr = coherence_count_per_entity_batch(
        offsets, flat_genes, W, float(threshold), rst,
    )

    # Decide demotion per entity
    n_genes_per_ent = counts
    bad_few = (n_genes_per_ent < min_n_genes)
    # n_finite == 0 returns C=0 from kernel; that's caught by "C ≤ min_C"
    # for any positive min_C.
    bad_low = (~bad_few) & (C_arr <= min_C)

    bad_set = set(entity_ids[bad_few | bad_low].tolist())

    # Option A — coherence-triggered sibling promotion. When a failing MAIN
    # `cell` has a surviving (floor-clearing) sibling from the SAME Phase-1
    # family, promote that sibling to the main's label instead of orphaning
    # the whole nucleus. Families are keyed by the tracer_id LABEL structure
    # ({C} main / {C}-{k} partial), NOT the cell_id column — cell_id drifts
    # from the label after transcript reassignment, so grouping by it would
    # match entities across nuclei and mislabel the promoted cell.
    promotions: dict[str, str] = {}   # sibling label -> failed main's label
    if bad_set and "cell_id" in df_out.columns:
        ent_C = {str(e): float(c) for e, c in zip(entity_ids, C_arr)}
        gb = df_out.groupby(entity_col, sort=False)
        # Vectorized mode of cell_id per entity (was a per-group Python lambda
        # calling .mode() over ~150k groups). Count (entity, cell_id) pairs and
        # per entity take the max-count cell_id, ties broken by the smallest
        # cell_id — bit-identical to pandas `mode().iat[0]` (mode returns values
        # sorted ascending, so iat[0] is the smallest among the ties).
        _cc = pd.DataFrame({
            "_e": df_out[entity_col].astype(str).to_numpy(),
            "_cid": df_out["cell_id"].astype(str).to_numpy(),
        })
        _mode = (
            _cc.groupby(["_e", "_cid"], sort=False).size().reset_index(name="_n")
               .sort_values(["_e", "_n", "_cid"], ascending=[True, False, True],
                            kind="stable")
               .drop_duplicates("_e", keep="first")
               .set_index("_e")["_cid"]
        )
        # Reindex to first-appearance order (what `groupby(sort=False)` yields),
        # so the ent_cid iteration order — and hence the order fam_to_ents lists
        # are built, which decides max(sibs, key=(size, C)) ties — is bit-
        # identical to the old per-group agg.
        _order = pd.unique(df_out[entity_col].astype(str))
        ent_cid = _mode.reindex(_order).to_dict()
        ent_etype = (gb["_etype"].first().astype(str).to_dict()
                     if "_etype" in df_out.columns else {})
        ent_size = {str(e): int(n) for e, n in gb.size().items()}  # tx per entity

        def _family(label: str) -> str | None:
            """Parent cell_id C if `label` is C (main) or C-{k...} (partial),
            using the entity's own cell_id to strip the suffix (cell_ids may
            themselves contain dashes). None if the label doesn't structurally
            match its cell_id — i.e. a reassigned/relabeled entity."""
            cid = ent_cid.get(label)
            if cid is None:
                return None
            if label == cid:
                return cid
            if label.startswith(cid + "-"):
                suf = label[len(cid) + 1:]
                if suf and all(p.isdigit() for p in suf.split("-")):
                    return cid
            return None

        fam_to_ents: dict[str, list[str]] = {}
        ent_fam: dict[str, str | None] = {}
        for e in ent_cid:
            f = _family(e)
            ent_fam[e] = f
            if f is not None:
                fam_to_ents.setdefault(f, []).append(e)
        for M in bad_set:
            if ent_etype.get(M) != "cell":
                continue   # only rescue a failing main cell
            fam = ent_fam.get(M)
            if fam is None:
                continue   # drifted/relabeled main: no clean Phase-1 family
            sibs = [s for s in fam_to_ents.get(fam, [])
                    if s != M and s not in bad_set and s not in promotions
                    and ent_etype.get(s) in ("cell", "partial")]
            if sibs:
                # Floor is the quality GATE (already applied via `s not in
                # bad_set`). Among survivors pick the LARGEST by tx count —
                # count-coherence is inflated at small tx counts, so it's a
                # poor selector, and the promoted main should carry the
                # nucleus's mass (and act as the rescue sink for the released
                # tx). Coherence only breaks size ties.
                promotions[max(
                    sibs,
                    key=lambda s: (ent_size.get(s, 0), ent_C.get(s, -1e18)),
                )] = M

    # Release failing entities' original tx to unassigned first...
    n_demoted = 0
    if bad_set:
        mask = df_out[entity_col].isin(bad_set)
        df_out.loc[mask, entity_col] = unassigned_id
        n_demoted = int(mask.sum())
        if "_etype" in df_out.columns:
            df_out.loc[mask, "_etype"] = "unknown"

    # ...then promote surviving siblings into the now-freed main labels.
    # Vectorized: a single `.map` over the label column, instead of a full-
    # column string `==` per promotion (the old loop was O(n_promotions × n_tx)
    # object comparisons — the dominant Mid-QC cost at scale). The sib→main map
    # is 1:1 with disjoint keys and values (a sib is never itself a target, and
    # each freed main gets exactly one sib), so one masked assignment is
    # bit-identical to applying the promotions one at a time.
    if promotions:
        mapped = df_out[entity_col].map(promotions)   # sib→main, NaN elsewhere
        pmask = mapped.notna().to_numpy()
        if pmask.any():
            col = df_out[entity_col]
            if isinstance(col.dtype, pd.CategoricalDtype):
                new_cats = set(mapped[pmask].unique()) - set(col.cat.categories)
                if new_cats:
                    df_out[entity_col] = col.cat.add_categories(sorted(new_cats))
            df_out.loc[pmask, entity_col] = mapped[pmask].to_numpy()
            if "_etype" in df_out.columns:
                df_out.loc[pmask, "_etype"] = "cell"

    return df_out, {
        "entities_examined": int(entity_ids.size),
        "entities_demoted_low_C": int(bad_low.sum()),
        "entities_demoted_few_genes": int(bad_few.sum()),
        "tx_demoted": n_demoted,
        "entities_promoted": len(promotions),
    }


NUCLEAR_ONLY_ADMIT = True   # restrict 1b/1c to nuclear tx; cyto via Rescue
# Fallback cells (<3 nuclear genes → whole-cell seed) are exempted from the
# nuclear-only restriction: they admit whole-cell tx in 1b/1c. Without this,
# a thin nucleus whose 1-2 nuclear reads don't fit the whole-cell seed dies
# even when its local program is perfectly coherent (~726 recoverable cells,
# 0 lost, on the pdac cPMI ROI). Only coherence-floor-failing seeds still die.
FALLBACK_WHOLE_CELL_ADMIT = True
# 2026-05-13: RESCUE_NEG_THR paired with PMI_THR (both 0.2 in PMI scale).
# Symmetric ± 0.2 dead zone around chance.
RESCUE_NEG_THR = -0.2
ANNOTATE_NEG_THR = -0.1 * (PMI_THR / 0.05)  # scales with PMI_THR; -0.4 at PMI_THR=0.2
# Iterative Rescue caps: 3 passes captures ≥98 % of asymptotic gain at
# any scale (per /tmp/iterative_rescue_*.png diagnostic). Early-stop
# fires when a pass adds zero tx — covers tight crops in 1–2 passes.
RESCUE_MAX_PASSES = 3
# Rescue veto: hybrid two-stage admission for tx of gene g into entity E:
#   1. If g ∈ E.genes  → admit (no test; E was deemed compatible w/ g earlier).
#   2. Else if min PMI(g, E.genes) > RESCUE_MIN_ADMIT → admit (unanimous-pos).
#   3. Else if mean PMI(g, E.genes, finite) > RESCUE_MEAN_ADMIT → admit.
#   4. Else → reject. Tx remains "-1".
RESCUE_VETO_MODE = "hybrid"
RESCUE_MIN_ADMIT = 0.0      # any negative pair drops to mean-stage
# 2026-05-13: RESCUE_MEAN_ADMIT raised 0.1 → 0.5 (PMI scale; ≈1.65× chance).
# Validated: substantially cleaner low-coherence tail (cell C p10 0.57→0.82
# on PDAC, partial p10 0.56→0.79). −2 pp coverage / ~−0.3 pp retention.
RESCUE_MEAN_ADMIT = 0.5     # aggregate must be solidly positive

# Stitch spatial-gate tightening — opt-in knobs. Defaults preserve
# current production behavior; raise values to tighten.
#   STITCH_MIN_LOCAL_TX     0 = off (current). >=1 requires that many
#                              UNIQUE tx of EACH candidate entity in
#                              the shared xy 8-Moore + z-window bins.
#                              Catches single-bridging-tx pairs where
#                              two spatially-separated entities are
#                              glued by one diagonal-Moore pair.
#   STITCH_GZ_UM            None = use auto (current). 1.0 → max
#                              Δz reach in candidate enumeration =
#                              2·G_z = 2 µm, matching SPLIT_PHASE1_DZ.
STITCH_MIN_LOCAL_TX: int = 3
STITCH_GZ_UM: float | None = 1.0

# Stitch acceptance-bypass: when set, a pair that fails ΔC ≥ deltaC_min
# is still accepted if the raw post-merge coherence C(union) ≥ this
# value. Recovers same-program fragment absorptions where both parents
# are already at C ≈ 1.0 and ΔC has no headroom. Spatial-witness and
# candidate-source gates still apply. None = off (legacy ΔC-only gate).
STITCH_C_UNION_BYPASS: float | None = 0.9

# Size cap on the C(union) bypass path. The bypass is only allowed
# when the merger's resulting tx count is at or below this threshold.
# Trusts gene-fit alone for small within-cell fragment consolidations
# (where ΔC has no headroom because both parents are at C ≈ 1.0) while
# requiring strong ΔC signal for any larger merger. Calibrated against
# the natural per-cell tx-count distribution in PDAC (50 tx ≈ between
# the 75th and 90th percentiles of natural cell sizes). None = no size
# cap on the bypass.
STITCH_C_UNION_BYPASS_MAX_N_TX: int | None = 50

# Stitch merger-tree depth cap. Per-component counter: each pre-stitch
# entity starts at depth 0; on union, the new root's depth becomes
# `max(child_depths) + 1`. A merger is blocked if either side has
# already reached the cap. Balanced N-entity merges cost log2(N) depth;
# chain merges cost N-1 — so the cap rewards balanced consolidations
# and penalises chain growth (one component repeatedly absorbing
# neighbours, the over-merge failure mode). None = off. Recommended 3.
STITCH_MAX_MERGER_DEPTH: int | None = 3

# Mid-pipeline QC (after Group, before Stitch). Two opt-in controls;
# both default off so current production behavior is unchanged.
#
#   MID_SPLIT_UNASSIGNED_DZ  None = off (current). Float (e.g. 2.0)
#                            applies sorted-Δz fragmentation to
#                            UNASSIGNED_<base> components from Group,
#                            mirroring Split-Phase1's logic. Group's
#                            spatial graph is essentially 2D, so its
#                            components routinely span > 2 µm in z.
#                            ROI data: 57 % of components have max-gap
#                            > 2 µm without this stage.
#
#   MID_QC_C_FLOOR           0.0 = off (current). Float (e.g. 0.05)
#                            demotes any entity (cell / partial /
#                            component) with coherence ≤ floor OR
#                            n_genes < 2. Catches incoherent Phase-1
#                            entities that survived size QC and any
#                            low-C Group components.
MID_SPLIT_UNASSIGNED_DZ: float | None = 2.0
MID_QC_C_FLOOR: float = 0.05

# "Real players" gate for Mid-QC coherence + Rescue veto.
#   0.0 = legacy n_finite-denominator (every non-NaN pair counts —
#         this conflates explicit zeros, tight_nulls, and dead_zones
#         with informative pairs, biasing C toward 0 on sparse W).
#   >0  = panel-shape-agnostic gate. Pairs with |W[i,j]| ≤ floor are
#         excluded from BOTH numerator and denominator; coherence
#         and rescue-veto reflect the (signed) strength of pairs
#         that actually carry information. Required for sparse
#         bootstrap / Visium HD panels where most off-diagonal cells
#         are implicit zero rather than NaN.
REAL_SIGNAL_THRESHOLD: float = 0.05
# Percentile of real-signal PMIs used in the Rescue mean/hybrid veto.
# 50 = median. <50 = stricter (more pairs must clear mean_threshold).
# >50 = liberal (tolerates a long left tail of weak/negative pairs).
# 2026-05-13: lowered 50 → 25 — pairs with stricter Rescue produced
# substantially cleaner low-coherence tail at zero retention cost in the
# pdac_pmi_sweep validation. See benchmarks/pdac_pmi_sweep/.
RESCUE_AGGREGATOR_PERCENTILE: float = 25.0

# Post-Group Rescue (between Group and Stitch). Closes the gap where
# Group's UNASSIGNED_* components — freshly created — cannot serve as
# Rescue targets in the main 3-pass Rescue (which runs BEFORE Group).
# Without this stage, tx that "belong" to a Group component sit as
# "-1" through Stitch (which then sees an incomplete component) and
# Demote (which may cull on size before the component is fully grown),
# only getting a chance at Final Rescue.
#
#   0 = off (current production behavior)
#  >0 = number of post-Group rescue passes; admits "-1" tx to BOTH
#        Phase-1 entities AND Group components via the same hybrid
#        veto used elsewhere. Same compute profile as the main Rescue.
RESCUE_POST_GROUP_PASSES: int = 3

# Phase-1 post-1c nuclear reassignment (opt-in). Gap B: a nuclear tx
# weakly admitted to the main seed via mean-PMI test, whose gene later
# turns out to be a STRONG fit to a Phase-1c-emitted partial's
# sub-seed, is currently locked in the main entity. This stage moves
# such tx to the partial sibling that gives strictly higher mean PMI.
#
#   False = off (legacy).
#   True  = enable post-1c reassignment.
# Promoted to default-on 2026-05-11 after numpy vectorization eliminated
# the wall-cost objection (3.1x full-pipeline speedup, byte-identical;
# see benchmarks/reassign_full_tissue_speedup.json). The cells-only
# coherence evidence (mean Δ +0.00316, win/loss 2070/954 on full tissue)
# already supported promotion; only the +250s wall cost was the blocker.
PHASE1_REASSIGN_AFTER_1C: bool = True

PHASE1_RERANK_ENABLED: bool = True     # rerank depth-1 entities — promoted to default-on 2026-05-13
                                        # under each parent by nuclear-tx
                                        # count. See
                                        # docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
PHASE1_RERANK_MARGIN_TX: int = 1

# The etype-aware path is now the only path (hardwired in commit
# 3fbcc67). Legacy regex-based _phase1_rerank_within_parent and
# _reassign_nuclear_post_1c functions remain in this module for
# back-compat with tests that import them directly; they're not
# called by run_segmented_pipeline anymore.


# Opt-in: replace Group's `annotate_unassigned_components_fast` (G=8 self,
# spatial-only connected components) with the density-cascade Phase 1 on
# the same post-Rescue residual. Cascade emits `cascade_<n>` synthetic
# anchors, classified as 'cell' (``_etype == "cell"``). Floor is selected at runtime
# from runtime tx-coverage in the residual pool — no hand-tuned thresholds.
#
# On 500 µm Xenium ROI, head-to-head:
#   default Group (G=8 self):     1,826 components,  ARI vs raw +0.6808
#   cascade-residual ('auto'):     ~602 components,  ARI vs raw +0.6877
#       (3x fewer components, ~99 % of default's assignment, equal-or-better ARI)
#
# See density-cascade-handoff.md for full design and bench results.
#
#   False = off (legacy: spatial-only annotate_unassigned_components_fast).
#   True  = on (default 2026-05-07 onward): cascade as Group replacement.
#
# Default flip rationale (full-tissue Xenium, /tmp/bench_cascade_step6.py):
#   default Group  : V_β=2 +0.9410, ARI +0.6896, 62,980 entities
#   cascade partial: V_β=2 +0.9435, ARI +0.6987, 59,207 entities (closest
#                    to input cell_id's 58,405).
# Cascade emits `cascade_<n>-1` labels; downstream Stitch merges fragments
# of the same biological cell via the existing two-dash partial-merger logic.
PHASE1_SEG_RESIDUAL_CASCADE: bool = True
# Cascade auto-floor target: fraction of residual tx mass to capture in
# the R=1 Moore-dilated anchor mask. 0.65 was reverse-engineered from the
# empirical NOSEG winner (66.5 % cov at floor=4 on full Xenium tissue) and
# also lands on floor=2 on the SEG-residual ROI (which can't reach 65 %
# coverage at any floor, so falls back to hard_min=2).
PHASE1_SEG_RESIDUAL_CASCADE_TARGET_COV: float = 0.65
PHASE1_SEG_RESIDUAL_CASCADE_HARD_MIN: int = 2


# Replace the NOSEG path's Group call (de-facto cell-finder, not
# residual handler) with the density-cascade. NOSEG runs at higher density
# than SEG-residual since the entire pool is the input — auto-floor will
# typically pick floor=4 (~66 % tx coverage) on Xenium full pool.
#
# See density-cascade-handoff.md for ROI ARI improvements (+0.085 over
# baseline NOSEG) and full-tissue homogeneity bench results.
#
#   False = off (legacy: G=8 self spatial-only annotate_unassigned_
#           components_fast; bin restriction acts as a crude
#           anchoring substitute).
#   True  = on (default 2026-05-09 onward): cascade as Phase-1
#           replacement in the NOSEG path. Promotes cascade to default
#           symmetrically with PHASE1_SEG_RESIDUAL_CASCADE (flipped
#           2026-05-07). Test references that pinned legacy NOSEG
#           entity counts will need refresh.
PHASE1_NOSEG_CASCADE: bool = True
PHASE1_NOSEG_CASCADE_TARGET_COV: float = 0.65
PHASE1_NOSEG_CASCADE_HARD_MIN: int = 2


def _state_dict(df: pd.DataFrame, col: str) -> dict[str, int]:
    """Stage-snapshot accounting. Reads the upstream-emitted ``_etype``
    column when present (correct on FFPE cell_ids); otherwise falls
    back to the canonical label parser :func:`tracer._etype.infer_etype_from_label`.
    """
    from tracer._etype import infer_etype_from_label

    s = df[col].astype(str)
    if "_etype" in df.columns:
        etypes = df["_etype"].astype(str)
    else:
        etypes = pd.Series(
            np.asarray(infer_etype_from_label(s)).astype(str),
            index=df.index,
        )
    # Map etype categories → snapshot keys. "unknown" and "drop" both
    # collapse to "unassigned" (matches the legacy `_classify` schema).
    bucket = etypes.where(
        etypes.isin(["cell", "partial", "component"]),
        other="unassigned",
    )
    n_ent = s.groupby(bucket).nunique().to_dict()
    n_tx = bucket.value_counts().to_dict()
    return {
        "n_cells": int(n_ent.get("cell", 0)),
        "n_partials": int(n_ent.get("partial", 0)),
        "n_components": int(n_ent.get("component", 0)),
        "n_unassigned_tx": int(n_tx.get("unassigned", 0)),
    }


def _record_stage(progression: list, stage_name: str, df: pd.DataFrame, col: str):
    """Append a stage snapshot. Records wall-clock elapsed since the
    previous _record_stage call so callers can see which stage dominates.

    When the env var ``TRACER_STAGE_VERBOSE`` is set (any truthy value),
    prints a one-line summary as each stage completes — useful for live
    progress on long-running benches. Workers inherit the env var
    through fork/spawn so process-pool runs are also chatty.
    """
    import os as _os
    import time as _t
    now = _t.time()
    prev_ts = progression[-1]["_ts"] if progression else None
    stage_seconds = (
        round(now - prev_ts, 3) if prev_ts is not None else None
    )
    entry = {
        "stage": stage_name,
        "_ts": now,
        "stage_seconds": stage_seconds,
        **_state_dict(df, col),
    }
    progression.append(entry)
    if _os.environ.get("TRACER_STAGE_VERBOSE"):
        tag = _os.environ.get("TRACER_STAGE_TAG", "")
        prefix = f"[stage {tag}]" if tag else "[stage]"
        secs_str = (
            f"{stage_seconds:>7.2f}s" if stage_seconds is not None
            else f"{'(t0)':>8s}"
        )
        print(
            f"{prefix} {stage_name:<22s} "
            f"{secs_str}  "
            f"cells={entry.get('n_cells', 0):>7,} "
            f"partials={entry.get('n_partials', 0):>7,} "
            f"unassigned_tx={entry.get('n_unassigned_tx', 0):>9,}",
            flush=True,
        )


# Intentionally a superset of the finalize sentinel: adds `__GUARD_SKIP__`
# (`prune_rejected` is already collapsed to `UNASSIGNED` by
# `finalize_unassigned`, which runs before this helper).
_OUTPUT_UNASSIGNED_TOKENS = frozenset(
    {"UNASSIGNED", "-1", "nan", "DROP", "group_rejected",
     "demote_rejected", "__GUARD_SKIP__"}
)


def _canonicalize_output(df, input_cell_id, *,
                         entity_col="stitched", txid_col="transcript_id"):
    """tracer_id <- final entity label (unassigned tokens -> '-1', exact match
    so UNASSIGNED_<n> components are preserved); drop entity_col; restore
    pristine cell_id from input_cell_id (Series indexed by transcript_id).
    Identity only — type lives in _etype."""
    df = df.copy()
    final = df[entity_col].astype(str).to_numpy()
    is_un = np.isin(final, list(_OUTPUT_UNASSIGNED_TOKENS))
    df["tracer_id"] = np.where(is_un, "-1", final)
    df = df.drop(columns=[entity_col])
    df["cell_id"] = df[txid_col].map(input_cell_id)
    return df


def _grid_3d_graph_fn(df_in, *, k=None, dist_threshold=None,
                      coord_cols=("x", "y", "z"),
                      G_z=2.0, z_neighborhood_depth=1):
    return build_grid_graph_xyz(
        df_in, k=k, dist_threshold=dist_threshold, coord_cols=coord_cols,
        G_xy=2.0, G_z=G_z, xy_neighborhood="8",
        z_neighborhood_depth=z_neighborhood_depth,
        exact_distance_filter=False,
    )


def _grid_self_graph_fn(df_in, *, k=None, dist_threshold=None,
                        coord_cols=("x", "y", "z")):
    return build_grid_graph_xy(
        df_in, k=k, dist_threshold=dist_threshold or 1.5, coord_cols=coord_cols,
        G=8.0, neighborhood="self", exact_distance_filter=False,
    )


def _resolve_pipeline_cfg(cfg):
    """Resolve a `PipelineConfig` for the runner — Phase B of config
    migration.

    When ``cfg`` is provided, it is returned as-is. When ``cfg is None``
    (the legacy call path), a `PipelineConfig` is **built from the
    current module-global constants** so that every cfg consumer below
    sees today's production values bit-for-bit. The module globals are
    read at call time so test monkey-patches still take effect.

    Determinism guarantee (verified by the determinism bench in
    ``benchmarks/bench_pdac_roi_seed_determinism.py``):
    ``run_*_pipeline(df, panel, cfg=None)`` produces a partition
    bit-identical to ``run_*_pipeline(df, panel, cfg=load_config())``.
    Knobs not yet represented in `PipelineConfig` (Phase-1 prune,
    Group, post-group-pass-count, Mid-QC) remain read from the module
    globals at their call sites — wire them in subsequent Phase-B
    increments.
    """
    if cfg is not None:
        return cfg
    from tracer.config import (
        DemoteConfig,
        PipelineConfig,
        RescueConfig,
        StitchConfig,
    )
    rescue_kwargs = dict(
        veto_mode=RESCUE_VETO_MODE,
        min_admit_threshold=RESCUE_MIN_ADMIT,
        mean_admit_threshold=RESCUE_MEAN_ADMIT,
        neg_threshold=RESCUE_NEG_THR,
        max_passes=RESCUE_MAX_PASSES,
        bin_size_um=2.0,
        z_bound_um=None,
        cluster_guard_n=3,
        small_entity_guard_n=0,
        aggregator_percentile=RESCUE_AGGREGATOR_PERCENTILE,
        real_signal_threshold=REAL_SIGNAL_THRESHOLD,
        post_group_passes=RESCUE_POST_GROUP_PASSES,
    )
    rescue_cfg = RescueConfig(**rescue_kwargs)
    # Final Rescue: SEG-friendly default of 3 passes (matches the
    # promoted 2026-05-15 setting). NOSEG users load via
    # `load_config(platform="noseg")` to get 5 passes instead.
    final_rescue_kwargs = {
        **rescue_kwargs,
        "max_passes": 3,
    }
    final_rescue_cfg = RescueConfig(**final_rescue_kwargs)
    stitch_cfg = StitchConfig(
        mode="count", metric="pmi", penalize_simplicity=True,
        deltaC_min=0.03,
        c_union_bypass=STITCH_C_UNION_BYPASS,
        c_union_bypass_max_n_tx=STITCH_C_UNION_BYPASS_MAX_N_TX,
        max_merger_depth=STITCH_MAX_MERGER_DEPTH,
        candidate_source="grid",
        bin_size_um=2.0,
        g_z_um=STITCH_GZ_UM,
        z_neighbor_depth=1,
        neighborhood="8",
        dist_threshold_um=5.0,
        min_local_tx_per_entity=STITCH_MIN_LOCAL_TX,
    )
    demote_cfg = DemoteConfig(min_size=5)
    return PipelineConfig(
        rescue=rescue_cfg,
        stitch=stitch_cfg,
        demote=demote_cfg,
        final_rescue=final_rescue_cfg,
    )


def run_segmented_pipeline(df: pd.DataFrame,
                           npmi_panel: pd.DataFrame,
                           cfg=None,
                           ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the segmented workflow on ``df`` (must have ``cell_id`` set).

    Parameters
    ----------
    df, npmi_panel
        Input transcripts (with ``cell_id``) and bootstrap PMI panel.
    cfg : PipelineConfig | None, optional
        Phase-B config. When ``None`` (default), a `PipelineConfig` is
        built from the current module-global constants so today's
        production behavior is preserved bit-exactly (see
        `_resolve_pipeline_cfg`). When provided, drives the Rescue,
        Stitch, Demote, and Final Rescue call sites. Knobs not yet in
        `PipelineConfig` (Phase-1 prune, Group, post-group-pass-count)
        continue to read from module globals.

    Returns
    -------
    df_final : DataFrame with ``stitched`` column carrying the final per-tx label.
    stage_progression : list of state dicts, one per stage.
    """
    cfg = _resolve_pipeline_cfg(cfg)
    _input_cell_id = pd.Series(
        df["cell_id"].astype(str).to_numpy(),
        index=df["transcript_id"].to_numpy(),
    )
    if not _input_cell_id.index.is_unique:
        raise ValueError("transcript_id must be unique to restore pristine cell_id")
    progression: list[dict[str, Any]] = []
    _record_stage(progression, "input", df.assign(_lbl=df["cell_id"].astype(str)), "_lbl")

    # Auto-derive within-cell |Δz| threshold + recommended G_z. The
    # recommended_G_z is bimodality-aware:
    #   - bimodal data (z-stratified, Cohen's d ≥ 3): G_z = floor(thr),
    #     small enough to leave an empty-bin moat between the within-
    #     cell mode and the cross-stratum mode.
    #   - unimodal data: G_z = ceil(thr), wide enough to admit cell-
    #     spanning merges at depth=1.
    # The same G_z drives both Split's grid (where depth=1 is fixed)
    # and Stitch's grid + Δz guard.
    dz_stats = estimate_within_cell_dz_threshold(df, entity_col="cell_id")
    auto_dz = dz_stats["threshold"]
    auto_Gz = dz_stats.get("recommended_G_z", float("nan"))
    if not np.isfinite(auto_dz):
        auto_dz, auto_n = None, 0
        auto_Gz = 1.0
    if not np.isfinite(auto_Gz):
        auto_Gz = 1.0

    # Stage 1 — Prune (nuclear-seed when overlaps_nucleus is available).
    # Use PMI column when available; nuclear-seed identity prune
    # anchors each cell on its compact nucleus, then admits cytoplasmic
    # tx via mean PMI to the seed. Recursive Phase 1c surfaces
    # secondary modules as partials. Fall back to NPMI/whole-cell prune
    # if the panel lacks a PMI column or the input has no nuclear flag
    # (legacy synthetic panels).
    metric_col = "PMI" if "PMI" in npmi_panel.columns else "NPMI"
    # Phase 1b admission gate — pulled from cfg.phase1 when available.
    # Defaults preserve the legacy "mean" gate bit-exactly.
    _p1 = getattr(cfg, "phase1", None)
    _p1_veto_mode = getattr(_p1, "veto_mode", "mean") if _p1 is not None else "mean"
    _p1_mean_admit = getattr(_p1, "mean_admit_threshold", 0.2) if _p1 is not None else 0.2
    _p1_min_admit = getattr(_p1, "min_admit_threshold", 0.0) if _p1 is not None else 0.0
    _p1_agg_pct = getattr(_p1, "aggregator_percentile", 25.0) if _p1 is not None else 25.0
    _p1_rs_thr = getattr(_p1, "real_signal_threshold", 0.05) if _p1 is not None else 0.05
    _p1_neg_thr = getattr(_p1, "neg_npmi_threshold", -0.2) if _p1 is not None else -0.2
    if "overlaps_nucleus" in df.columns:
        df_pruned, aux = prune_transcripts_nuclear_seed(
            df, npmi_panel,
            cell_id_col="cell_id", gene_col="feature_name",
            nuclear_col="overlaps_nucleus",
            threshold=PMI_THR, unassigned_id="-1",
            metric_col=metric_col, nan_fill=0.0,
            min_nuclear_genes=3,
            seed_coherence_floor=SEED_COHERENCE_FLOOR,
            nuclear_only_admit=NUCLEAR_ONLY_ADMIT,
            fallback_whole_cell_admit=FALLBACK_WHOLE_CELL_ADMIT,
            tx_weighted=TX_WEIGHTED_PRUNE,
            veto_mode=_p1_veto_mode,
            mean_admit_threshold=_p1_mean_admit,
            min_admit_threshold=_p1_min_admit,
            aggregator_percentile=_p1_agg_pct,
            real_signal_threshold=_p1_rs_thr,
            neg_npmi_threshold=_p1_neg_thr,
            n_jobs=-1, show_progress=False,
        )
    else:
        df_pruned, aux = prune_transcripts_fast(
            df, npmi_panel,
            cell_id_col="cell_id", gene_col="feature_name",
            threshold=PMI_THR, unassigned_id="-1",
            metric_col=metric_col, nan_fill=0.0,
            n_jobs=-1, show_progress=False,
        )
    _record_stage(progression, "Prune", df_pruned, "tracer_id")

    # Phase-1 post-1c nuclear reassignment (opt-in). Closes Gap B:
    # nuclear tx weakly admitted to the main seed, whose gene fits a
    # 1c partial's sub-seed strictly better, get moved to the partial.
    if PHASE1_REASSIGN_AFTER_1C and "overlaps_nucleus" in df_pruned.columns:
        df_pruned, _reassign_stats = _reassign_nuclear_post_1c_etype(
            df_pruned, entity_col="tracer_id", aux=aux,
            cell_id_col="cell_id", gene_col="feature_name",
            nuclear_col="overlaps_nucleus",
            margin=0.05, threshold=PMI_THR,
        )
        _record_stage(progression, "Phase1-Reassign-1c", df_pruned, "tracer_id")

    # Spatial-split Phase 1 entities. Phase 1c is purely gene-based —
    # if a cell has TWO contamination sources contributing similar
    # gene programs (e.g. lymphoid tx from cells above AND below the
    # target cell in z), Phase 1c emits one merged partial. Split it
    # into spatially-distinct sub-partials so downstream Stitch sees
    # them as separate entities.
    df_pruned, _split_stats = _spatial_split_phase1_entities(
        df_pruned, entity_col="tracer_id",
        coord_cols=("x", "y", "z"),
        dz_threshold=SPLIT_PHASE1_DZ,
        min_size=SPLIT_PHASE1_MIN_TX,
        min_entity_size=SPLIT_PHASE1_MIN_ENTITY,
        unassigned_id="-1",
    )
    _record_stage(progression, "Split-Phase1", df_pruned, "tracer_id")

    # Phase1-Rerank (opt-in): within each parent cell_id, promote the
    # depth-1 entity with the most nuclear tx to the main `{cell_id}`
    # label. Defuses Phase 1's greedy 1a->1b->1c privilege.
    # Only runs when overlaps_nucleus is present (nuclear-seed prune path).
    if PHASE1_RERANK_ENABLED and "overlaps_nucleus" in df_pruned.columns:
        df_pruned, _rerank_stats = _phase1_rerank_within_parent_etype(
            df_pruned, entity_col="tracer_id",
            cell_id_col="cell_id",
            nuclear_col="overlaps_nucleus",
            margin_tx=PHASE1_RERANK_MARGIN_TX,
        )
        _record_stage(progression, "Phase1-Rerank", df_pruned, "tracer_id")

    # Post-split QC: demote tiny Phase 1 entities (1-2 tx) so they
    # don't act as degenerate routing anchors in Rescue.
    df_pruned, _qc_stats = _qc_demote_small_phase1_entities(
        df_pruned, entity_col="tracer_id",
        min_size=PHASE1_QC_MIN_TX,
        unassigned_id="-1",
    )
    _record_stage(progression, "Phase1-QC", df_pruned, "tracer_id")

    # Phase-1-time Mahalanobis-gated remerge (opt-in). Sibling of the
    # Stitch-time rescue applied one stage earlier so EMT-like
    # over-splits collapse before Rescue/Group/Stitch see them. No-op
    # when ``cfg.phase1.maha_remerge_d is None`` (default).
    _p1_maha_d = getattr(_p1, "maha_remerge_d", None) if _p1 is not None else None
    if _p1_maha_d is not None:
        from tracer.phase1_rescue import phase1_maha_remerge
        _p1_maha_floor = float(
            getattr(_p1, "maha_remerge_delta_c_floor", -0.2)
        )
        df_pruned, _p1_resc_stats = phase1_maha_remerge(
            df_pruned, npmi_panel,
            threshold=float(_p1_maha_d),
            floor=_p1_maha_floor,
            cfg=cfg,
            entity_col="tracer_id",
            gene_col="feature_name",
            verbose=False,
        )
        _record_stage(progression, "Phase1-Maha-Remerge", df_pruned, "tracer_id")

    # Split stage REMOVED. The nuclear-seed prune emits spatially
    # compact entities by construction (anchored on the nucleus), so
    # there are no spatially-disconnected gene clusters for Split to
    # fragment. For legacy whole-cell prune (no overlaps_nucleus), we
    # still run Split as a safety net.
    if "overlaps_nucleus" not in df.columns:
        def _split_graph_fn(df_in, *, k=None, dist_threshold=None,
                            coord_cols=("x", "y", "z")):
            return _grid_3d_graph_fn(df_in, k=k, dist_threshold=dist_threshold,
                                     coord_cols=coord_cols,
                                     G_z=auto_Gz, z_neighborhood_depth=1)
        df_pruned = enforce_spatial_coherence_fast(
            df_stitched=df_pruned, build_graph_fn=_split_graph_fn,
            entity_col="tracer_id", coord_cols=("x", "y", "z"),
            k=5, dist_threshold=5.0,
            out_col="tracer_id", show_progress=False,
        )
        _record_stage(progression, "Split", df_pruned, "tracer_id")

    # Prune → Rescue → Group → Stitch (re-ordered for nuclear-only
    # Phase 1). Rationale: under nuclear-only admission, ~50% of tx
    # exit Phase 1 unassigned — most are cyto tx of cells that already
    # have a nuclear-anchored seed. Running Rescue first lets those tx
    # attach to nearby seeded entities before Group clusters the
    # residual into orphan UNASSIGNED_* components.
    df_rescued = df_pruned
    n_rescued = 0
    for _pass in range(cfg.rescue.max_passes):
        df_rescued, n_pass_rescued, _, _ = guarded_rescue(
            df_rescued, aux=aux,
            entity_col="tracer_id", gene_col="feature_name",
            coord_cols=("x", "y", "z"), out_col="tracer_id",
            G=cfg.rescue.bin_size_um,
            pos_npmi_threshold=PMI_THR,
            neg_npmi_threshold=cfg.rescue.neg_threshold,
            cluster_guard_n=cfg.rescue.cluster_guard_n,
            veto_mode=cfg.rescue.veto_mode,
            mean_threshold=cfg.rescue.mean_admit_threshold,
            min_admit_threshold=cfg.rescue.min_admit_threshold,
            small_entity_guard_n=cfg.rescue.small_entity_guard_n,
            real_signal_threshold=cfg.rescue.real_signal_threshold,
            aggregator_percentile=cfg.rescue.aggregator_percentile,
            rank_policy=cfg.rescue.rank_policy,
            witness_min_admit=cfg.rescue.witness_min_admit,
            witness_cap=cfg.rescue.witness_cap,
            witness_small_component_cap_divisor=cfg.rescue.witness_small_component_cap_divisor,
            witness_tiebreak=cfg.rescue.witness_tiebreak,
            offpanel_first_entity=cfg.rescue.offpanel_first_entity,
        )
        n_rescued += n_pass_rescued
        if n_pass_rescued == 0:
            break
    _record_stage(progression, "Rescue", df_rescued, "tracer_id")

    if PHASE1_SEG_RESIDUAL_CASCADE:
        df_grouped = cascade_as_residual_handler(
            df_pruned=df_rescued, aux=aux,
            entity_col="tracer_id",
            G=2.0, thresholds="auto",
            territory_radius_bins=1,
            pmi_threshold=PMI_THR,
            min_anchor_tx=3,
            auto_target_cov=PHASE1_SEG_RESIDUAL_CASCADE_TARGET_COV,
            auto_hard_min=PHASE1_SEG_RESIDUAL_CASCADE_HARD_MIN,
        )
    else:
        df_grouped = annotate_unassigned_components_fast(
            df_pruned=df_rescued, aux=aux,
            build_graph_fn=_grid_self_graph_fn, prune_fn=prune_genes_by_pmi_greedy,
            coord_cols=("x", "y", "z"),
            k=8, dist_threshold=1.5, min_comp_size=4,
            npmi_threshold=ANNOTATE_NEG_THR,
            entity_col="tracer_id", out_col="tracer_id",
            cell_id_col="cell_id", gene_col="feature_name",
            transcript_id_col="transcript_id", show_progress=False,
        )
    _record_stage(progression, "Group", df_grouped, "tracer_id")

    # Mid-pipeline QC (after Group, before Stitch). Both stages are
    # opt-in via constants at the top of this module.
    mid_did_anything = False
    if MID_SPLIT_UNASSIGNED_DZ is not None:
        df_grouped, _ = _split_unassigned_components(
            df_grouped, entity_col="tracer_id", coord_cols=("x", "y", "z"),
            dz_threshold=float(MID_SPLIT_UNASSIGNED_DZ),
            min_size=1, min_entity_size=2, unassigned_id="-1",
        )
        mid_did_anything = True
    if MID_QC_C_FLOOR > 0:
        df_grouped, _ = _qc_demote_low_coherence(
            df_grouped, entity_col="tracer_id", aux=aux,
            min_C=float(MID_QC_C_FLOOR), min_n_genes=2,
            threshold=PMI_THR, metric="pmi", unassigned_id="-1",
            # rst defaults to threshold (tau): informative-edges denominator,
            # panel-shape-agnostic. Decoupled from the Rescue veto's
            # REAL_SIGNAL_THRESHOLD (0.05), which stays as-is.
        )
        mid_did_anything = True
    if mid_did_anything:
        _record_stage(progression, "Mid-QC", df_grouped, "tracer_id")

    # Post-Group Rescue (opt-in). Admits any remaining "-1" tx to
    # Phase-1 entities AND Group components — closing the gap where
    # Group's UNASSIGNED_* couldn't be Rescue targets in the main pass.
    if cfg.rescue.post_group_passes > 0:
        for _pass in range(cfg.rescue.post_group_passes):
            df_grouped, n_pass_rescued, _, _ = guarded_rescue(
                df_grouped, aux=aux,
                entity_col="tracer_id", gene_col="feature_name",
                coord_cols=("x", "y", "z"), out_col="tracer_id",
                G=cfg.rescue.bin_size_um,
                pos_npmi_threshold=PMI_THR,
                neg_npmi_threshold=cfg.rescue.neg_threshold,
                cluster_guard_n=cfg.rescue.cluster_guard_n,
                veto_mode=cfg.rescue.veto_mode,
                mean_threshold=cfg.rescue.mean_admit_threshold,
                min_admit_threshold=cfg.rescue.min_admit_threshold,
                small_entity_guard_n=cfg.rescue.small_entity_guard_n,
                real_signal_threshold=cfg.rescue.real_signal_threshold,
                aggregator_percentile=cfg.rescue.aggregator_percentile,
                rank_policy=cfg.rescue.rank_policy,
                witness_min_admit=cfg.rescue.witness_min_admit,
                witness_cap=cfg.rescue.witness_cap,
                witness_small_component_cap_divisor=cfg.rescue.witness_small_component_cap_divisor,
                witness_tiebreak=cfg.rescue.witness_tiebreak,
                offpanel_first_entity=cfg.rescue.offpanel_first_entity,
            )
            if n_pass_rescued == 0:
                break
        _record_stage(progression, "Post-Group Rescue", df_grouped, "tracer_id")

    # Stitch — symmetric kwargs with NOSEG path (close-edges guard removed
    # 2026-05-09: empirical bench on 500 µm Xenium ROI showed the guard is
    # dormant under current production constants — flipping it on/off in
    # SEG / NOSEG produced byte-identical Stitch output. Removing for
    # symmetry; auto_dz is still computed but no longer fed in here.
    # entity_col reads `tracer_id` directly (the active partition column
    # after Post-Group Rescue); previous code aliased this as
    # "post_stage4", a stale name from the numbered-stages era.
    _stitch_g_z, _stitch_z_depth = _resolve_stitch_g_z(
        df_grouped, cfg, auto_Gz, coord_cols=("x", "y", "z"))
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_grouped, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode=cfg.stitch.mode, threshold=PMI_THR, metric=cfg.stitch.metric,
        penalize_simplicity=cfg.stitch.penalize_simplicity,
        deltaC_min=cfg.stitch.deltaC_min,
        c_union_bypass=cfg.stitch.c_union_bypass,
        c_union_bypass_max_n_tx=cfg.stitch.c_union_bypass_max_n_tx,
        max_merger_depth=cfg.stitch.max_merger_depth,
        dist_threshold=cfg.stitch.dist_threshold_um,
        out_col="stitched", show_progress=False,
        candidate_source=cfg.stitch.candidate_source,
        G=cfg.stitch.bin_size_um,
        stitch_neighborhood=cfg.stitch.neighborhood,
        G_z=_stitch_g_z,
        z_neighbor_depth=_stitch_z_depth,
        min_local_tx_per_entity=cfg.stitch.min_local_tx_per_entity,
        mahalanobis_d_rescue=cfg.stitch.mahalanobis_d_rescue,
        rescue_delta_c_floor=cfg.stitch.rescue_delta_c_floor,
    )
    _record_stage(progression, "Stitch", df_stitched, "stitched")

    # Demote
    df_stitched, n_demoted = demote_small_entities(
        df_stitched, entity_col="stitched", out_col="stitched",
        min_size=cfg.demote.min_size, unassigned_label="-1",
    )
    _record_stage(progression, "Demote", df_stitched, "stitched")

    # Final Rescue — iterative with two exit conditions:
    #   (a) hard ceiling at cfg.final_rescue.max_passes
    #   (b) convergence gate at early_exit_admit_ratio (if > 0): break
    #       when a pass admits fewer than ratio * pre-pass-pool tx.
    for _pass in range(cfg.final_rescue.max_passes):
        df_stitched, n_reassigned, stats = reassign_unassigned_grid_pool(
            df_stitched, aux=aux,
            entity_col="stitched", gene_col="feature_name",
            coord_cols=("x", "y", "z"), out_col="stitched",
            G=cfg.final_rescue.bin_size_um,
            pos_npmi_threshold=PMI_THR,
            neg_npmi_threshold=cfg.final_rescue.neg_threshold,
            only_partial_component=False,
            veto_mode=cfg.final_rescue.veto_mode,
            mean_threshold=cfg.final_rescue.mean_admit_threshold,
            min_admit_threshold=cfg.final_rescue.min_admit_threshold,
            small_entity_guard_n=cfg.final_rescue.small_entity_guard_n,
            real_signal_threshold=cfg.final_rescue.real_signal_threshold,
            aggregator_percentile=cfg.final_rescue.aggregator_percentile,
            rank_policy=cfg.final_rescue.rank_policy,
            witness_min_admit=cfg.final_rescue.witness_min_admit,
            witness_cap=cfg.final_rescue.witness_cap,
            witness_small_component_cap_divisor=cfg.final_rescue.witness_small_component_cap_divisor,
            witness_tiebreak=cfg.final_rescue.witness_tiebreak,
            offpanel_first_entity=cfg.final_rescue.offpanel_first_entity,
        )
        if n_reassigned == 0:
            break
        # Convergence gate (no-op when early_exit_admit_ratio == 0.0).
        n_un_before = int(stats.get("total_unassigned", 0))
        if (cfg.final_rescue.early_exit_admit_ratio > 0.0
                and n_un_before > 0
                and (n_reassigned / n_un_before
                     < cfg.final_rescue.early_exit_admit_ratio)):
            break
    _record_stage(progression, "Final Rescue", df_stitched, "stitched")

    # Finalize: collapse all stage-rejected / sentinel labels in the
    # entity column to a single canonical "DROP". Mid-pipeline labels
    # like "group_rejected", "demote_rejected", "-1", "UNASSIGNED",
    # "nan" all become "DROP" — published output has exactly two label
    # categories: real entity IDs (cell/partial/component) and "DROP".
    # Diagnostic info (which stage rejected each tx) is recoverable
    # via the per-stage progression snapshots and the
    # `unassigned_qc_status` column emitted by Group.
    finalize_unassigned(df_stitched, col="stitched")
    _record_stage(progression, "Finalize", df_stitched, "stitched")

    df_stitched = _canonicalize_output(df_stitched, _input_cell_id)

    return df_stitched, progression


def run_noseg_pipeline(df: pd.DataFrame, npmi_panel: pd.DataFrame,
                       cfg=None,
                       ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the no-segmentation workflow on ``df``.

    The input ``cell_id`` column is overwritten to ``"-1"`` everywhere
    (so any prior segmentation is discarded). Stages: Group → Stitch →
    Demote → Final Rescue.

    Accepts xy-grid-only input (Visium HD or any 2D imaging modality):
    when the ``z`` column is absent it is synthesised as ``0`` so the
    3D-aware stages (which expect ``coord_cols=("x", "y", "z")``) run
    unchanged. With z constant, ``z_neighborhood_depth`` is a no-op —
    every transcript lands in the same z-bin and any depth ≥ 0 admits
    every candidate pair.

    Parameters
    ----------
    df, npmi_panel
        Input transcripts and bootstrap PMI panel.
    cfg : PipelineConfig | None, optional
        Phase-B config. See `run_segmented_pipeline` for semantics —
        when ``None`` (default) a config is built from module globals
        to preserve today's behavior bit-exactly.
    """
    cfg = _resolve_pipeline_cfg(cfg)
    df = df.copy()
    _input_cell_id = pd.Series(
        df["cell_id"].astype(str).to_numpy() if "cell_id" in df.columns
        else np.full(len(df), "-1"),
        index=df["transcript_id"].to_numpy(),
    )
    if not _input_cell_id.index.is_unique:
        raise ValueError("transcript_id must be unique to restore pristine cell_id")
    if "z" not in df.columns:
        df["z"] = 0.0
    df["cell_id"] = "-1"

    progression: list[dict[str, Any]] = []
    _record_stage(progression, "input (cell_id all -1)",
                  df.assign(_lbl=df["cell_id"].astype(str)), "_lbl")

    # Init aux via Stage 1 prune at -inf threshold (no actual pruning).
    # PMI is the default/production panel column; fall back to NPMI only for
    # legacy panels lacking a PMI column (mirrors run_segmented_pipeline).
    metric_col = "PMI" if "PMI" in npmi_panel.columns else "NPMI"
    df, aux = prune_transcripts_fast(
        df, npmi_panel,
        cell_id_col="cell_id", gene_col="feature_name",
        threshold=-1e9, unassigned_id="-1",
        metric_col=metric_col,
        n_jobs=-1, show_progress=False,
    )
    df["tracer_id"] = "-1"
    _record_stage(progression, "after init", df, "tracer_id")

    # Group (or cascade replacement for cell-finding in NOSEG).
    # Cascade grid/threshold params come from cfg.group when present
    # (so the grouping grid can scale to the input bin pitch — e.g. 8 um
    # VisiumHD bins need an 8 um grid, not the 2 um default). The cfg
    # defaults equal the historical literals below, so the cfg=None /
    # bundled-defaults path is byte-for-byte unchanged.
    grp = getattr(cfg, "group", None)
    if PHASE1_NOSEG_CASCADE:
        df_grouped = cascade_as_residual_handler(
            df_pruned=df, aux=aux,
            entity_col="tracer_id",
            G=(grp.cascade_bin_size_um if grp is not None else 2.0),
            thresholds="auto",
            territory_radius_bins=(grp.cascade_territory_radius_bins if grp is not None else 1),
            pmi_threshold=(grp.cascade_pmi_threshold if grp is not None else PMI_THR),
            min_anchor_tx=(grp.cascade_min_anchor_tx if grp is not None else 3),
            auto_target_cov=(grp.noseg_cascade_target_cov if grp is not None else PHASE1_NOSEG_CASCADE_TARGET_COV),
            auto_hard_min=(grp.noseg_cascade_hard_min if grp is not None else PHASE1_NOSEG_CASCADE_HARD_MIN),
        )
    else:
        df_grouped = annotate_unassigned_components_fast(
            df_pruned=df, aux=aux,
            build_graph_fn=_grid_self_graph_fn, prune_fn=prune_genes_by_pmi_greedy,
            coord_cols=("x", "y", "z"),
            k=8, dist_threshold=1.5, min_comp_size=5,
            npmi_threshold=ANNOTATE_NEG_THR,
            entity_col="tracer_id", out_col="tracer_id",
            cell_id_col="cell_id", gene_col="feature_name",
            transcript_id_col="transcript_id", show_progress=False,
        )
    _record_stage(progression, "Group", df_grouped, "tracer_id")

    # Mid-pipeline QC (after Group, before Stitch). Same opt-in
    # knobs as the segmented path.
    mid_did_anything = False
    if MID_SPLIT_UNASSIGNED_DZ is not None:
        df_grouped, _ = _split_unassigned_components(
            df_grouped, entity_col="tracer_id", coord_cols=("x", "y", "z"),
            dz_threshold=float(MID_SPLIT_UNASSIGNED_DZ),
            min_size=1, min_entity_size=2, unassigned_id="-1",
        )
        mid_did_anything = True
    if MID_QC_C_FLOOR > 0:
        df_grouped, _ = _qc_demote_low_coherence(
            df_grouped, entity_col="tracer_id", aux=aux,
            min_C=float(MID_QC_C_FLOOR), min_n_genes=2,
            threshold=PMI_THR, metric="pmi", unassigned_id="-1",
            # rst defaults to threshold (tau): informative-edges denominator,
            # panel-shape-agnostic. Decoupled from the Rescue veto's
            # REAL_SIGNAL_THRESHOLD (0.05), which stays as-is.
        )
        mid_did_anything = True
    if mid_did_anything:
        _record_stage(progression, "Mid-QC", df_grouped, "tracer_id")

    # Post-Group Rescue (opt-in) — see segmented runner for rationale.
    if cfg.rescue.post_group_passes > 0:
        for _pass in range(cfg.rescue.post_group_passes):
            df_grouped, n_pass_rescued, _, _ = guarded_rescue(
                df_grouped, aux=aux,
                entity_col="tracer_id", gene_col="feature_name",
                coord_cols=("x", "y", "z"), out_col="tracer_id",
                G=cfg.rescue.bin_size_um,
                pos_npmi_threshold=PMI_THR,
                neg_npmi_threshold=cfg.rescue.neg_threshold,
                cluster_guard_n=cfg.rescue.cluster_guard_n,
                veto_mode=cfg.rescue.veto_mode,
                mean_threshold=cfg.rescue.mean_admit_threshold,
                min_admit_threshold=cfg.rescue.min_admit_threshold,
                small_entity_guard_n=cfg.rescue.small_entity_guard_n,
                real_signal_threshold=cfg.rescue.real_signal_threshold,
                aggregator_percentile=cfg.rescue.aggregator_percentile,
                rank_policy=cfg.rescue.rank_policy,
                witness_min_admit=cfg.rescue.witness_min_admit,
                witness_cap=cfg.rescue.witness_cap,
                witness_small_component_cap_divisor=cfg.rescue.witness_small_component_cap_divisor,
                witness_tiebreak=cfg.rescue.witness_tiebreak,
                offpanel_first_entity=cfg.rescue.offpanel_first_entity,
            )
            if n_pass_rescued == 0:
                break
        _record_stage(progression, "Post-Group Rescue", df_grouped, "tracer_id")

    # Stitch — entity_col reads `tracer_id` directly (was aliased as
    # "post_stage4" — a stale name from the numbered-stages era).
    # NOSEG has no within-cell prior, so the legacy auto_Gz estimator is
    # unavailable; the resolver falls back to 1.0 µm when g_z_um is null.
    _stitch_g_z, _stitch_z_depth = _resolve_stitch_g_z(
        df_grouped, cfg, auto_Gz=1.0, coord_cols=("x", "y", "z"))
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_grouped, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode=cfg.stitch.mode, threshold=PMI_THR, metric=cfg.stitch.metric,
        penalize_simplicity=cfg.stitch.penalize_simplicity,
        deltaC_min=cfg.stitch.deltaC_min,
        c_union_bypass=cfg.stitch.c_union_bypass,
        c_union_bypass_max_n_tx=cfg.stitch.c_union_bypass_max_n_tx,
        max_merger_depth=cfg.stitch.max_merger_depth,
        dist_threshold=cfg.stitch.dist_threshold_um,
        out_col="stitched", show_progress=False,
        candidate_source=cfg.stitch.candidate_source,
        G=cfg.stitch.bin_size_um,
        stitch_neighborhood=cfg.stitch.neighborhood,
        G_z=_stitch_g_z,
        z_neighbor_depth=_stitch_z_depth,
        min_local_tx_per_entity=cfg.stitch.min_local_tx_per_entity,
        mahalanobis_d_rescue=cfg.stitch.mahalanobis_d_rescue,
        rescue_delta_c_floor=cfg.stitch.rescue_delta_c_floor,
    )
    _record_stage(progression, "Stitch", df_stitched, "stitched")

    # Demote
    df_stitched, n_demoted = demote_small_entities(
        df_stitched, entity_col="stitched", out_col="stitched",
        min_size=cfg.demote.min_size, unassigned_label="-1",
    )
    _record_stage(progression, "Demote", df_stitched, "stitched")

    # Final Rescue — iterative with two exit conditions:
    #   (a) hard ceiling at cfg.final_rescue.max_passes
    #   (b) convergence gate at early_exit_admit_ratio (if > 0): break
    #       when a pass admits fewer than ratio * pre-pass-pool tx.
    for _pass in range(cfg.final_rescue.max_passes):
        df_stitched, n_reassigned, stats = reassign_unassigned_grid_pool(
            df_stitched, aux=aux,
            entity_col="stitched", gene_col="feature_name",
            coord_cols=("x", "y", "z"), out_col="stitched",
            G=cfg.final_rescue.bin_size_um,
            pos_npmi_threshold=PMI_THR,
            neg_npmi_threshold=cfg.final_rescue.neg_threshold,
            only_partial_component=False,
            veto_mode=cfg.final_rescue.veto_mode,
            mean_threshold=cfg.final_rescue.mean_admit_threshold,
            min_admit_threshold=cfg.final_rescue.min_admit_threshold,
            small_entity_guard_n=cfg.final_rescue.small_entity_guard_n,
            real_signal_threshold=cfg.final_rescue.real_signal_threshold,
            aggregator_percentile=cfg.final_rescue.aggregator_percentile,
            rank_policy=cfg.final_rescue.rank_policy,
            witness_min_admit=cfg.final_rescue.witness_min_admit,
            witness_cap=cfg.final_rescue.witness_cap,
            witness_small_component_cap_divisor=cfg.final_rescue.witness_small_component_cap_divisor,
            witness_tiebreak=cfg.final_rescue.witness_tiebreak,
            offpanel_first_entity=cfg.final_rescue.offpanel_first_entity,
        )
        if n_reassigned == 0:
            break
        # Convergence gate (no-op when early_exit_admit_ratio == 0.0).
        n_un_before = int(stats.get("total_unassigned", 0))
        if (cfg.final_rescue.early_exit_admit_ratio > 0.0
                and n_un_before > 0
                and (n_reassigned / n_un_before
                     < cfg.final_rescue.early_exit_admit_ratio)):
            break
    _record_stage(progression, "Final Rescue", df_stitched, "stitched")

    # Finalize unassigned-class labels → "DROP" (see segmented runner
    # for full rationale).
    finalize_unassigned(df_stitched, col="stitched")
    _record_stage(progression, "Finalize", df_stitched, "stitched")

    df_stitched = _canonicalize_output(df_stitched, _input_cell_id)

    return df_stitched, progression
