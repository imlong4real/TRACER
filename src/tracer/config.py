"""Pipeline configuration — typed dataclasses + TOML loader.

Phase A of the config migration: this module defines the dataclasses
that codify every tunable knob in the segmentation pipeline, plus a
loader that builds a `PipelineConfig` from layered TOML files.

Design notes
------------
* Code defaults are canonical. `configs/defaults.toml` is a
  human-readable export of those defaults. `tests/test_config.py`
  verifies the two agree, so the TOML stays in lock-step.
* Layered composition: ``defaults`` ← ``platforms/<name>.toml`` ← user
  override file. Each layer patches keys; sections are merged, not
  replaced wholesale.
* `[final_rescue]` accepts an ``inherit = "rescue"`` directive:
  resolved values from `[rescue]` are copied first, then the local
  keys override. One-level inherit only — no transitive chains.
* Frozen dataclasses → configs are hashable, can pin a run.
* `dump_receipt(cfg, path)` writes resolved values as JSON for the
  per-run receipt that ships alongside outputs (reproducibility).

Phase B will switch the runner to consume `PipelineConfig`; this
module is currently standalone — importing it has no effect on the
pipeline.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

# Python 3.11+ has tomllib in the stdlib; 3.10 (still in our CI matrix
# per pyproject.toml `requires-python = ">=3.9"`) needs the `tomli`
# backport. Same API; module name aliased.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

# ---------------------------------------------------------------------------
# Per-stage configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase1Config:
    """Phase 1 (a/b/c) — nuclear-anchored greedy prune + admission.

    Phase 1b admission gate
    -----------------------
    The 1b gate decides whether a cytoplasmic / non-seed nuclear tx
    admits to the main cell via gene-fit to the seed. ``veto_mode``
    selects the gate:

    - ``"mean"`` (default; back-compat): admit when mean PMI(gene,seed)
      >= ``pmi_threshold``. Permissive — at lineage interfaces a few
      housekeeping/IFN positives can dilute strong opposing-lineage
      signal (see the EPCAM-vs-macrophage case study in the hybrid
      admission spec).
    - ``"hybrid"``: real-signal filter + unanimous-min fast-pass +
      percentile-aggregator gate. Mirrors the Rescue hybrid kernel
      (see `_cy_prune._admission_test`). Recommended for lineage-aware
      panels; rejects cross-lineage candidates that the mean gate
      admits. Note: switching the default to ``"hybrid"`` will
      regenerate the Phase-1 partition. Regression refs will need
      deliberate regeneration before flipping the default.
    - ``"min"``: veto on any seed pair with PMI < ``neg_npmi_threshold``.
      Strictest, rarely used outside diagnostic runs.
    """
    # 2026-05-13: pmi_threshold raised 0.05 → 0.2 to match the new
    # bootstrap-PMI calibration (PMI=0.2 = 1.22× chance in natural-log).
    pmi_threshold: float = 0.2
    seed_coherence_floor: float = 0.10
    tx_weighted_prune: bool = True
    nuclear_only_admit: bool = True

    # 1b admission gate (mirrors RescueConfig's veto knobs)
    veto_mode: Literal["min", "mean", "hybrid"] = "hybrid"
    mean_admit_threshold: float = 0.2
    min_admit_threshold: float = 0.0
    aggregator_percentile: float = 25.0
    real_signal_threshold: float = 0.05
    neg_npmi_threshold: float = -0.2

    # ------------------------------------------------------------------
    # Phase-1-time Mahalanobis-gated remerge (opt-in).
    # ------------------------------------------------------------------
    # Sibling of StitchConfig.mahalanobis_d_rescue applied one stage
    # earlier — between Phase1-QC and Rescue. For each entity pair
    # sharing a bin neighborhood (xy 8-Moore + ±1 z), if
    #
    #     maha_remerge_delta_c_floor < ΔC < 0   AND   D ≤ maha_remerge_d
    #
    # the two roots are unioned via DSU. The ΔC floor (≤ 0) protects
    # against fusing engulfment doublets where composition rejects
    # strongly. Default 1.0 ships the stage on; set to ``None`` to
    # disable (the stage becomes a no-op).
    maha_remerge_d: float | None = 1.0
    maha_remerge_delta_c_floor: float = -0.2

    def __post_init__(self) -> None:
        if not (-1.0 <= self.pmi_threshold <= 1.0):
            raise ValueError(
                f"phase1.pmi_threshold out of range: {self.pmi_threshold}"
            )
        if not (0.0 <= self.seed_coherence_floor <= 1.0):
            raise ValueError(
                f"phase1.seed_coherence_floor out of range: "
                f"{self.seed_coherence_floor}"
            )
        if self.veto_mode not in ("min", "mean", "hybrid"):
            raise ValueError(
                f"phase1.veto_mode must be 'min'/'mean'/'hybrid'; "
                f"got {self.veto_mode!r}"
            )
        if not (0.0 <= self.aggregator_percentile <= 100.0):
            raise ValueError(
                f"phase1.aggregator_percentile must be in [0, 100]; "
                f"got {self.aggregator_percentile}"
            )
        if self.real_signal_threshold < 0.0:
            raise ValueError(
                f"phase1.real_signal_threshold must be >= 0; "
                f"got {self.real_signal_threshold}"
            )
        if self.maha_remerge_d is not None and not (
            self.maha_remerge_d > 0.0
        ):
            raise ValueError(
                f"phase1.maha_remerge_d must be > 0 when set; "
                f"got {self.maha_remerge_d}"
            )
        if not math.isfinite(self.maha_remerge_delta_c_floor):
            raise ValueError(
                f"phase1.maha_remerge_delta_c_floor must be finite; "
                f"got {self.maha_remerge_delta_c_floor}"
            )
        if self.maha_remerge_delta_c_floor > 0.0:
            raise ValueError(
                f"phase1.maha_remerge_delta_c_floor must be <= 0; "
                f"got {self.maha_remerge_delta_c_floor}"
            )


@dataclass(frozen=True)
class SplitPhase1Config:
    """Post-Phase-1 z-gap splitter (no-op when z column absent)."""
    dz_threshold_um: float = 2.0
    min_tx: int = 1
    min_entity_size: int = 2

    def __post_init__(self) -> None:
        if self.dz_threshold_um <= 0:
            raise ValueError(
                f"split_phase1.dz_threshold_um must be > 0; got {self.dz_threshold_um}"
            )
        if self.min_entity_size < 2:
            raise ValueError(
                f"split_phase1.min_entity_size must be >= 2; got {self.min_entity_size}"
            )


@dataclass(frozen=True)
class Phase1QcConfig:
    """Demote Phase-1 entities below this size threshold."""
    min_tx: int = 3

    def __post_init__(self) -> None:
        if self.min_tx < 1:
            raise ValueError(f"phase1_qc.min_tx must be >= 1; got {self.min_tx}")


@dataclass(frozen=True)
class Phase1RerankConfig:
    """Re-rank depth-1 entities under each parent cell by nuclear-tx
    count; promote the largest to the main `{cell_id}` slot.

    Defuses Phase 1's greedy 1a→1b→1c privilege when a partial ends up
    with more nuclear tx than the main. See
    `docs/superpowers/specs/2026-05-11-phase1-rerank-design.md`.

    2026-05-13: promoted to default-on after PDAC + lung cross-tissue
    validation (89%/67% retention, cell C mean 0.93 with strict-PMI
    defaults; no failure mode observed).
    """
    enabled: bool = True
    margin_tx: int = 1   # minimum (n_largest - n_runner_up) required
                         # to swap. margin_tx=1 ⇒ strict >.

    def __post_init__(self) -> None:
        if self.margin_tx < 1:
            raise ValueError(
                f"phase1_rerank.margin_tx must be >= 1; got {self.margin_tx}"
            )


@dataclass(frozen=True)
class Phase1ReassignConfig:
    """Phase-1 post-1c nuclear reassignment (Gap-B fix).

    Nuclear tx weakly admitted to the main seed, but strong-fit to a
    1c partial sub-seed, get moved to the partial. Wired in commit
    `8454454`; numpy-vectorized in `0217585` (3.1× speedup); Cython
    kernel `_cy_reassign` with OpenMP prange added 2026-05-13.

    Default-on since 2026-05-11 (commit `a9718fd`); promoted to FROZEN
    2026-05-13.
    """
    enabled: bool = True
    margin: float = 0.05   # min Δ(partial-fit - main-fit) PMI to trigger move

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError(
                f"phase1_reassign.margin must be >= 0; got {self.margin}"
            )


@dataclass(frozen=True)
class DemoteConfig:
    """Demote entities smaller than `min_size` to unassigned (post-Stitch).

    Runs once between Stitch and Final Rescue. Catches sub-threshold
    entities that survived Stitch via lucky merges but are too small
    to be biologically meaningful.

    Promoted to FROZEN 2026-05-13.
    """
    min_size: int = 5

    def __post_init__(self) -> None:
        if self.min_size < 1:
            raise ValueError(
                f"demote.min_size must be >= 1; got {self.min_size}"
            )


@dataclass(frozen=True)
class RescueConfig:
    """Spatial-prior rescue veto (used by main Rescue and Final Rescue)."""
    veto_mode: Literal["min", "mean", "hybrid"] = "hybrid"
    min_admit_threshold: float = 0.0      # hybrid: unanimous-pos cutoff
    # 2026-05-13: mean_admit_threshold raised 0.1 → 0.5 to match the new
    # bootstrap-PMI calibration. Validated cell C mean 0.80→0.93 on PDAC.
    mean_admit_threshold: float = 0.5     # hybrid/mean: aggregate-pos cutoff
    neg_threshold: float = -0.2           # cluster-guard / min-mode veto; paired with phase1.pmi_threshold
    max_passes: int = 3
    bin_size_um: float = 2.0
    z_bound_um: float | None = None       # None → G * sqrt(2)
    cluster_guard_n: int = 3
    small_entity_guard_n: int = 0
    # 2026-05-13: aggregator_percentile lowered 50 → 25 (stricter; mean
    # is computed over the bottom-quartile of real-signal pairs).
    aggregator_percentile: float = 25.0
    # Real-players gate (cross-cutting, but Rescue-overridable). Pairs
    # with |PMI| ≤ this contribute neither to mean nor to count gates.
    # Default 0.05 matches the cross-cutting REAL_SIGNAL_THRESHOLD.
    real_signal_threshold: float = 0.05

    # ------------------------------------------------------------------
    # Rank policy — how a non-vetoed candidate is chosen among the
    # entities in the 9-bin × z-bound neighborhood of an orphan tx.
    # ------------------------------------------------------------------
    # "distance" : nearest-tx distance wins (legacy production behavior).
    # "witness"  : count of supporting tx (capped) wins; tie broken by
    #              `witness_tiebreak`. Inspired by Stitch's witness gate
    #              — see docs/superpowers/specs/2026-05-14-rescue-witness-
    #              rank-design.md (forthcoming).
    # 2026-05-15: promoted "distance" → "witness" after Cython port
    # (`_cy_prune.rescue_per_tx_batch` gained witness branch) and 2×2
    # PDAC ROI bench. Witness improves small-cell ARI in NOSEG
    # (0.496 → 0.581 on <20-tx cells) with ~11% wall overhead vs
    # distance Cython. SEG ARI cost is 0.047 — accepted in exchange
    # for coherence and small-cell wins.
    rank_policy: Literal["distance", "witness"] = "witness"

    # The following knobs are meaningful only when ``rank_policy ==
    # "witness"``; they are ignored under "distance". Defaults reflect
    # the 50 µm-ROI bench-derived starting point (MIN_ADMIT=3, CAP=3,
    # gene-fit tiebreak, small-component damper).
    witness_min_admit: int = 3
    witness_cap: int = 3
    # Small-component witness damper. An entity contributes at most
    # ``ceil(entity_size / witness_small_component_cap_divisor)``
    # witnesses, capped further by ``witness_cap``. Damps the influence
    # of small entities without making them ineligible.
    witness_small_component_cap_divisor: int = 2
    # Tie-breaker when multiple candidates have the same (capped)
    # witness count: "distance" (nearest-tx) or "gene_fit" (highest
    # mean PMI of the orphan gene against the candidate's seed gene set).
    witness_tiebreak: Literal["distance", "gene_fit"] = "gene_fit"

    # ------------------------------------------------------------------
    # Convergence-aware early exit. After each Rescue pass, compare the
    # number admitted to the pre-pass unassigned-pool size. If the
    # ratio falls below this threshold, break the loop. Diminishing-
    # returns guard — saves wall on the asymptotic tail of large pools
    # (NOSEG) while letting fast-converging runs (SEG) exit naturally.
    #   0.0 = disabled (legacy: break only on zero admits)
    #   0.01 = 1 % gate (recommended for Final Rescue per 2x2 bench)
    # Applied independently to each rescue invocation (main Rescue,
    # Post-Group Rescue, Final Rescue).
    # ------------------------------------------------------------------
    early_exit_admit_ratio: float = 0.0

    # Post-Group Rescue pass count (the second Rescue stage in both
    # pipelines, after Group/cascade). Pulled out of the module-global
    # ``RESCUE_POST_GROUP_PASSES`` 2026-05-15. SEG default 3; NOSEG
    # platform preset raises to 5 — NOSEG enters Post-Group Rescue
    # with a much larger orphan pool, so the asymptotic admit-tail
    # extends further.
    post_group_passes: int = 3

    def __post_init__(self) -> None:
        if self.veto_mode not in ("min", "mean", "hybrid"):
            raise ValueError(
                f"rescue.veto_mode must be 'min'/'mean'/'hybrid'; got {self.veto_mode!r}"
            )
        if self.max_passes < 1:
            raise ValueError(
                f"rescue.max_passes must be >= 1; got {self.max_passes}"
            )
        if self.bin_size_um <= 0:
            raise ValueError(
                f"rescue.bin_size_um must be > 0; got {self.bin_size_um}"
            )
        if not (0.0 <= self.aggregator_percentile <= 100.0):
            raise ValueError(
                f"rescue.aggregator_percentile must be in [0, 100]; "
                f"got {self.aggregator_percentile}"
            )
        if self.real_signal_threshold < 0.0:
            raise ValueError(
                f"rescue.real_signal_threshold must be >= 0; "
                f"got {self.real_signal_threshold}"
            )
        if self.rank_policy not in ("distance", "witness"):
            raise ValueError(
                f"rescue.rank_policy must be 'distance' or 'witness'; "
                f"got {self.rank_policy!r}"
            )
        if self.witness_min_admit < 1:
            raise ValueError(
                f"rescue.witness_min_admit must be >= 1; "
                f"got {self.witness_min_admit}"
            )
        if self.witness_cap < 1:
            raise ValueError(
                f"rescue.witness_cap must be >= 1; got {self.witness_cap}"
            )
        if self.witness_small_component_cap_divisor < 1:
            raise ValueError(
                f"rescue.witness_small_component_cap_divisor must be >= 1; "
                f"got {self.witness_small_component_cap_divisor}"
            )
        if self.witness_tiebreak not in ("distance", "gene_fit"):
            raise ValueError(
                f"rescue.witness_tiebreak must be 'distance' or 'gene_fit'; "
                f"got {self.witness_tiebreak!r}"
            )
        if not (0.0 <= self.early_exit_admit_ratio <= 1.0):
            raise ValueError(
                f"rescue.early_exit_admit_ratio must be in [0.0, 1.0]; "
                f"got {self.early_exit_admit_ratio}"
            )
        if self.post_group_passes < 0:
            raise ValueError(
                f"rescue.post_group_passes must be >= 0; "
                f"got {self.post_group_passes}"
            )


@dataclass(frozen=True)
class GroupConfig:
    """Phase 2 — entity grouping.

    Two backends, selected per-pipeline by ``seg_residual_cascade`` /
    ``noseg_cascade``:

    * **Cascade** (default since 2026-05-07 for SEG-residual,
      2026-05-09 for NOSEG): density-anchor + Phase-1a/b purity prune.
      Emits ``cascade_<n>-1`` partial labels. Auto-floor selects the
      density-threshold floor at runtime by walking down candidate
      thresholds until ``target_cov`` of residual tx is captured (with
      a ``hard_min`` lower bound).

    * **Legacy spatial-CC** (when ``*_cascade=False``): the original
      ``annotate_unassigned_components_fast`` with G=8 µm self-bin
      connectivity + post-hoc gene-set prune. Emits ``UNASSIGNED_<n>``
      component labels. Retained as a fallback for parity with
      pre-cascade pipeline state.

    The cascade backend's internal params (bin size, territory radius,
    per-pair PMI threshold, min anchor tx) are shared between the SEG
    and NOSEG paths; the auto-floor params (target_cov, hard_min) are
    per-pipeline because SEG-residual is sparser than the full NOSEG
    pool — auto-floor selects different floors in those regimes.
    """
    # Backend selection (per pipeline)
    seg_residual_cascade: bool = True
    noseg_cascade: bool = True

    # Cascade auto-floor — per pipeline (different residual density regimes)
    seg_cascade_target_cov: float = 0.65
    seg_cascade_hard_min: int = 2
    noseg_cascade_target_cov: float = 0.65
    noseg_cascade_hard_min: int = 2

    # Cascade internal params — shared between SEG and NOSEG paths
    cascade_bin_size_um: float = 2.0           # G in cascade_as_residual_handler
    cascade_territory_radius_bins: int = 1      # Moore radius (3×3 = 9 bins)
    cascade_pmi_threshold: float = 0.2          # mirrors phase1.pmi_threshold
    cascade_min_anchor_tx: int = 3

    # Legacy spatial-CC fallback — used when *_cascade=False
    legacy_bin_size_um: float = 8.0             # G in build_grid_graph_xy
    legacy_neighborhood: Literal["self", "moore"] = "self"
    legacy_k: int = 8
    legacy_dist_threshold: float = 1.5
    legacy_min_comp_size: int = 5
    legacy_npmi_threshold: float = -0.1         # post-prune negative cutoff

    def __post_init__(self) -> None:
        # Cascade params
        if not (0.0 < self.seg_cascade_target_cov <= 1.0):
            raise ValueError(
                f"group.seg_cascade_target_cov must be in (0, 1]; got {self.seg_cascade_target_cov}"
            )
        if not (0.0 < self.noseg_cascade_target_cov <= 1.0):
            raise ValueError(
                f"group.noseg_cascade_target_cov must be in (0, 1]; got {self.noseg_cascade_target_cov}"
            )
        if self.seg_cascade_hard_min < 2:
            raise ValueError(
                f"group.seg_cascade_hard_min must be >= 2; got {self.seg_cascade_hard_min}"
            )
        if self.noseg_cascade_hard_min < 2:
            raise ValueError(
                f"group.noseg_cascade_hard_min must be >= 2; got {self.noseg_cascade_hard_min}"
            )
        if self.cascade_bin_size_um <= 0:
            raise ValueError(
                f"group.cascade_bin_size_um must be > 0; got {self.cascade_bin_size_um}"
            )
        if self.cascade_territory_radius_bins < 1:
            raise ValueError(
                f"group.cascade_territory_radius_bins must be >= 1; got {self.cascade_territory_radius_bins}"
            )
        if not (-1.0 <= self.cascade_pmi_threshold <= 1.0):
            raise ValueError(
                f"group.cascade_pmi_threshold out of range: {self.cascade_pmi_threshold}"
            )
        if self.cascade_min_anchor_tx < 1:
            raise ValueError(
                f"group.cascade_min_anchor_tx must be >= 1; got {self.cascade_min_anchor_tx}"
            )
        # Legacy params
        if self.legacy_bin_size_um <= 0:
            raise ValueError(
                f"group.legacy_bin_size_um must be > 0; got {self.legacy_bin_size_um}"
            )
        if self.legacy_neighborhood not in ("self", "moore"):
            raise ValueError(
                f"group.legacy_neighborhood must be 'self' or 'moore'; got {self.legacy_neighborhood!r}"
            )
        if self.legacy_min_comp_size < 1:
            raise ValueError(
                f"group.legacy_min_comp_size must be >= 1; got {self.legacy_min_comp_size}"
            )


@dataclass(frozen=True)
class StitchConfig:
    """Phase 4 — hierarchical entity stitching.

    Mirrors the production call in
    ``tests/_pipeline_runner.py::run_segmented_pipeline`` (and the
    symmetric NOSEG path) at the time of typing (2026-05-14). All
    values are the production defaults active after the
    `bugfix/stitch-dist-threshold` merge:

      * ``deltaC_min=0.03`` (raised 0.0 → 0.03 on 2026-05-09 to prevent
        NOSEG supercell formation; SEG insensitive).
      * ``c_union_bypass=0.9`` + ``c_union_bypass_max_n_tx=50``: admit
        ΔC-failing pairs when C(union) ≥ 0.9 AND merged n_tx ≤ 50.
        Recovers within-cell fragment consolidations where both parents
        are at C ≈ 1.0 and ΔC has no headroom; size cap keeps the
        bypass targeted at within-cell, not cross-compartment.
      * ``max_merger_depth=3``: post-acceptance merger-tree depth cap.
        Blocks chain-style growth.
      * ``min_local_tx_per_entity=3``: per-entity witness floor in the
        shared bin neighborhood. Capped at min(threshold, entity n_tx)
        internally.

    Decomposable-coherence + DSU + heap fast path (``use_decomposable_
    stitch=True`` in ``stitching.py``) is ⚪ PLANNED as opt-in — not
    represented here yet.
    """
    # Coherence metric
    mode: Literal["count", "primitives"] = "count"
    metric: Literal["pmi", "npmi"] = "pmi"
    penalize_simplicity: bool = True
    deltaC_min: float = 0.03

    # C(union) bypass for ΔC-failing pairs
    c_union_bypass: float | None = 0.9
    c_union_bypass_max_n_tx: int | None = 50

    # Merger-tree depth cap
    max_merger_depth: int | None = 3

    # Spatial gate
    candidate_source: Literal["grid", "delaunay"] = "grid"
    bin_size_um: float = 2.0                    # xy bin (G)
    g_z_um: float | None = 1.0                  # z bin; None → auto_Gz from estimator
    z_neighbor_depth: int = 1                   # ±depth z bins
    neighborhood: Literal["4", "8"] = "8"       # xy Moore reach
    dist_threshold_um: float = 5.0              # max 3D distance for candidate pairs

    # Witness floor (per-entity tx count in shared bin neighborhood)
    min_local_tx_per_entity: int = 3

    # Mahalanobis-D RESCUE on borderline-ΔC pairs.
    # When `mahalanobis_d_rescue` is set (recommended ~1.0), the loop
    # OVERRIDES a ΔC reject for candidate pairs whose:
    #     rescue_delta_c_floor < ΔC < 0    AND    D ≤ mahalanobis_d_rescue
    # i.e. when composition borderline-rejects AND the two tx clouds
    # are geometrically enmeshed (low Mahalanobis D relative to the
    # pooled covariance structure). Recovers EMT-like cells where the
    # panel's epi/mes anti-correlation drags ΔC slightly negative on a
    # legitimate single-cell merge. Default 1.0 ships the rescue on
    # (PDAC 2 mm: ~1.8 k cell-consolidations that ΔC alone misses);
    # set to ``None`` to disable.
    #
    # The `rescue_delta_c_floor` (default -0.2, must be ≤ 0) protects
    # against fusing engulfment doublets (jikammne-like: ΔC = -0.49,
    # D ≈ 0.5 — D is low, but ΔC is well below the floor → no rescue).
    # An earlier veto-direction Maha implementation was superseded —
    # the witness floor `min_local_tx_per_entity` already gates the
    # accept path adequately; geometry's useful contribution is the
    # rescue, not a veto.
    mahalanobis_d_rescue: float | None = 1.0
    rescue_delta_c_floor: float = -0.2

    def __post_init__(self) -> None:
        if not (-1.0 <= self.deltaC_min <= 1.0):
            raise ValueError(
                f"stitch.deltaC_min out of range: {self.deltaC_min}"
            )
        if self.c_union_bypass is not None and not (0.0 <= self.c_union_bypass <= 1.0):
            raise ValueError(
                f"stitch.c_union_bypass out of range: {self.c_union_bypass}"
            )
        if (self.c_union_bypass_max_n_tx is not None
                and self.c_union_bypass_max_n_tx < 1):
            raise ValueError(
                f"stitch.c_union_bypass_max_n_tx must be >= 1; "
                f"got {self.c_union_bypass_max_n_tx}"
            )
        if self.max_merger_depth is not None and self.max_merger_depth < 1:
            raise ValueError(
                f"stitch.max_merger_depth must be >= 1; got {self.max_merger_depth}"
            )
        if self.bin_size_um <= 0:
            raise ValueError(
                f"stitch.bin_size_um must be > 0; got {self.bin_size_um}"
            )
        if self.g_z_um is not None and self.g_z_um <= 0:
            raise ValueError(
                f"stitch.g_z_um must be > 0 (or None for auto); got {self.g_z_um}"
            )
        if self.z_neighbor_depth < 0:
            raise ValueError(
                f"stitch.z_neighbor_depth must be >= 0; got {self.z_neighbor_depth}"
            )
        if self.dist_threshold_um <= 0:
            raise ValueError(
                f"stitch.dist_threshold_um must be > 0; got {self.dist_threshold_um}"
            )
        if self.min_local_tx_per_entity < 0:
            raise ValueError(
                f"stitch.min_local_tx_per_entity must be >= 0; "
                f"got {self.min_local_tx_per_entity}"
            )
        if self.mahalanobis_d_rescue is not None and not (
            self.mahalanobis_d_rescue > 0.0
        ):
            raise ValueError(
                f"stitch.mahalanobis_d_rescue must be > 0 when set; "
                f"got {self.mahalanobis_d_rescue}"
            )
        if not math.isfinite(self.rescue_delta_c_floor):
            raise ValueError(
                f"stitch.rescue_delta_c_floor must be finite; "
                f"got {self.rescue_delta_c_floor}"
            )
        if self.rescue_delta_c_floor > 0.0:
            raise ValueError(
                f"stitch.rescue_delta_c_floor must be <= 0; "
                f"got {self.rescue_delta_c_floor}"
            )


# Mid-QC and Post-Group Rescue remain untyped for now. Mid-QC's
# coherence-floor knob is still being tuned per-platform; Post-Group
# Rescue shares `RescueConfig` (just runs the same call site with the
# same knobs). Both will be promoted as their scope settles.


@dataclass(frozen=True)
class BootstrapConfig:
    """`compute_pmi_bootstrap` config-recommended defaults.

    These are the flavor-C "production" settings (PMI metric, per-gene
    size-band filter, dual-tau, set_neg_one). The function's own Python
    signature defaults stay backward-compatible (legacy NPMI, no
    per-gene filter) for callers that don't go through the config — but
    callers using `load_config()` get the recommended settings. The
    `tests/test_config.py::test_defaults_toml_matches_dataclass_defaults`
    test enforces lockstep with `configs/defaults.toml`.

    Platform presets (e.g. `xenium.toml`) layer on top of these to add
    platform-specific knobs that depend on column availability (e.g.
    `nuclear_only=true` requires `overlaps_nucleus`).
    """
    # Pre-filter pipeline
    nuclear_only: bool = False                  # platform-only (needs overlaps_nucleus column)
    nucleus_col: str = "overlaps_nucleus"
    percentile_filter: tuple[float, float] | None = None       # global percentile is biased; off
    per_gene_percentile_filter: tuple[float, float] | None = (5.0, 95.0)  # ← flavor-C default
    memory_optimize: bool = True
    # Bootstrap engine
    metric: Literal["pmi", "npmi"] = "pmi"      # ← flavor-C default (was npmi)
    tau_low: float = 0.05
    tau_high: float = 0.20                      # ← flavor-C dual-tau default (was single 0.05)
    alpha: float = 0.1
    ci_level: float = 0.95
    max_bootstraps: int = 10_000
    coarse_block: int = 200
    refine_block: int = 500
    min_samples_for_ci: int = 30
    subsample_size: int | None = 25_000         # ← flavor-C default (was None / full pop)
    # Evidence gates
    min_occurrences_per_context: int = 2
    min_expected_cooccur_for_evidence: float = 10.0
    # When None → falls back to min_expected_cooccur_for_evidence (legacy
    # behavior: same threshold gates evidence and bootstrap eligibility).
    # Set explicitly to control how rare-cooccurrence pairs are routed:
    #   higher value → more pairs sent to legacy_only (no bootstrap CI)
    #   0.0          → bootstrap every high-evidence pair (wide CIs for sparse)
    min_expected_cooccur_for_bootstrap: float | None = 10.0  # ← flavor-C default (explicit; None → fall back to evidence threshold)
    # When True, k=0 pairs with E_full ≥ min_expected_cooccur_for_evidence
    # are classified as `neg_one` (mutual-exclusion sentinel; W = -1 for
    # NPMI metric, W = -log(E_full) for PMI). When False, those pairs
    # are classified as `indeterminate` and left absent from W. Matches
    # the legacy compute_npmi `set_neg_one` semantics, but the gate uses
    # E_full (not marginal probability).
    set_neg_one: bool = True

    def __post_init__(self) -> None:
        # Coerce TOML lists → tuples for frozen-dataclass hashability.
        for fname in ("percentile_filter", "per_gene_percentile_filter"):
            v = getattr(self, fname)
            if isinstance(v, list):
                object.__setattr__(self, fname, tuple(v))
        if self.metric not in ("pmi", "npmi"):
            raise ValueError(
                f"bootstrap.metric must be 'pmi' or 'npmi'; got {self.metric!r}"
            )
        if not (0.0 <= self.tau_low <= self.tau_high):
            raise ValueError(
                f"bootstrap requires 0 <= tau_low <= tau_high; got "
                f"({self.tau_low}, {self.tau_high})"
            )
        if not (0.0 <= self.alpha):
            raise ValueError(f"bootstrap.alpha must be >= 0; got {self.alpha}")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(
                f"bootstrap.ci_level must be in (0, 1); got {self.ci_level}"
            )
        if self.percentile_filter is not None:
            lo, hi = self.percentile_filter
            if not (0.0 <= lo < hi <= 100.0):
                raise ValueError(
                    f"bootstrap.percentile_filter must satisfy 0 <= lo < hi <= 100; "
                    f"got {self.percentile_filter}"
                )
        if self.per_gene_percentile_filter is not None:
            lo, hi = self.per_gene_percentile_filter
            if not (0.0 <= lo < hi <= 100.0):
                raise ValueError(
                    f"bootstrap.per_gene_percentile_filter must satisfy "
                    f"0 <= lo < hi <= 100; got {self.per_gene_percentile_filter}"
                )

    @property
    def tau(self) -> float | tuple[float, float]:
        """Convenience: returns tau as the function expects it."""
        if self.tau_low == self.tau_high:
            return self.tau_low
        return (self.tau_low, self.tau_high)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline config. `final_rescue` defaults to a copy of
    `rescue` with `small_entity_guard_n = 0`; override by passing an
    explicit `RescueConfig` or via the `[final_rescue] inherit = "rescue"`
    pattern in TOML."""
    phase1: Phase1Config = field(default_factory=Phase1Config)
    split_phase1: SplitPhase1Config = field(default_factory=SplitPhase1Config)
    phase1_qc: Phase1QcConfig = field(default_factory=Phase1QcConfig)
    phase1_rerank: Phase1RerankConfig = field(default_factory=Phase1RerankConfig)
    phase1_reassign: Phase1ReassignConfig = field(default_factory=Phase1ReassignConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)
    group: GroupConfig = field(default_factory=GroupConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    demote: DemoteConfig = field(default_factory=DemoteConfig)
    final_rescue: RescueConfig = field(
        default_factory=lambda: RescueConfig(
            small_entity_guard_n=0,
            # 2026-05-15: SEG-friendly default — 3 passes, no gate.
            # Validated against 2×2 PDAC ROI: SEG FR pass 3 admits ~0.3%
            # of pool (asymptote). NOSEG benefits from more passes; use
            # `load_config(platform="noseg")` to get max_passes=5.
            max_passes=3,
        )
    )
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIGS_DIR = _PKG_DIR / "configs"

_SECTION_TO_CLS: dict[str, type] = {
    "phase1": Phase1Config,
    "split_phase1": SplitPhase1Config,
    "phase1_qc": Phase1QcConfig,
    "phase1_rerank": Phase1RerankConfig,
    "phase1_reassign": Phase1ReassignConfig,
    "rescue": RescueConfig,
    "group": GroupConfig,
    "stitch": StitchConfig,
    "demote": DemoteConfig,
    "final_rescue": RescueConfig,
    "bootstrap": BootstrapConfig,
}


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — override wins, sections merge, scalars replace."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_inherit(merged: dict[str, Any]) -> dict[str, Any]:
    """Resolve any `inherit = "<other_section>"` directives.

    One level only: the listed section's resolved values are copied
    in, then the local keys override. The `inherit` key itself is
    stripped from the output. Raises on cycles or bad targets.
    """
    out = dict(merged)
    for section, body in list(merged.items()):
        if not isinstance(body, dict):
            continue
        target = body.get("inherit")
        if target is None:
            continue
        if target == section:
            raise ValueError(f"[{section}] inherits from itself")
        if target not in merged or not isinstance(merged[target], dict):
            raise ValueError(
                f"[{section}] inherits from missing section [{target}]"
            )
        if "inherit" in merged[target]:
            raise ValueError(
                f"[{section}] inherits from [{target}] which itself inherits "
                f"— transitive inherit not supported"
            )
        resolved = dict(merged[target])
        for k, v in body.items():
            if k == "inherit":
                continue
            resolved[k] = v
        out[section] = resolved
    return out


def _to_dataclass(merged: dict[str, Any]) -> PipelineConfig:
    """Map a resolved dict to PipelineConfig, ignoring unknown sections."""
    kwargs: dict[str, Any] = {}
    for section, cls in _SECTION_TO_CLS.items():
        body = merged.get(section, {})
        if not isinstance(body, dict):
            raise ValueError(f"section [{section}] must be a table; got {type(body).__name__}")
        # Filter unknown keys with a clear error rather than silently dropping.
        valid_fields = {f.name for f in fields(cls)}
        unknown = set(body) - valid_fields
        if unknown:
            raise ValueError(
                f"[{section}] contains unknown keys: {sorted(unknown)} "
                f"(valid: {sorted(valid_fields)})"
            )
        kwargs[section] = cls(**body)
    return PipelineConfig(**kwargs)


def load_config(
    path: str | Path | None = None,
    *,
    platform: str | None = None,
) -> PipelineConfig:
    """Load a pipeline config.

    Layering: ``configs/defaults.toml``  ← (optional) ``configs/platforms/<platform>.toml``
    ← (optional) ``path``. Each layer patches keys.

    Parameters
    ----------
    path
        Optional user-override TOML file. Top of the layer stack.
    platform
        Optional platform-preset name (file under ``configs/platforms/``,
        without the ``.toml`` suffix). E.g. ``"xenium_3d"`` or
        ``"vhd_unsegmented"``.

    Returns
    -------
    PipelineConfig
        Frozen, fully-resolved config.
    """
    defaults_path = _DEFAULT_CONFIGS_DIR / "defaults.toml"
    merged: dict[str, Any] = _load_toml(defaults_path) if defaults_path.exists() else {}

    if platform is not None:
        plat_path = _DEFAULT_CONFIGS_DIR / "platforms" / f"{platform}.toml"
        if not plat_path.exists():
            available = sorted(
                p.stem for p in (_DEFAULT_CONFIGS_DIR / "platforms").glob("*.toml")
            ) if (_DEFAULT_CONFIGS_DIR / "platforms").exists() else []
            raise FileNotFoundError(
                f"Unknown platform {platform!r}; available: {available}"
            )
        merged = _deep_merge(merged, _load_toml(plat_path))

    if path is not None:
        merged = _deep_merge(merged, _load_toml(Path(path)))

    merged = _resolve_inherit(merged)
    return _to_dataclass(merged)


# ---------------------------------------------------------------------------
# Run-receipt dumper (JSON — deps-free, machine-readable, easy to diff)
# ---------------------------------------------------------------------------


def _normalize_for_json(obj: Any) -> Any:
    """Recursively coerce tuples → lists so JSON roundtrip is symmetric.
    asdict() preserves tuples, but json.dumps converts them to lists;
    without this normalization, `loaded == asdict(cfg)` after a roundtrip
    fails on any tuple-typed field."""
    if isinstance(obj, dict):
        return {k: _normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(v) for v in obj]
    return obj


def to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Recursively convert a PipelineConfig to a plain nested dict.
    Tuples are normalized to lists so the dict round-trips through JSON
    cleanly (`json.dumps` converts tuples → lists, and the loaded form
    must compare equal to this dict)."""
    return _normalize_for_json(asdict(cfg))


def dump_receipt(cfg: PipelineConfig, path: str | Path) -> None:
    """Write resolved config to JSON. Companion to a pipeline run; lets
    anyone reading the output later replay the exact same parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_dict(cfg), f, indent=2, sort_keys=True)


__all__ = [
    "Phase1Config",
    "SplitPhase1Config",
    "Phase1QcConfig",
    "Phase1RerankConfig",
    "Phase1ReassignConfig",
    "RescueConfig",
    "GroupConfig",
    "DemoteConfig",
    "BootstrapConfig",
    "PipelineConfig",
    "load_config",
    "to_dict",
    "dump_receipt",
]
