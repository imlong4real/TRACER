# Design note: one prune for cells and grouped entities

Status: proposal, 2026-11-13. Based on the code at `1e7e1a4` and on prototype runs on an internal Xenium dataset. Results from that dataset are unpublished and are kept out of this repository.

## Problem

TRACER prunes in two places:

- **Phase-1 Prune** works on segmented cells.
- **The Group cascade** works on seed windows of unassigned transcripts.

Conceptually these are the same operation: find the coherent gene program in a set of transcripts, keep it, and carve out or release the rest. The two paths share the greedy kernel, but everything around the kernel was written separately and has drifted apart. Each recent cascade fix has rebuilt something Phase-1 already does.

## What is shared today

Both paths call the same greedy core, `_cy_prune._greedy_prune_to_retained`:

- Conflict scores are weighted by transcript count.
- A gene whose own transcript count exceeds its conflict score is protected from removal.
- Pairs with PMI below `PMI_THR` (0.2) count as conflicts.
- Pairs missing from the panel are skipped.

The callers are:

| Caller | Where |
|---|---|
| Phase 1a seed (nuclear or whole cell) | `_prune_cells_nuclear_seed_core` |
| Phase 1c sub-seed | `_prune_cells_nuclear_seed_core` |
| Non-nuclear whole-cell prune | `prune_cells_retained` (via `prune_transcripts_fast`) |
| Cascade seed | `greedy_prune_retained` (via `density_cascade._greedy_prune_retained`) |

Both paths also treat a pair that is stored with value 0 as a conflict, because 0 < 0.2. So that is a panel-format question, not a difference between the paths.

## Where the paths diverge

| Behaviour | Phase-1 Prune | Cascade seed prune | Observed effect |
|---|---|---|---|
| **W contract** | Builds symmetric CSR arrays itself (`_symmetric_csr_arrays`) | Passes `W.tocsr()` as received. On the nuclear-seed path `aux["W"]` is upper-triangle only. | Conflict bookkeeping goes negative and pruning stops early, so cascade seeds keep conflicting genes. Symmetrizing W fixed this in the prototype and lowered the unassigned fraction. |
| **Membership after the seed** | 1b: every transcript is tested against the seed with `_admission_test` (mean, percentile and veto modes). A gene outside the seed can still be admitted if it fits. | A transcript joins only if its gene is in the retained set. There is no admission test. | Cascade entities are stricter on membership. Not measured separately. |
| **Rejected transcripts** | 1c: the rejected nuclear genes are pruned again into a sub-seed (at least `min_nuclear_genes`), with admission. That produces a carved partial; the rest go to unassigned. | All go back to the pool (window mode, the repo default). | In the prototype, cell-style carving (`TRACER_CASCADE_PRUNE=cell`) raised partial coherence and reduced unassigned transcripts. |
| **Size minimum** | Phase1-QC sends entities under `PHASE1_QC_MIN_TX` (3) to unassigned. | A seed needs at least `min_anchor_tx` (3) committed transcripts. The prototype sibling had no minimum. | The prototype produced many siblings under 3 transcripts. |
| **z-split** | Split-Phase1 splits at gaps over 2 µm. | None. The seed window is 3×3 bins in xy across all z. | Not measured. |
| **Re-ranking the main entity** | Phase1-Rerank picks the main by nuclear transcript count. | Not applicable (no nucleus). | – |
| **Remerging over-splits** | Phase1-Maha-Remerge (D ≤ 1, ΔC > −0.2) | Only later, through Stitch's Mahalanobis rescue | – |
| **Config source** | `cfg.phase1.*` plus module constants | NOSEG path: `cfg.group.*`. SEG path: hard-coded literals (`G=2.0`, `territory_radius_bins=1`, `min_anchor_tx=3`, `PMI_THR`), so `cfg.group` is ignored. | Settings silently differ between SEG and NOSEG. |

There is also a legacy Group path (`annotate_unassigned_components_fast`, used when `PHASE1_SEG_RESIDUAL_CASCADE=False`). It uses a third implementation, the pure-Python `prune_genes_by_pmi_greedy`, which:

- counts each conflicting gene once instead of weighting by transcripts;
- uses a different default threshold (`ANNOTATE_NEG_THR`).

It is off by default.

## Proposal

One orchestration function that both paths call:

```python
prune_entity(tx_idx, gene_idx, coords, W_sym, cfg, *, seed_mask=None)
    -> PruneResult(main: idx, carved: list[idx], rejected: idx)
```

1. **One W contract.** `W_sym` is a symmetric CSR built once, at panel load, by a single helper. `aux["W"]` always carries this form, with the convention stated in its docstring. Every consumer reads the same object: Prune, cascade, Rescue, Mid-QC, Stitch and Reassign-1c. Stitch's `W + W.T` becomes a no-op, or an assertion.
2. **Seed.** Greedy prune over the seed genes:
   - Phase-1 sets `seed_mask` to nuclear or whole-cell transcripts.
   - The cascade uses the whole window.
   - The kernel is unchanged.
3. **Admission (1b).** Each transcript is tested against the seed with `_admission_test` under `cfg.prune.admission`. The cascade gains admission; whether it should use the same veto settings as Phase-1 is an open question.
4. **Carve (1c).** The prune is repeated on the rejected genes to give a sub-seed with admission, up to `cfg.prune.max_carves` (default 1). The prototype's cell-style sibling becomes this step.
5. **Shared post-steps, applied to main and carved:** z-split (`cfg.prune.split_dz`), then the size minimum (`cfg.prune.min_tx`), then the optional Maha remerge.
6. **The caller decides what happens to rejected transcripts.**
   - Phase-1 sends them to unassigned, for Rescue.
   - The cascade returns them to the pool for later seeds.

   This is the only intended behavioural difference between the two paths.

Nuclear-specific steps stay outside `prune_entity` as Phase-1 wrappers: the seed mask, nuclear-only admission, Reassign-1c and Rerank.

### Settings that collapse

| Today | After |
|---|---|
| `PHASE1_QC_MIN_TX`, `group.cascade_min_anchor_tx`, prototype `TRACER_CASCADE_SIB_MIN` | `prune.min_tx` |
| `SPLIT_PHASE1_DZ`, plus a missing cascade equivalent | `prune.split_dz` |
| Prototype `TRACER_CASCADE_PRUNE` | `prune.max_carves` (0 = window behaviour, 1 = cell-style) |
| SEG-path hard-coded cascade literals | `cfg.group.*`, read on both paths |

## Migration plan (each step is its own PR)

1. **Lock in current output.** Add bit-identical regression fixtures for Phase-1 and Group outputs on a small Xenium ROI, with the nucleus column, and on a NOSEG ROI.
2. **One W contract.** Symmetrize at panel load. Audit every `aux["W"]` consumer: Stitch, Rescue, Final Rescue, Mid-QC, Reassign-1c, coherence scoring. Expected diff: the cascade changes; Stitch must stay bit-identical. This is the bug fix and can land on its own.
3. **Extract `prune_entity` from the Phase-1 path, behaviour unchanged.** Phase-1 output must stay bit-identical.
4. **Move the cascade onto `prune_entity` with `max_carves=0` and no admission.** Group output must stay bit-identical to step 2.
5. **Change the cascade's behaviour, one flag at a time:** carving (`max_carves=1`), the size minimum on carved entities, the z-split, then admission. Measure each on a real ROI (coverage, coherence, RCTD max weight), and check runtime and peak memory with `scripts/bench_stages.py`.
6. **SEG path reads `cfg.group`.** Remove the hard-coded literals.
7. **Retire the legacy Group path** (`annotate_unassigned_components_fast` with `prune_genes_by_pmi_greedy`), or route it through `prune_entity`.

## Open questions

- Should the cascade use Phase-1's admission settings (`veto_mode`, `mean_admit_threshold`, `aggregator_percentile`)? The cascade has no nucleus to anchor identity, so a stricter admission may be warranted.
- Should 1c be allowed to recurse more than once (`max_carves > 1`)?
- Explicit zeros: should a pair stored as 0 count as a conflict, or be skipped like a missing pair? This affects both paths equally, for example on panels with zeroed unsupported pairs.
- Phase-1 sends rejected transcripts to unassigned while the cascade sends them back to its pool. Should Phase-1 rejects instead feed the cascade pool directly?

## Evidence

Prototype runs on an internal dataset compared cell-style carving, cell-style carving with a sibling size minimum, and the repo's window mode. Cell-style carving gave more partials with higher coherence and fewer unassigned transcripts, and left whole-cell coherence unchanged. The numbers are kept outside this repository because the dataset is unpublished.

The prototype patches are environment flags in a scratch copy of TRACER; none are in the repo. The flags are:

- `TRACER_CASCADE_SYM`
- `TRACER_CASCADE_PRUNE`
- `TRACER_CASCADE_SIB_MIN`
- `STITCH_CUNION_MIN`
- `STITCH_CELL_DCMIN`
- `STITCH_CUM_DCMIN`
- `TRACER_SIM_REEXPAND`
