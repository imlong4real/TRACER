# Figure 2 — PDAC Xenium: biological insight from TRACER

Dataset: `Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE` (Human Multi-Tissue &
Cancer panel, **380 genes**, 20.73 M transcripts). TRACER assignment from
`pdac_io_partition_sequential.parquet` (joined to the raw transcript table on
`transcript_id`; original 10x `cell_id` = original segmentation, `label` =
TRACER assignment). Reference for label transfer: PDAC scRNA atlas
`pk_all_30k_stratified.h5ad` (10 cell types).

All panels share the dark "Nature Methods" aesthetic in `fig2_style.py` and live
in `outputs/` (PNG + SVG, ≥300 dpi).

---

## 1. Ranked biological stories (strongest → weakest)

**① TRACER removes cross-lineage transcript admixture and recovers cells.**
Median cross-lineage contamination drops **0.200 → 0.125** (original → TRACER
complete; −37%) and label-transfer confidence rises **0.75 → 0.81**. Cleanup is
strongest for the small immune/vascular cells most corrupted by neighbouring
epithelium: **Macrophage 0.33→0.13, B 0.20→0.00, Endothelial 0.33→0.17, T
0.13→0.00**. TRACER additionally yields **108,586** confidently-typed *partial*
cells absent from the original segmentation (enriched for stroma: Fibroblast
40.7k, Stellate 13.5k; and immune: Macrophage 10.9k, B 9.1k).
→ *Strong, quantified, panel-independent. The core methods result.*

**② TRACER recovers immunoregulatory VSIG4⁺ macrophages and their T-cell niches.**
VSIG4 (immunosuppressive TAM marker) is badly mis-attributed in the original
segmentation — only **17%** of VSIG4⁺ cells are macrophages (46% wrongly
fibroblast, 25% tumour). TRACER nearly doubles macrophage specificity to **34%**
(now the top type). VSIG4⁺ TAMs sit close to T cells (median **117 → 72 µm** to
nearest T after TRACER), consistent with immunosuppressive niches.
→ *Strong, biologically meaningful, directly tied to story ①.*

**③ TRACER sharpens plausible local immune interactions and purifies ligand senders.**
Spatial LR-adjacency enrichment (permutation null) vs original: co-stimulatory
**sender-lineage purity rises** (CD86→CD28 **0.51→0.72**, CD86→CTLA4 0.57→0.70;
mean 49%→56%), and plausible chemokine interactions strengthen (CCL19→CCR7 z
36→58; CCL5→CCR7, CD80→CTLA4, CXCL10→CXCR4 all up). Low-abundance pairs
(CD274→PDCD1, CXCL9→CXCR4) become too sparse to call after de-mixing (flagged).
→ *Moderate; supports ①/②. Main-or-supplement.*

**④ (Negative) "EMT recovery" is largely a segmentation artifact.**
With no epithelial EMT genes on the panel (VIM/CDH1/2/ZEB/TWIST absent;
FN1/SPARC/TGFB1 shared with CAFs), "EMT-high" ductal cells are the most
cross-lineage-contaminated (25% vs 5%), and TRACER *reduces* the EMT-high
fraction **5.0% → 3.2%** — i.e. much apparent EMT is fibroblast spillover that
TRACER removes. *Honest negative → supplement, not a main EMT-recovery claim.*

**⑤ (Negative) Hypoxia–VISTA axis is weak on this panel.**
Hypoxia surrogate is limited to **VEGFA + HIF1A** (core hypoxia genes
CA9/SLC2A1/LDHA/NDRG1 absent). VSIR/VSIG4 vs local-hypoxia association is
negligible (grid Spearman ρ ≈ 0.03–0.08; rank-biserial < 0.04). *Honest
negative → supplement, with the panel-coverage caveat.*

---

## 2. Best ROI candidates

| ROI | Window (µm) | Why | Used in |
|-----|-------------|-----|---------|
| **R1 — immune–tumour 3D ROI** | x0≈7097, y0≈1569, **80×80** | 46 reconstructed partial immune cells embedded among tumour; new contacts | `fig2_3d_roi` |
| **R2 — VSIG4⁺ TAM / T niche** | x0≈7697, y0≈1211, **450×450** | dense VSIG4⁺ TAMs co-localising with T cells | `fig2_vista_vsig4b` |
| R3 — partial-immune hotspot | x0≈2357, y0≈1749, 170×170 | very high partial-immune density (carpet; good for zoomed insets) | candidate |

(ROIs are auto-selected in the scripts; coordinates are in global Xenium µm.)

---

## 3. Main figure vs supplement

**Main figure (recommended panel order):**
1. `pdac_full_tier_a` / `pdac_full_tier_b` — transcript-fate alluvials (phase ladder)
2. `pdac_zstack_reconstruction` — 3D z-stack of complete vs reconstructed partial cells
3. `fig2_composition_cleanliness` — **story ①** (composition + admixture cleanup + new cells)
4. `fig2_3d_roi` — **story ①/②** Open3D cell-hull ROI: partials + new immune–tumour contacts
5. `fig2_vista_vsig4` — **story ②** VSIG4⁺ TAM recovery + T-cell niche

**Supplement:**
- `fig2_cci` — **story ③** spatial LR enrichment & sender purity (promote to main if space)
- `fig2_supp_hypoxia_emt` — **stories ④/⑤** honest negatives with panel-coverage caveats

---

## 4. Methods (brief)
- **Matrices** (`build_matrices.py`): transcripts qv≥20, real genes only; grouped by
  original `cell_id` and TRACER `label`; per-cell centroid, depth, entity class
  (complete/partial/neighboring from `etype_at_finalize`).
- **Label transfer** (`scripts/label_transfer_spatial.py`, cosine-centroid softmax,
  379 shared genes) onto both matrices via the atlas reference.
- **Cleanliness** = 1 − dominant-lineage marker fraction over 9 in-panel lineage panels.
- **VISTA/hypoxia** (`vista_hypoxia.py`): scanpy `score_genes`, k=15 spatial-NN local
  field, MWU + grid Spearman; KDTree proximity.
- **CCI** (`cci_spatial.py`): directed spatial contacts ≤30 µm, degree-preserving
  receptor-label permutation (100×) → z/fold; sender-lineage purity.
- **3D ROI** (`build_3d_roi.py`): Open3D convex hull + Loop subdivision + Laplacian
  smoothing per cell (filament offscreen unavailable headless → matplotlib 3D render).

## 5. Key caveats
- 380-gene targeted panel: hypoxia (2 genes) and EMT (CAF-shared genes) are
  under-powered — reported as honest negatives, not forced into the main figure.
- Partial cells are small (median 8 counts); analyses use confidence/depth filters.
- Label transfer is centroid-cosine (targeted-panel appropriate), not a deep model.

## 6. Reproduce
```bash
python scripts/reproducibility/fig2/build_matrices.py
python scripts/label_transfer_spatial.py --query_h5ad .../original_cells.h5ad \
  --reference_type cervical_atera_plus_scrna --reference_h5ad .../atlas_ref_for_lt.h5ad \
  --label_harmonization passthrough --min_transcripts 5 --max_transcripts 100000 \
  --outdir .../lt_original --sample_prefix pdac_orig         # (+ tracer)
python scripts/reproducibility/fig2/analysis_core.py
python scripts/reproducibility/fig2/vista_hypoxia.py
python scripts/reproducibility/fig2/cci_spatial.py
python scripts/reproducibility/fig2/build_3d_roi.py
python scripts/reproducibility/fig2/fig_composition.py
python scripts/reproducibility/fig2/fig_vista.py
python scripts/reproducibility/fig2/fig_cci.py
python scripts/reproducibility/fig2/fig_supplement.py
```
