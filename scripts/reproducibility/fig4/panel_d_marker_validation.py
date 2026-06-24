#!/usr/bin/env python3
"""Panel D — canonical marker validation (transcriptional coherence).

For every method we compute mean CP10k-log1p expression of canonical kidney
lineage markers within each reconstructed lineage (whole-transcriptome
matrices). A coherent reconstruction shows each marker block lighting up on
its own lineage's diagonal.

Layout: a method x (marker-gene grouped-by-lineage) heatmap, one row block per
method, columns = genes grouped by their target lineage, color = z-scored mean
expression across lineages (per gene, within method).

Message: reconstructed profiles are transcriptionally coherent, not just
spatially plausible — and TRACER 2um/8um match 10x/bin2cell marker structure.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import fig4_config as C
import utils as U


def _zrows(df):
    mu = df.mean(1).to_numpy()[:, None]
    sd = df.std(1).replace(0, 1).to_numpy()[:, None]
    return (df - mu) / sd


def make():
    plt = U.setup_style()
    genes, gene_lineage = [], []
    for lin in C.LINEAGES:
        for g in C.MARKERS[lin]:
            genes.append(g); gene_lineage.append(lin)
    gl = pd.Series(gene_lineage, index=genes)

    # method -> DataFrame (gene x lineage) mean expression
    per_method = {}
    long_rows = []
    for m in C.METHOD_ORDER:
        expr = U.lineage_mean_expression(m, genes)         # genes x lineage
        expr = expr.reindex(index=[g for g in genes if g in expr.index],
                            columns=C.LINEAGES)
        per_method[m] = expr
        for g in expr.index:
            for lin in C.LINEAGES:
                long_rows.append({"method": C.METHOD_DISPLAY[m], "gene": g,
                                  "target_lineage": gl[g], "lineage": lin,
                                  "mean_log1p_cp10k": expr.loc[g, lin]})
    pd.DataFrame(long_rows).to_csv(C.SRCDIR / "panel_D_marker_expression.csv", index=False)

    genes_present = [g for g in genes if g in per_method[C.METHOD_ORDER[0]].index]
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.4), sharey=True)
    for ax, m in zip(axes, C.METHOD_ORDER):
        Z = _zrows(per_method[m].reindex(genes_present))
        im = ax.imshow(Z.to_numpy(), aspect="auto", cmap="RdBu_r",
                       vmin=-2, vmax=2)
        ax.set_xticks(range(len(C.LINEAGES)))
        ax.set_xticklabels([C.LINEAGE_DISPLAY[l] for l in C.LINEAGES],
                           rotation=90, fontsize=6)
        ax.set_title(C.METHOD_DISPLAY[m], fontsize=9)
        ax.set_yticks(range(len(genes_present)))
        if m == C.METHOD_ORDER[0]:
            ax.set_yticklabels(genes_present, fontsize=5)
        # lineage block separators
        bounds = np.where(gl.reindex(genes_present).values[:-1]
                          != gl.reindex(genes_present).values[1:])[0]
        for b in bounds:
            ax.axhline(b + 0.5, color="k", lw=0.4)
        ax.tick_params(length=0)
    cax = fig.add_axes([1.0, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="row z-score (mean log1p CP10k)")
    # lineage color strip on the left of first axis
    for i, g in enumerate(genes_present):
        axes[0].add_patch(plt.Rectangle((-1.4, i - 0.5), 0.6, 1.0,
                          color=C.PALETTE[gl[g]], clip_on=False, lw=0))
    axes[0].set_xlim(left=-1.5)
    fig.suptitle("Canonical lineage markers across methods (whole-transcriptome)",
                 fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.16, wspace=0.06)
    U.save_fig(fig, "panel_D_marker_validation")


if __name__ == "__main__":
    make()
