#!/usr/bin/env python3
"""Panel F — resolution / computational tradeoff (compact summary).

Sub-panels:
  1. Runtime (bars) + peak memory (overlaid line, twin axis) — compute methods
     only (10x is spaceranger output, not benchmarked here).
  2. Number of reconstructed profiles / cells — all methods.
  3. Genes-per-profile and UMIs-per-profile FULL distributions as raincloud
     half-violins (log10 axis) — all methods.
  4. Fraction of input bins left unassigned — per bin-based method.

Framing: profile *number* and unassigned-bin fraction reflect the granularity
of the starting bins, NOT reconstruction quality.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import fig4_config as C
import utils as U


def _bench(method):
    p = C.BENCH_METRICS.get(method)
    return json.loads(p.read_text()) if (p and p.exists()) else {}


def _unassigned_fraction(method):
    """(frac_unassigned, n_assigned, n_input, bin_um) for bin-based methods."""
    if method == "tracer_2um":
        b = _bench("tracer_2um"); n_in = b.get("input_bin_count")
        n_as = pd.read_parquet(C.BIN_TO_PROFILE["tracer_2um"], columns=["bin_id"])["bin_id"].nunique()
        return (1 - n_as / n_in, n_as, n_in, 2)
    if method == "tracer_8um":
        b = _bench("tracer_8um"); n_in = b.get("input_bin_count")
        n_as = pd.read_parquet(C.BIN_TO_PROFILE["tracer_8um"], columns=["bin_id"])["bin_id"].nunique()
        return (1 - n_as / n_in, n_as, n_in, 8)
    if method == "bin2cell":
        b = _bench("bin2cell"); n_in = b.get("n_input_bins"); n_as = b.get("n_bins_assigned")
        if n_in and n_as:
            return (1 - n_as / n_in, n_as, n_in, 2)
    return (np.nan, np.nan, np.nan, np.nan)


def build_table() -> pd.DataFrame:
    wt = json.loads((C.WT / "whole_transcriptome_stats.json").read_text())
    skey = {"10x_segmented": "10x_segmented", "bin2cell": "bin2cell_2um",
            "tracer_2um": "tracer_2um", "tracer_8um": "tracer_8um"}
    rows = []
    for m in C.METHOD_ORDER:
        b = _bench(m); ws = wt.get(skey[m], {})
        fu, n_as, n_in, binum = _unassigned_fraction(m)
        rows.append({
            "method": C.METHOD_DISPLAY[m],
            "runtime_min": round(b["total_wallclock_s"] / 60, 1) if b.get("total_wallclock_s") else np.nan,
            "peak_rss_gb": b.get("peak_rss_gb", np.nan),
            "n_profiles": ws.get("n_profiles", np.nan),
            "median_genes_per_profile": ws.get("median_genes_per_profile", np.nan),
            "median_umis_per_profile": ws.get("median_umis_per_profile", np.nan),
            "input_bin_size_um": binum,
            "n_input_bins": n_in, "n_assigned_bins": n_as,
            "frac_unassigned_bins": round(fu, 4) if fu == fu else np.nan,
        })
    return pd.DataFrame(rows).set_index("method")


def make():
    plt = U.setup_style()
    df = build_table()
    df.to_csv(C.SRCDIR / "panel_F_resolution_tradeoff.csv")
    disp = {m: C.METHOD_DISPLAY[m] for m in C.METHOD_ORDER}
    col = {m: C.METHOD_COLOR[m] for m in C.METHOD_ORDER}

    fig = plt.figure(figsize=(15, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.85, 1.5, 0.95], wspace=0.42)

    # (1) runtime bars + memory line  — compute methods only (no 10x)
    cm = ["bin2cell", "tracer_2um", "tracer_8um"]
    ax = fig.add_subplot(gs[0, 0])
    rt = [df.loc[disp[m], "runtime_min"] for m in cm]
    mem = [df.loc[disp[m], "peak_rss_gb"] for m in cm]
    xs = np.arange(len(cm))
    ax.bar(xs, rt, color=[col[m] for m in cm], width=0.62, zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels([disp[m] for m in cm], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Runtime (min)"); ax.set_title("Runtime & peak memory", fontsize=8.5)
    for i, v in enumerate(rt):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
    ax2 = ax.twinx()
    ax2.plot(xs, mem, "o-", color="#222", lw=1.4, ms=5, zorder=5)
    ax2.set_ylabel("Peak memory (GB)"); ax2.set_ylim(0, max(mem) * 1.4)
    ax2.spines["right"].set_visible(True)
    for i, v in enumerate(mem):
        ax2.text(i, v, f" {v:.1f} GB", ha="left", va="bottom", fontsize=6, color="#222")

    # (2) n profiles  — all methods
    axn = fig.add_subplot(gs[0, 1])
    npr = [df.loc[disp[m], "n_profiles"] for m in C.METHOD_ORDER]
    axn.bar(range(4), npr, color=[col[m] for m in C.METHOD_ORDER], width=0.7)
    axn.set_xticks(range(4)); axn.set_xticklabels([disp[m] for m in C.METHOD_ORDER], rotation=35, ha="right", fontsize=6.5)
    axn.set_title("Reconstructed\nprofiles / cells", fontsize=8.5)
    for i, v in enumerate(npr):
        axn.text(i, v, f"{v/1000:.0f}k", ha="center", va="bottom", fontsize=6)
    axn.margins(y=0.18)

    # (3) genes & UMIs per profile — raincloud half-violins (log10), all methods
    axd = fig.add_subplot(gs[0, 2])
    counts = {m: U.per_profile_counts(m) for m in C.METHOD_ORDER}
    gene_data = [counts[m]["n_genes"].to_numpy() for m in C.METHOD_ORDER]
    umi_data = [counts[m]["n_umis"].to_numpy() for m in C.METHOD_ORDER]
    cols4 = [col[m] for m in C.METHOD_ORDER]
    U.half_violin(axd, gene_data, list(range(4)), cols4, log=True, width=0.8)
    U.half_violin(axd, umi_data, list(range(6, 10)), cols4, log=True, width=0.8)
    axd.set_xticks(list(range(4)) + list(range(6, 10)))
    axd.set_xticklabels([disp[m] for m in C.METHOD_ORDER] * 2, rotation=40, ha="right", fontsize=6)
    axd.set_ylabel("log10(count + 1)")
    axd.set_title("Genes / profile          UMIs / profile", fontsize=8.5)
    axd.axvline(5, color="#999", lw=0.6, ls="--")

    # (4) unassigned bins fraction — 10x / bin2cell / TRACER 2µm on the SAME
    #     filtered 2µm bin set (common denominator), plus TRACER 8µm (8µm bins).
    axu = fig.add_subplot(gs[0, 3])
    u2 = pd.read_csv(C.SRCDIR / "unassigned_bins_2um.csv").set_index("method")
    key2disp = {"10x": "10x_segmented", "bin2cell": "bin2cell", "TRACER 2 µm": "tracer_2um"}
    fu_methods = ["10x", "bin2cell", "TRACER 2 µm"]
    fu = [u2.loc[k, "frac_unassigned"] * 100 for k in fu_methods]
    barcol = [col[key2disp[k]] for k in fu_methods]
    labs = [f"{k}\n(2µm bins)" for k in fu_methods]
    # append TRACER 8µm (its own 8µm grid)
    fu.append(df.loc[disp["tracer_8um"], "frac_unassigned_bins"] * 100)
    barcol.append(col["tracer_8um"]); labs.append("TRACER 8 µm\n(8µm bins)")
    axu.bar(range(len(fu)), fu, color=barcol, width=0.66)
    axu.set_xticks(range(len(fu))); axu.set_xticklabels(labs, rotation=0, fontsize=5.0)
    axu.set_title("Input bins left\nunassigned (%)", fontsize=8.5)
    for i, v in enumerate(fu):
        axu.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=6)
    axu.margins(y=0.18)

    fig.suptitle("Resolution / compute tradeoff  —  profile count & unassigned-bin fraction reflect input bin granularity, not quality",
                 fontsize=8.5)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.26)
    U.save_fig(fig, "panel_F_resolution_tradeoff")
    print(df.to_string())


if __name__ == "__main__":
    make()
