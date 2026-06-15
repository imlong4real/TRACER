#!/usr/bin/env python3
"""
Plot a chromosome-ordered TRACER expression heatmap for GBM Slide 3.

The default path is offline/reproducible: read the TRACER finetuned AnnData
object, read a cached GRCh38 gene-position table, subset to annotated cancer
and oligodendrocyte cells, and render a rasterized heatmap.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Iterable

GENE_FEATURE_TYPE = "Gene Expression"
DEFAULT_CHROMOSOME_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
DEFAULT_CELL_TYPE_PREFIXES = ("cancer",)
DEFAULT_CELL_TYPE_EXACT = ("oligo",)
DEFAULT_FOCUS_CHROMOSOMES = ("2", "7", "10", "14")
DEFAULT_BOOTSTRAP_ITERATIONS = 500
DEFAULT_HCLUST_K = (4, 5)
SMALL_CLUSTER_N = 100
ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/id"
DEFAULT_THREAD_LIMIT = "8"


def _configure_thread_env() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, DEFAULT_THREAD_LIMIT)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_h5ad() -> Path:
    return (
        _repo_root()
        / "tutorials"
        / "gbm"
        / "output"
        / "slide3_profile_comparison"
        / "adata_ft.h5ad"
    )


def _default_gene_positions() -> Path:
    return _repo_root() / "tutorials" / "gbm" / "data" / "gene_positions_grch38.tsv"


def _default_outdir() -> Path:
    return _repo_root() / "tutorials" / "gbm" / "output" / "slide3_chromosome_heatmap"


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _btc_gbm_root_candidates(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in (repo_root, *repo_root.parents):
        if path.name == "BTC_GBM":
            candidates.append(path)

    if len(repo_root.parents) > 1:
        candidates.append(repo_root.parents[1])

    candidates.append(Path("/mnt/storage/dept/medonc/beroukhim/youyun/BTC_GBM"))
    return _unique_paths(candidates)


def _feature_tsv_candidates() -> list[Path]:
    repo_root = _repo_root()
    rel = Path(
        "data/xenium/output-XETG00323__0023274__Patient4__20241004__181038/"
        "cell_feature_matrix/features.tsv.gz"
    )
    return _unique_paths(
        [
            *(root / rel for root in _btc_gbm_root_candidates(repo_root)),
            repo_root / "tutorials" / "gbm" / "data" / "features.tsv.gz",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a chromosome-ordered heatmap for TRACER-finetuned GBM cells."
    )
    parser.add_argument(
        "--h5ad", default=str(_default_h5ad()), help="TRACER finetuned AnnData h5ad."
    )
    parser.add_argument(
        "--features-tsv",
        default=None,
        help="Xenium cell_feature_matrix/features.tsv.gz. Auto-discovered by default.",
    )
    parser.add_argument(
        "--gene-positions",
        default=str(_default_gene_positions()),
        help="Cached GRCh38 gene-position TSV with gene_id, gene, chromosome, start, end.",
    )
    parser.add_argument(
        "--outdir", default=str(_default_outdir()), help="Output directory."
    )
    parser.add_argument(
        "--cell-type-prefix",
        action="append",
        default=list(DEFAULT_CELL_TYPE_PREFIXES),
        help="Keep annotated cell types starting with this prefix. May be repeated.",
    )
    parser.add_argument(
        "--cell-type-exact",
        action="append",
        default=list(DEFAULT_CELL_TYPE_EXACT),
        help="Keep annotated cell types exactly matching this value. May be repeated.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional deterministic subsample size for preview renders. Default keeps all selected cells.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for --max-cells subsampling."
    )
    parser.add_argument(
        "--clip-z",
        type=float,
        default=2.0,
        help="Symmetric clipping limit after gene-wise z scoring.",
    )
    parser.add_argument(
        "--leiden-resolution",
        type=float,
        default=0.1,
        help="Resolution for Leiden re-clustering on the cancer_* + oligo subset.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
        help="Bootstrap iterations for chromosome-level cluster-vs-oligo confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for chromosome-level bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--cnv-effect-threshold",
        type=float,
        default=0.5,
        help="Minimum oligo-centered chromosome effect required for gain/loss calls.",
    )
    parser.add_argument(
        "--cnv-min-genes-per-chromosome",
        type=int,
        default=5,
        help="Flag chromosome calls with fewer positioned panel genes as low confidence.",
    )
    parser.add_argument(
        "--hclust-k",
        type=int,
        action="append",
        default=None,
        help="Cancer-only hierarchical cluster count to report. May be repeated.",
    )
    parser.add_argument(
        "--hclust-n-pcs",
        type=int,
        default=30,
        help="Number of PCA components used for graph-constrained hierarchical clustering.",
    )
    parser.add_argument(
        "--hclust-n-neighbors",
        type=int,
        default=30,
        help="Number of kNN edges used to constrain hierarchical clustering.",
    )
    parser.add_argument(
        "--hclust-n-jobs",
        type=int,
        default=1,
        help="Worker count for hierarchical-clustering kNN graph construction.",
    )
    parser.add_argument(
        "--download-gene-positions",
        action="store_true",
        help="Create/update --gene-positions from Ensembl REST using the features TSV, then continue.",
    )
    return parser.parse_args()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_features_tsv(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"features TSV not found: {path}")
        return path

    for candidate in _feature_tsv_candidates():
        if candidate.exists():
            return candidate

    candidates = "\n  ".join(str(path) for path in _feature_tsv_candidates())
    raise FileNotFoundError(
        "Could not auto-discover Xenium features.tsv.gz. Checked:\n  " + candidates
    )


def _read_gene_features(features_tsv: Path):
    import pandas as pd

    opener = gzip.open if features_tsv.suffix == ".gz" else open
    rows: list[tuple[str, str, str]] = []
    with opener(features_tsv, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            rows.append((fields[0], fields[1], fields[2]))

    features = pd.DataFrame(rows, columns=["gene_id", "gene", "feature_type"])
    return features[features["feature_type"] == GENE_FEATURE_TYPE].copy()


def _normalize_chromosome(value: object) -> str:
    chrom = str(value).strip()
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def _chromosome_rank(chromosome: str) -> int:
    try:
        return DEFAULT_CHROMOSOME_ORDER.index(_normalize_chromosome(chromosome))
    except ValueError:
        return len(DEFAULT_CHROMOSOME_ORDER)


def _download_gene_positions(gene_features, output_path: Path) -> None:
    import pandas as pd
    import requests

    ids = sorted(gene_features["gene_id"].dropna().astype(str).unique())
    if not ids:
        raise ValueError("No Ensembl gene IDs found in gene-expression features.")

    _log(
        f"Downloading GRCh38 gene positions from Ensembl REST for {len(ids):,} genes ..."
    )
    response = requests.post(
        ENSEMBL_LOOKUP_URL,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        data=json.dumps({"ids": ids}),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    feature_name_by_id = (
        gene_features.drop_duplicates("gene_id").set_index("gene_id")["gene"].to_dict()
    )
    rows = []
    for gene_id in ids:
        item = payload.get(gene_id)
        if not item:
            continue
        rows.append(
            {
                "gene_id": gene_id,
                "gene": feature_name_by_id.get(gene_id, item.get("display_name")),
                "ensembl_display_name": item.get("display_name"),
                "chromosome": _normalize_chromosome(item.get("seq_region_name")),
                "start": int(item.get("start")),
                "end": int(item.get("end")),
                "strand": int(item.get("strand", 0)),
                "source": "Ensembl REST lookup/id",
                "assembly": item.get("assembly_name", "GRCh38"),
            }
        )

    positions = pd.DataFrame(rows)
    if positions.empty:
        raise ValueError("Ensembl REST returned no gene-position records.")

    positions["chromosome_rank"] = positions["chromosome"].map(_chromosome_rank)
    positions = positions.sort_values(["chromosome_rank", "start", "end", "gene"]).drop(
        columns=["chromosome_rank"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(output_path, sep="\t", index=False)
    _log(f"Wrote gene-position cache: {output_path}")


def _read_gene_positions(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(
            f"Gene-position cache not found: {path}\n"
            "Create it once with:\n"
            "  conda run -n segmentation python tutorials/gbm/plot_chromosome_heatmap.py "
            "--download-gene-positions --max-cells 1000"
        )

    positions = pd.read_csv(path, sep="\t")
    required = {"gene_id", "gene", "chromosome", "start", "end"}
    missing = sorted(required - set(positions.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    positions = positions.copy()
    positions["chromosome"] = positions["chromosome"].map(_normalize_chromosome)
    positions["start"] = pd.to_numeric(positions["start"], errors="coerce")
    positions["end"] = pd.to_numeric(positions["end"], errors="coerce")
    positions = positions.dropna(subset=["chromosome", "start", "end", "gene"])
    return positions


def _selected_cell_mask(obs, *, prefixes: Iterable[str], exact: Iterable[str]):
    import pandas as pd

    if "cell_type" not in obs.columns:
        raise ValueError(
            "AnnData obs has no 'cell_type' column. Run compare_profiles.py with annotations first."
        )

    cell_type = obs["cell_type"].astype("string")
    mask = pd.Series(False, index=obs.index)
    for prefix in prefixes:
        mask = mask | cell_type.str.startswith(prefix, na=False)
    exact_values = {value for value in exact if value}
    if exact_values:
        mask = mask | cell_type.isin(exact_values)
    return mask


def _prefix_cell_mask(obs, *, prefixes: Iterable[str]):
    import pandas as pd

    if "cell_type" not in obs.columns:
        raise ValueError("AnnData obs has no 'cell_type' column.")

    cell_type = obs["cell_type"].astype("string")
    mask = pd.Series(False, index=obs.index)
    for prefix in prefixes:
        mask = mask | cell_type.str.startswith(prefix, na=False)
    return mask.fillna(False).astype(bool).to_numpy()


def _oligo_cell_mask(obs, *, oligo_label: str = "oligo"):
    cell_type = obs["cell_type"].astype("string").fillna("")
    return (cell_type == oligo_label).to_numpy()


def _unique_positive_ints(values, *, default: Iterable[int]) -> list[int]:
    unique: list[int] = []
    for value in values if values is not None else default:
        value = int(value)
        if value < 2:
            raise ValueError("Cluster counts must be >= 2.")
        if value not in unique:
            unique.append(value)
    return unique


def _as_dense_float_array(matrix):
    import numpy as np
    from scipy import sparse

    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _counts_matrix(adata, *, selected_cells, ordered_genes):
    sub = adata[list(selected_cells), ordered_genes].copy()
    if "counts" in sub.layers:
        counts = sub.layers["counts"].copy()
    else:
        counts = sub.X.copy()
    return sub, _as_dense_float_array(counts)


def _prepare_log_expression_matrix(
    adata,
    *,
    selected_cells,
    ordered_genes,
):
    import numpy as np
    from scipy import sparse

    selected_cells = list(selected_cells)
    sub = adata[selected_cells, ordered_genes].copy()
    if "counts" in sub.layers:
        counts = sub.layers["counts"].copy()
    else:
        counts = sub.X.copy()

    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    scale[nonzero] = 1e4 / row_sums[nonzero]

    if sparse.issparse(counts):
        normalized = counts.multiply(scale[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
        sub.X = normalized
    else:
        normalized = np.asarray(counts, dtype=np.float32) * scale[:, None]
        normalized = np.log1p(normalized)
        sub.X = normalized

    return sub, _as_dense_float_array(sub.X)


def _prepare_heatmap_matrix(
    adata,
    *,
    selected_cells,
    ordered_genes,
    max_cells: int | None,
    seed: int,
    clip_z: float,
):
    import numpy as np

    selected_cells = list(selected_cells)
    if max_cells is not None and len(selected_cells) > max_cells:
        rng = np.random.default_rng(seed)
        sampled = set(rng.choice(selected_cells, size=max_cells, replace=False).tolist())
        selected_cells = [cell for cell in selected_cells if cell in sampled]

    sub, matrix = _prepare_log_expression_matrix(
        adata, selected_cells=selected_cells, ordered_genes=ordered_genes
    )

    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    matrix = np.clip(matrix, -clip_z, clip_z)
    return sub, matrix


def _cluster_selected_subset(adata_sel, *, resolution: float, seed: int):
    import scanpy as sc

    n_comps = max(1, min(30, adata_sel.n_vars - 1, adata_sel.n_obs - 1))
    sc.pp.pca(adata_sel, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(adata_sel, random_state=seed)
    sc.tl.umap(adata_sel, random_state=seed)
    sc.tl.leiden(
        adata_sel,
        resolution=resolution,
        random_state=seed,
        key_added="heatmap_leiden",
    )
    return adata_sel


def _add_hierarchical_clusters(
    adata_sel,
    gene_score_matrix,
    *,
    cancer_mask,
    ks: Iterable[int],
    n_pcs: int,
    n_neighbors: int,
    n_jobs: int,
    seed: int,
):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.neighbors import kneighbors_graph

    cancer_mask = np.asarray(cancer_mask, dtype=bool)
    n_cancer = int(cancer_mask.sum())
    if n_cancer < 2:
        raise ValueError("Hierarchical clustering requires at least two cancer cells.")

    max_k = max(ks)
    if n_cancer < max_k:
        raise ValueError(
            f"Hierarchical clustering requested k={max_k}, "
            f"but only {n_cancer} cancer cells are available."
        )

    scores = np.asarray(gene_score_matrix[cancer_mask, :], dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    n_components = max(1, min(int(n_pcs), scores.shape[0] - 1, scores.shape[1]))
    pcs = PCA(n_components=n_components, random_state=seed).fit_transform(scores)

    n_graph_neighbors = max(1, min(int(n_neighbors), n_cancer - 1))
    connectivity = kneighbors_graph(
        pcs,
        n_neighbors=n_graph_neighbors,
        mode="connectivity",
        include_self=False,
        n_jobs=int(n_jobs),
    )

    keys: list[str] = []
    obs_index = adata_sel.obs.index
    for k in ks:
        key = f"hclust_k{k}"
        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                connectivity=connectivity,
                linkage="ward",
                metric="euclidean",
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=k,
                connectivity=connectivity,
                linkage="ward",
                affinity="euclidean",
            )
        raw_labels = model.fit_predict(pcs)
        labels = pd.Series("oligo_ref", index=obs_index, dtype="string")
        labels.iloc[np.flatnonzero(cancer_mask)] = [
            str(label) for label in raw_labels
        ]
        adata_sel.obs[key] = labels
        keys.append(key)
    return keys


def _ordered_cluster_categorical(values):
    import pandas as pd

    raw = pd.Series(values).astype(str)

    def _key(label: str):
        try:
            return (0, int(label))
        except ValueError:
            return (1, label)

    categories = sorted(raw.unique().tolist(), key=_key)
    return pd.Categorical(raw, categories=categories, ordered=True)


def _prepare_plot_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache = output_path.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))


def _cluster_palette(n: int):
    import matplotlib.pyplot as plt

    base = plt.get_cmap("tab10").colors
    if n <= len(base):
        return list(base[:n])
    extra = plt.get_cmap("tab20").colors
    palette = list(base) + [c for c in extra if c not in base]
    if n <= len(palette):
        return palette[:n]
    cmap = plt.get_cmap("hsv", n)
    return [cmap(i) for i in range(n)]


def _plot_umap_panels(adata_sel, output_path: Path, *, resolution: float) -> None:
    _prepare_plot_output(output_path)

    import matplotlib.pyplot as plt
    import numpy as np

    if "X_umap" not in adata_sel.obsm:
        raise ValueError(
            "adata_sel is missing obsm['X_umap']; clustering must run first."
        )

    coords = adata_sel.obsm["X_umap"]
    cluster_cat = _ordered_cluster_categorical(adata_sel.obs["heatmap_leiden"].tolist())
    cluster_colors = _cluster_palette(len(cluster_cat.categories))

    cell_type_cat = adata_sel.obs["cell_type"].astype("string").fillna("unannotated")
    cell_type_categories = sorted(cell_type_cat.unique().tolist())
    cell_type_colors = _cluster_palette(len(cell_type_categories))
    cell_type_color_map = dict(zip(cell_type_categories, cell_type_colors))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.5))

    for code, label in enumerate(cluster_cat.categories):
        mask = cluster_cat.codes == code
        ax_l.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            c=[cluster_colors[code]],
            label=f"{label} (n={int(mask.sum()):,})",
            linewidths=0,
            rasterized=True,
        )
    ax_l.set_title(
        f"Leiden (resolution={resolution:g}, k={len(cluster_cat.categories)})"
    )

    for label in cell_type_categories:
        mask = (cell_type_cat == label).to_numpy()
        ax_r.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            c=[cell_type_color_map[label]],
            label=f"{label} (n={int(mask.sum()):,})",
            linewidths=0,
            rasterized=True,
        )
    ax_r.set_title("cell_type annotation")

    for ax in (ax_l, ax_r):
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=7, frameon=False, markerscale=2)

    fig.suptitle(f"cancer_* + oligo subset UMAP ({adata_sel.n_obs:,} cells)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_hclust_umap_panels(adata_sel, output_path: Path, *, hclust_keys) -> None:
    _prepare_plot_output(output_path)

    import matplotlib.pyplot as plt
    import numpy as np

    if not hclust_keys:
        return
    if "X_umap" not in adata_sel.obsm:
        raise ValueError("adata_sel is missing obsm['X_umap'].")

    coords = adata_sel.obsm["X_umap"]
    n_panels = len(hclust_keys)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.2 * n_panels, 5.5),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, key in zip(axes, hclust_keys):
        labels = adata_sel.obs[key].astype("string").fillna("oligo_ref")
        oligo_mask = (labels == "oligo_ref").to_numpy()
        if oligo_mask.any():
            ax.scatter(
                coords[oligo_mask, 0],
                coords[oligo_mask, 1],
                s=4,
                c=["#bdbdbd"],
                label=f"oligo_ref (n={int(oligo_mask.sum()):,})",
                linewidths=0,
                rasterized=True,
            )

        cancer_labels = sorted(
            [label for label in labels.unique().tolist() if label != "oligo_ref"],
            key=lambda label: (
                (0, int(label)) if str(label).isdigit() else (1, str(label))
            ),
        )
        colors = _cluster_palette(len(cancer_labels))
        for code, label in enumerate(cancer_labels):
            mask = (labels == label).to_numpy()
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=4,
                c=[colors[code]],
                label=f"{label} (n={int(mask.sum()):,})",
                linewidths=0,
                rasterized=True,
            )

        ax.set_title(key)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=7, frameon=False, markerscale=2)

    fig.suptitle("Cancer-only hierarchical clusters on cancer_* + oligo UMAP")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _chromosome_groups(ordered_positions):
    chromosomes = ordered_positions["chromosome"].tolist()
    if not chromosomes:
        return []

    groups: list[tuple[str, int, int]] = []
    start = 0
    current = chromosomes[0]
    for idx, chrom in enumerate(chromosomes):
        if chrom != current:
            groups.append((current, start, idx))
            start = idx
            current = chrom
    groups.append((current, start, len(chromosomes)))
    return groups


def _oligo_center_gene_scores(log_matrix, obs, *, oligo_label: str = "oligo"):
    import numpy as np

    cell_type = obs["cell_type"].astype("string").fillna("")
    oligo_mask = (cell_type == oligo_label).to_numpy()
    if not oligo_mask.any():
        raise ValueError("No oligo cells found for chromosome effect reference.")

    oligo_matrix = log_matrix[oligo_mask, :]
    oligo_means = oligo_matrix.mean(axis=0, keepdims=True)
    oligo_stds = oligo_matrix.std(axis=0, keepdims=True)
    oligo_stds[oligo_stds == 0] = 1.0
    return (log_matrix - oligo_means) / oligo_stds, oligo_mask


def _chromosome_score_frame(gene_score_matrix, ordered_positions):
    import pandas as pd

    groups = _chromosome_groups(ordered_positions)
    scores = {
        chromosome: gene_score_matrix[:, start:end].mean(axis=1)
        for chromosome, start, end in groups
    }
    return pd.DataFrame(scores)


def _bootstrap_mean_difference(values, reference_values, *, iterations: int, rng):
    import numpy as np

    if iterations <= 0:
        raise ValueError("--bootstrap-iterations must be > 0.")

    values = np.asarray(values, dtype=np.float32)
    reference_values = np.asarray(reference_values, dtype=np.float32)
    observed = float(values.mean() - reference_values.mean())
    boot = np.empty(iterations, dtype=np.float32)
    for idx in range(iterations):
        sample = rng.choice(values, size=values.size, replace=True)
        reference_sample = rng.choice(
            reference_values, size=reference_values.size, replace=True
        )
        boot[idx] = sample.mean() - reference_sample.mean()
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return observed, float(ci_low), float(ci_high)


def _chromosome_cluster_effects(
    chromosome_scores,
    obs,
    ordered_positions,
    *,
    focus_chromosomes=DEFAULT_FOCUS_CHROMOSOMES,
    bootstrap_iterations: int,
    bootstrap_seed: int,
):
    import numpy as np
    import pandas as pd

    if "heatmap_leiden" not in obs.columns:
        raise ValueError("obs is missing heatmap_leiden cluster labels.")
    if "cell_type" not in obs.columns:
        raise ValueError("obs is missing cell_type labels.")

    cluster_labels = _ordered_cluster_categorical(obs["heatmap_leiden"].tolist())
    cell_type = obs["cell_type"].astype("string").fillna("")
    oligo_mask = (cell_type == "oligo").to_numpy()
    if not oligo_mask.any():
        raise ValueError("No oligo cells found for chromosome effect reference.")

    gene_counts = (
        ordered_positions.groupby("chromosome", sort=False)["gene"].size().to_dict()
    )
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for cluster in cluster_labels.categories:
        cluster_mask = cluster_labels == cluster
        for chromosome in chromosome_scores.columns:
            values = chromosome_scores.loc[cluster_mask, chromosome].to_numpy()
            reference_values = chromosome_scores.loc[oligo_mask, chromosome].to_numpy()
            effect, ci_low, ci_high = _bootstrap_mean_difference(
                values,
                reference_values,
                iterations=bootstrap_iterations,
                rng=rng,
            )
            rows.append(
                {
                    "cluster": str(cluster),
                    "chromosome": str(chromosome),
                    "mean_score": float(values.mean()),
                    "oligo_mean_score": float(reference_values.mean()),
                    "effect_vs_oligo": effect,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_cells": int(values.size),
                    "n_oligo_reference_cells": int(reference_values.size),
                    "n_genes_on_chromosome": int(gene_counts.get(chromosome, 0)),
                    "is_focus_chromosome": str(chromosome) in set(focus_chromosomes),
                    "small_n_flag": bool(values.size < SMALL_CLUSTER_N),
                }
            )
    return pd.DataFrame(rows)


def _plot_chromosome_effect_heatmap(effects, output_path: Path) -> None:
    _prepare_plot_output(output_path)

    import matplotlib.pyplot as plt
    import numpy as np

    pivot = effects.pivot(
        index="cluster", columns="chromosome", values="effect_vs_oligo"
    )
    ordered_columns = [
        chrom for chrom in DEFAULT_CHROMOSOME_ORDER if chrom in set(pivot.columns)
    ]
    ordered_index = sorted(
        pivot.index.tolist(),
        key=lambda label: (0, int(label)) if str(label).isdigit() else (1, str(label)),
    )
    pivot = pivot.loc[ordered_index, ordered_columns]
    vmax = max(0.25, float(np.nanmax(np.abs(pivot.to_numpy()))))

    fig_width = max(10.0, min(20.0, 0.45 * len(pivot.columns)))
    fig_height = max(3.5, 0.55 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(
        pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"cluster {label}" for label in pivot.index], fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_title("Mean oligo-centered chromosome expression effect")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("effect vs all oligo cells")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_focus_chromosome_effects(effects, output_path: Path) -> None:
    _prepare_plot_output(output_path)

    import matplotlib.pyplot as plt
    import numpy as np

    focus = effects[effects["is_focus_chromosome"]].copy()
    if focus.empty:
        return
    chromosomes = [
        chrom
        for chrom in DEFAULT_FOCUS_CHROMOSOMES
        if chrom in set(focus["chromosome"])
    ]
    clusters = sorted(
        focus["cluster"].unique().tolist(),
        key=lambda label: (0, int(label)) if str(label).isdigit() else (1, str(label)),
    )
    x = np.arange(len(chromosomes))
    offsets = np.linspace(-0.32, 0.32, num=len(clusters))

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for offset, cluster in zip(offsets, clusters):
        sub = focus[focus["cluster"] == cluster].set_index("chromosome")
        y = np.array([sub.loc[chrom, "effect_vs_oligo"] for chrom in chromosomes])
        yerr = np.array(
            [
                [
                    sub.loc[chrom, "effect_vs_oligo"] - sub.loc[chrom, "ci_low"]
                    for chrom in chromosomes
                ],
                [
                    sub.loc[chrom, "ci_high"] - sub.loc[chrom, "effect_vs_oligo"]
                    for chrom in chromosomes
                ],
            ]
        )
        ax.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt="o",
            capsize=2.5,
            markersize=4,
            linewidth=1,
            label=f"cluster {cluster}",
        )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"chr{chrom}" for chrom in chromosomes])
    ax.set_ylabel("effect vs all oligo cells")
    ax.set_title("Focused chromosome effects with bootstrap 95% CI")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _log1p_normalized_vector(counts):
    import numpy as np

    counts = np.asarray(counts, dtype=np.float32).ravel()
    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float32)
    return np.log1p(counts * (1e4 / total)).astype(np.float32, copy=False)


def _call_cnv_state(
    effect: float, ci_low: float, ci_high: float, *, threshold: float
) -> str:
    if effect >= threshold and ci_low > 0:
        return "gain"
    if effect <= -threshold and ci_high < 0:
        return "loss"
    if abs(effect) < threshold and ci_low <= 0 <= ci_high:
        return "neutral"
    return "uncertain"


def _chromosome_cnv_calls(
    counts_matrix,
    gene_score_matrix,
    log_matrix,
    obs,
    ordered_positions,
    *,
    grouping_keys,
    cancer_mask,
    oligo_mask,
    effect_threshold: float,
    min_genes_per_chromosome: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
):
    import numpy as np
    import pandas as pd

    cancer_mask = np.asarray(cancer_mask, dtype=bool)
    oligo_mask = np.asarray(oligo_mask, dtype=bool)
    if not cancer_mask.any():
        raise ValueError("No cancer cells available for CNV-like calls.")
    if not oligo_mask.any():
        raise ValueError("No oligo cells available for CNV-like reference.")

    chromosome_groups = _chromosome_groups(ordered_positions)
    if not chromosome_groups:
        raise ValueError("No chromosome groups available for CNV-like calls.")

    oligo_log = log_matrix[oligo_mask, :]
    oligo_means = oligo_log.mean(axis=0)
    oligo_stds = oligo_log.std(axis=0)
    oligo_stds[oligo_stds == 0] = 1.0

    chromosome_scores = _chromosome_score_frame(gene_score_matrix, ordered_positions)
    chromosome_scores.index = obs.index
    rng = np.random.default_rng(bootstrap_seed)

    rows = []
    for grouping in grouping_keys:
        if grouping not in obs.columns:
            raise ValueError(f"obs is missing grouping column: {grouping}")
        labels = obs[grouping].astype("string").fillna("")
        target_labels = sorted(
            labels[cancer_mask].unique().tolist(),
            key=lambda label: (
                (0, int(label)) if str(label).isdigit() else (1, str(label))
            ),
        )
        for label in target_labels:
            group_mask = cancer_mask & (labels == label).to_numpy()
            n_group = int(group_mask.sum())
            if n_group == 0:
                continue

            group_counts = counts_matrix[group_mask, :].sum(axis=0)
            group_log = _log1p_normalized_vector(group_counts)
            group_gene_scores = (group_log - oligo_means) / oligo_stds

            for chromosome, start, end in chromosome_groups:
                n_genes = int(end - start)
                effect = float(np.median(group_gene_scores[start:end]))
                values = chromosome_scores.loc[group_mask, chromosome].to_numpy()
                reference_values = chromosome_scores.loc[oligo_mask, chromosome].to_numpy()
                cell_effect, ci_low, ci_high = _bootstrap_mean_difference(
                    values,
                    reference_values,
                    iterations=bootstrap_iterations,
                    rng=rng,
                )
                # Use the cell bootstrap width but center it on the pseudo-bulk effect.
                ci_shift = effect - cell_effect
                ci_low = float(ci_low + ci_shift)
                ci_high = float(ci_high + ci_shift)
                rows.append(
                    {
                        "grouping": grouping,
                        "group": str(label),
                        "chromosome": str(chromosome),
                        "state": _call_cnv_state(
                            effect,
                            ci_low,
                            ci_high,
                            threshold=effect_threshold,
                        ),
                        "effect": effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_cancer_cells": n_group,
                        "n_oligo_reference_cells": int(oligo_mask.sum()),
                        "n_genes_on_chromosome": n_genes,
                        "low_gene_count_flag": bool(
                            n_genes < min_genes_per_chromosome
                        ),
                        "small_group_flag": bool(n_group < SMALL_CLUSTER_N),
                    }
                )
    return pd.DataFrame(rows)


def _plot_cnv_call_heatmap(calls, output_path: Path, *, grouping: str) -> None:
    _prepare_plot_output(output_path)

    import matplotlib.pyplot as plt
    import numpy as np

    sub = calls[calls["grouping"] == grouping].copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="group", columns="chromosome", values="effect")
    state_pivot = sub.pivot(index="group", columns="chromosome", values="state")
    ordered_columns = [
        chrom for chrom in DEFAULT_CHROMOSOME_ORDER if chrom in set(pivot.columns)
    ]
    ordered_index = sorted(
        pivot.index.tolist(),
        key=lambda label: (0, int(label)) if str(label).isdigit() else (1, str(label)),
    )
    pivot = pivot.loc[ordered_index, ordered_columns]
    state_pivot = state_pivot.loc[ordered_index, ordered_columns]
    values = pivot.to_numpy(dtype=float)
    vmax = max(0.5, float(np.nanmax(np.abs(values))))

    fig_width = max(10.0, min(20.0, 0.45 * len(pivot.columns)))
    fig_height = max(3.5, 0.55 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{grouping} {label}" for label in pivot.index], fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_title(f"Whole-chromosome CNV-like calls ({grouping})")

    state_symbol = {"gain": "+", "loss": "-", "neutral": "0", "uncertain": "?"}
    for y_idx, label in enumerate(pivot.index):
        for x_idx, chrom in enumerate(pivot.columns):
            state = state_pivot.loc[label, chrom]
            ax.text(
                x_idx,
                y_idx,
                state_symbol.get(state, "?"),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("pseudo-bulk median oligo-centered gene score")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ordered_cells_by_group(adata_sel, *, group_key: str) -> list[str]:
    selected_obs_for_order = adata_sel.obs.copy()
    selected_obs_for_order["cell_id_finetuned"] = selected_obs_for_order.index.astype(str)
    selected_obs_for_order[group_key] = _ordered_cluster_categorical(
        selected_obs_for_order[group_key].tolist()
    )
    sort_cols = [group_key, "cell_type"]
    ascending = [True, True]
    if "n_transcripts" in selected_obs_for_order.columns:
        sort_cols.append("n_transcripts")
        ascending.append(False)
    sort_cols.append("cell_id_finetuned")
    ascending.append(True)
    selected_obs_for_order = selected_obs_for_order.sort_values(
        sort_cols, ascending=ascending
    )
    return selected_obs_for_order.index.tolist()


def _plot_heatmap(
    matrix,
    ordered_positions,
    output_path: Path,
    *,
    n_cells: int,
    clip_z: float,
    cluster_labels,
    annotation_title: str = "heatmap_leiden",
) -> None:
    _prepare_plot_output(output_path)

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cluster_cat = _ordered_cluster_categorical(cluster_labels)
    cluster_codes = np.asarray(cluster_cat.codes)
    cluster_colors = _cluster_palette(len(cluster_cat.categories))

    n_genes = matrix.shape[1]
    chromosome_groups = _chromosome_groups(ordered_positions)
    if not chromosome_groups:
        raise ValueError("No chromosome groups available for heatmap plotting.")
    if n_genes != chromosome_groups[-1][2]:
        raise ValueError(
            "Heatmap matrix column count does not match ordered gene-position rows."
        )

    fig_width = max(12.0, min(30.0, n_genes * 0.045))
    fig_height = max(7.0, min(24.0, n_cells / 5500))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.022, 1.0], wspace=0.01)
    ax_bar = fig.add_subplot(gs[0, 0])
    heatmap_gs = gs[0, 1].subgridspec(
        1,
        len(chromosome_groups),
        width_ratios=[end - start for _, start, end in chromosome_groups],
        wspace=0.08,
    )

    ax_bar.imshow(
        cluster_codes.reshape(-1, 1),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(cluster_colors),
        vmin=-0.5,
        vmax=len(cluster_cat.categories) - 0.5,
    )
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("Annotated cancer_* + oligo cells")

    heatmap_axes = []
    image = None
    for idx, (chromosome, start, end) in enumerate(chromosome_groups):
        ax = fig.add_subplot(heatmap_gs[0, idx], sharey=ax_bar)
        heatmap_axes.append(ax)
        image = ax.imshow(
            matrix[:, start:end],
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-clip_z,
            vmax=clip_z,
            rasterized=True,
        )
        ax.set_xticks([(end - start - 1) / 2])
        ax.set_xticklabels([chromosome], rotation=0, fontsize=8)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_alpha(0.7)

    heatmap_axes[0].set_ylabel("")

    cluster_breaks = np.where(np.diff(cluster_codes) != 0)[0] + 1
    for row in cluster_breaks:
        for ax in heatmap_axes:
            ax.axhline(row - 0.5, color="black", linewidth=0.5, alpha=0.7)

    fig.suptitle(
        f"TRACER finetuned expression heatmap ({n_cells:,} cells x {n_genes:,} genes, "
        f"{len(cluster_cat.categories)} clusters)"
    )
    fig.supxlabel("Genes ordered by GRCh38 chromosome position")

    cluster_counts = {
        str(label): int((cluster_codes == code).sum())
        for code, label in enumerate(cluster_cat.categories)
    }
    legend_handles = [
        Patch(
            facecolor=cluster_colors[code],
            edgecolor="none",
            label=f"cluster {label} (n={cluster_counts[str(label)]:,})",
        )
        for code, label in enumerate(cluster_cat.categories)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
        title=annotation_title,
        title_fontsize=9,
    )

    cbar = fig.colorbar(image, ax=heatmap_axes, fraction=0.02, pad=0.08)
    cbar.set_label(
        f"gene-wise z score of log1p normalized expression, clipped +/-{clip_z:g}"
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _configure_thread_env()

    import anndata as ad
    import pandas as pd

    h5ad_path = Path(args.h5ad)
    features_tsv = _resolve_features_tsv(args.features_tsv)
    gene_positions_path = Path(args.gene_positions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hclust_ks = _unique_positive_ints(args.hclust_k, default=DEFAULT_HCLUST_K)

    _log(f"Reading gene features: {features_tsv}")
    gene_features = _read_gene_features(features_tsv)
    _log(f"Gene-expression features: {len(gene_features):,}")

    if args.download_gene_positions:
        _download_gene_positions(gene_features, gene_positions_path)

    positions = _read_gene_positions(gene_positions_path)
    gene_meta = gene_features.merge(positions, on=["gene_id", "gene"], how="inner")
    gene_meta["chromosome_rank"] = gene_meta["chromosome"].map(_chromosome_rank)
    gene_meta = gene_meta.sort_values(["chromosome_rank", "start", "end", "gene"]).drop(
        columns=["chromosome_rank"]
    )

    _log(f"Reading AnnData: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    ordered_genes = [
        gene for gene in gene_meta["gene"].tolist() if gene in adata.var_names
    ]
    missing_gene_features = sorted(set(gene_features["gene"]) - set(ordered_genes))
    if not ordered_genes:
        raise ValueError(
            "No positioned Gene Expression features matched AnnData var_names."
        )

    selected_mask = _selected_cell_mask(
        adata.obs, prefixes=args.cell_type_prefix, exact=args.cell_type_exact
    )
    selected_mask = selected_mask.fillna(False).astype(bool).to_numpy()
    if not selected_mask.any():
        raise ValueError("No cells matched the requested cell-type filters.")

    adata_sel = adata[selected_mask, :].copy()
    _log(
        f"Clustering cancer_* + oligo subset ({adata_sel.n_obs:,} cells) with "
        f"Leiden resolution={args.leiden_resolution:g} ..."
    )
    _cluster_selected_subset(
        adata_sel, resolution=args.leiden_resolution, seed=args.seed
    )
    cluster_categories = sorted(
        adata_sel.obs["heatmap_leiden"].astype(str).unique().tolist(),
        key=lambda label: (0, int(label)) if label.isdigit() else (1, label),
    )
    _log(
        f"Leiden clusters discovered: {len(cluster_categories)} -> {cluster_categories}"
    )

    umap_path = outdir / "umap_clusters.png"
    _plot_umap_panels(adata_sel, umap_path, resolution=args.leiden_resolution)
    _log(f"Saved UMAP: {umap_path}")

    all_selected_cells = adata_sel.obs_names.tolist()
    ordered_gene_table = gene_meta[gene_meta["gene"].isin(ordered_genes)].copy()
    ordered_gene_table["gene"] = pd.Categorical(
        ordered_gene_table["gene"], categories=ordered_genes, ordered=True
    )
    ordered_gene_table = ordered_gene_table.sort_values("gene")
    ordered_gene_table.to_csv(outdir / "ordered_genes.tsv", sep="\t", index=False)

    _log("Computing oligo-centered gene and chromosome scores ...")
    effect_sub, log_matrix = _prepare_log_expression_matrix(
        adata_sel,
        selected_cells=all_selected_cells,
        ordered_genes=ordered_genes,
    )
    gene_score_matrix, _ = _oligo_center_gene_scores(log_matrix, effect_sub.obs)
    cancer_mask = _prefix_cell_mask(effect_sub.obs, prefixes=args.cell_type_prefix)
    oligo_mask = _oligo_cell_mask(effect_sub.obs)

    _log(
        f"Hierarchical clustering {int(cancer_mask.sum()):,} cancer cells for "
        f"k={','.join(str(k) for k in hclust_ks)} ..."
    )
    hclust_keys = _add_hierarchical_clusters(
        adata_sel,
        gene_score_matrix,
        cancer_mask=cancer_mask,
        ks=hclust_ks,
        n_pcs=args.hclust_n_pcs,
        n_neighbors=args.hclust_n_neighbors,
        n_jobs=args.hclust_n_jobs,
        seed=args.seed,
    )
    for key in hclust_keys:
        effect_sub.obs[key] = (
            adata_sel.obs.loc[effect_sub.obs_names, key].astype(str).values
        )

    hclust_umap_path = outdir / "umap_hclust.png"
    _plot_hclust_umap_panels(adata_sel, hclust_umap_path, hclust_keys=hclust_keys)
    _log(f"Saved hierarchical-cluster UMAP: {hclust_umap_path}")

    selected_cells = _ordered_cells_by_group(adata_sel, group_key="heatmap_leiden")

    _log(f"Selected cells before optional subsampling: {len(selected_cells):,}")
    _log(f"Positioned genes matched in AnnData: {len(ordered_genes):,}")

    sub, matrix = _prepare_heatmap_matrix(
        adata_sel,
        selected_cells=selected_cells,
        ordered_genes=ordered_genes,
        max_cells=args.max_cells,
        seed=args.seed,
        clip_z=args.clip_z,
    )

    cluster_labels = (
        adata_sel.obs.loc[sub.obs_names, "heatmap_leiden"].astype(str).tolist()
    )

    selected_obs = sub.obs[["cell_type", "n_transcripts", "n_genes"]].copy()
    selected_obs["heatmap_leiden"] = (
        adata_sel.obs.loc[sub.obs_names, "heatmap_leiden"].astype(str).values
    )
    for key in hclust_keys:
        selected_obs[key] = adata_sel.obs.loc[sub.obs_names, key].astype(str).values
    selected_obs.index.name = "cell_id_finetuned"
    selected_obs.to_csv(outdir / "selected_cells.csv")

    _log("Computing oligo-centered chromosome effects ...")
    chromosome_scores = _chromosome_score_frame(gene_score_matrix, ordered_gene_table)
    chromosome_scores.index = effect_sub.obs_names
    chromosome_effects = _chromosome_cluster_effects(
        chromosome_scores,
        effect_sub.obs,
        ordered_gene_table,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    effects_tsv = outdir / "chromosome_cluster_effects.tsv"
    chromosome_effects.to_csv(effects_tsv, sep="\t", index=False)
    _log(f"Saved chromosome effects: {effects_tsv}")

    effects_heatmap = outdir / "chromosome_cluster_effect_heatmap.png"
    _plot_chromosome_effect_heatmap(chromosome_effects, effects_heatmap)
    _log(f"Saved chromosome-effect heatmap: {effects_heatmap}")

    focus_effects = outdir / "focus_chromosome_effects.png"
    _plot_focus_chromosome_effects(chromosome_effects, focus_effects)
    _log(f"Saved focused chromosome effects: {focus_effects}")

    _log("Computing pseudo-bulk whole-chromosome CNV-like calls ...")
    _, counts_matrix = _counts_matrix(
        adata_sel, selected_cells=all_selected_cells, ordered_genes=ordered_genes
    )
    cnv_grouping_keys = ["heatmap_leiden", *hclust_keys]
    cnv_calls = _chromosome_cnv_calls(
        counts_matrix,
        gene_score_matrix,
        log_matrix,
        effect_sub.obs,
        ordered_gene_table,
        grouping_keys=cnv_grouping_keys,
        cancer_mask=cancer_mask,
        oligo_mask=oligo_mask,
        effect_threshold=args.cnv_effect_threshold,
        min_genes_per_chromosome=args.cnv_min_genes_per_chromosome,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    cnv_calls_tsv = outdir / "chromosome_cnv_calls.tsv"
    cnv_calls.to_csv(cnv_calls_tsv, sep="\t", index=False)
    _log(f"Saved CNV-like calls: {cnv_calls_tsv}")
    for grouping in cnv_grouping_keys:
        cnv_heatmap = outdir / f"chromosome_cnv_calls_{grouping}.png"
        _plot_cnv_call_heatmap(cnv_calls, cnv_heatmap, grouping=grouping)
        _log(f"Saved CNV-like call heatmap: {cnv_heatmap}")

    summary = pd.DataFrame(
        [
            {
                "h5ad": str(h5ad_path),
                "features_tsv": str(features_tsv),
                "gene_positions": str(gene_positions_path),
                "n_selected_cells_available": len(selected_cells),
                "n_selected_cells_plotted": int(sub.n_obs),
                "n_gene_expression_features": int(len(gene_features)),
                "n_positioned_genes_plotted": int(len(ordered_genes)),
                "n_gene_expression_features_missing_or_unpositioned": int(
                    len(missing_gene_features)
                ),
                "max_cells": args.max_cells,
                "clip_z": args.clip_z,
                "leiden_resolution": args.leiden_resolution,
                "bootstrap_iterations": args.bootstrap_iterations,
                "bootstrap_seed": args.bootstrap_seed,
                "cnv_effect_threshold": args.cnv_effect_threshold,
                "cnv_min_genes_per_chromosome": args.cnv_min_genes_per_chromosome,
                "hclust_k": ",".join(str(k) for k in hclust_ks),
                "hclust_n_pcs": args.hclust_n_pcs,
                "hclust_n_neighbors": args.hclust_n_neighbors,
                "hclust_n_jobs": args.hclust_n_jobs,
                "n_clusters": len(cluster_categories),
            }
        ]
    )
    summary.to_csv(outdir / "heatmap_summary.csv", index=False)

    output_png = outdir / "chromosome_ordered_heatmap.png"
    _plot_heatmap(
        matrix,
        ordered_gene_table,
        output_png,
        n_cells=sub.n_obs,
        clip_z=args.clip_z,
        cluster_labels=cluster_labels,
        annotation_title="heatmap_leiden",
    )
    _log(f"Saved heatmap: {output_png}")

    for key in hclust_keys:
        hclust_selected_cells = _ordered_cells_by_group(adata_sel, group_key=key)
        hclust_sub, hclust_matrix = _prepare_heatmap_matrix(
            adata_sel,
            selected_cells=hclust_selected_cells,
            ordered_genes=ordered_genes,
            max_cells=args.max_cells,
            seed=args.seed,
            clip_z=args.clip_z,
        )
        hclust_labels = (
            adata_sel.obs.loc[hclust_sub.obs_names, key].astype(str).tolist()
        )
        hclust_heatmap = outdir / f"chromosome_ordered_heatmap_{key}.png"
        _plot_heatmap(
            hclust_matrix,
            ordered_gene_table,
            hclust_heatmap,
            n_cells=hclust_sub.n_obs,
            clip_z=args.clip_z,
            cluster_labels=hclust_labels,
            annotation_title=key,
        )
        _log(f"Saved {key}-ordered heatmap: {hclust_heatmap}")
    _log(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
