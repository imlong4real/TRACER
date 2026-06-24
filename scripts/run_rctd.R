#!/usr/bin/env Rscript
# =============================================================================
# run_rctd.R — RCTD (spacexr) as an orthogonal cell-type mixture diagnostic.
# =============================================================================
#
# Reads a spatial cell-by-gene h5ad and a reference scRNA h5ad, runs
# spacexr::run.RCTD in doublet mode, and emits per-cell weights, dominant-
# celltype calls, Shannon entropy of the weight distribution, and tumor/
# immune/stromal mixture scores. Treats RCTD as an orthogonal diagnostic
# (NOT ground truth).
#
# USAGE
# =====
# Rscript scripts/run_rctd.R \
#   --spatial-h5ad <h5ad>      \
#   --reference-h5ad <h5ad>    \
#   --reference-celltype-col cell_type_harmonized \
#   --outdir results/benchmark/lung_xenium/rctd \
#   --doublet-mode doublet     \
#   --umi-min 10               \
#   --max-cores 4              \
#   [--celltype-category-json <map.json>]   # optional, for mixture scores
#
# Optional pre/post:
#   --spatial-h5ad-pre <pre-method h5ad>
#
# If spacexr / anndata / reticulate are not installed, exits 1 with a clear
# message.
#
# ENVIRONMENT
# ===========
# Tested with an R env that has:
#   spacexr           (Stickels et al.)
#   anndata           (CRAN)  — reads .h5ad
#   optparse, jsonlite, Matrix, readr
# Recommended: /Users/lyuan13/anaconda3/envs/tracer_benchmark_r/bin/Rscript
# =============================================================================

suppressPackageStartupMessages({
  required <- c("optparse", "jsonlite", "Matrix", "anndata")
  missing  <- required[!sapply(required, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    stop(sprintf(
      "Missing R packages: %s. Install before running this script.",
      paste(missing, collapse = ", ")
    ))
  }
  if (!requireNamespace("spacexr", quietly = TRUE)) {
    stop("Missing R package 'spacexr'. Install via remotes::install_github('dmcable/spacexr').")
  }
  library(optparse)
  library(jsonlite)
  library(Matrix)
  library(anndata)
  library(spacexr)
})


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parse_args_local <- function() {
  opts <- list(
    make_option(c("--spatial-h5ad"), type = "character", default = NULL,
                help = "Spatial cell-by-gene h5ad (post-method)."),
    make_option(c("--spatial-h5ad-pre"), type = "character", default = NULL,
                help = "Optional pre-method spatial h5ad for paired comparison."),
    make_option(c("--reference-h5ad"), type = "character", default = NULL,
                help = "Reference scRNA h5ad."),
    make_option(c("--reference-celltype-col"), type = "character",
                default = "cell_type_harmonized",
                help = "obs column for reference cell-type labels."),
    make_option(c("--outdir"), type = "character", default = NULL),
    make_option(c("--doublet-mode"), type = "character", default = "doublet",
                help = "doublet | full | multi (default doublet)."),
    make_option(c("--umi-min"), type = "integer", default = 10,
                help = "Min UMI per pixel/cell."),
    make_option(c("--umi-min-sigma"), type = "integer", default = 100,
                help = "spacexr UMI_min_sigma. Lower for sparse spatial pixels."),
    make_option(c("--max-cores"), type = "integer", default = 4),
    make_option(c("--celltype-category-json"), type = "character", default = NULL,
                help = "JSON mapping cell types to categories (tumor/immune/stromal/...). "),
    make_option(c("--min-cells-per-celltype-reference"), type = "integer",
                default = 25, help = "Drop reference celltypes below this count."),
    make_option(c("--gene-cutoff"), type = "numeric", default = 0.000125),
    make_option(c("--fc-cutoff"), type = "numeric", default = 0.5),
    make_option(c("--seed"), type = "integer", default = 1)
  )
  parser <- OptionParser(option_list = opts,
                         description = "Run RCTD as an orthogonal diagnostic.")
  args <- parse_args(parser)
  required <- c("spatial-h5ad", "reference-h5ad", "outdir")
  missing <- required[sapply(required, function(x) is.null(args[[x]]))]
  if (length(missing) > 0) {
    print_help(parser)
    stop(sprintf("Missing required: %s", paste(missing, collapse = ", ")))
  }
  args
}


# -----------------------------------------------------------------------------
# H5AD loading
# -----------------------------------------------------------------------------
.load_h5ad_counts <- function(path, want_raw = TRUE) {
  message(sprintf("[h5ad] reading %s", path))
  a <- anndata::read_h5ad(path)
  # Prefer layers$counts; fall back to raw$X; finally adata$X.
  X <- NULL
  if ("counts" %in% names(a$layers)) {
    X <- a$layers[["counts"]]
    message("[h5ad]   using layers$counts")
  } else if (!is.null(a$raw)) {
    X <- a$raw$X
    message("[h5ad]   using raw$X (caller responsibility: validate it's raw counts)")
  } else {
    X <- a$X
    message("[h5ad]   using X (caller responsibility: validate it's raw counts)")
  }
  # Force numeric (double) storage and CSC layout. anndata may return an
  # integer-typed dgRMatrix, which downstream Matrix operations + spacexr
  # reject ("'x' slot is not of type 'double'"). Round-trip via dgCMatrix
  # with explicit double cast.
  if (inherits(X, "Matrix")) {
    Xd <- as(X, "CsparseMatrix")
    if (typeof(Xd@x) != "double") {
      Xd@x <- as.double(Xd@x)
    }
    X <- Xd
  } else {
    # dense fallback
    X <- as(as.matrix(X) * 1.0, "CsparseMatrix")
  }
  list(X = X, obs = a$obs, var = a$var, var_names = a$var_names, obs_names = a$obs_names)
}


# -----------------------------------------------------------------------------
# Build spacexr Reference + SpatialRNA
# -----------------------------------------------------------------------------
build_reference <- function(ref_path, celltype_col, min_cells, seed,
                             restrict_genes = NULL) {
  set.seed(seed)
  obj <- .load_h5ad_counts(ref_path)
  ct <- as.character(obj$obs[[celltype_col]])
  if (any(is.na(ct))) {
    keep <- !is.na(ct)
    obj$X <- obj$X[keep, , drop = FALSE]
    obj$obs <- obj$obs[keep, , drop = FALSE]
    ct <- ct[keep]
    obj$obs_names <- obj$obs_names[keep]
  }
  # Subset reference genes to those shared with spatial panel BEFORE coercing
  # to dense in spacexr. On a 72k-gene whole-transcriptome reference, the
  # internal dense coercion blows past R's 16 GB ceiling; restricting to the
  # spatial panel (~300 genes for Xenium) avoids it and is also what RCTD
  # ultimately uses anyway.
  if (!is.null(restrict_genes)) {
    var_chr <- as.character(obj$var_names)
    keep_g <- var_chr %in% restrict_genes
    message(sprintf("[ref] gene restriction: kept %d / %d reference genes (overlap with spatial panel)",
                    sum(keep_g), length(keep_g)))
    obj$X <- obj$X[, keep_g, drop = FALSE]
    obj$var_names <- obj$var_names[keep_g]
  }
  # Drop sparse celltypes.
  tab <- table(ct)
  keep_types <- names(tab[tab >= min_cells])
  keep_rows <- ct %in% keep_types
  obj$X <- obj$X[keep_rows, , drop = FALSE]
  ct <- ct[keep_rows]
  obj$obs_names <- obj$obs_names[keep_rows]
  message(sprintf("[ref] %d cells across %d celltypes (after min_cells=%d filter)",
                  nrow(obj$X), length(unique(ct)), min_cells))
  # spacexr expects genes × cells.
  counts <- t(obj$X)
  rownames(counts) <- as.character(obj$var_names)
  colnames(counts) <- as.character(obj$obs_names)
  cell_types <- factor(ct, levels = sort(unique(ct)))
  names(cell_types) <- as.character(obj$obs_names)
  nUMI <- Matrix::colSums(counts)
  ref <- spacexr::Reference(counts, cell_types, nUMI)
  ref
}


build_spatial <- function(spatial_path, umi_min) {
  obj <- .load_h5ad_counts(spatial_path)
  # spacexr SpatialRNA expects genes × cells and a coords frame with rownames = cell ids.
  counts <- t(obj$X)
  rownames(counts) <- as.character(obj$var_names)
  colnames(counts) <- as.character(obj$obs_names)
  nUMI <- Matrix::colSums(counts)
  # If the obs has x_centroid / y_centroid use them; otherwise plant dummy coords.
  if ("x_centroid" %in% colnames(obj$obs) && "y_centroid" %in% colnames(obj$obs)) {
    coords <- data.frame(x = obj$obs$x_centroid, y = obj$obs$y_centroid)
  } else {
    coords <- data.frame(x = seq_along(nUMI), y = 0)
  }
  rownames(coords) <- as.character(obj$obs_names)
  keep <- nUMI >= umi_min
  message(sprintf("[spatial] %d cells; after UMI_min=%d filter: %d",
                  length(nUMI), umi_min, sum(keep)))
  counts <- counts[, keep, drop = FALSE]
  coords <- coords[keep, , drop = FALSE]
  nUMI <- nUMI[keep]
  spacexr::SpatialRNA(coords = coords, counts = counts, nUMI = nUMI)
}


# -----------------------------------------------------------------------------
# Metrics from RCTD results
# -----------------------------------------------------------------------------
shannon_entropy <- function(w_row) {
  p <- w_row[w_row > 0]
  if (length(p) == 0) return(NA_real_)
  -sum(p * log(p))
}

dominant_celltype <- function(weights_norm) {
  apply(weights_norm, 1, function(r) {
    if (all(is.na(r))) return(NA_character_)
    colnames(weights_norm)[which.max(r)]
  })
}

celltype_category_for <- function(celltype, category_map) {
  if (is.null(category_map)) return(NA_character_)
  for (cat in names(category_map)) {
    pats <- unlist(category_map[[cat]])
    if (any(grepl(paste(pats, collapse = "|"), celltype, ignore.case = TRUE))) {
      return(cat)
    }
  }
  NA_character_
}

mixture_score_pair <- function(weights_norm, category_map, cat_a, cat_b) {
  if (is.null(category_map)) return(rep(NA_real_, nrow(weights_norm)))
  in_a <- sapply(colnames(weights_norm),
                  function(g) celltype_category_for(g, category_map) == cat_a)
  in_b <- sapply(colnames(weights_norm),
                  function(g) celltype_category_for(g, category_map) == cat_b)
  in_a[is.na(in_a)] <- FALSE
  in_b[is.na(in_b)] <- FALSE
  if (sum(in_a) == 0 || sum(in_b) == 0) return(rep(NA_real_, nrow(weights_norm)))
  apply(weights_norm, 1, function(r) min(sum(r[in_a]), sum(r[in_b])))
}


# -----------------------------------------------------------------------------
# Run one RCTD instance
# -----------------------------------------------------------------------------
run_one_rctd <- function(tag, ref, spatial, args, category_map, outdir) {
  message(sprintf("[%s] create.RCTD ...", tag))
  myRCTD <- spacexr::create.RCTD(
    spatial, ref,
    max_cores = args$`max-cores`,
    UMI_min = args$`umi-min`,
    gene_cutoff = args$`gene-cutoff`,
    fc_cutoff  = args$`fc-cutoff`,
    UMI_min_sigma = args$`umi-min-sigma`
  )
  message(sprintf("[%s] run.RCTD doublet_mode=%s", tag, args$`doublet-mode`))
  myRCTD <- spacexr::run.RCTD(myRCTD, doublet_mode = args$`doublet-mode`)

  res <- myRCTD@results
  weights_norm <- as.matrix(spacexr::normalize_weights(res$weights))
  dom <- dominant_celltype(weights_norm)
  entropy <- apply(weights_norm, 1, shannon_entropy)
  max_w <- apply(weights_norm, 1, function(r) if (all(is.na(r))) NA_real_ else max(r))

  # Doublet/singlet calls (only meaningful when doublet_mode='doublet').
  doublet_status <- NA_character_
  if (!is.null(res$results_df)) {
    rdf <- res$results_df
    doublet_status <- rdf$spot_class
  }

  mix_tumor_immune  <- mixture_score_pair(weights_norm, category_map, "tumor",  "immune")
  mix_tumor_stromal <- mixture_score_pair(weights_norm, category_map, "tumor",  "stromal")
  mix_immune_stromal<- mixture_score_pair(weights_norm, category_map, "immune", "stromal")

  per_cell <- data.frame(
    cell_id            = rownames(weights_norm),
    dominant_celltype  = dom,
    max_weight         = max_w,
    entropy            = entropy,
    doublet_status     = if (length(doublet_status) == nrow(weights_norm))
                            doublet_status else NA_character_,
    mixture_tumor_immune  = mix_tumor_immune,
    mixture_tumor_stromal = mix_tumor_stromal,
    mixture_immune_stromal= mix_immune_stromal
  )
  write.table(per_cell,
              file = file.path(outdir, sprintf("rctd_cell_assignments_%s.tsv", tag)),
              sep = "\t", quote = FALSE, row.names = FALSE)
  # Full weights table compressed.
  wdf <- as.data.frame(weights_norm)
  wdf$cell_id <- rownames(weights_norm)
  wdf <- wdf[, c("cell_id", setdiff(colnames(wdf), "cell_id"))]
  gz <- gzfile(file.path(outdir, sprintf("rctd_weights_%s.tsv.gz", tag)), "w")
  write.table(wdf, gz, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz)

  summary <- list(
    tag = tag,
    n_cells = nrow(weights_norm),
    n_celltypes = ncol(weights_norm),
    median_entropy = if (any(!is.na(entropy))) median(entropy, na.rm = TRUE) else NA_real_,
    median_max_weight = if (any(!is.na(max_w))) median(max_w, na.rm = TRUE) else NA_real_,
    fraction_doublet = if (!is.null(res$results_df))
        mean(res$results_df$spot_class == "doublet_certain", na.rm = TRUE) else NA_real_,
    fraction_singlet = if (!is.null(res$results_df))
        mean(res$results_df$spot_class == "singlet", na.rm = TRUE) else NA_real_
  )
  list(per_cell = per_cell, summary = summary,
       weights = weights_norm, results_df = res$results_df)
}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main <- function() {
  args <- parse_args_local()
  outdir <- args$outdir
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  category_map <- NULL
  if (!is.null(args$`celltype-category-json`)) {
    category_map <- jsonlite::fromJSON(args$`celltype-category-json`,
                                         simplifyVector = FALSE)
    message(sprintf("[main] loaded category map: %s",
                    paste(names(category_map), collapse = ", ")))
  } else {
    # Sensible defaults for lung adenocarcinoma.
    category_map <- list(
      tumor   = list("epithelial", "Epithelial", "Type II", "Type I",
                      "AT1", "AT2", "Club", "Ciliated", "Basal",
                      "Cancer", "tumor"),
      immune  = list("T cells", "T_cell", "B cells", "B_cell",
                      "NK", "macrophage", "Mac", "DC", "monocyte",
                      "neutrophil", "mast", "Plasma", "plasma", "RBC"),
      stromal = list("fibroblast", "Fibro", "endothelial", "Endo",
                      "Smooth muscle", "Pericyte")
    )
  }

  # Probe spatial panel up-front so the reference can be subset to those
  # genes before spacexr densifies (avoids 16 GB R ceiling on whole-trans-
  # criptome references).
  panel_post <- .load_h5ad_counts(args$`spatial-h5ad`)
  panel_genes <- as.character(panel_post$var_names)
  if (!is.null(args$`spatial-h5ad-pre`)) {
    panel_pre <- .load_h5ad_counts(args$`spatial-h5ad-pre`)
    panel_genes <- union(panel_genes, as.character(panel_pre$var_names))
    rm(panel_pre)
  }
  rm(panel_post); gc()
  message(sprintf("[main] spatial-panel gene universe: %d genes", length(panel_genes)))

  ref <- build_reference(args$`reference-h5ad`,
                          args$`reference-celltype-col`,
                          args$`min-cells-per-celltype-reference`,
                          args$seed,
                          restrict_genes = panel_genes)

  runs <- list()
  t0 <- Sys.time()
  spatial_post <- build_spatial(args$`spatial-h5ad`, args$`umi-min`)
  runs$post <- run_one_rctd("post", ref, spatial_post, args, category_map, outdir)
  if (!is.null(args$`spatial-h5ad-pre`)) {
    spatial_pre <- build_spatial(args$`spatial-h5ad-pre`, args$`umi-min`)
    runs$pre <- run_one_rctd("pre", ref, spatial_pre, args, category_map, outdir)
  }
  t1 <- Sys.time()

  # Pre/post comparison if both present
  if ("pre" %in% names(runs) && "post" %in% names(runs)) {
    pre_s <- runs$pre$summary; post_s <- runs$post$summary
    pp <- data.frame(
      metric = c("median_entropy", "median_max_weight",
                  "fraction_doublet", "fraction_singlet",
                  "n_cells"),
      pre   = c(pre_s$median_entropy, pre_s$median_max_weight,
                 pre_s$fraction_doublet, pre_s$fraction_singlet,
                 pre_s$n_cells),
      post  = c(post_s$median_entropy, post_s$median_max_weight,
                 post_s$fraction_doublet, post_s$fraction_singlet,
                 post_s$n_cells)
    )
    pp$delta <- pp$post - pp$pre
    write.table(pp,
                file = file.path(outdir, "rctd_pre_post_metrics.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }

  # Single-row entropy metric per tag
  rows <- do.call(rbind, lapply(names(runs), function(tag) {
    s <- runs[[tag]]$summary
    data.frame(tag = tag,
                n_cells = s$n_cells,
                median_entropy = s$median_entropy,
                median_max_weight = s$median_max_weight,
                fraction_doublet = s$fraction_doublet,
                fraction_singlet = s$fraction_singlet)
  }))
  write.table(rows,
              file = file.path(outdir, "rctd_entropy_metrics.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)

  summary <- list(
    command   = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
    args      = args,
    runs      = lapply(runs, function(r) r$summary),
    runtime_seconds = as.numeric(difftime(t1, t0, units = "secs")),
    timestamp_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  )
  writeLines(jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE),
             file.path(outdir, "rctd_run_summary.json"))
  message(sprintf("DONE in %.1f s. Outputs at %s",
                  summary$runtime_seconds, outdir))
}


# Trampoline for clean error reporting.
status <- tryCatch({ main(); 0 },
                   error = function(e) {
                     message(sprintf("ERROR: %s", conditionMessage(e)))
                     1
                   })
quit(save = "no", status = status)
