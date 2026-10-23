"""Arm-level CNV calling downstream of InSituCNV (PROTOTYPE).

Reads the AnnData InSituCNV writes (`adata_cnv.h5ad`: layers["counts"], var
chromosome/start/end, obs compartment/annotation/cnv_leiden) and calls chromosome arms
gain / loss / neutral from the entities' OWN counts -- no cell-to-cell smoothing.

Primary unit: the CLUSTER (score_groups). Each cluster's cells are summed into one
profile; at that depth (1e5-1e6 transcripts) nearly every gene has counts, so each gene
gets its own fold change and its own vote. Single entities can also be scored
(score_matrix), but at Xenium depth one-copy arm changes are mostly below detection.

Per profile, per gene
  expected mu   = profile total x the gene's share in the normal reference
  fold change   LFC = log((y + 1) / (mu + 1))           0 vs a tiny expectation ~ neutral
  weight        w = mu/(mu + 1): at most one vote per gene (~1 for every gene at cluster depth)
  reference     "pooled" (default): one share over all normal cells. "envelope": bounds
                across normal types at the gene level -- tested and found to zero most genes
                when the normal types are as different as fibroblast / myeloid / T
Per profile
  subtract the weighted median LFC over all genes (whole-profile shift)
Per profile, per arm
  arm effect    weighted median of the arm's genes (default). At cluster depth it resists the
                normal-cell genes tumour lacks, which dragged the 10%-trimmed mean down on
                most tumour arms. In sparse single cells the median has ~no power; use
                stat="wtrim" there.
  location test the same statistic on n_null random groups of the same number of genes from
                the REST of the genome; z = (arm - null mean)/null SD -> one-sided p for gain
                and loss (pmode="z"; an empirical rank p floors at 1/(n_null+1), which BH
                across ~38 arms can never bring under 0.05 for a single arm)
  calls         two-sided p = 2*min, BH across arms within the profile, gain/loss at q < alpha

Reference cells are never scored against themselves: single normal cells are cross-fitted
(fold-excluded reference); a cluster is scored against normal cells OUTSIDE that cluster.

Known limits (stated, not yet handled): (1) real arms co-vary more than random gene groups
even without CNV, so z-tails are optimistic in normal single cells; (2) with no normal
epithelium, an epithelial programme concentrated on one arm cannot be told apart from a
copy change of that arm -- normal types themselves differ by up to ~1.5 z on some arms.
"""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm
import cnv_review_common as C

CEN = C.BENCH / "data/hg38_centromeres.tsv"      # derived from UCSC hg38 cytoBand (acen)


# ----------------------------------------------------------------------------- arms
def arm_map(var: pd.DataFrame, min_genes: int = 30, cen_path=CEN):
    """Gene -> arm ("1p", "1q", ...) for autosomes; arms with < min_genes genes dropped."""
    cen = pd.read_csv(cen_path, sep="\t").set_index("chromosome")
    chrom = var["chromosome"].astype(str)
    chrom = chrom.where(chrom.str.startswith("chr"), "chr" + chrom)
    start = pd.to_numeric(var["start"], errors="coerce")
    end = pd.to_numeric(var["end"], errors="coerce")
    cs, ce = chrom.map(cen["cen_start"]), chrom.map(cen["cen_end"])
    mid, cm = (start + end) / 2, (cs + ce) / 2
    side = np.where(start < cs, "p", np.where(start >= ce, "q", np.where(mid < cm, "p", "q")))
    auto = chrom.isin([f"chr{i}" for i in range(1, 23)]) & start.notna()
    arm = pd.Series(np.where(auto, chrom.str.replace("chr", "", regex=False) + side, None),
                    index=var.index, dtype=object)
    n_all = arm.value_counts()
    arm = arm.where(arm.map(n_all) >= min_genes)
    order = sorted(arm.dropna().unique(), key=lambda a: (int(a[:-1]), a[-1]))
    dropped = sorted(set(n_all.index) - set(order), key=lambda a: (int(a[:-1]), a[-1]))
    return arm, order, n_all, dropped


# ------------------------------------------------------------------------ primitives
def wmedian(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted median along the last axis (lower weighted median)."""
    o = np.argsort(V, axis=-1)
    Vs, Ws = np.take_along_axis(V, o, -1), np.take_along_axis(W, o, -1)
    c = np.cumsum(Ws, -1)
    k = np.minimum((c < c[..., -1:] / 2).sum(-1, keepdims=True), V.shape[-1] - 1)
    return np.take_along_axis(Vs, k, -1)[..., 0]


def wtrim(V: np.ndarray, W: np.ndarray, trim: float = 0.10) -> np.ndarray:
    """Weighted trimmed mean along the last axis: drop the lowest and highest `trim` of
    the total weight, weighted mean of the rest."""
    o = np.argsort(V, axis=-1)
    Vs, Ws = np.take_along_axis(V, o, -1), np.take_along_axis(W, o, -1)
    c = np.cumsum(Ws, -1)
    tot = c[..., -1:]
    mid = (c - Ws / 2) / np.where(tot > 0, tot, 1)                 # weight midpoint of each gene
    keep = (mid >= trim) & (mid <= 1 - trim)
    Wk = np.where(keep, Ws, 0)
    s = Wk.sum(-1)
    return np.where(s > 0, (Vs * Wk).sum(-1) / np.where(s > 0, s, 1), 0.0)


def wmean(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    s = W.sum(-1)
    return np.where(s > 0, (V * W).sum(-1) / np.where(s > 0, s, 1), 0.0)


STATS = {"wmedian": wmedian, "wtrim": wtrim, "wmean": wmean}


def type_shares(Y, types_of_cells: np.ndarray, types: list, pseudo: float = 0.5) -> np.ndarray:
    """(T, G) gene shares per normal type, pooled over that type's cells."""
    S = np.vstack([np.asarray(Y[types_of_cells == t].sum(0)).ravel() for t in types]) + pseudo
    return (S / S.sum(1, keepdims=True)).astype(np.float64)


def gene_scores(Y: np.ndarray, shares: np.ndarray):
    """Bounded, centred per-gene log fold changes L and weights w, both (n, G)."""
    Y = np.asarray(Y, dtype=np.float64)
    N = Y.sum(1)
    mu = N[:, None, None] * shares[None, :, :]                       # (n, T, G)
    lfc = np.log((Y[:, None, :] + 1.0) / (mu + 1.0))
    L = np.where((lfc > 0).all(1), lfc.min(1), np.where((lfc < 0).all(1), lfc.max(1), 0.0))
    mbar = mu.mean(1)
    w = mbar / (mbar + 1.0)
    L = L - wmedian(L, w)[:, None]
    return L.astype(np.float32), w.astype(np.float32)


def bh_rows(P: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg within each row."""
    n = P.shape[1]
    o = np.argsort(P, 1)
    q = np.take_along_axis(P, o, 1) * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[:, ::-1], 1)[:, ::-1]
    out = np.empty_like(q)
    np.put_along_axis(out, o, np.minimum(q, 1.0), 1)
    return out


def drop_to(y: np.ndarray, total: int, rng) -> np.ndarray:
    """Randomly drop transcripts from a count vector until it holds `total`."""
    return rng.multivariate_hypergeometric(np.asarray(y, dtype=np.int64), int(total))


# ---------------------------------------------------------------------------- caller
class ArmCNV:
    def __init__(self, adata, reference_key="compartment", reference_cat="reference",
                 type_key="annotation", min_type_cells=25, n_folds=5, n_null=1000,
                 min_arm_genes=30, alpha=0.05, block=32, seed=0, stat="wmedian",
                 reference="pooled", pmode="z"):
        self.alpha, self.block, self.n_null = alpha, block, n_null
        self.stat = STATS[stat]                 # arm summary of per-gene fold changes
        # "envelope": bounds across normal types; "pooled": one mean over all normal cells
        self.reference = reference
        # "z": p from the random-group null's mean/SD (resolves small p);
        # "empirical": rank in the null (floor 1/(n_null+1))
        self.pmode = pmode
        arm, self.arms, self.arm_n_all, self.arms_dropped = arm_map(adata.var, min_arm_genes)
        keep = arm.notna().to_numpy()
        self.genes = adata.var_names[keep]
        self.gene_arm = arm[keep].to_numpy()
        X = adata.layers["counts"]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        self.Y = X[:, keep].astype(np.float64)
        self.obs = adata.obs.copy()
        rng = np.random.default_rng(seed)

        ref = (self.obs[reference_key].astype(str) == reference_cat).to_numpy()
        typ = self.obs[type_key].astype(str).to_numpy()
        vc = pd.Series(typ[ref]).value_counts()
        self.types = [t for t in vc.index if vc[t] >= min_type_cells]
        self.is_ref = ref & np.isin(typ, self.types)            # cells that BUILD references
        self.typ = typ
        self.fold = np.full(len(typ), -1)
        for t in self.types:                                     # stratified folds
            idx = np.where(self.is_ref & (typ == t))[0]
            self.fold[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
        # pooled reference = every reference cell as ONE type
        ref_types = self.types if reference == "envelope" else ["__pooled__"]
        tlab = typ if reference == "envelope" else np.where(self.is_ref, "__pooled__", typ)
        self.ref_types, self.tlab = ref_types, tlab
        Yr, tr = self.Y[self.is_ref], tlab[self.is_ref]
        self.shares_all = type_shares(Yr, tr, ref_types)
        self.shares_fold = {f: type_shares(self.Y[self.is_ref & (self.fold != f)],
                                           tlab[self.is_ref & (self.fold != f)], ref_types)
                            for f in range(n_folds)}

        self.arm_idx = [np.where(self.gene_arm == a)[0] for a in self.arms]
        G = len(self.genes)
        self.null_idx = []
        for ix in self.arm_idx:                                  # random groups, rest of genome
            pool = np.setdiff1d(np.arange(G), ix)
            r = rng.random((n_null, len(pool)))
            self.null_idx.append(pool[np.argpartition(r, len(ix), axis=1)[:, :len(ix)]])

    # --- core scoring of a count matrix against one set of shares
    def _score(self, Y: np.ndarray, shares: np.ndarray):
        L, w = gene_scores(Y, shares)
        D = np.stack([self.stat(L[:, ix], w[:, ix]) for ix in self.arm_idx], 1)
        pg = np.empty_like(D); pl = np.empty_like(D)
        B = self.n_null
        for a, R in enumerate(self.null_idx):
            nul = self.stat(L[:, R], w[:, R])                     # (n, B)
            if self.pmode == "z":
                z = (D[:, a] - nul.mean(1)) / (nul.std(1) + 1e-9)
                pg[:, a], pl[:, a] = norm.sf(z), norm.cdf(z)
            else:
                pg[:, a] = (1 + (nul >= D[:, [a]]).sum(1)) / (B + 1)
                pl[:, a] = (1 + (nul <= D[:, [a]]).sum(1)) / (B + 1)
        return D, pg, pl

    def score_matrix(self, Y, folds=None):
        """Score rows of Y (dense or sparse counts on self.genes). folds[i] >= 0 -> that
        row is scored with the reference built WITHOUT fold folds[i] (cross-fitting)."""
        n = Y.shape[0]
        folds = np.full(n, -1) if folds is None else np.asarray(folds)
        D = np.empty((n, len(self.arms)), np.float32); pg = np.empty_like(D); pl = np.empty_like(D)
        for f in np.unique(folds):
            rows = np.where(folds == f)[0]
            sh = self.shares_all if f < 0 else self.shares_fold[int(f)]
            for s in range(0, len(rows), self.block):
                r = rows[s:s + self.block]
                Yb = Y[r]
                Yb = Yb.toarray() if sp.issparse(Yb) else np.asarray(Yb)
                D[r], pg[r], pl[r] = self._score(Yb, sh)
        return self.finish(D, pg, pl)

    def finish(self, D, pg, pl):
        p2 = np.minimum(1.0, 2 * np.minimum(pg, pl))
        q = bh_rows(p2)
        call = np.where(q < self.alpha, np.where(pg < pl, 1, -1), 0).astype(np.int8)
        omni = (norm.isf(np.clip(p2, 1e-300, 1) / 2) ** 2).sum(1)
        return dict(D=D, p_gain=pg, p_loss=pl, p=p2, q=q, call=call, omnibus=omni,
                    n_called=(call != 0).sum(1))

    def score_all(self):
        """Every entity in the AnnData; reference cells cross-fitted."""
        return self.score_matrix(self.Y, self.fold)

    def score_groups(self, labels, min_cells: int = 20) -> pd.DataFrame:
        """Cluster-level calls. Each group's cells are summed into one profile and scored
        against a reference built from the normal cells OUTSIDE that group.
        Returns a tidy group x arm table."""
        labels = pd.Series(np.asarray(labels).astype(str))
        out = []
        for g, idx in labels.groupby(labels).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) < min_cells:
                continue
            inside = np.zeros(len(labels), bool); inside[idx] = True
            rm = self.is_ref & ~inside
            shares = type_shares(self.Y[rm], self.tlab[rm], self.ref_types)
            prof = np.asarray(self.Y[idx].sum(0))
            D, pg, pl = self._score(prof, shares)
            r = self.finish(D, pg, pl)
            for a, arm in enumerate(self.arms):
                out.append(dict(group=g, arm=arm, n_cells=len(idx), n_tx=int(prof.sum()),
                                n_ref_used=int(rm.sum()), effect=float(D[0, a]),
                                fold_change=float(np.exp(D[0, a])), p_gain=float(pg[0, a]),
                                p_loss=float(pl[0, a]), p=float(r["p"][0, a]),
                                q=float(r["q"][0, a]), call=int(r["call"][0, a])))
        return pd.DataFrame(out)


# ------------------------------------------------------------ test on InSituCNV values
def xcnv_window_arms(adata, cen_path=CEN) -> np.ndarray:
    """Arm of each X_cnv column (window), by the window's bp midpoint."""
    wins = C.window_intervals(adata.var)
    if len(wins) != adata.obsm["X_cnv"].shape[1]:
        raise ValueError(f"rebuilt {len(wins)} windows, X_cnv has {adata.obsm['X_cnv'].shape[1]}")
    cen = pd.read_csv(cen_path, sep="\t").set_index("chromosome")
    cm = wins["chromosome"].map((cen["cen_start"] + cen["cen_end"]) / 2)
    return (wins["chromosome"].str.replace("chr", "", regex=False)
            + np.where(wins["bp_mid"] < cm, "p", "q")).to_numpy()


def xcnv_score_groups(adata, labels, arms, min_cells: int = 20, alpha: float = 0.05,
                      gap: int = 5) -> pd.DataFrame:
    """Arm calls from InSituCNV's own X_cnv, cluster level.

    Each group's X_cnv is averaged over its cells; arm effect = mean over the arm's windows.
    Location test: the same mean over EVERY contiguous stretch of the same number of
    windows elsewhere in the genome (genome order). Random genes would be wrong here:
    X_cnv windows are 60-gene running means, so an arm's windows share genes and move
    together, while scattered windows do not. Stretches keep that sharing. Stretches
    within `gap` windows of the arm are left out (windows 5 steps apart still share 10
    genes). z -> one-sided p for gain / loss, two-sided 2*min, BH across arms per group.
    In an aneuploid profile many stretches sit on other altered arms, so the null is wide
    and the test conservative; in a profile that is mostly thresholded zeros it is narrow.
    """
    X = C.dense_cnv(adata)
    warm = xcnv_window_arms(adata)
    W = X.shape[1]
    labels = pd.Series(np.asarray(labels).astype(str))
    out = []
    for g, idx in labels.groupby(labels).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < min_cells:
            continue
        v = X[idx].mean(0)
        cs = np.concatenate([[0.0], np.cumsum(v)])
        D = np.full(len(arms), np.nan); pg = np.full(len(arms), np.nan); pl = pg.copy()
        n_null = np.zeros(len(arms), int)
        for a, arm in enumerate(arms):
            w = np.where(warm == arm)[0]
            if not len(w):
                continue
            k, lo, hi = len(w), w.min(), w.max()
            st = np.arange(0, W - k + 1)
            ok = (st + k - 1 < lo - gap) | (st > hi + gap)
            nul = (cs[st[ok] + k] - cs[st[ok]]) / k
            D[a] = v[w].mean(); n_null[a] = ok.sum()
            z = (D[a] - nul.mean()) / (nul.std() + 1e-12)
            pg[a], pl[a] = norm.sf(z), norm.cdf(z)
        p2 = np.minimum(1.0, 2 * np.minimum(pg, pl))
        fin = np.isfinite(p2)
        q = np.full(len(arms), np.nan)
        q[fin] = bh_rows(p2[fin][None])[0]
        call = np.where(fin & (q < alpha), np.where(pg < pl, 1, -1), 0)
        for a, arm in enumerate(arms):
            out.append(dict(group=g, arm=arm, n_cells=len(idx),
                            n_windows=int((warm == arm).sum()), n_null=int(n_null[a]),
                            effect=float(D[a]), p_gain=float(pg[a]), p_loss=float(pl[a]),
                            p=float(p2[a]), q=float(q[a]), call=int(call[a])))
    return pd.DataFrame(out)
