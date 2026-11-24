"""Small shared helpers used across tracer submodules."""

import numpy as np
import pandas as pd


def ensure_string_categorical(df: pd.DataFrame, col: str) -> None:
    """Convert `df[col]` to pd.Categorical in place if it isn't already.

    Low-cardinality string columns (`feature_name` ≈ 300 unique, `cell_id`
    ≈ 60K unique) store one Python-object reference per row when left as
    object dtype — tens to hundreds of MB at 1.4M rows, multiple GB at
    100M. Categorical dtype stores an int32 code + the vocabulary once,
    cutting memory 10–50× and accelerating groupby/isin/equality checks.

    Idempotent: if the column is already categorical, this is a no-op.
    Mutates `df` in place (no new DataFrame allocation).

    If the column is a numeric dtype it is first coerced to string via
    `.astype(str)` — matching the pipeline contract that cell/gene IDs
    are strings.
    """
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return
    if not pd.api.types.is_string_dtype(s):
        s = s.astype(str)
    df[col] = s.astype("category")


def prepare_transcript_df(
    df: pd.DataFrame,
    *,
    gene_col: str = "feature_name",
    fov_col: str = "fov_name",
    cell_id_col: str = "cell_id",
    transcript_id_col: str = "transcript_id",
) -> pd.DataFrame:
    """Normalise a transcript-level DataFrame for the TRACER pipeline.

    Performs four idempotent memory-saving conversions in place:

    1. `feature_name` → Categorical. Gene vocabulary is tiny (hundreds)
       and highly repeated (millions of transcripts) — drops the column
       from ~5 GB of object pointers at 100M rows to ~100 MB of int
       codes, and speeds up every downstream groupby/isin/equality.

    2. `fov_name` → Categorical. FOV vocabulary is small (~100 typical)
       and the column is just carried along by the pipeline. Drops a
       further ~70 MiB at 1.4M rows / ~5 GB at 100M.

    3. `cell_id` int64 → int32 if all values fit (signed, ≤ 2^31). Cuts
       this column in half. At 1.4M rows: ~5 MiB → ~3 MiB; at 100M:
       ~800 MiB → ~400 MiB.

    4. `transcript_id` uint64 → uint32 if all values fit (≤ 2^32 − 1).
       Same proportional saving.

    `cell_id` is **not** converted to Categorical even when it's a
    string: downstream stages create derived columns that add new
    string labels (`"{cid}-1"` partials, `"UNASSIGNED_N"` components)
    via `.loc` assignment, which would error against a frozen
    categorical vocabulary. Pre-expansion would be possible but lives
    in stage-specific code, not this generic helper.

    Idempotent and safe to call repeatedly — a no-op when the column
    is already in the target dtype, or when the column is missing.
    """
    if gene_col in df.columns:
        ensure_string_categorical(df, gene_col)
    if fov_col in df.columns:
        ensure_string_categorical(df, fov_col)

    # Numeric downcasts: only when safe (values fit in target dtype).
    if cell_id_col in df.columns:
        s = df[cell_id_col]
        if s.dtype == np.int64:
            mn, mx = int(s.min()), int(s.max())
            if -(2**31) <= mn and mx < 2**31:
                df[cell_id_col] = s.astype(np.int32)

    if transcript_id_col in df.columns:
        s = df[transcript_id_col]
        if s.dtype == np.uint64:
            if int(s.max()) < 2**32:
                df[transcript_id_col] = s.astype(np.uint32)
        elif s.dtype == np.int64:
            mn, mx = int(s.min()), int(s.max())
            if 0 <= mn and mx < 2**32:
                df[transcript_id_col] = s.astype(np.uint32)
            elif -(2**31) <= mn and mx < 2**31:
                df[transcript_id_col] = s.astype(np.int32)

    return df


def relu_symmetric(x, tau):
    """
    Two-sided ReLU with dead zone [-tau, tau].

    Values in [-tau, tau] are zeroed out.
    Values above tau are shifted down by tau.
    Values below -tau are shifted up by tau.

    Parameters
    ----------
    x : array_like
        Input values
    tau : float
        Dead zone threshold

    Returns
    -------
    out : np.ndarray
        ReLU-transformed values
    """
    out = np.zeros_like(x)
    out[x > tau] = x[x > tau] - tau
    out[x < -tau] = x[x < -tau] + tau
    return out


def deprecated_legacy_group(fn):
    """Mark a function of the retired spatial-CC Group backend.

    Unified-prune step 7 removed that backend from the pipeline (Group is
    always the density cascade). The function still works, for one
    release, and warns on every call.
    """
    import functools
    import warnings

    msg = (f"tracer.{fn.__name__} belongs to the retired spatial-CC Group "
           "backend (unified-prune step 7) and will be removed in a future "
           "release; the pipeline's Group is the density cascade "
           "(tracer.density_cascade.cascade_as_residual_handler).")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return fn(*args, **kwargs)

    return wrapper


def symmetric_csr(W):
    """Return ``W`` as a symmetric CSR PMI matrix. Idempotent.

    The single contract for the gene x gene panel ``aux["W"]``: every
    stored pair ``(i, j)`` is also stored as ``(j, i)`` with the same value.
    Panel builders produce the upper triangle only (``i < j``); this fills
    in the mirror entry wherever it is missing, so it accepts an upper-
    triangle, lower-triangle or already-symmetric matrix and never doubles
    a value (unlike ``W + W.T``, which doubles every pair of a symmetric
    input). On an upper-triangle input it is value-identical to
    ``W + W.T``.

    Explicit zeros are kept as stored entries (an observed PMI of exactly
    0.0 is evidence, unlike a structurally absent pair); ``W + W.T`` would
    drop them. Indices are sorted within each row. Dense input is returned
    as a float32 ndarray with missing mirror entries filled from the other
    triangle (NaN and 0 count as missing).
    """
    import scipy.sparse as sp

    if not sp.issparse(W):
        A = np.asarray(W, dtype=np.float32)
        missing = (A == 0) | np.isnan(A)
        return np.where(missing, A.T, A).astype(np.float32, copy=False)

    C = W.tocoo()
    G = C.shape[0]
    row = C.row.astype(np.int64)
    col = C.col.astype(np.int64)
    key = row * G + col
    mirror = col * G + row
    add = ~np.isin(mirror, key) & (row != col)
    S = sp.csr_matrix(
        (
            np.concatenate([C.data, C.data[add]]).astype(np.float32),
            (np.concatenate([row, col[add]]), np.concatenate([col, row[add]])),
        ),
        shape=C.shape,
        dtype=np.float32,
    )
    S.sort_indices()
    return S


def fold_symmetric_dense(A):
    """Symmetric view of a dense per-entity PMI block. Idempotent.

    ``A`` is a block sliced from ``aux["W"]`` in arbitrary gene order. If
    ``W`` was triangular, each pair is stored on exactly one side; if it
    was symmetric, on both. Taking the stored side (``A`` where nonzero,
    else ``A.T``) is correct for both, and equals ``A + A.T`` on a
    triangular input.
    """
    return np.where(A != 0, A, A.T)
