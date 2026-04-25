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
