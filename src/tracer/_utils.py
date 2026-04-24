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
) -> pd.DataFrame:
    """Normalise a transcript-level DataFrame for the TRACER pipeline.

    Currently just converts `feature_name` to Categorical — the gene
    vocabulary is small (hundreds) and highly repeated (millions of
    transcripts), so object→categorical is a ~10× memory drop on that
    column and speeds up every `groupby('feature_name')`, `isin`, and
    equality check that touches it downstream.

    `cell_id` is intentionally left alone: downstream stages create
    derived columns (`cell_id_npmi_cons_p1`, `cell_id_stitched`, …)
    that add *new* string labels (e.g. `"{cid}-1"` partials). Making
    `cell_id` categorical would propagate the dtype to those derived
    columns and break `.loc` assignment of unseen categories. A future
    change can pre-expand the category vocabulary to keep those
    columns categorical too.

    Call at the top of any stage that ingests a fresh transcript df.
    Idempotent — a no-op when the column is already categorical.
    Mutates `df` in place and returns it for convenience.
    """
    if gene_col in df.columns:
        ensure_string_categorical(df, gene_col)
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
