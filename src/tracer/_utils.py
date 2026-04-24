"""Small shared helpers used across tracer submodules."""

import numpy as np


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
