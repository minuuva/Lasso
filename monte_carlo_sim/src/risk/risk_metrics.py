"""
Path-level risk metrics — all reductions are NumPy vectorized (axis reductions, not path loops).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def p_default(defaulted: np.ndarray) -> float:
    """
    Probability of default as the sample mean of boolean default flags.

    Parameters
    ----------
    defaulted:
        Shape (n_paths,), boolean array.
    """
    return float(np.mean(defaulted.astype(np.float64)))


def expected_loss(losses: np.ndarray) -> float:
    """
    Expected loss as the mean of per-path loss amounts (non-default paths typically 0).

    Parameters
    ----------
    losses:
        Shape (n_paths,), nonnegative loss amounts.
    """
    return float(np.mean(losses))


def cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Conditional Value-at-Risk: mean of losses at or above the ``alpha`` percentile.

    Parameters
    ----------
    losses:
        Shape (n_paths,), loss samples.
    alpha:
        Upper tail probability mass (e.g. 0.95 keeps the worst ~5% tail).
    """
    if losses.size == 0:
        return 0.0
    thr = float(np.percentile(losses, alpha * 100.0))
    tail = losses[losses >= thr]
    if tail.size == 0:
        return float(np.max(losses))
    return float(np.mean(tail))


def time_to_default_dist(default_months: np.ndarray) -> dict[str, float]:
    """
    Percentiles of time-to-default for paths that actually default.

    Parameters
    ----------
    default_months:
        Shape (n_paths,), month index or ``-1`` if no default.
    """
    dm = default_months[default_months >= 0]
    if dm.size == 0:
        return {"p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p90": float("nan")}
    return {
        "p25": float(np.percentile(dm, 25)),
        "p50": float(np.percentile(dm, 50)),
        "p75": float(np.percentile(dm, 75)),
        "p90": float(np.percentile(dm, 90)),
    }


def income_envelope(income_matrix: np.ndarray, percentiles: Iterable[float]) -> np.ndarray:
    """
    Month-by-month income quantiles across paths.

    Parameters
    ----------
    income_matrix:
        Shape (n_paths, horizon_months).
    percentiles:
        Iterable of percent values in [0, 100] passed to ``numpy.percentile``.

    Returns
    -------
    np.ndarray
        Shape (len(percentiles), horizon_months).
    """
    qs = list(percentiles)
    return np.stack([np.percentile(income_matrix, q, axis=0) for q in qs], axis=0)
