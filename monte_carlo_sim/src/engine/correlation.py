"""
Multi-stream correlation modeling, portfolio (mu, sigma), and correlated Gaussian draws.

Array shapes are documented in each function docstring.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky

from src.types import CorrelationMode, GigType, SimulationConfig, WorkerProfile


def _is_platform_gig(gig_type: GigType) -> bool:
    return gig_type in (GigType.DELIVERY, GigType.RIDESHARE)


def _pairwise_correlation(ti: GigType, tj: GigType) -> float:
    """Build off-diagonal correlation for two stream gig types (symmetric)."""
    if ti == tj:
        return 0.7
    pi, pj = _is_platform_gig(ti), _is_platform_gig(tj)
    if pi and pj:
        return 0.4
    if (pi and tj == GigType.FREELANCE) or (pj and ti == GigType.FREELANCE):
        return 0.2
    if ti == GigType.MIXED or tj == GigType.MIXED:
        if {ti, tj} == {GigType.MIXED, GigType.MIXED}:
            return 0.7
        if (pi and tj == GigType.MIXED) or (pj and ti == GigType.MIXED):
            return 0.35
        return 0.3
    return 0.25


def build_correlation_matrix(streams: list, mode: CorrelationMode, custom: np.ndarray | None) -> np.ndarray:
    """
    Construct an n_streams × n_streams correlation matrix.

    Parameters
    ----------
    streams:
        Sequence of objects with attribute ``gig_type: GigType`` (e.g. GigStream).
    mode:
        INDEPENDENT, CORRELATED, or CUSTOM_MATRIX.
    custom:
        When mode is CUSTOM_MATRIX, a square PSD correlation matrix; otherwise ignored.

    Returns
    -------
    np.ndarray
        Shape (n, n), symmetric, unit diagonal.
    """
    n = len(streams)
    if n == 0:
        raise ValueError("streams must be non-empty")
    if mode == CorrelationMode.CUSTOM_MATRIX:
        if custom is None:
            raise ValueError("custom correlation matrix required for CUSTOM_MATRIX mode")
        mat = np.asarray(custom, dtype=np.float64)
        if mat.shape != (n, n):
            raise ValueError("custom matrix shape must match number of streams")
        return mat
    if mode == CorrelationMode.INDEPENDENT:
        return np.eye(n, dtype=np.float64)
    rho = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ti = streams[i].gig_type
            tj = streams[j].gig_type
            r = _pairwise_correlation(ti, tj)
            rho[i, j] = r
            rho[j, i] = r
    return rho


def portfolio_sigma(weights: np.ndarray, sigmas: np.ndarray, correlation: np.ndarray) -> float:
    """
    Portfolio volatility using σ_p = sqrt(w^T Σ w).

    Parameters
    ----------
    weights:
        Shape (n_streams,), nonnegative, should sum to 1.
    sigmas:
        Shape (n_streams,), per-stream standard deviations.
    correlation:
        Shape (n_streams, n_streams,) correlation matrix.

    Returns
    -------
    float
        Portfolio standard deviation.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    cov = (s[:, None] * correlation) * s[None, :]
    v = float(w @ cov @ w)
    return float(np.sqrt(max(v, 0.0)))


def correlated_standard_normals(
    correlation: np.ndarray, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw correlated standard normals.

    Parameters
    ----------
    correlation:
        Shape (n_streams, n_streams), PSD correlation matrix.
    n_paths:
        Number of Monte Carlo paths.
    rng:
        Seeded NumPy Generator (never constructed inside this function).

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_streams), each column marginally N(0,1) with correlation ``correlation``.
    """
    rho = np.asarray(correlation, dtype=np.float64)
    n = rho.shape[0]
    l = cholesky(rho, lower=True)
    z = rng.standard_normal((n, n_paths))
    y = (l @ z).T
    return y


def correlated_income_draws(
    means: np.ndarray,
    sigmas: np.ndarray,
    correlation: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Correlated income draws: scale/add per stream from correlated standard normals.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_streams); column k is income for stream k.
    """
    z = correlated_standard_normals(correlation, n_paths, rng)
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    return m + z * s


def effective_portfolio_mu_sigma(profile: WorkerProfile, config: SimulationConfig) -> tuple[float, float]:
    """
    Collapse multiple gig streams into a single (mu, sigma) for the univariate engine.

    Portfolio mean is the sum of individual stream means. Portfolio sigma is calculated
    using correlation-aware covariance matrix. For a single stream, returns that stream's
    mean and standard deviation with no correlation adjustment.

    Returns
    -------
    tuple[float, float]
        (effective_mu_base, effective_sigma_base).
    """
    streams = profile.streams
    if len(streams) == 1:
        st = streams[0]
        return float(st.mean_monthly_income), float(np.sqrt(max(st.income_variance, 0.0)))
    means = np.array([s.mean_monthly_income for s in streams], dtype=np.float64)
    sigmas = np.sqrt(np.maximum([s.income_variance for s in streams], 0.0))
    total = float(np.sum(means))
    if total <= 0:
        raise ValueError("total mean monthly income must be positive")
    weights = means / total
    custom = None
    if config.correlation_mode == CorrelationMode.CUSTOM_MATRIX and profile.correlation_matrix is not None:
        custom = np.asarray(profile.correlation_matrix, dtype=np.float64)
    rho = build_correlation_matrix(streams, config.correlation_mode, custom)
    mu_p = total
    sigma_p = portfolio_sigma(weights, sigmas, rho)
    return mu_p, sigma_p
