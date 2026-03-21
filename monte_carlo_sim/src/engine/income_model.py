"""
Vectorized jump-diffusion style monthly income draws across all Monte Carlo paths.
"""

from __future__ import annotations

import numpy as np


def draw_monthly_income(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    lambda_t: np.ndarray,
    mu_jump: float,
    sigma_jump: float,
    echo: np.ndarray,
    discrete_jump_amount: float,
    discrete_jump_variance: float,
    rng: np.random.Generator,
    jump_echo_decay: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw one month of gross incomes for all paths.

    Parameters
    ----------
    mu_t:
        Shape (n_paths,), location of the Gaussian component per path.
    sigma_t:
        Shape (n_paths,), scale of the Gaussian component per path.
    lambda_t:
        Shape (n_paths,), Poisson jump probability mass per path (clipped to [0,1] implicitly by mask).
    mu_jump, sigma_jump:
        Jump magnitude distribution parameters (scalars).
    echo:
        Shape (n_paths,), carry-in echo from prior month jumps.
    discrete_jump_amount, discrete_jump_variance:
        If no discrete shock this month, pass ``amount=0`` and ``variance=0`` (deterministic zero draw).
    rng:
        Seeded ``numpy.random.Generator`` supplied by the caller.
    jump_echo_decay:
        Factor applied to this month's jump magnitudes for next month's echo state.

    Returns
    -------
    tuple
        ``(income, echo_next, discrete_draw)`` where ``income`` and ``echo_next`` have
        shape (n_paths,), and ``discrete_draw`` records the Gaussian discrete shock
        for auditing / optional echo chaining.
    """
    n_paths = mu_t.shape[0]
    baseline = rng.normal(0.0, 1.0, size=n_paths) * sigma_t + mu_t
    jump_mask = rng.random(n_paths) < lambda_t
    jump_magnitudes = rng.normal(mu_jump, sigma_jump, size=n_paths)
    jump_term = jump_magnitudes * jump_mask
    if discrete_jump_variance > 0.0 or discrete_jump_amount != 0.0:
        std = float(np.sqrt(max(discrete_jump_variance, 0.0)))
        discrete = rng.normal(discrete_jump_amount, std, size=n_paths)
    else:
        discrete = np.zeros(n_paths, dtype=np.float64)
    total = baseline + jump_term + echo + discrete
    income = np.maximum(total, 0.0)
    echo_next = jump_term * float(jump_echo_decay)
    return income, echo_next, discrete
