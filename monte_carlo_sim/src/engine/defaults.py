"""
Vectorized default detection and loss-at-default on amortizing loans.

Rolling 3-month sums use cumulative sums along the time axis (vectorized, no Python
month loops for the rolling window). Buffer updates use a short horizon loop with
inner operations vectorized across all paths — paths are never iterated in Python.

Default detection parameters and recovery rates are loaded from data_pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_pipeline.loaders import DataLoader
except ImportError:
    _data_loader = None
else:
    _data_loader = DataLoader()


def _get_default_params() -> dict[str, float]:
    """Load default detection parameters from data_pipeline."""
    if _data_loader is None:
        return {
            "rolling_window_months": 3,
            "cash_flow_threshold_multiplier": -1.5,
            "buffer_threshold_multiplier": 1.0,
            "loss_given_default_rate": 0.7,
        }
    try:
        expense_data = _data_loader.get_expense_data()
        default_detection = expense_data.get("default_detection", {})
        default_params = expense_data.get("default_parameters", {})
        
        rolling_window = default_detection.get("rolling_window_months", 3)
        cf_threshold = default_detection.get("cash_flow_threshold_multiplier", -1.5)
        buffer_threshold = default_detection.get("buffer_threshold_multiplier", 1.0)
        
        recovery_rate = default_params.get("recovery_rate", 0.3)
        lgd = 1.0 - recovery_rate
        
        return {
            "rolling_window_months": rolling_window,
            "cash_flow_threshold_multiplier": cf_threshold,
            "buffer_threshold_multiplier": buffer_threshold,
            "loss_given_default_rate": lgd,
        }
    except Exception:
        return {
            "rolling_window_months": 3,
            "cash_flow_threshold_multiplier": -1.5,
            "buffer_threshold_multiplier": 1.0,
            "loss_given_default_rate": 0.7,
        }


def _outstanding_balances(
    principal: float, annual_rate: float, term_months: int, default_month: np.ndarray
) -> np.ndarray:
    """
    Vectorized outstanding principal at end of default month (per path).

    Parameters
    ----------
    default_month:
        Shape (n_paths,), ``-1`` if no default; otherwise 0-based month index.
    """
    k = np.where(default_month >= 0, np.minimum(default_month + 1, term_months), 0).astype(np.int64)
    r = annual_rate / 12.0
    if abs(r) < 1e-15:
        return np.where(default_month >= 0, principal * (1.0 - k.astype(np.float64) / term_months), 0.0)
    num = (1.0 + r) ** term_months - (1.0 + r) ** k.astype(np.float64)
    den = (1.0 + r) ** term_months - 1.0
    bal = principal * num / den
    return np.where(default_month >= 0, bal, 0.0)


def detect_defaults_and_losses(
    income_matrix: np.ndarray,
    expenses_by_month: np.ndarray,
    monthly_payment: float,
    initial_buffer: float,
    loan_principal: float,
    annual_rate: float,
    loan_term_months: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify first default month per path and loss given default.

    Default at month ``t`` when **both**:
    - rolling N-month sum of net cash flows (ending at ``t``) < ``threshold_mult * monthly_payment``
    - current buffer after month ``t`` updates < ``buffer_mult * monthly_payment``

    Parameters are loaded from data_pipeline/data/expenses.json.

    Parameters
    ----------
    income_matrix:
        Shape (n_paths, horizon_months), gross income paths.
    expenses_by_month:
        Shape (horizon_months,), nonnegative expenses per month (same across paths).
    monthly_payment:
        Loan payment (scalar).
    initial_buffer:
        Starting liquid buffer (same for all paths).
    loan_principal, annual_rate, loan_term_months:
        Amortization terms for loss given default.

    Returns
    -------
    defaulted:
        Shape (n_paths,), boolean default flag.
    default_month:
        Shape (n_paths,), first default month index or ``-1`` if no default.
    loss_given_default:
        Shape (n_paths,), economic loss (0 if no default); LGD based on recovery_rate from pipeline.
    """
    params = _get_default_params()
    rolling_window = int(params["rolling_window_months"])
    cf_threshold_mult = params["cash_flow_threshold_multiplier"]
    buffer_mult = params["buffer_threshold_multiplier"]
    lgd_rate = params["loss_given_default_rate"]
    
    n_paths, h = income_matrix.shape
    exp = np.asarray(expenses_by_month, dtype=np.float64)
    pay = np.broadcast_to(monthly_payment, (n_paths, h))
    exp_b = np.broadcast_to(exp, (n_paths, h))
    net = income_matrix - exp_b - pay

    cs = np.cumsum(net, axis=1)
    rolling_sum = np.empty_like(cs)
    rolling_sum[:, :rolling_window] = cs[:, :rolling_window]
    rolling_sum[:, rolling_window:] = cs[:, rolling_window:] - cs[:, :-rolling_window]

    rolling_stress = rolling_sum < cf_threshold_mult * monthly_payment

    buffer = np.full(n_paths, initial_buffer, dtype=np.float64)
    default_month = np.full(n_paths, -1, dtype=np.int64)
    defaulted = np.zeros(n_paths, dtype=bool)
    mp_thresh = buffer_mult * monthly_payment

    for t in range(h):
        buffer = np.maximum(buffer + net[:, t], 0.0)
        cond = rolling_stress[:, t] & (buffer < mp_thresh) & (~defaulted)
        defaulted |= cond
        default_month = np.where(cond & (default_month < 0), t, default_month)

    bal = _outstanding_balances(loan_principal, annual_rate, loan_term_months, default_month)
    loss = np.where(defaulted, lgd_rate * bal, 0.0)
    return defaulted, default_month, loss
