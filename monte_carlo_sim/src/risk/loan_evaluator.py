"""
Loan recommendation engine built on ``SimulationResult`` metrics.

Risk tier thresholds are loaded from data_pipeline/data/expenses.json.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.types import LoanConfig, LoanRecommendation, RiskTier, SimulationResult

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_pipeline.loaders import DataLoader
except ImportError:
    _data_loader = None
else:
    _data_loader = DataLoader()


def _get_risk_tier_thresholds() -> dict[str, float]:
    """Load risk tier thresholds from data_pipeline."""
    if _data_loader is None:
        return {
            "prime": 0.04,
            "near_prime": 0.08,
            "subprime": 0.15,
        }
    try:
        expense_data = _data_loader.get_expense_data()
        risk_tiers = expense_data.get("risk_tiers", {})
        return {
            "prime": risk_tiers.get("prime", {}).get("threshold", 0.04),
            "near_prime": risk_tiers.get("near_prime", {}).get("threshold", 0.08),
            "subprime": risk_tiers.get("subprime", {}).get("threshold", 0.15),
        }
    except Exception:
        return {
            "prime": 0.04,
            "near_prime": 0.08,
            "subprime": 0.15,
        }


def _tier_from_p(p: float) -> RiskTier:
    """Classify risk tier based on P(default), using thresholds from data_pipeline."""
    thresholds = _get_risk_tier_thresholds()
    if p < thresholds["prime"]:
        return RiskTier.PRIME
    if p < thresholds["near_prime"]:
        return RiskTier.NEAR_PRIME
    if p < thresholds["subprime"]:
        return RiskTier.SUBPRIME
    return RiskTier.DECLINE


def evaluate_loan(simulation: SimulationResult, loan: LoanConfig) -> LoanRecommendation:
    """
    Produce an approval decision, tier, and human-readable rationale strings.

    Parameters
    ----------
    simulation:
        Aggregated Monte Carlo outputs for a single loan configuration.
    loan:
        Loan terms corresponding to the simulation run.
    """
    p = simulation.p_default
    tier = _tier_from_p(p)
    approved = tier != RiskTier.DECLINE

    reasoning: list[str] = []
    if simulation.cvar_95 > loan.amount:
        reasoning.append(
            f"Tail CVaR ({simulation.cvar_95:,.0f}) exceeds principal ({loan.amount:,.0f}), indicating severe downside tail risk."
        )
    ttd = simulation.time_to_default_percentiles
    if not np.isnan(ttd.get("p50", np.nan)):
        reasoning.append(
            f"Median time-to-default among defaulting paths is near month {ttd['p50']:.1f} "
            f"(p90 ~= {ttd['p90']:.1f}), highlighting mid-horizon stress."
        )

    inc = simulation.raw_paths
    h = inc.shape[1]
    v_month = np.var(inc, axis=0)
    early = float(np.mean(v_month[: max(1, h // 3)]))
    late = float(np.mean(v_month[2 * h // 3 :]))
    if late < 0.85 * early:
        reasoning.append("Cross-path income dispersion tends to decline late in the horizon (diversification / mean-reversion effect).")

    mu_m = simulation.median_income_by_month
    if mu_m.size >= 12:
        q4 = float(np.mean(mu_m[9:12]))
        spring = float(np.mean(mu_m[3:6]))
        if q4 < 0.92 * spring:
            reasoning.append("Seasonal pattern suggests a softer Q4 median income window versus spring.")

    if not reasoning:
        reasoning.append("Key risk metrics are within nominal ranges for the requested structure.")

    alts: list[dict[str, Any]] = []
    return LoanRecommendation(
        approved=approved,
        optimal_amount=loan.amount,
        optimal_term_months=loan.term_months,
        optimal_rate=loan.annual_rate,
        risk_tier=tier,
        reasoning=reasoning,
        alternative_structures=alts,
    )


def suggest_restructuring(
    simulation: SimulationResult,
    loan: LoanConfig,
    simulate_fn: Callable[[float], SimulationResult],
) -> dict[str, Any] | None:
    """
    If the loan is DECLINE, binary-search a smaller principal down to 25% of requested.

    Returns the largest amount whose simulated P(default) stays below the SUBPRIME ceiling.
    """
    if simulation.recommended_loan.risk_tier != RiskTier.DECLINE:
        return None
    
    thresholds = _get_risk_tier_thresholds()
    subprime_threshold = thresholds["subprime"]
    
    lo_amt = 0.25 * loan.amount
    hi_amt = loan.amount
    best: dict[str, Any] | None = None
    for _ in range(32):
        mid = 0.5 * (lo_amt + hi_amt)
        if mid <= lo_amt + 1e-6:
            break
        res = simulate_fn(mid)
        if res.p_default < subprime_threshold:
            best = {
                "amount": mid,
                "term_months": loan.term_months,
                "annual_rate": loan.annual_rate,
                "p_default": res.p_default,
            }
            lo_amt = mid
        else:
            hi_amt = mid
    return best
