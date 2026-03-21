"""
Monte Carlo orchestration: income paths, defaults, lender economics, and loan sweeps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.engine.correlation import effective_portfolio_mu_sigma
from src.engine.income_model import draw_monthly_income
from src.engine.parameter_state import effective_parameters, macro_scaling
from src.engine.seasonality import get_multipliers
from src.engine.defaults import detect_defaults_and_losses
from src.risk import loan_evaluator, risk_metrics
from src.types import (
    AIScenario,
    JumpParams,
    LoadResult,
    LoanConfig,
    LoanRecommendation,
    RiskTier,
    SimulationConfig,
    SimulationResult,
    WorkerProfile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_pipeline.loaders import DataLoader
except ImportError:
    _data_loader = None
else:
    _data_loader = DataLoader()


def _get_jump_params_from_pipeline() -> dict[str, float]:
    """Load default jump parameters from data_pipeline."""
    if _data_loader is None:
        return {
            "monthly_jump_probability": 0.25,
            "jump_mean_multiplier": -0.15,
            "jump_sigma_multiplier": 0.22,
        }
    try:
        expense_data = _data_loader.get_expense_data()
        return expense_data.get("jump_parameters", {
            "monthly_jump_probability": 0.25,
            "jump_mean_multiplier": -0.15,
            "jump_sigma_multiplier": 0.22,
        })
    except Exception:
        return {
            "monthly_jump_probability": 0.25,
            "jump_mean_multiplier": -0.15,
            "jump_sigma_multiplier": 0.22,
        }


def _monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    if term_months <= 0 or principal <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r <= 0:
        return principal / term_months
    return float(principal * r / (1.0 - (1.0 + r) ** (-term_months)))


def _scheduled_total_interest(principal: float, annual_rate: float, term_months: int) -> float:
    pmt = _monthly_payment(principal, annual_rate, term_months)
    return float(pmt * term_months - principal)


def _dominant_gig_type(profile: WorkerProfile):
    prim = [s for s in profile.streams if s.is_primary]
    pool = prim if prim else profile.streams
    return max(pool, key=lambda s: s.mean_monthly_income).gig_type


def run_simulation(
    profile: WorkerProfile,
    config: SimulationConfig,
    loan: LoanConfig,
    load: LoadResult,
    scenario: AIScenario | None = None,
    refine_alternatives: bool = True,
) -> SimulationResult:
    """
    Run the full vectorized Monte Carlo pipeline for one loan configuration.

    Parameters
    ----------
    profile:
        Worker characteristics and gig streams.
    config:
        Path count, horizon, RNG seed, macro and correlation modes.
    loan:
        Loan amount, term, and annual rate.
    load:
        Derived fields from the JSON loader (obligations, portfolio mu/sigma, jump overrides).
    scenario:
        Optional AI scenario with deterministic shifts and discrete jumps.
    refine_alternatives:
        When ``False``, skip DECLINE restructuring search (avoids nested recursion).

    Returns
    -------
    SimulationResult
        Risk metrics, income envelopes, default timing, and a ``LoanRecommendation``.
    """
    rng = np.random.default_rng(config.random_seed)
    n_paths = config.n_paths
    h = config.horizon_months

    mu0, sig0 = load.effective_mu_base, load.effective_sigma_base
    
    jump_defaults = _get_jump_params_from_pipeline()
    lam0 = jump_defaults.get("monthly_jump_probability", 0.25)
    mu_jump = jump_defaults.get("jump_mean_multiplier", -0.15) * mu0
    sigma_jump = jump_defaults.get("jump_sigma_multiplier", 0.22) * mu0
    
    if load.jump_params.jump_lambda is not None:
        lam0 = float(load.jump_params.jump_lambda)
    if load.jump_params.mu_jump is not None:
        mu_jump = float(load.jump_params.mu_jump)
    if load.jump_params.sigma_jump is not None:
        sigma_jump = float(load.jump_params.sigma_jump)

    m_mu, m_sig, m_lam, m_exp = macro_scaling(config.macro_state)
    mu_base0 = mu0 * m_mu
    sigma_base0 = sig0 * m_sig
    lambda_base0 = lam0 * m_lam
    obligations0 = load.total_monthly_obligations * m_exp

    shifts = scenario.parameter_shifts if scenario else []

    dgig = _dominant_gig_type(profile)

    income = np.zeros((n_paths, h), dtype=np.float64)
    poisson_echo = np.zeros(n_paths, dtype=np.float64)
    disc_echo_carry = np.zeros(n_paths, dtype=np.float64)
    disc_echo_end = -1
    disc_decay = 1.0

    for t in range(h):
        if disc_echo_end < 0 or t > disc_echo_end:
            disc_echo_carry.fill(0.0)
        elif disc_decay != 1.0:
            disc_echo_carry *= disc_decay

        mu_b, sig_b, lam_b, exp_b = effective_parameters(
            t, mu_base0, sigma_base0, lambda_base0, obligations0, shifts
        )
        smu, ssig = get_multipliers(dgig, t % 12)
        mu_t = np.full(n_paths, mu_b * smu, dtype=np.float64)
        sigma_t = np.full(n_paths, max(sig_b * ssig, 1e-6), dtype=np.float64)
        lambda_t = np.full(n_paths, min(max(lam_b, 0.0), 1.0), dtype=np.float64)

        djumps = [j for j in (scenario.discrete_jumps if scenario else []) if j.month == t]
        d_amt = float(sum(j.amount for j in djumps))
        d_var = float(sum(j.variance for j in djumps))

        echo_in = poisson_echo + disc_echo_carry
        inc_col, poisson_echo, disc_draw = draw_monthly_income(
            mu_t,
            sigma_t,
            lambda_t,
            mu_jump,
            sigma_jump,
            echo_in,
            d_amt,
            d_var,
            rng,
            jump_echo_decay=1.0,
        )
        income[:, t] = inc_col

        echo_candidates = [j for j in djumps if j.echo_months is not None and j.echo_months > 0]
        if echo_candidates:
            jm = max(echo_candidates, key=lambda j: int(j.echo_months or 0))
            disc_echo_end = max(disc_echo_end, t + int(jm.echo_months))
            disc_decay = float(jm.echo_decay_rate) if jm.echo_decay_rate is not None else 0.7
            disc_echo_carry = disc_echo_carry + disc_draw * disc_decay

    expenses = np.array(
        [effective_parameters(t, mu_base0, sigma_base0, lambda_base0, obligations0, shifts)[3] for t in range(h)],
        dtype=np.float64,
    )

    pmt = _monthly_payment(loan.amount, loan.annual_rate, loan.term_months)
    defaulted, default_month, losses = detect_defaults_and_losses(
        income,
        expenses,
        pmt,
        profile.liquid_savings,
        loan.amount,
        loan.annual_rate,
        loan.term_months,
    )

    p_def = risk_metrics.p_default(defaulted)
    el = risk_metrics.expected_loss(losses)
    cvar95 = risk_metrics.cvar(losses, 0.95)
    ttd = risk_metrics.time_to_default_dist(default_month)

    med = risk_metrics.income_envelope(income, [50.0])[0]
    p10 = risk_metrics.income_envelope(income, [10.0])[0]
    p90 = risk_metrics.income_envelope(income, [90.0])[0]

    placeholder = LoanRecommendation(
        approved=False,
        optimal_amount=loan.amount,
        optimal_term_months=loan.term_months,
        optimal_rate=loan.annual_rate,
        risk_tier=RiskTier.PRIME,
        reasoning=[],
        alternative_structures=[],
    )
    sim = SimulationResult(
        p_default=p_def,
        expected_loss=el,
        cvar_95=cvar95,
        median_income_by_month=med,
        p10_income_by_month=p10,
        p90_income_by_month=p90,
        time_to_default_percentiles=ttd,
        recommended_loan=placeholder,
        raw_paths=income,
    )

    rec = loan_evaluator.evaluate_loan(sim, loan)
    sim.recommended_loan = rec

    if refine_alternatives and rec.risk_tier == RiskTier.DECLINE:

        def _sim_amt(amt: float) -> SimulationResult:
            lc = LoanConfig(amount=amt, term_months=loan.term_months, annual_rate=loan.annual_rate)
            return run_simulation(profile, config, lc, load, scenario, refine_alternatives=False)

        alt = loan_evaluator.suggest_restructuring(sim, loan, _sim_amt)
        if alt is not None:
            rec.alternative_structures.append(alt)

    return sim


def sweep_loan_space(
    profile: WorkerProfile,
    config: SimulationConfig,
    load: LoadResult,
    scenario: AIScenario | None = None,
) -> tuple[list[dict], dict | None]:
    """
    Grid search over principal multiples, terms, and annual rates.

    Returns
    -------
    grid:
        List of result dicts for every evaluated configuration.
    optimal:
        Configuration with **maximum** principal among cells with P(default) < approval_threshold, if any.
    """
    if _data_loader is not None:
        try:
            expense_data = _data_loader.get_expense_data()
            sweep_config = expense_data.get("loan_sweep_grid", {})
            amount_mults = sweep_config.get("amount_multipliers", [0.25, 0.5, 0.75, 1.0, 1.25])
            terms = sweep_config.get("term_months", [12, 24, 36, 48])
            rates = sweep_config.get("annual_rates", [0.08, 0.12, 0.16, 0.20, 0.24])
            approval_threshold = sweep_config.get("approval_threshold", 0.08)
        except Exception:
            amount_mults = [0.25, 0.5, 0.75, 1.0, 1.25]
            terms = [12, 24, 36, 48]
            rates = [0.08, 0.12, 0.16, 0.20, 0.24]
            approval_threshold = 0.08
    else:
        amount_mults = [0.25, 0.5, 0.75, 1.0, 1.25]
        terms = [12, 24, 36, 48]
        rates = [0.08, 0.12, 0.16, 0.20, 0.24]
        approval_threshold = 0.08
    
    amounts = [m * profile.loan_request_amount for m in amount_mults]
    grid: list[dict] = []
    best: dict | None = None
    for a in amounts:
        for tm in terms:
            for r in rates:
                loan = LoanConfig(amount=a, term_months=tm, annual_rate=r)
                res = run_simulation(profile, config, loan, load, scenario, refine_alternatives=False)
                cell = {
                    "amount": a,
                    "term_months": tm,
                    "annual_rate": r,
                    "p_default": res.p_default,
                    "expected_loss": res.expected_loss,
                    "cvar_95": res.cvar_95,
                }
                grid.append(cell)
                if res.p_default < approval_threshold:
                    if best is None or a > best["amount"] or (a == best["amount"] and tm > best["term_months"]):
                        best = {
                            "amount": a,
                            "term_months": tm,
                            "annual_rate": r,
                            "p_default": res.p_default,
                            "expected_loss": res.expected_loss,
                            "cvar_95": res.cvar_95,
                        }
    return grid, best


def load_and_prepare(profile: WorkerProfile, config: SimulationConfig) -> LoadResult:
    """
    Convenience wrapper when JSON loading is bypassed — rebuilds ``LoadResult`` from a profile.
    """
    mu, sig = effective_portfolio_mu_sigma(profile, config)
    tot = profile.monthly_fixed_expenses + profile.existing_debt_obligations
    return LoadResult(
        profile=profile,
        jump_params=JumpParams(),
        total_monthly_obligations=tot,
        effective_mu_base=mu,
        effective_sigma_base=sig,
    )
