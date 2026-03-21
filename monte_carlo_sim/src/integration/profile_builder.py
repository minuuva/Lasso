"""
Bridge between data_pipeline research parameters and monte_carlo_sim WorkerProfile.

This module converts customer application data + data pipeline parameters into
a complete WorkerProfile ready for Monte Carlo simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_pipeline.loaders import DataLoader
except ImportError:
    raise ImportError(
        "data_pipeline not found. Ensure the data_pipeline directory exists "
        "at the repository root and contains the required JSON files."
    )

from src.types import GigStream, GigType, WorkerProfile


class CustomerApplication:
    """
    Customer-specific inputs collected during loan application.
    
    These are the ONLY parameters that change per borrower — everything else
    comes from the data pipeline's research-backed parameters.
    """

    def __init__(
        self,
        platforms_and_hours: list[tuple[str, float, int]],
        metro_area: str,
        months_as_gig_worker: int,
        has_vehicle: bool,
        has_dependents: bool,
        liquid_savings: float,
        monthly_fixed_expenses: float,
        existing_debt_obligations: float,
        credit_score_range: tuple[int, int],
        loan_request_amount: float,
        requested_term_months: int,
        acceptable_rate_range: tuple[float, float],
    ):
        """
        Customer-specific loan application inputs.

        Parameters
        ----------
        platforms_and_hours:
            List of (platform_name, hours_per_week, tenure_months) tuples.
            Example: [("doordash", 30.0, 18), ("uber", 15.0, 6)]
        metro_area:
            Geographic region — "national", "san_francisco", "new_york", "atlanta", "dallas", "rural".
        months_as_gig_worker:
            Total months working in the gig economy (across all platforms).
        has_vehicle:
            Whether they own a vehicle (required for delivery/rideshare).
        has_dependents:
            Whether they support dependents.
        liquid_savings:
            Current cash reserves (emergency buffer).
        monthly_fixed_expenses:
            Rent, utilities, food, etc. (non-gig expenses).
        existing_debt_obligations:
            Monthly payments on other debts (car loan, credit cards, etc.).
        credit_score_range:
            (min, max) credit score range.
        loan_request_amount:
            How much they're asking to borrow.
        requested_term_months:
            How many months to repay.
        acceptable_rate_range:
            (min, max) annual rate they'll accept.
        """
        self.platforms_and_hours = platforms_and_hours
        self.metro_area = metro_area
        self.months_as_gig_worker = months_as_gig_worker
        self.has_vehicle = has_vehicle
        self.has_dependents = has_dependents
        self.liquid_savings = liquid_savings
        self.monthly_fixed_expenses = monthly_fixed_expenses
        self.existing_debt_obligations = existing_debt_obligations
        self.credit_score_range = credit_score_range
        self.loan_request_amount = loan_request_amount
        self.requested_term_months = requested_term_months
        self.acceptable_rate_range = acceptable_rate_range


def _map_platform_to_gig_type(platform: str) -> GigType:
    """Map customer's platform string to Monte Carlo GigType enum."""
    delivery = ["doordash", "grubhub", "instacart", "ubereats"]
    rideshare = ["uber", "lyft"]
    p = platform.lower()
    if p in delivery:
        return GigType.DELIVERY
    if p in rideshare:
        return GigType.RIDESHARE
    return GigType.FREELANCE


def _get_metro_multipliers(loader: DataLoader, metro: str) -> tuple[float, float]:
    """
    Extract metro income/expense multipliers from data pipeline.
    
    Returns
    -------
    tuple[float, float]
        (income_multiplier, expense_multiplier)
    """
    from data_pipeline.ingest.metro_adjustments import get_metro_adjustment

    adj = get_metro_adjustment(metro)
    return float(adj["income_multiplier"]), float(adj["expense_multiplier"])


def _calculate_stream_mu_sigma(
    loader: DataLoader,
    platform: str,
    hours_per_week: float,
    metro: str,
) -> tuple[float, float]:
    """
    Convert platform + hours/week → (μ, σ²) using data pipeline parameters.

    Uses the data pipeline's calibration logic:
    1. Platform hourly rate (from static_params)
    2. Metro income multiplier
    3. Expense calculation
    4. Volatility (36% CV from JPMorgan)
    5. Platform-specific variance multiplier
    """
    from data_pipeline.ingest.static_params import PLATFORM_EARNINGS, EXPENSES, INCOME_VOLATILITY
    from data_pipeline.ingest.metro_adjustments import adjust_income, get_metro_adjustment

    if platform.lower() not in PLATFORM_EARNINGS:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(PLATFORM_EARNINGS.keys())}")

    hourly = PLATFORM_EARNINGS[platform.lower()]["hourly_rate"]
    variance_mult = PLATFORM_EARNINGS[platform.lower()]["variance_multiplier"]

    weekly_gross = hourly * hours_per_week
    monthly_gross = weekly_gross * 4.33

    monthly_gross_adj = adjust_income(monthly_gross, metro, platform.lower())

    is_fulltime = hours_per_week >= 35
    if is_fulltime:
        gas_mid = sum(EXPENSES["gas_weekly_fulltime"]) / 2
    else:
        gas_mid = sum(EXPENSES["gas_weekly_parttime"]) / 2
    gas_monthly = gas_mid * 4.33
    maint = sum(EXPENSES["maintenance_monthly"]) / 2
    deprec = EXPENSES["vehicle_depreciation_monthly"]
    insurance = EXPENSES["insurance_monthly"]
    phone = EXPENSES["phone_data_monthly"]
    total_exp = gas_monthly + maint + deprec + insurance + phone
    
    metro_adj = get_metro_adjustment(metro)
    total_exp_adj = total_exp * metro_adj["expense_multiplier"]

    tax = EXPENSES["self_employment_tax"]
    net = monthly_gross_adj - total_exp_adj - (monthly_gross_adj * tax)

    cv = INCOME_VOLATILITY["median_cv"]
    sigma = net * cv * variance_mult
    variance = sigma**2

    return float(net), float(variance)


def build_profile_from_application(
    application: CustomerApplication,
    loader: DataLoader | None = None,
) -> WorkerProfile:
    """
    Convert customer application + data pipeline parameters → WorkerProfile.

    This is the MAIN INTEGRATION FUNCTION. It:
    1. Reads fixed parameters from data_pipeline (hourly rates, CV, expense formulas)
    2. Takes customer-specific inputs (platforms, hours, savings, etc.)
    3. Computes derived (μ, σ²) per stream using data pipeline calibration
    4. Returns a WorkerProfile ready for Monte Carlo simulation

    Parameters
    ----------
    application:
        Customer-specific loan application data (collected via web form, API, etc.).
    loader:
        Optional DataLoader instance; creates one if None.

    Returns
    -------
    WorkerProfile
        Fully populated profile with research-backed parameters + customer specifics.
    """
    if loader is None:
        loader = DataLoader()

    streams: list[GigStream] = []
    total_hours = sum(h for _, h, _ in application.platforms_and_hours)

    for idx, (platform, hours, tenure) in enumerate(application.platforms_and_hours):
        mu, var = _calculate_stream_mu_sigma(loader, platform, hours, application.metro_area)
        gig_type = _map_platform_to_gig_type(platform)
        is_primary = hours == max(h for _, h, _ in application.platforms_and_hours)

        streams.append(
            GigStream(
                platform_name=platform,
                gig_type=gig_type,
                mean_monthly_income=mu,
                income_variance=var,
                tenure_months=tenure,
                is_primary=is_primary,
            )
        )

    return WorkerProfile(
        streams=streams,
        metro_area=application.metro_area,
        months_as_gig_worker=application.months_as_gig_worker,
        has_vehicle=application.has_vehicle,
        has_dependents=application.has_dependents,
        liquid_savings=application.liquid_savings,
        monthly_fixed_expenses=application.monthly_fixed_expenses,
        existing_debt_obligations=application.existing_debt_obligations,
        credit_score_range=application.credit_score_range,
        loan_request_amount=application.loan_request_amount,
        requested_term_months=application.requested_term_months,
        acceptable_rate_range=application.acceptable_rate_range,
        correlation_matrix=None,
    )


def scenario_from_data_pipeline(
    loader: DataLoader,
    category: str,
    scenario_name: str,
    start_month: int,
    gig_type: GigType,
) -> dict[str, Any]:
    """
    Convert data_pipeline macro scenario → AIScenario dict (ready for scenario_parser).

    Parameters
    ----------
    loader:
        DataLoader instance.
    category:
        "recession", "gas_spike", "regulatory", "tech_disruption".
    scenario_name:
        E.g., "recession_2020", "gas_spike_moderate".
    start_month:
        When the shock begins (0-based month index).
    gig_type:
        Customer's dominant gig type (used to look up platform_impacts).

    Returns
    -------
    dict
        Raw dict ready for ``parse_ai_scenario``.
    """
    scenario = loader.get_scenario(category, scenario_name)

    gig_key_map = {GigType.DELIVERY: "delivery", GigType.RIDESHARE: "rideshare", GigType.FREELANCE: "freelance"}
    gig_key = gig_key_map.get(gig_type, "general_gig")

    income_impact = scenario.get("platform_impacts", {}).get(gig_key, 1.0)
    duration = scenario.get("duration_months", 12)
    if duration == "permanent":
        duration = 240

    expense_changes = scenario.get("expense_changes", {})
    gas_mult = expense_changes.get("gas_price_multiplier", 1.0)

    volatility_reduction = scenario.get("volatility_reduction", 1.0)
    sigma_mult = 1.0 / volatility_reduction if volatility_reduction > 0 else 1.0

    lambda_mult = 1.0
    if income_impact < 0.9:
        lambda_mult = 1.0 + (1.0 - income_impact) * 0.8
    elif income_impact > 1.1:
        lambda_mult = 0.85

    shifts = [
        {
            "target": "mu_base",
            "type": "multiplicative",
            "magnitude": income_impact,
            "start_month": start_month,
            "duration_months": min(duration, 240),
            "decay": "linear",
        },
        {
            "target": "sigma_base",
            "type": "multiplicative",
            "magnitude": sigma_mult,
            "start_month": start_month,
            "duration_months": min(duration, 240),
            "decay": "linear",
        },
        {
            "target": "lambda",
            "type": "multiplicative",
            "magnitude": lambda_mult,
            "start_month": start_month,
            "duration_months": min(duration, 240),
            "decay": "linear",
        },
        {
            "target": "expenses",
            "type": "multiplicative",
            "magnitude": gas_mult,
            "start_month": start_month,
            "duration_months": min(duration, 240),
            "decay": "linear",
        },
    ]

    return {
        "narrative": scenario.get("name", "Unnamed scenario"),
        "parameter_shifts": shifts,
        "discrete_jumps": [],
    }
