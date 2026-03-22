#!/usr/bin/env python3
"""
API Runner for VarLend Monte Carlo Simulation

Accepts JSON input via stdin, runs simulation, outputs JSON to stdout.
Designed to be called from Next.js API routes via child_process.

Usage:
    echo '{"profile": {...}, "scenario": {...}}' | python api_runner.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.scenario_parser import parse_ai_scenario
from src.engine.monte_carlo import load_and_prepare, run_simulation
from src.types import (
    GigStream,
    GigType,
    LoanConfig,
    SimulationConfig,
    WorkerProfile,
)


def build_profile_from_json(data: dict) -> WorkerProfile:
    """Convert JSON profile to WorkerProfile dataclass."""
    streams = []
    for s in data.get("streams", []):
        gig_type = GigType(s.get("gig_type", "delivery"))
        streams.append(
            GigStream(
                platform_name=s.get("platform_name", "Unknown"),
                gig_type=gig_type,
                mean_monthly_income=float(s.get("mean_monthly_income", 3000)),
                income_variance=float(s.get("income_variance", 30000)),
                tenure_months=int(s.get("tenure_months", 12)),
                is_primary=s.get("is_primary", True),
            )
        )

    # Default stream if none provided
    if not streams:
        streams = [
            GigStream(
                platform_name="DoorDash",
                gig_type=GigType.DELIVERY,
                mean_monthly_income=3500.0,
                income_variance=35000.0,
                tenure_months=12,
                is_primary=True,
            )
        ]

    credit_range = data.get("credit_score_range", [620, 680])
    rate_range = data.get("acceptable_rate_range", [0.08, 0.18])

    return WorkerProfile(
        streams=streams,
        metro_area=data.get("metro_area", "National"),
        months_as_gig_worker=int(data.get("months_as_gig_worker", 18)),
        has_vehicle=data.get("has_vehicle", True),
        has_dependents=data.get("has_dependents", False),
        liquid_savings=float(data.get("liquid_savings", 3000)),
        monthly_fixed_expenses=float(data.get("monthly_fixed_expenses", 1800)),
        existing_debt_obligations=float(data.get("existing_debt_obligations", 200)),
        credit_score_range=(int(credit_range[0]), int(credit_range[1])),
        loan_request_amount=float(data.get("loan_request_amount", 5000)),
        requested_term_months=int(data.get("requested_term_months", 24)),
        acceptable_rate_range=(float(rate_range[0]), float(rate_range[1])),
        correlation_matrix=None,
    )


def run_api(input_data: dict) -> dict:
    """Run simulation and return results as JSON-serializable dict."""
    try:
        # Extract configuration
        profile_data = input_data.get("profile", {})
        scenario_data = input_data.get("scenario", None)
        config_data = input_data.get("config", {})
        loan_data = input_data.get("loan", {})

        # Build profile
        profile = build_profile_from_json(profile_data)

        # Build config
        config = SimulationConfig(
            n_paths=int(config_data.get("n_paths", 5000)),
            horizon_months=int(config_data.get("horizon_months", 24)),
            random_seed=config_data.get("random_seed", 42),
        )

        # Build loan config
        loan = LoanConfig(
            amount=float(loan_data.get("amount", profile.loan_request_amount)),
            term_months=int(loan_data.get("term_months", profile.requested_term_months)),
            annual_rate=float(loan_data.get("annual_rate", 0.12)),
        )

        # Parse AI scenario if provided
        ai_scenario = None
        if scenario_data:
            ai_scenario = parse_ai_scenario(scenario_data, config.horizon_months)

        # Load and prepare
        load = load_and_prepare(profile, config)

        # Run simulation
        result = run_simulation(profile, config, loan, load, ai_scenario)

        # Format response
        response = {
            "success": True,
            "result": {
                "p_default": float(result.p_default),
                "expected_loss": float(result.expected_loss),
                "cvar_95": float(result.cvar_95),
                "median_income_by_month": result.median_income_by_month.tolist(),
                "p10_income_by_month": result.p10_income_by_month.tolist(),
                "p90_income_by_month": result.p90_income_by_month.tolist(),
                "time_to_default_percentiles": result.time_to_default_percentiles,
                "recommendation": {
                    "approved": result.recommended_loan.approved,
                    "optimal_amount": float(result.recommended_loan.optimal_amount),
                    "optimal_term_months": result.recommended_loan.optimal_term_months,
                    "optimal_rate": float(result.recommended_loan.optimal_rate),
                    "risk_tier": result.recommended_loan.risk_tier.value,
                    "reasoning": result.recommended_loan.reasoning,
                    "alternative_structures": result.recommended_loan.alternative_structures,
                },
            },
            "scenario_applied": scenario_data.get("narrative", "") if scenario_data else None,
        }

        return response

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def main():
    """Read JSON from stdin, run simulation, write JSON to stdout."""
    try:
        input_text = sys.stdin.read()
        input_data = json.loads(input_text) if input_text.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    result = run_api(input_data)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
