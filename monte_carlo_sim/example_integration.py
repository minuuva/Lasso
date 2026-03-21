"""
Example: How to use data_pipeline parameters with customer application data.

This demonstrates the COMPLETE integration flow:
1. Customer submits application (platforms, hours, savings, loan request)
2. System pulls research-backed parameters from data_pipeline
3. ProfileBuilder combines them → WorkerProfile
4. Monte Carlo engine runs simulation
5. System returns loan decision
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.scenario_parser import parse_ai_scenario
from src.engine.monte_carlo import load_and_prepare, run_simulation, sweep_loan_space
from src.integration.profile_builder import CustomerApplication, build_profile_from_application, scenario_from_data_pipeline
from src.types import GigType, LoanConfig, SimulationConfig

from data_pipeline.loaders import DataLoader


def example_loan_application_flow():
    """
    Simulate a real loan application workflow using data pipeline integration.
    """
    print("="*80)
    print("VarLend Loan Application — Data Pipeline Integration Example")
    print("="*80)

    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    loader = DataLoader()

    print("\n1. CUSTOMER SUBMITS APPLICATION\n")
    print("   Customer: Marcus Johnson")
    print("   Location: Atlanta, GA")
    print("   Platforms: DoorDash (35 hrs/week, 18 months), Uber (15 hrs/week, 12 months)")
    print("   Savings: $6,800")
    print("   Fixed expenses: $1,200/month")
    print("   Other debt: $180/month")
    print("   Credit score: 640-680")
    print("   Loan request: $4,500 @ 24 months")

    application = CustomerApplication(
        platforms_and_hours=[
            ("doordash", 35.0, 18),
            ("uber", 15.0, 12),
        ],
        metro_area="atlanta",
        months_as_gig_worker=18,
        has_vehicle=True,
        has_dependents=False,
        liquid_savings=6800.0,
        monthly_fixed_expenses=1200.0,
        existing_debt_obligations=180.0,
        credit_score_range=(640, 680),
        loan_request_amount=4500.0,
        requested_term_months=24,
        acceptable_rate_range=(0.10, 0.18),
    )

    print("\n2. SYSTEM PULLS RESEARCH-BACKED PARAMETERS FROM DATA PIPELINE\n")
    
    print("   Platform hourly rates (from Gridwise research):")
    from data_pipeline.ingest.static_params import PLATFORM_EARNINGS
    print(f"     - DoorDash: ${PLATFORM_EARNINGS['doordash']['hourly_rate']}/hr")
    print(f"     - Uber: ${PLATFORM_EARNINGS['uber']['hourly_rate']}/hr")
    
    print("\n   Volatility parameters (from JPMorgan research):")
    vol = loader.get_income_volatility_params()
    print(f"     - Coefficient of variation: {vol['median_cv']:.1%}")
    print(f"     - Extreme swing probability: {vol['extreme_swing']:.1%}")
    
    print("\n   Metro adjustment (Atlanta):")
    from data_pipeline.ingest.metro_adjustments import get_metro_adjustment
    metro_adj = get_metro_adjustment("atlanta")
    print(f"     - Income multiplier: {metro_adj['income_multiplier']:.2f}×")
    print(f"     - Expense multiplier: {metro_adj['expense_multiplier']:.2f}×")
    
    print("\n   Seasonality (delivery):")
    dec = loader.get_seasonality("delivery", "dec")
    jul = loader.get_seasonality("delivery", "jul")
    print(f"     - December: {dec:.2f}x (holiday surge)")
    print(f"     - July: {jul:.2f}x (summer lull)")

    print("\n3. PROFILE BUILDER COMBINES CUSTOMER + PIPELINE DATA\n")
    
    profile = build_profile_from_application(application, loader)
    
    print(f"   Generated WorkerProfile with {len(profile.streams)} streams:")
    for s in profile.streams:
        print(f"     - {s.platform_name} ({s.gig_type.value}): mu=${s.mean_monthly_income:,.2f}, sigma=${np.sqrt(s.income_variance):,.2f}")
    print(f"   Total monthly obligations: ${profile.monthly_fixed_expenses + profile.existing_debt_obligations:,.2f}")
    print(f"   Liquid buffer: ${profile.liquid_savings:,.2f}")

    print("\n4. MONTE CARLO SIMULATION RUNS\n")
    
    config = SimulationConfig(n_paths=5000, horizon_months=24, random_seed=123)
    load = load_and_prepare(profile, config)
    
    print(f"   Portfolio parameters (correlation-adjusted):")
    print(f"     - Effective mu: ${load.effective_mu_base:,.2f}/month")
    print(f"     - Effective sigma: ${load.effective_sigma_base:,.2f}/month")
    print(f"     - CV: {load.effective_sigma_base / load.effective_mu_base:.1%}")
    
    loan = LoanConfig(
        amount=profile.loan_request_amount,
        term_months=profile.requested_term_months,
        annual_rate=0.5 * (profile.acceptable_rate_range[0] + profile.acceptable_rate_range[1]),
    )
    
    print(f"\n   Running 5,000 × 24-month income paths with jump-diffusion...")
    baseline = run_simulation(profile, config, loan, load, None)

    print("\n5. LOAN DECISION OUTPUT\n")
    
    rec = baseline.recommended_loan
    print(f"   [+] DECISION: {'APPROVED' if rec.approved else 'DECLINED'}")
    print(f"   [+] Risk Tier: {rec.risk_tier.value.upper()}")
    print(f"   [+] Probability of Default: {baseline.p_default:.2%}")
    print(f"   [+] Expected Loss: ${baseline.expected_loss:,.2f}")
    print(f"   [+] CVaR_95 (worst 5% tail): ${baseline.cvar_95:,.2f}")
    print(f"   [+] Recommended Amount: ${rec.optimal_amount:,.2f}")
    print(f"   [+] Recommended Term: {rec.optimal_term_months} months")
    print(f"   [+] Recommended Rate: {rec.optimal_rate:.2%}")
    
    if rec.reasoning:
        print(f"\n   Risk Analysis:")
        for r in rec.reasoning:
            print(f"     • {r}")

    print("\n6. STRESS TEST WITH DATA PIPELINE RECESSION SCENARIO\n")
    
    print("   Loading COVID-2020 scenario from macro_params.json...")
    covid_dict = scenario_from_data_pipeline(
        loader,
        category="recession",
        scenario_name="recession_2020",
        start_month=3,
        gig_type=GigType.DELIVERY,
    )
    covid = parse_ai_scenario(covid_dict, config.horizon_months)
    
    covid_res = run_simulation(profile, config, loan, load, covid)
    rec_covid = covid_res.recommended_loan
    
    print(f"   Scenario: {covid_dict['narrative']}")
    print(f"   Impact: Delivery income x1.3 (surge), rideshare x0.4 (collapse)")
    print(f"\n   Stressed Results:")
    print(f"     - P(default): {covid_res.p_default:.2%}")
    print(f"     - Expected Loss: ${covid_res.expected_loss:,.2f}")
    print(f"     - Risk Tier: {rec_covid.risk_tier.value.upper()}")
    print(f"     - Decision: {'APPROVED' if rec_covid.approved else 'DECLINED'}")

    print("\n7. OPTIMAL LOAN SWEEP (BASELINE SCENARIO)\n")
    
    print("   Testing 100 configurations (5 amounts × 4 terms × 5 rates)...")
    grid, optimal = sweep_loan_space(profile, config, load, None)
    
    if optimal:
        print(f"\n   [+] OPTIMAL CONFIGURATION:")
        print(f"     - Amount: ${optimal['amount']:,.2f} ({optimal['amount']/profile.loan_request_amount:.0%} of requested)")
        print(f"     - Term: {optimal['term_months']} months")
        print(f"     - Rate: {optimal['annual_rate']:.2%}")
        print(f"     - P(default): {optimal['p_default']:.2%}")
        print(f"     - Expected Loss: ${optimal['expected_loss']:,.2f}")
    else:
        print("   [-] No configuration met P(default) < 8% threshold")

    print("\n" + "="*80)


if __name__ == "__main__":
    import numpy as np
    example_loan_application_flow()
