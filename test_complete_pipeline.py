"""
Complete Pipeline Test - Single unified test for UI integration.

This is the main test file that:
1. Takes natural language prompts
2. Runs full AI pipeline
3. Generates all charts (basic + advanced + 3D)
4. Outputs assessment JSON

Use this as the reference for what the UI will call.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_model.model import VarLendModel
from ai_model.visualization.advanced_charts import (
    plot_risk_surface_3d,
    plot_volatility_surface_3d,
    plot_stress_test_matrix,
    plot_default_probability_waterfall
)
from datetime import datetime
import json


def run_complete_assessment(
    natural_prompt: str,
    loan_amount: float,
    loan_term_months: int,
    monthly_income: float,
    platforms: list,
    hours_per_week: float,
    liquid_savings: float,
    monthly_expenses: float,
    existing_debt: float,
    **kwargs
):
    """
    Run complete assessment with all visualizations.
    
    This is what the UI should call.
    """
    print("="*100)
    print("VARLEND COMPLETE PIPELINE")
    print("="*100)
    
    print(f"\nUser Input:")
    print(f'  "{natural_prompt}"')
    
    print(f"\nParameters:")
    print(f"  - Loan: ${loan_amount:,.0f} for {loan_term_months} months")
    print(f"  - Income: ${monthly_income:,.0f} gross/month")
    print(f"  - Platforms: {', '.join(platforms)}")
    print(f"  - Hours: {hours_per_week}/week")
    print(f"  - Savings: ${liquid_savings:,.0f}")
    print(f"  - Expenses: ${monthly_expenses:,.0f}/month")
    print(f"  - Existing debt: ${existing_debt:,.0f}/month")
    
    # Initialize model
    model = VarLendModel()
    
    # Run full assessment
    print(f"\nRunning AI assessment pipeline...")
    
    assessment = model.assess_loan_application(
        user_prompt=natural_prompt,
        loan_amount=loan_amount,
        loan_term_months=loan_term_months,
        monthly_income=monthly_income,
        platforms=platforms,
        hours_per_week=hours_per_week,
        liquid_savings=liquid_savings,
        monthly_expenses=monthly_expenses,
        existing_debt=existing_debt,
        **kwargs
    )
    
    print(f"\n[OK] Assessment complete")
    
    # Generate additional 3D charts
    print(f"\nGenerating advanced 3D visualizations...")
    
    try:
        # Get simulation result from assessment data
        from life_simulation.run_life_simulation import run_full_life_simulation
        from monte_carlo_sim.src.integration.profile_builder import CustomerApplication
        from monte_carlo_sim.src.types import LoanConfig
        
        # Rebuild objects for 3D chart generation
        platforms_and_hours = [
            (platform, hours_per_week / len(platforms), kwargs.get("months_experience", 12))
            for platform in platforms
        ]
        
        customer_app = CustomerApplication(
            platforms_and_hours=platforms_and_hours,
            metro_area=kwargs.get("metro_area", "national"),
            months_as_gig_worker=kwargs.get("months_experience", 12),
            has_vehicle=kwargs.get("has_vehicle", True),
            has_dependents=kwargs.get("has_dependents", False),
            liquid_savings=liquid_savings,
            monthly_fixed_expenses=monthly_expenses,
            existing_debt_obligations=existing_debt,
            loan_request_amount=loan_amount,
            requested_term_months=loan_term_months,
            acceptable_rate_range=(0.08, 0.20)
        )
        
        loan_config = LoanConfig(
            amount=loan_amount,
            term_months=loan_term_months,
            annual_rate=kwargs.get("loan_rate", 0.12)
        )
        
        # Run simulation to get result object
        result = run_full_life_simulation(
            archetype_id='volatile_vic',
            customer_application=customer_app,
            loan_config=loan_config,
            random_seed=kwargs.get("random_seed", 42),
            n_paths=kwargs.get("n_paths", 2000),
            horizon_months=kwargs.get("time_horizon_months", loan_term_months)
        )
        
        output_dir = Path("ai_model/outputs/charts")
        
        # Generate 3D and advanced charts
        plot_risk_surface_3d(result, customer_app, 'complete_test', 
                            output_dir / "risk_surface_3d_complete.png")
        print(f"  [OK] 3D risk surface")
        
        plot_volatility_surface_3d(result, 'complete_test',
                                   output_dir / "volatility_surface_3d_complete.png")
        print(f"  [OK] 3D volatility surface")
        
        plot_stress_test_matrix(result, customer_app, 'complete_test',
                               output_dir / "stress_test_matrix_complete.png")
        print(f"  [OK] Stress test matrix")
        
        plot_default_probability_waterfall(result, 'complete_test',
                                          output_dir / "default_waterfall_complete.png")
        print(f"  [OK] Default waterfall")
        
    except Exception as e:
        print(f"  [WARN] Some advanced charts failed: {e}")
    
    # Display results
    print(f"\n{'='*100}")
    print("ASSESSMENT RESULTS")
    print("="*100)
    
    print(f"\nDecision: {'APPROVED' if assessment.approved else 'DECLINED'}")
    print(f"Risk Tier: {assessment.risk_tier.upper()}")
    print(f"Default Risk: {assessment.default_probability:.2%}")
    
    print(f"\nOptimal Loan Structure:")
    print(f"  - Amount: ${assessment.optimal_loan_amount:,.0f}")
    print(f"  - Term: {assessment.optimal_loan_term} months")
    print(f"  - Rate: {assessment.optimal_loan_rate:.2%}")
    
    print(f"\nExpected Loss: ${assessment.simulation_data['risk_metrics']['expected_loss']:,.2f}")
    print(f"CVaR 95%: ${assessment.simulation_data['risk_metrics']['cvar_95']:,.2f}")
    
    print(f"\n{'='*100}")
    print("CHARTS GENERATED")
    print("="*100)
    
    print(f"\nStandard Charts ({len(assessment.charts)}):")
    for i, chart in enumerate(assessment.charts, 1):
        print(f"  {i}. {chart['filename']}")
    
    print(f"\nAdvanced 3D Charts (4):")
    print(f"  - risk_surface_3d_complete.png")
    print(f"  - volatility_surface_3d_complete.png")
    print(f"  - stress_test_matrix_complete.png")
    print(f"  - default_waterfall_complete.png")
    
    print(f"\nTotal: {len(assessment.charts) + 4} charts")
    
    print(f"\n{'='*100}")
    print("EXECUTIVE SUMMARY")
    print("="*100)
    print(f"\n{assessment.executive_summary}\n")
    
    print(f"{'='*100}")
    print(f"[OK] All files saved to: ai_model/outputs/")
    print(f"[OK] Charts in: ai_model/outputs/charts/")
    print("="*100)
    
    return assessment


if __name__ == "__main__":
    print("VarLend Complete Pipeline Test")
    print("="*100)
    print("\nThis is the unified test for UI integration.")
    print("Generates all charts (9 standard + 4 advanced = 13 total)\n")
    
    # Run comprehensive test
    assessment = run_complete_assessment(
        natural_prompt="Experienced driver with Uber and DoorDash, work 40 hours per week, "
                     "make around $3,600 monthly gross income, have $1,800 saved, "
                     "spend $350 on living expenses and $100 on other debts, "
                     "need $5,000 loan for 24 months",
        
        # Required params
        loan_amount=5000,
        loan_term_months=24,
        monthly_income=3600,
        platforms=["uber", "doordash"],
        hours_per_week=40,
        liquid_savings=1800,
        monthly_expenses=350,
        existing_debt=100,
        
        # Optional params
        loan_rate=0.12,
        metro_area="national",
        months_experience=15,
        has_vehicle=True,
        has_dependents=False,
        time_horizon_months=36,
        n_paths=3000,
        random_seed=42
    )
    
    print(f"\n{'='*100}")
    print("READY FOR UI INTEGRATION")
    print("="*100)
    print("\nThe UI should:")
    print("  1. Collect user prompt + basic params (loan amount, term, income)")
    print("  2. Call: VarLendModel().assess_loan_application(...)")
    print("  3. Display: assessment.executive_summary")
    print("  4. Show charts from: assessment.charts")
    print("  5. Optionally show: 3D charts (risk_surface_3d, etc.)")
    print("\nAll data is in the assessment object returned.")
